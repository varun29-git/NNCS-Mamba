"""
Master Training Pipeline for NNCS-Mamba

Usage Examples:
1. Smoke Test (Fast check):
   python train.py --phase smoke --outdir runs/smoke

2. Baseline Imitation (The heavy lifting):
   python train.py --phase imitation --num-traj 5000 --batch-size 32 --outdir runs/imitation

3. CEGIS Refinement (Hunting edge cases):
   python train.py --phase cegis --resume runs/imitation/best_imitation.pt --cegis-cycles 20

4. Full Pipeline (Imitation → CEGIS, one command):
   python train.py --phase all --outdir runs/full
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

from drone_env import DronePlant, DroneExpertController
from mamba_learner import MambaController
from cegis_loop import falsify_cem, fix_and_merge


# =============================================================================
# Helper: CPU-Backed Dataloader (Prevents T4 OOM Errors)
# =============================================================================
def create_cpu_dataloaders(dataset, batch_size, val_split=0.15):
    """Converts the list of trajectories into CPU-bound PyTorch DataLoaders."""
    data_list = list(dataset)
    np.random.shuffle(data_list)

    split_idx = int(len(data_list) * (1 - val_split))
    train_raw = data_list[:split_idx]
    val_raw = data_list[split_idx:]

    def to_dataset(raw_list):
        if not raw_list:
            return None
        # KEEP ON CPU: Do not use device='cuda' here!
        # The MambaController will move batches to GPU during training.
        o = torch.tensor(np.stack([x[0] for x in raw_list]), dtype=torch.float32)
        a = torch.tensor(np.stack([x[1] for x in raw_list]), dtype=torch.float32)
        return TensorDataset(o, a)

    train_ds = to_dataset(train_raw)
    val_ds = to_dataset(val_raw)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True) if val_ds else None

    return train_loader, val_loader


# =============================================================================
# Helper: Generate Baseline Data
# =============================================================================
def generate_expert_data(num_traj, seq_steps, dt):
    print(f"[*] Harvesting {num_traj} expert trajectories...")
    plant = DronePlant()
    expert = DroneExpertController()
    dataset = []

    for i in range(num_traj):
        y = plant.reset()
        # Randomize start-time to simulate partially burned fuel (Low-Mass training)
        if np.random.rand() > 0.5:
            plant.time = np.random.uniform(0.0, 60.0) 
            y = plant.state.copy()
            
        expert.reset()
        expert.set_plant_ref(plant)

        o_seq, a_seq = [], []
        for _ in range(seq_steps):
            u = expert.compute_action(y)
            o_seq.append(y)
            a_seq.append(u)
            y = plant.step(u, dt)

        dataset.append((np.array(o_seq, dtype=np.float32), np.array(a_seq, dtype=np.float32)))
        if (i + 1) % 500 == 0:
            print(f"    ...generated {i + 1}/{num_traj}")

    return dataset


# =============================================================================
# Phase Runners
# =============================================================================
def run_smoke(controller, args, outdir):
    """Quick sanity check — kernels compile, backward pass works."""
    print("\n=== PHASE: SMOKE TEST ===")
    print("Testing kernel compilation and backward pass...")
    dataset = generate_expert_data(10, 100, 0.1)
    train_loader, val_loader = create_cpu_dataloaders(dataset, batch_size=2)
    metrics = controller.update(train_loader, val_loader, epochs=2, fit_normalizer=True)
    print(f"Smoke Test Complete. Train Loss: {metrics['train_loss']:.4f} | Val Loss: {metrics['val_loss']:.4f}")


def run_imitation(controller, args, outdir, dataset):
    """Pure imitation learning from expert demonstrations."""
    print("\n=== PHASE: BASELINE IMITATION ===")
    train_loader, val_loader = create_cpu_dataloaders(dataset, args.batch_size)

    best_val_loss = float("inf")
    csv_log = open(outdir / "imitation_log.csv", "w")
    csv_log.write("epoch,train_loss,val_loss,lr\n")

    for epoch in range(1, args.epochs + 1):
        metrics = controller.update(
            train_loader, val_loader,
            epochs=1,
            fit_normalizer=(epoch == 1 and not controller.normalizer_fitted),
        )
        t_loss = metrics["train_loss"]
        v_loss = metrics["val_loss"]
        lr = metrics["lr"]

        csv_log.write(f"{epoch},{t_loss:.6f},{v_loss:.6f},{lr:.2e}\n")
        csv_log.flush()

        print(f"Epoch {epoch:03d}/{args.epochs} | Train MSE: {t_loss:.5f} | Val MSE: {v_loss:.5f} | LR: {lr:.2e}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            controller.save_checkpoint(str(outdir / "best_imitation.pt"), phase="imitation", epoch=epoch)
            print("   -> Saved new best checkpoint.")

    csv_log.close()
    controller.save_checkpoint(str(outdir / "last_imitation.pt"), phase="imitation", epoch=args.epochs)
    print("\n[*] Imitation Phase Complete.")
    return outdir / "best_imitation.pt"


def run_cegis(controller, args, outdir, dataset):
    """CEGIS refinement — falsify, fix, retrain."""
    print("\n=== PHASE: CEGIS REFINEMENT ===")

    plant = DronePlant()
    expert = DroneExpertController()
    expert.set_plant_ref(plant)

    csv_log = open(outdir / "cegis_log.csv", "w")
    csv_log.write("cycle,fails,coverage,train_loss,val_loss\n")

    for cycle in range(1, args.cegis_cycles + 1):
        print(f"\n--- CEGIS Cycle {cycle}/{args.cegis_cycles} ---")

        # 1. Hunt for failures
        print("[*] Falsifier hunting for edge cases...")
        failures = falsify_cem(
            controller, plant, expert,
            pop_size=20, seq_steps=args.seq_steps, dt=0.1,
        )

        num_fails = len(failures)
        coverage = (1.0 - min(1.0, num_fails / 60.0)) * 100
        print(f"    Found {num_fails} unique failures. Safety Coverage: {coverage:.1f}%")

        if num_fails == 0:
            print("\n✔️  CEGIS Converged! 100% Safety Coverage achieved.")
            controller.save_checkpoint(str(outdir / "best_cegis.pt"), phase="cegis", cycle=cycle)
            break

        # 2. Expert Intervention
        print("[*] Generating expert corrections...")
        fix_and_merge(failures, expert, plant, dataset, args.seq_steps, 0.1)

        # 3. Retrain
        print(f"[*] Retraining on updated dataset (Size: {len(dataset)})...")
        train_loader, val_loader = create_cpu_dataloaders(dataset, args.batch_size)
        metrics = controller.update(
            train_loader, val_loader,
            epochs=5,
            fit_normalizer=(cycle == 1 and not controller.normalizer_fitted),
        )
        t_loss = metrics["train_loss"]
        v_loss = metrics["val_loss"]

        csv_log.write(f"{cycle},{num_fails},{coverage:.1f},{t_loss:.6f},{v_loss:.6f}\n")
        csv_log.flush()

        # Save progress
        controller.save_checkpoint(str(outdir / f"cegis_cycle_{cycle}.pt"), phase="cegis", cycle=cycle)

    csv_log.close()
    controller.save_checkpoint(str(outdir / "last_cegis.pt"), phase="cegis", cycle=args.cegis_cycles)
    print("\n[*] CEGIS Pipeline Finished.")
    return outdir / "best_cegis.pt"


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NNCS-Mamba Robust Trainer")
    parser.add_argument("--phase", choices=["smoke", "imitation", "cegis", "all"], required=True)
    parser.add_argument("--outdir", type=str, default="runs/experiment")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume from")

    # Hyperparameters
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)

    # Data & Training
    parser.add_argument("--num-traj", type=int, default=5000)
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--cegis-cycles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(outdir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize Controller
    print("[*] Initializing Mamba Controller...")
    controller = MambaController(
        obs_dim=12, action_dim=4,
        d_model=args.d_model, d_state=args.d_state,
        num_layers=args.layers, lr=args.lr,
    )
    print(f"    Device: {controller.device}")

    if args.resume:
        print(f"[*] Resuming from checkpoint: {args.resume}")
        controller.load_checkpoint(args.resume)

    # ── Smoke ──
    if args.phase == "smoke":
        run_smoke(controller, args, outdir)
        return

    # ── Dataset Management ──
    # Cache the baseline dataset to avoid re-harvesting (saves 5-10 mins on restart)
    cache_path = Path("baseline_dataset.pt")
    dataset = deque(maxlen=args.num_traj + 1000)
    
    if cache_path.exists() and args.num_traj > 0:
        print(f"[*] Loading cached baseline dataset from {cache_path}...")
        try:
            cached_data = torch.load(cache_path, weights_only=True)
            # If the user requested a different number of trajectories, adjust
            if len(cached_data) > args.num_traj:
                cached_data = cached_data[:args.num_traj]
            dataset.extend(cached_data)
            print(f"    Loaded {len(dataset)} trajectories.")
        except Exception as e:
            print(f"    [!] Failed to load cache: {e}. Re-harvesting...")
            raw_data = generate_expert_data(args.num_traj, args.seq_steps, 0.1)
            dataset.extend(raw_data)
            torch.save(list(dataset), cache_path)
    elif args.num_traj > 0:
        raw_data = generate_expert_data(args.num_traj, args.seq_steps, 0.1)
        dataset.extend(raw_data)
        torch.save(list(dataset), cache_path)
        print(f"[*] Baseline dataset cached to {cache_path}")

    # ── Imitation ──
    if args.phase == "imitation":
        run_imitation(controller, args, outdir, dataset)
        return

    # ── CEGIS ──
    if args.phase == "cegis":
        # If resuming for CEGIS, we might not need to harvest if we just want to hunt
        run_cegis(controller, args, outdir, dataset)
        return

    # ── All: Imitation → CEGIS ──
    run_imitation(controller, args, outdir, dataset)
    run_cegis(controller, args, outdir, dataset)


if __name__ == "__main__":
    main()
