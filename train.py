"""
Master Training Pipeline for NNCS-Mamba

Usage Examples:
1. Smoke Test (Fast check):
   python train.py --phase smoke --outdir runs/smoke

2. Baseline Imitation (The heavy lifting):
   python train.py --phase imitation --num-traj 5000 --batch-size 32 --outdir runs/imitation

2b. Best T4 Budget Run (recommended for 3-hour Colab/T4 sessions):
   python train.py --phase all --profile t4-sota --outdir runs/t4_sota

3. CEGIS Refinement (Hunting edge cases):
   python train.py --phase cegis --resume runs/imitation/best_imitation.pt --cegis-cycles 20

4. Full Pipeline (Imitation → CEGIS, one command):
   python train.py --phase all --outdir runs/full
"""

import argparse
import json
import numpy as np
import torch
import time
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader, TensorDataset

# Enable TF32 for optimal performance on A100 GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from drone_env import DronePlant, DroneExpertController, CHECKPOINTS
from drone_env import CHECKPOINT_RADIUS, FORCE_LIMIT
from controller_factory import build_controller
from cegis_loop import falsify_cem


PROFILE_OVERRIDES = {
    "default": {},
    "t4-fast": {
        "d_model": 96,
        "d_state": 16,
        "layers": 2,
        "lr": 3e-4,
        "batch_size": 16,
        "num_traj": 1500,
        "seq_steps": 200,
        "epochs": 6,
        "cegis_cycles": 1,
        "optimizer": "adamw",
        "num_workers": 2,
        "val_split": 0.10,
        "max_hours": 1.25,
        "cegis_pop_size": 512,
        "cegis_generations": 1,
        "cegis_retrain_epochs": 2,
        "disable_gradient_checkpointing": True,
        "controller": "mamba",
        "late_timestep_weight": 1.0,
        "recurrent_dropout": 0.0,
    },
    "t4-sota": {
        "controller": "gru",
        "d_model": 192,
        "d_state": 16,
        "layers": 2,
        "lr": 3e-4,
        "batch_size": 64,
        "num_traj": 4096,
        "seq_steps": 300,
        "epochs": 12,
        "optimizer": "adamw",
        "num_workers": 2,
        "val_split": 0.10,
        "max_hours": 2.8,
        "cegis_cycles": 1,
        "cegis_pop_size": 256,
        "cegis_generations": 1,
        "cegis_retrain_epochs": 2,
        "disable_gradient_checkpointing": True,
        "late_timestep_weight": 2.5,
        "recurrent_dropout": 0.1,
    },
    "t4-generalize": {
        "controller": "structured_gru",
        "d_model": 256,
        "d_state": 16,
        "layers": 2,
        "lr": 2e-4,
        "batch_size": 16,
        "num_traj": 8192,
        "seq_steps": 300,
        "epochs": 18,
        "optimizer": "adamw",
        "num_workers": 2,
        "val_split": 0.10,
        "max_hours": 2.8,
        "cegis_cycles": 1,
        "cegis_pop_size": 128,
        "cegis_generations": 1,
        "cegis_retrain_epochs": 1,
        "disable_gradient_checkpointing": True,
        "late_timestep_weight": 1.5,
        "recurrent_dropout": 0.1,
    },
}

DATASET_CACHE_VERSION = "v3"


def apply_profile(args, defaults):
    overrides = PROFILE_OVERRIDES.get(args.profile, {})
    for key, value in overrides.items():
        if getattr(args, key) == defaults[key]:
            setattr(args, key, value)


def build_dataset_cache_path(args):
    cache_dir = Path("cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / (
        f"baseline_dataset_{DATASET_CACHE_VERSION}_profile-{args.profile}"
        f"_traj-{args.num_traj}_steps-{args.seq_steps}.pt"
    )


# =============================================================================
# Helper: CPU-Backed Dataloader (Prevents T4 OOM Errors)
# =============================================================================
def create_cpu_dataloaders(dataset, batch_size, val_split=0.15, num_workers=2):
    """Converts the list of trajectories into CPU-bound PyTorch DataLoaders."""
    data_list = list(dataset)
    np.random.shuffle(data_list)

    split_idx = int(len(data_list) * (1 - val_split))
    train_raw = data_list[:split_idx]
    val_raw = data_list[split_idx:]

    def to_dataset(raw_list):
        if not raw_list:
            return None
        o = torch.from_numpy(np.stack([x[0] for x in raw_list]).astype(np.float32, copy=False))
        a = torch.from_numpy(np.stack([x[1] for x in raw_list]).astype(np.float32, copy=False))
        # Sequentially map input causality
        o_in = o[:, :-1, :]
        a_in = a[:, :-1, :]
        next_o = o[:, 1:, 0:12]
        return TensorDataset(o_in, a_in, next_o)

    train_ds = to_dataset(train_raw)
    val_ds = to_dataset(val_raw)
    effective_num_workers = num_workers if torch.cuda.is_available() else 0

    loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": effective_num_workers,
        "persistent_workers": effective_num_workers > 0,
    }
    if effective_num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    ) if val_ds else None

    return train_loader, val_loader


# =============================================================================
# Helper: Generate Baseline Data
# =============================================================================
def generate_expert_data(num_traj, seq_steps, dt):
    print(f"[*] Harvesting {num_traj} expert trajectories (batched fast path)...")
    checkpoints = np.asarray(CHECKPOINTS, dtype=np.float32)
    dataset = []
    chunk_size = min(1024, max(128, num_traj))
    generated = 0

    while generated < num_traj:
        batch = min(chunk_size, num_traj - generated)
        states = np.zeros((batch, 12), dtype=np.float32)
        states[:, 0:3] = np.random.uniform(-10.0, 10.0, size=(batch, 3)).astype(np.float32)
        states[:, 2] += 12.0

        times = np.zeros(batch, dtype=np.float32)
        low_mass_mask = np.random.rand(batch) > 0.5
        if np.any(low_mass_mask):
            times[low_mass_mask] = np.random.uniform(0.0, 60.0, size=int(np.sum(low_mass_mask))).astype(np.float32)

        phase_idx = np.zeros(batch, dtype=np.int64)
        obs_seq = np.empty((batch, seq_steps, 15), dtype=np.float32)
        act_seq = np.empty((batch, seq_steps, 4), dtype=np.float32)

        for step_idx in range(seq_steps):
            targets = checkpoints[phase_idx]
            pos = states[:, 0:3]

            reached_mask = (phase_idx < 3) & (
                np.linalg.norm(pos - targets, axis=1) < CHECKPOINT_RADIUS
            )
            phase_idx = np.where(reached_mask, phase_idx + 1, phase_idx)
            targets = checkpoints[phase_idx]

            vel = states[:, 6:9]
            error = targets - pos
            v_target = 1.5 * error
            acc_desired = 3.0 * (v_target - vel)
            acc_desired[:, 2] += 9.81

            masses = np.maximum(1.0, 2.5 - 0.02 * times).astype(np.float32)
            force_cmd = np.clip(acc_desired * masses[:, None], -FORCE_LIMIT, FORCE_LIMIT).astype(np.float32)
            action = np.zeros((batch, 4), dtype=np.float32)
            action[:, 0:3] = force_cmd

            obs_seq[:, step_idx, :12] = states
            obs_seq[:, step_idx, 12:] = targets
            act_seq[:, step_idx] = action

            fx, fy, fz, yaw_rate = action[:, 0], action[:, 1], action[:, 2], action[:, 3]
            ax = fx / masses
            ay = fy / masses
            az = fz / masses
            wind = np.where(np.sin(2.0 * np.pi * times / 5.0) > 0.8, 5.0 / masses, 0.0).astype(np.float32)

            phi = states[:, 3]
            theta = states[:, 4]
            vx = states[:, 6]
            vy = states[:, 7]
            vz = states[:, 8]
            p = states[:, 9]
            q = states[:, 10]

            derivatives = np.stack(
                [
                    vx,
                    vy,
                    vz,
                    p,
                    q,
                    states[:, 11],
                    9.81 * theta + wind + ax,
                    -9.81 * phi + ay,
                    az - 9.81,
                    -10.0 * phi - 2.0 * p,
                    -10.0 * theta - 2.0 * q,
                    yaw_rate,
                ],
                axis=1,
            ).astype(np.float32)
            derivatives += np.random.normal(0.0, 0.05, size=(batch, 12)).astype(np.float32)

            states = states + derivatives * dt
            times = times + dt

        dataset.extend((obs_seq[i], act_seq[i]) for i in range(batch))
        generated += batch
        if generated % 500 == 0 or generated == num_traj:
            print(f"    ...generated {generated}/{num_traj}")

    return dataset


def fix_and_merge(failures, expert, plant, buffer, seq_steps, dt):
    """Takes failure states, has the expert fix them, and adds to buffer."""
    for state in failures:
        plant.state = state.copy()
        plant.time = 0.0
        expert.reset()
        expert.set_plant_ref(plant)
        
        o_seq = np.empty((seq_steps, 15), dtype=np.float32)
        a_seq = np.empty((seq_steps, 4), dtype=np.float32)
        y = state.copy()
        for step_idx in range(seq_steps):
            u = expert.compute_action(y)
            target = CHECKPOINTS[expert.phase_idx]
            o_seq[step_idx, :12] = y
            o_seq[step_idx, 12:] = target
            a_seq[step_idx] = u
            y = plant.step(u, dt)
            
        buffer.append((o_seq, a_seq))



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


def run_imitation(controller, args, outdir, dataset, global_start_time):
    """Pure imitation learning from expert demonstrations."""
    print("\n=== PHASE: BASELINE IMITATION ===")
    train_loader, val_loader = create_cpu_dataloaders(
        dataset,
        args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    best_val_loss = float("inf")
    csv_log = open(outdir / "imitation_log.csv", "w")
    csv_log.write("epoch,train_loss,val_loss,lr\n")

    for epoch in range(1, args.epochs + 1):
        if time.time() - global_start_time > args.max_hours * 3600:
            print(f"[!] Time limit reached ({args.max_hours:.2f} hours). Gracefully exiting Imitation Phase...")
            break
            
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


def run_cegis(controller, args, outdir, dataset, global_start_time):
    """CEGIS refinement — falsify, fix, retrain."""
    print("\n=== PHASE: CEGIS REFINEMENT ===")

    plant = DronePlant()
    expert = DroneExpertController()
    expert.set_plant_ref(plant)
    
    cegis_buffer = []

    csv_log = open(outdir / "cegis_log.csv", "w")
    csv_log.write("cycle,fails,coverage,train_loss,val_loss\n")
    best_coverage = -float("inf")
    best_val_loss = float("inf")

    for cycle in range(1, args.cegis_cycles + 1):
        if time.time() - global_start_time > args.max_hours * 3600:
            print(f"[!] Time limit reached ({args.max_hours:.2f} hours). Gracefully exiting CEGIS Phase...")
            break
        
        print(f"\n--- CEGIS Cycle {cycle}/{args.cegis_cycles} ---")

        # 1. Hunt for failures
        print("[*] Falsifier hunting for edge cases...")
        failures = falsify_cem(
            controller, plant, expert,
            num_generations=args.cegis_generations,
            pop_size=args.cegis_pop_size,
            elite_frac=0.2, seq_steps=args.seq_steps, dt=0.1,
        )

        num_fails = len(failures)
        coverage = (1.0 - min(1.0, num_fails / float(args.cegis_pop_size))) * 100
        print(f"    Found {num_fails} unique failures. Safety Coverage: {coverage:.2f}%")

        if num_fails == 0:
            print("\n✔️  CEGIS Converged! 100% Safety Coverage achieved.")
            controller.save_checkpoint(str(outdir / "best_cegis.pt"), phase="cegis", cycle=cycle)
            break

        # 2. Expert Intervention
        print("[*] Generating expert corrections...")
        fix_and_merge(failures, expert, plant, cegis_buffer, args.seq_steps, 0.1)

        # 3. Retrain (Prioritized Experience Replay)
        target_cegis_size = int(len(dataset) * (0.3 / 0.7))
        mixed_dataset = list(dataset)
        if len(cegis_buffer) > 0:
            import random
            upsampled_cegis = random.choices(cegis_buffer, k=target_cegis_size)
            mixed_dataset.extend(upsampled_cegis)

        print(f"[*] Retraining PER (Size: {len(mixed_dataset)} | Baseline: {len(dataset)} | Recovery: {len(mixed_dataset)-len(dataset)})...")
        train_loader, val_loader = create_cpu_dataloaders(
            mixed_dataset,
            args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
        )
        metrics = controller.update(
            train_loader, val_loader,
            epochs=args.cegis_retrain_epochs,
            fit_normalizer=(cycle == 1 and not controller.normalizer_fitted),
        )
        t_loss = metrics["train_loss"]
        v_loss = metrics["val_loss"]

        csv_log.write(f"{cycle},{num_fails},{coverage:.1f},{t_loss:.6f},{v_loss:.6f}\n")
        csv_log.flush()

        # Save progress
        controller.save_checkpoint(str(outdir / f"cegis_cycle_{cycle}.pt"), phase="cegis", cycle=cycle)
        meaningful_coverage = coverage > 0.0
        if meaningful_coverage and (coverage > best_coverage or (coverage == best_coverage and v_loss < best_val_loss)):
            best_coverage = coverage
            best_val_loss = v_loss
            controller.save_checkpoint(str(outdir / "best_cegis.pt"), phase="cegis", cycle=cycle)
            print("   -> Saved new best CEGIS checkpoint.")
        elif not meaningful_coverage:
            print("   -> CEGIS coverage stayed at 0.00%; keeping imitation checkpoint as the recommended model.")

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
    parser.add_argument("--profile", choices=sorted(PROFILE_OVERRIDES.keys()), default="default")
    parser.add_argument("--outdir", type=str, default="runs/experiment")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume from")
    parser.add_argument("--controller", choices=["mamba", "gru", "structured_gru"], default="mamba")

    # Hyperparameters ('T4 Blitz' — Scaled for 3-hour window)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optimizer", choices=["split_muon", "adamw"], default="split_muon")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--late-timestep-weight", type=float, default=1.0)
    parser.add_argument("--recurrent-dropout", type=float, default=0.1)

    # Data & Training
    parser.add_argument("--num-traj", type=int, default=5000)
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cegis-cycles", type=int, default=5)
    parser.add_argument("--cegis-pop-size", type=int, default=2000)
    parser.add_argument("--cegis-generations", type=int, default=3)
    parser.add_argument("--cegis-retrain-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-hours", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)

    defaults = {
        "controller": "mamba",
        "d_model": 128,
        "d_state": 16,
        "layers": 3,
        "lr": 3e-4,
        "batch_size": 32,
        "optimizer": "split_muon",
        "disable_gradient_checkpointing": False,
        "late_timestep_weight": 1.0,
        "recurrent_dropout": 0.1,
        "num_traj": 5000,
        "seq_steps": 300,
        "epochs": 10,
        "cegis_cycles": 5,
        "cegis_pop_size": 2000,
        "cegis_generations": 3,
        "cegis_retrain_epochs": 5,
        "num_workers": 2,
        "val_split": 0.15,
        "max_hours": 2.5,
    }
    args = parser.parse_args()
    apply_profile(args, defaults)
    if args.controller in {"gru", "structured_gru"} and args.optimizer != "adamw":
        print("[*] Switching optimizer to AdamW for recurrent controller.")
        args.optimizer = "adamw"
    
    global_start_time = time.time()

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
    print("[*] Initializing controller...")
    # Increase obs_dim to 15 to account for concatenated target coordinates (12D state + 3D target)
    controller = build_controller(
        controller_type=args.controller,
        obs_dim=15,
        action_dim=4,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.layers,
        lr=args.lr,
        optimizer_name=args.optimizer,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        late_timestep_weight=args.late_timestep_weight,
        recurrent_dropout=args.recurrent_dropout,
    )
    print(f"    Device: {controller.device}")
    print(
        f"    Profile: {args.profile} | Controller: {args.controller} | "
        f"Optimizer: {args.optimizer} | Gradient checkpointing: {not args.disable_gradient_checkpointing}"
    )
    print(f"    Late-timestep loss weight: {args.late_timestep_weight:.2f}")

    if args.controller == "mamba":
        # NOTE: torch.compile disabled — CUDA Graphs consume ~1.2GB extra VRAM
        # which causes OOM on A100 MIG 3g.20gb (19.5GB) slices. The sequential
        # for-loop in selective_scan causes graph breaks anyway, negating benefits.
        pass

    if args.resume:
        print(f"[*] Resuming from checkpoint: {args.resume}")
        controller.load_checkpoint(args.resume)

    # ── Smoke ──
    if args.phase == "smoke":
        run_smoke(controller, args, outdir)
        return

    # ── Dataset Management ──
    # Cache the baseline dataset to avoid re-harvesting (saves 5-10 mins on restart)
    cache_path = build_dataset_cache_path(args)
    dataset = deque(maxlen=args.num_traj + 1000)
    
    if cache_path.exists() and args.num_traj > 0:
        print(f"[*] Loading cached baseline dataset from {cache_path}...")
        try:
            cached_data = torch.load(cache_path, weights_only=False)
            
            sample_obs = cached_data[0][0]
            if sample_obs.shape[-1] != 15 or sample_obs.shape[0] != args.seq_steps:
                print(
                    f"[!] Cache shape mismatch ({tuple(sample_obs.shape)} != ({args.seq_steps}, 15)). "
                    "Invalidating cache..."
                )
                raise ValueError("Old cache format")
                
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
        run_imitation(controller, args, outdir, dataset, global_start_time)
        return

    # ── CEGIS ──
    if args.phase == "cegis":
        # If resuming for CEGIS, we might not need to harvest if we just want to hunt
        run_cegis(controller, args, outdir, dataset, global_start_time)
        return

    # ── All: Imitation → CEGIS ──
    run_imitation(controller, args, outdir, dataset, global_start_time)
    run_cegis(controller, args, outdir, dataset, global_start_time)


if __name__ == "__main__":
    main()
