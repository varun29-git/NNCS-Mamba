"""
Master Training Pipeline for NNCS-Mamba

Usage Examples:
1. Smoke Test (Fast check):
   python train.py --phase smoke --outdir runs/smoke

2. Baseline Imitation (The heavy lifting):
   python train.py --phase imitation --num-traj 5000 --batch-size 32 --outdir runs/imitation

3. Budget GRU Baseline:
   python train.py --phase imitation --profile t4-sota --outdir runs/t4_sota
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

from controller_factory import build_controller
from safe_control_gym_config import (
    SAFE_CONTROL_GYM_MPC_CONFIG,
    SAFE_CONTROL_GYM_SOURCE,
    SAFE_CONTROL_GYM_TASK_CONFIG,
    SAFE_CONTROL_GYM_VERSION,
    make_env_and_mpc,
    reset_gym_env,
    task_config_for_initial_state,
    action_bounds,
    clip_to_env_action_space,
    get_goal_position,
    step_gym_env,
    write_benchmark_manifest,
)
from stl_monitor import STLSpec, evaluate_stabilization_stl


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
        "optimizer": "adamw",
        "num_workers": 2,
        "val_split": 0.10,
        "max_hours": 1.25,
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
        "disable_gradient_checkpointing": True,
        "late_timestep_weight": 2.5,
        "recurrent_dropout": 0.1,
    },
    "mlp-baseline": {
        "controller": "mlp",
        "d_model": 192,
        "d_state": 16,
        "layers": 3,
        "lr": 3e-4,
        "batch_size": 64,
        "num_traj": 4096,
        "seq_steps": 300,
        "epochs": 12,
        "optimizer": "adamw",
        "num_workers": 2,
        "val_split": 0.10,
        "max_hours": 2.8,
        "disable_gradient_checkpointing": True,
        "late_timestep_weight": 1.0,
        "recurrent_dropout": 0.0,
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
        f"baseline_dataset_{DATASET_CACHE_VERSION}_safe_control_gym_profile-{args.profile}"
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


def generate_safe_control_gym_expert_data(num_traj, seq_steps, args):
    print(f"[*] Harvesting {num_traj} Safe-Control-Gym MPC trajectories...")
    env, mpc = make_env_and_mpc(str(Path(args.outdir) / "safe_control_gym"), args.seed)
    write_benchmark_manifest(Path(args.outdir) / "benchmark_manifest.json", env)

    dataset = []
    mpc_wall_times = []
    try:
        for traj_idx in range(num_traj):
            obs, info = reset_gym_env(env)
            mpc.reset_before_run(obs=obs, info=info, env=env)

            obs_seq = np.empty((seq_steps, 12), dtype=np.float32)
            act_seq = np.empty((seq_steps, 4), dtype=np.float32)

            for step_idx in range(seq_steps):
                action_start = time.perf_counter()
                action = np.asarray(mpc.select_action(obs, info), dtype=np.float32)
                mpc_wall_times.append(time.perf_counter() - action_start)
                obs_seq[step_idx] = np.asarray(obs, dtype=np.float32)
                act_seq[step_idx] = action

                obs, _, done, info = step_gym_env(env, action)
                if done and step_idx < seq_steps - 1:
                    obs_seq[step_idx + 1:] = np.asarray(obs, dtype=np.float32)
                    act_seq[step_idx + 1:] = 0.0
                    break

            dataset.append((obs_seq, act_seq))
            if (traj_idx + 1) % 100 == 0 or traj_idx + 1 == num_traj:
                print(f"    ...generated {traj_idx + 1}/{num_traj}")
    finally:
        mpc.close()
        env.close()

    if mpc_wall_times:
        print(f"    MPC label time: {np.mean(mpc_wall_times) * 1000.0:.2f} ms/action")
    return dataset


def generate_safe_control_gym_expert_data_from_states(initial_states, seq_steps, args):
    dataset = []
    for idx, initial_state in enumerate(initial_states):
        fixed_config = task_config_for_initial_state(initial_state)
        env, mpc = make_env_and_mpc(
            str(Path(args.outdir) / "safe_control_gym_cegis"),
            args.seed + idx,
            task_config=fixed_config,
        )
        try:
            obs, info = reset_gym_env(env)
            mpc.reset_before_run(obs=obs, info=info, env=env)
            obs_seq = np.empty((seq_steps, 12), dtype=np.float32)
            act_seq = np.empty((seq_steps, 4), dtype=np.float32)
            for step_idx in range(seq_steps):
                action = np.asarray(mpc.select_action(obs, info), dtype=np.float32)
                obs_seq[step_idx] = np.asarray(obs, dtype=np.float32)
                act_seq[step_idx] = action
                obs, _, done, info = step_gym_env(env, action)
                if done and step_idx < seq_steps - 1:
                    obs_seq[step_idx + 1:] = np.asarray(obs, dtype=np.float32)
                    act_seq[step_idx + 1:] = 0.0
                    break
            dataset.append((obs_seq, act_seq))
        finally:
            mpc.close()
            env.close()
    return dataset


def generate_dataset(args, num_traj=None, seq_steps=None):
    num_traj = args.num_traj if num_traj is None else num_traj
    seq_steps = args.seq_steps if seq_steps is None else seq_steps
    return generate_safe_control_gym_expert_data(num_traj, seq_steps, args)


def checkpoint_metadata(args, **metadata):
    metadata.update({
        "plant_backend": "safe-control-gym",
        "plant_source": SAFE_CONTROL_GYM_SOURCE,
        "safe_control_gym_version": SAFE_CONTROL_GYM_VERSION,
        "expert_controller": "safe-control-gym mpc",
        "safe_control_gym_task_config": SAFE_CONTROL_GYM_TASK_CONFIG,
        "safe_control_gym_mpc_config": SAFE_CONTROL_GYM_MPC_CONFIG,
    })
    return metadata



# =============================================================================
# Phase Runners
# =============================================================================
def run_smoke(controller, args, outdir):
    """Quick sanity check — kernels compile, backward pass works."""
    print("\n=== PHASE: SMOKE TEST ===")
    print("Testing kernel compilation and backward pass...")
    dataset = generate_dataset(args, num_traj=10, seq_steps=100)
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
            controller.save_checkpoint(str(outdir / "best_imitation.pt"), **checkpoint_metadata(args, phase="imitation", epoch=epoch))
            print("   -> Saved new best checkpoint.")

    csv_log.close()
    controller.save_checkpoint(str(outdir / "last_imitation.pt"), **checkpoint_metadata(args, phase="imitation", epoch=args.epochs))
    print("\n[*] Imitation Phase Complete.")
    return outdir / "best_imitation.pt"


def find_counterexample_initial_states(controller, args):
    env, mpc = make_env_and_mpc(str(Path(args.outdir) / "safe_control_gym_falsify"), args.seed)
    action_low, action_high = action_bounds(env)
    failures = []
    robustness_values = []
    try:
        for _ in range(args.cegis_pop_size):
            obs, info = reset_gym_env(env)
            initial_obs = np.asarray(obs, dtype=np.float32).copy()
            controller.reset()
            states = [initial_obs]
            actions = []
            for _ in range(args.seq_steps):
                action = np.asarray(controller.forward(np.asarray(obs, dtype=np.float32)), dtype=np.float32)
                action = clip_to_env_action_space(env, action)
                actions.append(action)
                obs, _, done, info = step_gym_env(env, action)
                states.append(np.asarray(obs, dtype=np.float32))
                if done:
                    break
            stl = evaluate_stabilization_stl(
                np.stack(states),
                np.stack(actions) if actions else None,
                STLSpec(goal_position=get_goal_position(env)),
                action_low,
                action_high,
            )
            robustness_values.append(stl["stl_robustness"])
            if stl["stl_robustness"] < 0.0:
                failures.append((stl["stl_robustness"], initial_obs))
    finally:
        mpc.close()
        env.close()

    failures.sort(key=lambda item: item[0])
    selected = [state for _, state in failures[: args.cegis_max_failures]]
    return selected, robustness_values


def run_cegis(controller, args, outdir, dataset, global_start_time):
    print("\n=== PHASE: CEGIS REFINEMENT ===")
    recovery_buffer = []
    csv_log = open(outdir / "cegis_log.csv", "w")
    csv_log.write("iteration,failures,mean_stl_robustness,min_stl_robustness,train_loss,val_loss\n")
    try:
        for iteration in range(1, args.cegis_iterations + 1):
            if time.time() - global_start_time > args.max_hours * 3600:
                print(f"[!] Time limit reached ({args.max_hours:.2f} hours). Exiting CEGIS.")
                break
            failures, robustness_values = find_counterexample_initial_states(controller, args)
            mean_robustness = float(np.mean(robustness_values)) if robustness_values else float("nan")
            min_robustness = float(np.min(robustness_values)) if robustness_values else float("nan")
            print(
                f"CEGIS iteration {iteration:02d}: failures={len(failures)} "
                f"mean_STL={mean_robustness:.3f} min_STL={min_robustness:.3f}"
            )
            if not failures:
                controller.save_checkpoint(
                    str(outdir / "best_cegis.pt"),
                    **checkpoint_metadata(args, phase="cegis", cegis_iteration=iteration),
                )
                break

            recovery_buffer.extend(generate_safe_control_gym_expert_data_from_states(failures, args.seq_steps, args))
            mixed_dataset = list(dataset) + recovery_buffer
            train_loader, val_loader = create_cpu_dataloaders(
                mixed_dataset,
                args.batch_size,
                val_split=args.val_split,
                num_workers=args.num_workers,
            )
            metrics = controller.update(
                train_loader,
                val_loader,
                epochs=args.cegis_retrain_epochs,
                fit_normalizer=not controller.normalizer_fitted,
            )
            csv_log.write(
                f"{iteration},{len(failures)},{mean_robustness:.6f},{min_robustness:.6f},"
                f"{metrics['train_loss']:.6f},{metrics['val_loss']:.6f}\n"
            )
            csv_log.flush()
            controller.save_checkpoint(
                str(outdir / f"cegis_iteration_{iteration}.pt"),
                **checkpoint_metadata(args, phase="cegis", cegis_iteration=iteration),
            )
    finally:
        csv_log.close()
    controller.save_checkpoint(
        str(outdir / "last_cegis.pt"),
        **checkpoint_metadata(args, phase="cegis", cegis_iteration=args.cegis_iterations),
    )
    return outdir / "last_cegis.pt"


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NNCS-Mamba Robust Trainer")
    parser.add_argument("--phase", choices=["smoke", "imitation", "cegis", "all"], required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_OVERRIDES.keys()), default="default")
    parser.add_argument("--outdir", type=str, default="runs/experiment")
    parser.add_argument("--resume", type=str, default=None, help="Path to .pt checkpoint to resume from")
    parser.add_argument("--controller", choices=["mamba", "gru", "mlp"], default="mamba")

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
    parser.add_argument("--cegis-iterations", type=int, default=3)
    parser.add_argument("--cegis-pop-size", type=int, default=50)
    parser.add_argument("--cegis-max-failures", type=int, default=10)
    parser.add_argument("--cegis-retrain-epochs", type=int, default=2)
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
        "cegis_iterations": 3,
        "cegis_pop_size": 50,
        "cegis_max_failures": 10,
        "cegis_retrain_epochs": 2,
        "num_workers": 2,
        "val_split": 0.15,
        "max_hours": 2.5,
    }
    args = parser.parse_args()
    apply_profile(args, defaults)
    if args.controller in {"gru", "mlp"} and args.optimizer != "adamw":
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
    controller = build_controller(
        controller_type=args.controller,
        obs_dim=12,
        action_dim=4,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.layers,
        lr=args.lr,
        optimizer_name=args.optimizer,
        use_gradient_checkpointing=not args.disable_gradient_checkpointing,
        late_timestep_weight=args.late_timestep_weight,
        recurrent_dropout=args.recurrent_dropout,
        action_clip=None,
    )
    print(f"    Device: {controller.device}")
    print(
        f"    Profile: {args.profile} | Controller: {args.controller} | "
        f"Plant: Safe-Control-Gym quadrotor | Expert: Safe-Control-Gym MPC | "
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
            if sample_obs.shape[-1] != 12 or sample_obs.shape[0] != args.seq_steps:
                print(
                    f"[!] Cache shape mismatch ({tuple(sample_obs.shape)} != ({args.seq_steps}, 12)). "
                    "Invalidating cache..."
                )
                raise ValueError("Old cache format")
                
            if len(cached_data) > args.num_traj:
                cached_data = cached_data[:args.num_traj]
            dataset.extend(cached_data)
            print(f"    Loaded {len(dataset)} trajectories.")
        except Exception as e:
            print(f"    [!] Failed to load cache: {e}. Re-harvesting...")
            raw_data = generate_dataset(args)
            dataset.extend(raw_data)
            torch.save(list(dataset), cache_path)
    elif args.num_traj > 0:
        raw_data = generate_dataset(args)
        dataset.extend(raw_data)
        torch.save(list(dataset), cache_path)
        print(f"[*] Baseline dataset cached to {cache_path}")

    # ── Imitation ──
    if args.phase == "imitation":
        run_imitation(controller, args, outdir, dataset, global_start_time)
        return

    if args.phase == "cegis":
        run_cegis(controller, args, outdir, dataset, global_start_time)
        return

    if args.phase == "all":
        run_imitation(controller, args, outdir, dataset, global_start_time)
        run_cegis(controller, args, outdir, dataset, global_start_time)
        return


if __name__ == "__main__":
    main()
