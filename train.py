import argparse
import json
from collections import deque
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from cegis_loop import (
    falsify_cem,
    fix_and_merge,
    generate_expert_demonstrations,
    prepare_dataloaders,
    run_validation_rollouts,
)
from drone_env import DroneExpertController, DronePlant
from mamba_learner import MambaController


DatasetType = List[Tuple[np.ndarray, np.ndarray]]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_total_training_units(args) -> int:
    phase = args.phase
    if phase == "smoke":
        return max(1, args.smoke_epochs)
    if phase == "imitation":
        return max(1, args.epochs)
    if phase == "cegis":
        return max(1, args.cegis_cycles)
    return max(1, args.epochs + args.cegis_cycles)


class ProgressTracker:
    def __init__(self, total_units: int):
        self.total_units = max(1, total_units)
        self.completed_units = 0
        self.start_time = time.time()

    def advance(self, units: int = 1) -> None:
        self.completed_units = min(self.total_units, self.completed_units + units)

    def eta_string(self) -> str:
        elapsed = time.time() - self.start_time
        if self.completed_units <= 0:
            return "estimating..."
        avg_time_per_unit = elapsed / self.completed_units
        remaining_units = max(0, self.total_units - self.completed_units)
        return format_seconds(avg_time_per_unit * remaining_units)

    def elapsed_string(self) -> str:
        return format_seconds(time.time() - self.start_time)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(path: Path, dataset: DatasetType) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        [
            (
                np.asarray(obs_seq, dtype=np.float32),
                np.asarray(act_seq, dtype=np.float32),
            )
            for obs_seq, act_seq in dataset
        ],
        path,
    )


def load_dataset(path: Path) -> DatasetType:
    raw_dataset = torch.load(path, map_location="cpu", weights_only=False)
    dataset = []
    for obs_seq, act_seq in raw_dataset:
        dataset.append(
            (
                np.asarray(obs_seq, dtype=np.float32),
                np.asarray(act_seq, dtype=np.float32),
            )
        )
    return dataset


def ensure_dataset(
    dataset_path: Path,
    num_trajectories: int,
    seq_steps: int,
    dt: float,
) -> DatasetType:
    if dataset_path.exists():
        print(f"Loading dataset from {dataset_path}")
        return load_dataset(dataset_path)

    print(f"Generating baseline dataset with {num_trajectories} trajectories...")
    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
    expert = DroneExpertController()
    dataset = generate_expert_demonstrations(
        plant,
        expert,
        num_trajectories=num_trajectories,
        seq_steps=seq_steps,
        dt=dt,
    )
    save_dataset(dataset_path, dataset)
    print(f"Saved dataset to {dataset_path}")
    return dataset


def append_metrics(log_path: Path, payload: Dict[str, float]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_controller(args) -> MambaController:
    return MambaController(
        obs_dim=12,
        action_dim=4,
        d_model=args.d_model,
        d_state=args.d_state,
        num_layers=args.num_layers,
        lr=args.lr,
    )


def maybe_resume(controller: MambaController, resume_path: Optional[str]) -> Optional[Dict]:
    if not resume_path:
        return None
    print(f"Resuming checkpoint from {resume_path}")
    return controller.load_checkpoint(resume_path)


def checkpoint_score(rollout_metrics: Dict[str, float]) -> float:
    return (
        2.0 * rollout_metrics["docking_success_rate"]
        + rollout_metrics["checkpoint_order_success_rate"]
        - 0.01 * rollout_metrics["avg_final_distance"]
    )


def save_model_checkpoint(
    controller: MambaController,
    checkpoint_path: Path,
    phase: str,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    controller.save_checkpoint(
        str(checkpoint_path),
        phase=phase,
        epoch=epoch,
        metrics=metrics,
    )


def run_smoke_phase(args, outdir: Path, tracker: Optional[ProgressTracker] = None) -> Path:
    print("=== Phase 1: Smoke Test ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    controller = build_controller(args)
    maybe_resume(controller, args.resume)

    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
    expert = DroneExpertController()
    dataset = generate_expert_demonstrations(
        plant,
        expert,
        num_trajectories=args.smoke_trajectories,
        seq_steps=args.smoke_seq_steps,
        dt=args.dt,
    )
    train_loader, val_loader = prepare_dataloaders(
        dataset,
        latest_demos=[],
        batch_size=min(args.batch_size, 16),
        val_split=0.2,
        num_workers=args.num_workers,
        pin_memory=controller.device.type == "cuda",
    )
    metrics = controller.update(
        train_loader,
        val_loader,
        epochs=args.smoke_epochs,
        fit_normalizer=not controller.normalizer_fitted,
    )
    rollout_metrics = run_validation_rollouts(
        controller,
        plant,
        num_missions=min(3, args.validation_rollouts),
        seq_steps=args.smoke_seq_steps,
        dt=args.dt,
    )
    merged = {
        "phase": "smoke",
        "epoch": args.smoke_epochs,
        **metrics,
        **rollout_metrics,
    }
    append_metrics(outdir / "metrics.jsonl", merged)
    checkpoint_path = outdir / "smoke_last.pt"
    save_model_checkpoint(controller, checkpoint_path, "smoke", args.smoke_epochs, merged)
    print(
        f"Smoke test complete | train={metrics['train_loss']:.4f} "
        f"| val={metrics['val_loss']:.4f} "
        f"| dock={rollout_metrics['docking_success_rate']:.2f}"
    )
    if tracker is not None:
        tracker.advance(args.smoke_epochs)
        print(f"Overall progress | elapsed={tracker.elapsed_string()} | ETA={tracker.eta_string()}")
    return checkpoint_path


def run_imitation_phase(
    args,
    outdir: Path,
    resume_path: Optional[str] = None,
    tracker: Optional[ProgressTracker] = None,
) -> Path:
    print("=== Phase 2: Pure Imitation ===")
    controller = build_controller(args)
    maybe_resume(controller, resume_path or args.resume)

    dataset = ensure_dataset(args.dataset_path, args.num_trajectories, args.seq_steps, args.dt)
    train_loader, val_loader = prepare_dataloaders(
        dataset,
        latest_demos=[],
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        pin_memory=controller.device.type == "cuda",
    )

    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
    best_score = -float("inf")
    best_checkpoint = outdir / "best_imitation.pt"
    last_checkpoint = outdir / "last_imitation.pt"
    phase_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        metrics = controller.update(
            train_loader,
            val_loader,
            epochs=1,
            fit_normalizer=(epoch == 1 and not controller.normalizer_fitted),
        )

        rollout_metrics = {
            "checkpoint_order_success_rate": float("nan"),
            "docking_success_rate": float("nan"),
            "avg_final_distance": float("nan"),
            "avg_checkpoints_reached": float("nan"),
        }
        if epoch % args.validation_rollout_interval == 0 or epoch == args.epochs:
            rollout_metrics = run_validation_rollouts(
                controller,
                plant,
                num_missions=args.validation_rollouts,
                seq_steps=args.seq_steps,
                dt=args.dt,
            )

        merged = {"phase": "imitation", "epoch": epoch, **metrics, **rollout_metrics}
        append_metrics(outdir / "metrics.jsonl", merged)
        save_model_checkpoint(controller, last_checkpoint, "imitation", epoch, merged)

        current_score = checkpoint_score(rollout_metrics) if not np.isnan(rollout_metrics["docking_success_rate"]) else -float("inf")
        if current_score > best_score:
            best_score = current_score
            save_model_checkpoint(controller, best_checkpoint, "imitation", epoch, merged)

        epoch_duration = time.time() - epoch_start
        remaining_epochs = args.epochs - epoch
        phase_eta = format_seconds((time.time() - phase_start) / epoch * remaining_epochs)
        if tracker is not None:
            tracker.advance(1)
            overall_eta = tracker.eta_string()
            overall_elapsed = tracker.elapsed_string()
        else:
            overall_eta = "n/a"
            overall_elapsed = format_seconds(time.time() - phase_start)
        print(
            f"Epoch {epoch:03d} | train={metrics['train_loss']:.4f} | "
            f"val={metrics['val_loss']:.4f} | lr={metrics['lr']:.2e} | "
            f"dock={rollout_metrics['docking_success_rate']:.2f} | "
            f"checkpoints={rollout_metrics['checkpoint_order_success_rate']:.2f} | "
            f"epoch_time={format_seconds(epoch_duration)} | "
            f"phase_eta={phase_eta} | overall_eta={overall_eta} | elapsed={overall_elapsed}"
        )

    return best_checkpoint if best_checkpoint.exists() else last_checkpoint


def run_cegis_phase(
    args,
    outdir: Path,
    resume_path: Optional[str] = None,
    tracker: Optional[ProgressTracker] = None,
) -> Path:
    print("=== Phase 3: CEGIS Refinement ===")
    controller = build_controller(args)
    maybe_resume(controller, resume_path or args.resume)

    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
    expert = DroneExpertController()
    expert.set_plant_ref(plant)

    base_dataset = ensure_dataset(args.dataset_path, args.num_trajectories, args.seq_steps, args.dt)
    dataset = deque(base_dataset, maxlen=max(len(base_dataset) + 1000, 4000))
    latest_demos = []
    best_checkpoint = outdir / "best_cegis.pt"
    last_checkpoint = outdir / "last_cegis.pt"
    best_score = -float("inf")
    phase_start = time.time()

    for cycle in range(1, args.cegis_cycles + 1):
        cycle_start = time.time()
        train_loader, val_loader = prepare_dataloaders(
            dataset,
            latest_demos,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            pin_memory=controller.device.type == "cuda",
        )

        metrics = controller.update(
            train_loader,
            val_loader,
            epochs=args.cegis_epochs,
            fit_normalizer=(cycle == 1 and not controller.normalizer_fitted),
        )
        rollout_metrics = run_validation_rollouts(
            controller,
            plant,
            num_missions=args.validation_rollouts,
            seq_steps=args.seq_steps,
            dt=args.dt,
        )
        failures = falsify_cem(
            controller,
            plant,
            expert,
            num_generations=args.cem_generations,
            pop_size=args.cem_population,
            seq_steps=args.seq_steps,
            dt=args.dt,
        )
        latest_demos = fix_and_merge(
            failures,
            expert,
            plant,
            dataset,
            seq_steps=args.seq_steps,
            dt=args.dt,
        )

        merged = {
            "phase": "cegis",
            "cycle": cycle,
            "failures": float(len(failures)),
            "dataset_size": float(len(dataset)),
            **metrics,
            **rollout_metrics,
        }
        append_metrics(outdir / "metrics.jsonl", merged)
        save_model_checkpoint(controller, last_checkpoint, "cegis", cycle, merged)

        current_score = checkpoint_score(rollout_metrics) - 0.05 * len(failures)
        if current_score > best_score:
            best_score = current_score
            save_model_checkpoint(controller, best_checkpoint, "cegis", cycle, merged)

        cycle_duration = time.time() - cycle_start
        remaining_cycles = args.cegis_cycles - cycle
        phase_eta = format_seconds((time.time() - phase_start) / cycle * remaining_cycles)
        if tracker is not None:
            tracker.advance(1)
            overall_eta = tracker.eta_string()
            overall_elapsed = tracker.elapsed_string()
        else:
            overall_eta = "n/a"
            overall_elapsed = format_seconds(time.time() - phase_start)
        print(
            f"Cycle {cycle:02d} | train={metrics['train_loss']:.4f} | "
            f"val={metrics['val_loss']:.4f} | failures={len(failures)} | "
            f"dock={rollout_metrics['docking_success_rate']:.2f} | "
            f"cycle_time={format_seconds(cycle_duration)} | "
            f"phase_eta={phase_eta} | overall_eta={overall_eta} | elapsed={overall_elapsed}"
        )
        if not failures:
            print("CEGIS converged with zero discovered failures.")
            break

    return best_checkpoint if best_checkpoint.exists() else last_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train NNCS-Mamba on a single T4-friendly pipeline.")
    parser.add_argument("--phase", choices=["smoke", "imitation", "cegis", "all"], default="imitation")
    parser.add_argument("--dataset-path", type=Path, default=Path("artifacts/baseline_dataset.pt"))
    parser.add_argument("--outdir", type=Path, default=Path("runs/default"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--num-trajectories", type=int, default=5000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--validation-rollouts", type=int, default=5)
    parser.add_argument("--validation-rollout-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-trajectories", type=int, default=32)
    parser.add_argument("--smoke-seq-steps", type=int, default=120)
    parser.add_argument("--smoke-epochs", type=int, default=3)
    parser.add_argument("--cegis-cycles", type=int, default=20)
    parser.add_argument("--cegis-epochs", type=int, default=5)
    parser.add_argument("--cem-generations", type=int, default=1)
    parser.add_argument("--cem-population", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    tracker = ProgressTracker(estimate_total_training_units(args))

    checkpoint_path = None
    if args.phase == "smoke":
        run_smoke_phase(args, args.outdir, tracker=tracker)
        return

    if args.phase == "imitation":
        run_imitation_phase(args, args.outdir, tracker=tracker)
        return

    if args.phase == "cegis":
        run_cegis_phase(args, args.outdir, tracker=tracker)
        return

    checkpoint_path = run_imitation_phase(args, args.outdir, tracker=tracker)
    run_cegis_phase(args, args.outdir, resume_path=str(checkpoint_path), tracker=tracker)


if __name__ == "__main__":
    main()
