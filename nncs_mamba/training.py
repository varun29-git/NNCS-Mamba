"""Small supervised training loop for the dual-head Mamba controller.

The dataset already contains two kinds of supervision:

- MPC actions, which train the action/control head;
- STL robustness scores, which train the value head.

This first trainer is intentionally simple. It samples fixed-length windows from
the expert trajectories and attaches the trajectory-level STL robustness label
to every step in that window. That is enough to prove the model and dataset are
connected before we add richer value targets or CEGIS.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nncs_mamba.dataset import ROLLOUT_SCORE_KEYS, to_jsonable
from nncs_mamba.models.mamba import DualHeadMambaController, MambaControllerConfig


class ExpertWindowDataset(Dataset):
    """Fixed-length windows cut from MPC expert rollouts."""

    def __init__(
        self,
        dataset_path: Path,
        rollout_indices: np.ndarray,
        sequence_len: int,
        stride: int,
    ) -> None:
        if sequence_len <= 0:
            raise ValueError("sequence_len must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        with np.load(dataset_path) as data:
            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            action_mask = data["action_mask"].astype(bool)
            rollout_scores = data["rollout_scores"].astype(np.float32)

        stl_idx = ROLLOUT_SCORE_KEYS.index("stl_robustness")
        self.states = states
        self.actions = actions
        self.action_mask = action_mask
        self.stl_robustness = rollout_scores[:, stl_idx]
        self.sequence_len = int(sequence_len)

        windows: list[tuple[int, int]] = []
        for rollout_idx in rollout_indices:
            valid_steps = int(np.sum(action_mask[rollout_idx]))
            if valid_steps <= 0:
                continue
            if valid_steps <= sequence_len:
                windows.append((int(rollout_idx), 0))
                continue

            last_start = valid_steps - sequence_len
            for start in range(0, last_start + 1, stride):
                windows.append((int(rollout_idx), int(start)))
            if windows[-1] != (int(rollout_idx), int(last_start)):
                windows.append((int(rollout_idx), int(last_start)))

        if not windows:
            raise ValueError("No training windows could be created from the dataset.")
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rollout_idx, start = self.windows[idx]
        end = start + self.sequence_len

        states = self.states[rollout_idx, start:end]
        actions = self.actions[rollout_idx, start:end]
        mask = self.action_mask[rollout_idx, start:end]

        value = np.full((self.sequence_len, 1), self.stl_robustness[rollout_idx], dtype=np.float32)
        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "value": torch.from_numpy(value),
            "mask": torch.from_numpy(mask.astype(np.float32)).reshape(self.sequence_len, 1),
        }


def split_rollouts(rollout_count: int, validation_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic train/validation split."""

    if rollout_count < 2:
        raise ValueError("At least two rollouts are needed for a train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(rollout_count)
    rng.shuffle(indices)
    validation_count = max(1, int(round(rollout_count * validation_fraction)))
    validation_indices = np.sort(indices[:validation_count])
    train_indices = np.sort(indices[validation_count:])
    if len(train_indices) == 0:
        raise ValueError("Train split is empty. Reduce validation_fraction.")
    return train_indices, validation_indices


def masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a per-step loss over real, non-padding steps."""

    return torch.sum(loss * mask) / torch.clamp(torch.sum(mask), min=1.0)


def run_epoch(
    model: DualHeadMambaController,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    value_weight: float,
    device: torch.device,
) -> dict[str, float]:
    """Run one train or validation epoch."""

    is_train = optimizer is not None
    model.train(is_train)
    action_loss_fn = nn.MSELoss(reduction="none")
    value_loss_fn = nn.HuberLoss(reduction="none")

    totals = {
        "loss": 0.0,
        "action_mse": 0.0,
        "action_mae": 0.0,
        "value_huber": 0.0,
        "value_mae": 0.0,
        "steps": 0.0,
    }

    for batch in loader:
        states = batch["states"].to(device)
        action_target = batch["actions"].to(device)
        value_target = batch["value"].to(device)
        mask = batch["mask"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        action_pred, value_pred = model.forward_torch(states)
        action_mask = mask.expand_as(action_target)
        action_mse = masked_mean(action_loss_fn(action_pred, action_target), action_mask)
        action_mae = masked_mean(torch.abs(action_pred - action_target), action_mask)
        value_huber = masked_mean(value_loss_fn(value_pred, value_target), mask)
        value_mae = masked_mean(torch.abs(value_pred - value_target), mask)
        loss = action_mse + value_weight * value_huber

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        valid_steps = float(torch.sum(mask).item())
        totals["loss"] += float(loss.detach().item()) * valid_steps
        totals["action_mse"] += float(action_mse.detach().item()) * valid_steps
        totals["action_mae"] += float(action_mae.detach().item()) * valid_steps
        totals["value_huber"] += float(value_huber.detach().item()) * valid_steps
        totals["value_mae"] += float(value_mae.detach().item()) * valid_steps
        totals["steps"] += valid_steps

    steps = max(totals.pop("steps"), 1.0)
    return {key: value / steps for key, value in totals.items()}


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dual-head Mamba controller on MPC + STL data.")
    parser.add_argument("--dataset", default="runs/expert_dataset_core_tok64/expert_dataset_core_tok64.npz")
    parser.add_argument("--output-dir", default="runs/mamba_small_train")
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--sequence-len", type=positive_int, default=64)
    parser.add_argument("--stride", type=positive_int, default=32)
    parser.add_argument("--limit-rollouts", type=positive_int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--d-model", type=positive_int, default=128)
    parser.add_argument("--d-state", type=positive_int, default=16)
    parser.add_argument("--layers", type=positive_int, default=3)
    parser.add_argument("--head-hidden-dim", type=positive_int, default=None)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with np.load(dataset_path) as data:
        rollout_count = int(data["states"].shape[0])

    if args.limit_rollouts is not None:
        rollout_count = min(rollout_count, args.limit_rollouts)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    all_indices = np.arange(rollout_count)
    train_indices, validation_indices = split_rollouts(len(all_indices), args.validation_fraction, args.seed)
    train_indices = all_indices[train_indices]
    validation_indices = all_indices[validation_indices]

    train_dataset = ExpertWindowDataset(dataset_path, train_indices, args.sequence_len, args.stride)
    validation_dataset = ExpertWindowDataset(dataset_path, validation_indices, args.sequence_len, args.stride)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    config = MambaControllerConfig(
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.layers,
        head_hidden_dim=args.head_hidden_dim,
    )
    model = DualHeadMambaController(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    summary_path = output_dir / "summary.json"

    print(f"dataset: {dataset_path}")
    print(f"device: {device}")
    print(f"parameters: {parameter_count(model)}")
    print(f"train_windows: {len(train_dataset)}")
    print(f"validation_windows: {len(validation_dataset)}")

    history = []
    best_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, args.value_weight, device)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, None, args.value_weight, device)

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_action_mse={train_metrics['action_mse']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_action_mse={validation_metrics['action_mse']:.6f} "
            f"val_value_mae={validation_metrics['value_mae']:.6f}"
        )

        if validation_metrics["loss"] < best_validation:
            best_validation = validation_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "parameter_count": parameter_count(model),
                    "epoch": epoch,
                    "validation": validation_metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "device": str(device),
        "parameter_count": parameter_count(model),
        "model_config": asdict(config),
        "train_rollouts": train_indices.tolist(),
        "validation_rollouts": validation_indices.tolist(),
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "history": history,
        "best_validation_loss": best_validation,
        "checkpoint": str(checkpoint_path),
    }
    with open(summary_path, "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
