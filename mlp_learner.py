import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from abstract_env import LearnerController


class MLPController(LearnerController, nn.Module):
    """Memoryless MLP baseline trained on the same MPC demonstrations."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 192,
        d_state: int = 16,
        num_layers: int = 3,
        lr: float = 3e-4,
        optimizer_name: str = "adamw",
        use_gradient_checkpointing: bool = False,
        aux_state_weight: float = 0.5,
        late_timestep_weight: float = 1.0,
        recurrent_dropout: float = 0.0,
        action_clip: Optional[float] = None,
    ):
        del d_state, use_gradient_checkpointing, recurrent_dropout
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.noise_std = 0.01
        self.default_epochs = 4
        self.default_batch_size = 64
        self.obs_norm_eps = 1e-6
        self.grad_clip_norm = 1.0
        self.normalizer_fitted = False
        self.optimizer_name = optimizer_name
        self.aux_state_weight = aux_state_weight
        self.late_timestep_weight = late_timestep_weight
        self.action_clip = action_clip

        self.register_buffer("obs_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("obs_scale", torch.ones(obs_dim, dtype=torch.float32))
        self.input_norm = nn.LayerNorm(obs_dim)

        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, d_model), nn.SiLU(), nn.LayerNorm(d_model)])
            in_dim = d_model
        self.net = nn.Sequential(*layers)
        self.action_head = nn.Linear(d_model, action_dim)
        self.state_head = nn.Linear(d_model, 12)

        self.to(self.device)
        if optimizer_name != "adamw":
            raise ValueError("MLPController supports optimizer_name='adamw' only")
        self.optimizer_adamw = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_adamw,
            mode="min",
            factor=0.5,
            patience=3,
        )
        self.use_amp = self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None

    @property
    def current_lr(self) -> float:
        return float(self.optimizer_adamw.param_groups[0]["lr"])

    def get_config(self) -> Dict[str, Any]:
        return {
            "controller_type": "mlp",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "d_state": 0,
            "num_layers": self.num_layers,
            "lr": self.current_lr,
            "noise_std": self.noise_std,
            "grad_clip_norm": self.grad_clip_norm,
            "optimizer_name": self.optimizer_name,
            "use_gradient_checkpointing": False,
            "aux_state_weight": self.aux_state_weight,
            "late_timestep_weight": self.late_timestep_weight,
            "recurrent_dropout": 0.0,
            "action_clip": self.action_clip,
        }

    def reset(self, max_seq_len: int = 5000, max_batch_size: int = 1) -> None:
        del max_seq_len, max_batch_size

    def _normalize_observations(self, batch_obs: torch.Tensor) -> torch.Tensor:
        return (batch_obs - self.obs_mean.view(1, 1, -1)) / self.obs_scale.view(1, 1, -1)

    def _forward_sequence(self, batch_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_obs = self._normalize_observations(batch_obs)
        return self._forward_sequence_from_normalized(normalized_obs)

    def _forward_sequence_from_normalized(self, normalized_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(normalized_obs)
        features = self.net(x)
        return self.action_head(features), self.state_head(features)

    def _build_time_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.late_timestep_weight <= 1.0 or seq_len <= 1:
            return None
        weights = torch.linspace(1.0, self.late_timestep_weight, seq_len, device=device, dtype=dtype)
        return weights / weights.mean()

    def _sequence_mse(self, prediction: torch.Tensor, target: torch.Tensor, time_weights=None) -> torch.Tensor:
        loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        if time_weights is not None:
            loss = loss * time_weights.view(1, -1)
        return loss.mean()

    def _compute_loss(self, pred_act, batch_act, pred_next, batch_next) -> torch.Tensor:
        time_weights = self._build_time_weights(pred_act.shape[1], pred_act.device, pred_act.dtype)
        action_loss = self._sequence_mse(pred_act, batch_act, time_weights=time_weights)
        next_state_loss = self._sequence_mse(pred_next, batch_next, time_weights=time_weights)
        return action_loss + self.aux_state_weight * next_state_loss

    def _move_batch_to_device(self, batch_obs: torch.Tensor, batch_act: torch.Tensor):
        batch_obs = batch_obs.to(self.device, non_blocking=self.device.type == "cuda")
        batch_act = batch_act.to(self.device, non_blocking=self.device.type == "cuda")
        return batch_obs, batch_act

    def _fit_observation_normalizer(self, loader: DataLoader) -> None:
        total_count = 0
        feature_sum = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        feature_sq_sum = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for batch_obs, _, _ in loader:
                batch_obs = batch_obs.to(self.device, non_blocking=self.device.type == "cuda")
                flat_obs = batch_obs.reshape(-1, self.obs_dim)
                feature_sum += flat_obs.sum(dim=0)
                feature_sq_sum += (flat_obs * flat_obs).sum(dim=0)
                total_count += flat_obs.shape[0]
        if total_count == 0:
            return
        mean = feature_sum / total_count
        variance = torch.clamp(feature_sq_sum / total_count - mean * mean, min=self.obs_norm_eps ** 2)
        self.obs_mean.copy_(mean)
        self.obs_scale.copy_(torch.clamp(torch.sqrt(variance), min=self.obs_norm_eps))
        self.normalizer_fitted = True

    def forward(self, y: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
            out, _ = self._forward_sequence(obs_tensor)
            action = out[0, 0, :].cpu().numpy()
            if self.action_clip is None:
                return action
            return np.clip(action, -self.action_clip, self.action_clip)

    def _build_loader_from_dataset(self, dataset, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory=None):
        if dataset is None or len(dataset) == 0:
            return None
        obs_tensor = torch.from_numpy(np.stack([x[0] for x in dataset]).astype(np.float32, copy=False))
        act_tensor = torch.from_numpy(np.stack([x[1] for x in dataset]).astype(np.float32, copy=False))
        tensor_dataset = TensorDataset(obs_tensor[:, :-1, :], act_tensor[:, :-1, :], obs_tensor[:, 1:, :12])
        effective_num_workers = num_workers if self.device.type == "cuda" else 0
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_num_workers,
            pin_memory=self.device.type == "cuda" if pin_memory is None else pin_memory,
        )

    def _evaluate_loader(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return float("nan")
        self.eval()
        total_loss = 0.0
        batches_processed = 0
        with torch.no_grad():
            for batch_obs, batch_act, batch_next in loader:
                batch_obs, batch_act = self._move_batch_to_device(batch_obs, batch_act)
                batch_next = batch_next.to(self.device, non_blocking=True)
                context = torch.amp.autocast("cuda", enabled=self.use_amp) if self.use_amp else torch.inference_mode()
                with context:
                    pred_act, pred_next = self._forward_sequence(batch_obs)
                    loss = self._compute_loss(pred_act, batch_act, pred_next, batch_next)
                total_loss += loss.item()
                batches_processed += 1
        return total_loss / max(1, batches_processed)

    def update(self, dataset_or_loader, val_loader=None, epochs: int = 4, batch_size: int = 32,
               fit_normalizer: bool = True, num_workers: int = 0, pin_memory=None) -> Dict[str, float]:
        train_loader = dataset_or_loader if isinstance(dataset_or_loader, DataLoader) else self._build_loader_from_dataset(
            dataset_or_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        if train_loader is None:
            return {"train_loss": 0.0, "val_loss": float("nan"), "lr": self.current_lr, "avg_grad_norm": 0.0, "epochs": 0.0}
        if fit_normalizer:
            self._fit_observation_normalizer(train_loader)

        self.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        batches_processed = 0
        for _ in range(epochs):
            for batch_obs, batch_act, batch_next in train_loader:
                batch_obs, batch_act = self._move_batch_to_device(batch_obs, batch_act)
                batch_next = batch_next.to(self.device, non_blocking=True)
                self.optimizer_adamw.zero_grad(set_to_none=True)
                normalized_obs = self._normalize_observations(batch_obs)
                noisy_obs = normalized_obs.clone()
                noisy_obs[..., 0:3] = noisy_obs[..., 0:3] + torch.randn_like(noisy_obs[..., 0:3]) * self.noise_std
                context = torch.amp.autocast("cuda", enabled=self.use_amp) if self.use_amp else torch.inference_mode(False)
                with context:
                    pred_act, pred_next = self._forward_sequence_from_normalized(noisy_obs)
                    loss = self._compute_loss(pred_act, batch_act, pred_next, batch_next)
                if self.grad_scaler:
                    self.grad_scaler.scale(loss).backward()
                    self.grad_scaler.unscale_(self.optimizer_adamw)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
                    self.grad_scaler.step(self.optimizer_adamw)
                    self.grad_scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
                    self.optimizer_adamw.step()
                total_loss += loss.item()
                total_grad_norm += float(grad_norm)
                batches_processed += 1
        train_loss = total_loss / max(1, batches_processed)
        val_loss = self._evaluate_loader(val_loader) if val_loader is not None else float("nan")
        self.scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)
        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": self.current_lr,
            "avg_grad_norm": total_grad_norm / max(1, batches_processed),
            "epochs": float(epochs),
        }

    def save_checkpoint(self, path: str, **metadata: Any) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_adamw_state_dict": self.optimizer_adamw.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.get_config(),
                "metadata": metadata,
                "normalizer_fitted": self.normalizer_fitted,
            },
            checkpoint_path,
        )

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_adamw_state_dict" in checkpoint:
            self.optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.normalizer_fitted = bool(checkpoint.get("normalizer_fitted", True))
        return checkpoint
