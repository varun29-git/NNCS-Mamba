import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from abstract_env import LearnerController
from drone_env import FORCE_LIMIT


class StructuredGRUController(LearnerController, nn.Module):
    """
    Higher-capacity recurrent controller with physics-informed action synthesis.

    Instead of asking the network to emit force commands from scratch, it learns
    a latent state that estimates hidden mass and adds a bounded residual on top
    of an analytic PD-style controller. This is a much stronger inductive bias
    for the drone task and tends to generalize better per unit of wall clock.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 2,
        lr: float = 2e-4,
        optimizer_name: str = "adamw",
        use_gradient_checkpointing: bool = False,
        aux_state_weight: float = 0.5,
        late_timestep_weight: float = 1.0,
        recurrent_dropout: float = 0.1,
        mass_loss_weight: float = 0.35,
        residual_loss_weight: float = 0.01,
        residual_force_limit: float = 6.0,
        min_mass: float = 1.0,
        max_mass: float = 2.5,
    ):
        nn.Module.__init__(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.max_seq_len = 5000
        self.max_batch_size = 1
        self.noise_std = 0.01
        self.default_epochs = 4
        self.default_batch_size = 32
        self.obs_norm_eps = 1e-6
        self.grad_clip_norm = 1.0
        self.normalizer_fitted = False
        self.optimizer_name = optimizer_name
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.aux_state_weight = aux_state_weight
        self.late_timestep_weight = late_timestep_weight
        self.recurrent_dropout = recurrent_dropout
        self.mass_loss_weight = mass_loss_weight
        self.residual_loss_weight = residual_loss_weight
        self.residual_force_limit = residual_force_limit
        self.min_mass = min_mass
        self.max_mass = max_mass

        self.register_buffer("obs_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("obs_scale", torch.ones(obs_dim, dtype=torch.float32))
        self.input_norm = nn.LayerNorm(obs_dim)
        self.input_layer = nn.Linear(obs_dim, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout if num_layers > 1 else 0.0,
        )
        self.post_norm = nn.LayerNorm(d_model)
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        self.mass_head = nn.Linear(d_model, 1)
        self.residual_head = nn.Linear(d_model, action_dim)
        self.state_head = nn.Linear(d_model, 12)

        self.hidden_state: Optional[torch.Tensor] = None
        self._init_weights()
        self.to(self.device)

        if optimizer_name != "adamw":
            raise ValueError("StructuredGRUController currently supports optimizer_name='adamw' only")
        self.optimizer_adamw = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_adamw,
            mode="min",
            factor=0.5,
            patience=3,
        )
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.grad_scaler = None

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for module in self.post_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.mass_head.weight)
        nn.init.zeros_(self.mass_head.bias)
        nn.init.xavier_uniform_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.xavier_uniform_(self.state_head.weight)
        nn.init.zeros_(self.state_head.bias)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer_adamw.param_groups[0]["lr"])

    def get_config(self) -> Dict[str, Any]:
        return {
            "controller_type": "structured_gru",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "num_layers": self.num_layers,
            "lr": self.current_lr,
            "noise_std": self.noise_std,
            "grad_clip_norm": self.grad_clip_norm,
            "optimizer_name": self.optimizer_name,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "aux_state_weight": self.aux_state_weight,
            "late_timestep_weight": self.late_timestep_weight,
            "recurrent_dropout": self.recurrent_dropout,
            "mass_loss_weight": self.mass_loss_weight,
            "residual_loss_weight": self.residual_loss_weight,
            "residual_force_limit": self.residual_force_limit,
            "min_mass": self.min_mass,
            "max_mass": self.max_mass,
        }

    def reset(self, max_seq_len: int = 5000, max_batch_size: int = 1) -> None:
        del max_seq_len
        self.max_batch_size = max_batch_size
        self.hidden_state = torch.zeros(
            self.num_layers,
            max_batch_size,
            self.d_model,
            device=self.device,
            dtype=torch.float32,
        )

    def _ensure_inference_state(self) -> None:
        if self.hidden_state is None:
            self.reset(max_batch_size=self.max_batch_size)

    def _normalize_observations(self, batch_obs: torch.Tensor) -> torch.Tensor:
        return (batch_obs - self.obs_mean.view(1, 1, -1)) / self.obs_scale.view(1, 1, -1)

    def _encode_normalized_inputs(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        return self.input_layer(self.input_norm(normalized_obs))

    def _encode_inputs(self, batch_obs: torch.Tensor) -> torch.Tensor:
        normalized_obs = self._normalize_observations(batch_obs)
        return self._encode_normalized_inputs(normalized_obs)

    def _refine_features(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.post_mlp(self.post_norm(features))

    def _analytic_acceleration(self, batch_obs: torch.Tensor) -> torch.Tensor:
        pos = batch_obs[..., 0:3]
        vel = batch_obs[..., 6:9]
        target = batch_obs[..., 12:15]
        error = target - pos
        v_target = 1.5 * error
        acc_desired = 3.0 * (v_target - vel)
        acc_desired[..., 2] = acc_desired[..., 2] + 9.81
        return acc_desired

    def _estimate_mass(self, features: torch.Tensor) -> torch.Tensor:
        mass_unit = torch.sigmoid(self.mass_head(features))
        return self.min_mass + (self.max_mass - self.min_mass) * mass_unit

    def _compose_action(
        self,
        batch_obs: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mass = self._estimate_mass(features)
        acc_desired = self._analytic_acceleration(batch_obs)

        baseline = torch.zeros(
            batch_obs.shape[0],
            batch_obs.shape[1],
            self.action_dim,
            device=batch_obs.device,
            dtype=batch_obs.dtype,
        )
        baseline[..., 0:3] = acc_desired * mass

        residual = self.residual_force_limit * torch.tanh(self.residual_head(features))
        action = torch.clamp(baseline + residual, -FORCE_LIMIT, FORCE_LIMIT)
        return action, mass, residual

    def _forward_sequence(self, batch_obs: torch.Tensor):
        normalized_obs = self._normalize_observations(batch_obs)
        return self._forward_sequence_from_normalized(batch_obs, normalized_obs)

    def _forward_sequence_from_normalized(self, raw_obs: torch.Tensor, normalized_obs: torch.Tensor):
        x = self._encode_normalized_inputs(normalized_obs)
        x, _ = self.gru(x)
        x = self._refine_features(x)
        pred_act, pred_mass, residual = self._compose_action(raw_obs, x)
        pred_next = self.state_head(x)
        return pred_act, pred_next, pred_mass, residual

    def _move_batch_to_device(self, batch_obs: torch.Tensor, batch_act: torch.Tensor):
        if batch_obs.device == self.device and batch_act.device == self.device:
            return batch_obs, batch_act
        batch_obs = batch_obs.to(self.device, non_blocking=self.device.type == "cuda")
        batch_act = batch_act.to(self.device, non_blocking=self.device.type == "cuda")
        return batch_obs, batch_act

    def _build_time_weights(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if self.late_timestep_weight <= 1.0 or seq_len <= 1:
            return None
        weights = torch.linspace(1.0, self.late_timestep_weight, seq_len, device=device, dtype=dtype)
        return weights / weights.mean()

    def _sequence_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        if time_weights is not None:
            loss = loss * time_weights.view(1, -1)
        return loss.mean()

    def _mass_supervision(
        self,
        batch_obs: torch.Tensor,
        batch_act: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acc_desired = self._analytic_acceleration(batch_obs)
        acc_z = acc_desired[..., 2]
        act_z = batch_act[..., 2]
        valid_mask = (acc_z.abs() > 1.0) & (act_z.abs() < (FORCE_LIMIT - 0.5))
        pseudo_mass = act_z / torch.clamp(acc_z, min=1e-3)
        pseudo_mass = torch.clamp(pseudo_mass, min=self.min_mass, max=self.max_mass)
        return pseudo_mass.unsqueeze(-1), valid_mask.unsqueeze(-1)

    def _masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=prediction.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        diff = ((prediction - target) ** 2) * mask_f
        return diff.sum() / denom

    def _compute_loss(
        self,
        batch_obs: torch.Tensor,
        batch_act: torch.Tensor,
        pred_act: torch.Tensor,
        pred_next: torch.Tensor,
        batch_next: torch.Tensor,
        pred_mass: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        time_weights = self._build_time_weights(pred_act.shape[1], pred_act.device, pred_act.dtype)
        action_loss = self._sequence_mse(pred_act, batch_act, time_weights=time_weights)
        next_state_loss = self._sequence_mse(pred_next, batch_next, time_weights=time_weights)

        pseudo_mass, valid_mask = self._mass_supervision(batch_obs, batch_act)
        mass_loss = self._masked_mse(pred_mass, pseudo_mass, valid_mask)

        residual_penalty = residual.pow(2).mean()
        return (
            action_loss
            + self.aux_state_weight * next_state_loss
            + self.mass_loss_weight * mass_loss
            + self.residual_loss_weight * residual_penalty
        )

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
        std = torch.sqrt(variance)

        self.obs_mean.copy_(mean)
        self.obs_scale.copy_(torch.clamp(std, min=self.obs_norm_eps))
        self.normalizer_fitted = True

    def _run_cached_step(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        x = self._encode_inputs(obs_tensor)
        self._ensure_inference_state()
        assert self.hidden_state is not None
        hidden_state = self.hidden_state
        if hidden_state.shape[1] != obs_tensor.shape[0]:
            self.reset(max_batch_size=obs_tensor.shape[0])
            hidden_state = self.hidden_state
        x, hidden_state = self.gru(x, hidden_state)
        self.hidden_state = hidden_state.detach()
        x = self._refine_features(x)
        pred_act, _, _ = self._compose_action(obs_tensor, x)
        return pred_act

    def forward(self, y: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
            out = self._run_cached_step(obs_tensor)
            return np.clip(out[0, 0, :].cpu().numpy(), -FORCE_LIMIT, FORCE_LIMIT)

    def _build_loader_from_dataset(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
    ) -> Optional[DataLoader]:
        if dataset is None or len(dataset) == 0:
            return None

        obs_list = [item[0] for item in dataset]
        act_list = [item[1] for item in dataset]

        obs_tensor = torch.from_numpy(np.stack(obs_list).astype(np.float32, copy=False))
        act_tensor = torch.from_numpy(np.stack(act_list).astype(np.float32, copy=False))
        obs_in = obs_tensor[:, :-1, :]
        act_in = act_tensor[:, :-1, :]
        next_obs = obs_tensor[:, 1:, :12]
        tensor_dataset = TensorDataset(obs_in, act_in, next_obs)
        if pin_memory is None:
            pin_memory = self.device.type == "cuda"
        effective_num_workers = num_workers if self.device.type == "cuda" else 0
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
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
                    pred_act, pred_next, pred_mass, residual = self._forward_sequence(batch_obs)
                    loss = self._compute_loss(batch_obs, batch_act, pred_act, pred_next, batch_next, pred_mass, residual)
                total_loss += loss.item()
                batches_processed += 1

        return total_loss / max(1, batches_processed)

    def update(
        self,
        dataset_or_loader,
        val_loader=None,
        epochs: int = 4,
        batch_size: int = 32,
        fit_normalizer: bool = True,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
    ) -> Dict[str, float]:
        if isinstance(dataset_or_loader, DataLoader):
            train_loader = dataset_or_loader
        else:
            train_loader = self._build_loader_from_dataset(
                dataset_or_loader,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )

        if train_loader is None:
            return {
                "train_loss": 0.0,
                "val_loss": float("nan"),
                "lr": self.current_lr,
                "avg_grad_norm": 0.0,
                "epochs": 0.0,
            }

        if fit_normalizer:
            self._fit_observation_normalizer(train_loader)
        self.train()
        total_loss = 0.0
        batches_processed = 0
        total_grad_norm = 0.0

        for _ in range(epochs):
            for batch_obs, batch_act, batch_next in train_loader:
                batch_obs, batch_act = self._move_batch_to_device(batch_obs, batch_act)
                batch_next = batch_next.to(self.device, non_blocking=True)

                self.optimizer_adamw.zero_grad(set_to_none=True)

                noisy_obs = batch_obs.clone()
                noisy_obs[..., 0:3] = noisy_obs[..., 0:3] + torch.randn_like(noisy_obs[..., 0:3]) * self.noise_std
                normalized_noisy_obs = self._normalize_observations(noisy_obs)

                context = torch.amp.autocast("cuda", enabled=self.use_amp) if self.use_amp else torch.inference_mode(False)
                with context:
                    pred_act, pred_next, pred_mass, residual = self._forward_sequence_from_normalized(noisy_obs, normalized_noisy_obs)
                    loss = self._compute_loss(batch_obs, batch_act, pred_act, pred_next, batch_next, pred_mass, residual)

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
        avg_grad_norm = total_grad_norm / max(1, batches_processed)

        validation_loss = float("nan")
        if val_loader is not None:
            validation_loss = self._evaluate_loader(val_loader)
            self.scheduler.step(validation_loss if not np.isnan(validation_loss) else train_loss)
        else:
            self.scheduler.step(train_loss)

        return {
            "train_loss": train_loss,
            "val_loss": validation_loss,
            "lr": self.current_lr,
            "avg_grad_norm": avg_grad_norm,
            "epochs": float(epochs),
        }

    def save_checkpoint(self, path: str, **metadata: Any) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "optimizer_adamw_state_dict": self.optimizer_adamw.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.get_config(),
            "metadata": metadata,
            "normalizer_fitted": self.normalizer_fitted,
        }
        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_adamw_state_dict" in checkpoint:
            self.optimizer_adamw.load_state_dict(checkpoint["optimizer_adamw_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.normalizer_fitted = bool(checkpoint.get("normalizer_fitted", True))
        return checkpoint
