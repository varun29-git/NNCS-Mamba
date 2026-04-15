import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from abstract_env import LearnerController

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

class MambaController(LearnerController, nn.Module):
    """
    Production-ready Neural Network Controller using `mamba-ssm`.
    Implements proper O(1) causal inference cache parameters, batched sequence learning,
    and device scaling operations.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_model: int = 64,
        d_state: int = 16,
        num_layers: int = 2,
        lr: float = 1e-4,
    ):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        
        self.register_buffer("obs_mean", torch.zeros(obs_dim, dtype=torch.float32))
        self.register_buffer("obs_scale", torch.ones(obs_dim, dtype=torch.float32))
        self.input_norm = nn.LayerNorm(obs_dim)
        self.input_layer = nn.Linear(obs_dim, d_model)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, action_dim)
        
        # Internal state manager for autoregressive steps
        self.inference_params = None

        self.to(self.device)
        
        # Persistent optimizer & loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        self.criterion = nn.MSELoss()
        self.use_amp = self.device.type == "cuda"
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @property
    def current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def get_config(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "d_state": self.d_state,
            "num_layers": self.num_layers,
            "lr": self.current_lr,
            "noise_std": self.noise_std,
            "grad_clip_norm": self.grad_clip_norm,
        }
        
    def reset(self, max_seq_len: int = 5000, max_batch_size: int = 1) -> None:
        """Resets the internal hidden memory cache for autoregressive steps across episodes."""
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.inference_params = InferenceParams(max_seqlen=max_seq_len, max_batch_size=max_batch_size)
        self.inference_params.seqlen_offset = 0

    def _ensure_inference_state(self) -> None:
        if self.inference_params is None:
            self.reset(max_seq_len=self.max_seq_len, max_batch_size=self.max_batch_size)
            return

        if self.inference_params.seqlen_offset >= self.max_seq_len - 1:
            self.reset(max_seq_len=self.max_seq_len, max_batch_size=self.max_batch_size)

    def _encode_inputs(self, batch_obs: torch.Tensor) -> torch.Tensor:
        normalized_obs = self._normalize_observations(batch_obs)
        return self._encode_normalized_inputs(normalized_obs)

    def _encode_normalized_inputs(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        return self.input_layer(self.input_norm(normalized_obs))

    def _normalize_observations(self, batch_obs: torch.Tensor) -> torch.Tensor:
        return (batch_obs - self.obs_mean.view(1, 1, -1)) / self.obs_scale.view(1, 1, -1)

    def _move_batch_to_device(self, batch_obs: torch.Tensor, batch_act: torch.Tensor):
        if batch_obs.device == self.device and batch_act.device == self.device:
            return batch_obs, batch_act
        batch_obs = batch_obs.to(self.device, non_blocking=self.device.type == "cuda")
        batch_act = batch_act.to(self.device, non_blocking=self.device.type == "cuda")
        return batch_obs, batch_act

    def _fit_observation_normalizer(self, loader: DataLoader) -> None:
        total_count = 0
        feature_sum = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)
        feature_sq_sum = torch.zeros(self.obs_dim, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for batch_obs, _ in loader:
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
        for block, norm in zip(self.mamba_blocks, self.norms):
            x = x + block(norm(x), inference_params=self.inference_params)
        self.inference_params.seqlen_offset += 1
        return self.output_layer(x)
        
    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Autoregressive step in O(1) utilizing hardware-efficient cached recurrent transitions.
        """
        self.eval()
        with torch.no_grad():
            self._ensure_inference_state()
            obs_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
            out = self._run_cached_step(obs_tensor)

            return np.clip(out[0, 0, :].cpu().numpy(), -20.0, 20.0)

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

        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        act_tensor = torch.tensor(np.stack(act_list), dtype=torch.float32)
        tensor_dataset = TensorDataset(obs_tensor, act_tensor)
        if pin_memory is None:
            pin_memory = self.device.type == "cuda"
        return DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def _forward_sequence(self, batch_obs: torch.Tensor) -> torch.Tensor:
        x = self._encode_inputs(batch_obs)
        for block, norm in zip(self.mamba_blocks, self.norms):
            x = x + block(norm(x))
        return self.output_layer(x)

    def _evaluate_loader(self, loader: Optional[DataLoader]) -> float:
        if loader is None:
            return float("nan")

        self.eval()
        total_loss = 0.0
        batches_processed = 0

        with torch.no_grad():
            for batch_obs, batch_act in loader:
                batch_obs, batch_act = self._move_batch_to_device(batch_obs, batch_act)
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self._forward_sequence(batch_obs)
                    loss = self.criterion(predictions, batch_act)
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
        """
        Trains on either:
        - a raw dataset of (obs_seq, act_seq) tuples, or
        - a prebuilt training DataLoader plus optional validation DataLoader.
        """
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
            for batch_obs, batch_act in train_loader:
                batch_obs, batch_act = self._move_batch_to_device(batch_obs, batch_act)
                self.optimizer.zero_grad(set_to_none=True)
                normalized_obs = self._normalize_observations(batch_obs)
                noisy_obs = normalized_obs + torch.randn_like(normalized_obs) * self.noise_std

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self._forward_sequence_from_normalized(noisy_obs)
                    loss = self.criterion(predictions, batch_act)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_norm)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

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

    def _forward_sequence_from_normalized(self, normalized_obs: torch.Tensor) -> torch.Tensor:
        x = self._encode_normalized_inputs(normalized_obs)
        for block, norm in zip(self.mamba_blocks, self.norms):
            x = x + block(norm(x))
        return self.output_layer(x)

    def save_checkpoint(self, path: str, **metadata: Any) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.get_config(),
            "metadata": metadata,
            "normalizer_fitted": self.normalizer_fitted,
        }
        torch.save(payload, checkpoint_path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.normalizer_fitted = bool(checkpoint.get("normalizer_fitted", True))
        return checkpoint
 
