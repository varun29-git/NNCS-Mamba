import torch
import torch.nn as nn
from typing import Any, List, Optional, Tuple
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
    def __init__(self, obs_dim: int, action_dim: int, d_model: int = 64, d_state: int = 16, num_layers: int = 2, lr: float = 1e-4):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = 5000
        self.max_batch_size = 1
        self.noise_std = 0.01
        self.default_epochs = 4
        
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
        return self.input_layer(self.input_norm(batch_obs))

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

    def _build_loader_from_dataset(self, dataset, batch_size: int, shuffle: bool) -> Optional[DataLoader]:
        if dataset is None or len(dataset) == 0:
            return None

        obs_list = [item[0] for item in dataset]
        act_list = [item[1] for item in dataset]

        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(np.stack(act_list), dtype=torch.float32, device=self.device)
        tensor_dataset = TensorDataset(obs_tensor, act_tensor)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

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
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self._forward_sequence(batch_obs)
                    loss = self.criterion(predictions, batch_act)
                total_loss += loss.item()
                batches_processed += 1

        return total_loss / max(1, batches_processed)

    def update(self, dataset_or_loader, val_loader=None, epochs: int = 4, batch_size: int = 32):
        """
        Trains on either:
        - a raw dataset of (obs_seq, act_seq) tuples, or
        - a prebuilt training DataLoader plus optional validation DataLoader.

        Returns:
            float: train loss when called with a raw dataset
            Tuple[float, float]: (train_loss, val_loss) when a validation loader is provided
        """
        if isinstance(dataset_or_loader, DataLoader):
            train_loader = dataset_or_loader
            return_tuple = val_loader is not None
        else:
            train_loader = self._build_loader_from_dataset(dataset_or_loader, batch_size=batch_size, shuffle=True)
            return_tuple = False

        if train_loader is None:
            return (0.0, float("nan")) if return_tuple else 0.0

        self.train()
        total_loss = 0.0
        batches_processed = 0

        for _ in range(epochs):
            for batch_obs, batch_act in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                noisy_obs = batch_obs + torch.randn_like(batch_obs) * self.noise_std

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    predictions = self._forward_sequence(noisy_obs)
                    loss = self.criterion(predictions, batch_act)

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                total_loss += loss.item()
                batches_processed += 1

        train_loss = total_loss / max(1, batches_processed)

        validation_loss = float("nan")
        if return_tuple:
            validation_loss = self._evaluate_loader(val_loader)
            self.scheduler.step(validation_loss if not np.isnan(validation_loss) else train_loss)
            return train_loss, validation_loss

        self.scheduler.step(train_loss)

        return train_loss
 
