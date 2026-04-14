import torch
import torch.nn as nn
from typing import List, Tuple, Any
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from abstract_env import LearnerController

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

class MambaController(LearnerController, nn.Module):
    """
    Production Neural Network Controller utilizing advanced hardware inference
    mechanisms and PyTorch Mixed Precision loops for scalable CUDA scaling.
    """
    def __init__(self, obs_dim: int, action_dim: int, d_model: int = 64, d_state: int = 16, num_layers: int = 2, lr: float = 1e-4):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        self.input_layer = nn.Linear(obs_dim, d_model)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, action_dim)
        
        # Internal auto-regressive state manager
        self.inference_params = None

        self.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        # Standard AMP scaling utility for modern GPUs
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def reset(self) -> None:
        """
        Pre-allocates precise contiguous cache memory required for extended recursive Mamba logic
        without drifting / OOM crashes during single-sequence $O(1)$ control.
        """
        max_batch_size = 1
        max_seq_len = 5000
        
        self.inference_params = InferenceParams(max_seqlen=max_seq_len, max_batch_size=max_batch_size)
        self.inference_params.key_value_memory_dict = {}
        
        # Rigorously allocate memory blocks directly onto CUDA layers resolving Tensor dimension leaks
        for i, block in enumerate(self.mamba_blocks):
            if hasattr(block, "allocate_inference_cache"):
                # Native datatype extraction ensuring Mamba inference dtype matches PyTorch params
                cache = block.allocate_inference_cache(
                    batch_size=max_batch_size, 
                    max_seqlen=max_seq_len, 
                    dtype=next(self.parameters()).dtype,
                    device=self.device
                )
                self.inference_params.key_value_memory_dict[i] = cache

        self.inference_params.seqlen_offset = 0 
        
    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Causally shifts $y$ integrating across explicitly allocated hidden boundary caches.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(1, 1, -1) 
            x = self.input_layer(x)
            
            if self.inference_params is None:
                self.reset()
                
            # Single sequence evaluation (B=1, L=1, D)
            for i, block in enumerate(self.mamba_blocks):
                x = block(x, inference_params=self.inference_params)
                
            # Enact internal ring-buffer cache index offset
            self.inference_params.seqlen_offset += 1
            
            out = self.output_layer(x)
            # Remove seq and batch dims defensively via explicit indexing
            return out[0, 0, :].cpu().numpy()
            
    def update(self, train_data_loader: DataLoader, val_data_loader: DataLoader = None, epochs: int = 1) -> Tuple[float, float]:
        """
        Receives pre-packaged batched DataLoader datasets integrating proper validation checks 
        to track Mamba generalizability vs Overfitting index matrices.
        """
        self.train()
        total_train_loss = 0.0
        batches_processed = 0
        
        # Enforce AMP logic wrapper context explicitly
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for _ in range(epochs):
            for batch_obs, batch_act in train_data_loader:
                self.optimizer.zero_grad()
                
                # Defaulting to float16 prevents compatibility crashes on older architectures
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                    x = self.input_layer(batch_obs)
                    for block in self.mamba_blocks:
                        x = block(x) # Parallel Full-Scan mode execution
                        
                    predictions = self.output_layer(x)
                    loss = self.criterion(predictions, batch_act)
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer) 
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_train_loss += loss.item()
                batches_processed += 1
                
        avg_train_loss = total_train_loss / max(1, batches_processed)
        
        avg_val_loss = 0.0
        # Formal offline Validation Matrix
        if val_data_loader is not None and len(val_data_loader) > 0:
            self.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_obs, v_act in val_data_loader:
                    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                        x = self.input_layer(v_obs)
                        for block in self.mamba_blocks:
                            x = block(x)
                        preds = self.output_layer(x)
                        v_loss = self.criterion(preds, v_act)
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / max(1, val_batches)
            
        return avg_train_loss, avg_val_loss
