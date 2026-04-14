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
        
        # Rigorously allocate memory blocks directly onto CUDA layers
        for block in self.mamba_blocks:
            if hasattr(block, "allocate_inference_cache"):
                # Allocates required kwargs corresponding to causal-conv1d parameters
                block.allocate_inference_cache(batch_size=max_batch_size, max_seqlen=max_seq_len, dtype=torch.bfloat16)

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
            for block in self.mamba_blocks:
                x = block(x, inference_params=self.inference_params)
                
            # Enact internal ring-buffer cache index offset
            self.inference_params.seqlen_offset += 1
            
            out = self.output_layer(x)
            # Remove seq and batch dims defensively via explicit indexing
            return out[0, 0, :].cpu().numpy()
            
    def update(self, train_data_loader: DataLoader, epochs: int = 1) -> float:
        """
        Receives pre-packaged batched DataLoader (containing balanced representations
        of memory + new CEGIS flaws) and scales operations utilizing PyTorch AMP.
        """
        self.train()
        total_loss = 0.0
        batches_processed = 0
        
        # Enforce AMP logic wrapper context explicitly
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        for _ in range(epochs):
            for batch_obs, batch_act in train_data_loader:
                self.optimizer.zero_grad()
                
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=(device_type == 'cuda')):
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
                
                total_loss += loss.item()
                batches_processed += 1
                
        return total_loss / max(1, batches_processed)
