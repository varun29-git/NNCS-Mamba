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
        
        self.input_layer = nn.Linear(obs_dim, d_model)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, action_dim)
        
        # Internal state manager for autoregressive steps
        self.inference_params = None

        self.to(self.device)
        
        # Persistent optimizer & loss function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def reset(self) -> None:
        """Resets the internal hidden memory cache for autoregressive steps across episodes."""
        self.inference_params = InferenceParams(max_seqlen=5000, max_batch_size=1)
        self.inference_params.seqlen_offset = 0 # Vital to reset the seq slice pointer
        
    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Autoregressive step in O(1) utilizing hardware-efficient cached recurrent transitions.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(y, dtype=torch.float32, device=self.device).reshape(1, 1, -1) # (batch=1, seq_len=1, obs_dim)
            x = self.input_layer(x)
            
            if self.inference_params is None:
                self.reset()
                
            # Run the O(1) step updating the cache block-by-block
            for block in self.mamba_blocks:
                x = block(x, inference_params=self.inference_params)
                
            # Crucial: increment the sequence offset so the next step correctly shifts the tensor buffers
            self.inference_params.seqlen_offset += 1
            
            out = self.output_layer(x)
            # Remove seq and batch dims -> (action_dim)
            return out.squeeze().cpu().numpy()
            
    def update(self, dataset, epochs: int = 2, batch_size: int = 16) -> float:
        """
        Imitation learning process utilizing parallel hardware scans and Mini-Batches.
        """
        self.train()
        if len(dataset) == 0:
            return 0.0
            
        # Convert buffer directly to batched tensors
        # Assuming fixed simulated sequence lengths
        obs_list = [item[0] for item in dataset]
        act_list = [item[1] for item in dataset]
        
        obs_tensor = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device) # (B, S, D_obs)
        act_tensor = torch.tensor(np.stack(act_list), dtype=torch.float32, device=self.device) # (B, S, D_act)
        
        train_data = TensorDataset(obs_tensor, act_tensor)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        total_loss = 0.0
        batches_processed = 0
        
        for _ in range(epochs):
            for batch_obs, batch_act in train_loader:
                self.optimizer.zero_grad()
                
                x = self.input_layer(batch_obs)
                for block in self.mamba_blocks:
                    x = block(x) # (B, S, d_model) - Parallel Execution Mode
                    
                predictions = self.output_layer(x) # (B, S, D_act)
                
                loss = self.criterion(predictions, batch_act)
                loss.backward()
                
                # Gradient Clipping for explosive derivatives stability 
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                batches_processed += 1
                
        return total_loss / max(1, batches_processed)
 