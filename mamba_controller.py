import torch
import torch.nn as nn
from typing import List, Tuple, Any
import numpy as np

from abstract_env import LearnerController

from mamba_ssm import Mamba
# Note: inference params is required for the O(1) recurrent step inference
from mamba_ssm.utils.generation import InferenceParams


class MambaController(LearnerController, nn.Module):
    """
    Neural Network Controller using the official `mamba-ssm` architecture.
    During training, it processes entire trajectory sequences in parallel (O(log L) parallel scan).
    During online evaluation, it processes observations autoregressively in O(1) time using InferenceParams.
    """
    def __init__(self, obs_dim: int, action_dim: int, d_model: int = 64, d_state: int = 16, num_layers: int = 2):
        nn.Module.__init__(self)
            
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        self.input_layer = nn.Linear(obs_dim, d_model)
        
        # Mamba module handles the parallel scan and recurrent steps entirely
        # Stacking Mamba blocks can be done sequentially
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(d_model, action_dim)
        
        # Internal state manager for autoregressive steps
        self.inference_params = None
        
    def reset(self) -> None:
        """Resets the internal hidden memory cache for autoregressive steps across episodes."""
        # InferenceParams pre-allocates cache sizes used by `mamba.step`
        # max_seqlen bounds the recurrent rollout cache size if causal convolution is active. 
        # For infinite continuous control loops, we can set max_seqlen sufficiently large.
        self.inference_params = InferenceParams(max_seqlen=2000, max_batch_size=1)
        
    def forward(self, y: np.ndarray) -> np.ndarray:
        """
        Autoregressive step in O(1) utilizing hardware-efficient cached recurrent transitions.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(y, dtype=torch.float32).unsqueeze(0) # (1, obs_dim)
            x = self.input_layer(x)
            
            if self.inference_params is None:
                self.reset()
                
            # Run the O(1) step
            for block in self.mamba_blocks:
                x = block.step(x, self.inference_params)
                
            out = self.output_layer(x)
            return out.squeeze(0).numpy()
            
    def update(self, dataset: List[Tuple[np.ndarray, np.ndarray]]) -> Any:
        """
        Imitation learning process utilizing parallel hardware scans.
        It processes full trajectory sequences at once without nested Python for-loops.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        self.train()
        total_loss = 0.0
        
        for obs_seq, act_seq in dataset:
            optimizer.zero_grad()
            
            # (1, seq_len, obs_dim) -> explicit batch dimension for parallel scan
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0) 
            act_tensor = torch.tensor(act_seq, dtype=torch.float32).unsqueeze(0) 
            
            # 1. Parallel sequence processing
            x = self.input_layer(obs_tensor)
            
            # 2. Parallel Hardware Scan over the full sequence (The key benefit of Mamba)
            for block in self.mamba_blocks:
                x = block(x)
                
            # 3. Output predictions
            predictions = self.output_layer(x) # (1, seq_len, act_dim)
            
            loss = criterion(predictions, act_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataset) if dataset else 0.0
