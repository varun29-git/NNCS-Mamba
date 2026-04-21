import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint

class MambaCache:
    """
    Storage for recurrent states (h) and convolution buffers (x) 
    to enable O(1) inference.
    """
    def __init__(self, batch_size: int, d_model: int, d_state: int, d_conv: int, device: torch.device):
        self.batch_size = batch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.device = device
        
        # [Batch, Inner_Dim, D_State]
        self.h = torch.zeros(batch_size, d_model * 2, d_state, device=device)
        # [Batch, Inner_Dim, D_Conv]
        self.conv_buffer = torch.zeros(batch_size, d_model * 2, d_conv, device=device)

class MambaBlock(nn.Module):
    """
    Pure PyTorch implementation of the Mamba (S6) block.
    Avoids all custom CUDA dependencies.
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_checkpointing: bool = True,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.use_checkpointing = use_checkpointing

        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = int(dt_rank)

        # Input projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        # Conv1d
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.activation = F.silu

        # S6 Projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S6 Parameters
        # A matrix: [D_Inner, D_State]
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

        # Initialization
        self.dt_init_floor = dt_init_floor
        self._dt_init(dt_min, dt_max, dt_init, dt_scale)

    def _dt_init(self, dt_min, dt_max, dt_init, dt_scale):
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=self.dt_init_floor)
        
        # Inverse softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

    def forward(self, x: torch.Tensor, cache: Optional[MambaCache] = None):
        """
        x: [Batch, SeqLen, D_Model]
        """
        if self.training and cache is None and self.use_checkpointing:
            # Trade compute for memory using gradient checkpointing during parallel scan phase
            # use_reentrant=False is the PyTorch >= 2.0 standard
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x, cache)

    def _forward_impl(self, x: torch.Tensor, cache: Optional[MambaCache] = None):
        """
        x: [Batch, SeqLen, D_Model]
        """
        batch, seq_len, _ = x.shape
        
        # 1. Input Projection
        # [B, L, 2*D_Inner]
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 2. Convolution (handling causal padding)
        # [B, D_Inner, L]
        x = x.transpose(1, 2)
        
        if cache:
            # Shift buffer and add new token
            cache.conv_buffer.copy_(torch.roll(cache.conv_buffer, shifts=-1, dims=-1))
            cache.conv_buffer[:, :, -1] = x[:, :, 0]
            # Use buffer for convolution
            x = F.conv1d(
                cache.conv_buffer,
                self.conv1d.weight,
                self.conv1d.bias,
                groups=self.d_inner
            )
        else:
            x = self.conv1d(x)[:, :, :seq_len]
        
        x = self.activation(x)

        # 3. Selective Scan (S6)
        # [B, L, D_Inner]
        x = x.transpose(1, 2)
        y = self.selective_scan(x, cache)

        # 4. Gating & Out
        y = y * self.activation(z)
        return self.out_proj(y)

    def selective_scan(self, x: torch.Tensor, cache: Optional[MambaCache] = None):
        """
        x: [B, L, D_Inner]
        """
        batch, seq_len, d_inner = x.shape
        
        # Projections
        # [B, L, dt_rank + 2*d_state]
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # [B, L, D_Inner]
        dt = F.softplus(self.dt_proj(dt))
        
        # Discretize A and B
        # A: [D_Inner, D_State]
        A = -torch.exp(self.A_log)
        
        # Explicit FP32 casting to completely bypass AMP and prevent NaN/Inf overflow during exponentiation
        dt_fp32 = dt.float()
        A_fp32 = A.float()
        B_fp32 = B.float()
        C_fp32 = C.float()
        x_fp32 = x.float()
        
        if cache:
            # Sequential/Recurrent mode (Inference)
            # h: [B, D_Inner, D_State]
            # [B, D_Inner, D_State] * [B, D_Inner, 1] -> [B, D_Inner, D_State]
            dA = torch.exp(dt_fp32.transpose(1, 2) * A_fp32)
            dB = dt_fp32.transpose(1, 2) * B_fp32
            
            # Recurrence
            # x: [B, 1, D_Inner] -> [B, D_Inner, 1]
            new_h = dA * cache.h.float() + dB * x_fp32.transpose(1, 2)
            cache.h.copy_(new_h)
            
            # [B, D_Inner, D_State] @ [B, D_State, 1] -> [B, D_Inner, 1]
            y = torch.bmm(new_h, C_fp32.transpose(1, 2)).transpose(1, 2)
        else:
            # Parallel mode (Training)
            # This is a simplified version of the associative scan using pre-broadcasted dA/dB
            # [B, L, D_Inner, D_State]
            dA = torch.exp(dt_fp32.unsqueeze(-1) * A_fp32.unsqueeze(0).unsqueeze(0))
            dB = dt_fp32.unsqueeze(-1) * B_fp32.unsqueeze(-2)
            
            # Recurrent scan
            h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=torch.float32)
            ys = []
            for t in range(seq_len):
                h = dA[:, t] * h + dB[:, t] * x_fp32[:, t].unsqueeze(-1)
                # [B, D_Inner, D_State] @ [B, D_State, 1] -> [B, D_Inner, 1]
                y_t = torch.bmm(h, C_fp32[:, t].unsqueeze(-1))
                ys.append(y_t)
            
            y = torch.stack(ys, dim=1).squeeze(-1)

        # Cast safely back to original precision (FP16 or FP32)
        y = y.to(dtype=x.dtype)

        # Residual connection additive part (D)
        y = y + x * self.D
        return y
