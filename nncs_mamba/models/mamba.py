"""Dual-head Mamba-style controller.

This is the first trainable controller core for the project. It uses a small
structured state-space block instead of an attention block. The block scans a
sequence during training and stores a fixed-size state during online control.

The important deployment property is the cached step:

```text
action, value, cache = controller.step(obs, cache)
```

For a fixed model size, that step updates only the fixed-size cache. It does
not reprocess the whole history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from nncs_mamba.models.controller import (
    ACTION_PRED_KEY,
    VALUE_PRED_KEY,
    ControllerCache,
    ControllerSpec,
    as_single_observation,
    as_state_sequence,
)
from nncs_mamba.models.heads import ActionHead, ValueHead
from nncs_mamba.safe_control_gym_config import ACTION_LABELS, STATE_LABELS


@dataclass(frozen=True)
class MambaControllerConfig:
    """Dimensions for the first dual-head structured-SSM controller."""

    state_dim: int = len(STATE_LABELS)
    action_dim: int = len(ACTION_LABELS)
    value_dim: int = 1
    d_model: int = 64
    d_state: int = 8
    n_layers: int = 2
    head_hidden_dim: int | None = None
    dropout: float = 0.0


class SelectiveSSMBlock(nn.Module):
    """A compact selective state-space block.

    Each layer keeps a hidden state shaped `(batch, d_model, d_state)`. At every
    time step, the current input chooses how much of the old state to keep and
    how much new information to write. That is the Mamba-like part: the memory
    update is structured, recurrent, and input-dependent.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)

        self.norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Linear(d_model, 2 * d_model)
        self.state_proj = nn.Linear(d_model, d_model * d_state)
        self.decay_proj = nn.Linear(d_model, d_model)
        self.base_decay = nn.Parameter(torch.zeros(d_model))
        self.readout = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def empty_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self.d_state, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one time step and update the fixed-size SSM state."""

        z = self.norm(x)
        drive, gate = self.input_proj(z).chunk(2, dim=-1)
        drive = torch.nn.functional.silu(drive)
        gate = torch.nn.functional.silu(gate)

        update = self.state_proj(drive).reshape(x.shape[0], self.d_model, self.d_state)
        decay = torch.sigmoid(self.base_decay.reshape(1, self.d_model) + self.decay_proj(z)).unsqueeze(-1)
        next_state = decay * state + (1.0 - decay) * update

        mixed = torch.sum(next_state * self.readout.reshape(1, self.d_model, self.d_state), dim=-1)
        y = self.output_proj(mixed * gate)
        return x + self.dropout(y), next_state

    def forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scan a full sequence and return the output sequence plus final state."""

        if x.ndim != 3:
            raise ValueError(f"Expected x shaped (batch, time, d_model), got {tuple(x.shape)}.")

        state = self.empty_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        outputs = []
        for step_idx in range(x.shape[1]):
            y, state = self.step(x[:, step_idx], state)
            outputs.append(y)
        return torch.stack(outputs, dim=1), state


class DualHeadMambaController(nn.Module):
    """Shared structured-SSM encoder with action and STL value heads."""

    def __init__(
        self,
        config: MambaControllerConfig | None = None,
        action_low: np.ndarray | torch.Tensor | None = None,
        action_high: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MambaControllerConfig()
        self.spec = ControllerSpec(
            name="dual_head_mamba",
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            value_dim=self.config.value_dim,
        )

        self.input_norm = nn.LayerNorm(self.config.state_dim)
        self.input_proj = nn.Linear(self.config.state_dim, self.config.d_model)
        self.blocks = nn.ModuleList([
            SelectiveSSMBlock(
                d_model=self.config.d_model,
                d_state=self.config.d_state,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.n_layers)
        ])
        self.output_norm = nn.LayerNorm(self.config.d_model)
        self.action_head = ActionHead(
            input_dim=self.config.d_model,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.head_hidden_dim,
            action_low=None if action_low is None else torch.as_tensor(action_low, dtype=torch.float32),
            action_high=None if action_high is None else torch.as_tensor(action_high, dtype=torch.float32),
        )
        self.value_head = ValueHead(
            input_dim=self.config.d_model,
            value_dim=self.config.value_dim,
            hidden_dim=self.config.head_hidden_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def _check_torch_sequence(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            states = states.reshape(1, states.shape[0], states.shape[1])
        if states.ndim != 3:
            raise ValueError(
                f"Expected states shaped (time, {self.spec.state_dim}) or "
                f"(batch, time, {self.spec.state_dim}), got {tuple(states.shape)}."
            )
        if states.shape[-1] != self.spec.state_dim:
            raise ValueError(f"Expected final state dimension {self.spec.state_dim}, got {states.shape[-1]}.")
        return states

    def _check_torch_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.reshape(1, obs.shape[0])
        if obs.ndim != 2 or obs.shape[-1] != self.spec.state_dim:
            raise ValueError(f"Expected obs shaped ({self.spec.state_dim},) or (batch, {self.spec.state_dim}), got {tuple(obs.shape)}.")
        return obs

    def forward_torch(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Trainable full-sequence path.

        Training code should call this method because it keeps PyTorch tensors
        and gradients alive.
        """

        states = self._check_torch_sequence(states.to(device=self.device, dtype=self.dtype))
        x = self.input_proj(self.input_norm(states))
        for block in self.blocks:
            x, _ = block.forward_sequence(x)
        features = self.output_norm(x)
        return self.action_head(features), self.value_head(features)

    def forward(self, states: torch.Tensor) -> dict[str, torch.Tensor]:
        action_pred, value_pred = self.forward_torch(states)
        return {
            ACTION_PRED_KEY: action_pred,
            VALUE_PRED_KEY: value_pred,
        }

    def forward_sequence(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inference wrapper that satisfies the NumPy controller contract."""

        state_sequence = as_state_sequence(states, self.spec.state_dim)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state_sequence, device=self.device, dtype=self.dtype)
            action_pred, value_pred = self.forward_torch(state_tensor)
        return (
            action_pred.detach().cpu().numpy().astype(np.float32),
            value_pred.detach().cpu().numpy().astype(np.float32),
        )

    def initial_cache(self, batch_size: int = 1) -> ControllerCache:
        """Create the fixed-size online cache."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return {
            "layer_states": [
                block.empty_state(batch_size=batch_size, device=self.device, dtype=self.dtype)
                for block in self.blocks
            ],
            "steps": torch.zeros(batch_size, dtype=torch.long, device=self.device),
        }

    def _prepare_cache(self, cache: ControllerCache | None, batch_size: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        if cache is None:
            cache = self.initial_cache(batch_size=batch_size)

        layer_states = cache.get("layer_states")
        if layer_states is None:
            raise ValueError("Mamba cache is missing 'layer_states'.")
        if len(layer_states) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} layer states, got {len(layer_states)}.")

        prepared_states = []
        for layer_idx, (block, state) in enumerate(zip(self.blocks, layer_states)):
            state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
            expected_shape = (batch_size, block.d_model, block.d_state)
            if tuple(state.shape) != expected_shape:
                raise ValueError(f"Layer {layer_idx} cache state must have shape {expected_shape}, got {tuple(state.shape)}.")
            prepared_states.append(state)

        steps = torch.as_tensor(cache.get("steps", torch.zeros(batch_size)), device=self.device, dtype=torch.long).reshape(-1)
        if tuple(steps.shape) != (batch_size,):
            raise ValueError(f"Cache steps must have shape ({batch_size},), got {tuple(steps.shape)}.")
        return prepared_states, steps

    def step_torch(self, obs: torch.Tensor, cache: ControllerCache | None = None) -> tuple[torch.Tensor, torch.Tensor, ControllerCache]:
        """Torch tensor path for one online control step."""

        obs = self._check_torch_observation(obs.to(device=self.device, dtype=self.dtype))
        layer_states, steps = self._prepare_cache(cache, batch_size=obs.shape[0])

        x = self.input_proj(self.input_norm(obs))
        next_layer_states = []
        for block, state in zip(self.blocks, layer_states):
            x, next_state = block.step(x, state)
            next_layer_states.append(next_state)

        features = self.output_norm(x)
        action = self.action_head(features)
        value = self.value_head(features)
        next_cache = {
            "layer_states": next_layer_states,
            "steps": steps + 1,
        }
        return action, value, next_cache

    def step(self, obs: np.ndarray, cache: ControllerCache | None = None) -> tuple[np.ndarray, np.ndarray, ControllerCache]:
        """NumPy single-environment deployment path."""

        observation = as_single_observation(obs, self.spec.state_dim)
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation.reshape(1, -1), device=self.device, dtype=self.dtype)
            action, value, next_cache = self.step_torch(obs_tensor, cache)
        return (
            action.reshape(-1).detach().cpu().numpy().astype(np.float32),
            value.reshape(-1).detach().cpu().numpy().astype(np.float32),
            next_cache,
        )


__all__ = [
    "DualHeadMambaController",
    "MambaControllerConfig",
    "SelectiveSSMBlock",
]
