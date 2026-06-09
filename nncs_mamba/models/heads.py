"""Output heads for the dual-head controller.

The shared sequence model produces one feature vector for every time step. The
heads convert those features into the two quantities this project needs:

- the next quadrotor control action;
- the predicted STL robustness or safety margin.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    """A small feed-forward head used after the shared sequence encoder."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class ActionHead(nn.Module):
    """Predict the 4D control action.

    If action bounds are provided, the raw output is squashed through `tanh` and
    scaled into that interval. If no bounds are provided, the head returns raw
    action predictions and the rollout code can still clip them to the
    environment limits.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
        action_low: torch.Tensor | None = None,
        action_high: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.net = MLPHead(input_dim=input_dim, output_dim=action_dim, hidden_dim=hidden_dim)

        if action_low is not None and action_high is not None:
            low = torch.as_tensor(action_low, dtype=torch.float32).reshape(1, 1, action_dim)
            high = torch.as_tensor(action_high, dtype=torch.float32).reshape(1, 1, action_dim)
            if torch.any(high <= low):
                raise ValueError("action_high must be greater than action_low for every action dimension.")
            self.register_buffer("action_low", low)
            self.register_buffer("action_high", high)
        else:
            self.register_buffer("action_low", None)
            self.register_buffer("action_high", None)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_action = self.net(features)
        if self.action_low is None or self.action_high is None:
            return raw_action

        center = 0.5 * (self.action_high + self.action_low)
        radius = 0.5 * (self.action_high - self.action_low)
        return center + radius * torch.tanh(raw_action)


class ValueHead(nn.Module):
    """Predict STL robustness or a safety-margin value."""

    def __init__(self, input_dim: int, value_dim: int = 1, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = MLPHead(input_dim=input_dim, output_dim=value_dim, hidden_dim=hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


__all__ = [
    "ActionHead",
    "MLPHead",
    "ValueHead",
]
