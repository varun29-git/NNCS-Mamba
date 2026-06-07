"""Controller contracts for the learned quadrotor controller.

This file defines the shape of the controller before we define the real Mamba
model. That is important because training, rollout evaluation, and deployment
should all agree on the same interface.

The controller has two outputs:

- an action/control head, which predicts the next 4 motor command values;
- an STL value head, which predicts a robustness or safety-margin value.

The STL value head is only a learned estimate. The actual STL monitor remains
the authority for deciding whether a trajectory is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping, Protocol

import numpy as np

from nncs_mamba.safe_control_gym_config import ACTION_LABELS, STATE_LABELS


ACTION_PRED_KEY = "action_pred"
VALUE_PRED_KEY = "value_pred"

ControllerCache = MutableMapping[str, Any]


@dataclass(frozen=True)
class ControllerSpec:
    """The dimensions that every controller must respect."""

    name: str = "controller"
    state_dim: int = len(STATE_LABELS)
    action_dim: int = len(ACTION_LABELS)
    value_dim: int = 1


class ControllerProtocol(Protocol):
    """The shared API for training, evaluation, and deployment.

    During training, a model can process a full sequence at once.
    During deployment, it must process one observation at a time and carry a
    cache. For Mamba, that cache is what makes online inference fixed-cost per
    controller step for a fixed model size.
    """

    spec: ControllerSpec

    def forward_sequence(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict actions and STL values for a full state sequence."""

    def initial_cache(self, batch_size: int = 1) -> ControllerCache:
        """Create the empty online-inference cache."""

    def step(self, obs: np.ndarray, cache: ControllerCache | None = None) -> tuple[np.ndarray, np.ndarray, ControllerCache]:
        """Predict one action and one STL value from the current observation."""


def as_state_sequence(states: np.ndarray, state_dim: int) -> np.ndarray:
    """Convert states into a batch-first sequence array.

    A single sequence shaped `(time, state_dim)` becomes
    `(1, time, state_dim)`. A batch shaped `(batch, time, state_dim)` is kept as
    it is. Anything else fails early because hidden shape mistakes are expensive
    in control.
    """

    array = np.asarray(states, dtype=np.float32)
    if array.ndim == 2:
        array = array.reshape(1, array.shape[0], array.shape[1])
    if array.ndim != 3:
        raise ValueError(f"Expected states shaped (time, {state_dim}) or (batch, time, {state_dim}), got {array.shape}.")
    if array.shape[-1] != state_dim:
        raise ValueError(f"Expected final state dimension {state_dim}, got {array.shape[-1]}.")
    return array


def as_single_observation(obs: np.ndarray, state_dim: int) -> np.ndarray:
    """Convert one environment observation into a flat state vector."""

    array = np.asarray(obs, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array.reshape(array.shape[1])
    if array.ndim != 1 or array.shape[0] != state_dim:
        raise ValueError(f"Expected one observation shaped ({state_dim},), got {array.shape}.")
    return array


def as_vector(value: np.ndarray | float, dim: int, name: str) -> np.ndarray:
    """Convert a scalar or vector into one fixed-size float32 vector."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = np.full((dim,), float(array), dtype=np.float32)
    else:
        array = array.reshape(-1)

    if array.shape[0] == 1 and dim > 1:
        array = np.full((dim,), float(array[0]), dtype=np.float32)

    if array.shape != (dim,):
        raise ValueError(f"Expected {name} shaped ({dim},), got {array.shape}.")
    return array.astype(np.float32, copy=True)


class ConstantController:
    """A tiny controller used to test the pipeline before Mamba exists.

    It always returns the same action and the same STL value estimate. This is
    not a useful flight controller. Its job is to prove that rollout code,
    clipping, caching, and tensor shapes work without needing a neural network.
    """

    def __init__(
        self,
        action: np.ndarray | float | None = None,
        value: np.ndarray | float = 0.0,
        spec: ControllerSpec | None = None,
    ) -> None:
        self.spec = spec or ControllerSpec(name="constant_controller")
        action_value = np.zeros(self.spec.action_dim, dtype=np.float32) if action is None else action
        self.action = as_vector(action_value, self.spec.action_dim, "action")
        self.value = as_vector(value, self.spec.value_dim, "value")

    def forward_sequence(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return constant predictions for every state in the sequence."""

        state_sequence = as_state_sequence(states, self.spec.state_dim)
        batch_size, steps = state_sequence.shape[:2]
        action_pred = np.broadcast_to(
            self.action.reshape(1, 1, self.spec.action_dim),
            (batch_size, steps, self.spec.action_dim),
        ).copy()
        value_pred = np.broadcast_to(
            self.value.reshape(1, 1, self.spec.value_dim),
            (batch_size, steps, self.spec.value_dim),
        ).copy()
        return action_pred, value_pred

    def initial_cache(self, batch_size: int = 1) -> ControllerCache:
        """Create a simple step counter cache.

        The real Mamba cache will hold recurrent state. This dummy cache only
        counts how many online steps have been taken.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return {"steps": np.zeros(batch_size, dtype=np.int64)}

    def step(self, obs: np.ndarray, cache: ControllerCache | None = None) -> tuple[np.ndarray, np.ndarray, ControllerCache]:
        """Return one constant action and one constant STL value estimate."""

        as_single_observation(obs, self.spec.state_dim)
        if cache is None:
            cache = self.initial_cache(batch_size=1)

        steps = np.asarray(cache.get("steps", np.zeros(1, dtype=np.int64)), dtype=np.int64).reshape(-1).copy()
        if steps.shape != (1,):
            raise ValueError("ConstantController.step is the single-environment deployment path, so cache['steps'] must have shape (1,).")

        next_cache = dict(cache)
        next_cache["steps"] = steps + 1
        return self.action.copy(), self.value.copy(), next_cache


__all__ = [
    "ACTION_PRED_KEY",
    "VALUE_PRED_KEY",
    "ConstantController",
    "ControllerCache",
    "ControllerProtocol",
    "ControllerSpec",
    "as_single_observation",
    "as_state_sequence",
    "as_vector",
]
