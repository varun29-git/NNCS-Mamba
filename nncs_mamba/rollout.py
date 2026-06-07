"""Roll out a controller through an environment.

This file keeps the environment loop separate from the model. The model decides
the action and predicts an STL value. The rollout code clips the action if the
environment exposes action bounds, steps the environment, and records what
happened.

That separation matters because MPC, a dummy controller, and the future Mamba
controller should all be evaluated through clear and reviewable paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nncs_mamba.models.controller import ControllerProtocol
from nncs_mamba.safe_control_gym_config import (
    ACTION_LABELS,
    STATE_LABELS,
    clip_to_env_action_space,
    reset_gym_env,
    step_gym_env,
)


@dataclass(frozen=True)
class ControllerRollout:
    """One rollout produced by a learned or dummy controller."""

    states: np.ndarray
    actions: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    total_reward: float
    done: bool
    final_info: dict[str, Any]


def positive_steps(steps: int) -> int:
    """Validate rollout length before touching the environment."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    return int(steps)


def controller_dimensions(controller: ControllerProtocol) -> tuple[int, int, int]:
    """Read dimensions from a controller, with benchmark defaults as fallback."""

    spec = getattr(controller, "spec", None)
    state_dim = int(getattr(spec, "state_dim", len(STATE_LABELS)))
    action_dim = int(getattr(spec, "action_dim", len(ACTION_LABELS)))
    value_dim = int(getattr(spec, "value_dim", 1))
    return state_dim, action_dim, value_dim


def run_controller_rollout(
    env,
    controller: ControllerProtocol,
    steps: int,
    clip_actions: bool = True,
    initial_cache: dict[str, Any] | None = None,
) -> ControllerRollout:
    """Run one controller-controlled episode.

    The returned arrays are not padded. If the environment ends early, the
    arrays simply stop early too. Dataset writers can decide later whether they
    want padding and masks.
    """

    rollout_steps = positive_steps(steps)
    state_dim, action_dim, value_dim = controller_dimensions(controller)

    obs, _ = reset_gym_env(env)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.shape != (state_dim,):
        raise ValueError(f"Expected reset observation shaped ({state_dim},), got {obs.shape}.")

    cache = initial_cache if initial_cache is not None else controller.initial_cache(batch_size=1)
    states = [obs.copy()]
    actions = []
    values = []
    rewards = []
    dones = []
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    for _ in range(rollout_steps):
        action, value, cache = controller.step(obs, cache)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        value = np.asarray(value, dtype=np.float32).reshape(-1)

        if action.shape != (action_dim,):
            raise ValueError(f"Expected controller action shaped ({action_dim},), got {action.shape}.")
        if value.shape != (value_dim,):
            raise ValueError(f"Expected controller value shaped ({value_dim},), got {value.shape}.")

        if clip_actions:
            action = np.asarray(clip_to_env_action_space(env, action), dtype=np.float32).reshape(-1)

        obs, reward, done, info = step_gym_env(env, action)
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs.shape != (state_dim,):
            raise ValueError(f"Expected step observation shaped ({state_dim},), got {obs.shape}.")

        actions.append(action.copy())
        values.append(value.copy())
        rewards.append(float(reward))
        dones.append(bool(done))
        states.append(obs.copy())
        total_reward += float(reward)

        if done:
            break

    if actions:
        action_array = np.asarray(actions, dtype=np.float32)
        value_array = np.asarray(values, dtype=np.float32)
    else:
        action_array = np.empty((0, action_dim), dtype=np.float32)
        value_array = np.empty((0, value_dim), dtype=np.float32)

    return ControllerRollout(
        states=np.asarray(states, dtype=np.float32),
        actions=action_array,
        values=value_array,
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        total_reward=float(total_reward),
        done=bool(done),
        final_info=dict(info),
    )


__all__ = [
    "ControllerRollout",
    "controller_dimensions",
    "positive_steps",
    "run_controller_rollout",
]
