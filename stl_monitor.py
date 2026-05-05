"""Quantitative STL robustness for the Safe-Control-Gym stabilization task.

The formulas here use standard min/max quantitative semantics:
    G phi = min_t rho(phi, t)
    F phi = max_t rho(phi, t)
    phi AND psi = min(rho(phi), rho(psi))

The current benchmark task is stabilization, so the default specification is:
    G safe_state
    AND G input_within_bounds
    AND F_[0,T] near_goal
    AND F_[0,T] G_[t,T] settled
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from safe_control_gym_config import state_angles, state_position, state_velocity


@dataclass(frozen=True)
class STLSpec:
    goal_position: np.ndarray
    position_tolerance: float = 0.25
    settle_position_tolerance: float = 0.35
    settle_speed_tolerance: float = 0.25
    x_bound: float = 2.5
    y_bound: float = 2.5
    z_min: float = 0.0
    z_max: float = 2.5
    max_abs_angle: float = 0.75
    reach_deadline_step: Optional[int] = None


def suffix_always(signal: np.ndarray) -> np.ndarray:
    out = np.empty_like(signal, dtype=np.float64)
    running = np.inf
    for idx in range(len(signal) - 1, -1, -1):
        running = min(running, float(signal[idx]))
        out[idx] = running
    return out


def eventually(signal: np.ndarray, end_idx: Optional[int] = None) -> float:
    if len(signal) == 0:
        return -np.inf
    if end_idx is None:
        end_idx = len(signal) - 1
    end_idx = min(end_idx, len(signal) - 1)
    return float(np.max(signal[: end_idx + 1]))


def always(signal: np.ndarray) -> float:
    if len(signal) == 0:
        return -np.inf
    return float(np.min(signal))


def evaluate_stabilization_stl(
    states: np.ndarray,
    actions: Optional[np.ndarray],
    spec: STLSpec,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    states = np.asarray(states, dtype=np.float64)
    positions = state_position(states)
    velocities = state_velocity(states)
    angles = state_angles(states)

    distance_to_goal = np.linalg.norm(positions - spec.goal_position.reshape(1, 3), axis=1)
    speed = np.linalg.norm(velocities, axis=1)

    state_safety_signal = np.minimum.reduce([
        spec.x_bound - np.abs(positions[:, 0]),
        spec.y_bound - np.abs(positions[:, 1]),
        positions[:, 2] - spec.z_min,
        spec.z_max - positions[:, 2],
        spec.max_abs_angle - np.max(np.abs(angles), axis=1),
    ])
    state_safety = always(state_safety_signal)

    input_safety = np.inf
    if actions is not None and action_low is not None and action_high is not None and len(actions) > 0:
        actions = np.asarray(actions, dtype=np.float64)
        low_margin = actions - np.asarray(action_low, dtype=np.float64).reshape(1, -1)
        high_margin = np.asarray(action_high, dtype=np.float64).reshape(1, -1) - actions
        input_safety = always(np.minimum(np.min(low_margin, axis=1), np.min(high_margin, axis=1)))

    reach_signal = spec.position_tolerance - distance_to_goal
    eventually_reach = eventually(reach_signal, spec.reach_deadline_step)

    settled_signal = np.minimum(
        spec.settle_position_tolerance - distance_to_goal,
        spec.settle_speed_tolerance - speed,
    )
    eventually_always_settled = eventually(suffix_always(settled_signal))

    robustness = min(state_safety, input_safety, eventually_reach, eventually_always_settled)
    return {
        "stl_robustness": float(robustness),
        "stl_satisfied": float(robustness >= 0.0),
        "state_safety_robustness": float(state_safety),
        "input_safety_robustness": float(input_safety),
        "eventually_reach_robustness": float(eventually_reach),
        "eventually_always_settled_robustness": float(eventually_always_settled),
        "min_distance_to_goal": float(np.min(distance_to_goal)) if len(distance_to_goal) else float("nan"),
        "final_distance_to_goal": float(distance_to_goal[-1]) if len(distance_to_goal) else float("nan"),
    }
