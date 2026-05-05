"""Safe-Control-Gym quadrotor benchmark configuration and helpers."""

from functools import partial
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


# Configures the safe-control-gym benchmark environment and maps 
# the standard 12-dimensional state vector (3D position, linear velocity, Euler angles, and body angular rates) 
# required for quadrotor physical dynamics.

SAFE_CONTROL_GYM_SOURCE = "https://github.com/learnsyslab/safe-control-gym"
SAFE_CONTROL_GYM_VERSION = "v1.0.0"

STATE_LABELS = [
    "x",
    "x_dot",
    "y",
    "y_dot",
    "z",
    "z_dot",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
]


# Specifies physical units for the state variables, 
# labels the four motor control inputs, and defines NumPy index arrays for slicing 
# of position, velocity, and attitude data from the full 12D state vector.

STATE_UNITS = ["m", "m/s", "m", "m/s", "m", "m/s", "rad", "rad", "rad", "rad/s", "rad/s", "rad/s"]
ACTION_LABELS = ["motor_1", "motor_2", "motor_3", "motor_4"]

POSITION_IDX = np.array([0, 2, 4])
VELOCITY_IDX = np.array([1, 3, 5])
ANGLE_IDX = np.array([6, 7, 8])


# Defines the core simulation parameters, stabilization objectives, initial state randomization boundaries, 
# and quadratic cost functions for the PyBullet quadrotor environment.

SAFE_CONTROL_GYM_TASK_CONFIG: Dict[str, Any] = {
    "seed": 1337,
    "ctrl_freq": 50,
    "pyb_freq": 1000,
    "gui": False,
    "physics": "pyb",
    "quad_type": 3,
    "init_state_randomization_info": {
        "init_x": {"distrib": "uniform", "low": -1, "high": 1},
        "init_x_dot": {"distrib": "uniform", "low": -0.1, "high": 0.1},
        "init_y": {"distrib": "uniform", "low": -1, "high": 1},
        "init_y_dot": {"distrib": "uniform", "low": -0.1, "high": 0.1},
        "init_z": {"distrib": "uniform", "low": 0.5, "high": 1.5},
        "init_z_dot": {"distrib": "uniform", "low": -0.1, "high": 0.1},
        "init_phi": {"distrib": "uniform", "low": -0.2, "high": 0.2},
        "init_theta": {"distrib": "uniform", "low": -0.2, "high": 0.2},
        "init_psi": {"distrib": "uniform", "low": -0.2, "high": 0.2},
        "init_p": {"distrib": "uniform", "low": -0.1, "high": 0.1},
        "init_q": {"distrib": "uniform", "low": -0.1, "high": 0.1},
        "init_r": {"distrib": "uniform", "low": -0.1, "high": 0.1},
    },
    "randomized_init": True,
    "randomized_inertial_prop": False,
    "task": "stabilization",
    "task_info": {
        "stabilization_goal": [0, 0, 1],
        "stabilization_goal_tolerance": 0.0,
        "proj_point": [0, 0, 0.5],
        "proj_normal": [0, 1, 1],
    },
    "episode_len_sec": 6,
    "cost": "quadratic",
    "rew_state_weight": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "rew_act_weight": [0.1],
    "done_on_out_of_bound": True,
    "constraints": [
        {"constraint_form": "default_constraint", "constrained_variable": "input"},
        {"constraint_form": "default_constraint", "constrained_variable": "state"},
    ],
    "done_on_violation": False,
    "disturbances": None,
}


# Configures the Model Predictive Control (MPC) parameters, defining the 20-step prediction horizon, 
# state and action penalty matrices for optimization, and specifying the IPOPT solver with warm-starting for 
# real-time performance.

SAFE_CONTROL_GYM_MPC_CONFIG: Dict[str, Any] = {
    "horizon": 20,
    "r_mpc": [0.1, 0.1, 0.1, 0.1],
    "q_mpc": [5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "prior_info": {
        "prior_prop": None,
        "randomize_prior_prop": False,
        "prior_prop_rand_info": None,
    },
    "warmstart": True,
    "solver": "ipopt",
}


# Verifies the installation of the safe-control-gym dependency, 
# raising a guided runtime error if missing, and returns the environment creation factory function.
def require_safe_control_gym():
    if find_spec("safe_control_gym") is None:
        raise RuntimeError(
            "safe_control_gym is required. "
            "Install the pinned upstream benchmark from "
            f"{SAFE_CONTROL_GYM_SOURCE} at {SAFE_CONTROL_GYM_VERSION}."
        )
    from safe_control_gym.utils.registration import make

    return make

# Provides a compatibility wrapper for environment initialization, 
# ensuring the output always conforms to the modern Gymnasium API standard of returning an 
# (observation, info) tuple, preventing unpacking errors across different library versions.

def reset_gym_env(env):
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result
    return reset_result, {}


# Provides a compatibility wrapper for environment stepping, converting the modern Gymnasium 5-tuple return 
# (obs, reward, terminated, truncated, info) back into the legacy Gym 4-tuple format (obs, reward, done, info) 
# to prevent unpacking errors during the training loop.
def step_gym_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated or truncated), info
    obs, reward, done, info = step_result
    return obs, reward, bool(done), info


# Orchestrates the instantiation of both the quadrotor environment and the MPC expert controller, 
# seamlessly merging configuration overrides and providing the MPC with a functional blueprint to simulate 
# future trajectories.
def make_env_and_mpc(output_dir: str, seed: int, task_config: Dict[str, Any] | None = None):
    make = require_safe_control_gym()
    merged_task_config = dict(SAFE_CONTROL_GYM_TASK_CONFIG)
    if task_config:
        merged_task_config.update(task_config)
    merged_task_config["seed"] = seed

    env_func = partial(make, "quadrotor", output_dir=output_dir, **merged_task_config)
    env = env_func()
    mpc = make(
        "mpc",
        env_func,
        training=False,
        output_dir=output_dir,
        seed=seed,
        **SAFE_CONTROL_GYM_MPC_CONFIG,
    )
    mpc.reset()
    return env, mpc


# Generates a configuration dictionary to enforce a specific,
# deterministic starting state by overriding the environment's randomization boundaries to have zero variance.
def task_config_for_initial_state(initial_state: np.ndarray) -> Dict[str, Any]:
    state = np.asarray(initial_state, dtype=float).reshape(-1)
    if state.shape[0] != len(STATE_LABELS):
        raise ValueError(f"Expected 12D initial state, got shape {state.shape}")
    randomization = {}
    for label, value in zip(STATE_LABELS, state):
        randomization[f"init_{label}"] = {"distrib": "uniform", "low": float(value), "high": float(value)}
    return {
        "randomized_init": True,
        "init_state_randomization_info": randomization,
    }

# Safely extracts the 3D target position (x, y, z) by dynamically checking the active environment's 
# internal state first, falling back to the configuration dictionary if the environment is not yet 
# initialized.
def get_goal_position(env=None, task_config: Dict[str, Any] | None = None) -> np.ndarray:
    if env is not None and hasattr(env, "X_GOAL"):
        goal = np.asarray(env.X_GOAL).reshape(-1)
        if goal.shape[0] >= 3:
            if goal.shape[0] >= 5:
                return goal[POSITION_IDX]
            return goal[:3]
    cfg = task_config or SAFE_CONTROL_GYM_TASK_CONFIG
    return np.asarray(cfg["task_info"]["stabilization_goal"], dtype=np.float32)

# Provides helper functions to extract specific physical 
# components (position, velocity, or Euler angles) from a state array, utilizing the Ellipsis (...) operator to safely process both single observations and multi-dimensional batches.
def state_position(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[..., POSITION_IDX]


def state_velocity(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[..., VELOCITY_IDX]


def state_angles(obs: np.ndarray) -> np.ndarray:
    return np.asarray(obs)[..., ANGLE_IDX]


def clip_to_env_action_space(env, action: np.ndarray) -> np.ndarray:
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return action
    return np.clip(action, action_space.low, action_space.high)


def action_bounds(env) -> Tuple[np.ndarray | None, np.ndarray | None]:
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return None, None
    return np.asarray(action_space.low), np.asarray(action_space.high)


def benchmark_manifest(env=None) -> Dict[str, Any]:
    return {
        "source": SAFE_CONTROL_GYM_SOURCE,
        "version": SAFE_CONTROL_GYM_VERSION,
        "task": "quadrotor stabilization",
        "state_order": STATE_LABELS,
        "state_units": STATE_UNITS,
        "action_order": ACTION_LABELS,
        "ctrl_freq_hz": SAFE_CONTROL_GYM_TASK_CONFIG["ctrl_freq"],
        "pybullet_freq_hz": SAFE_CONTROL_GYM_TASK_CONFIG["pyb_freq"],
        "sample_time_sec": 1.0 / float(SAFE_CONTROL_GYM_TASK_CONFIG["ctrl_freq"]),
        "physics": SAFE_CONTROL_GYM_TASK_CONFIG["physics"],
        "quad_type": SAFE_CONTROL_GYM_TASK_CONFIG["quad_type"],
        "constraints": SAFE_CONTROL_GYM_TASK_CONFIG["constraints"],
        "disturbances": SAFE_CONTROL_GYM_TASK_CONFIG["disturbances"],
        "mpc_config": SAFE_CONTROL_GYM_MPC_CONFIG,
        "env_action_low": getattr(getattr(env, "action_space", None), "low", None),
        "env_action_high": getattr(getattr(env, "action_space", None), "high", None),
    }


def write_benchmark_manifest(path: str | Path, env=None) -> None:
    import json

    manifest = benchmark_manifest(env)
    serializable = {
        key: (value.tolist() if hasattr(value, "tolist") else value)
        for key, value in manifest.items()
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
