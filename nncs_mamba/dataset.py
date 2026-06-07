"""Collect MPC expert trajectories and attach STL robustness labels.

This file is the bridge between the trusted controller and the learned
controller. The MPC expert creates the actions. The STL monitor judges the full
trajectory. The saved dataset then gives the Mamba model two things to learn:
what action to take, and how safe or unsafe the resulting behavior was.

The first dataset should be small. Its purpose is to prove that the data shape,
the action labels, and the STL labels all line up before we spend hours
collecting many rollouts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from nncs_mamba.safe_control_gym_config import (
    ACTION_LABELS,
    SAFE_CONTROL_GYM_MPC_CONFIG,
    SAFE_CONTROL_GYM_TASK_CONFIG,
    STATE_LABELS,
    action_bounds,
    benchmark_manifest,
    clip_to_env_action_space,
    get_goal_position,
    make_env_and_mpc,
    reset_gym_env,
    step_gym_env,
)
from nncs_mamba.stl_monitor import STLSpec, evaluate_stabilization_stl


COMPONENT_ROBUSTNESS_KEYS = (
    "state_safety_robustness",
    "input_safety_robustness",
    "eventually_reach_robustness",
    "eventually_always_settled_robustness",
)

ROLLOUT_SCORE_KEYS = (
    "stl_robustness",
    "stl_satisfied",
    "min_distance_to_goal",
    "final_distance_to_goal",
)


@dataclass(frozen=True)
class ExpertRollout:
    """One complete MPC-controlled episode.

    The states have one more row than the actions because the first state is the
    reset observation. Each action moves the system to the next state.
    """

    states: np.ndarray
    actions: np.ndarray
    total_reward: float
    done: bool
    scores: Dict[str, float]


def require_runtime_dependencies() -> None:
    """Stop early if the simulator stack is not ready.

    Safe-Control-Gym provides the plant and MPC controller. PyBullet provides
    the physics engine. CasADi provides the nonlinear optimization machinery
    used by the MPC expert. If one of them is missing, dataset collection cannot
    start yet.
    """

    missing = [name for name in ("safe_control_gym", "pybullet", "casadi") if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install Safe-Control-Gym with a working PyBullet build before collecting trajectories."
        )


def run_mpc_rollout(env, mpc, steps: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Run one expert rollout and return raw state/action arrays."""

    if hasattr(mpc, "reset"):
        mpc.reset()

    obs, info = reset_gym_env(env)
    states = [np.asarray(obs, dtype=np.float64)]
    actions = []
    total_reward = 0.0
    done = False

    for _ in range(steps):
        action = np.asarray(mpc.select_action(obs, info), dtype=np.float64)
        action = clip_to_env_action_space(env, action)
        obs, reward, done, info = step_gym_env(env, action)

        actions.append(np.asarray(action, dtype=np.float64))
        states.append(np.asarray(obs, dtype=np.float64))
        total_reward += float(reward)

        if done:
            break

    if actions:
        action_array = np.asarray(actions, dtype=np.float64)
    else:
        action_array = np.empty((0, len(ACTION_LABELS)), dtype=np.float64)

    return np.asarray(states, dtype=np.float64), action_array, total_reward, done


def score_rollout(
    states: np.ndarray,
    actions: np.ndarray,
    goal_position: np.ndarray,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
) -> Dict[str, float]:
    """Use the STL monitor as the authority for rollout safety."""

    spec = STLSpec(goal_position=goal_position)
    return evaluate_stabilization_stl(states, actions, spec, action_low, action_high)


def collect_expert_rollouts(output_dir: Path, rollouts: int, steps: int, seed: int) -> tuple[List[ExpertRollout], Dict[str, Any]]:
    """Create the environment once and ask MPC for several demonstrations."""

    env = None
    try:
        env, mpc = make_env_and_mpc(str(output_dir), seed)
        goal_position = get_goal_position(env)
        action_low, action_high = action_bounds(env)

        records: List[ExpertRollout] = []
        for rollout_idx in range(rollouts):
            states, actions, total_reward, done = run_mpc_rollout(env, mpc, steps)
            scores = score_rollout(states, actions, goal_position, action_low, action_high)
            records.append(
                ExpertRollout(
                    states=states,
                    actions=actions,
                    total_reward=total_reward,
                    done=done,
                    scores=scores,
                )
            )
            print(
                f"rollout={rollout_idx:03d} "
                f"steps={len(actions)} "
                f"reward={total_reward:.4f} "
                f"stl={scores['stl_robustness']:.6f}"
            )

        manifest = {
            "benchmark": benchmark_manifest(env),
            "goal_position": goal_position,
            "action_low": action_low,
            "action_high": action_high,
        }
        return records, manifest
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


def pad_rollouts(records: Iterable[ExpertRollout], max_steps: int) -> Dict[str, np.ndarray]:
    """Convert variable-length rollouts into fixed arrays plus masks.

    Padding keeps the file simple. The masks tell training code which rows are
    real data and which rows are empty padding after an early termination.
    """

    records = list(records)
    rollout_count = len(records)
    state_dim = len(STATE_LABELS)
    action_dim = len(ACTION_LABELS)

    states = np.zeros((rollout_count, max_steps + 1, state_dim), dtype=np.float32)
    actions = np.zeros((rollout_count, max_steps, action_dim), dtype=np.float32)
    state_mask = np.zeros((rollout_count, max_steps + 1), dtype=bool)
    action_mask = np.zeros((rollout_count, max_steps), dtype=bool)
    state_lengths = np.zeros(rollout_count, dtype=np.int32)
    action_lengths = np.zeros(rollout_count, dtype=np.int32)
    total_rewards = np.zeros(rollout_count, dtype=np.float32)
    dones = np.zeros(rollout_count, dtype=bool)
    rollout_scores = np.zeros((rollout_count, len(ROLLOUT_SCORE_KEYS)), dtype=np.float32)
    component_robustness = np.zeros((rollout_count, len(COMPONENT_ROBUSTNESS_KEYS)), dtype=np.float32)

    for idx, record in enumerate(records):
        state_len = min(len(record.states), max_steps + 1)
        action_len = min(len(record.actions), max_steps)

        states[idx, :state_len] = record.states[:state_len]
        actions[idx, :action_len] = record.actions[:action_len]
        state_mask[idx, :state_len] = True
        action_mask[idx, :action_len] = True
        state_lengths[idx] = state_len
        action_lengths[idx] = action_len
        total_rewards[idx] = float(record.total_reward)
        dones[idx] = bool(record.done)

        for score_idx, key in enumerate(ROLLOUT_SCORE_KEYS):
            rollout_scores[idx, score_idx] = float(record.scores[key])
        for score_idx, key in enumerate(COMPONENT_ROBUSTNESS_KEYS):
            component_robustness[idx, score_idx] = float(record.scores[key])

    return {
        "states": states,
        "actions": actions,
        "state_mask": state_mask,
        "action_mask": action_mask,
        "state_lengths": state_lengths,
        "action_lengths": action_lengths,
        "total_rewards": total_rewards,
        "dones": dones,
        "rollout_scores": rollout_scores,
        "component_robustness": component_robustness,
    }


def to_jsonable(value: Any) -> Any:
    """Convert NumPy-heavy metadata into plain JSON values."""

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_dataset(
    output_dir: Path,
    dataset_name: str,
    records: List[ExpertRollout],
    max_steps: int,
    seed: int,
    runtime_manifest: Dict[str, Any],
) -> tuple[Path, Path]:
    """Write arrays to NPZ and human-readable metadata to JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / dataset_name
    manifest_path = output_dir / f"{Path(dataset_name).stem}_manifest.json"

    arrays = pad_rollouts(records, max_steps)
    np.savez_compressed(dataset_path, **arrays)

    manifest = {
        "dataset_format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_file": dataset_path.name,
        "rollouts": len(records),
        "max_steps": max_steps,
        "seed": seed,
        "state_labels": STATE_LABELS,
        "action_labels": ACTION_LABELS,
        "rollout_score_keys": ROLLOUT_SCORE_KEYS,
        "component_robustness_keys": COMPONENT_ROBUSTNESS_KEYS,
        "task_config": SAFE_CONTROL_GYM_TASK_CONFIG,
        "mpc_config": SAFE_CONTROL_GYM_MPC_CONFIG,
        "runtime": runtime_manifest,
        "notes": (
            "States are padded to max_steps + 1 because each rollout starts with "
            "an initial observation. Actions are padded to max_steps. Use the masks "
            "to ignore padding during training."
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    return dataset_path, manifest_path


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Safe-Control-Gym MPC expert trajectories.")
    parser.add_argument("--rollouts", type=positive_int, default=10, help="Number of expert rollouts to collect.")
    parser.add_argument("--steps", type=positive_int, default=300, help="Maximum controller steps per rollout.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="runs/expert_dataset")
    parser.add_argument("--dataset-name", default="expert_dataset.npz")
    args = parser.parse_args()

    try:
        require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"[blocked] {exc}")
        return 2

    output_dir = Path(args.output_dir)
    try:
        records, runtime_manifest = collect_expert_rollouts(output_dir, args.rollouts, args.steps, args.seed)
        dataset_path, manifest_path = write_dataset(
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            records=records,
            max_steps=args.steps,
            seed=args.seed,
            runtime_manifest=runtime_manifest,
        )
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}")
        return 1

    robustness = np.asarray([record.scores["stl_robustness"] for record in records], dtype=np.float64)
    print(f"dataset: {dataset_path}")
    print(f"manifest: {manifest_path}")
    print(f"mean_stl_robustness: {float(np.mean(robustness)):.6f}")
    print(f"min_stl_robustness:  {float(np.min(robustness)):.6f}")
    print("[ok] Expert dataset collected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
