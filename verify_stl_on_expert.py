"""Verify STL robustness on Safe-Control-Gym MPC expert rollouts."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from safe_control_gym_config import (
    action_bounds,
    clip_to_env_action_space,
    get_goal_position,
    make_env_and_mpc,
    reset_gym_env,
    step_gym_env,
)
from stl_monitor import STLSpec, evaluate_stabilization_stl


def require_runtime_dependencies() -> None:
    missing = [name for name in ("safe_control_gym", "pybullet", "casadi") if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install Safe-Control-Gym with a working PyBullet build before running this verifier."
        )


def run_expert_rollout(env, mpc, steps: int):
    obs, info = reset_gym_env(env)
    states = [np.asarray(obs, dtype=np.float64)]
    actions = []
    total_reward = 0.0

    for _ in range(steps):
        action = np.asarray(mpc.select_action(obs, info), dtype=np.float64)
        action = clip_to_env_action_space(env, action)
        obs, reward, done, info = step_gym_env(env, action)
        states.append(np.asarray(obs, dtype=np.float64))
        actions.append(action)
        total_reward += float(reward)
        if done:
            break

    return np.asarray(states), np.asarray(actions), total_reward


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate STL robustness on MPC expert rollouts.")
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="runs/verify_stl_on_expert")
    args = parser.parse_args()

    try:
        require_runtime_dependencies()
    except RuntimeError as exc:
        print(f"[blocked] {exc}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        env, mpc = make_env_and_mpc(str(output_dir), args.seed)
        action_low, action_high = action_bounds(env)
        spec = STLSpec(goal_position=get_goal_position(env))

        robustness_values = []
        for rollout_idx in range(args.rollouts):
            states, actions, total_reward = run_expert_rollout(env, mpc, args.steps)
            scores = evaluate_stabilization_stl(states, actions, spec, action_low, action_high)
            robustness_values.append(scores["stl_robustness"])
            print(f"rollout={rollout_idx:03d} steps={len(actions)} reward={total_reward:.4f}")
            for key in (
                "stl_robustness",
                "stl_satisfied",
                "state_safety_robustness",
                "input_safety_robustness",
                "eventually_reach_robustness",
                "eventually_always_settled_robustness",
                "min_distance_to_goal",
                "final_distance_to_goal",
            ):
                print(f"  {key}: {scores[key]:.6f}")

        robustness_values = np.asarray(robustness_values, dtype=np.float64)
        print(f"mean_stl_robustness: {float(np.mean(robustness_values)):.6f}")
        print(f"min_stl_robustness:  {float(np.min(robustness_values)):.6f}")
        print("[ok] STL monitor evaluated on MPC expert rollouts.")
        return 0
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}")
        return 1
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
