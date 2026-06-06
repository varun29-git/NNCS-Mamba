"""Verify the Safe-Control-Gym quadrotor + MPC interface.

This script is intentionally small. It checks the trusted plant/expert boundary
before any learner code is introduced.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from safe_control_gym_config import (
    ACTION_LABELS,
    SAFE_CONTROL_GYM_VERSION,
    STATE_LABELS,
    action_bounds,
    clip_to_env_action_space,
    get_goal_position,
    make_env_and_mpc,
    reset_gym_env,
    state_angles,
    state_position,
    state_velocity,
    step_gym_env,
    write_benchmark_manifest,
)


def require_runtime_dependencies() -> None:
    missing = [name for name in ("safe_control_gym", "pybullet", "casadi") if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Install Safe-Control-Gym with a working PyBullet build before running this verifier."
        )


def describe_array(name: str, value) -> str:
    array = np.asarray(value)
    return f"{name}: shape={array.shape}, dtype={array.dtype}, value={np.array2string(array, precision=4)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Safe-Control-Gym quadrotor MPC interface.")
    parser.add_argument("--steps", type=int, default=5, help="Number of MPC-controlled environment steps.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", default="runs/verify_safe_control_gym")
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
        obs, info = reset_gym_env(env)
        action_low, action_high = action_bounds(env)
        goal_position = get_goal_position(env)

        print("Safe-Control-Gym verification")
        print(f"  pinned version: {SAFE_CONTROL_GYM_VERSION}")
        print(f"  state labels:   {STATE_LABELS}")
        print(f"  action labels:  {ACTION_LABELS}")
        print("  " + describe_array("initial observation", obs))
        print("  " + describe_array("position", state_position(obs)))
        print("  " + describe_array("velocity", state_velocity(obs)))
        print("  " + describe_array("angles", state_angles(obs)))
        print("  " + describe_array("goal_position", goal_position))
        print("  " + describe_array("action_low", action_low))
        print("  " + describe_array("action_high", action_high))

        if np.asarray(obs).shape[-1] != len(STATE_LABELS):
            raise RuntimeError(f"Expected {len(STATE_LABELS)}D state, got shape {np.asarray(obs).shape}")

        total_reward = 0.0
        for step in range(args.steps):
            action = np.asarray(mpc.select_action(obs, info), dtype=np.float64)
            clipped_action = clip_to_env_action_space(env, action)
            next_obs, reward, done, info = step_gym_env(env, clipped_action)
            total_reward += float(reward)
            print(
                f"  step={step:03d} "
                f"action_shape={action.shape} "
                f"reward={float(reward): .4f} "
                f"done={done}"
            )
            obs = next_obs
            if done:
                break

        write_benchmark_manifest(output_dir / "benchmark_manifest.json", env)
        print(f"  total_reward: {total_reward:.4f}")
        print(f"  manifest: {output_dir / 'benchmark_manifest.json'}")
        print("[ok] Safe-Control-Gym environment and MPC interface verified.")
        return 0
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}")
        return 1
    finally:
        if env is not None and hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
