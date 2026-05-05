"""
Evaluate a trained NNCS checkpoint.

Usage:
    python evaluate.py --checkpoint runs/experiment/best_imitation.pt
    python evaluate.py --checkpoint runs/experiment/best_imitation.pt --missions 20
"""
import argparse
import numpy as np
import torch
import time
from pathlib import Path

from controller_factory import build_controller_from_config
from safe_control_gym_config import (
    action_bounds,
    clip_to_env_action_space,
    get_goal_position,
    make_env_and_mpc,
    reset_gym_env,
    state_position,
    step_gym_env,
)
from stl_monitor import STLSpec, evaluate_stabilization_stl


def task_config_for_profile(profile: str):
    if profile == "wide-init":
        return {
            "init_state_randomization_info": {
                "init_x": {"distrib": "uniform", "low": -1.5, "high": 1.5},
                "init_x_dot": {"distrib": "uniform", "low": -0.2, "high": 0.2},
                "init_y": {"distrib": "uniform", "low": -1.5, "high": 1.5},
                "init_y_dot": {"distrib": "uniform", "low": -0.2, "high": 0.2},
                "init_z": {"distrib": "uniform", "low": 0.35, "high": 1.8},
                "init_z_dot": {"distrib": "uniform", "low": -0.2, "high": 0.2},
                "init_phi": {"distrib": "uniform", "low": -0.35, "high": 0.35},
                "init_theta": {"distrib": "uniform", "low": -0.35, "high": 0.35},
                "init_psi": {"distrib": "uniform", "low": -0.35, "high": 0.35},
                "init_p": {"distrib": "uniform", "low": -0.2, "high": 0.2},
                "init_q": {"distrib": "uniform", "low": -0.2, "high": 0.2},
                "init_r": {"distrib": "uniform", "low": -0.2, "high": 0.2},
            }
        }
    return {}


def stl_spec_for_profile(profile: str, goal_position: np.ndarray):
    if profile == "constraint-tight":
        return STLSpec(
            goal_position=goal_position,
            position_tolerance=0.18,
            settle_position_tolerance=0.25,
            settle_speed_tolerance=0.18,
            x_bound=1.75,
            y_bound=1.75,
            z_min=0.25,
            z_max=1.75,
            max_abs_angle=0.5,
        )
    return STLSpec(goal_position=goal_position)


def evaluate_safe_control_gym(controller, args):
    checkpoint_parent = Path(args.checkpoint).parent if args.checkpoint else Path("runs/expert_eval")
    output_dir = str(checkpoint_parent / "safe_control_gym_eval")
    env, mpc = make_env_and_mpc(output_dir, args.seed, task_config=task_config_for_profile(args.robustness_profile))
    action_low, action_high = action_bounds(env)

    episode_returns = []
    final_position_errors = []
    action_mses = []
    action_maes = []
    learner_step_times = []
    mpc_step_times = []
    constraint_violations = []
    stl_robustness = []
    stl_satisfaction = []

    try:
        for _ in range(args.missions):
            obs, info = reset_gym_env(env)
            if controller is not None:
                controller.reset()
            mpc.reset_before_run(obs=obs, info=info, env=env)

            total_reward = 0.0
            states = [np.asarray(obs, dtype=np.float32)]
            learner_actions = []
            expert_actions = []
            violation_count = 0

            for _ in range(args.seq_steps):
                mpc_start = time.perf_counter()
                expert_action = np.asarray(mpc.select_action(obs, info), dtype=np.float32)
                mpc_step_times.append(time.perf_counter() - mpc_start)
                if controller is None:
                    learner_action = expert_action
                else:
                    learner_start = time.perf_counter()
                    learner_action = np.asarray(controller.forward(np.asarray(obs, dtype=np.float32)), dtype=np.float32)
                    learner_step_times.append(time.perf_counter() - learner_start)
                learner_action = clip_to_env_action_space(env, learner_action)

                learner_actions.append(learner_action)
                expert_actions.append(expert_action)

                obs, reward, done, info = step_gym_env(env, learner_action)
                states.append(np.asarray(obs, dtype=np.float32))
                total_reward += float(reward)
                if info.get("constraint_violation", False) or info.get("violation", False):
                    violation_count += 1
                if done:
                    break

            goal_pos = get_goal_position(env)
            final_position_errors.append(float(np.linalg.norm(state_position(np.asarray(obs).reshape(1, -1))[0] - goal_pos)))
            episode_returns.append(total_reward)
            constraint_violations.append(violation_count)

            if learner_actions:
                learner_stack = np.stack(learner_actions)
                expert_stack = np.stack(expert_actions)
                action_mses.append(float(np.mean((learner_stack - expert_stack) ** 2)))
                action_maes.append(float(np.mean(np.abs(learner_stack - expert_stack))))
                stl = evaluate_stabilization_stl(
                    np.stack(states),
                    learner_stack,
                    stl_spec_for_profile(args.robustness_profile, goal_pos),
                    action_low,
                    action_high,
                )
                stl_robustness.append(stl["stl_robustness"])
                stl_satisfaction.append(stl["stl_satisfied"])

    finally:
        mpc.close()
        env.close()

    return {
        "avg_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "avg_final_position_error": float(np.mean(final_position_errors)) if final_position_errors else float("nan"),
        "avg_action_mse_vs_mpc": float(np.mean(action_mses)) if action_mses else float("nan"),
        "avg_action_mae_vs_mpc": float(np.mean(action_maes)) if action_maes else float("nan"),
        "avg_constraint_violations": float(np.mean(constraint_violations)) if constraint_violations else float("nan"),
        "avg_stl_robustness": float(np.mean(stl_robustness)) if stl_robustness else float("nan"),
        "stl_satisfaction_rate": float(np.mean(stl_satisfaction)) if stl_satisfaction else float("nan"),
        "learner_ms_per_step": float(np.mean(learner_step_times) * 1000.0) if learner_step_times else 0.0,
        "mpc_ms_per_step": float(np.mean(mpc_step_times) * 1000.0) if mpc_step_times else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained controller.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--missions", type=int, default=10, help="Number of evaluation rollouts")
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expert-only", action="store_true", help="Evaluate Safe-Control-Gym MPC without a learned checkpoint")
    parser.add_argument("--robustness-profile", choices=["nominal", "wide-init", "constraint-tight"], default="nominal")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    controller = None
    config = {"controller_type": "safe-control-gym-mpc"}
    meta = {"phase": "expert"}
    if not args.expert_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --expert-only is set")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        controller = build_controller_from_config(config)
        checkpoint = controller.load_checkpoint(args.checkpoint)
        meta = checkpoint.get("metadata", {})

    print(f"Loaded: {'Safe-Control-Gym MPC expert' if args.expert_only else args.checkpoint}")
    print(f"  Controller: {config.get('controller_type', 'mamba')}")
    print(f"  Plant backend: safe-control-gym")
    print(f"  Robustness profile: {args.robustness_profile}")
    print(f"  Phase: {meta.get('phase', '?')}  Epoch/Cycle: {meta.get('epoch', meta.get('cycle', '?'))}")
    print()

    print(f"Results over {args.missions} missions:")
    metrics = evaluate_safe_control_gym(controller, args)
    print(f"  Avg return:                  {metrics['avg_return']:.3f}")
    print(f"  Avg final position error:    {metrics['avg_final_position_error']:.3f} m")
    print(f"  Action MSE vs MPC:           {metrics['avg_action_mse_vs_mpc']:.6f}")
    print(f"  Action MAE vs MPC:           {metrics['avg_action_mae_vs_mpc']:.6f}")
    print(f"  Avg constraint violations:   {metrics['avg_constraint_violations']:.2f}")
    print(f"  STL satisfaction rate:       {metrics['stl_satisfaction_rate']:.0%}")
    print(f"  Avg STL robustness:          {metrics['avg_stl_robustness']:.3f}")
    print(f"  Learner inference:           {metrics['learner_ms_per_step']:.3f} ms/step")
    print(f"  MPC label/control time:      {metrics['mpc_ms_per_step']:.3f} ms/step")


if __name__ == "__main__":
    main()
