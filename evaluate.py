"""
Evaluate a trained NNCS checkpoint.

Usage:
    python evaluate.py --checkpoint runs/full/best_cegis.pt
    python evaluate.py --checkpoint runs/full/best_cegis.pt --missions 20 --plot
    python evaluate.py --checkpoint runs/scg/best_imitation.pt --plant-backend safe-control-gym
"""
import argparse
import numpy as np
import torch
from functools import partial
from pathlib import Path

from drone_env import DronePlant, DroneExpertController, CHECKPOINTS, CHECKPOINT_RADIUS
from controller_factory import build_controller_from_config
from cegis_loop import (
    check_stl_score,
    run_validation_rollouts,
    plot_trajectory,
)
from train import (
    SAFE_CONTROL_GYM_MPC_CONFIG,
    SAFE_CONTROL_GYM_TASK_CONFIG,
    require_safe_control_gym,
    reset_gym_env,
    step_gym_env,
)


def clip_to_env_action_space(env, action):
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return action
    return np.clip(action, action_space.low, action_space.high)


def evaluate_safe_control_gym(controller, args):
    make = require_safe_control_gym()
    task_config = dict(SAFE_CONTROL_GYM_TASK_CONFIG)
    task_config["seed"] = args.seed
    output_dir = str(Path(args.checkpoint).parent / "safe_control_gym_eval")
    env_func = partial(make, "quadrotor", output_dir=output_dir, **task_config)
    env = env_func()
    mpc = make(
        "mpc",
        env_func,
        training=False,
        output_dir=output_dir,
        seed=args.seed,
        **SAFE_CONTROL_GYM_MPC_CONFIG,
    )
    mpc.reset()

    episode_returns = []
    final_position_errors = []
    action_mses = []
    constraint_violations = []

    try:
        for _ in range(args.missions):
            obs, info = reset_gym_env(env)
            controller.reset()
            mpc.reset_before_run(obs=obs, info=info, env=env)

            total_reward = 0.0
            learner_actions = []
            expert_actions = []
            violation_count = 0

            for _ in range(args.seq_steps):
                expert_action = np.asarray(mpc.select_action(obs, info), dtype=np.float32)
                learner_action = np.asarray(controller.forward(np.asarray(obs, dtype=np.float32)), dtype=np.float32)
                learner_action = clip_to_env_action_space(env, learner_action)

                learner_actions.append(learner_action)
                expert_actions.append(expert_action)

                obs, reward, done, info = step_gym_env(env, learner_action)
                total_reward += float(reward)
                if info.get("constraint_violation", False) or info.get("violation", False):
                    violation_count += 1
                if done:
                    break

            goal = np.asarray(getattr(env, "X_GOAL", task_config["task_info"]["stabilization_goal"])).reshape(-1)
            goal_pos = goal[:3] if goal.shape[0] >= 3 else np.asarray(task_config["task_info"]["stabilization_goal"])
            final_position_errors.append(float(np.linalg.norm(np.asarray(obs).reshape(-1)[:3] - goal_pos)))
            episode_returns.append(total_reward)
            constraint_violations.append(violation_count)

            if learner_actions:
                learner_stack = np.stack(learner_actions)
                expert_stack = np.stack(expert_actions)
                action_mses.append(float(np.mean((learner_stack - expert_stack) ** 2)))

    finally:
        mpc.close()
        env.close()

    return {
        "avg_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "avg_final_position_error": float(np.mean(final_position_errors)) if final_position_errors else float("nan"),
        "avg_action_mse_vs_mpc": float(np.mean(action_mses)) if action_mses else float("nan"),
        "avg_constraint_violations": float(np.mean(constraint_violations)) if constraint_violations else float("nan"),
    }


def evaluate_prototype(controller, args):
    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)

    metrics = run_validation_rollouts(
        controller, plant,
        num_missions=args.missions,
        seq_steps=args.seq_steps,
        dt=args.dt,
    )

    plant.reset()
    controller.reset()
    y = plant.state.copy()
    traj = [y]
    phase_idx = 0
    for _ in range(args.seq_steps):
        target = CHECKPOINTS[phase_idx]
        y_with_target = np.concatenate([y, target])
        u = controller.forward(y_with_target)
        y = plant.step(u, args.dt)
        traj.append(y)
        if phase_idx < 3 and np.linalg.norm(y[0:3] - target) < CHECKPOINT_RADIUS:
            phase_idx += 1
    metrics["stl_robustness_single_traj"] = check_stl_score(traj)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained controller.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--missions", type=int, default=10, help="Number of evaluation rollouts")
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--plot", action="store_true", help="Save a trajectory comparison plot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plant-backend", choices=["auto", "safe-control-gym", "prototype"], default="auto")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    controller = build_controller_from_config(config)
    checkpoint = controller.load_checkpoint(args.checkpoint)
    meta = checkpoint.get("metadata", {})
    plant_backend = args.plant_backend
    if plant_backend == "auto":
        plant_backend = meta.get("plant_backend")
        if plant_backend is None:
            plant_backend = "prototype" if config.get("obs_dim", 15) == 15 else "safe-control-gym"

    print(f"Loaded: {args.checkpoint}")
    print(f"  Controller: {config.get('controller_type', 'mamba')}")
    print(f"  Plant backend: {plant_backend}")
    print(f"  Phase: {meta.get('phase', '?')}  Epoch/Cycle: {meta.get('epoch', meta.get('cycle', '?'))}")
    print()

    print(f"Results over {args.missions} missions:")
    if plant_backend == "safe-control-gym":
        metrics = evaluate_safe_control_gym(controller, args)
        print(f"  Avg return:                  {metrics['avg_return']:.3f}")
        print(f"  Avg final position error:    {metrics['avg_final_position_error']:.3f} m")
        print(f"  Action MSE vs MPC:           {metrics['avg_action_mse_vs_mpc']:.6f}")
        print(f"  Avg constraint violations:   {metrics['avg_constraint_violations']:.2f}")
        if args.plot:
            print("  Plotting is currently available only for prototype checkpoint rollouts.")
        return

    metrics = evaluate_prototype(controller, args)
    print(f"  Checkpoint ordering success: {metrics['checkpoint_order_success_rate']:.0%}")
    print(f"  Docking success:             {metrics['docking_success_rate']:.0%}")
    print(f"  Avg final distance to dock:  {metrics['avg_final_distance']:.2f} m")
    print(f"  Avg checkpoints reached:     {metrics['avg_checkpoints_reached']:.1f} / 3")
    stl = metrics["stl_robustness_single_traj"]
    print(f"  STL robustness (1 traj):     {stl:.3f} {'✓' if stl > 0 else '✗'}")

    # Optional plot
    if args.plot:
        plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)
        init_state = plant.reset()
        expert = DroneExpertController()
        expert.set_plant_ref(plant)

        # Mamba trajectory (15D input with phase tracking)
        plant.state = init_state.copy()
        plant.time = 0.0
        controller.reset()
        m_traj, y = [], plant.state.copy()
        phase_idx = 0
        for _ in range(args.seq_steps):
            target = CHECKPOINTS[phase_idx]
            y_with_target = np.concatenate([y, target])
            u = controller.forward(y_with_target)
            y = plant.step(u, args.dt)
            m_traj.append(y)
            if phase_idx < 3 and np.linalg.norm(y[0:3] - target) < CHECKPOINT_RADIUS:
                phase_idx += 1

        # Expert trajectory (same init)
        plant.state = init_state.copy()
        plant.time = 0.0
        expert.reset()
        e_traj, y = [], plant.state.copy()
        for _ in range(args.seq_steps):
            u = expert.compute_action(y)
            y = plant.step(u, args.dt)
            e_traj.append(y)

        out = Path(args.checkpoint).parent / "eval_trajectory.png"
        plot_trajectory(m_traj, e_traj, cycle="eval", filename=str(out))
        print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
