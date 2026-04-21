"""
Evaluate a trained NNCS-Mamba checkpoint.

Usage:
    python evaluate.py --checkpoint runs/full/best_cegis.pt
    python evaluate.py --checkpoint runs/full/best_cegis.pt --missions 20 --plot
"""
import argparse
import numpy as np
import torch
from pathlib import Path

from drone_env import DronePlant, DroneExpertController, CHECKPOINTS, CHECKPOINT_RADIUS
from mamba_learner import MambaController
from cegis_loop import (
    check_stl_score,
    run_validation_rollouts,
    plot_trajectory,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Mamba controller.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--missions", type=int, default=10, help="Number of evaluation rollouts")
    parser.add_argument("--seq-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--plot", action="store_true", help="Save a trajectory comparison plot")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    controller = MambaController(
        obs_dim=config.get("obs_dim", 15),
        action_dim=config.get("action_dim", 4),
        d_model=config.get("d_model", 128),
        d_state=config.get("d_state", 16),
        num_layers=config.get("num_layers", 3),
        lr=config.get("lr", 3e-4),
        optimizer_name=config.get("optimizer_name", "split_muon"),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", True),
        aux_state_weight=config.get("aux_state_weight", 0.5),
    )
    checkpoint = controller.load_checkpoint(args.checkpoint)
    meta = checkpoint.get("metadata", {})
    print(f"Loaded: {args.checkpoint}")
    print(f"  Phase: {meta.get('phase', '?')}  Epoch/Cycle: {meta.get('epoch', meta.get('cycle', '?'))}")
    print()

    plant = DronePlant(m0=2.5, fuel_rate=0.02, m_min=1.0)

    # Run rollouts
    metrics = run_validation_rollouts(
        controller, plant,
        num_missions=args.missions,
        seq_steps=args.seq_steps,
        dt=args.dt,
    )

    print(f"Results over {args.missions} missions:")
    print(f"  Checkpoint ordering success: {metrics['checkpoint_order_success_rate']:.0%}")
    print(f"  Docking success:             {metrics['docking_success_rate']:.0%}")
    print(f"  Avg final distance to dock:  {metrics['avg_final_distance']:.2f} m")
    print(f"  Avg checkpoints reached:     {metrics['avg_checkpoints_reached']:.1f} / 3")

    # STL robustness on a single trajectory (with 15D target concatenation)
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
    stl = check_stl_score(traj)
    print(f"  STL robustness (1 traj):     {stl:.3f} {'✓' if stl > 0 else '✗'}")

    # Optional plot
    if args.plot:
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
