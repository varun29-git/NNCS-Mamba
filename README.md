# NNCS-Mamba

This repository is rebuilding a safe neural network control system for the Safe-Control-Gym 3D quadrotor stabilization task.

The current focus is the core architecture:

```text
MPC expert trajectories
    -> STL robustness labels
    -> dual-head Mamba controller
        -> action/control head
        -> STL value head
```

The learned value head predicts robustness. The STL monitor remains the actual safety evaluator.

## Repository Layout

```text
nncs_mamba/        Core Python package.
scripts/           Command-line verification and dataset collection scripts.
tests/             Unit tests.
docs/              Research notes, implementation plan, and reports.
runs/              Generated datasets and experiment outputs. Ignored by git.
```

## Current Pipeline

- Safe-Control-Gym provides the quadrotor plant and MPC expert.
- `nncs_mamba.stl_monitor` computes quantitative STL robustness.
- `nncs_mamba.dataset` collects MPC trajectories and saves STL-labeled datasets.
- `nncs_mamba.models.controller` defines the action/control head plus STL value head interface.
- `nncs_mamba.models.mamba` implements the first dual-head structured-SSM controller core.
- `nncs_mamba.training` trains the dual-head controller from MPC actions and STL robustness labels.
- `nncs_mamba.rollout` evaluates any controller through the same environment loop.
- The first real datasets were generated on the approved CPU VM `tok64`.
- The first 1M-parameter model was trained on an AWS `g6e.xlarge` L40S GPU.

The model files require PyTorch. The simulator foundation is kept separate
because Safe-Control-Gym pins an older Torch range, while this Python 3.11
environment uses modern Torch for the neural controller.

## Quick Checks

Run local STL tests:

```bash
python -m unittest
```

Verify the Safe-Control-Gym + MPC boundary when PyBullet is available:

```bash
python scripts/verify_safe_control_gym.py --steps 5
```

Evaluate STL robustness on MPC expert rollouts:

```bash
python scripts/verify_stl_on_expert.py --rollouts 3 --steps 100
```

Collect a small expert dataset:

```bash
python scripts/collect_dataset.py --rollouts 10 --steps 300 --output-dir runs/expert_dataset_tiny
```

Benchmark the controller core on CPU:

```bash
python scripts/benchmark_mamba_controller.py --steps 1000
```

Run a small supervised training pass:

```bash
python scripts/train_mamba.py --epochs 1 --limit-rollouts 8 --device cpu
```

## Generated Datasets

The generated dataset files are ignored by git and live under `runs/`.

Current local artifacts:

```text
runs/expert_dataset_tiny_tok64/
runs/expert_dataset_core_tok64/
runs/mamba_l40s_1m/
```

The core dataset has:

```text
states:  (240, 301, 12)
actions: (240, 300, 4)
STL satisfaction rate: 1.0
```

## Research Docs

- [Research Configuration](docs/RESEARCH.md)
- [Next Implementation Plan](docs/NEXT_IMPLEMENTATION_PLAN.md)
- [AlphaGo Pattern](docs/ALPHAGO_PATTERN_FOR_NNCS.md)
- [Quadrotor Basics](docs/notes/Quadrator_Basics.md)
