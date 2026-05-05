# NNCS-Mamba (under development)

This repository trains sequence controllers to imitate an MPC expert on the Safe-Control-Gym 3D quadrotor benchmark.

## Research Hardening Status

The plant and MPC expert are provided directly by Safe-Control-Gym: https://github.com/learnsyslab/safe-control-gym

## Overview
- Safe-Control-Gym provides the physics-based 3D quadrotor environment.
- Safe-Control-Gym MPC generates expert state/action demonstrations.
- Neural controllers learn to imitate the MPC from trajectory data.

## Neural Architecture (Mamba) (`mamba_learner.py`)
- Uses a Mamba block (a selective state‑space model) that maintains a hidden state across time steps.
- Takes the 12‑D Safe-Control-Gym quadrotor observation and outputs 4D control actions.

## Baseline Controller (`gru_learner.py`)
- Adds a cuDNN-backed GRU baseline for comparison under the same data and evaluation setup.

## Optimizer
- Split optimizer: **Muon** for the linear weight matrices and **AdamW** for all other parameters.
- Mixed‑precision training with PyTorch AMP.
- Learning‑rate scheduler reduces the rate when validation loss stops improving.

## Training Pipeline (`train.py`)
1. **Smoke test** – quick run to verify the model compiles.
2. **Expert data generation** – collect trajectories from Safe-Control-Gym MPC.
3. **Imitation learning** – train Mamba or GRU on expert trajectories.

## Evaluation (`evaluate.py`)
- Loads a saved checkpoint and runs a number of rollouts.
- Reports return, final position error, action MSE versus MPC, and constraint violations.

## Current Status
- The project has been simplified to one defensible environment/controller source.
- Counterexample-guided retraining and formal STL robustness should be reintroduced only against Safe-Control-Gym trajectories.

## How to Run
```bash
# Clone the repository
git clone https://github.com/varun29-git/NNCS-Mamba.git
cd NNCS-Mamba

# Install dependencies (requires a CUDA‑enabled machine)
pip install -r requirements.txt

# Research-grade plant/MPC dependency
git clone https://github.com/learnsyslab/safe-control-gym.git
cd safe-control-gym
python -m pip install -e .
cd ../NNCS-Mamba

# Train on Safe-Control-Gym quadrotor MPC demonstrations
python train.py --phase imitation --epochs 10

# Evaluate on the same Safe-Control-Gym physics plant
python evaluate.py --checkpoint runs/experiment/best_imitation.pt

# GRU baseline
python train.py --phase imitation --controller gru --profile t4-sota --outdir runs/gru_baseline
```

Feel free to open issues or pull requests as the project evolves.
