# NNCS-Mamba (under development)

This repository implements a framework for training a neural‑network controller for a 12‑dimensional drone using Counter‑Example Guided Inductive Synthesis (CEGIS). The code is still being written and is not yet complete.

## Overview
- A simulated drone with a 12‑D state (position, velocity, tilt, angular rates).
- The mission consists of three steps: reach checkpoint A, then checkpoint B, then dock at the origin.
- Hidden disturbances: a leaky fuel tank (mass decreases) and invisible wind gusts.

## Simulation Environment (`drone_env.py`)
- Provides the physics of the drone, including fuel loss and wind.
- Defines three checkpoints and a radius (`CHECKPOINT_RADIUS = 1.5`) for successful visits.
- Keeps a phase counter that tells the controller which checkpoint is next.

## Expert (Teacher) Controller (`drone_env.py`)
- A physics‑based controller that knows the exact mass and wind.
- Generates perfect trajectories that the neural controller can imitate.
- Used to create corrective examples during the CEGIS refinement phase.

## Neural Architecture (Mamba) (`mamba_learner.py`)
- Uses a Mamba block (a selective state‑space model) that maintains a hidden state across time steps.
- The hidden state works like a small notebook, remembering which checkpoints have been visited.
- Takes a 15‑D observation (12‑D state + 3‑D target waypoint) and outputs thrust commands.

## Optimizer
- Split optimizer: **Muon** for the linear weight matrices and **AdamW** for all other parameters.
- Mixed‑precision training with PyTorch AMP.
- Learning‑rate scheduler reduces the rate when validation loss stops improving.

## Training Pipeline (`train.py`)
1. **Smoke test** – quick run to verify the model compiles.
2. **Imitation learning** – train on expert trajectories (default 5,000) for a configurable number of epochs.
3. **CEGIS refinement** – generate failure states with a Cross‑Entropy Method (CEM), let the expert correct them, and retrain on the combined data.

## CEGIS Loop (`cegis_loop.py`)
- Generates a population of 2,000 candidate states and evaluates them with a Signal Temporal Logic (STL) score.
- Failures are fixed by the expert (`fix_and_merge`) and added to the training buffer.
- Logs the number of failures, safety coverage, and loss after each cycle.

## Evaluation (`evaluate.py`)
- Loads a saved checkpoint and runs a number of rollouts.
- Reports the success rate for visiting checkpoints in order and the average number of checkpoints reached.
- Can optionally plot a single trajectory with checkpoint markers.

## Current Status
- A 45‑minute run on a free Tesla T4 GPU showed the model can reach checkpoints A and B reliably.
- Docking still fails; the model needs more training steps.
- The new Muon optimizer is expected to double the learning speed.

## How to Run
```bash
# Clone the repository
git clone https://github.com/varun29-git/NNCS-Mamba.git
cd NNCS-Mamba

# Install dependencies (requires a CUDA‑enabled machine)
pip install -r requirements.txt

# Run the full training and CEGIS loop
python train.py --epochs 10 --cegis-cycles 5

# Evaluate a saved checkpoint
python evaluate.py --checkpoint runs/full/best_cegis.pt --missions 20 --plot
```

Feel free to open issues or pull requests as the project evolves.
