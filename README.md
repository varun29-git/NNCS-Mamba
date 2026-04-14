# NNCS-Mamba: Counter-Example Guided Imitation Learning

This repository implements a framework to synthesize Neural Network Control Systems (NNCS) utilizing recurrent state-space models (**Mamba / SSMs**) rather than standard stateless Multi-Layer Perceptrons (MLPs). It focuses on guaranteeing Parametric Signal Temporal Logic (PSTL) specifications for dynamic unobserved states in control environments.

## Current Capabilities

- **12D Drone Plant Simulation (`drone_env.py`)**: A continuous drone simulation evaluating aggressive 3D docking maneuvers, incorporating hidden mathematical dynamics (unobservable periodic wind gusts) designed to force recurrent memory usage.
- **Sequential Expert Controller (`drone_env.py`)**: A robust algorithmic demonstrator that perfectly executes sequential logic pathways (visiting a Pre-Dock coordinate before descending to the absolute Origin), generating flawless reference trajectories.
- **Mamba SSM Controller (`mamba_learner.py`)**: A neural controller integrating `mamba-ssm`. It achieves native parallel scan scaling during trajectory training sequences and shifts to constant $O(1)$ recurrent sequence processing using `InferenceParams` during online tracking. 
- **Automated CEGIS Loop (`cegis_loop.py`)**: A discrete programmatic iteration script that pits a Temporal Logic Falsifier against the neural model, structurally isolating edge-cases to iteratively synthesize a perfectly stable control policy.

## The Operational Workflow 

1. **Initialization**: The CEGIS framework starts the Mamba model with an extremely small baseline imitation dataset (~5 trajectories).
2. **Parallel Train**: The model minimizes MSE error natively across the sequential data lengths using its hardware-accelerated mapping layer.
3. **Falsify**: A test matrix simulates the neural model acting live on `30` severe zero-shot edge conditions. A formal routine verifies bounded logic parameters (e.g., maximum target overshoot $s_{ov} < 15.0$ and ultimate target stabilization $s_{st} < 2.0$). 
4. **Fix & Merge**: If an initial state condition breaks the neural controller's logic loop, it triggers the Expert Controller to simulate perfectly from that exact coordinate, splicing the correct algorithmic response back into Mamba's dataset.
5. **Convergence**: This iterative interaction continues recursively until the *Performance Similarity index* confirms zero failures, meaning the SSM policy formally bounds to the logic specifications.

## What You Need to Do Next

1. **GPU Environment Transfer**: Push this codebase to a CUDA-enabled machine/cluster. Set up the environment running `pip install -r requirements.txt`. (Strictly requires GPU configuration to correctly build both `mamba-ssm` and `causal-conv1d`).
2. **Run Execution**: Initiate the core `python cegis_loop.py` script. Watch the terminal iteratively increase its performance similarity parameter until it caps at 100%. 
3. **Transition to Target Software Simulation**: Currently, the `DronePlant` utilizes an internalized affine set of derivatives. Once verified, refactor `DronePlant.step()` to connect directly with your formal Quadrotor aerodynamic simulator or API (e.g. IsaacGym, PyBullet).
