# Next Implementation Plan

## Current Decision

We are rebuilding the active implementation from scratch so every file can be reviewed and defended. The old learner/training stack has been removed because it predates the current Safe-Control-Gym + STL direction and is too hard to trust without re-reviewing every assumption.

The preserved foundation is:

- Safe-Control-Gym plant and MPC integration.
- STL robustness monitor.
- STL tests and reports.
- AlphaGo-inspired idea reports.
- Research documentation.

The new implementation should be built in small, reviewable files. Each file should have a clear role and should be understandable before the next file is added.

## Guiding Research Direction

We are following the AlphaGo-inspired framing:

```text
expensive improvement procedure -> richer supervision -> fast neural controller
```

For this project:

```text
MPC + STL falsification -> expert/corrective supervision -> Mamba/MLP/GRU controller
```

The implementation should eventually include:

- MPC imitation learning.
- STL robustness evaluation.
- CEGIS retraining from negative-STL counterexamples.
- MLP and GRU baselines.
- Mamba controller.
- STL value head for robustness prediction.
- Optional value-guided CEGIS prioritization.
- Optional soft MPC target distributions around the MPC action.

## Files to Preserve

These files are the trusted/current foundation:

```text
safe_control_gym_config.py
stl_monitor.py
tests/test_stl_monitor.py
tests/__init__.py
README.md
RESEARCH.md
requirements.txt
ALPHAGO_PATTERN_FOR_NNCS.md
AlphaGo_Architecture_Course.pdf
docs/STL_Monitor_Report.tex
docs/STL_Monitor_Report.pdf
docs/ALPHAGO_IDEAS_FOR_NNCS.md
docs/ALPHAGO_IDEAS_FOR_NNCS.tex
docs/ALPHAGO_IDEAS_FOR_NNCS.pdf
```

## Implementation Order

## Current Local Status

Completed in the rebuild process:

- Removed the old learner/training stack.
- Created a fresh local `.venv` with Python 3.11.
- Installed the non-PyBullet foundation dependencies.
- Installed the pinned Safe-Control-Gym package with `--no-deps` to avoid the old Torch dependency conflict.
- Confirmed `tests/test_stl_monitor.py` passes in `.venv`.
- Added `verify_safe_control_gym.py`.
- Added `verify_stl_on_expert.py`.

Current local blocker:

```text
pybullet is not installed.
```

Both verifier scripts currently stop with a clear blocked message because PyBullet failed to build from source on this macOS environment. This is a local simulator dependency issue, not a logic issue in the verifier scripts.

Observed failure:

```text
pip install pybullet==3.2.7
-> Failed building wheel for pybullet
```

Next environment task:

- Install PyBullet through a working route, preferably a Python/environment combination with a prebuilt wheel, or a Conda environment where PyBullet is available.
- Then rerun:

```text
.venv/bin/python verify_safe_control_gym.py --steps 5
.venv/bin/python verify_stl_on_expert.py --rollouts 3 --steps 100
```

### Step 1: Verify Safe-Control-Gym Interface

Create:

```text
verify_safe_control_gym.py
```

Purpose:

- Instantiate the Safe-Control-Gym quadrotor environment.
- Instantiate the Safe-Control-Gym MPC controller.
- Reset the environment.
- Run a small number of MPC-controlled steps.
- Print state shape, action shape, action bounds, goal position, and basic rollout status.

Acceptance criteria:

- Environment creation works.
- MPC returns valid actions.
- State vector matches the documented 12D order.
- Action bounds are available and understood.
- No learner code is involved.

### Step 2: Verify STL Monitor Against Real Rollouts

Create:

```text
verify_stl_on_expert.py
```

Purpose:

- Run the MPC expert for several rollouts.
- Collect states and actions.
- Evaluate each rollout with `evaluate_stabilization_stl`.
- Print component robustness scores.

Acceptance criteria:

- Expert rollouts produce interpretable STL scores.
- Positive/negative robustness values are explainable.
- Component scores identify the limiting requirement.

### Step 3: Expert Dataset Collection

Create:

```text
dataset.py
```

Purpose:

- Collect MPC demonstrations from Safe-Control-Gym.
- Store trajectories in a simple format.
- Save both state/action arrays and metadata.

Minimum saved fields:

```text
states
actions
goal_position
action_low
action_high
safe_control_gym_config
mpc_config
seed
```

Acceptance criteria:

- Dataset can be loaded independently.
- Shapes are documented.
- A small dataset can be inspected by hand.

### Step 4: Minimal MLP Baseline

Create:

```text
models/mlp.py
```

Purpose:

- Implement a simple memoryless MLP.
- Input: current observation/state.
- Output: action.
- Train using supervised MPC action imitation.

Acceptance criteria:

- Code is short and reviewable.
- Uses AdamW only.
- No auxiliary losses initially.
- Can overfit a tiny dataset as a sanity check.

### Step 5: Minimal Training Script

Create:

```text
train_imitation.py
```

Purpose:

- Load expert dataset.
- Train one selected model.
- Save checkpoint and metadata.
- Report train/validation action MSE.

Acceptance criteria:

- Works first for MLP only.
- No CEGIS yet.
- No value head yet.
- No soft targets yet.

### Step 6: Evaluation Script

Create:

```text
evaluate_controller.py
```

Purpose:

- Load a trained controller.
- Roll it out in Safe-Control-Gym.
- Compare against MPC actions when needed.
- Compute STL robustness and task metrics.

Metrics:

- action MSE/MAE versus MPC;
- final distance to goal;
- minimum distance to goal;
- STL satisfaction rate;
- average STL robustness;
- component robustness values;
- inference time per step.

Acceptance criteria:

- Same evaluation script can evaluate MPC expert and learned controllers.
- Results are saved as JSON/CSV.

### Step 7: GRU Baseline

Create:

```text
models/gru.py
```

Purpose:

- Implement a small recurrent baseline.
- Use the same dataset, splits, and metrics as MLP.

Acceptance criteria:

- Parameter count is reported.
- Same training/evaluation interface as MLP.
- No special tricks beyond what is documented.

### Step 8: Mamba Controller

Create:

```text
models/mamba.py
```

Purpose:

- Implement or wrap a reviewable Mamba-style sequence controller.
- Keep the architecture as simple as possible.
- Match the same interface as MLP/GRU.

Acceptance criteria:

- Each architectural choice is documented.
- Parameter count is comparable to baselines.
- Same dataset and evaluation process.

### Step 9: CEGIS Loop

Create:

```text
cegis.py
```

Purpose:

- Roll out the current learner.
- Use STL robustness to find failed trajectories.
- Treat negative-STL rollouts as counterexamples.
- Relabel failed initial states or failure regions with MPC.
- Add corrected data to the dataset.
- Retrain.

Acceptance criteria:

- Clearly separates:
  - epoch;
  - CEGIS iteration;
  - expert-labeled sample count.
- Saves each CEGIS iteration result.
- Reports STL improvement across iterations.

### Step 10: STL Value Head

Create or extend:

```text
models/value_heads.py
```

Purpose:

- Add a value head that predicts STL robustness.
- Start with total STL robustness only.
- Later support component robustness prediction.

Training target:

```text
value_target = trajectory STL robustness
```

Loss:

```text
L_total = L_action + lambda_value * L_value
```

Acceptance criteria:

- Reports robustness prediction MAE.
- Reports safe/unsafe classification quality.
- Does not replace the STL monitor.

### Step 11: Value-Guided CEGIS Prioritization

Extend:

```text
cegis.py
```

Purpose:

- Use predicted STL robustness to prioritize rollouts for real STL checking and MPC relabeling.

Priority score:

```text
priority = -predicted_stl_robustness
```

Optional uncertainty:

```text
priority = -predicted_stl_robustness + beta * uncertainty
```

Acceptance criteria:

- Compare against random CEGIS selection.
- Measure counterexamples found per rollout.
- Measure MPC relabeling calls needed per improvement.

### Step 12: Soft MPC Target Distributions

Create:

```text
soft_targets.py
```

Purpose:

- Around each MPC action, sample candidate actions.
- Score candidates with short-horizon cost or STL-inspired safety margin.
- Convert scores into soft weights.
- Train the policy on weighted action targets.

Candidate generation:

```text
u_candidate = clip(u_mpc + noise, action_low, action_high)
```

Weighted imitation loss:

```text
L_soft = sum_k w_k * ||u_pred - u_candidate_k||^2
```

Acceptance criteria:

- First tested on a small number of states.
- Scores and weights are inspectable.
- Compared against single-point MPC imitation.

## Final Experiment Structure

The final research comparison should include:

```text
MLP
GRU
Mamba
Mamba + STL value head
Mamba + STL value head + value-guided CEGIS
Mamba + soft MPC targets
```

All models must use:

- same Safe-Control-Gym environment;
- same MPC expert;
- same train/validation/test split;
- comparable parameter budgets;
- same STL monitor;
- same evaluation seeds;
- same disturbance/robustness profiles.

## Do Not Implement Yet

Do not add the following until the basic pipeline is verified:

- value-guided CEGIS;
- soft MPC targets;
- uncertainty estimation;
- distributional action policies;
- custom docking task;
- custom quadrotor dynamics.

The first priority is a simple, defensible MPC imitation pipeline with STL evaluation.
