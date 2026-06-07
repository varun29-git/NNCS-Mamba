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
MPC + STL falsification -> expert/corrective supervision -> dual-head Mamba controller
```

The implementation should focus first on the core architecture:

- MPC expert trajectory collection.
- Quantitative STL robustness labels for complete rollouts.
- A shared Mamba sequence encoder over state/action history.
- An action/control head that predicts the next 4D quadrotor control action.
- An STL value head that predicts trajectory robustness or safety margin.
- Cached online inference so deployment uses constant-time `O(1)` per-step control updates for a fixed model size.
- STL-guided CEGIS retraining from negative-robustness counterexamples.

MLP and GRU baselines are still useful, but they should not drive the rebuild. They are comparison models to add after the core dual-head Mamba pipeline works.

## Core Architecture Target

The target controller is:

```text
historical trajectory window
    -> shared Mamba sequence encoder
        -> action/control head: 4D control action
        -> STL value head: predicted STL robustness / safety margin
```

Use `action head` or `control head` in implementation and project documentation. The action head is analogous to AlphaGo's policy head, but this project is a continuous-control NNCS, so the control terminology is clearer.

The learned STL value head is not the mathematical verifier. It is a learned judge/predictor used for robustness awareness and later CEGIS prioritization. The actual STL monitor in `nncs_mamba/stl_monitor.py` remains the authority for safety/task satisfaction.

The deployed controller must expose an online API:

```text
action, value, cache = controller.step(obs, cache)
```

For a fixed model size, this API must avoid reprocessing the full history at every control step. The implementation should benchmark mean, p95, and p99 per-step latency and report whether cached Mamba inference is fixed-cost, `O(1)` per step.

## Files to Preserve

These files are the trusted/current foundation:

```text
nncs_mamba/safe_control_gym_config.py
nncs_mamba/stl_monitor.py
tests/test_stl_monitor.py
tests/__init__.py
README.md
docs/RESEARCH.md
requirements.txt
docs/ALPHAGO_PATTERN_FOR_NNCS.md
docs/reference/AlphaGo_Architecture_Course.pdf
docs/STL_Monitor_Report.tex
docs/STL_Monitor_Report.pdf
docs/ALPHAGO_IDEAS_FOR_NNCS.md
docs/ALPHAGO_IDEAS_FOR_NNCS.tex
docs/ALPHAGO_IDEAS_FOR_NNCS.pdf
```

## Current Local Status

Completed in the rebuild process:

- Removed the old learner/training stack.
- Created a fresh local `.venv` with Python 3.11.
- Installed the non-PyBullet foundation dependencies.
- Installed the pinned Safe-Control-Gym package with `--no-deps` to avoid the old Torch dependency conflict.
- Confirmed `tests/test_stl_monitor.py` passes in `.venv`.
- Added `scripts/verify_safe_control_gym.py`.
- Added `scripts/verify_stl_on_expert.py`.
- Added `nncs_mamba/dataset.py` and `scripts/collect_dataset.py` for MPC expert trajectory collection with STL labels.
- Generated real MPC expert datasets on the approved CPU VM `tok64`:
  - `runs/expert_dataset_tiny_tok64`: 10 rollouts, 300 steps each.
  - `runs/expert_dataset_core_tok64`: 240 rollouts, 300 steps each.

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
.venv/bin/python scripts/verify_safe_control_gym.py --steps 5
.venv/bin/python scripts/verify_stl_on_expert.py --rollouts 3 --steps 100
```

## Implementation Order

### Step 1: Verify Safe-Control-Gym Interface

Create:

```text
scripts/verify_safe_control_gym.py
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
scripts/verify_stl_on_expert.py
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

### Step 3: Expert Dataset Collection With STL Labels

Create:

```text
nncs_mamba/dataset.py
scripts/collect_dataset.py
```

Purpose:

- Collect MPC demonstrations from Safe-Control-Gym.
- Store trajectories in a simple format.
- Evaluate each trajectory with `evaluate_stabilization_stl`.
- Save state/action arrays, STL labels, and benchmark metadata.

Minimum saved fields:

```text
states
actions
sample_time_sec
initial_state
goal_position
action_low
action_high
stl_robustness
stl_satisfied
component_robustness
total_reward
done
safe_control_gym_config
mpc_config
seed
```

Acceptance criteria:

- Dataset can be loaded independently.
- Shapes are documented.
- STL labels can be reproduced by rerunning `nncs_mamba/stl_monitor.py` on saved states/actions.
- A small dataset can be inspected by hand.

### Step 4: Controller Interface And Rollout Contracts

Create:

```text
models/controller.py
rollout.py
```

Purpose:

- Define the contract shared by training, evaluation, and deployment.
- Support full-sequence training calls and cached online control calls.
- Keep environment rollout logic separate from model architecture.
- Standardize action scaling/clipping and value-head output naming.

Required controller API:

```text
controller.forward_sequence(states) -> action_pred, value_pred
controller.initial_cache(batch_size) -> cache
controller.step(obs, cache) -> action, value, cache
```

Acceptance criteria:

- A dummy controller can be rolled out through the evaluation path.
- `step` has stable input/output shapes for single-env deployment.
- The API makes cached inference explicit before Mamba is implemented.
- No model-specific logic is embedded in the Safe-Control-Gym wrappers.

### Step 5: Dual-Head Mamba Core

Create:

```text
models/mamba.py
models/heads.py
```

Purpose:

- Implement or wrap a reviewable Mamba/structured-SSM sequence encoder.
- Add an action/control head for 4D quadrotor actions.
- Add an STL value head for predicted robustness or safety margin.
- Support both sequence training and cached online inference.

Acceptance criteria:

- Each architectural choice is documented.
- Parameter count is reported.
- `forward_sequence` runs on batched trajectories.
- Repeated `step` calls are tested against sequence-mode outputs where practical.
- Cached `step` does not reprocess the full history at every control step.
- Mean, p95, and p99 per-step inference latency are reported.

### Step 6: Core Dual-Head Training

Create:

```text
train_mamba.py
```

Purpose:

- Load the expert dataset with STL labels.
- Train the Mamba action/control head to imitate MPC actions.
- Train the STL value head to predict robustness.
- Save checkpoints and metadata.
- Keep the first version free of CEGIS, soft targets, uncertainty, and baseline comparisons.

Training target:

```text
action_target = MPC action
value_target = trajectory STL robustness
```

Loss:

```text
L_total = L_action + lambda_value * L_value
```

Preferred first losses:

```text
L_action = MSE(action_pred, action_target)
L_value = Huber(value_pred, stl_robustness)
```

The first implementation may attach the same trajectory-level robustness label to every timestep/window. A later refinement can train prefix-conditioned or component-wise value targets.

Acceptance criteria:

- Can overfit a tiny dataset as a sanity check.
- Reports train/validation action MSE and MAE.
- Reports robustness prediction MAE.
- Reports safe/unsafe classification quality from `value_pred >= 0`.
- Saves model config, dataset manifest, loss weights, and benchmark metadata.

### Step 7: Core Evaluation Script

Create:

```text
evaluate_controller.py
```

Purpose:

- Load a trained Mamba controller.
- Roll it out in Safe-Control-Gym using the cached `step` API.
- Compare actions against MPC when needed.
- Compute STL robustness and task metrics.
- Measure whether the value head predicts STL outcomes reliably.

Metrics:

- action MSE/MAE versus MPC;
- final distance to goal;
- minimum distance to goal;
- STL satisfaction rate;
- average STL robustness;
- component robustness values;
- robustness prediction MAE;
- value-head violation detection AUROC if both safe and unsafe examples exist;
- mean, p95, and p99 inference time per step;
- parameter count and checkpoint size.

Acceptance criteria:

- Evaluation can compare MPC expert and learned Mamba rollouts.
- Results are saved as JSON/CSV.
- The reported safety decision always comes from the STL monitor, not from the value head alone.

### Step 8: STL-Guided CEGIS Correction Loop

Create:

```text
cegis.py
```

Purpose:

- Roll out the current Mamba controller.
- Use the STL monitor to find negative-robustness trajectories.
- Treat negative-STL rollouts as counterexamples.
- Relabel failed initial states or failure regions with MPC.
- Add corrected trajectories and STL labels to the dataset.
- Retrain the dual-head Mamba model.

Acceptance criteria:

- Clearly separates:
  - epoch;
  - CEGIS iteration;
  - expert-labeled sample count.
- Saves each CEGIS iteration result.
- Reports STL improvement across iterations.
- Tracks how many MPC relabeling calls were required.
- Keeps the STL monitor as the final counterexample authority.

### Step 9: Value-Guided CEGIS Prioritization

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
- Confirm that prioritized candidates are still checked by the real STL monitor.

### Step 10: Minimal Baselines For Comparison

Create:

```text
models/mlp.py
models/gru.py
```

Purpose:

- Add small MLP and GRU baselines only after the core Mamba pipeline works.
- Use the same dataset, splits, rollout logic, STL monitor, and evaluation metrics.
- Keep baseline implementations simple enough to defend.

Acceptance criteria:

- Parameter counts are reported.
- Same controller API as Mamba where possible.
- No special tricks beyond what is documented.
- Baselines are used for comparison, not as the main implementation spine.

### Step 11: Soft MPC Target Distributions

Create:

```text
soft_targets.py
```

Purpose:

- Around each MPC action, sample candidate actions.
- Score candidates with short-horizon cost or STL-inspired safety margin.
- Convert scores into soft weights.
- Train the action/control head on weighted action targets.

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

The core experiment should include:

```text
Mamba action/control head only
Dual-head Mamba: action/control head + STL value head
Dual-head Mamba + STL-guided CEGIS
Dual-head Mamba + value-guided CEGIS
```

Later comparison experiments may add:

```text
MLP
GRU
Mamba + soft MPC targets
```

All models must use:

- same Safe-Control-Gym environment;
- same MPC expert;
- same train/validation/test split;
- comparable parameter budgets where practical;
- same STL monitor;
- same evaluation seeds;
- same disturbance/robustness profiles.

## Do Not Implement Yet

Do not add the following until the core dual-head Mamba pipeline is verified:

- soft MPC targets;
- uncertainty estimation;
- distributional action policies;
- custom docking task;
- custom quadrotor dynamics;
- heavy baseline sweeps.

The first priority is a simple, defensible dual-head Mamba controller with STL-labeled expert data, cached online inference, and STL-monitor evaluation.
