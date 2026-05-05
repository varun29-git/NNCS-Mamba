# Research Configuration

## Benchmark Source

- Plant/controller source: Safe-Control-Gym
- Upstream URL: https://github.com/learnsyslab/safe-control-gym
- Pinned dependency: `safe-control-gym @ git+https://github.com/learnsyslab/safe-control-gym.git@v1.0.0`
- Secondary sanity-check reference: MathWorks nonlinear MPC quadrotor example and dynamics derivation.

## Plant

The research plant is the Safe-Control-Gym 3D quadrotor stabilization task using PyBullet physics.

State order:

| Index | Name | Unit |
|---:|---|---|
| 0 | `x` | m |
| 1 | `x_dot` | m/s |
| 2 | `y` | m |
| 3 | `y_dot` | m/s |
| 4 | `z` | m |
| 5 | `z_dot` | m/s |
| 6 | `phi` | rad |
| 7 | `theta` | rad |
| 8 | `psi` | rad |
| 9 | `p` | rad/s |
| 10 | `q` | rad/s |
| 11 | `r` | rad/s |

Control setup:

- Control frequency: 50 Hz
- PyBullet frequency: 1000 Hz
- Sample time: 0.02 s
- Task: stabilization to `[x, y, z] = [0, 0, 1]`
- Constraints: Safe-Control-Gym default state and input constraints
- Disturbances: none in the nominal benchmark

The machine-readable version of this configuration is written to `benchmark_manifest.json` in each training output directory.

## Expert Controller

The teacher is Safe-Control-Gym `mpc`, not a recreated controller.

Current MPC settings:

- Horizon: 20
- Solver: IPOPT
- Warm start: enabled
- State cost diagonal: `[5.0, 0.1, 5.0, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]`
- Input cost diagonal: `[0.1, 0.1, 0.1, 0.1]`

Run the expert-only validation:

```bash
python evaluate.py --expert-only --missions 20 --seq-steps 300
```

## Training Terms

- `epoch`: one optimization pass over the current imitation dataset.
- `CEGIS iteration`: rollout learner, identify negative-STL counterexamples, label those initial states with Safe-Control-Gym MPC, add them to the data, retrain.
- `sample count`: number of MPC-labeled trajectories used for imitation.

## Metrics

Evaluation reports:

- Controller similarity: action MSE and MAE versus MPC.
- Task performance: return and final position error.
- Safety: constraint violation count.
- STL: satisfaction rate and quantitative robustness.
- Runtime: learner inference time and MPC control/labeling time per step.

## STL Specification

The current formal specification is for stabilization:

```text
G safe_state
AND G input_within_bounds
AND F near_goal
AND F G settled
```

Robustness uses standard quantitative min/max semantics:

- `G phi = min_t rho(phi, t)`
- `F phi = max_t rho(phi, t)`
- `phi AND psi = min(rho(phi), rho(psi))`

The implementation lives in `stl_monitor.py`; basic satisfying and violating trajectories are tested in `tests/test_stl_monitor.py`.

## Experiment Commands

Compare MLP, GRU, and Mamba:

```bash
python research_experiments.py compare --num-traj 1024 --epochs 8 --missions 20
```

Sample-efficiency sweep:

```bash
python research_experiments.py sample-efficiency --sample-counts 128 512 2048
```

Robustness sweep:

```bash
python research_experiments.py robustness --checkpoint runs/experiment/best_imitation.pt
```

CEGIS ablation:

```bash
python train.py --phase imitation --outdir runs/imitation
python train.py --phase all --resume runs/imitation/best_imitation.pt --outdir runs/cegis
```
