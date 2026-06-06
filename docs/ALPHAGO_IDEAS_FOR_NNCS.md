# AlphaGo-Inspired Ideas for NNCS-Mamba

## Purpose

This document identifies the AlphaGo concepts that can be meaningfully reused in the NNCS-Mamba drone-control project. The goal is not to copy AlphaGo directly. The drone problem is continuous-control, physics-based, and safety-constrained, while Go is a discrete board game. The useful transfer is at the level of training architecture:

```text
slow improvement procedure -> richer supervision -> fast neural policy
```

In AlphaGo, Monte Carlo Tree Search (MCTS) improves the raw policy network and produces a better action distribution. The neural network then learns to imitate this improved distribution. In NNCS-Mamba, the corresponding improvement mechanism is not MCTS; it is MPC plus STL-based falsification/counterexample generation.

The strongest link is therefore:

```text
AlphaGo MCTS improvement       <->   MPC correction + STL falsification
AlphaGo policy head            <->   neural controller action head
AlphaGo value head             <->   predicted STL robustness / safety margin
AlphaGo self-play data loop     <->   rollout, falsify, relabel, retrain loop
AlphaGo soft move distribution  <->   soft action landscape around MPC action
```

## Why This Is Relevant

AlphaGo did not rely only on a neural network making a one-shot guess. It used a slow, expensive, higher-quality procedure to improve decisions, then distilled those improved decisions back into a fast neural model.

That pattern fits this project because MPC is accurate but expensive, while Mamba is fast but must be trained carefully. STL robustness gives a formal way to decide whether a rollout is safe and successful. Therefore, the project can be framed as:

```text
Use MPC and STL to generate high-quality supervision.
Train Mamba to approximate that supervision with fast inference.
Use CEGIS to find failure regions and improve the dataset.
```

This is a defensible AlphaGo-inspired research direction because it borrows the improvement-distillation principle, not the game-specific tree-search machinery.

## Idea 1: Add a Value Head for STL Robustness Prediction

### Core Idea

AlphaGo uses two outputs:

- a policy head, which predicts what action to take;
- a value head, which predicts how good the current state is.

For NNCS-Mamba, the policy head already corresponds to action prediction. The natural value target is not game win probability. It should be a safety/task score that already exists in the project:

```text
predicted STL robustness
```

The controller would become a two-head network:

```text
shared Mamba encoder
    -> policy head: predicted action
    -> value head: predicted STL robustness
```

The value head estimates whether the current state/history is likely to produce a safe and successful rollout.

### Why STL Robustness Is the Right Value Target

STL robustness is better than an arbitrary reward because it has formal meaning:

- positive robustness means the trajectory satisfies the specification;
- negative robustness means the trajectory violates at least one requirement;
- the magnitude indicates safety margin or violation severity.

This makes the value head interpretable. It is not merely predicting "reward"; it is predicting the formal safety/task margin used by the project.

### Possible Training Target

For each rollout, the STL monitor returns:

```text
stl_robustness
state_safety_robustness
input_safety_robustness
eventually_reach_robustness
eventually_always_settled_robustness
```

The simplest version predicts only final STL robustness:

```text
value_target = final trajectory STL robustness
```

A stronger version predicts multiple component scores:

```text
value_head =
    [
      total STL robustness,
      state safety robustness,
      input safety robustness,
      reach robustness,
      settled robustness
    ]
```

The multi-output version is more useful diagnostically because it can indicate why the controller is likely to fail.

### Loss Function

The training loss could combine imitation loss and value prediction loss:

```text
L_total = L_action + lambda_value * L_value
```

Where:

```text
L_action = MSE(predicted_action, MPC_action)
L_value  = Huber(predicted_STL_robustness, measured_STL_robustness)
```

Huber loss is preferable to plain MSE if robustness scores contain large outliers.

An optional classification term can also be added:

```text
safe_label = 1 if STL robustness >= 0 else 0
```

Then:

```text
L_total = L_action
        + lambda_value * L_robustness_regression
        + lambda_safe * L_satisfaction_classification
```

This lets the model learn both the continuous robustness margin and the binary safe/unsafe decision.

## Optional Extension: Use the Value Head for CEGIS Prioritization

### Motivation

CEGIS can become expensive because many rollouts may need to be evaluated and relabeled by MPC. A value head can help prioritize which cases deserve expensive expert attention.

The value head should not be treated as a safety certificate. It is only a triage mechanism.

### Proposed Loop

```text
1. Train Mamba on MPC demonstrations.
2. Roll out Mamba on many initial states or disturbance settings.
3. Use the value head to predict STL robustness.
4. Prioritize trajectories predicted to have low or negative robustness.
5. Evaluate those trajectories with the real STL monitor.
6. For confirmed failures, ask MPC to relabel/correct behavior.
7. Add these corrected examples to the dataset.
8. Retrain Mamba.
```

This is similar in spirit to AlphaGo using a value network to focus search. In this project, the predicted value focuses counterexample discovery and relabeling.

### Prioritization Score

A simple prioritization score is:

```text
priority = - predicted_STL_robustness
```

Lower predicted robustness means higher priority.

A better score can include uncertainty:

```text
priority = - predicted_STL_robustness + beta * uncertainty
```

Uncertainty can be estimated using an ensemble, dropout, or disagreement between Mamba/GRU/MLP value predictions.

### Evaluation

This idea should be judged by whether it improves the efficiency of CEGIS:

- number of unsafe trajectories found per rollout;
- number of MPC relabeling calls needed to improve the controller;
- STL satisfaction rate after each CEGIS iteration;
- robustness distribution before and after value-guided prioritization.

The main question is:

```text
Does value-guided CEGIS find useful failures faster than random rollout selection?
```

## Idea 2: Soft Target Distributions Around MPC Actions

### Core Idea

The current imitation setup treats the MPC action as one exact target:

```text
state x -> MPC action u*
```

This is simple, but it loses information. In many control states, several nearby actions may be acceptable, and some may be only slightly worse than the MPC action. AlphaGo avoids this kind of hard target by training on an improved move distribution, not only a single best move.

For NNCS-Mamba, the equivalent idea is to build a soft local action distribution around the MPC action.

Instead of teaching:

```text
Mamba should output exactly 0.80
```

teach:

```text
0.80 is best,
0.78 and 0.82 are also good,
0.85 is acceptable but worse,
0.70 is unsafe or poor.
```

This teaches Mamba the local landscape of safe actions rather than forcing it to memorize a single rigid number.

### Example

Suppose MPC gives:

```text
u* = 0.80
```

Sample candidate actions near it:

```text
0.78, 0.80, 0.82, 0.85
```

Each candidate is tested using a short rollout, local model prediction, MPC cost, constraint margin, or STL-inspired score. The scores are converted into weights:

```text
0.78 -> 0.25
0.80 -> 0.40
0.82 -> 0.25
0.85 -> 0.10
```

The learner is then trained toward this soft target, not a single point.

### General Continuous-Action Version

At state \(x_t\), MPC returns action \(u_t^*\). Generate \(K\) candidate actions:

```text
u_t^(k) = clip(u_t^* + epsilon_k, action_low, action_high)
```

Each candidate receives a score:

```text
score_k = short_horizon_quality(x_t, u_t^(k))
```

The score can be based on:

- short-horizon tracking error;
- MPC stage cost;
- distance to constraints;
- predicted STL margin;
- a weighted combination of cost and safety margin.

Convert candidate scores into a probability distribution:

```text
w_k = softmax(score_k / temperature)
```

If lower cost is better, use:

```text
w_k = softmax(-cost_k / temperature)
```

The temperature controls how sharp the target is. A low temperature makes the target close to the best candidate. A high temperature keeps more actions partially acceptable.

### Possible Training Loss

For a deterministic Mamba policy, use weighted regression:

```text
L_soft = sum_k w_k * || predicted_action - u_t^(k) ||^2
```

This pulls the prediction toward the center of the good-action region, not blindly toward one number.

A more advanced version makes Mamba output a Gaussian action distribution:

```text
policy output = mean and covariance
```

Then the sampled candidate distribution can supervise both:

- the mean action;
- the uncertainty or width of acceptable actions.

This would be useful when the safe action region is wide in some states and narrow near constraints.

### Why This Could Help Mamba

Mamba is sequence-based. It should benefit from learning not only what the expert did, but also how sensitive the action decision is around the expert action.

Soft targets may improve:

- smoothness of learned control;
- robustness to small state-estimation errors;
- generalization near unseen states;
- safety near constraints;
- sample efficiency, because each MPC query produces multiple learning signals.

This is directly analogous to the AlphaGo idea that an improved probability distribution gives more learning information than a single hard label.

## Relationship Between the Two Ideas

The value head and soft target distribution can support each other.

The value head predicts whether a rollout or state is safety-critical:

```text
Which states need more attention?
```

The soft target distribution teaches the policy how actions behave locally:

```text
Which nearby actions are good or unsafe?
```

Together:

```text
value head -> selects important states
soft targets -> provide richer action supervision at those states
```

This creates a stronger improvement-distillation loop:

```text
1. Roll out Mamba.
2. Predict STL robustness with value head.
3. Prioritize risky states/trajectories.
4. Query MPC at those states.
5. Sample action candidates around MPC.
6. Score candidates using short-horizon cost/STL margin.
7. Train Mamba on soft action targets and robustness targets.
```

## Recommended Priority

The ideas should not all be implemented at once. The recommended order is:

### Phase 1: Validate the Existing Pipeline

Before adding AlphaGo-inspired extensions:

```text
MPC expert works
STL monitor works
imitation training works
CEGIS loop works
MLP/GRU/Mamba baseline comparison works
```

This phase is required because the project must first be defensible as a control-learning system.

### Phase 2: Add the STL Value Head

This is the safest AlphaGo-inspired extension because the target already exists. The STL monitor already computes the labels.

Minimum experiment:

```text
Mamba without value head
vs
Mamba with STL value head
```

Compare:

- action imitation error;
- STL satisfaction rate;
- robustness prediction MAE;
- violation detection AUROC;
- CEGIS counterexamples found per rollout.

### Phase 3: Add Value-Guided CEGIS Prioritization

Only after the value head is reasonably accurate, use it to prioritize counterexamples.

The value head should not replace the STL monitor. The actual STL monitor remains the final judge.

### Phase 4: Add Soft MPC Target Distributions

This is the more ambitious idea. It is promising, but it adds compute and design choices:

- how many candidate actions to sample;
- how to score candidates;
- how long the short rollout should be;
- how to choose the softmax temperature;
- whether to train deterministic or distributional policies.

It should be treated as a second-stage contribution after the basic MPC imitation plus CEGIS results are solid.

## What Should Not Be Claimed

The project should not claim that drone control needs literal MCTS. That would be forced.

Reasons:

- quadrotor control has continuous actions;
- MPC is already the standard planning/control tool for this domain;
- tree search over continuous thrust vectors is less natural and may be computationally expensive;
- safety claims should remain tied to MPC, Safe-Control-Gym, and STL robustness.

The appropriate claim is:

```text
We borrow AlphaGo's improvement-distillation pattern:
an expensive optimizer/verifier produces richer supervision,
and a neural controller distills that supervision into fast inference.
```

## Possible Writeup Paragraph

Inspired by AlphaGo's separation between fast neural intuition and slower search-based improvement, this project can treat MPC and STL falsification as the improvement mechanism for quadrotor control. MPC provides high-quality expert actions, while STL robustness evaluates whether complete trajectories satisfy formal safety and task requirements. The Mamba controller then amortizes this expensive procedure into a fast policy. As an extension, a value head can be trained to predict STL robustness, analogous to AlphaGo's value network, and can be used to prioritize counterexamples during CEGIS. A further extension is to replace single-action MPC imitation with soft local action distributions around the MPC action, allowing the controller to learn the landscape of safe actions rather than only one expert point.

## Summary of Proposed Contributions

| Idea | AlphaGo analogy | NNCS-Mamba version | Risk level |
|---|---|---|---|
| MPC imitation | Policy learns from improved search | Mamba imitates MPC actions | Low |
| STL value head | Value network predicts win probability | Value head predicts STL robustness | Medium |
| Value-guided CEGIS | Value focuses search | Predicted robustness prioritizes failures | Medium |
| Soft action targets | Train on MCTS visit distribution | Train on scored action samples around MPC | Higher |

The most practical next contribution is the STL value head. The most original second-stage contribution is the soft MPC target distribution.
