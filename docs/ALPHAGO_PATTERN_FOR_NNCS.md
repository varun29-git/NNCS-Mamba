# AlphaGo Pattern Worth Borrowing

## Core Link

The useful AlphaGo idea is not Go, board games, or literal MCTS. The useful pattern is:

```text
expensive improvement procedure -> dense supervision -> fast neural policy
```

In AlphaGo:

- raw policy network proposes moves
- MCTS improves the move distribution
- the network is trained to imitate the improved search distribution
- over time, expensive search is distilled into a fast forward pass

In this project:

- Mamba/GRU/MLP propose quadrotor actions
- Safe-Control-Gym MPC improves or corrects behavior
- STL robustness verifies whether trajectories satisfy safety/task constraints
- the neural controller is trained to imitate MPC and then corrected through CEGIS

So the strong analogy is:

```text
AlphaGo MCTS  <->  MPC + STL falsification
AlphaGo policy net  <->  neural controller
AlphaGo value net  <->  learned STL/cost-to-go predictor
AlphaGo self-play  <->  rollout/falsify/relabel/retrain
```

This is a real conceptual fit because both systems use a slow deliberative procedure to generate better supervision than the raw neural policy can produce alone.

## The Best Concrete Idea

Add an **improvement-distillation view** to the project:

> The neural controller should not merely imitate MPC actions once. It should repeatedly distill MPC-improved behavior discovered by STL-guided counterexample search.

This gives a clean research framing:

```text
Initial imitation:
    train policy pi_theta on MPC demonstrations

Improvement:
    roll out pi_theta
    find negative-STL trajectories
    call MPC from failed initial states

Distillation:
    add MPC-corrected trajectories to the dataset
    retrain pi_theta

Repeat:
    policy becomes a fast amortized approximation of MPC under the falsifier distribution
```

That is already close to the current CEGIS loop, but the AlphaGo lens makes the contribution clearer:

> CEGIS is our test-time search/improvement operator, and Mamba is the amortized controller that distills it.

## Stronger Variant: Add A Value Head

AlphaGo did not only learn a policy; it also learned a value function.

For this project, the equivalent value target is not win probability. It should be:

```text
predicted STL robustness
```

or:

```text
predicted MPC cost-to-go / final tracking error
```

A useful extension would be to add a second head to each controller:

```text
policy head: action prediction
value head: predicted STL robustness or safety margin
```

Then evaluation can ask:

- Does Mamba imitate MPC actions?
- Does Mamba predict whether its own rollout will be safe?
- Can predicted robustness help prioritize CEGIS counterexamples?

This is more defensible than adding a value head for vague “reward” because STL robustness is already part of the project’s formal specification.

## Stronger Variant: Soft Targets Instead Of Single MPC Actions

AlphaGo trains on an improved action distribution, not just one hard move.

MPC normally gives one action. But we can create richer supervision by sampling candidate actions around the MPC action:

1. At a state `x`, get MPC action `u*`.
2. Sample candidate actions near `u*`.
3. Roll each candidate briefly or score it with model/MPC cost/STL margin.
4. Convert scores into a soft target distribution or weighted regression target.
5. Train the policy to prefer actions with better robustness/cost, not merely copy one action.

This could increase “bits per sample,” which is one of the key AlphaGo lessons from the PDF.

Do not implement this immediately. It is a second-stage idea after the current MPC imitation + CEGIS pipeline is validated.

## What Not To Force

Do **not** claim the drone problem needs literal MCTS.

Reasons:

- Quadrotor control is continuous-action, not a discrete board game.
- MPC is already the natural planning/search method for this domain.
- Tree search over continuous controls would be expensive and less defensible than using the trusted MPC implementation.

The right statement is:

> We borrow AlphaGo’s improvement-distillation principle, not its game-specific tree search implementation.

## How To Use This In The Project Writeup

A good research framing would be:

> Inspired by AlphaGo’s search-improvement distillation, we use MPC as a trusted planner to generate expert demonstrations and STL-guided CEGIS as a counterexample search mechanism. The neural controller learns to amortize this expensive planning process into fast inference, and is compared against MLP and GRU baselines under identical MPC supervision.

This is a genuine link to the PDF and does not overclaim.

## Recommended Next Step

Do not change the code yet.

First validate the current pipeline:

```bash
python evaluate.py --expert-only --missions 20 --seq-steps 300
python train.py --phase imitation --controller mamba --epochs 10
python evaluate.py --checkpoint runs/experiment/best_imitation.pt
```

Only after that, consider adding:

1. a robustness/value head,
2. soft MPC-improvement targets,
3. CEGIS prioritization using predicted robustness.

