# TinyWorlds Autoresearch Program

## Decision

Use `MiniGrid-Empty-8x8-v0` as the first low-entropy world-modeling dataset.

Why this first:

- It has a tiny, discrete, visually interpretable state space.
- It has known discrete actions, so action conditioning can be evaluated directly instead of only through inferred latent actions.
- It is cheap enough for an AI-Scientist-style loop to run many short experiments.
- It can scale later into harder MiniGrid variants such as FourRooms, DoorKey, DynamicObstacles, and BabyAI tasks.

Keep Pong as the second benchmark, not the first. Pong is visually low entropy and closer to existing TinyWorlds video data, but the currently convenient TinyWorlds Pong file is actionless. The action-labeled Atari Pong dataset exists, but its ArrayRecord format adds loader work before the research loop can start.

## Task

Build a small automated research loop for action-conditioned visual world models.

The system should propose cheap model or training changes, run short experiments, evaluate the resulting world model, and update a compact memory of which ideas improved controllable prediction. The goal is not a large paper-generation wrapper. The goal is a substrate where foundation-model agents can discover useful world-modeling tricks under strict compute constraints.

## Dataset

Primary dataset:

- Environment: `MiniGrid-Empty-8x8-v0`
- Observation: rendered RGB frames resized to `64x64`
- Action space: MiniGrid's discrete actions
- Stored format: `data/minigrid_empty8x8_actions.h5`
- Default size: `128` episodes, up to `128` steps each
- Required arrays:
  - `frames`: `uint8`, shape `[N, H, W, C]`
  - `actions`: `int64`, shape `[N - 1]`, where `actions[t]` leads from `frames[t]` to `frames[t + 1]`
  - `episode_starts`: `int64`, start frame indices for each episode
  - `episode_lengths`: `int64`, frame count per episode
  - `splits/train`, `splits/val`, `splits/test`: frame index ranges

The setup script creates this dataset locally. It should be treated as generated data and not committed.

## Optimization Metric

Primary metric: action-conditioned next-frame MSE.

For a model `f`, frame `x_t`, and ground-truth action `a_t`, evaluate:

```text
MSE_true = mean((f(x_<=t, a_t) - x_{t+1})^2)
```

This is the first score because it is simple, cheap, stable, and directly aligned with one-step dynamics.

Secondary metric: counterfactual action sensitivity.

For the same context frame, predict next frames under each legal action:

```text
y_a = f(x_<=t, a) for all a in A
```

Then measure:

```text
CounterfactualSpread = mean_pairwise_mse({y_a})
ActionMargin = min_{a != a_t} MSE(y_a, x_{t+1}) - MSE(y_{a_t}, x_{t+1})
```

The model is better when:

- `MSE_true` is low.
- `ActionMargin` is positive.
- `CounterfactualSpread` is nonzero on states where actions should visibly matter.

This guards against a degenerate model that ignores actions and predicts the most likely next frame.

Tertiary metric: short rollout MSE.

Starting from one true context frame and a logged action sequence, roll out `K in {4, 8, 16}` steps autoregressively:

```text
x_hat_{t+1} = f(x_hat_<=t, a_t)
```

Report per-step MSE and rollout drift. Rollout should be a secondary metric at first because it is noisier and slower, but it is necessary for judging whether one-step gains produce usable world models.

## Evaluation Rule

Rank experiments by:

```text
score = MSE_true
      + 0.25 * rollout_mse_8
      - 0.10 * max(ActionMargin, 0)
      + 0.01 * compute_minutes
```

Lower is better. Keep component metrics visible; do not hide the tradeoff behind the scalar score.

## Autoresearch Loop

Each research iteration should:

1. Read this file and the current experiment memory.
2. Propose one small hypothesis.
3. Modify one config or one local model component.
4. Run a short training budget.
5. Evaluate next-frame, counterfactual, and rollout metrics.
6. Save a compact result record.
7. Decide whether to keep, mutate, or discard the idea.

Good first hypotheses:

- Increase or decrease action embedding dimension.
- Add explicit action dropout and test whether counterfactual margin improves.
- Compare direct pixel prediction against tokenized next-frame prediction.
- Add a contrastive counterfactual loss that pushes predictions for different actions apart only when the dataset says the next frame differs.
- Search context length `1, 2, 4`.
- Test whether MiniGrid symbolic observations are useful as an auxiliary target while RGB remains the primary prediction target.

## Constraints

- Optimize for short, repeatable experiments over impressive single runs.
- Prefer changes that produce interpretable failure cases.
- Log every experiment with config, git diff summary, metrics, and a short conclusion.
- Do not use generated paper quality as the objective. The objective is useful experimental discovery.

## Setup

Create the default dataset with:

```bash
python3 setup.py
```

For a fast smoke test:

```bash
python3 setup.py --episodes 8 --max-steps 32 --out data/minigrid_smoke.h5
```

Run one five-minute autoresearch trial with:

```bash
python3 train.py
```

`train.py` is the editable genome. `setup.py` is fixed infrastructure for dataset creation, dataloading, and architecture-agnostic evaluation.
