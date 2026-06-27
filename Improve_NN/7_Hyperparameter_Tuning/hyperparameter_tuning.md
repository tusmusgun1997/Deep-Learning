# Hyperparameter Tuning — Finding the Best Network Configuration

## Parameters vs Hyperparameters

- **Parameters** — the **weights** (`W`) and **biases** (`b`). Learned **automatically** by the model during training via backpropagation. You never set these by hand.
- **Hyperparameters** — configuration you set **manually before** training. They control *how* the model learns and *what shape* it has.

```
Parameters      → learned BY the model    (millions of them, set by gradient descent)
Hyperparameters → chosen BY you           (a handful, set before training starts)
```

**Analogy:** Hyperparameters are the **oven settings** (temperature, time, rack position); parameters are the **batter** that actually turns into the cake. Same recipe, wrong oven settings → ruined cake.

---

## The Key Hyperparameters (and what they do)

| Hyperparameter | Too Low | Too High | Typical Range |
|----------------|---------|----------|---------------|
| **Learning rate** ⭐ | Crawls, gets stuck | Overshoots, diverges (NaN) | 1e-4 – 1e-1 (log scale) |
| **# Hidden layers** | Underfits | Vanishing grads, overfits | 1 – 5 (more with ResNet) |
| **# Neurons / layer** | Underfits | Overfits, slow | 32 – 512 |
| **Batch size** | Noisy, slow wall-clock | Poor generalization, memory-heavy | 16 – 256 |
| **Epochs** | Underfits | Overfits | use early stopping |
| **Dropout rate** | Little regularization | Underfits | 0.2 – 0.5 |
| **Weight decay (λ)** | Overfits | Underfits | 1e-5 – 1e-2 (log scale) |
| **Optimizer** | — | — | Adam (default), SGD+Momentum |

> ⭐ **Learning rate is the most important.** If you can only tune one thing, tune this.

### Quick guidance on the structural ones
```
Underfitting?  → MORE capacity: add layers/neurons, train longer, reduce regularization
Overfitting?   → LESS capacity: fewer layers/neurons, add dropout/weight decay, get more data
```
- **Depth:** start with **1–2 hidden layers**; a single hidden layer is already a universal approximator. Add depth only if underfitting.
- **Width:** a common starting point is `≈ 2 × num_input_features`; funnel down (e.g. 128 → 64 → 32) is popular.
- **Batch size:** **32–128** is the usual sweet spot. Small batches add helpful noise (better generalization); large batches train faster per epoch but can land in "sharp" minima that generalize worse.

---

## Search Strategy 1: Grid Search

Try **every combination** of a predefined set of values.

```
learning_rate ∈ {0.001, 0.01, 0.1}
hidden_size   ∈ {32, 64, 128}
→ 3 × 3 = 9 runs (try all of them)
```
- ✅ Simple; guarantees full coverage of the grid.
- ❌ **Curse of dimensionality** — cost explodes: 5 hyperparameters × 5 values each = **3125** runs. Wastes time re-testing bad values of one parameter across all values of the others.

---

## Search Strategy 2: Random Search

Sample random combinations from ranges/distributions.

```
Grid Search (9 pts)          Random Search (9 pts)
   •   •   •                    •      •
   •   •   •                       •  •     •
   •   •   •                    •        •     •
 only 3 distinct LR values    9 DISTINCT LR values explored
```
- ✅ **Statistically better than grid** (Bergstra & Bengio, 2012): most hyperparameters barely matter, so random search spends its budget exploring *many* distinct values of the few that *do* matter — instead of wasting runs on redundant values of the unimportant ones.
- ❌ Still "blind" — it doesn't learn from earlier trials.
- **Verdict:** for the same compute, random search usually beats grid search. A great default.

---

## Search Strategy 3: Bayesian Optimization

**Learns from past trials.** Build a probabilistic **surrogate model** (often a Gaussian Process) mapping hyperparameters → validation score, then use an **acquisition function** to pick the most promising next configuration — balancing:
- **Exploitation** — try near the best-known point.
- **Exploration** — try uncertain regions that might hide something better.

```
Trial 1,2,3 (random) → fit surrogate → "where's the score likely highest?" → Trial 4
→ update surrogate → Trial 5 → ... converges on the optimum in FEW trials
```
- ✅ **Very sample-efficient** — finds great configs in far fewer runs. Ideal when each training run is expensive.
- ❌ More complex; harder to parallelize than random search. (Tools: Optuna, Hyperopt, scikit-optimize.)

---

## Search Strategy 4: Hyperband (Successive Halving)

**Spend compute on winners; cut losers early.** Train *many* random configs for a *tiny* budget, keep the top half, double their budget, repeat.

```
81 configs × 1 epoch  → keep top 27
27 configs × 3 epochs → keep top 9
 9 configs × 9 epochs → keep top 3
 3 configs × 27 epochs → pick the best
```
- ✅ **Blazing fast** — kills hopeless runs after just a few epochs instead of training them fully.
- ⚠️ Assumes early performance predicts final performance (usually true, occasionally not).

---

## Strategy Comparison

| Strategy | Learns from trials? | Efficiency | Complexity |
|----------|:--:|------------|------------|
| **Grid Search** | No | Poor (explodes with dimensions) | Trivial |
| **Random Search** | No | Good (great default) | Trivial |
| **Bayesian Opt** | Yes | Excellent (few trials) | High |
| **Hyperband** | Via early-stopping | Excellent (cheap) | Medium |

---

## Practical Tuning Recipe

1. **Search on a LOG scale** for LR and weight decay: `{1e-4, 1e-3, 1e-2, 1e-1}`, not `{0.01, 0.02, 0.03}`. These matter most across orders of magnitude.
2. **Coarse-to-fine.** First a wide search with few epochs to find a promising region; then a narrow search there with more epochs.
3. **Tune in priority order:** learning rate → batch size → architecture (layers/width) → regularization (dropout, weight decay).
4. **Always judge on a VALIDATION set**, never training loss — otherwise you'll pick a configuration that overfits.
5. **Use early stopping during the search** so bad configs die quickly and you test more of them.
6. **Change one thing at a time** (or use a proper search method) so you can attribute what helped.
7. **Set a seed** so results are comparable run-to-run.

---

## Common Mistakes

1. **Tuning on the test set.** That contaminates your final estimate — validation only.
2. **Linear-scale LR search.** You'll miss the order of magnitude that actually matters.
3. **Grid search over many hyperparameters.** Use random search or Bayesian opt instead.
4. **Tuning everything at once with no baseline.** Get a simple model working first, then tune.
5. **Over-tuning.** Chasing a 0.1% gain by obsessively tuning is rarely worth it — better data usually beats better hyperparameters.

---

## TL;DR

- **Parameters** are learned; **hyperparameters** are chosen by you before training.
- **Learning rate** is king — tune it first, on a **log scale**.
- Prefer **Random Search** over Grid Search; step up to **Bayesian Optimization** or **Hyperband** when runs are expensive.
- Always select on **validation** performance, go **coarse-to-fine**, and lean on **early stopping** to test more configs cheaply.
