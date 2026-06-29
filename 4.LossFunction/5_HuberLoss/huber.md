# Huber Loss — Deep Dive

## What It Does

Huber Loss is the **best of both MSE and MAE**. For small errors it behaves like MSE (smooth, penalizes more), for large errors it behaves like MAE (linear, robust to outliers). A threshold δ (delta) controls the boundary.

```
Formula:
  L(error) = 0.5 · error²              if |error| ≤ δ   ← MSE zone
             δ · (|error| - 0.5·δ)     if |error| > δ   ← MAE zone

  where error = ŷ - y

Derivative:
  dL/dŷ = error                        if |error| ≤ δ   ← smooth gradient
           δ · sign(error)             if |error| > δ   ← capped gradient

Output Range: [0, ∞)   (0 = perfect)
```

**Intuition:** Think of δ as a dial — small errors are treated with MSE precision, large errors (outliers) are treated with MAE robustness. Typical δ values: 1.0, 1.35.

---

## The Threshold δ in Action

```
δ = 1.0:

  error = 0.5  → MSE zone: loss = 0.5 · 0.5² = 0.125,  gradient = 0.5
  error = 1.0  → boundary:  loss = 0.5 · 1.0² = 0.5,   gradient = 1.0
  error = 2.0  → MAE zone:  loss = 1.0·(2.0 - 0.5) = 1.5, gradient = 1.0 (capped)
  error = 5.0  → MAE zone:  loss = 1.0·(5.0 - 0.5) = 4.5, gradient = 1.0 (capped)

Compare:
  MSE at error=5.0: loss = 25.0  ← massive! gradient = 10.0 ← exploding!
  Huber at error=5.0: loss = 4.5 ← much less, gradient = 1.0 ← bounded
```

---

## Advantages

1. **Outlier robust** — large errors don't dominate the loss or cause exploding gradients.
2. **Smooth everywhere** — differentiable at error = δ (unlike MAE at 0).
3. **Faster convergence than MAE** — squared gradients for small errors push harder.
4. **Tunable** — δ lets you control how sensitive to outliers you want to be.

---

## Disadvantages

1. **Extra hyperparameter δ** — you need to tune it. Wrong δ hurts performance.
2. **More complex** — two-regime formula is harder to implement and reason about.
3. **Not as standard** — MSE is the default; Huber requires explicit choice.

---

## When to Use Huber

✅ **Regression with outliers** — predicting prices, demand, sensor values with noise spikes.
✅ When MSE's exploding gradients cause instability but MAE's flat gradients are too slow.
✅ **Reinforcement learning** (DQN) — the original paper used Huber for the Bellman error.
❌ **Clean data without outliers** — just use MSE.
❌ **Classification** — use Cross-Entropy.

---

## Dry Run — Forward & Backward Pass

### Setup
```
δ = 1.0
Predictions:  ŷ = [2.0, 5.5, 10.0]
True values:  y  = [2.0, 4.0,  4.0]
n = 3
```

### Step 1: Compute Errors
```
  error₁ = 2.0 - 2.0  =  0.0
  error₂ = 5.5 - 4.0  =  1.5   ← |error| > δ=1.0  (MAE zone)
  error₃ = 10.0 - 4.0 =  6.0   ← |error| > δ=1.0  (MAE zone — outlier!)
```

### Step 2: Compute Per-Sample Loss
```
  Sample 1: |0.0| ≤ 1.0  →  MSE zone:  0.5 · 0.0² = 0.000
  Sample 2: |1.5| > 1.0  →  MAE zone:  1.0 · (1.5 - 0.5) = 1.000
  Sample 3: |6.0| > 1.0  →  MAE zone:  1.0 · (6.0 - 0.5) = 5.500
```

### Step 3: Average Loss
```
  L = (1/3) · (0.000 + 1.000 + 5.500)
    = (1/3) · 6.500
    = 2.1667

  Compare with MSE:  (0² + 1.5² + 6²) / 3 = (0 + 2.25 + 36) / 3 = 12.75
  Huber is MUCH lower because it caps the outlier's contribution.
```

### Step 4: Compute Gradients
```
  dL/dŷ₁ = (1/3) · 0.0              = 0.0000   (in MSE zone)
  dL/dŷ₂ = (1/3) · 1.0 · sign(1.5) = +0.3333  (in MAE zone, capped at δ)
  dL/dŷ₃ = (1/3) · 1.0 · sign(6.0) = +0.3333  (in MAE zone, capped at δ)

  Compare with MSE gradients:
    dL/dŷ₂ (MSE) = (2/3) · 1.5 = +1.000
    dL/dŷ₃ (MSE) = (2/3) · 6.0 = +4.000   ← exploding due to outlier!
```

**Key Insight:** The outlier (sample 3, error = 6.0) gets the **same gradient as sample 2** (error = 1.5) under Huber. MSE would give sample 3 a gradient 4× larger — letting the outlier dominate training.
