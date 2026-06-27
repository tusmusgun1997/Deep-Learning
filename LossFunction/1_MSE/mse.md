# Mean Squared Error (MSE) — Deep Dive

## What It Does

MSE measures the **average squared difference** between predictions and true values. Squaring punishes large errors much more than small ones.

```
Formula:   L = (1/n) · Σ (ŷᵢ - yᵢ)²

Derivative: dL/dŷ = (2/n) · (ŷ - y)

Output Range: [0, ∞)   (0 = perfect)
```

**Intuition:** If your prediction is off by 2, the loss is 4. If off by 10, the loss is 100. Big errors are **disproportionately punished**.

---

## Advantages

1. **Differentiable everywhere** — smooth gradient, great for backpropagation.
2. **Heavily penalizes outliers** — the model is forced to pay attention to large mistakes.
3. **Simple to understand and compute** — just square the error and average it.
4. **Convex** — for linear regression, guaranteed to find the global minimum.

---

## Disadvantages

1. **Sensitive to outliers** — one bad data point with a huge error can dominate the loss.
2. **Units are squared** — if you're predicting salary in dollars, loss is in dollars², which is hard to interpret. Use RMSE (√MSE) for interpretability.
3. **Not robust** — outliers pull the model's predictions toward themselves.

---

## When to Use MSE

✅ **Regression problems** — predicting house prices, temperatures, scores.
✅ When **large errors are especially bad** and should be penalized hard.
❌ When your data has many **outliers** — use MAE or Huber instead.
❌ **Classification** — use Cross-Entropy instead.

---

## Dry Run — Forward & Backward Pass

### Setup
```
Predictions:  ŷ = [3.0, 5.0, 2.5]
True values:  y  = [2.0, 5.0, 4.0]
n = 3
```

### Step 1: Compute Errors
```
  error₁ = ŷ₁ - y₁ = 3.0 - 2.0 =  1.0
  error₂ = ŷ₂ - y₂ = 5.0 - 5.0 =  0.0
  error₃ = ŷ₃ - y₃ = 2.5 - 4.0 = -1.5
```

### Step 2: Square the Errors
```
  error₁² = 1.0² = 1.00
  error₂² = 0.0² = 0.00
  error₃² = 1.5² = 2.25
```

### Step 3: Compute Loss
```
  L = (1/3) · (1.00 + 0.00 + 2.25)
    = (1/3) · 3.25
    = 1.0833
```

### Step 4: Compute Gradients (for backprop)
```
  dL/dŷ₁ = (2/3) · (3.0 - 2.0) = (2/3) · 1.0  =  0.6667
  dL/dŷ₂ = (2/3) · (5.0 - 5.0) = (2/3) · 0.0  =  0.0000
  dL/dŷ₃ = (2/3) · (2.5 - 4.0) = (2/3) · (-1.5) = -1.0000
```

**Key Insight:** Sample 3 (error = -1.5) gets a gradient of -1.0, while sample 1 (error = 1.0) gets 0.667. The larger error dominates — that's MSE's "outlier sensitivity" in action.
