# Mean Absolute Error (MAE) — Deep Dive

## What It Does

MAE measures the **average absolute difference** between predictions and true values. Every error is treated equally — no squaring, no extra penalty for large mistakes.

```
Formula:    L = (1/n) · Σ |ŷᵢ - yᵢ|

Derivative: dL/dŷ = (1/n) · sign(ŷ - y)
            where sign(x) = +1 if x > 0, -1 if x < 0, 0 if x = 0

Output Range: [0, ∞)   (0 = perfect)
```

**Intuition:** Off by 2? Loss increases by 2. Off by 10? Loss increases by 10. Every unit of error costs exactly the same.

---

## Advantages

1. **Robust to outliers** — a huge error contributes linearly, not quadratically. One bad point won't dominate.
2. **Interpretable units** — if predicting salary in dollars, loss is in dollars (not dollars²).
3. **Simple gradient** — always ±1/n, never explodes.

---

## Disadvantages

1. **Non-differentiable at zero** — the absolute value has a "kink" at 0. Subgradient is used in practice.
2. **Flat gradients** — gradient magnitude is constant (±1/n) regardless of error size. Large errors get the same update signal as small ones → slower convergence.
3. **Not convex in some settings** — can make optimization harder vs. MSE.

---

## When to Use MAE

✅ **Regression with outliers** — predicting noisy real-world data (stock prices, sensor readings).
✅ When you want **equal treatment** for all error magnitudes.
❌ When **large errors are especially costly** — use MSE instead.
❌ **Classification** — use Cross-Entropy.

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

### Step 2: Absolute Values
```
  |error₁| = 1.0
  |error₂| = 0.0
  |error₃| = 1.5
```

### Step 3: Compute Loss
```
  L = (1/3) · (1.0 + 0.0 + 1.5)
    = (1/3) · 2.5
    = 0.8333
```

### Step 4: Compute Gradients
```
  dL/dŷ₁ = (1/3) · sign(1.0)  = (1/3) · (+1) = +0.3333
  dL/dŷ₂ = (1/3) · sign(0.0)  = (1/3) ·   0  =  0.0000
  dL/dŷ₃ = (1/3) · sign(-1.5) = (1/3) · (-1) = -0.3333
```

**Key Insight:** Sample 3 has a larger error (1.5) than sample 1 (1.0), but both get the **same gradient magnitude** (0.3333). This is MAE's "flat gradient" — it doesn't push harder for bigger errors.

### MSE vs MAE Side-by-Side
```
Same data, same errors:

          error    MSE gradient    MAE gradient
sample 1:  1.0       0.6667          0.3333
sample 2:  0.0       0.0000          0.0000
sample 3: -1.5      -1.0000         -0.3333

MSE gives bigger gradient to bigger error. MAE treats them equally.
```
