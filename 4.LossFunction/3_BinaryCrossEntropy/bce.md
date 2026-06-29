# Binary Cross-Entropy (BCE) — Deep Dive

## What It Does

Binary Cross-Entropy measures how different your **predicted probability** is from the true binary label (0 or 1). It's the go-to loss for **binary classification**.

```
Formula:   L = -(1/n) · Σ [ yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ) ]

Derivative: dL/dŷ = (1/n) · [ -y/ŷ + (1-y)/(1-ŷ) ]
            (simplifies to: (ŷ - y) / (ŷ · (1 - ŷ))  per sample)

Output Range: [0, ∞)   (0 = perfect)
```

**Intuition:**
- If y = 1 (spam): only the left term activates → `-log(ŷ)`. Predicting ŷ = 0.99 gives tiny loss. Predicting ŷ = 0.01 gives huge loss.
- If y = 0 (not spam): only the right term activates → `-log(1 - ŷ)`. Predicting ŷ = 0.01 gives tiny loss. Predicting ŷ = 0.99 gives huge loss.

The **log** creates a sharp penalty as predictions approach the wrong extreme.

---

## Why Log and Not Squared Error?

```
If y = 1 and ŷ = 0.01 (very wrong):
  MSE loss  = (0.01 - 1)² = 0.9801
  BCE loss  = -log(0.01)  = 4.6052   ← much larger penalty!

If y = 1 and ŷ = 0.99 (very right):
  MSE loss  = (0.99 - 1)² = 0.0001
  BCE loss  = -log(0.99)  = 0.0101   ← both tiny, but BCE is slightly bigger
```

BCE punishes **confident wrong predictions** much more aggressively than MSE.

---

## Advantages

1. **Perfect for probability outputs** — works naturally with sigmoid (output in (0,1)).
2. **Convex** — single global minimum for logistic regression.
3. **Sharp gradients for confident wrong predictions** — fast learning when far off.
4. **Probabilistic interpretation** — derived from maximum likelihood estimation.

---

## Disadvantages

1. **Numerically unstable** if ŷ → 0 or ŷ → 1 (log of 0 = -∞). Need clipping.
2. **Only for binary problems** — use Categorical Cross-Entropy for multi-class.
3. **Sensitive to class imbalance** — if 99% of labels are 0, predicting all 0s gives low loss.

---

## When to Use BCE

✅ **Binary classification** — spam/not-spam, disease/healthy, cat/not-cat.
✅ **Multi-label classification** — each output neuron predicts independently (e.g., image has cat AND dog).
❌ **Multi-class (mutually exclusive)** — use Categorical Cross-Entropy.
❌ **Regression** — use MSE or MAE.

---

## Dry Run — Forward & Backward Pass

### Setup
```
3 samples, true labels:  y  = [1, 0, 1]
Model predictions (after sigmoid): ŷ = [0.9, 0.2, 0.4]
n = 3
```

### Step 1: Compute Per-Sample Loss

```
  Sample 1 (y=1): -[1·log(0.9) + 0·log(0.1)] = -log(0.9) = 0.1054
  Sample 2 (y=0): -[0·log(0.2) + 1·log(0.8)] = -log(0.8) = 0.2231
  Sample 3 (y=1): -[1·log(0.4) + 0·log(0.6)] = -log(0.4) = 0.9163
```

### Step 2: Average Loss
```
  L = (1/3) · (0.1054 + 0.2231 + 0.9163)
    = (1/3) · 1.2448
    = 0.4149
```

### Step 3: Compute Gradients
```
  dL/dŷ₁ = (1/3) · (-1/0.9  + 0/0.1)  = (1/3) · (-1.111) = -0.3704
  dL/dŷ₂ = (1/3) · (-0/0.2  + 1/0.8)  = (1/3) · (+1.250) = +0.4167
  dL/dŷ₃ = (1/3) · (-1/0.4  + 0/0.6)  = (1/3) · (-2.500) = -0.8333
```

**Key Insight:** Sample 3 (y=1, ŷ=0.4) gets the largest gradient magnitude (0.8333) because the model is most wrong and confident. Sample 1 (y=1, ŷ=0.9) gets the smallest (0.3704) because the model is nearly correct.

### What Happens as ŷ → 0 When y = 1?
```
  ŷ = 0.5:  loss = 0.693,  gradient = -0.667
  ŷ = 0.1:  loss = 2.303,  gradient = -3.333
  ŷ = 0.01: loss = 4.605,  gradient = -33.33   ← exploding gradient!
```
This is why we **clip** ŷ to (ε, 1-ε) like (1e-7, 1-1e-7) in practice.
