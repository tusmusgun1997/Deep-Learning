# Categorical Cross-Entropy (CCE) — Deep Dive

## What It Does

Categorical Cross-Entropy measures how far off your predicted **probability distribution** is from the true class distribution. It's the standard loss for **multi-class classification** (one correct class per sample).

```
Formula:   L = -(1/n) · Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ)

For one-hot labels (one correct class per sample), this simplifies to:

  L = -(1/n) · Σᵢ log(ŷᵢ,correct_class)

Derivative w.r.t. softmax input (z): dL/dz = ŷ - y   (clean!)

Output Range: [0, ∞)   (0 = perfect)
```

**Intuition:** Only the predicted probability for the **correct class** matters. If the true class is "cat" and your model says P(cat) = 0.9, loss is low. If P(cat) = 0.1, loss is high.

---

## Connection to Softmax

CCE is always used with **Softmax** as the final activation:

```
Softmax output:  ŷⱼ = e^(zⱼ) / Σₖ e^(zₖ)

This ensures:
  - All ŷⱼ ∈ (0, 1)
  - Σⱼ ŷⱼ = 1  (valid probability distribution)
```

The combined Softmax + CCE has a beautifully simple gradient: `dL/dz = ŷ - y`.

---

## Advantages

1. **Perfect for multi-class problems** — handles any number of classes.
2. **Clean gradient** — Softmax + CCE gradient is just `ŷ - y`, numerically stable and simple.
3. **Probabilistic** — directly optimizes the probability of the correct class.
4. **Sharp penalties** — confidently wrong predictions get huge losses.

---

## Disadvantages

1. **Requires Softmax** — outputs must be a valid probability distribution summing to 1.
2. **Assumes mutually exclusive classes** — each sample belongs to exactly one class.
3. **Sensitive to class imbalance** — rare classes get fewer gradient signals.
4. **Numerically unstable** if log(ŷ) is computed naively (need log-sum-exp trick).

---

## When to Use CCE

✅ **Multi-class classification** — digit recognition (0-9), image classification (cat/dog/bird).
✅ When using **Softmax** output layer.
❌ **Multi-label** (image can have multiple correct labels) — use BCE per class instead.
❌ **Binary classification** — use BCE (simpler, same result).
❌ **Regression** — use MSE or MAE.

---

## Dry Run — Forward & Backward Pass

### Setup
```
3 classes: Cat, Dog, Bird
True labels (one-hot):  y  = [1, 0, 0]   (correct class is Cat)
Softmax outputs:        ŷ  = [0.7, 0.2, 0.1]
```

### Step 1: Compute Loss (only correct class matters!)
```
  L = -log(ŷ_cat) = -log(0.7) = 0.3567
```

Even though the model gives P(cat) = 0.7, the loss is non-zero because it's not 1.0.

### Step 2: What if the model was more wrong?
```
  ŷ = [0.3, 0.5, 0.2]   → L = -log(0.3) = 1.2040   ← much higher!
  ŷ = [0.9, 0.05, 0.05] → L = -log(0.9) = 0.1054   ← much lower!
```

### Step 3: Gradient (Softmax + CCE combined)
```
  True labels (one-hot): y  = [1, 0, 0]
  Softmax output:        ŷ  = [0.7, 0.2, 0.1]

  Gradient w.r.t. softmax input z:
    dL/dz₁ = ŷ₁ - y₁ = 0.7 - 1 = -0.3   ← push up (too low)
    dL/dz₂ = ŷ₂ - y₂ = 0.2 - 0 = +0.2   ← push down (too high)
    dL/dz₃ = ŷ₃ - y₃ = 0.1 - 0 = +0.1   ← push down (too high)
```

**Key Insight:** The gradient is just `ŷ - y`. No complex chain rule needed — this clean formula is why Softmax + CCE is the standard combo. The model will increase z₁ (making P(cat) higher) and decrease z₂, z₃.

### Batch Example (n=3 samples)
```
Labels:     y  = [[1,0,0], [0,1,0], [0,0,1]]
Softmax:    ŷ  = [[0.7,0.2,0.1], [0.1,0.8,0.1], [0.2,0.3,0.5]]

Per-sample losses:
  Sample 1 (Cat):  -log(0.7) = 0.3567
  Sample 2 (Dog):  -log(0.8) = 0.2231
  Sample 3 (Bird): -log(0.5) = 0.6931

Average Loss: (0.3567 + 0.2231 + 0.6931) / 3 = 0.4243
```
