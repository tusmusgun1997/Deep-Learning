# Leaky ReLU Activation Function — Deep Dive

## What It Does

Leaky ReLU is an improved variant of the standard ReLU activation function. Instead of outputting exactly zero for negative inputs, it allows a small, non-zero slope (controlled by hyperparameter $\alpha$, typically set to $0.01$).

```
Formula:   f(z) = max(αz, z)   (where α is a small constant, e.g. 0.01)

Derivative: f'(z) = 1.0 if z > 0 else α

Output Range: (-∞, ∞)
```

**Shape:** A 45-degree diagonal line (`y = z`) for positive inputs, and a very flat diagonal line (`y = αz`) for negative inputs.
- Very negative inputs → output is slightly negative (scaled by $\alpha$)
- Zero input → output = 0
- Positive inputs → output = input

---

## Advantages

1. **Solves the "Dying ReLU" Problem**
   - Because the derivative for negative inputs is $\alpha$ (e.g. $0.01$) rather than 0.0, the gradients never completely die.
   - Neurons that output negative values can still receive small gradients and update their weights to recover.
2. **Fast to Compute**
   - No exponent calculations. Just a simple threshold: `z if z > 0 else alpha * z`.
3. **No Saturation**
   - Unlike Sigmoid/Tanh, it does not saturate in either the positive or negative directions (if $\alpha > 0$).

---

## Disadvantages

1. **New Hyperparameter $\alpha$**
   - Introduces another hyperparameter that needs to be tuned. (Though $0.01$ works well in most cases; sometimes a learnable parameter $\alpha$ is used, called **PReLU** or Parametric ReLU).
2. **Not zero-centered**
   - Like ReLU, the average activation is still biased slightly positive, though less so than standard ReLU because negative activations are allowed.

---

## When to Use Leaky ReLU

✅ **Whenever you face the Dying ReLU problem** (e.g., when a large fraction of your ReLU units become inactive and network performance plateaus).
✅ **A solid default** for deep networks where standard ReLU fails.
❌ **Output layer** — Similar to ReLU, it's not bounded or normalized, so it shouldn't be used as the final classification layer.

---

## Dry Run — Forward & Backward Pass Through a Simple MLP

### Network Setup
A tiny network: **2 inputs → 2 hidden neurons (leaky relu with α = 0.01) → 1 output (linear)**

```
Inputs:  x₁ = 1.0,  x₂ = 0.5
Target:  y = 1.0
α = 0.01

Weights (layer 1):
  w₁₁ = 0.4,  w₁₂ = 0.3   (neuron h₁)
  w₂₁ = 0.2,  w₂₂ = 0.5   (neuron h₂)
  b₁ = 0.1,  b₂ = 0.1

Weights (layer 2):
  v₁ = 0.6,  v₂ = 0.4
  b₃ = 0.1
```

### Step 1: Forward Pass — Hidden Layer

```
Neuron h₁:
  z₁ = w₁₁·x₁ + w₁₂·x₂ + b₁
     = 0.4·1.0 + 0.3·0.5 + 0.1 = 0.65
  Since z₁ > 0:
  a₁ = max(0.01·0.65, 0.65) = 0.65

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
     = 0.2·1.0 + 0.5·0.5 + 0.1 = 0.55
  Since z₂ > 0:
  a₂ = max(0.01·0.55, 0.55) = 0.55
```

*What if a neuron received negative input?* If $z_1 = -0.5$, then its activation would be:
$a_1 = 0.01 \cdot (-0.5) = -0.005$

### Step 2: Forward Pass — Output Layer (Linear)

```
  ŷ = v₁·a₁ + v₂·a₂ + b₃
    = 0.6·0.65 + 0.4·0.55 + 0.1
    = 0.39 + 0.22 + 0.1 = 0.71
```

### Step 3: Loss (MSE)

```
  L = (ŷ - y)² = (0.71 - 1.0)² = 0.0841
  dL/dŷ = 2·(ŷ - y) = 2·(-0.29) = -0.58
```

### Step 4: Backward Pass — Output Layer

```
  dL/dv₁ = dL/dŷ · a₁ = -0.58 · 0.65 = -0.3780
  dL/dv₂ = dL/dŷ · a₂ = -0.58 · 0.55 = -0.3190
  dL/db₃ = dL/dŷ     = -0.58
```

### Step 5: Backward Pass — Hidden Layer (Leaky ReLU gradients)

Since both $z_1 > 0$ and $z_2 > 0$, the derivatives are $1.0$:
```
  f'(z₁) = 1.0
  f'(z₂) = 1.0

Gradients reaching the pre-activation:
  dL/da₁ = dL/dŷ · v₁ = -0.58 · 0.6 = -0.3480
  dL/da₂ = dL/dŷ · v₂ = -0.58 · 0.4 = -0.2320

  dL/dz₁ = dL/da₁ · f'(z₁) = -0.3480 · 1.0 = -0.3480
  dL/dz₂ = dL/da₂ · f'(z₂) = -0.2320 · 1.0 = -0.2320
```

*But what if a neuron had a negative input (e.g., $z_1 = -0.5$)?*
```
  f'(z₁) = α = 0.01
  dL/dz₁ = dL/da₁ · f'(z₁) = -0.3480 · 0.01 = -0.00348
```
The gradient is small but **non-zero**. It allows the neuron's weights to update and potentially pull the neuron back to an active state.

### Step 6: Weight Updates (lr = 0.1)

```
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.3480·1.0) = 0.4348
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.3480·0.5) = 0.3174
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(-0.2320·1.0) = 0.2232
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(-0.2320·0.5) = 0.5116

  v₁  ← 0.6 - 0.1·(-0.3780) = 0.6378
  v₂  ← 0.4 - 0.1·(-0.3190) = 0.4319
```
