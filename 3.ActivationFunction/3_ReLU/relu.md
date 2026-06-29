# ReLU Activation Function — Deep Dive

## What It Does

ReLU (Rectified Linear Unit) is a simple threshold function. If the input is negative, it outputs zero. If it is positive, it passes the input directly through unchanged.

```
Formula:   f(z) = max(0, z)

Derivative: f'(z) = 1 if z > 0 else 0 (at z=0, typically defined as 0 or 0.5, here we use 0)

Output Range: [0, ∞)
```

**Shape:** A flat line at 0 for negative values, and a 45-degree diagonal line (`y = z`) for positive values.
- Very negative inputs → output = 0
- Zero input → output = 0
- Positive inputs → output = input

---

## Advantages

1. **Solves Vanishing Gradients**
   - For all positive inputs, the derivative is exactly **1.0**.
   - This means gradients are passed through layers back to the beginning of the network without any shrinkage.
2. **Computational Efficiency**
   - Extremely fast to calculate. No exponentials (`e^z`) or division. Just a comparison: `max(0, z)` (often compiled as a single instruction `max`).
3. **Sparsity**
   - Negative inputs output exactly 0.
   - This results in a sparse representation where only a subset of neurons are active (firing) at any given time, which is biologically realistic and computationally efficient.

---

## Disadvantages

1. **The "Dying ReLU" Problem**
   - If a neuron receives a large gradient that updates its weights such that it always outputs negative values, its input $z$ will always be negative.
   - Once $z < 0$, its output is 0 and its derivative is 0.
   - A derivative of 0 means the weights of that neuron will **never** be updated again. The neuron is effectively "dead" and contributes nothing to learning.
2. **Not zero-centered**
   - Since outputs are always $\ge 0$, the activations in the hidden layers are biased positive, which can lead to zig-zagging gradient descent steps (similar to Sigmoid).

---

## When to Use ReLU

✅ **Default choice for all hidden layers** in feedforward neural networks, CNNs, and Transformers.
❌ **RNNs / LSTMs** — ReLU can cause activations to grow out of control across time steps; Tanh is preferred.
❌ **Output layer** — Output is bounded on the left by 0 and unbounded on the right, which is generally not suitable for classification (use Sigmoid/Softmax) or regression where target values can be negative.

---

## Dry Run — Forward & Backward Pass Through a Simple MLP

### Network Setup
A tiny network: **2 inputs → 2 hidden neurons (relu) → 1 output (linear)**

```
Inputs:  x₁ = 1.0,  x₂ = 0.5
Target:  y = 1.0

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
     = 0.4·1.0 + 0.3·0.5 + 0.1
     = 0.4 + 0.15 + 0.1 = 0.65

  a₁ = max(0, 0.65) = 0.65

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
     = 0.2·1.0 + 0.5·0.5 + 0.1
     = 0.2 + 0.25 + 0.1 = 0.55

  a₂ = max(0, 0.55) = 0.55
```

### Step 2: Forward Pass — Output Layer (Linear)

```
  ŷ = v₁·a₁ + v₂·a₂ + b₃
    = 0.6·0.65 + 0.4·0.55 + 0.1
    = 0.39 + 0.22 + 0.1
    = 0.71
```

### Step 3: Loss (MSE)

```
  L = (ŷ - y)² = (0.71 - 1.0)² = (-0.29)² = 0.0841
  dL/dŷ = 2·(ŷ - y) = 2·(-0.29) = -0.58
```

### Step 4: Backward Pass — Output Layer

```
  dL/dv₁ = dL/dŷ · a₁ = -0.58 · 0.65 = -0.3780
  dL/dv₂ = dL/dŷ · a₂ = -0.58 · 0.55 = -0.3190
  dL/db₃ = dL/dŷ     = -0.58
```

### Step 5: Backward Pass — Hidden Layer (ReLU gradients)

```
ReLU derivatives (since both inputs z₁ and z₂ are positive):
  f'(z₁) = 1.0
  f'(z₂) = 1.0

Gradients reaching the pre-activation:
  dL/da₁ = dL/dŷ · v₁ = -0.58 · 0.6 = -0.3480
  dL/da₂ = dL/dŷ · v₂ = -0.58 · 0.4 = -0.2320

  dL/dz₁ = dL/da₁ · f'(z₁) = -0.3480 · 1.0 = -0.3480
  dL/dz₂ = dL/da₂ · f'(z₂) = -0.2320 · 1.0 = -0.2320
```

**Notice:** Because the derivative is exactly 1.0 for positive inputs, the gradients are passed backward completely unchanged ($dL/dz_1 = dL/da_1$). There is absolutely no vanishing gradient here!

### Step 6: Weight Updates (lr = 0.1)

```
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.3480·1.0) = 0.4348
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.3480·0.5) = 0.3174
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(-0.2320·1.0) = 0.2232
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(-0.2320·0.5) = 0.5116

  v₁  ← 0.6 - 0.1·(-0.3780) = 0.6378
  v₂  ← 0.4 - 0.1·(-0.3190) = 0.4319
```
