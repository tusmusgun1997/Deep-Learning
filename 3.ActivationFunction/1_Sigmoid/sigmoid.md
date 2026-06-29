# Sigmoid Activation Function — Deep Dive

## What It Does

Sigmoid squashes **any** input into a value between 0 and 1. Think of it as a "confidence meter" — how confident is the neuron that it should fire?

```
Formula:   σ(z) = 1 / (1 + e⁻ᶻ)

Derivative: σ'(z) = σ(z) · (1 - σ(z))

Output Range: (0, 1)
```

**Shape:** An S-curve (hence "sigmoid" — S-shaped).
- Very negative inputs → output ≈ 0
- Zero input → output = 0.5
- Very positive inputs → output ≈ 1

---

## Advantages

1. **Smooth & differentiable** — backpropagation needs derivatives; sigmoid gives clean ones everywhere.
2. **Output is bounded (0, 1)** — perfect for probabilities. "There's a 0.87 chance this email is spam."
3. **Historically important** — enabled the backpropagation revolution in the 1980s.
4. **Intuitive** — output can be interpreted as a probability directly.

---

## Disadvantages

1. **Vanishing Gradient (THE killer problem)**
   - The maximum derivative is only **0.25** (at z = 0).
   - During backpropagation, gradients are **multiplied** through layers.
   - After just 5 layers: `0.25⁵ = 0.00098` → gradient is practically **zero**.
   - Early layers stop learning completely.

2. **Not zero-centered**
   - Output is always positive (0 to 1).
   - This means gradients for weights are always all-positive or all-negative.
   - Causes zig-zag updates during optimization (slower convergence).

3. **Computationally expensive**
   - Uses `e^(-z)` (exponential) — slower than a simple `max(0, z)` (ReLU).

4. **Saturation**
   - For |z| > 5, the output barely changes (flat region of the S-curve).
   - Neuron becomes "stuck" — essentially dead to learning.

---

## When to Use Sigmoid

✅ **Binary classification output layer** — you want a probability between 0 and 1.
❌ **Hidden layers** — almost never. Use ReLU or its variants instead.

---

## Dry Run — Forward & Backward Pass Through a Simple MLP

### Network Setup
A tiny network: **2 inputs → 2 hidden neurons (sigmoid) → 1 output (linear)**

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

  a₁ = σ(0.65) = 1 / (1 + e⁻⁰·⁶⁵)
     = 1 / (1 + 0.5220) = 1 / 1.5220
     = 0.6570

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
     = 0.2·1.0 + 0.5·0.5 + 0.1
     = 0.2 + 0.25 + 0.1 = 0.55

  a₂ = σ(0.55) = 1 / (1 + e⁻⁰·⁵⁵)
     = 1 / (1 + 0.5769) = 1 / 1.5769
     = 0.6341
```

### Step 2: Forward Pass — Output Layer (Linear)

```
  ŷ = v₁·a₁ + v₂·a₂ + b₃
   = 0.6·0.6570 + 0.4·0.6341 + 0.1
   = 0.3942 + 0.2536 + 0.1
   = 0.7479
```

### Step 3: Loss (MSE)

```
  L = (ŷ - y)² = (0.7479 - 1.0)² = (-0.2521)² = 0.0636
  dL/dŷ = 2·(ŷ - y) = 2·(-0.2521) = -0.5043
```

### Step 4: Backward Pass — Output Layer

```
  dL/dv₁ = dL/dŷ · a₁ = -0.5043 · 0.6570 = -0.3313
  dL/dv₂ = dL/dŷ · a₂ = -0.5043 · 0.6341 = -0.3198
  dL/db₃ = dL/dŷ     = -0.5043
```

### Step 5: Backward Pass — Hidden Layer (HERE is where sigmoid hurts!)

```
Sigmoid derivatives:
  σ'(z₁) = σ(0.65) · (1 - σ(0.65)) = 0.6570 · 0.3430 = 0.2254  ← less than 0.25!
  σ'(z₂) = σ(0.55) · (1 - σ(0.55)) = 0.6341 · 0.3659 = 0.2320  ← less than 0.25!

  dL/da₁ = dL/dŷ · v₁ = -0.5043 · 0.6 = -0.3026
  dL/da₂ = dL/dŷ · v₂ = -0.5043 · 0.4 = -0.2017

  dL/dz₁ = dL/da₁ · σ'(z₁) = -0.3026 · 0.2254 = -0.0682  ← SHRUNK by ~4.4x!
  dL/dz₂ = dL/da₂ · σ'(z₂) = -0.2017 · 0.2320 = -0.0468  ← SHRUNK by ~4.3x!
```

**Notice:** The gradient went from `-0.5043` at the output to `-0.0682` at the hidden layer — shrunk by ~7x in just **one layer**! In a 10-layer network, gradients would essentially vanish.

### Step 6: Weight Updates (lr = 0.1)

```
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.0682·1.0) = 0.4 + 0.0068 = 0.4068
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.0682·0.5) = 0.3 + 0.0034 = 0.3034
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(-0.0468·1.0) = 0.2 + 0.0047 = 0.2047
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(-0.0468·0.5) = 0.5 + 0.0023 = 0.5023

  v₁  ← 0.6 - 0.1·(-0.3313) = 0.6 + 0.0331 = 0.6331
  v₂  ← 0.4 - 0.1·(-0.3198) = 0.4 + 0.0320 = 0.4320
```

**Key takeaway:** The hidden-layer weight updates (`0.0068`, `0.0034`, etc.) are **tiny** compared to output-layer updates (`0.0331`, `0.0320`). This is the vanishing gradient in action — early layers barely learn.
