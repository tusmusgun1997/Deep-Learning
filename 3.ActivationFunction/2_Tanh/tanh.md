# Tanh Activation Function — Deep Dive

## What It Does

Tanh (Hyperbolic Tangent) squashes **any** input into a value between -1 and 1. It is a zero-centered, shifted, and scaled version of the Sigmoid function.

```
Formula:   tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Derivative: tanh'(z) = 1 - tanh²(z)

Output Range: (-1, 1)
```

**Shape:** An S-curve that crosses exactly through the origin `(0, 0)`.
- Very negative inputs → output ≈ -1
- Zero input → output = 0.0
- Very positive inputs → output ≈ 1

---

## Advantages

1. **Zero-centered outputs**
   - The output of the tanh function ranges from -1 to 1. This means the mean of the outputs of the neurons is close to 0.
   - This prevents the "zig-zag" parameter updates during gradient descent, allowing the network to converge faster than Sigmoid.
2. **Stronger gradients**
   - The maximum derivative is **1.0** (at z = 0), compared to Sigmoid's maximum of 0.25.
   - Gradients are larger, meaning they vanish less quickly than they do with Sigmoid.
3. **Smooth & differentiable** — continuous derivative everywhere.

---

## Disadvantages

1. **Vanishing Gradient (still exists)**
   - For larger values of $|z|$ (e.g., $|z| > 3$), the slope of the curve becomes very flat.
   - The derivative quickly approaches 0. In deep networks, gradients still vanish in early layers.
2. **Computationally expensive**
   - Uses floating-point exponents (`e^z` and `e^-z`), which are slow to compute compared to simple threshold functions like ReLU.

---

## When to Use Tanh

✅ **RNNs (Recurrent Neural Networks)** — Tanh is standard in LSTM and GRU hidden states because it helps regulate value magnitude without letting them grow infinitely.
✅ **Zero-centered data** — When you want negative inputs to lead to negative activations, and positive inputs to lead to positive activations.
❌ **Deep Feedforward Networks** — Use ReLU or its variants instead to prevent vanishing gradients.

---

## Dry Run — Forward & Backward Pass Through a Simple MLP

### Network Setup
A tiny network: **2 inputs → 2 hidden neurons (tanh) → 1 output (linear)**

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

  a₁ = tanh(0.65) = (e⁰·⁶⁵ - e⁻⁰·⁶⁵) / (e⁰·⁶⁵ + e⁻⁰·⁶⁵)
     = (1.9155 - 0.5220) / (1.9155 + 0.5220) = 1.3935 / 2.4376
     = 0.5717

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂
     = 0.2·1.0 + 0.5·0.5 + 0.1
     = 0.2 + 0.25 + 0.1 = 0.55

  a₂ = tanh(0.55) = (e⁰·⁵⁵ - e⁻⁰·⁵⁵) / (e⁰·⁵⁵ + e⁻⁰·⁵⁵)
     = (1.7333 - 0.5769) / (1.7333 + 0.5769) = 1.1563 / 2.3102
     = 0.5005
```

### Step 2: Forward Pass — Output Layer (Linear)

```
  ŷ = v₁·a₁ + v₂·a₂ + b₃
    = 0.6·0.5717 + 0.4·0.5005 + 0.1
    = 0.3430 + 0.2002 + 0.1
    = 0.6432
```

### Step 3: Loss (MSE)

```
  L = (ŷ - y)² = (0.6432 - 1.0)² = (-0.3568)² = 0.1273
  dL/dŷ = 2·(ŷ - y) = 2·(-0.3568) = -0.7136
```

### Step 4: Backward Pass — Output Layer

```
  dL/dv₁ = dL/dŷ · a₁ = -0.7136 · 0.5717 = -0.4080
  dL/dv₂ = dL/dŷ · a₂ = -0.7136 · 0.5005 = -0.3572
  dL/db₃ = dL/dŷ     = -0.7136
```

### Step 5: Backward Pass — Hidden Layer (Tanh gradients)

```
Tanh derivatives:
  tanh'(z₁) = 1 - a₁² = 1 - (0.5717)² = 1 - 0.3268 = 0.6732
  tanh'(z₂) = 1 - a₂² = 1 - (0.5005)² = 1 - 0.2505 = 0.7495

Gradients reaching the pre-activation:
  dL/da₁ = dL/dŷ · v₁ = -0.7136 · 0.6 = -0.4282
  dL/da₂ = dL/dŷ · v₂ = -0.7136 · 0.4 = -0.2854

  dL/dz₁ = dL/da₁ · tanh'(z₁) = -0.4282 · 0.6732 = -0.2883
  dL/dz₂ = dL/da₂ · tanh'(z₂) = -0.2854 · 0.7495 = -0.2139
```

**Notice:** Compare this to Sigmoid's $dL/dz_1 = -0.0682$. With Tanh, the gradient is much larger ($-0.2883$) because the derivative values are higher. This means Tanh converges faster and suffers less from vanishing gradients than Sigmoid does.

### Step 6: Weight Updates (lr = 0.1)

```
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.2883·1.0) = 0.4288
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.2883·0.5) = 0.3144
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(-0.2139·1.0) = 0.2214
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(-0.2139·0.5) = 0.5107

  v₁  ← 0.6 - 0.1·(-0.4080) = 0.6408
  v₂  ← 0.4 - 0.1·(-0.3572) = 0.4357
```
