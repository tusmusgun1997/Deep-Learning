# Swish Activation Function — Deep Dive

## What It Does

Swish is a self-gated activation function discovered by Google Brain researchers in 2017 using automated search techniques. It has been shown to outperform ReLU on deep networks in many applications.

```
Formula:   f(z) = z · σ(βz) = z / (1 + e^(-βz))   (usually β = 1.0 is used)

Derivative: f'(z) = f(z) + σ(z)(1 - f(z))  (for β = 1.0)

Output Range: [-0.28, ∞)
```

**Shape:** It looks very similar to ReLU but has a smooth, continuous curve near $z = 0$, and it dips slightly below 0 into negative values (down to approximately $-0.278$) before going back up to asymptote at 0 for highly negative inputs. This property is called **non-monotonicity**.

---

## Advantages

1. **Outperforms ReLU**
   - Empirical studies show that Swish consistently yields 1-2% higher validation accuracy on large-scale datasets (like ImageNet) compared to ReLU.
2. **Smooth & Differentiable**
   - Swish is smooth at all points, including $z = 0$. This prevents sudden changes in loss during gradient descent and stabilizes training.
3. **Non-Monotonic**
   - The small negative dip allows small negative activations to pass through. This enhances the network's capacity to represent complex functions.
4. **No Dying Neurons**
   - Because the derivative doesn't drop abruptly to zero, neurons do not freeze ("die") as easily as standard ReLUs.

---

## Disadvantages

1. **Computationally Expensive**
   - Requires computing the Sigmoid function `1 / (1 + e^-z)` and multiplying it by the input, which is much slower than ReLU's simple comparison `max(0, z)`.
2. **Harder to Implement on Hardware**
   - Certain edge devices and GPUs have hardware-accelerated instructions for ReLU, but not for transcendental functions like Swish.

---

## When to Use Swish

✅ **Default choice for modern state-of-the-art vision models** (e.g., EfficientNet).
✅ **When training very deep networks** (e.g., over 40 layers) where ReLU's sharp non-linearity causes training instabilities.
❌ **Resource-constrained environments** where training speed or latency is a priority.

---

## Dry Run — Forward & Backward Pass Through a Simple MLP

### Network Setup
A tiny network: **2 inputs → 2 hidden neurons (swish with β = 1.0) → 1 output (linear)**

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
  z₁ = w₁₁·x₁ + w₁₂·x₂ + b₁ = 0.4·1.0 + 0.3·0.5 + 0.1 = 0.65
  σ(0.65) = 1 / (1 + e⁻⁰·⁶⁵) = 0.6570
  a₁ = z₁ · σ(z₁) = 0.65 · 0.6570 = 0.4271

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂ = 0.2·1.0 + 0.5·0.5 + 0.1 = 0.55
  σ(0.55) = 1 / (1 + e⁻⁰·⁵⁵) = 0.6341
  a₂ = z₂ · σ(z₂) = 0.55 · 0.6341 = 0.3488
```

### Step 2: Forward Pass — Output Layer (Linear)

```
  ŷ = v₁·a₁ + v₂·a₂ + b₃
    = 0.6·0.4271 + 0.4·0.3488 + 0.1
    = 0.2563 + 0.1395 + 0.1 = 0.4958
```

### Step 3: Loss (MSE)

```
  L = (ŷ - y)² = (0.4958 - 1.0)² = 0.2542
  dL/dŷ = 2·(ŷ - y) = 2·(-0.5042) = -1.0084
```

### Step 4: Backward Pass — Output Layer

```
  dL/dv₁ = dL/dŷ · a₁ = -1.0084 · 0.4271 = -0.4307
  dL/dv₂ = dL/dŷ · a₂ = -1.0084 · 0.3488 = -0.3517
  dL/db₃ = dL/dŷ     = -1.0084
```

### Step 5: Backward Pass — Hidden Layer (Swish gradients)

```
Swish derivatives:
  f'(z) = σ(z) · [1 + z · (1 - σ(z))]

At z₁ = 0.65:
  f'(z₁) = 0.6570 · [1 + 0.65 · (1 - 0.6570)] = 0.6570 · 1.2230 = 0.8035

At z₂ = 0.55:
  f'(z₂) = 0.6341 · [1 + 0.55 · (1 - 0.6341)] = 0.6341 · 1.2012 = 0.7617

Gradients on hidden pre-activations:
  dL/da₁ = dL/dŷ · v₁ = -1.0084 · 0.6 = -0.6050
  dL/da₂ = dL/dŷ · v₂ = -1.0084 · 0.4 = -0.4034

  dL/dz₁ = dL/da₁ · f'(z₁) = -0.6050 · 0.8035 = -0.4861
  dL/dz₂ = dL/da₂ · f'(z₂) = -0.4034 · 0.7617 = -0.3073
```

*What if a neuron received a negative input, like $z_1 = -1.0$?*
```
  σ(-1.0) = 0.2689
  a₁ = -1.0 · 0.2689 = -0.2689
  f'(-1.0) = 0.2689 · [1 + (-1.0) · (1 - 0.2689)] = 0.2689 · 0.2689 = 0.0723
```
Unlike ReLU, which has 0 gradient, Swish passes a small gradient of `0.0723` through, preventing dead neurons.

### Step 6: Weight Updates (lr = 0.1)

```
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.4861·1.0) = 0.4486
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.4861·0.5) = 0.3243
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(-0.3073·1.0) = 0.2307
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(-0.3073·0.5) = 0.5154

  v₁  ← 0.6 - 0.1·(-0.4307) = 0.6431
  v₂  ← 0.4 - 0.1·(-0.3517) = 0.4352
```
