# Softmax Activation Function — Deep Dive

## What It Does

Softmax is a special activation function used almost exclusively in the **output layer** of a multi-class neural network. Instead of processing neurons independently, it takes a vector of raw scores (logits) and turns them into a probability distribution that sums to 1.

```
Formula:   aᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

Derivative: ∂aᵢ/∂zⱼ = aᵢ(1 - aᵢ) if i = j, else -aᵢ aⱼ

Output Range: (0, 1) and the sum of all outputs is exactly 1.0.
```

**What it means:** The output is a relative confidence score. If neuron 1 outputs `0.7` and neuron 2 outputs `0.3`, the model is 70% confident in class 1 and 30% confident in class 2.

---

## Advantages

1. **Creates a True Probability Distribution**
   - Outputs are bounded between 0 and 1, and sum to 1.
   - Ideal for multi-class classification where classes are mutually exclusive.
2. **Differentiable**
   - Smooth derivative means we can optimize classification networks using gradient descent.
3. **Works perfectly with Cross-Entropy Loss**
   - When combined with Cross-Entropy Loss, the derivative becomes extremely simple and stable: `dL/dzᵢ = aᵢ - yᵢ` (predicted probability minus target).

---

## Disadvantages

1. **Only for the Output Layer**
   - Cannot be used in hidden layers. It locks activations together globally across the layer, which destroys spatial feature properties.
2. **Computationally Expensive**
   - Requires computing exponents `e^z` for every class and then summing them up.
3. **Sensitive to Outliers**
   - A single huge logit can dominate the sum of exponentials, pushing other activations to near 0 and causing high overconfidence.

---

## When to Use Softmax

✅ **Multi-class classification output layer** (e.g., classifying an image as cat, dog, or horse).
❌ **Binary classification** — Use Sigmoid instead (it is a special 2-class case of Softmax and is more efficient).
❌ **Multi-label classification** — If an image can be *both* a cat and a dog, use Sigmoid on each output unit instead of Softmax.
❌ **Hidden layers** — Never.

---

## Dry Run — Forward & Backward Pass Through a Classification MLP

### Network Setup
A network predicting **2 mutually exclusive classes**: **2 inputs → 2 hidden units (relu) → 2 output logits (softmax)**

```
Inputs:  x₁ = 1.0,  x₂ = 0.5
Target:  y = [1.0, 0.0]  (representing class 1 is the true class)

Weights (layer 1 - Hidden ReLU):
  w₁₁ = 0.4,  w₁₂ = 0.3,  b₁ = 0.1   (neuron h₁)
  w₂₁ = 0.2,  w₂₂ = 0.5,  b₂ = 0.1   (neuron h₂)

Weights (layer 2 - Output Softmax):
  v₁₁ = 0.6,  v₁₂ = 0.4,  c₁ = 0.1   (logit u₁)
  v₂₁ = 0.3,  v₂₂ = 0.7,  c₂ = 0.1   (logit u₂)
```

### Step 1: Forward Pass — Hidden Layer (ReLU)

```
Neuron h₁:
  z₁ = w₁₁·x₁ + w₁₂·x₂ + b₁ = 0.4·1.0 + 0.3·0.5 + 0.1 = 0.65
  a₁ = max(0, 0.65) = 0.65

Neuron h₂:
  z₂ = w₂₁·x₁ + w₂₂·x₂ + b₂ = 0.2·1.0 + 0.5·0.5 + 0.1 = 0.55
  a₂ = max(0, 0.55) = 0.55
```

### Step 2: Forward Pass — Output Layer (Softmax)

Compute the logits (pre-activation outputs):
```
Logit u₁:
  u₁ = v₁₁·a₁ + v₁₂·a₂ + c₁ = 0.6·0.65 + 0.4·0.55 + 0.1 = 0.71

Logit u₂:
  u₂ = v₂₁·a₁ + v₂₂·a₂ + c₂ = 0.3·0.65 + 0.7·0.55 + 0.1 = 0.68
```

Compute the Softmax activation:
```
Exponents:
  e^(u₁) = e^(0.71) = 2.0340
  e^(u₂) = e^(0.68) = 1.9739

Sum = 2.0340 + 1.9739 = 4.0079

Probabilities:
  ŷ₁ = 2.0340 / 4.0079 = 0.5075
  ŷ₂ = 1.9739 / 4.0079 = 0.4925

Check sum: 0.5075 + 0.4925 = 1.0000
```

### Step 3: Loss (Categorical Cross-Entropy)

```
  L = - Σ yᵢ · ln(ŷᵢ) = - [1.0 · ln(0.5075) + 0.0 · ln(0.4925)]
    = - ln(0.5075) = 0.6783
```

### Step 4: Backward Pass — Output Layer (Softmax + Cross Entropy)

The combined gradient $dL/du_i$ simplifies elegantly to:
```
  dL/du₁ = ŷ₁ - y₁ = 0.5075 - 1.0 = -0.4925
  dL/du₂ = ŷ₂ - y₂ = 0.4925 - 0.0 = 0.4925

Output Weights Gradients:
  dL/dv₁₁ = dL/du₁ · a₁ = -0.4925 · 0.65 = -0.3201
  dL/dv₁₂ = dL/du₁ · a₂ = -0.4925 · 0.55 = -0.2709
  dL/dc₁  = dL/du₁     = -0.4925

  dL/dv₂₁ = dL/du₂ · a₁ = 0.4925 · 0.65 = 0.3201
  dL/dv₂₂ = dL/du₂ · a₂ = 0.4925 · 0.55 = 0.2709
  dL/dc₂  = dL/du₂     = 0.4925
```

### Step 5: Backward Pass — Hidden Layer (ReLU)

```
Gradients reaching the hidden layer activations:
  dL/da₁ = dL/du₁·v₁₁ + dL/du₂·v₂₁ = -0.4925·0.6 + 0.4925·0.3 = -0.1478
  dL/da₂ = dL/du₁·v₁₂ + dL/du₂·v₂₂ = -0.4925·0.4 + 0.4925·0.7 = 0.1478

ReLU derivatives:
  f'(z₁) = 1.0
  f'(z₂) = 1.0

Gradients on hidden pre-activations:
  dL/dz₁ = dL/da₁ · f'(z₁) = -0.1478 · 1.0 = -0.1478
  dL/dz₂ = dL/da₂ · f'(z₂) = 0.1478 · 1.0 = 0.1478
```

### Step 6: Weight Updates (lr = 0.1)

```
Hidden weights updates:
  w₁₁ ← 0.4 - 0.1·(dL/dz₁·x₁) = 0.4 - 0.1·(-0.1478·1.0) = 0.4148
  w₁₂ ← 0.3 - 0.1·(dL/dz₁·x₂) = 0.3 - 0.1·(-0.1478·0.5) = 0.3074
  w₂₁ ← 0.2 - 0.1·(dL/dz₂·x₁) = 0.2 - 0.1·(0.1478·1.0) = 0.1852
  w₂₂ ← 0.5 - 0.1·(dL/dz₂·x₂) = 0.5 - 0.1·(0.1478·0.5) = 0.4926
```
