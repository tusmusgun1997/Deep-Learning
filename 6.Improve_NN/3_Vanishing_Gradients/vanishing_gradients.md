# Vanishing & Exploding Gradients — Why Deep Networks Stop Learning

## The Core Problem

During backpropagation, the gradient for an early layer is a **long product** of terms, one per layer it passes through (chain rule):

```
dL/dW₁ = dL/dy · dy/dz_n · dz_n/da_(n-1) · ... · da₁/dz₁ · dz₁/dW₁
         └──────────────── a chain of many multiplications ────────────────┘
```

Multiplying many numbers together is dangerous:

- If each term is **< 1** → the product shrinks toward **0** → **VANISHING gradient**.
- If each term is **> 1** → the product blows up toward **∞** → **EXPLODING gradient**.

**Analogy:** Whisper a message down a line of 50 people. If everyone softens it slightly (×0.9), the last person hears silence (vanishing). If everyone exaggerates it (×1.5), it becomes an incoherent scream (exploding). You want everyone to pass it on at roughly the same volume (×1).

---

## Vanishing Gradient — Dry Run

Sigmoid's derivative maxes out at **0.25**. Suppose every layer multiplies the gradient by its best case, 0.25:

```
Start gradient:        1.0
After layer 1: ×0.25 = 0.25
After layer 2: ×0.25 = 0.0625
After layer 3: ×0.25 = 0.0156
After layer 4: ×0.25 = 0.0039
After layer 5: ×0.25 = 0.00098      ← already <0.001
...
After layer 10:       = 0.25¹⁰ = 0.00000095   ← essentially ZERO
```

**Consequence:** the early layers receive almost no gradient → their weights barely change → they never learn. The network is "deep" on paper but only the last few layers actually train.

---

## Exploding Gradient — Dry Run

Now suppose large weights make each term ≈ 2:

```
Start gradient:        1.0
After layer 10: ×2¹⁰ = 1024
After layer 20: ×2²⁰ = 1,048,576    ← huge updates
```

**Consequence:** weight updates become enormous, the loss oscillates wildly, and you see **NaN** (not-a-number) losses as values overflow. Training crashes. This is especially common in **RNNs**, where the same weights are reused across many time steps.

---

## How to Spot Which One You Have

| Symptom | Likely Cause |
|---------|-------------|
| Loss decreases painfully slowly or plateaus immediately | Vanishing gradients |
| Early-layer weights barely change across epochs | Vanishing gradients |
| Loss suddenly jumps to NaN / Inf | Exploding gradients |
| Loss oscillates violently | Exploding gradients |

---

## Solution 1: Better Activation Functions (`activation_functions.py`)

Replace sigmoid with activations whose derivative doesn't squash everything below 1.

| Activation | Formula | Max Derivative | Vanishing? |
|-----------|---------|----------------|-----------|
| **Sigmoid** | `1/(1+e⁻ᶻ)` | **0.25** | YES (severe) |
| **Tanh** | `(eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | **1.0** (only at z=0) | Better, still can vanish |
| **ReLU** | `max(0, z)` | **1.0** (for z>0) | **NO** — gradient is exactly 1 |
| **Leaky ReLU** | `z if z>0 else 0.01z` | **1.0** (for z>0) | NO, and avoids dead neurons |

### Why ReLU fixes it
For any positive input, ReLU's derivative is exactly **1**. Multiplying by 1 repeatedly keeps the gradient *intact* no matter how deep the network:
```
1.0 × 1 × 1 × 1 × ... × 1 = 1.0    ← gradient survives all layers ✓
```

### The "Dying ReLU" catch
If a neuron's input is always negative, ReLU outputs 0 and its gradient is 0 forever — the neuron is **dead** and never recovers. **Leaky ReLU** fixes this by allowing a small slope (0.01) for negative inputs, so a trickle of gradient always flows.

> See the [ActivationFunction](../../3.ActivationFunction/activation_functions.md) folder for the full deep-dive on each one.

---

## Solution 2: Smart Weight Initialization (`weight_initialization.py`)

The goal: choose initial weights so the **variance of activations stays roughly constant** layer to layer — gradients then neither shrink nor blow up.

| Method | Formula (std of weights) | Best With |
|--------|--------------------------|-----------|
| **Xavier / Glorot** | `√(2 / (fan_in + fan_out))` | Sigmoid, Tanh |
| **He** | `√(2 / fan_in)` | ReLU, Leaky ReLU |
| **LeCun** | `√(1 / fan_in)` | SELU |

(`fan_in` = number of inputs to the neuron, `fan_out` = number of outputs.)

### Why the √(2/fan_in) — Intuition + Dry Run
A neuron sums `fan_in` products `wᵢ·xᵢ`. Variances of independent terms add up:
```
Var(z) ≈ fan_in · Var(w) · Var(x)
```
To keep `Var(z) ≈ Var(x)` (signal neither grows nor shrinks), we need `fan_in · Var(w) = 1`, i.e. `Var(w) = 1/fan_in`. ReLU zeros out half the inputs, so He **doubles** it to compensate → `Var(w) = 2/fan_in`.

```
Example: fan_in = 100 neurons feeding a ReLU layer
He std = √(2/100) = √0.02 = 0.1414
→ initialize each weight ~ Normal(0, 0.1414)
→ activations keep a stable spread across all layers → gradients flow cleanly.
```

**What goes wrong without it:**
- Weights too **large** → activations grow each layer → exploding gradients.
- Weights too **small** → activations shrink each layer → vanishing gradients.

---

## Other Modern Cures (worth knowing)

| Technique | How it helps |
|-----------|--------------|
| **Batch Normalization** | Re-centers activations each batch → keeps gradients in a healthy range (see [normalization](../2_Normalization/normalization.md)) |
| **Residual / Skip connections** | Add a shortcut `output = F(x) + x` so gradients can flow *directly* backward, bypassing the squashing — this is what enabled 100+ layer ResNets |
| **Gradient Clipping** | Caps exploding gradients (see [gradient checking & clipping](../4_Gradient_Checking_and_Clipping/gradient_checking_and_clipping.md)) |
| **LSTM/GRU gates** | Designed to preserve gradients across long sequences in RNNs |

---

## Common Mistakes

1. **Using sigmoid/tanh in deep hidden layers.** Default to ReLU (or a variant) for hidden layers.
2. **Initializing all weights to the same constant (or zero).** Every neuron then learns identically — they never differentiate ("symmetry" problem). Always use random init.
3. **Ignoring exploding gradients in RNNs.** Add gradient clipping by default there.
4. **Manual `std = 0.01` for everything.** Use He/Xavier matched to your activation instead.

---

## TL;DR

- Deep backprop multiplies many terms → gradients **vanish** (terms <1) or **explode** (terms >1).
- **Vanishing** = early layers stop learning; **Exploding** = NaN losses and crashes.
- Fix vanishing with **ReLU-family activations** + **He/Xavier initialization**.
- Fix exploding with **gradient clipping** and proper initialization.
- For very deep nets, **BatchNorm** + **residual connections** are the modern workhorses.
