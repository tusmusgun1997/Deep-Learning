# Vanishing Gradients — Why Deep Networks Stop Learning

## The Problem

During backpropagation, gradients are **multiplied** through each layer via the chain rule:

```
dL/dW1 = dL/dy * dy/dz_n * dz_n/da_(n-1) * ... * da_1/dz_1 * dz_1/dW1
```

If each of those intermediate derivatives is **less than 1** (e.g., sigmoid derivative max = 0.25):

```
Layer 10: gradient = original * 0.25^10 = original * 0.00000095

The gradient VANISHES! Early layers stop learning.
```

If they're **greater than 1** (large weights):

```
Layer 10: gradient = original * 2^10 = original * 1024

The gradient EXPLODES! Training crashes (NaN losses).
```

---

## 2 Solutions

### 1. Activation Functions (`activation_functions.py`)

Replace sigmoid with activations whose derivatives don't squash below 1:

| Activation | Max Derivative | Vanishing? |
|-----------|---------------|------------|
| Sigmoid | **0.25** | YES (severe) |
| Tanh | **1.0** | Somewhat |
| ReLU | **1.0** (for z>0) | NO |
| Leaky ReLU | **1.0** (for z>0) | NO |

### 2. Weight Initialization (`weight_initialization.py`)

Initialize weights so that the variance of activations stays **roughly constant** across layers:

| Method | Formula | Best With |
|--------|---------|-----------|
| Xavier | `std = sqrt(2 / (fan_in + fan_out))` | Sigmoid, Tanh |
| He | `std = sqrt(2 / fan_in)` | ReLU, Leaky ReLU |
