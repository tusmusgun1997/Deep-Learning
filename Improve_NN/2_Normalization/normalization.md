# Normalization — Making Inputs and Layers Well-Behaved

## Why Normalize?

Neural networks learn by adjusting weights based on gradients. If input features are on **wildly different scales**, the loss surface becomes elongated (elliptical), causing gradient descent to oscillate and converge slowly.

```
Without Normalization:              With Normalization:

   Loss                               Loss
    |   /----\                          |    /--\
    |  /      \   Elongated!            |   /    \   Circular!
    | /   *    \  Oscillates            |  /  *   \  Direct path
    |/  /   \   \                       | /   |    \
    +--/-----\----> w1                  +/----+------> w1
       w2                                w2
```

---

## Two Types of Normalization

### 1. Input Normalization (`input_normalization.py`)

Normalizes the **raw input features** before they enter the network.

**Methods:**
- **Min-Max:** Scale to [0, 1] — `(x - min) / (max - min)`
- **Z-Score:** Scale to mean=0, std=1 — `(x - mean) / std`

**Critical rule:** Compute stats from TRAINING data only, apply same transform to test data.

### 2. Batch Normalization (`batch_normalization.py`)

Normalizes the **activations inside the network** within each mini-batch.

**Formula:**
```
z_norm = (z - batch_mean) / sqrt(batch_var + epsilon)
output = gamma * z_norm + beta    (gamma and beta are learnable)
```

**Benefits:**
- Prevents internal covariate shift
- Allows higher learning rates
- Acts as mild regularization
- Smooths the loss landscape
