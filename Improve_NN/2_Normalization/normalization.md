# Normalization — Making Inputs and Layers Well-Behaved

## Why Normalize?

Neural networks learn by adjusting weights based on gradients. If input features live on **wildly different scales**, the loss surface becomes a stretched, elongated valley — and gradient descent zig-zags slowly down it instead of heading straight to the bottom.

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

**Analogy:** Imagine rolling a marble down a valley. A round bowl sends it straight to the center. A long, narrow canyon makes it bounce wall-to-wall, taking forever. Normalization turns the canyon into a bowl.

### Why scale mismatch causes this — Dry Run
```
Feature CGPA   ∈ [0, 10]      Feature Profile ∈ [0, 5]
A weight feeding from CGPA sees inputs ~2x bigger than one feeding from Profile.
→ Its gradient is ~2x bigger → that weight wants a big step while the other wants a small one.
→ One learning rate can't suit both → oscillation in the steep direction, crawling in the flat one.
```

---

## Type 1: Input Normalization (`input_normalization.py`)

Normalize the **raw input features** *before* they enter the network.

### Method A — Min-Max (scale to [0, 1])
```
x_norm = (x - x_min) / (x_max - x_min)
```

### Method B — Z-Score / Standardization (mean = 0, std = 1)
```
x_norm = (x - mean) / std
```

### Method C — Max-Abs (scale to [-1, 1])
```
x_norm = x / max(|x|)
```

### Dry Run — both on CGPA = [3, 5, 7, 8]
```
Min-Max:  min=3, max=8, range=5
  3 → (3-3)/5 = 0.00
  5 → (5-3)/5 = 0.40
  7 → (7-3)/5 = 0.80
  8 → (8-3)/5 = 1.00          Result: [0.00, 0.40, 0.80, 1.00]

Z-Score:  mean = 5.75
  variance = [(3-5.75)² + (5-5.75)² + (7-5.75)² + (8-5.75)²] / 4
           = [7.5625 + 0.5625 + 1.5625 + 5.0625] / 4 = 14.75/4 = 3.6875
  std = √3.6875 = 1.92
  3 → (3-5.75)/1.92 = -1.43
  5 → (5-5.75)/1.92 = -0.39
  7 → (7-5.75)/1.92 =  0.65
  8 → (8-5.75)/1.92 =  1.17     Result: [-1.43, -0.39, 0.65, 1.17]  (mean≈0, std≈1)
```

| Method | When to Use |
|--------|------------|
| Min-Max | You know fixed bounds (e.g., pixels 0–255) |
| Z-Score | **General purpose**, handles outliers better, most common |
| Max-Abs | Data already centered near 0 / sparse data |

### ⚠️ The Critical Rule: No Data Leakage
```
Compute mean/std (or min/max) from the TRAINING data ONLY.
Apply those SAME numbers to validation and test data.
```
**Why?** If you compute statistics using the test set, information about the test set "leaks" into training, and your reported accuracy becomes a lie. In production you won't have future data to normalize against — so simulate that honestly.

---

## Type 2: Batch Normalization (`batch_normalization.py`)

Normalize the **activations inside the network**, per mini-batch — so each layer receives well-behaved inputs even as earlier layers keep changing.

### The Formula
```
For a mini-batch of m pre-activation values z:

1. Batch mean:      μ      = (1/m) · Σ zᵢ
2. Batch variance:  σ²     = (1/m) · Σ (zᵢ - μ)²
3. Normalize:       ẑᵢ     = (zᵢ - μ) / √(σ² + ε)
4. Scale & shift:   yᵢ     = γ · ẑᵢ + β        ← γ, β are LEARNABLE
```

The learnable `γ` (scale) and `β` (shift) let the network *undo* the normalization if that turns out to be better — so it never loses representational power.

### Dry Run — mini-batch z = [2, 4, 6, 8]
```
μ  = (2+4+6+8)/4 = 5.0
σ² = [(2-5)² + (4-5)² + (6-5)² + (8-5)²]/4 = [9+1+1+9]/4 = 5.0
σ  = √(5.0 + ε) ≈ 2.236

ẑ = [(2-5), (4-5), (6-5), (8-5)] / 2.236
  = [-1.342, -0.447, 0.447, 1.342]      ← now mean 0, std 1

With γ = 2, β = 1:
y = 2·ẑ + 1 = [-1.684, 0.106, 1.894, 3.684]
```

### Internal Covariate Shift (the problem it solves)
As earlier layers update their weights, the **distribution of inputs** to later layers keeps shifting. Later layers are forever chasing a moving target. BatchNorm re-centers each layer's inputs every batch, so later layers see a stable distribution and can learn faster.

**Benefits:**
- ✅ Allows **higher learning rates** (more stable training)
- ✅ Acts as **mild regularization** (noise from per-batch statistics)
- ✅ **Smooths the loss landscape** → faster, more reliable convergence
- ✅ Reduces sensitivity to weight initialization

- **Where to apply:** between the linear step (`z = Wx + b`) and the activation.
- **At inference:** a single example has no "batch" to compute stats from, so use **running averages** of mean/variance collected during training.

---

## Quick Comparison of Normalization Layers

| Type | Normalizes across | Best for |
|------|-------------------|----------|
| **Batch Norm** | The batch dimension | CNNs, large batches |
| **Layer Norm** | All features of one example | Transformers, RNNs, small/variable batches |
| **Instance Norm** | Each channel of one example | Style transfer, image generation |
| **Group Norm** | Groups of channels | Small-batch vision tasks |

> Input normalization and BatchNorm are the two you'll use constantly. Layer Norm is worth knowing because it powers Transformers (batch size there is often tiny or variable, where BatchNorm struggles).

---

## Common Mistakes

1. **Leaking test statistics** — computing mean/std over the whole dataset including test. Train-only!
2. **Normalizing the target the wrong way** — if you scale `y`, remember to *un-scale* your predictions afterward.
3. **Using BatchNorm with batch size 1 or 2** — batch statistics become unreliable; use Layer Norm instead.
4. **Forgetting running averages at inference** — using live batch stats on test data gives inconsistent predictions.
5. **Putting BatchNorm after the activation when the convention is before** — be consistent; the common order is Linear → BatchNorm → Activation.

---

## TL;DR

- **Normalize inputs always** — it's free and almost always speeds up training.
- **Z-Score** is the safe default for tabular features; **Min-Max** for bounded data like pixels.
- **Compute stats on training data only** to avoid leakage.
- **Batch Normalization** stabilizes deep networks, enables higher learning rates, and adds light regularization.
- For Transformers/RNNs or tiny batches, reach for **Layer Norm** instead.
