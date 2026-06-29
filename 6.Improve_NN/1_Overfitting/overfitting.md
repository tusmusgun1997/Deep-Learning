# Overfitting — The #1 Problem in Deep Learning

## What is Overfitting?

The model **memorizes** the training data (including its random noise) instead of learning the **general patterns** that transfer to new data. It performs amazingly on training data but fails on new, unseen data.

```
                    Loss
                    ^
                    |    Training Loss
                    |      \
                    |       \      Validation Loss
                    |        \       /
                    |         \     /
                    |          \   /    <-- Sweet Spot (stop here!)
                    |           \_/
                    |            \
                    |             \_____ Training Loss keeps dropping
                    |
                    |                   Validation Loss goes UP!
                    +-----------------------------------------> Epochs
                              OVERFITTING ZONE -->
```

**Analogy:** A student who memorizes past exam papers word-for-word gets 100% on those exact papers but fails when the questions are rephrased. A student who *understands the concepts* scores well on any variation. Overfitting = memorizing. Generalization = understanding.

---

## Underfitting vs Good Fit vs Overfitting

Every model lives somewhere on this spectrum:

```
  UNDERFITTING            GOOD FIT              OVERFITTING
  (too simple)          (just right)           (too complex)

      o   o                o   o                  o   o
   ___________          _/‾‾‾\_/‾\            /\  o  /\ o
  /  o     o            o      o              o \/ \/ o\
 o    o   o            o   o                  o  (wiggles through
                                              every single point)

 High Bias            Balanced              High Variance
 Misses the pattern   Captures the trend    Captures the noise
 Bad on train+test    Good on train+test    Great on train, bad on test
```

| | Train Error | Test Error | Diagnosis |
|--|------------|-----------|-----------|
| **Underfitting** | High | High | Model too simple (high bias) |
| **Good fit** | Low | Low | Just right |
| **Overfitting** | Very low | High | Model too complex (high variance) |

---

## The Bias–Variance Tradeoff

This is the theory *behind* overfitting. Total error splits into three parts:

```
Total Error  =  Bias²  +  Variance  +  Irreducible Noise
```

- **Bias** = error from wrong assumptions (model too simple). High bias → underfitting.
- **Variance** = error from being too sensitive to the specific training set. High variance → overfitting.
- **Irreducible noise** = randomness in the data you can never remove.

```
Error
  ^
  |  \                                    /  Total Error
  |   \                                  /   (U-shaped)
  |    \            sweet spot          /
  |     \              |               /
  |      \  Variance   v             /
  |       \___        _____________/  <- Variance (rises with complexity)
  |           \      /
  |            \    /
  |   Bias      \  /
  |   ___________\/______________________
  +------------------------------------------> Model Complexity
```

**The goal:** find the complexity that minimizes *total* error — not too simple, not too complex.

---

## Signs of Overfitting

| Metric | Healthy | Overfitting |
|--------|---------|-------------|
| Train Loss | Decreasing | Decreasing |
| Val Loss | Decreasing (close to train) | **Increasing** (diverging from train) |
| Train Accuracy | Improving | Very high (99%+) |
| Val Accuracy | Improving (close to train) | **Stagnating or dropping** |
| Train–Val Gap | Small | **Large** |

**The #1 diagnostic:** a large and *growing* gap between training and validation performance.

---

## What Causes Overfitting?

1. **Model too complex** — too many layers/neurons for the amount of data.
2. **Too little data** — not enough examples to pin down the true pattern.
3. **Training too long** — the model runs out of real signal to learn and starts fitting noise.
4. **Noisy / mislabeled data** — the model dutifully memorizes the errors.
5. **No regularization** — nothing discourages over-complex solutions.

---

## Solution 1: Dropout (`dropout.py`)

Randomly **disable neurons** during training so the network can't rely on any single one. Forces redundancy and prevents "co-adaptation."

**How it works:** Each forward pass, every neuron is kept with probability `(1-p)` or zeroed with probability `p`.

### Dry Run — Inverted Dropout (p = 0.5)
```
Layer activations:  a       = [0.8, 0.3, 0.5, 0.9, 0.2]
Random keep mask:   mask    = [ 1 ,  0 ,  1 ,  0 ,  1 ]   (50% dropped this pass)
After masking:      a*mask  = [0.8, 0.0, 0.5, 0.0, 0.2]
Scale by 1/(1-p)=2: output  = [1.6, 0.0, 1.0, 0.0, 0.4]   <-- scale UP survivors
```

**Why scale up by `1/(1-p)`?** So the *expected* output stays the same as without dropout. Then at **inference** time you use all neurons with **no scaling** — the network behaves consistently.

```
Expected output of a neuron = (1-p) · (a / (1-p))  +  p · 0  =  a   ✓ unchanged
```

**Why it works:**
- **Prevents co-adaptation** — no neuron can depend on a specific partner always being present.
- **Ensemble effect** — each step trains a different random sub-network; the final model behaves like an average of `2ⁿ` networks.
- **Adds noise** — acts as regularization.

- **Where to apply:** between hidden layers (often the larger ones). **Never on the output layer.**
- **Typical values:** p = 0.2 to 0.5. Disable entirely at inference.

---

## Solution 2: L1 & L2 Regularization (`regularization.py`)

Add a **penalty for large weights** to the loss. This pushes the model toward smaller weights → simpler, smoother functions that generalize better.

```
L2 (Ridge):  Loss = Original_Loss + λ · Σ wᵢ²      → drives weights small
L1 (Lasso):  Loss = Original_Loss + λ · Σ |wᵢ|     → drives weights to exactly 0
```

### Why L2 is called "Weight Decay" — Dry Run
```
Setup: w = 0.50,  lr = 0.1,  λ = 0.01,  dL/dw = 0.20

L2 gradient = dL/dw + 2λw = 0.20 + 2(0.01)(0.50) = 0.20 + 0.010 = 0.210
w_new = 0.50 - 0.1 · 0.210 = 0.50 - 0.021 = 0.479

Compare WITHOUT regularization:
w_new = 0.50 - 0.1 · 0.20 = 0.480

The extra 0.001 shrinkage every step = weights continuously "decay" toward 0.
```

### Why L1 Creates Sparsity — Dry Run
```
L1 gradient adds: λ · sign(w) = 0.01 · (+1) = +0.01   (constant, regardless of how big w is)
Each step nudges w toward 0 by a FIXED amount → small weights hit exactly 0 and stay there.
```

| Property | L1 (Lasso) | L2 (Ridge) |
|----------|-----------|-----------|
| Penalty | Σ \|wᵢ\| | Σ wᵢ² |
| Effect | Weights → **exactly 0** (sparse) | Weights → **small** (but nonzero) |
| Feature selection | Yes (zeros out useless features) | No (keeps all, shrinks them) |
| Best for | Many irrelevant features | General-purpose smoothing |

- **When to use:** L2 is a great default. Use L1 when you suspect many features are useless.
- **Typical λ:** 0.0001 to 0.01.

---

## Solution 3: Early Stopping (`early_stopping.py`)

Monitor **validation loss** each epoch. When it stops improving for `patience` epochs, stop and **restore the best weights**.

### Dry Run
```
Epoch | Train Loss | Val Loss | Best? | Patience | Action
------|-----------|----------|-------|----------|------------------------
  10  |   2.000   |  1.900   |  ✓    |   0/5    | new best, save weights
  15  |   0.900   |  0.850   |  ✓    |   0/5    | new best, save weights
  18  |   0.500   |  0.860   |  ✗    |   1/5    | no improvement
  20  |   0.300   |  0.880   |  ✗    |   2/5    | val loss rising...
  23  |   0.150   |  0.950   |  ✗    |   5/5    | STOP! restore epoch-15 weights
```

**Why it works:** the moment validation loss starts rising is exactly where the model stops learning real patterns and starts memorizing noise. Stopping there freezes the best-generalizing version.

- **When to use:** **Always.** It's essentially free regularization.
- **Parameters:** `patience` (5–20 epochs), `min_delta` (smallest change that counts as improvement).

---

## Other Powerful Cures

| Cure | Idea |
|------|------|
| **Get more data** | The single most effective fix — more examples make noise harder to memorize. |
| **Data augmentation** | Cheaply "create" more data: flip/rotate/crop images, add noise, etc. |
| **Reduce model size** | Fewer layers/neurons = less capacity to memorize. |
| **Batch Normalization** | Adds noise via batch statistics → mild regularization (see [normalization](../2_Normalization/normalization.md)). |

---

## Common Mistakes

1. **Tuning on the test set.** Always use a separate **validation** set for decisions; touch the test set only once, at the very end.
2. **Adding dropout to the output layer.** It corrupts your final predictions.
3. **Forgetting to disable dropout at inference.** Predictions become random.
4. **Cranking λ too high.** Over-regularizing causes *underfitting* — the model becomes too simple.
5. **Judging overfitting by training loss alone.** The gap to validation is what matters.

---

## TL;DR

- Overfitting = memorizing noise instead of learning patterns → great on train, bad on test.
- Detect it via the **train–validation gap**.
- Fix it with **Dropout**, **L1/L2 regularization**, **Early Stopping**, **more data / augmentation**, or a **smaller model**.
- Stack several of these together for best results — and always validate on held-out data.
