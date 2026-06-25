# Overfitting — The #1 Problem in Deep Learning

## What is Overfitting?

The model **memorizes** the training data (including its noise) instead of learning the **general patterns**. It performs amazingly on training data but fails on new, unseen data.

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

**Analogy:** A student who memorizes past exam papers word-for-word gets 100% on those exact papers but fails when questions are rephrased. A student who *understands concepts* scores well on any variation.

---

## Signs of Overfitting

| Metric | Healthy | Overfitting |
|--------|---------|-------------|
| Train Loss | Decreasing | Decreasing |
| Val Loss | Decreasing (close to train) | **Increasing** (diverging from train) |
| Train Accuracy | Improving | Very high (99%+) |
| Val Accuracy | Improving (close to train) | **Stagnating or dropping** |
| Gap | Small | **Large** |

---

## 3 Solutions

### 1. Dropout (`dropout.py`)

Randomly disables neurons during training so the network can't rely on any single neuron. Forces redundancy and prevents co-adaptation.

- **What:** Zero out random neurons with probability `p` each forward pass
- **When to use:** Large networks, dense layers
- **Typical values:** p = 0.2 to 0.5

### 2. L1 & L2 Regularization (`regularization.py`)

Adds a penalty for large weights to the loss function, forcing the model to use smaller weights and learn simpler patterns.

- **L1 (Lasso):** Penalty = lambda * sum(|w|) — drives weights to exactly 0 (sparse)
- **L2 (Ridge):** Penalty = lambda * sum(w^2) — drives weights close to 0 (small)
- **When to use:** Always a good default. L2 is more common.
- **Typical lambda:** 0.0001 to 0.01

### 3. Early Stopping (`early_stopping.py`)

Monitor validation loss during training. When it stops improving for `patience` epochs, stop training and restore the best weights.

- **What:** Stop training before the model starts overfitting
- **When to use:** Always — it's free regularization
- **Typical patience:** 5 to 20 epochs
