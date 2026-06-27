# Loss Functions — Overview

## What Is a Loss Function?

A loss function measures **how wrong your model is**. It takes the model's prediction and the true answer, and returns a single number — the **loss**. The goal of training is to minimize this number.

```
Prediction (ŷ)  +  True label (y)  →  Loss Function  →  Loss value (a number)
                                                              ↓
                                                    Backpropagation uses this
                                                    to update weights
```

---

## Why Does the Choice of Loss Function Matter?

Different problems need different loss functions:

| Problem Type           | Use This Loss          |
|------------------------|------------------------|
| Regression             | MSE or MAE or Huber    |
| Binary Classification  | Binary Cross-Entropy   |
| Multi-class            | Categorical Cross-Entropy |

Using the **wrong** loss function leads to poor learning or misleading training signals.

---

## Loss Functions Covered

| # | Loss Function             | File                          |
|---|---------------------------|-------------------------------|
| 1 | Mean Squared Error (MSE)  | `1_MSE/mse.md`               |
| 2 | Mean Absolute Error (MAE) | `2_MAE/mae.md`               |
| 3 | Binary Cross-Entropy      | `3_BinaryCrossEntropy/bce.md`|
| 4 | Categorical Cross-Entropy | `4_CategoricalCrossEntropy/cce.md` |
| 5 | Huber Loss                | `5_HuberLoss/huber.md`       |

---

## Key Intuition

- **MSE** punishes big mistakes hard (squares the error). Sensitive to outliers.
- **MAE** treats all mistakes equally (absolute value). More robust to outliers.
- **Huber** — the best of both: squared for small errors, linear for large errors.
- **Cross-Entropy** — for classification. Measures how far off your probability is from the truth.
