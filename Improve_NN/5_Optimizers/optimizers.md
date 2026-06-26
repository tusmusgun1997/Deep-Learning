# Optimizers — How to Update Weights Smartly

## The Goal

All optimizers do the same thing: **update weights to minimize loss**. They differ in HOW they do it.

```
Vanilla SGD:  w = w - lr * gradient        (blind, same speed everywhere)
Smart:        w = w - lr * f(gradient)      (adapts based on history)
```

## Evolution

```
SGD (slow, oscillates)
 |
 +-- Momentum (adds velocity, smoother)
 |
 +-- Adagrad (adapts LR per param, but LR decays to 0)
      |
      +-- RMSprop (fixes Adagrad's decay with moving average)
           |
           +-- Adam = Momentum + RMSprop (best of both!)
```

## Comparison

| Optimizer | Per-param LR? | Momentum? | Key Formula |
|-----------|--------------|-----------|-------------|
| SGD+Momentum | No | Yes | `v = beta*v + grad; w -= lr*v` |
| Adagrad | Yes | No | `s += grad^2; w -= lr*grad/sqrt(s)` |
| RMSprop | Yes | No | `s = beta*s + (1-beta)*grad^2; w -= lr*grad/sqrt(s)` |
| **Adam** | **Yes** | **Yes** | `m,v + bias correction` |

## Which to Use?

- **Default:** Adam (works well on almost everything)
- **Computer Vision:** SGD+Momentum (often generalizes better with proper tuning)
- **Sparse data/NLP:** Adam or Adagrad
- **RNNs:** Adam or RMSprop
