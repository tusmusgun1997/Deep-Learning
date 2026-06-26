# Gradient Checking and Clipping

## 4.1 Gradient Checking — Debugging Backpropagation

**What:** A technique to verify your backprop math is correct by comparing your analytical gradients with numerically computed gradients.

**How:** For each parameter, nudge it by a tiny epsilon in both directions and measure the change in loss:

```
                    L(W + eps) - L(W - eps)
numerical_grad  =  -----------------------
                         2 * eps

Then compare:
                  |analytical - numerical|
relative_error = -------------------------
                  |analytical| + |numerical|

Error < 1e-5:  Backprop is CORRECT
Error < 1e-3:  Might be OK
Error > 1e-3:  BUG in backprop!
```

**Important:** Only use for debugging — it's extremely slow (2 forward passes per parameter).

---

## 4.2 Gradient Clipping — Preventing Explosions

**What:** Limits gradient magnitude to prevent exploding gradients from destabilizing training.

**Two methods:**

**Clip by Value:** Each gradient individually clamped to `[-max_val, max_val]`
```
g = max(-max_val, min(max_val, g))
```

**Clip by Norm:** Scale entire gradient vector if L2 norm exceeds threshold
```
if ||g|| > threshold:
    g = g * (threshold / ||g||)
```

Clip by norm is preferred because it **preserves gradient direction**.
