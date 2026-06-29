# Gradient Checking & Clipping

Two unrelated-but-handy gradient tools: one **verifies** your backprop math is correct (checking), the other **stabilizes** training when gradients blow up (clipping).

---

## 4.1 Gradient Checking — Debugging Backpropagation

**What:** A way to verify your hand-written backprop is correct by comparing its **analytical** gradients against **numerically estimated** ones. If they match, your math is right.

**Why you need it:** Backprop is easy to get subtly wrong (a missing term, a transpose, an off-by-one). The loss may still go down "well enough," hiding the bug. Gradient checking catches it definitively.

### The Idea — Finite Differences
A derivative is just *"how much does the loss change when I nudge this parameter?"* So nudge it and measure:

```
                     L(θ + ε) - L(θ - ε)
numerical_grad  =  -----------------------          ← two-sided (centered) difference
                          2 · ε

Where ε is tiny (e.g., 1e-7)
```

**Two-sided vs one-sided:** the centered version `[L(θ+ε) − L(θ−ε)] / 2ε` has error ~O(ε²), far more accurate than the one-sided `[L(θ+ε) − L(θ)] / ε` (error ~O(ε)). Always use two-sided.

### The Comparison Metric
```
                    |analytical_grad - numerical_grad|
relative_error  =  ------------------------------------------
                    |analytical_grad| + |numerical_grad| + ε

Error < 1e-7  →  Backprop is CORRECT ✓
Error < 1e-5  →  Probably fine
Error < 1e-3  →  Suspicious — check carefully
Error > 1e-3  →  There's a BUG in your backprop ✗
```
(We use *relative* error so the threshold works whether gradients are huge or tiny.)

### Dry Run — check `f(w) = w²` at `w = 3`
```
True derivative:      f'(w) = 2w = 6.0   (this is what backprop should produce)

Numerical estimate with ε = 1e-4:
  L(3 + 0.0001) = (3.0001)² = 9.00060001
  L(3 - 0.0001) = (2.9999)² = 8.99940001
  numerical = (9.00060001 - 8.99940001) / (2 · 0.0001)
            = 0.0012 / 0.0002
            = 6.000                       ✓ matches analytical 6.0

relative_error = |6.0 - 6.0| / (6.0 + 6.0) ≈ 0   →  backprop CORRECT
```

### Practical Rules
- ⚠️ **Debugging only** — it's painfully slow (2 forward passes *per parameter*). Never leave it on during real training.
- Turn **off dropout and BatchNorm** while checking (their randomness breaks the comparison).
- Use **float64** (double precision) — float32 rounding pollutes the tiny differences.
- Check just a **handful** of parameters, not all of them, for big networks.
- Pick ε around **1e-7** — too big = approximation error; too small = floating-point round-off.

---

## 4.2 Gradient Clipping — Taming Exploding Gradients

**What:** When the gradient magnitude exceeds a threshold, **shrink it** before the weight update — preventing one giant step from wrecking the model.

**Why:** In deep nets and especially **RNNs/LSTMs**, gradients can explode (see [vanishing & exploding gradients](../3_Vanishing_Gradients/vanishing_gradients.md)). A single huge update can throw weights into a bad region and produce **NaN** losses. Clipping puts a speed limit on updates.

### Method A — Clip by Value
Clamp each gradient element independently into `[-max_val, max_val]`:
```
g = max(-max_val, min(max_val, g))
```
Simple, but it can **change the gradient's direction** (different components get clipped by different amounts).

### Method B — Clip by Norm (preferred)
If the whole gradient vector's L2 norm exceeds a threshold, **rescale the entire vector** to that norm:
```
if ||g|| > threshold:
    g = g · (threshold / ||g||)
```
This shrinks the magnitude but **preserves the direction** — you still step the "right way," just a smaller distance.

### Dry Run — gradient g = [3, 4], threshold = 1.0
```
||g|| = √(3² + 4²) = √25 = 5.0     (exceeds threshold 1.0)

Clip by NORM:
  scale = 1.0 / 5.0 = 0.2
  g = [3·0.2, 4·0.2] = [0.6, 0.8]
  new ||g|| = √(0.36 + 0.64) = 1.0  ✓   direction 3:4 preserved ✓

Clip by VALUE (max_val = 1.0):
  g = [min(1,3), min(1,4)] = [1, 1]
  new ||g|| = √2 = 1.414            direction changed from 3:4 → 1:1 ✗
```

The contrast shows why **clip-by-norm is usually preferred**: same magnitude limiting, but the update keeps pointing downhill in the correct direction.

### When & How Much
- **Essential for:** RNNs, LSTMs, GRUs, Transformers, very deep networks.
- **Typical threshold:** 1.0 to 5.0.
- **How to tune:** watch your gradient norms during training; set the threshold a bit above the *typical* norm so it only catches the rare spikes.

---

## How They Relate

| | Gradient Checking | Gradient Clipping |
|--|-------------------|-------------------|
| Purpose | Verify backprop correctness | Stabilize training |
| When | Once, while debugging | Every step, during training |
| Speed | Very slow | Negligible cost |
| Fixes | Math bugs | Exploding gradients |

---

## TL;DR

- **Gradient Checking** = a debugging tool: compare backprop gradients to two-sided finite differences; relative error < 1e-7 means you're correct. Slow — use it once, then turn it off.
- **Gradient Clipping** = a training safeguard: cap gradient magnitude (prefer **clip-by-norm** to preserve direction) to stop exploding gradients and NaN losses. Default-on for RNNs.
