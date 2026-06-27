# Optimizers — How to Update Weights Smartly

## The Goal

Every optimizer does the same job: **update weights to minimize loss**. They differ only in *how cleverly* they use the gradient.

```
Vanilla SGD:  w = w - lr · gradient          (blind, same step everywhere)
Smart:        w = w - lr · f(gradient, history)   (adapts using past gradients)
```

**Analogy:** SGD is hiking downhill staring only at your feet. Momentum is a ball rolling downhill — it builds speed and powers through flat spots. Adam is a ball that *also* knows to slow down on steep, bumpy terrain and speed up on smooth slopes.

---

## The Evolution

```
SGD (slow, oscillates in ravines)
 │
 ├── + Momentum  (adds velocity → smoother, faster)
 │
 └── Adagrad  (per-parameter LR, but LR decays to 0)
      │
      └── RMSprop  (fixes the decay with a moving average)
           │
           └── Adam = Momentum + RMSprop  (the all-rounder default)
```

---

## SGD — the baseline

```
w = w - lr · gradient
```
**Problem:** in a **ravine** (a valley that's steep in one direction, gentle in another) plain SGD bounces wall-to-wall across the steep direction while barely progressing along the gentle one.

```
Without help:   /\/\/\/\___/\___   (oscillates, slow)
```

---

## 5.1 Momentum — add velocity

Accumulate a running "velocity" from past gradients, like a ball gaining speed downhill.

```
v_t = β · v_(t-1) + (1 - β) · gradient_t
w   = w - lr · v_t
                              (typical β = 0.9)
```

### Dry Run — consistent direction (grad = 1.0 every step)
```
v₁ = 0.9·0    + 0.1·1 = 0.100
v₂ = 0.9·0.10 + 0.1·1 = 0.190
v₃ = 0.9·0.19 + 0.1·1 = 0.271
...  velocity builds up toward 1.0  → the optimizer ACCELERATES
```

### Dry Run — oscillating direction (grad = +1, −1, +1, −1 …)
```
v₁ = 0.1·(+1)               =  0.100
v₂ = 0.9·0.10 + 0.1·(−1)    = -0.010
v₃ = 0.9·(−0.01)+ 0.1·(+1)  =  0.091
v₄ = 0.9·0.091 + 0.1·(−1)   = -0.018
...  velocity stays tiny  → oscillations get DAMPED
```

**That's the magic:** momentum *accelerates* in consistent directions (down the valley) and *cancels out* contradictory ones (across the valley).
```
With Momentum:   ────────\_________   (smooth, fast)
```

> **Nesterov Momentum** is a refinement: it "looks ahead" by computing the gradient at the position momentum is about to carry you to, giving a more responsive, slightly better update.

---

## 5.2 Adagrad — per-parameter learning rates

**Idea:** parameters that already received big updates should slow down; rarely-updated ones should keep large steps. Track each parameter's accumulated squared gradient and divide by its square root.

```
s_t = s_(t-1) + gradient_t²
w   = w - lr · gradient_t / (√s_t + ε)
```

### Dry Run — grad = 1.0 every step
```
step 1:  s=1    update = 0.1·1/√1   = 0.1000
step 2:  s=2    update = 0.1·1/√2   = 0.0707
step 3:  s=3    update = 0.1·1/√3   = 0.0577
step 10: s=10   update = 0.1·1/√10  = 0.0316
step 100:s=100  update = 0.1·1/√100 = 0.0100   ← shrinking toward 0
```

- ✅ Great for **sparse data** (NLP, recommendations) — rare features keep large updates.
- ❌ **Fatal flaw:** `s` only ever grows, so the learning rate **decays monotonically to zero** and training stalls. RMSprop was invented to fix exactly this.

---

## 5.3 RMSprop — fix the decay with a moving average

Replace the ever-growing sum with an **exponential moving average** of squared gradients — it "forgets" old history, so the learning rate stops collapsing.

```
s_t = β · s_(t-1) + (1 - β) · gradient_t²
w   = w - lr · gradient_t / (√s_t + ε)
                              (typical β = 0.9)
```

### Dry Run — grad = 1.0 every step
```
step 1:  s = 0.1·1 = 0.100   update = 0.1/√0.10 = 0.316
step 2:  s = 0.19          update = 0.1/√0.19 = 0.229
steady:  s → 1.0           update → 0.1/√1.0  = 0.100   ← STABILIZES (no collapse)
```

Versus Adagrad's slow death to 0, RMSprop settles at a healthy steady rate. Excellent for **RNNs** and non-stationary problems.

---

## 5.4 Adam — Momentum + RMSprop (the default)

Combine the **1st moment** (mean of gradients → momentum) and the **2nd moment** (mean of squared gradients → RMSprop), with a **bias correction** to fix the cold start.

```
m_t = β₁·m_(t-1) + (1-β₁)·gradient        # 1st moment  (momentum)
v_t = β₂·v_(t-1) + (1-β₂)·gradient²       # 2nd moment  (RMSprop)

m̂_t = m_t / (1 - β₁ᵗ)                      # bias correction
v̂_t = v_t / (1 - β₂ᵗ)                      # bias correction

w = w - lr · m̂_t / (√v̂_t + ε)
```
**Defaults:** `β₁=0.9, β₂=0.999, ε=1e-8, lr=0.001`.

### Why bias correction matters — Dry Run (step 1, grad = 1.0)
```
m₁ = 0.9·0 + 0.1·1   = 0.100
v₁ = 0.999·0 + 0.001·1 = 0.001

WITHOUT correction:  update = lr · 0.100/√0.001 = lr · 3.16   ← way too big & wrong!

WITH correction (t=1):
  m̂ = 0.100 / (1 - 0.9¹)   = 0.100/0.1   = 1.0
  v̂ = 0.001 / (1 - 0.999¹) = 0.001/0.001 = 1.0
  update = lr · 1.0/√1.0 = lr · 1.0       ← correct, well-scaled ✓
```
At step 1, `m` and `v` start at 0, so they're biased toward 0. Dividing by `(1 - βᵗ)` "inflates" them back to the right size. As `t` grows, `βᵗ → 0` and the correction fades away.

---

## Comparison

| Optimizer | Per-param LR? | Momentum? | Key Formula | Best For |
|-----------|:--:|:--:|-------------|----------|
| **SGD** | No | No | `w -= lr·g` | Simple problems, baselines |
| **Momentum** | No | Yes | `v=βv+(1−β)g; w-=lr·v` | General training speedup |
| **Adagrad** | Yes | No | `s+=g²; w-=lr·g/√s` | Sparse data (NLP) |
| **RMSprop** | Yes | No | `s=βs+(1−β)g²; w-=lr·g/√s` | RNNs, non-stationary |
| **Adam** | **Yes** | **Yes** | momentum + RMSprop + bias-corr | **Default for almost everything** |

---

## Which to Use?

- **Default:** **Adam** — works well out-of-the-box on almost anything.
- **Computer Vision (CNNs):** **SGD + Momentum** often *generalizes better* with careful LR tuning + a schedule (many SOTA image models use it).
- **Sparse data / NLP:** Adam or Adagrad.
- **RNNs / LSTMs:** Adam or RMSprop.

---

## Common Mistakes

1. **Adam learning rate too high.** Adam's default `0.001` is very different from SGD's typical `0.01–0.1`. Don't copy an SGD LR into Adam.
2. **Expecting Adam to always beat SGD.** On vision tasks, well-tuned SGD+Momentum frequently generalizes better.
3. **No learning-rate schedule.** Even Adam benefits from decaying the LR late in training (see [LR scheduling](../6_Learning_Rate_Scheduling/lr_scheduling.md)).
4. **Forgetting ε.** The `+ε` in the denominator prevents divide-by-zero — never drop it.

---

## TL;DR

- All optimizers minimize loss; they differ in how they use gradient history.
- **Momentum** smooths and accelerates. **Adagrad/RMSprop** adapt the LR per parameter. **Adam** fuses both + bias correction.
- **Start with Adam.** Switch to **SGD+Momentum (with a schedule)** when you need the last bit of generalization, especially in vision.
