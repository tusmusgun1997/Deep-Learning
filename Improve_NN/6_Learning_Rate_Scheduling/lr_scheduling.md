# Learning Rate Scheduling — Tuning Training Speed Dynamically

## Why Schedule the Learning Rate?

The learning rate (LR) is the single most important hyperparameter. A **fixed** LR forces one compromise for the whole run:

- **Too high** → the model overshoots and bounces around the minimum, never settling.
- **Too low** → training crawls and may get stuck in a poor spot.
- **Ideal** → start **high** to explore and cover ground fast, then gradually go **low** to settle precisely into the minimum.

```
High LR:       Bounces around the valley, never settles.
Low LR:        Takes forever to even reach the valley.
Scheduled LR:  Sprints in early, then tiptoes to the exact bottom.
```

**Analogy:** Parking a car. You drive fast down the street (high LR), slow down approaching the spot (decay), then inch back and forth to line up perfectly (tiny LR). Driving at full speed the whole way means smashing past the spot every time.

---

## 1. Step Decay

Drop the LR by a factor `γ` every `K` epochs.

```
α_t = α₀ · γ^⌊t / K⌋
```

### Dry Run — α₀ = 0.01, γ = 0.1, K = 30
```
Epoch  1–30:  α = 0.01
Epoch 31–60:  α = 0.01 · 0.1   = 0.001
Epoch 61–90:  α = 0.01 · 0.1²  = 0.0001
```
- ✅ Simple, predictable, easy to debug.
- ❌ The sudden cliffs can momentarily jolt training.
- **Use when:** you want a dependable baseline; common in classic CNN training.

---

## 2. Exponential Decay

Decay smoothly and continuously every epoch.

```
α_t = α₀ · e^(−k·t)     or equivalently     α_t = α₀ · γ^t   (γ < 1)
```

### Dry Run — α₀ = 0.01, γ = 0.95
```
Epoch  1:  0.01 · 0.95¹  = 0.0095
Epoch 10:  0.01 · 0.95¹⁰ = 0.0060
Epoch 50:  0.01 · 0.95⁵⁰ = 0.0008
```
- ✅ Smooth, no abrupt jumps; fast drop early, gentle tail.
- ❌ Can decay *too* fast if `γ` is mis-set, starving late training.
- **Use when:** you want continuous decay without manual step boundaries.

---

## 3. Cosine Annealing

Follow a cosine curve from a max LR down to a min LR over `T_max` epochs.

```
η_t = η_min + ½(η_max − η_min)·(1 + cos(π · T_cur / T_max))
```

### Behavior
```
LR
 |‾‾‾\___                        starts slow,
 |       ‾‾\__                   decays FAST in the middle,
 |           ‾‾\___             eases gently into the minimum
 |                 ‾‾‾‾‾‾----___
 +-------------------------------→ epoch
```
- ✅ Spends extra time at both high LR (explore) and low LR (fine-tune) — empirically excellent.
- ✅ A **modern default** for deep nets.
- **Warm Restarts (SGDR):** periodically jump the LR back up to `η_max` and anneal again — these restarts can kick the model out of poor minima and improve results.

---

## 4. Warm-up + Decay

Begin by **increasing** the LR linearly from ~0 to `α_max` over the first few hundred steps, *then* decay.

```
Phase 1 (warm-up):  LR ramps 0 → α_max over first N steps
Phase 2 (decay):    LR follows cosine / step / exponential down
```
- **Why:** at step 0 the weights are random; a full-size LR can cause a wild, destabilizing first update. Warm-up eases in gently.
- **Use when:** training **Transformers** (it's standard there) and large-batch training.

---

## 5. ReduceLROnPlateau

Adaptive, data-driven: watch the **validation loss**; if it hasn't improved for `patience` epochs, multiply the LR by a factor (e.g., 0.1).

```
if val_loss hasn't improved for `patience` epochs:
    LR = LR · factor
```
- ✅ Responds to the *actual* training dynamics instead of a fixed calendar.
- ✅ Pairs naturally with **Early Stopping** (see [overfitting](../1_Overfitting/overfitting.md)).
- **Use when:** you don't know the right decay schedule in advance — let the loss decide.

---

## Comparison

| Scheduler | Decay Style | Key Parameters | When to Use |
|-----------|-------------|----------------|-------------|
| **Step Decay** | Discrete steps | `step_size`, `γ` | Simple, debuggable baseline |
| **Exponential** | Smooth exponential | `decay_rate` | Continuous decay, no step boundaries |
| **Cosine Annealing** | Smooth cosine | `η_max`, `η_min`, `T_max` | Modern default; great convergence |
| **Warm-up + Decay** | Ramp up, then down | `warmup_steps`, decay rule | Transformers, large batches |
| **ReduceLROnPlateau** | Adaptive drops | `patience`, `factor` | When the best schedule is unknown |

---

## Practical Tips

1. **Tune the base LR first.** A schedule refines a good LR — it can't rescue a wildly wrong one.
2. **Decay late, not early.** Keep the LR high long enough to make real progress before annealing.
3. **Match schedule to optimizer.** SGD+Momentum leans heavily on a good schedule; Adam needs it less but still benefits.
4. **Plot your LR curve** before a long run — make sure it actually does what you intended.
5. **Warm-up for big models/batches** to avoid an unstable first few hundred steps.

---

## TL;DR

- A fixed LR is a compromise; **scheduling** gives you "fast early, precise late."
- **Step** and **Exponential** are simple; **Cosine Annealing** is the strong modern default; **Warm-up + Decay** is standard for Transformers; **ReduceLROnPlateau** adapts to the loss.
- Tune the **base LR first**, then layer a schedule on top.
