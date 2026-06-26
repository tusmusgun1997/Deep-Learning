# Learning Rate Scheduling — Tuning Training Speed dynamically

## Why Schedule the Learning Rate?

At the start of training, weights are randomly initialized and far from optimal. A **high learning rate** allows the model to make large updates and learn quickly.
Over time, as the model approaches a minimum, a high learning rate causes it to **oscillate** or overshoot the minimum. Reducing the learning rate allows the model to take smaller, finer steps to settle into a deep local minimum.

```
High LR: Bounces around the optimal valley, never settles.
Low LR: Takes forever to reach the valley.
Scheduled LR: Starts fast, then slows down to settle precisely.
```

---

## 3 Schedulers Covered

### 1. Step Decay
Reduces the learning rate by a factor $\gamma$ (e.g. 0.5) every $K$ epochs.
- **Formula:** 
  $$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / K \rfloor}$$
- **Behavior:** Step-wise drop. Good for simple models, but creates sudden drops.

### 2. Exponential Decay
Continuously decays the learning rate by an exponential factor.
- **Formula:**
  $$\alpha_t = \alpha_0 \cdot e^{-k \cdot t}$$
  or
  $$\alpha_t = \alpha_0 \cdot \gamma^t$$ (where $\gamma < 1$)
- **Behavior:** Smooth decay. Learning rate drops quickly at first, then flattens out.

### 3. Cosine Annealing
Decays the learning rate following a cosine wave from a maximum value $\eta_{max}$ down to a minimum value $\eta_{min}$ over $T_{max}$ epochs.
- **Formula:**
  $$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min}) \left(1 + \cos\left(\frac{T_{cur}}{T_{max}} \pi\right)\right)$$
- **Behavior:** Starts slowly decaying, decays rapidly in the middle, and flattens out slowly near the minimum. One of the most popular modern schedulers.

---

## Comparison Table

| Scheduler | Decay Style | Parameters | When to Use |
|-----------|-------------|------------|-------------|
| **Step Decay** | Discontinuous (Steps) | `step_size`, `gamma` | Standard baseline, easy to debug |
| **Exponential** | Continuous (Exponential) | `decay_rate` | Fast training early on |
| **Cosine Annealing** | Continuous (Cosine Wave) | `eta_max`, `eta_min`, `T_max` | Modern default, smooth convergence |
