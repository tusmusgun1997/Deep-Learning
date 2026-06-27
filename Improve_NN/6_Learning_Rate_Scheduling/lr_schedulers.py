# ============================================================
# LEARNING RATE SCHEDULING — From Scratch
# ============================================================
# Step Decay, Exponential Decay, Cosine Annealing,
# Warm-up + Cosine, and ReduceLROnPlateau.
# Compares how dynamically adjusting the learning rate
# improves training stability and convergence compared
# to a fixed high learning rate.
# ============================================================

import math, random
random.seed(42)

X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z): return z if z > 0 else 0.01 * z
def leaky_relu_deriv(z): return 1.0 if z > 0 else 0.01
def he_init(fan_in): return random.gauss(0, math.sqrt(2.0 / fan_in))

H1, H2 = 4, 4
EPOCHS = 100

# ------------------------------------------------------------
# Schedulers
# ------------------------------------------------------------

def step_decay(epoch, initial_lr=0.015, step_size=25, gamma=0.5):
    """Drops learning rate by gamma every step_size epochs."""
    # epoch is 1-indexed (1 to EPOCHS)
    return initial_lr * (gamma ** ((epoch - 1) // step_size))

def exponential_decay(epoch, initial_lr=0.015, decay_rate=0.015):
    """Continuously decays learning rate exponentially."""
    return initial_lr * math.exp(-decay_rate * (epoch - 1))

def cosine_annealing(epoch, initial_lr=0.015, eta_min=0.001, T_max=100):
    """Smoothly decays learning rate to eta_min using a cosine curve."""
    curr_t = epoch - 1
    if curr_t >= T_max:
        return eta_min
    return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * curr_t / T_max))


def warmup_cosine(epoch, initial_lr=0.015, warmup_epochs=10, eta_min=0.001, T_max=100):
    """Warm-up + Cosine: ramp the LR up linearly for the first `warmup_epochs`,
    THEN cosine-anneal down to eta_min.

    Why warm up? At epoch 1 the weights are random; a full-size LR can cause a
    wild, destabilizing first update. Ramping in gently avoids that. This is the
    standard schedule for training Transformers and large-batch models.
    """
    if epoch <= warmup_epochs:
        return initial_lr * epoch / warmup_epochs          # linear ramp 0 -> initial_lr
    progress = (epoch - warmup_epochs) / max(1, T_max - warmup_epochs)
    progress = min(1.0, progress)
    return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * progress))


class ReduceLROnPlateau:
    """Adaptive scheduler: drop the LR by `factor` when the monitored loss stops
    improving for `patience` epochs. Unlike the formula-based schedulers above,
    this responds to the ACTUAL training dynamics (pairs well with early stopping).

    Usage: read `.lr` for the current epoch, then call `.step(loss)` afterward.
    """
    def __init__(self, initial_lr=0.015, factor=0.5, patience=10, min_lr=1e-5, min_delta=1e-4):
        self.lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0

    def step(self, loss):
        """Call once per epoch with that epoch's loss; may lower self.lr."""
        if loss < self.best - self.min_delta:
            self.best = loss          # improvement -> reset patience
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.lr = max(self.min_lr, self.lr * self.factor)   # plateau -> drop LR
                self.counter = 0


def train(scheduler_func, initial_lr, label, plateau=None, **kwargs):
    random.seed(42)
    # Initialize weights
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    # Flatten parameters for simple vanilla SGD update
    flat = []
    for r in W1: flat.extend(r)
    for r in W2: flat.extend(r)
    flat.extend(W3[0]); flat.extend(b1); flat.extend(b2); flat.extend(b3)

    print(f"\n  {label}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        # Determine learning rate for this epoch
        if plateau is not None:
            lr = plateau.lr                                    # adaptive, set by loss feedback
        elif scheduler_func:
            lr = scheduler_func(epoch, initial_lr=initial_lr, **kwargs)
        else:
            lr = initial_lr

        grads = [0.0] * len(flat)
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]
            # Unflatten params
            idx = 0
            w1 = [[flat[idx+j*2+k] for k in range(2)] for j in range(H1)]; idx += H1*2
            w2 = [[flat[idx+j*H1+k] for k in range(H1)] for j in range(H2)]; idx += H2*H1
            w3 = [flat[idx:idx+H2]]; idx += H2
            b_1 = flat[idx:idx+H1]; idx += H1
            b_2 = flat[idx:idx+H2]; idx += H2
            b_3 = flat[idx:idx+1]

            # Forward pass
            z1 = [sum(w1[j][k]*x[k] for k in range(2))+b_1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(w2[j][k]*a1[k] for k in range(H1))+b_2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            yh = sum(w3[0][k]*a2[k] for k in range(H2)) + b_3[0]
            total_loss += (yh - y)**2

            # Backward pass
            dL = (2.0/n) * (yh - y)
            dW3 = [dL*a2[k] for k in range(H2)]
            dL_z2 = [dL*w3[0][j]*leaky_relu_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j]*a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j]*w2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j]*leaky_relu_deriv(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j]*x[k] for k in range(2)] for j in range(H1)]

            # Flat grads
            g = []
            for r in dW1: g.extend(r)
            for r in dW2: g.extend(r)
            g.extend(dW3); g.extend(dL_z1); g.extend(dL_z2); g.append(dL)
            for gi in range(len(grads)): grads[gi] += g[gi]

        # Update params using current epoch's learning rate
        flat = [p - lr * g for p, g in zip(flat, grads)]

        mse = total_loss / n
        losses.append(mse)

        if plateau is not None:
            plateau.step(mse)                                  # feed loss back for next epoch

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  LR: {lr:.5f}  |  MSE: {mse:.4f}")

    return losses


print("=" * 65)
print("  LEARNING RATE SCHEDULER COMPARISON")
print("=" * 65)

fixed = train(None, initial_lr=0.015, label="Fixed Learning Rate (lr=0.015)")
step  = train(step_decay, initial_lr=0.015, label="Step Decay (half every 25 epochs)", step_size=25, gamma=0.5)
exp   = train(exponential_decay, initial_lr=0.015, label="Exponential Decay (rate=0.015)", decay_rate=0.015)
cos   = train(cosine_annealing, initial_lr=0.015, label="Cosine Annealing (to 0.001)", eta_min=0.001, T_max=EPOCHS)
warm  = train(warmup_cosine, initial_lr=0.015, label="Warm-up (10) + Cosine", warmup_epochs=10, eta_min=0.001, T_max=EPOCHS)
# patience=20 here (vs the default 10) because this tiny raw-input demo is noisy
# early on -- a longer patience avoids over-reacting to those early loss spikes.
plat  = train(None, initial_lr=0.015, label="ReduceLROnPlateau (factor=0.5, patience=20)",
              plateau=ReduceLROnPlateau(initial_lr=0.015, factor=0.5, patience=20))

print(f"\n{'=' * 65}")
print(f"  Fixed LR final MSE:           {fixed[-1]:.4f}")
print(f"  Step Decay final MSE:         {step[-1]:.4f}")
print(f"  Exponential Decay final MSE:  {exp[-1]:.4f}")
print(f"  Cosine Annealing final MSE:   {cos[-1]:.4f}")
print(f"  Warm-up + Cosine final MSE:   {warm[-1]:.4f}")
print(f"  ReduceLROnPlateau final MSE:  {plat[-1]:.4f}")
print(f"{'=' * 65}")
