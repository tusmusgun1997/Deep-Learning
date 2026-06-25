# ============================================================
# EARLY STOPPING — From Scratch
# ============================================================
# Monitors validation loss during training. When it hasn't
# improved for 'patience' epochs, training stops and the best
# weights are restored.
#
# This is FREE regularization -- no extra hyperparameters
# to tune besides patience.
# ============================================================

import math
import random

random.seed(42)

# ---- Dataset (split into train + validation) ----
# Train: 3 samples, Validation: 1 sample
X_train = [[8, 4], [5, 2], [7, 3]]
Y_train = [15, 7, 12]
X_val = [[3, 1]]
Y_val = [4]

n_train = len(Y_train)
n_val = len(Y_val)

# ---- Activation ----
def leaky_relu(z):
    return z if z > 0 else 0.01 * z

def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


# ================================================================
# EARLY STOPPING IMPLEMENTATION
# ================================================================

class EarlyStopping:
    """Early Stopping: Stop training when validation loss stops improving.

    How it works:
        1. After each epoch, check validation loss
        2. If val_loss < best_val_loss - min_delta: save weights, reset counter
        3. If no improvement for 'patience' epochs: STOP and restore best weights

    Args:
        patience: how many epochs to wait for improvement
        min_delta: minimum change to qualify as improvement
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None

    def check(self, val_loss, weights, epoch):
        """Returns True if training should stop."""

        if val_loss < self.best_loss - self.min_delta:
            # IMPROVEMENT! Save best state
            self.best_loss = val_loss
            self.best_weights = weights[:]
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # STOP TRAINING!
            return False


# ================================================================
# NETWORK: 2 -> 4 -> 4 -> 1
# ================================================================

H1, H2 = 4, 4
LR = 0.01


def flatten(W1, b1, W2, b2, W3, b3):
    flat = []
    for row in W1: flat.extend(row)
    for row in W2: flat.extend(row)
    flat.extend(W3[0])
    flat.extend(b1)
    flat.extend(b2)
    flat.extend(b3)
    return flat


def unflatten(flat):
    idx = 0
    W1 = []
    for j in range(H1):
        W1.append([flat[idx], flat[idx+1]])
        idx += 2
    W2 = []
    for j in range(H2):
        row = flat[idx:idx+H1]
        W2.append(row)
        idx += H1
    W3 = [flat[idx:idx+H2]]
    idx += H2
    b1 = flat[idx:idx+H1]; idx += H1
    b2 = flat[idx:idx+H2]; idx += H2
    b3 = flat[idx:idx+1]; idx += 1
    return W1, b1, W2, b2, W3, b3


def compute_mse(W1, b1, W2, b2, W3, b3, X_data, Y_data):
    """Compute MSE on a dataset."""
    total = 0.0
    for i in range(len(Y_data)):
        x = X_data[i]
        z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
        a1 = [leaky_relu(z) for z in z1]
        z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
        a2 = [leaky_relu(z) for z in z2]
        y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]
        total += (y_hat - Y_data[i]) ** 2
    return total / len(Y_data)


def train_network(use_early_stopping=False, patience=10, max_epochs=200):
    """Train with or without early stopping."""
    random.seed(42)

    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    stopper = EarlyStopping(patience=patience) if use_early_stopping else None

    label = f"WITH Early Stopping (patience={patience})" if use_early_stopping else "WITHOUT Early Stopping"
    print(f"\n  {label}")
    print(f"  {'-' * 55}")

    train_losses = []
    val_losses = []
    actual_epochs = 0

    for epoch in range(1, max_epochs + 1):
        # ---- Train on training set ----
        for i in range(n_train):
            x, y = X_train[i], Y_train[i]

            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            dL = (2.0 / n_train) * (y_hat - y)

            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * leaky_relu_derivative(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j] * x[k] for k in range(2)] for j in range(H1)]

            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * dW1[j][k]
                b1[j] -= LR * dL_z1[j]
            for j in range(H2):
                for k in range(H1):
                    W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * dL_z2[j]
            for k in range(H2):
                W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * dL

        # ---- Compute losses ----
        train_mse = compute_mse(W1, b1, W2, b2, W3, b3, X_train, Y_train)
        val_mse = compute_mse(W1, b1, W2, b2, W3, b3, X_val, Y_val)
        train_losses.append(train_mse)
        val_losses.append(val_mse)
        actual_epochs = epoch

        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  Train: {train_mse:.4f}  |  Val: {val_mse:.4f}", end="")
            if stopper:
                print(f"  |  Patience: {stopper.counter}/{patience}", end="")
            print()

        # ---- Early Stopping Check ----
        if stopper:
            current_weights = flatten(W1, b1, W2, b2, W3, b3)
            if stopper.check(val_mse, current_weights, epoch):
                print(f"\n    ** EARLY STOP at epoch {epoch}! **")
                print(f"    ** Best val loss: {stopper.best_loss:.4f} at epoch {stopper.best_epoch} **")
                # Restore best weights
                W1, b1, W2, b2, W3, b3 = unflatten(stopper.best_weights)
                break

    print(f"\n    Total epochs trained: {actual_epochs}")
    print(f"    Final Train MSE: {train_losses[-1]:.4f}")
    print(f"    Final Val MSE:   {val_losses[-1]:.4f}")

    return train_losses, val_losses, actual_epochs


# ================================================================
# RUN COMPARISON
# ================================================================

print("=" * 60)
print("  EARLY STOPPING DEMO")
print("  Training set: 3 samples  |  Validation set: 1 sample")
print("  Network: 2 -> 4 -> 4 -> 1")
print("=" * 60)

t1, v1, e1 = train_network(use_early_stopping=False, max_epochs=200)
t2, v2, e2 = train_network(use_early_stopping=True, patience=15, max_epochs=200)

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>25} {'No Stop':>12} {'Early Stop':>12}")
print(f"  {'-' * 50}")
print(f"  {'Epochs trained':>25} {e1:>12} {e2:>12}")
print(f"  {'Final Train MSE':>25} {t1[-1]:>12.4f} {t2[e2-1]:>12.4f}")
print(f"  {'Final Val MSE':>25} {v1[-1]:>12.4f} {v2[e2-1]:>12.4f}")
print(f"  {'Best Val MSE':>25} {min(v1):>12.4f} {min(v2):>12.4f}")
print(f"{'=' * 60}")
print("\n  Early stopping saves training time and prevents")
print("  the model from memorizing the training set.")
