# ============================================================
# RANDOM SEARCH — From Scratch
# ============================================================
# Instead of trying EVERY combination on a fixed grid, randomly
# SAMPLE configurations from the hyperparameter ranges.
#
# Why it usually beats grid search for the SAME budget:
#   Most hyperparameters barely matter. Random search spends its
#   trials exploring MANY distinct values of the few that DO matter,
#   instead of redundantly re-testing the unimportant ones.
#   (Bergstra & Bengio, 2012)
#
# Also demonstrates sampling the learning rate on a LOG scale,
# which is the correct way to search it (see hyperparameter_tuning.md).
# ============================================================

import math, random

# Separate RNG for CONFIG sampling, kept independent from the global RNG
# that train_and_evaluate reseeds for reproducible weight initialization.
cfg_rng = random.Random(7)

# Dataset: CGPA, Profile Score -> Package (in LPA)
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z): return z if z > 0 else 0.01 * z
def leaky_relu_deriv(z): return 1.0 if z > 0 else 0.01
def he_init(fan_in): return random.gauss(0, math.sqrt(2.0 / fan_in))

EPOCHS = 100
N_TRIALS = 9   # same budget as grid_search.py's 3x3 grid, for a fair comparison


def train_and_evaluate(lr, hidden_dim):
    """Trains a 1-hidden-layer network and returns the final MSE.
    Same network/training as grid_search.py so results are comparable."""
    random.seed(42)   # reproducible weight init (global RNG only)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(hidden_dim)]
    b1 = [0.0] * hidden_dim
    W2 = [he_init(hidden_dim) for _ in range(hidden_dim)]
    b2 = 0.0

    for epoch in range(1, EPOCHS + 1):
        dW1 = [[0.0, 0.0] for _ in range(hidden_dim)]
        db1 = [0.0] * hidden_dim
        dW2 = [0.0] * hidden_dim
        db2_val = 0.0
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]

            # Forward pass
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(hidden_dim)]
            a1 = [leaky_relu(z) for z in z1]
            yh = sum(W2[j] * a1[j] for j in range(hidden_dim)) + b2
            total_loss += (yh - y) ** 2

            # Backward pass
            dL = (2.0 / n) * (yh - y)
            for j in range(hidden_dim):
                dW2[j] += dL * a1[j]
            db2_val += dL
            dL_a1 = [dL * W2[j] for j in range(hidden_dim)]
            dL_z1 = [dL_a1[j] * leaky_relu_deriv(z1[j]) for j in range(hidden_dim)]
            for j in range(hidden_dim):
                for k in range(2):
                    dW1[j][k] += dL_z1[j] * x[k]
                db1[j] += dL_z1[j]

        # Parameter updates
        for j in range(hidden_dim):
            for k in range(2):
                W1[j][k] -= lr * dW1[j][k]
            b1[j] -= lr * db1[j]
            W2[j] -= lr * dW2[j]
        b2 -= lr * db2_val

    return total_loss / n


def sample_lr():
    """Sample LR on a LOG scale: 10^U(-2.3, -1.7) ~ [0.005, 0.02].
    Log-scale is correct because LR matters across orders of magnitude."""
    return 10 ** cfg_rng.uniform(-2.3, -1.7)

def sample_hidden():
    return cfg_rng.randint(2, 6)


print("=" * 64)
print("  RANDOM SEARCH HYPERPARAMETER TUNING")
print(f"  {N_TRIALS} random trials  |  LR sampled on a log scale [~0.005, 0.02]")
print("=" * 64)

results = []
best_mse = float('inf')
best_config = None

for trial in range(1, N_TRIALS + 1):
    lr = sample_lr()
    hidden = sample_hidden()
    mse = train_and_evaluate(lr, hidden)
    results.append((lr, hidden, mse))

    # Track best (guard against divergence -> NaN)
    if not math.isnan(mse) and mse < best_mse:
        best_mse = mse
        best_config = (lr, hidden)

# Results table
print(f"{'Trial':<6} | {'Learning Rate':<15} | {'Hidden Neurons':<15} | {'Final MSE':<12}")
print("-" * 60)
for t, (lr, hidden, mse) in enumerate(results, 1):
    mse_str = f"{mse:.4f}" if not math.isnan(mse) else "NaN (Diverged)"
    print(f"{t:<6} | {lr:<15.5f} | {hidden:<15} | {mse_str:<12}")

print("=" * 64)
if best_config:
    print(f"  BEST CONFIGURATION:")
    print(f"  Learning Rate:  {best_config[0]:.5f}")
    print(f"  Hidden Neurons: {best_config[1]}")
    print(f"  Lowest MSE:     {best_mse:.4f}")
else:
    print("  All configurations diverged.")
print("=" * 64)
print("\n  Note: Grid search tried only 3 fixed LR values (re-used across")
print("  hidden sizes). Random search tried 9 DISTINCT learning rates for")
print("  the same 9-trial budget -> better coverage of the parameter that matters.")
