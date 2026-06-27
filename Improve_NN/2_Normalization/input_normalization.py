# ============================================================
# INPUT NORMALIZATION — From Scratch
# ============================================================
# Scales input features to a similar range so that no single
# feature dominates the gradients.
#
# Methods implemented:
#   1. Min-Max Normalization     -> scales to [0, 1]
#   2. Z-Score (Standardization) -> scales to mean=0, std=1
#   3. Max-Abs Normalization     -> scales to [-1, 1]
# ============================================================

import math
import random

random.seed(42)

# ---- Dataset ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ---- Activation ----
def leaky_relu(z):
    return z if z > 0 else 0.01 * z
def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01
def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


# ================================================================
# NORMALIZATION FUNCTIONS
# ================================================================

def min_max_normalize(data, feature_idx):
    """Min-Max: Scale feature to [0, 1].

    Formula: x_norm = (x - min) / (max - min)

    Example: CGPA values [3, 5, 7, 8]
      min=3, max=8, range=5
      3 -> 0.0, 5 -> 0.4, 7 -> 0.8, 8 -> 1.0
    """
    values = [row[feature_idx] for row in data]
    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v if max_v != min_v else 1.0

    result = [row[:] for row in data]
    for row in result:
        row[feature_idx] = (row[feature_idx] - min_v) / range_v

    return result, min_v, max_v


def z_score_normalize(data, feature_idx):
    """Z-Score / Standardization: Scale to mean=0, std=1.

    Formula: x_norm = (x - mean) / std

    Example: CGPA values [3, 5, 7, 8]
      mean=5.75, std=1.92
      3 -> -1.43, 5 -> -0.39, 7 -> 0.65, 8 -> 1.17
    """
    values = [row[feature_idx] for row in data]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance) if variance > 0 else 1.0

    result = [row[:] for row in data]
    for row in result:
        row[feature_idx] = (row[feature_idx] - mean) / std

    return result, mean, std


def max_abs_normalize(data, feature_idx):
    """Max-Abs: Scale feature to [-1, 1] by dividing by the largest |value|.

    Formula: x_norm = x / max(|x|)

    Example: CGPA values [3, 5, 7, 8]
      max(|x|)=8
      3 -> 0.375, 5 -> 0.625, 7 -> 0.875, 8 -> 1.0
    Best when data is already centered near 0 / sparse (it keeps zeros at zero).
    """
    values = [row[feature_idx] for row in data]
    max_abs = max(abs(v) for v in values)
    if max_abs == 0:
        max_abs = 1.0

    result = [row[:] for row in data]
    for row in result:
        row[feature_idx] = row[feature_idx] / max_abs

    return result, max_abs


def normalize_all(data, method="zscore"):
    """Normalize all features in the dataset."""
    result = [row[:] for row in data]
    stats = []
    for f in range(len(data[0])):
        if method == "minmax":
            result, min_v, max_v = min_max_normalize(result, f)
            stats.append(('minmax', min_v, max_v))
        elif method == "maxabs":
            result, max_abs = max_abs_normalize(result, f)
            stats.append(('maxabs', max_abs))
        else:
            result, mean, std = z_score_normalize(result, f)
            stats.append(('zscore', mean, std))
    return result, stats


# ================================================================
# DEMO: Show normalization in action
# ================================================================

print("=" * 60)
print("  INPUT NORMALIZATION DEMO")
print("=" * 60)

print("\n  Original data:")
print(f"    {'CGPA':>6} {'Profile':>8}")
for row in X:
    print(f"    {row[0]:>6} {row[1]:>8}")

X_minmax, mm_stats = normalize_all(X, method="minmax")
print(f"\n  After Min-Max Normalization (range [0, 1]):")
print(f"    {'CGPA':>8} {'Profile':>8}")
for row in X_minmax:
    print(f"    {row[0]:>8.4f} {row[1]:>8.4f}")

X_zscore, zs_stats = normalize_all(X, method="zscore")
print(f"\n  After Z-Score Normalization (mean=0, std=1):")
print(f"    {'CGPA':>8} {'Profile':>8}")
for row in X_zscore:
    print(f"    {row[0]:>8.4f} {row[1]:>8.4f}")

X_maxabs, ma_stats = normalize_all(X, method="maxabs")
print(f"\n  After Max-Abs Normalization (range [-1, 1]):")
print(f"    {'CGPA':>8} {'Profile':>8}")
for row in X_maxabs:
    print(f"    {row[0]:>8.4f} {row[1]:>8.4f}")


# ================================================================
# TRAIN: Compare normalized vs unnormalized
# ================================================================

H1, H2 = 4, 4
EPOCHS = 100
LR = 0.01


def train_network(X_data, label=""):
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n  Training: {label}")
    print(f"  {'-' * 50}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for i in range(n):
            x, y = X_data[i], Y[i]
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]
            error = y_hat - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * leaky_relu_derivative(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j] * x[k] for k in range(2)] for j in range(H1)]

            for j in range(H1):
                for k in range(2): W1[j][k] -= LR * dW1[j][k]
                b1[j] -= LR * dL_z1[j]
            for j in range(H2):
                for k in range(H1): W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * dL_z2[j]
            for k in range(H2): W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * dL

        mse = total_loss / n
        losses.append(mse)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")

    return losses


print(f"\n{'=' * 60}")
print("  TRAINING COMPARISON")
print(f"{'=' * 60}")

raw = train_network(X, label="Raw (no normalization)")
mm = train_network(X_minmax, label="Min-Max Normalized")
zs = train_network(X_zscore, label="Z-Score Normalized")
ma = train_network(X_maxabs, label="Max-Abs Normalized")

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>20} {'Raw':>10} {'Min-Max':>10} {'Z-Score':>10} {'Max-Abs':>10}")
print(f"  {'-' * 62}")
print(f"  {'Final MSE':>20} {raw[-1]:>10.4f} {mm[-1]:>10.4f} {zs[-1]:>10.4f} {ma[-1]:>10.4f}")
print(f"  {'Best MSE':>20} {min(raw):>10.4f} {min(mm):>10.4f} {min(zs):>10.4f} {min(ma):>10.4f}")
print(f"{'=' * 60}")
