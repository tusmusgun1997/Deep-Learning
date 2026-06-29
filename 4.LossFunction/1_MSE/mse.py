# ============================================================
# MEAN SQUARED ERROR — From Scratch Implementation
# ============================================================
# L = (1/n) · Σ (ŷ - y)²
# dL/dŷ = (2/n) · (ŷ - y)
#
# Trains a simple MLP for regression using only MSE loss,
# showing how it penalizes large errors harder.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Package (LPA) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1], [9, 5], [4, 2]]
Y = [15.0, 7.0, 12.0, 4.0, 18.0, 6.0]
n = len(Y)


# ============================================================
# MSE LOSS + DERIVATIVE
# ============================================================

def mse_loss(y_pred, y_true):
    """L = (1/n) · Σ (ŷ - y)²"""
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_true)

def mse_gradient(y_pred, y_true):
    """dL/dŷᵢ = (2/n) · (ŷᵢ - yᵢ)"""
    n = len(y_true)
    return [(2.0 / n) * (p - t) for p, t in zip(y_pred, y_true)]


# ============================================================
# LOSS TABLE — How MSE Scales With Error
# ============================================================

print("=" * 60)
print("  MSE — How Loss Scales With Error Size")
print("=" * 60)
print(f"  {'Error':>8}  |  {'MSE Loss':>10}  |  {'Gradient':>10}")
print(f"  {'-' * 42}")

for err in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]:
    loss = err ** 2
    grad = 2 * err
    print(f"  {err:>8.1f}  |  {loss:>10.4f}  |  {grad:>10.4f}")

print("\n  NOTE: Error doubles (1→2), loss QUADRUPLES (1→4).")
print("  Outliers dominate MSE training.")


# ============================================================
# OUTLIER SENSITIVITY DEMO
# ============================================================

print(f"\n{'=' * 60}")
print("  OUTLIER SENSITIVITY DEMO")
print("=" * 60)

clean_preds  = [3.0, 5.0, 2.5, 4.0]
clean_labels = [2.0, 5.0, 4.0, 4.0]

outlier_preds  = [3.0, 5.0, 2.5, 14.0]   # last sample is an outlier
outlier_labels = [2.0, 5.0, 4.0,  4.0]

print(f"  Clean data   MSE: {mse_loss(clean_preds, clean_labels):.4f}")
print(f"  Outlier data MSE: {mse_loss(outlier_preds, outlier_labels):.4f}")
print(f"  → One outlier (error=10) explodes the loss!")
print(f"    Clean gradients: {mse_gradient(clean_preds, clean_labels)}")
print(f"  Outlier gradients: {mse_gradient(outlier_preds, outlier_labels)}")


# ============================================================
# NETWORK HELPERS
# ============================================================

def relu(z):      return max(0.0, z)
def relu_d(z):    return 1.0 if z > 0 else 0.0
def he_init(fan): return random.gauss(0, math.sqrt(2.0 / fan))

H = 4
LR = 0.01
EPOCHS = 300

def make_net():
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H)]
    b1 = [0.0] * H
    W2 = [[he_init(H) for _ in range(H)]]
    b2 = [0.0]
    return W1, b1, W2, b2

def forward(x, W1, b1, W2, b2):
    z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H)]
    a1 = [relu(z) for z in z1]
    yh = sum(W2[0][k] * a1[k] for k in range(H)) + b2[0]
    return z1, a1, yh


# ============================================================
# TRAINING WITH MSE LOSS
# ============================================================

W1, b1, W2, b2 = make_net()

print(f"\n{'=' * 60}")
print(f"  TRAINING MLP WITH MSE LOSS (LR={LR}, {EPOCHS} epochs)")
print(f"{'=' * 60}")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for i in range(n):
        x, y = X[i], Y[i]
        z1, a1, yh = forward(x, W1, b1, W2, b2)

        error = yh - y
        total_loss += error ** 2

        # Backprop — MSE gradient is (2/n)*(ŷ - y)
        dL = (2.0 / n) * error

        dW2 = [dL * a1[k] for k in range(H)]
        dL_a1 = [dL * W2[0][k] for k in range(H)]
        dL_z1 = [dL_a1[j] * relu_d(z1[j]) for j in range(H)]

        for j in range(H):
            for k in range(2):
                W1[j][k] -= LR * dL_z1[j] * x[k]
            b1[j] -= LR * dL_z1[j]
        for k in range(H):
            W2[0][k] -= LR * dW2[k]
        b2[0] -= LR * dL

    mse = total_loss / n
    if epoch % 60 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  |  MSE: {mse:.4f}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print(f"\n{'=' * 60}")
print("  FINAL PREDICTIONS vs TARGETS")
print(f"{'=' * 60}")
print(f"  {'Input':>12}  |  {'Target':>8}  |  {'Predicted':>10}  |  {'Error':>8}")
print(f"  {'-' * 52}")
for i in range(n):
    _, _, yh = forward(X[i], W1, b1, W2, b2)
    print(f"  {str(X[i]):>12}  |  {Y[i]:>8.2f}  |  {yh:>10.4f}  |  {yh - Y[i]:>8.4f}")

print(f"\n  Final MSE: {mse_loss([forward(X[i], W1, b1, W2, b2)[2] for i in range(n)], Y):.4f}")
print(f"{'=' * 60}")
