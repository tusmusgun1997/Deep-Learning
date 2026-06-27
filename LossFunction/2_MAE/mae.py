# ============================================================
# MEAN ABSOLUTE ERROR — From Scratch Implementation
# ============================================================
# L = (1/n) · Σ |ŷ - y|
# dL/dŷ = (1/n) · sign(ŷ - y)
#
# Trains an MLP with MAE loss and compares its outlier
# robustness vs MSE on the same dataset.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset with one deliberate outlier ----
X = [[8, 4], [5, 2], [7, 3], [3, 1], [9, 5], [4, 2]]
Y = [15.0, 7.0, 12.0, 4.0, 18.0, 6.0]
n = len(Y)


# ============================================================
# MAE LOSS + DERIVATIVE
# ============================================================

def mae_loss(y_pred, y_true):
    """L = (1/n) · Σ |ŷ - y|"""
    return sum(abs(p - t) for p, t in zip(y_pred, y_true)) / len(y_true)

def mae_gradient(y_pred, y_true):
    """dL/dŷᵢ = (1/n) · sign(ŷᵢ - yᵢ)"""
    n = len(y_true)
    def sign(x): return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
    return [(1.0 / n) * sign(p - t) for p, t in zip(y_pred, y_true)]

def mse_loss(y_pred, y_true):
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_true)


# ============================================================
# LOSS TABLE — How MAE Scales With Error (vs MSE)
# ============================================================

print("=" * 70)
print("  MAE vs MSE — Loss and Gradient for Different Error Sizes")
print("=" * 70)
print(f"  {'Error':>6}  |  {'MAE Loss':>10}  |  {'MAE grad':>10}  |  {'MSE Loss':>10}  |  {'MSE grad':>10}")
print(f"  {'-' * 62}")

for err in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]:
    sign = 1.0 if err > 0 else (-1.0 if err < 0 else 0.0)
    print(f"  {err:>6.1f}  |  {abs(err):>10.4f}  |  {sign:>10.4f}  |  {err**2:>10.4f}  |  {2*err:>10.4f}")

print("\n  NOTE: MAE gradient is always ±1 (or 0). MSE gradient scales with error.")
print("  Outlier (error=5): MAE grad=1, MSE grad=10. MAE is 10x more stable!")


# ============================================================
# OUTLIER ROBUSTNESS DEMO
# ============================================================

print(f"\n{'=' * 60}")
print("  OUTLIER ROBUSTNESS: MAE vs MSE")
print("=" * 60)

clean_preds   = [3.0, 5.0, 2.5,  4.0]
outlier_preds = [3.0, 5.0, 2.5, 14.0]   # last sample is an outlier
labels        = [2.0, 5.0, 4.0,  4.0]

print(f"  Clean predictions:")
print(f"    MAE = {mae_loss(clean_preds, labels):.4f}   MSE = {mse_loss(clean_preds, labels):.4f}")
print(f"\n  With one outlier (pred=14 vs true=4, error=10):")
print(f"    MAE = {mae_loss(outlier_preds, labels):.4f}   MSE = {mse_loss(outlier_preds, labels):.4f}")
print(f"\n  MAE increased by {mae_loss(outlier_preds, labels) - mae_loss(clean_preds, labels):.4f}")
print(f"  MSE increased by {mse_loss(outlier_preds, labels) - mse_loss(clean_preds, labels):.4f}")
print(f"  → MSE is {(mse_loss(outlier_preds, labels) - mse_loss(clean_preds, labels)) / (mae_loss(outlier_preds, labels) - mae_loss(clean_preds, labels)):.1f}x more affected by the outlier!")


# ============================================================
# NETWORK HELPERS
# ============================================================

def relu(z):      return max(0.0, z)
def relu_d(z):    return 1.0 if z > 0 else 0.0
def sign(x):      return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
def he_init(fan): return random.gauss(0, math.sqrt(2.0 / fan))

H = 4
LR = 0.05
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
# TRAINING WITH MAE LOSS
# ============================================================

W1, b1, W2, b2 = make_net()

print(f"\n{'=' * 60}")
print(f"  TRAINING MLP WITH MAE LOSS (LR={LR}, {EPOCHS} epochs)")
print(f"{'=' * 60}")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for i in range(n):
        x, y = X[i], Y[i]
        z1, a1, yh = forward(x, W1, b1, W2, b2)

        error = yh - y
        total_loss += abs(error)

        # MAE gradient is (1/n) * sign(ŷ - y)
        dL = (1.0 / n) * sign(error)

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

    mae = total_loss / n
    if epoch % 60 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  |  MAE: {mae:.4f}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print(f"\n{'=' * 60}")
print("  FINAL PREDICTIONS vs TARGETS")
print(f"{'=' * 60}")
preds = []
for i in range(n):
    _, _, yh = forward(X[i], W1, b1, W2, b2)
    preds.append(yh)
    print(f"  Input {X[i]}  →  Target: {Y[i]:.2f}  |  Predicted: {yh:.4f}  |  Error: {yh - Y[i]:.4f}")

print(f"\n  Final MAE: {mae_loss(preds, Y):.4f}")
print(f"  Final MSE: {mse_loss(preds, Y):.4f}")
print(f"{'=' * 60}")
