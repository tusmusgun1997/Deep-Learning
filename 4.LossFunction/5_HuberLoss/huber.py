# ============================================================
# HUBER LOSS — From Scratch Implementation
# ============================================================
# L = 0.5·error²           if |error| ≤ δ   (MSE zone)
#     δ·(|error| - 0.5·δ)  if |error| > δ   (MAE zone)
#
# Trains an MLP with Huber loss and demonstrates its
# robustness to outliers vs pure MSE.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset with a deliberate outlier ----
# [CGPA, Profile Score] → Package (LPA)
# Last sample is an outlier (data entry error or exceptional case)
X = [[8, 4], [5, 2], [7, 3], [3, 1], [9, 5], [4, 2], [6, 3]]
Y = [15.0, 7.0, 12.0, 4.0, 18.0, 6.0, 50.0]   # 50.0 is the outlier
n = len(Y)

DELTA = 1.0   # threshold between MSE and MAE zones


# ============================================================
# HUBER LOSS + DERIVATIVE
# ============================================================

def huber_loss_single(error, delta=DELTA):
    """Per-sample Huber loss."""
    if abs(error) <= delta:
        return 0.5 * error ** 2
    else:
        return delta * (abs(error) - 0.5 * delta)

def huber_gradient_single(error, delta=DELTA):
    """Per-sample Huber gradient."""
    if abs(error) <= delta:
        return error                          # MSE zone: smooth gradient
    else:
        return delta * (1.0 if error > 0 else -1.0)   # MAE zone: capped

def huber_loss(y_pred, y_true, delta=DELTA):
    return sum(huber_loss_single(p - t, delta) for p, t in zip(y_pred, y_true)) / len(y_true)

def mse_loss(y_pred, y_true):
    return sum((p - t) ** 2 for p, t in zip(y_pred, y_true)) / len(y_true)


# ============================================================
# COMPARISON TABLE — Huber vs MSE vs MAE on Different Errors
# ============================================================

print("=" * 80)
print(f"  Huber vs MSE vs MAE (δ={DELTA})")
print("=" * 80)
print(f"  {'Error':>8}  |  {'Huber Loss':>12}  |  {'Huber Grad':>12}  |  {'MSE Loss':>10}  |  {'MSE Grad':>10}")
print(f"  {'-' * 68}")

for err in [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    h_loss = huber_loss_single(err)
    h_grad = huber_gradient_single(err)
    m_loss = err ** 2
    m_grad = 2 * err
    zone = "MSE" if abs(err) <= DELTA else "MAE"
    print(f"  {err:>8.1f}  |  {h_loss:>12.4f}  |  {h_grad:>12.4f}  |  {m_loss:>10.4f}  |  {m_grad:>10.4f}  [{zone}]")

print(f"\n  For error=10: Huber grad={huber_gradient_single(10):.1f}  vs  MSE grad={2*10:.1f}")
print(f"  Huber CAPS the gradient at δ={DELTA}, preventing outlier from dominating!")


# ============================================================
# OUTLIER IMPACT COMPARISON
# ============================================================

print(f"\n{'=' * 60}")
print("  OUTLIER IMPACT: Huber vs MSE")
print("=" * 60)

clean_preds   = [15.1, 7.2, 12.1, 4.1, 18.2, 6.1, 11.0]   # good predictions
outlier_preds = [15.1, 7.2, 12.1, 4.1, 18.2, 6.1, 11.0]   # same, but last true=50

print(f"  Dataset has an outlier: true label = 50.0, prediction = 11.0, error = -39.0")
print(f"  Huber loss: {huber_loss(clean_preds, Y):.4f}")
print(f"  MSE loss:   {mse_loss(clean_preds, Y):.4f}")
print(f"\n  Huber gradient on outlier: {huber_gradient_single(11.0 - 50.0):.4f}  (capped at -δ = -{DELTA})")
print(f"  MSE gradient on outlier:   {2 * (11.0 - 50.0):.4f}  (proportional — HUGE!)")


# ============================================================
# NETWORK HELPERS
# ============================================================

def relu(z):      return max(0.0, z)
def relu_d(z):    return 1.0 if z > 0 else 0.0
def he_init(fan): return random.gauss(0, math.sqrt(2.0 / fan))

H = 4
LR = 0.01
EPOCHS = 500

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
# TRAINING WITH HUBER LOSS
# ============================================================

W1, b1, W2, b2 = make_net()

print(f"\n{'=' * 60}")
print(f"  TRAINING MLP WITH HUBER LOSS (δ={DELTA}, LR={LR}, {EPOCHS} epochs)")
print(f"{'=' * 60}")

for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for i in range(n):
        x, y = X[i], Y[i]
        z1, a1, yh = forward(x, W1, b1, W2, b2)

        error = yh - y
        total_loss += huber_loss_single(error)

        # Huber gradient — capped for large errors
        dL = huber_gradient_single(error) / n

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

    hloss = total_loss / n
    if epoch % 100 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  |  Huber Loss: {hloss:.4f}")


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
    tag = " ← OUTLIER" if Y[i] == 50.0 else ""
    print(f"  {str(X[i]):>12}  |  Target: {Y[i]:>5.1f}  |  Pred: {yh:>8.4f}  |  Error: {yh-Y[i]:>8.4f}{tag}")

print(f"\n  Final Huber Loss: {huber_loss(preds, Y):.4f}")
print(f"  Final MSE:        {mse_loss(preds, Y):.4f}")
print(f"\n  The outlier (true=50, pred≈{preds[-1]:.1f}) is partially ignored by Huber.")
print(f"  This protects the other predictions from being corrupted by the outlier.")
print(f"{'=' * 60}")
