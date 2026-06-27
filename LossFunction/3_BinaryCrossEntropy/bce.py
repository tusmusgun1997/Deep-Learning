# ============================================================
# BINARY CROSS-ENTROPY — From Scratch Implementation
# ============================================================
# L = -(1/n) · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]
# dL/dŷ = (1/n) · [ -y/ŷ + (1-y)/(1-ŷ) ]
#
# Trains an MLP for binary classification using sigmoid
# output + BCE loss. Dataset: predict admission (1) or not (0)
# from CGPA and profile score.
# ============================================================

import math
import random
random.seed(42)

# ---- Binary Classification Dataset ----
# Features: [CGPA, Profile Score],  Label: 1=Admitted, 0=Not Admitted
X = [[8, 4], [9, 5], [7, 3], [5, 2], [4, 1], [3, 1], [8, 5], [6, 2]]
Y = [1,      1,      1,      0,      0,      0,      1,      0]
n = len(Y)


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):   return max(0.0, z)
def relu_d(z): return 1.0 if z > 0 else 0.0


# ============================================================
# BCE LOSS + DERIVATIVE
# ============================================================

EPS = 1e-7  # numerical stability — prevent log(0)

def bce_loss(y_pred, y_true):
    """L = -(1/n) · Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]"""
    total = 0.0
    for p, t in zip(y_pred, y_true):
        p = max(EPS, min(1 - EPS, p))   # clip to (ε, 1-ε)
        total += t * math.log(p) + (1 - t) * math.log(1 - p)
    return -total / len(y_true)

def bce_gradient(y_pred_single, y_true_single):
    """dL/dŷ = (1/n) · [ -y/ŷ + (1-y)/(1-ŷ) ]  per sample"""
    p = max(EPS, min(1 - EPS, y_pred_single))
    return -y_true_single / p + (1 - y_true_single) / (1 - p)


# ============================================================
# LOSS TABLE — How BCE Reacts to Different Predictions
# ============================================================

print("=" * 70)
print("  BCE — Loss for Different Predictions (True Label = 1)")
print("=" * 70)
print(f"  {'Prediction ŷ':>14}  |  {'Loss -log(ŷ)':>14}  |  {'Gradient':>10}")
print(f"  {'-' * 48}")

for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    loss = -math.log(p + EPS)
    grad = bce_gradient(p, 1)
    print(f"  {p:>14.2f}  |  {loss:>14.4f}  |  {grad:>10.4f}")

print(f"\n  NOTE: ŷ=0.01 (very wrong) → loss={-math.log(0.01):.2f}, huge gradient!")
print(f"  ŷ=0.99 (very right) → loss={-math.log(0.99):.4f}, tiny gradient.")


# ============================================================
# NETWORK
# ============================================================

H = 4
LR = 0.1
EPOCHS = 300

def he_init(fan): return random.gauss(0, math.sqrt(2.0 / fan))

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
    z_out = sum(W2[0][k] * a1[k] for k in range(H)) + b2[0]
    yh = sigmoid(z_out)          # sigmoid for binary classification!
    return z1, a1, z_out, yh


# ============================================================
# TRAINING WITH BCE LOSS
# ============================================================

W1, b1, W2, b2 = make_net()

print(f"\n{'=' * 60}")
print(f"  TRAINING MLP WITH BCE LOSS (LR={LR}, {EPOCHS} epochs)")
print(f"{'=' * 60}")

for epoch in range(1, EPOCHS + 1):
    preds_all = []
    for i in range(n):
        x, y = X[i], Y[i]
        z1, a1, z_out, yh = forward(x, W1, b1, W2, b2)
        preds_all.append(yh)

        # dL/dz_out = ŷ - y  (combined sigmoid + BCE gradient — very clean!)
        dL_zout = (yh - y) / n

        dW2 = [dL_zout * a1[k] for k in range(H)]
        dL_a1 = [dL_zout * W2[0][k] for k in range(H)]
        dL_z1 = [dL_a1[j] * relu_d(z1[j]) for j in range(H)]

        for j in range(H):
            for k in range(2):
                W1[j][k] -= LR * dL_z1[j] * x[k]
            b1[j] -= LR * dL_z1[j]
        for k in range(H):
            W2[0][k] -= LR * dW2[k]
        b2[0] -= LR * dL_zout

    loss = bce_loss(preds_all, Y)
    acc = sum(1 for p, t in zip(preds_all, Y) if (p >= 0.5) == bool(t)) / n

    if epoch % 60 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  |  BCE: {loss:.4f}  |  Accuracy: {acc:.2%}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print(f"\n{'=' * 60}")
print("  FINAL PREDICTIONS vs TARGETS")
print(f"{'=' * 60}")
preds_final = []
for i in range(n):
    _, _, _, yh = forward(X[i], W1, b1, W2, b2)
    preds_final.append(yh)
    label = "Admitted" if yh >= 0.5 else "Rejected"
    correct = "✓" if (yh >= 0.5) == bool(Y[i]) else "✗"
    print(f"  {str(X[i]):>12}  |  True: {Y[i]}  |  ŷ: {yh:.4f}  |  {label}  {correct}")

print(f"\n  Final BCE loss: {bce_loss(preds_final, Y):.4f}")
accuracy = sum(1 for p, t in zip(preds_final, Y) if (p >= 0.5) == bool(t)) / n
print(f"  Final Accuracy: {accuracy:.2%}")
print(f"{'=' * 60}")
