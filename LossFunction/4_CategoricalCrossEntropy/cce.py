# ============================================================
# CATEGORICAL CROSS-ENTROPY — From Scratch Implementation
# ============================================================
# L = -(1/n) · Σᵢ log(ŷᵢ,correct_class)
# Gradient w.r.t. softmax input: dL/dz = ŷ - y
#
# Trains an MLP for 3-class classification using Softmax
# output + CCE loss.
# Dataset: Predict grade (A / B / C) from study hours & score.
# ============================================================

import math
import random
random.seed(42)

# ---- 3-Class Dataset ----
# Features: [study_hours, test_score],  Label: 0=A, 1=B, 2=C
X = [[9, 90], [8, 80], [7, 70], [6, 60], [5, 50], [4, 40], [3, 30], [2, 20]]
Y = [0,       0,       1,       1,       1,        2,       2,       2]
n = len(Y)
NUM_CLASSES = 3


# ============================================================
# SOFTMAX + CCE
# ============================================================

def softmax(z_vec):
    """Numerically stable softmax using log-sum-exp trick."""
    max_z = max(z_vec)
    exp_z = [math.exp(z - max_z) for z in z_vec]
    total = sum(exp_z)
    return [e / total for e in exp_z]

def cce_loss(probs_list, labels):
    """L = -(1/n) · Σ log(ŷ_correct)"""
    EPS = 1e-7
    total = 0.0
    for probs, label in zip(probs_list, labels):
        p_correct = max(EPS, probs[label])
        total += math.log(p_correct)
    return -total / len(labels)

def one_hot(label, num_classes):
    v = [0.0] * num_classes
    v[label] = 1.0
    return v


# ============================================================
# LOSS TABLE — How CCE Reacts to Different Predictions
# ============================================================

print("=" * 70)
print("  CCE — Loss for Different Correct-Class Probabilities")
print("=" * 70)
print(f"  {'P(correct)':>12}  |  {'Loss -log(P)':>14}  |  Interpretation")
print(f"  {'-' * 60}")

for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
    loss = -math.log(p)
    quality = ("Very wrong!" if p < 0.2 else
               "Wrong" if p < 0.4 else
               "Uncertain" if p < 0.6 else
               "Good" if p < 0.8 else
               "Great!" if p < 0.95 else "Near perfect!")
    print(f"  {p:>12.2f}  |  {loss:>14.4f}  |  {quality}")

print("\n  Softmax+CCE gradient = ŷ - y  (very simple!)")
print("  Example: y=[1,0,0], ŷ=[0.7,0.2,0.1]  →  grad=[-0.3, +0.2, +0.1]")


# ============================================================
# NETWORK
# ============================================================

H = 8
LR = 0.005
EPOCHS = 500

def he_init(fan): return random.gauss(0, math.sqrt(2.0 / fan))

def make_net():
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H)]
    b1 = [0.0] * H
    W2 = [[he_init(H) for _ in range(H)] for _ in range(NUM_CLASSES)]
    b2 = [0.0] * NUM_CLASSES
    return W1, b1, W2, b2

def relu(z):   return max(0.0, z)
def relu_d(z): return 1.0 if z > 0 else 0.0

def forward(x, W1, b1, W2, b2):
    x_norm = [x[0] / 10.0, x[1] / 100.0]   # simple normalization
    z1 = [sum(W1[j][k] * x_norm[k] for k in range(2)) + b1[j] for j in range(H)]
    a1 = [relu(z) for z in z1]
    z2 = [sum(W2[c][k] * a1[k] for k in range(H)) + b2[c] for c in range(NUM_CLASSES)]
    probs = softmax(z2)
    return z1, a1, z2, probs


# ============================================================
# TRAINING WITH CCE LOSS
# ============================================================

W1, b1, W2, b2 = make_net()

print(f"\n{'=' * 60}")
print(f"  TRAINING MLP WITH CCE LOSS (LR={LR}, {EPOCHS} epochs)")
print(f"{'=' * 60}")

for epoch in range(1, EPOCHS + 1):
    all_probs = []
    for i in range(n):
        x, y = X[i], Y[i]
        z1, a1, z2, probs = forward(x, W1, b1, W2, b2)
        all_probs.append(probs)

        # Gradient of Softmax + CCE: dL/dz2 = (ŷ - y) / n  — that's it!
        y_oh = one_hot(y, NUM_CLASSES)
        dL_z2 = [(probs[c] - y_oh[c]) / n for c in range(NUM_CLASSES)]

        # Backprop to hidden layer
        dL_a1 = [sum(dL_z2[c] * W2[c][k] for c in range(NUM_CLASSES)) for k in range(H)]
        dL_z1 = [dL_a1[j] * relu_d(z1[j]) for j in range(H)]

        # Update W2
        for c in range(NUM_CLASSES):
            for k in range(H):
                W2[c][k] -= LR * dL_z2[c] * a1[k]
            b2[c] -= LR * dL_z2[c]

        # Update W1
        x_norm = [x[0] / 10.0, x[1] / 100.0]
        for j in range(H):
            for k in range(2):
                W1[j][k] -= LR * dL_z1[j] * x_norm[k]
            b1[j] -= LR * dL_z1[j]

    loss = cce_loss(all_probs, Y)
    acc = sum(1 for probs, t in zip(all_probs, Y) if probs.index(max(probs)) == t) / n

    if epoch % 100 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}  |  CCE: {loss:.4f}  |  Accuracy: {acc:.2%}")


# ============================================================
# FINAL PREDICTIONS
# ============================================================

print(f"\n{'=' * 60}")
print("  FINAL PREDICTIONS vs TARGETS")
print(f"{'=' * 60}")
class_names = ["A", "B", "C"]
all_probs_final = []
for i in range(n):
    _, _, _, probs = forward(X[i], W1, b1, W2, b2)
    all_probs_final.append(probs)
    pred_class = probs.index(max(probs))
    correct = "✓" if pred_class == Y[i] else "✗"
    probs_str = "  ".join(f"{class_names[c]}:{probs[c]:.2f}" for c in range(NUM_CLASSES))
    print(f"  {str(X[i]):>12}  |  True: {class_names[Y[i]]}  |  Pred: {class_names[pred_class]}  {correct}  |  [{probs_str}]")

final_loss = cce_loss(all_probs_final, Y)
final_acc = sum(1 for probs, t in zip(all_probs_final, Y) if probs.index(max(probs)) == t) / n
print(f"\n  Final CCE: {final_loss:.4f}  |  Accuracy: {final_acc:.2%}")
print(f"{'=' * 60}")
