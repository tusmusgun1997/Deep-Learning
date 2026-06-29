# ============================================================
# SIGMOID — From Scratch Implementation
# ============================================================
# σ(z) = 1 / (1 + e^-z)
# σ'(z) = σ(z) * (1 - σ(z))
#
# Trains a 2-hidden-layer MLP using ONLY sigmoid activation
# on the CGPA dataset, showing its convergence behavior
# and the vanishing gradient problem.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Package (LPA) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ============================================================
# SIGMOID FUNCTION + DERIVATIVE
# ============================================================

def sigmoid(z):
    """σ(z) = 1 / (1 + e^-z).  Output: (0, 1)"""
    z = max(-500, min(500, z))  # clip to prevent overflow
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_deriv(z):
    """σ'(z) = σ(z) * (1 - σ(z)).  Max value = 0.25 at z=0"""
    s = sigmoid(z)
    return s * (1.0 - s)


# ============================================================
# VALUE & DERIVATIVE TABLE
# ============================================================

print("=" * 60)
print("  SIGMOID — Value & Derivative Table")
print("=" * 60)
col1 = "z"
col2 = "sig(z)"
col3 = "sig'(z)"
print(f"  {col1:>6}  |  {col2:>10}  |  {col3:>10}")
print(f"  {'-' * 40}")

for z in [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
    print(f"  {z:>6.1f}  |  {sigmoid(z):>10.6f}  |  {sigmoid_deriv(z):>10.6f}")

print(f"\n  NOTE: Max derivative is 0.25 (at z=0).")
print(f"  After 10 layers: 0.25^10 = {0.25**10:.10f}  --> VANISHES!")


# ============================================================
# VANISHING GRADIENT DEMO — Chain through layers
# ============================================================

print(f"\n{'=' * 60}")
print("  VANISHING GRADIENT CHAIN DEMO")
print("=" * 60)

grad = 1.0
print(f"  Starting gradient: {grad:.6f}")
for layer in range(1, 11):
    grad *= sigmoid_deriv(0.0)  # best case: z=0, derivative=0.25
    print(f"  After layer {layer:>2}: {grad:.10f}" +
          (" <-- practically zero!" if grad < 0.001 else ""))


# ============================================================
# TRAINING: 2-Hidden-Layer MLP with Sigmoid
# ============================================================

H1, H2 = 4, 4
EPOCHS = 200
LR = 0.5  # Sigmoid needs higher LR because gradients are tiny

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def train_sigmoid():
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING MLP WITH SIGMOID (LR={LR}, {EPOCHS} epochs)")
    print(f"{'=' * 60}")
    losses = []
    grad_magnitudes = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        epoch_grad_sum = 0.0

        for i in range(n):
            x, y = X[i], Y[i]

            # Forward pass
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [sigmoid(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [sigmoid(z) for z in z2]
            yh = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = yh - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Backward pass
            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * sigmoid_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * sigmoid_deriv(z1[j]) for j in range(H1)]

            # Track gradient magnitude at layer 1 (to show vanishing)
            epoch_grad_sum += sum(abs(g) for g in dL_z1)

            # Update weights
            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * dL_z1[j] * x[k]
                b1[j] -= LR * dL_z1[j]
            for j in range(H2):
                for k in range(H1):
                    W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * dL_z2[j]
            for k in range(H2):
                W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * dL

        mse = total_loss / n
        losses.append(mse)
        avg_grad = epoch_grad_sum / (n * H1)
        grad_magnitudes.append(avg_grad)

        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}  |  MSE: {mse:.4f}  |  Avg Layer-1 |grad|: {avg_grad:.6f}")

    return losses, grad_magnitudes


losses, grads = train_sigmoid()

print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")
print(f"  Final MSE: {losses[-1]:.4f}")
print(f"  Final Avg Layer-1 gradient: {grads[-1]:.6f}")
print(f"\n  Observation: Sigmoid's tiny gradients make layer-1")
print(f"  learn VERY slowly. This is why ReLU replaced it for")
print(f"  hidden layers in modern networks.")
print(f"{'=' * 60}")
