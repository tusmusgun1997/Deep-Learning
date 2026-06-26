# ============================================================
# SWISH — From Scratch Implementation
# ============================================================
# f(z) = z * sigmoid(z)
# f'(z) = f(z) + sigmoid(z) * (1 - f(z))
#
# Trains a 2-hidden-layer MLP using ONLY Swish activation
# on the CGPA dataset, and shows its convergence behavior.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Package (LPA) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ============================================================
# SWISH FUNCTION + DERIVATIVE
# ============================================================

def sigmoid(z):
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def swish(z, beta=1.0):
    """f(z) = z * sigmoid(beta * z). Output: [-0.28, inf)"""
    return z * sigmoid(beta * z)

def swish_deriv(z, beta=1.0):
    """f'(z) = beta * f(z) + sigmoid(beta * z) * (1 - beta * f(z))"""
    s = sigmoid(beta * z)
    sz = swish(z, beta)
    return beta * sz + s * (1.0 - beta * sz)


# ============================================================
# VALUE & DERIVATIVE TABLE
# ============================================================

print("=" * 60)
print("  SWISH — Value & Derivative Table")
print("=" * 60)
col1 = "z"
col2 = "swish(z)"
col3 = "swish'(z)"
print(f"  {col1:>6}  |  {col2:>10}  |  {col3:>10}")
print(f"  {'-' * 40}")

for z in [-5.0, -3.0, -1.28, -1.0, 0.0, 1.0, 3.0, 5.0]:
    print(f"  {z:>6.2f}  |  {swish(z):>10.6f}  |  {swish_deriv(z):>10.6f}")

print(f"\n  NOTE: At z = -1.28, swish(z) = {swish(-1.28):.4f} (local minimum/dip).")
print(f"  This is the non-monotonic nature of Swish in action.")


# ============================================================
# TRAINING: 2-Hidden-Layer MLP with Swish
# ============================================================

H1, H2 = 4, 4
EPOCHS = 200
LR = 0.01

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def train_swish():
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING MLP WITH SWISH (LR={LR}, {EPOCHS} epochs)")
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
            a1 = [swish(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [swish(z) for z in z2]
            yh = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = yh - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Backward pass
            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * swish_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * swish_deriv(z1[j]) for j in range(H1)]

            # Track gradient magnitude at layer 1
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


losses, grads = train_swish()

print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")
print(f"  Final MSE: {losses[-1]:.4f}")
print(f"  Final Avg Layer-1 gradient: {grads[-1]:.6f}")
print(f"\n  Observation: Swish converges smoothly. Gradients are active")
print(f"  and do not die even for negative inputs, showing its stability.")
print(f"{'=' * 60}")
