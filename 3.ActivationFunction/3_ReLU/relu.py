# ============================================================
# RELU — From Scratch Implementation
# ============================================================
# f(z) = max(0, z)
# f'(z) = 1 if z > 0 else 0
#
# Trains a 2-hidden-layer MLP using ONLY ReLU activation
# on the CGPA dataset, and demonstrates the Dying ReLU issue.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Package (LPA) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ============================================================
# RELU FUNCTION + DERIVATIVE
# ============================================================

def relu(z):
    """f(z) = max(0, z). Output: [0, inf)"""
    return max(0.0, z)

def relu_deriv(z):
    """f'(z) = 1.0 if z > 0 else 0.0"""
    return 1.0 if z > 0.0 else 0.0


# ============================================================
# VALUE & DERIVATIVE TABLE
# ============================================================

print("=" * 60)
print("  RELU — Value & Derivative Table")
print("=" * 60)
col1 = "z"
col2 = "relu(z)"
col3 = "relu'(z)"
print(f"  {col1:>6}  |  {col2:>10}  |  {col3:>10}")
print(f"  {'-' * 40}")

for z in [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
    print(f"  {z:>6.1f}  |  {relu(z):>10.6f}  |  {relu_deriv(z):>10.6f}")

print(f"\n  NOTE: Derivative is exactly 1.0 for z > 0, preventing vanishing gradients!")
print(f"  However, it is exactly 0.0 for z <= 0, which can lead to dead neurons.")


# ============================================================
# DYING RELU DEMO
# ============================================================

print(f"\n{'=' * 60}")
print("  DYING RELU DEMO")
print("=" * 60)
print("  Let's simulate a neuron that has received weights that make it ")
print("  output negative values for all inputs. we try to update it.")

# Initial weights & bias for a single simulated neuron
w = [ -1.5, -2.0 ]
b = -1.0
print(f"  Initial Neuron: weights={w}, bias={b}")

# Try 5 update steps where inputs are positive but because weights/bias are negative, pre-activation z < 0
lr = 0.1
for step in range(1, 6):
    # Simulated input (positive)
    x = [8.0, 4.0]
    # Simulated incoming gradient from downstream layers
    dL_da = -5.0

    z = w[0]*x[0] + w[1]*x[1] + b
    a = relu(z)
    deriv = relu_deriv(z)
    dL_dz = dL_da * deriv # will be 0 since z < 0

    # Weight updates
    w[0] -= lr * dL_dz * x[0]
    w[1] -= lr * dL_dz * x[1]
    b -= lr * dL_dz

    print(f"  Step {step}: input={x} -> z={z:>5.1f} -> a={a:.1f} -> deriv={deriv:.1f} -> weights={w}, bias={b}")

print("  Observation: The weights and bias NEVER change because the derivative is 0!")
print("  The neuron is dead.")


# ============================================================
# TRAINING: 2-Hidden-Layer MLP with ReLU
# ============================================================

H1, H2 = 4, 4
EPOCHS = 200
LR = 0.01  # ReLU converges fast, high LR might cause dying ReLUs

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def train_relu():
    random.seed(42)
    # He initialization is critical for ReLU to prevent starting with dead neurons!
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING MLP WITH RELU (LR={LR}, {EPOCHS} epochs)")
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
            a1 = [relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [relu(z) for z in z2]
            yh = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = yh - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Backward pass
            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * relu_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * relu_deriv(z1[j]) for j in range(H1)]

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


losses, grads = train_relu()

print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")
print(f"  Final MSE: {losses[-1]:.4f}")
print(f"  Final Avg Layer-1 gradient: {grads[-1]:.6f}")
print(f"\n  Observation: ReLU converges to a very low MSE because there")
print(f"  is no vanishing gradient. The layer-1 gradients remain active.")
print(f"{'=' * 60}")
