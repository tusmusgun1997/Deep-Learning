# ============================================================
# ELU — From Scratch Implementation
# ============================================================
# f(z) = z if z > 0 else alpha * (e^z - 1)
# f'(z) = 1 if z > 0 else alpha * e^z
#
# Trains a 2-hidden-layer MLP using ONLY ELU activation
# on the CGPA dataset, and demonstrates recovery of "dead" units.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Package (LPA) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

ALPHA = 1.0  # ELU hyperparameter (default is 1.0)

# ============================================================
# ELU FUNCTION + DERIVATIVE
# ============================================================

def elu(z, alpha=ALPHA):
    """f(z) = z if z > 0 else alpha * (e^z - 1). Output: (-alpha, inf)"""
    if z > 0.0:
        return z
    else:
        # Clip z to prevent underflow/overflow in exp
        z = max(-500, min(500, z))
        return alpha * (math.exp(z) - 1.0)

def elu_deriv(z, alpha=ALPHA):
    """f'(z) = 1.0 if z > 0 else alpha * e^z"""
    if z > 0.0:
        return 1.0
    else:
        z = max(-500, min(500, z))
        return alpha * math.exp(z)


# ============================================================
# VALUE & DERIVATIVE TABLE
# ============================================================

print("=" * 60)
print("  ELU — Value & Derivative Table")
print("=" * 60)
col1 = "z"
col2 = "elu(z)"
col3 = "elu'(z)"
print(f"  {col1:>6}  |  {col2:>10}  |  {col3:>10}")
print(f"  {'-' * 40}")

for z in [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0]:
    print(f"  {z:>6.1f}  |  {elu(z):>10.6f}  |  {elu_deriv(z):>10.6f}")

print(f"\n  NOTE: Smooth transition around z = 0. f'(0) = 1.0 from both sides.")
print(f"  Negative inputs are not zeroed out completely, allowing recovery.")


# ============================================================
# ELU RECOVERY DEMO
# ============================================================

print(f"\n{'=' * 60}")
print("  ELU RECOVERY DEMO")
print("=" * 60)
print("  Let's simulate a neuron starting with negative weights and bias")
print("  (just like in the ReLU/Leaky ReLU demos) and see how ELU helps it recover.")

# Initial weights & bias for a single simulated neuron
w = [ -0.1, -0.2 ]
b = -0.5
print(f"  Initial Neuron: weights={w}, bias={b}")

# Try 5 update steps where inputs are positive but because weights/bias are negative, pre-activation z < 0
lr = 0.1
for step in range(1, 6):
    # Simulated input (positive)
    x = [8.0, 4.0]
    # Simulated incoming gradient from downstream layers
    dL_da = -5.0

    z = w[0]*x[0] + w[1]*x[1] + b
    a = elu(z)
    deriv = elu_deriv(z)
    dL_dz = dL_da * deriv # will be non-zero since deriv is alpha * e^z

    # Weight updates
    w[0] -= lr * dL_dz * x[0]
    w[1] -= lr * dL_dz * x[1]
    b -= lr * dL_dz

    print(f"  Step {step}: input={x} -> z={z:>5.1f} -> a={a:>5.2f} -> deriv={deriv:.4f} -> weights={[round(val, 4) for val in w]}, bias={round(b, 4)}")

print("  Observation: The weights and bias DO update because the gradient is non-zero (exponentials stay alive)!")
print("  The neuron is active and learning.")


# ============================================================
# TRAINING: 2-Hidden-Layer MLP with ELU
# ============================================================

H1, H2 = 4, 4
EPOCHS = 200
LR = 0.01

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def train_elu():
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n{'=' * 60}")
    print(f"  TRAINING MLP WITH ELU (LR={LR}, {EPOCHS} epochs)")
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
            a1 = [elu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [elu(z) for z in z2]
            yh = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = yh - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Backward pass
            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * elu_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * elu_deriv(z1[j]) for j in range(H1)]

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


losses, grads = train_elu()

print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")
print(f"  Final MSE: {losses[-1]:.4f}")
print(f"  Final Avg Layer-1 gradient: {grads[-1]:.6f}")
print(f"\n  Observation: ELU converges smoothly and provides the same gradient-passing")
print(f"  benefits as ReLU while maintaining zero-centered activations.")
print(f"{'=' * 60}")
