# ============================================================
# ACTIVATION FUNCTIONS — From Scratch
# ============================================================
# Compares Sigmoid, Tanh, ReLU, and Leaky ReLU by training
# the same network with each activation function.
#
# Shows how sigmoid causes vanishing gradients while ReLU
# variants solve it.
# ============================================================

import math
import random

# ---- Dataset ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)


# ================================================================
# ALL ACTIVATION FUNCTIONS
# ================================================================

def sigmoid(z):
    """Sigmoid: sigma(z) = 1 / (1 + e^-z)
    Output: (0, 1)
    Problem: Max derivative = 0.25 -> gradient vanishes after few layers!"""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_act(z):
    """Tanh: (e^z - e^-z) / (e^z + e^-z)
    Output: (-1, 1)
    Better than sigmoid (zero-centered), but derivative still < 1."""
    z = max(-500, min(500, z))
    return math.tanh(z)

def tanh_deriv(z):
    t = math.tanh(z)
    return 1.0 - t * t  # Max = 1.0 at z=0


def relu(z):
    """ReLU: f(z) = max(0, z)
    Output: [0, inf)
    Derivative = 1 for z > 0 -> NO vanishing gradient!
    Problem: 'Dying ReLU' if z is always negative."""
    return max(0.0, z)

def relu_deriv(z):
    return 1.0 if z > 0 else 0.0


def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: f(z) = z if z > 0, else alpha * z
    Fixes dying ReLU: small gradient for negative z."""
    return z if z > 0 else alpha * z

def leaky_relu_deriv(z, alpha=0.01):
    return 1.0 if z > 0 else alpha


# ================================================================
# SHOW ACTIVATION VALUES AND DERIVATIVES
# ================================================================

print("=" * 70)
print("  ACTIVATION FUNCTIONS -- Values & Derivatives at key points")
print("=" * 70)

test_z = [-3.0, -1.0, 0.0, 1.0, 3.0]

for name, fn, deriv in [("Sigmoid", sigmoid, sigmoid_deriv),
                         ("Tanh", tanh_act, tanh_deriv),
                         ("ReLU", relu, relu_deriv),
                         ("Leaky ReLU", leaky_relu, leaky_relu_deriv)]:
    print(f"\n  {name}:")
    print(f"    {'z':>6}  {'f(z)':>10}  {'f_prime(z)':>12}")
    print(f"    {'-' * 30}")
    for z in test_z:
        print(f"    {z:>6.1f}  {fn(z):>10.4f}  {deriv(z):>12.4f}")


# ================================================================
# GRADIENT VANISHING DEMO: Chain multiple derivatives
# ================================================================

print(f"\n{'=' * 70}")
print("  GRADIENT VANISHING DEMO -- 10 layers of chain rule")
print("=" * 70)

for name, deriv_fn, test_val in [("Sigmoid (z=0)", sigmoid_deriv, 0.0),
                                  ("Tanh (z=0)", tanh_deriv, 0.0),
                                  ("ReLU (z=1)", relu_deriv, 1.0),
                                  ("Leaky ReLU (z=1)", leaky_relu_deriv, 1.0)]:
    d = deriv_fn(test_val)
    grad_10 = d ** 10
    print(f"\n  {name}:")
    print(f"    Single layer derivative: {d:.4f}")
    print(f"    After 10 layers: {d:.4f}^10 = {grad_10:.10f}")
    if grad_10 < 0.001:
        print(f"    --> VANISHING! Gradient is practically zero.")
    else:
        print(f"    --> HEALTHY! Gradient flows through.")


# ================================================================
# TRAINING COMPARISON
# ================================================================

H1, H2 = 4, 4
EPOCHS = 100
LR = 0.01


def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


def train_with_activation(act_fn, act_deriv, name):
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n  {name}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for i in range(n):
            x, y = X[i], Y[i]
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [act_fn(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [act_fn(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]
            error = y_hat - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            dW3 = [dL * a2[k] for k in range(H2)]
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * act_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * act_deriv(z1[j]) for j in range(H1)]

            for j in range(H1):
                for k in range(2): W1[j][k] -= LR * dL_z1[j] * x[k]
                b1[j] -= LR * dL_z1[j]
            for j in range(H2):
                for k in range(H1): W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * dL_z2[j]
            for k in range(H2): W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * dL

        mse = total_loss / n
        losses.append(mse)
        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")

    return losses


print(f"\n{'=' * 70}")
print("  TRAINING COMPARISON -- Same network, different activations")
print(f"{'=' * 70}")

sig_loss = train_with_activation(sigmoid, sigmoid_deriv, "Sigmoid")
tanh_loss = train_with_activation(tanh_act, tanh_deriv, "Tanh")
relu_loss = train_with_activation(relu, relu_deriv, "ReLU")
lrelu_loss = train_with_activation(leaky_relu, leaky_relu_deriv, "Leaky ReLU")

print(f"\n{'=' * 70}")
print("  FINAL MSE COMPARISON")
print(f"{'=' * 70}")
print(f"  Sigmoid:     {sig_loss[-1]:.4f}")
print(f"  Tanh:        {tanh_loss[-1]:.4f}")
print(f"  ReLU:        {relu_loss[-1]:.4f}")
print(f"  Leaky ReLU:  {lrelu_loss[-1]:.4f}")
print(f"\n  Winner: {'Leaky ReLU' if lrelu_loss[-1] == min(sig_loss[-1], tanh_loss[-1], relu_loss[-1], lrelu_loss[-1]) else 'ReLU'}")
print(f"{'=' * 70}")
