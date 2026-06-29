# ============================================================
# GRADIENT CHECKING — Verify Backpropagation Is Correct
# ============================================================
# Compares analytical gradients (from backprop) with numerical
# gradients (from finite differences) to catch bugs.
#
# numerical_grad = (L(w+eps) - L(w-eps)) / (2*eps)
# ============================================================

import math
import random

random.seed(42)

X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z):
    return z if z > 0 else 0.01 * z
def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01
def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


# ================================================================
# SIMPLE NETWORK: 2 -> 3 -> 1
# ================================================================

H = 3

# Initialize
random.seed(42)
W1 = [[he_init(2) for _ in range(2)] for _ in range(H)]
b1 = [0.0] * H
W2 = [[he_init(H) for _ in range(H)]]
b2 = [0.0]


def flatten_params():
    flat = []
    for row in W1: flat.extend(row)
    flat.extend(W2[0])
    flat.extend(b1)
    flat.extend(b2)
    return flat


def unflatten_params(flat):
    idx = 0
    w1 = [[flat[idx + j*2 + k] for k in range(2)] for j in range(H)]
    idx += H * 2
    w2 = [flat[idx:idx+H]]
    idx += H
    b_1 = flat[idx:idx+H]
    idx += H
    b_2 = flat[idx:idx+1]
    return w1, b_1, w2, b_2


def compute_loss(params):
    """Compute MSE loss for given parameters."""
    w1, b_1, w2, b_2 = unflatten_params(params)
    total = 0.0
    for i in range(n):
        x = X[i]
        z1 = [sum(w1[j][k] * x[k] for k in range(2)) + b_1[j] for j in range(H)]
        a1 = [leaky_relu(z) for z in z1]
        y_hat = sum(w2[0][k] * a1[k] for k in range(H)) + b_2[0]
        total += (y_hat - Y[i]) ** 2
    return total / n


def compute_analytical_gradients():
    """Compute gradients using backpropagation."""
    dW1 = [[0.0] * 2 for _ in range(H)]
    db1 = [0.0] * H
    dW2 = [[0.0] * H]
    db2 = [0.0]

    for i in range(n):
        x, y = X[i], Y[i]
        z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H)]
        a1 = [leaky_relu(z) for z in z1]
        y_hat = sum(W2[0][k] * a1[k] for k in range(H)) + b2[0]

        dL = (2.0 / n) * (y_hat - y)

        for k in range(H):
            dW2[0][k] += dL * a1[k]
        db2[0] += dL

        dL_a1 = [dL * W2[0][k] for k in range(H)]
        dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H)]

        for j in range(H):
            for k in range(2):
                dW1[j][k] += dL_z1[j] * x[k]
            db1[j] += dL_z1[j]

    # Flatten gradients in same order as params
    flat = []
    for row in dW1: flat.extend(row)
    flat.extend(dW2[0])
    flat.extend(db1)
    flat.extend(db2)
    return flat


# ================================================================
# NUMERICAL GRADIENT (FINITE DIFFERENCES)
# ================================================================

def numerical_gradient(params, idx, epsilon=1e-7):
    """Two-sided finite difference for parameter at index idx.

    numerical_grad = (L(w+eps) - L(w-eps)) / (2*eps)

    This is slow (2 forward passes) but extremely accurate.
    """
    params_plus = params[:]
    params_plus[idx] += epsilon
    loss_plus = compute_loss(params_plus)

    params_minus = params[:]
    params_minus[idx] -= epsilon
    loss_minus = compute_loss(params_minus)

    return (loss_plus - loss_minus) / (2 * epsilon)


# ================================================================
# RUN GRADIENT CHECK
# ================================================================

print("=" * 65)
print("  GRADIENT CHECKING DEMO")
print("  Network: 2 -> 3 -> 1 (Leaky ReLU + Linear)")
print("=" * 65)

params = flatten_params()
analytical = compute_analytical_gradients()

print(f"\n  Checking all {len(params)} parameters:")
print(f"  {'#':>4}  {'Analytical':>14}  {'Numerical':>14}  {'Rel Error':>12}  {'Status':>8}")
print(f"  {'-' * 58}")

max_error = 0
all_ok = True

for i in range(len(params)):
    num = numerical_gradient(params, i)
    ana = analytical[i]

    denom = abs(ana) + abs(num) + 1e-10
    rel_error = abs(ana - num) / denom
    max_error = max(max_error, rel_error)

    if rel_error < 1e-5:
        status = "OK"
    elif rel_error < 1e-3:
        status = "WARN"
    else:
        status = "BUG!"
        all_ok = False

    print(f"  {i:>4}  {ana:>14.8f}  {num:>14.8f}  {rel_error:>12.2e}  {status:>8}")

print(f"\n  Max relative error: {max_error:.2e}")
if all_ok:
    print("  RESULT: Backpropagation implementation is CORRECT!")
else:
    print("  RESULT: WARNING -- Some gradients may have errors!")

print(f"\n  Interpretation guide:")
print(f"    < 1e-7:  Perfect match")
print(f"    < 1e-5:  Correct implementation")
print(f"    < 1e-3:  Suspicious, double-check")
print(f"    > 1e-3:  Bug in backpropagation!")
print(f"{'=' * 65}")
