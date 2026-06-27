# ============================================================
# GRADIENT CLIPPING — From Scratch
# ============================================================
# Prevents exploding gradients by limiting gradient magnitude.
# Two methods: clip by value, clip by norm.
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
# GRADIENT CLIPPING FUNCTIONS
# ================================================================

def clip_by_value(gradients, max_val):
    """Clip each gradient independently to [-max_val, max_val].

    Simple but can change gradient direction.
    Example: [10, -2, 5] with max_val=3 -> [3, -2, 3]
    """
    return [max(-max_val, min(max_val, g)) for g in gradients]


def clip_by_norm(gradients, max_norm):
    """Clip gradient vector by L2 norm.

    If ||g|| > max_norm, scale entire vector down proportionally.
    PRESERVES gradient direction (preferred method).

    Example: [3, 4] has norm=5. With max_norm=2.5:
             scale = 2.5/5 = 0.5 -> [1.5, 2.0]
    """
    norm = math.sqrt(sum(g ** 2 for g in gradients))
    if norm > max_norm:
        scale = max_norm / norm
        return [g * scale for g in gradients], norm
    return gradients[:], norm


# ================================================================
# DEMO: Show clipping in action
# ================================================================

print("=" * 60)
print("  GRADIENT CLIPPING DEMO")
print("=" * 60)

test_grads = [15.0, -8.0, 3.0, -20.0, 0.5, 12.0]
norm = math.sqrt(sum(g**2 for g in test_grads))

print(f"\n  Original gradients: {test_grads}")
print(f"  L2 Norm: {norm:.2f}")

# Clip by value
clipped_val = clip_by_value(test_grads, max_val=5.0)
print(f"\n  Clip by Value (max=5.0): {clipped_val}")
print(f"  Note: Direction changed! [15, -8] became [5, -5]")

# Clip by norm
clipped_norm, orig_norm = clip_by_norm(test_grads, max_norm=10.0)
print(f"\n  Clip by Norm (max=10.0): [{', '.join(f'{g:.2f}' for g in clipped_norm)}]")
print(f"  Scale factor: {10.0/orig_norm:.4f}")
print(f"  Direction preserved!")


# ================================================================
# TRAINING: With vs Without Gradient Clipping
# ================================================================

H1, H2 = 4, 4
EPOCHS = 100
LR = 0.01


def train_network(clip_method="none", clip_val=5.0):
    random.seed(42)
    # Deliberately large init to trigger exploding gradients
    W1 = [[random.uniform(-2, 2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[random.uniform(-2, 2) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[random.uniform(-2, 2) for _ in range(H2)]]
    b3 = [0.0]

    label = {"none": "No Clipping", "value": f"Clip by Value ({clip_val})",
             "norm": f"Clip by Norm ({clip_val})"}[clip_method]
    print(f"\n  {label}")
    print(f"  {'-' * 50}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        all_grads = []
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = y_hat - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            dW3 = [dL * a2[k] for k in range(H2)]
            dL_z2 = [dL * W3[0][j] * leaky_relu_derivative(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j] * x[k] for k in range(2)] for j in range(H1)]

            # Flatten all gradients
            flat_g = []
            for row in dW1: flat_g.extend(row)
            for row in dW2: flat_g.extend(row)
            flat_g.extend(dW3)
            flat_g.extend(dL_z1)
            flat_g.extend(dL_z2)
            flat_g.append(dL)

            # Apply clipping
            if clip_method == "value":
                flat_g = clip_by_value(flat_g, clip_val)
            elif clip_method == "norm":
                flat_g, _ = clip_by_norm(flat_g, clip_val)

            # Unflatten and update
            idx = 0
            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * flat_g[idx]; idx += 1
            for j in range(H2):
                for k in range(H1):
                    W2[j][k] -= LR * flat_g[idx]; idx += 1
            for k in range(H2):
                W3[0][k] -= LR * flat_g[idx]; idx += 1
            for j in range(H1):
                b1[j] -= LR * flat_g[idx]; idx += 1
            for j in range(H2):
                b2[j] -= LR * flat_g[idx]; idx += 1
            b3[0] -= LR * flat_g[idx]

        mse = total_loss / n
        # Check for NaN/Inf
        if math.isnan(mse) or math.isinf(mse):
            print(f"    Epoch {epoch:>3}  |  MSE: NaN/Inf -- EXPLODED!")
            losses.append(float('inf'))
            break

        losses.append(mse)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")

    return losses


print(f"\n{'=' * 60}")
print("  TRAINING (large init weights to trigger explosion)")
print(f"{'=' * 60}")

no_clip = train_network(clip_method="none")
val_clip = train_network(clip_method="value", clip_val=5.0)
norm_clip = train_network(clip_method="norm", clip_val=5.0)

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>20} {'No Clip':>10} {'By Value':>10} {'By Norm':>10}")
print(f"  {'-' * 50}")
f1 = f"{no_clip[-1]:.4f}" if not math.isinf(no_clip[-1]) else "EXPLODED"
f2 = f"{val_clip[-1]:.4f}" if not math.isinf(val_clip[-1]) else "EXPLODED"
f3 = f"{norm_clip[-1]:.4f}" if not math.isinf(norm_clip[-1]) else "EXPLODED"
print(f"  {'Final MSE':>20} {f1:>10} {f2:>10} {f3:>10}")
print(f"{'=' * 60}")
