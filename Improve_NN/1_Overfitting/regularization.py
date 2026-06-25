# ============================================================
# L1 & L2 REGULARIZATION — From Scratch
# ============================================================
# Adds a penalty to the loss function for large weights.
# This forces the model to keep weights small -> simpler model
# -> less overfitting.
#
# L2 (Ridge): Loss += lambda * sum(w^2)     -> shrinks weights
# L1 (Lasso): Loss += lambda * sum(|w|)     -> zeros out weights
# ============================================================

import math
import random

random.seed(42)

# ---- Dataset ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ---- Activation ----
def leaky_relu(z):
    return z if z > 0 else 0.01 * z

def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


# ================================================================
# REGULARIZATION FUNCTIONS
# ================================================================

def l2_penalty(weights, lambda_val):
    """L2 penalty: lambda * sum(w^2)
    Adds to the total loss. Makes loss higher when weights are large."""
    return lambda_val * sum(w ** 2 for w in weights)

def l2_gradient(w, lambda_val):
    """L2 gradient for one weight: 2 * lambda * w
    This is ADDED to the normal backprop gradient.

    Effect: w = w - lr * (dL/dw + 2*lambda*w)
          = w * (1 - 2*lr*lambda) - lr * dL/dw
                 ^^^^^^^^^^^^^^^^
                 "weight decay" -- weights shrink each step
    """
    return 2.0 * lambda_val * w

def l1_penalty(weights, lambda_val):
    """L1 penalty: lambda * sum(|w|)
    Drives weights to EXACTLY zero (sparsity / feature selection)."""
    return lambda_val * sum(abs(w) for w in weights)

def l1_gradient(w, lambda_val):
    """L1 gradient for one weight: lambda * sign(w)
    Constant push toward zero regardless of weight magnitude."""
    if w > 0:
        return lambda_val
    elif w < 0:
        return -lambda_val
    return 0.0


# ================================================================
# NETWORK: 2 -> 4 -> 4 -> 1
# ================================================================

H1, H2 = 4, 4
EPOCHS = 100
LR = 0.01


def train_network(reg_type="none", lambda_val=0.0):
    """Train with different regularization types.

    reg_type: 'none', 'l1', 'l2'
    lambda_val: regularization strength
    """
    random.seed(42)

    # Initialize weights
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    label = f"{reg_type.upper()} (lambda={lambda_val})" if reg_type != "none" else "No Regularization"
    print(f"\n  Training: {label}")
    print(f"  {'-' * 50}")

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for i in range(n):
            x = X[i]
            y = Y[i]

            # Forward
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = y_hat - y
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Backward (output)
            dW3 = [dL * a2[k] for k in range(H2)]
            db3_g = dL

            # Backward (hidden 2)
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            dL_z2 = [dL_a2[j] * leaky_relu_derivative(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            db2_g = dL_z2[:]

            # Backward (hidden 1)
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j] * x[k] for k in range(2)] for j in range(H1)]
            db1_g = dL_z1[:]

            # ADD REGULARIZATION GRADIENTS to weight gradients
            if reg_type == "l2":
                for j in range(H1):
                    for k in range(2):
                        dW1[j][k] += l2_gradient(W1[j][k], lambda_val)
                for j in range(H2):
                    for k in range(H1):
                        dW2[j][k] += l2_gradient(W2[j][k], lambda_val)
                for k in range(H2):
                    dW3[k] += l2_gradient(W3[0][k], lambda_val)

            elif reg_type == "l1":
                for j in range(H1):
                    for k in range(2):
                        dW1[j][k] += l1_gradient(W1[j][k], lambda_val)
                for j in range(H2):
                    for k in range(H1):
                        dW2[j][k] += l1_gradient(W2[j][k], lambda_val)
                for k in range(H2):
                    dW3[k] += l1_gradient(W3[0][k], lambda_val)

            # Update
            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * dW1[j][k]
                b1[j] -= LR * db1_g[j]
            for j in range(H2):
                for k in range(H1):
                    W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * db2_g[j]
            for k in range(H2):
                W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * db3_g

        mse = total_loss / n

        # Add regularization to reported loss
        all_w = []
        for row in W1: all_w.extend(row)
        for row in W2: all_w.extend(row)
        all_w.extend(W3[0])

        if reg_type == "l2":
            mse += l2_penalty(all_w, lambda_val)
        elif reg_type == "l1":
            mse += l1_penalty(all_w, lambda_val)

        loss_history.append(mse)

        if epoch % 20 == 0 or epoch == 1:
            # Count near-zero weights (L1 sparsity check)
            near_zero = sum(1 for w in all_w if abs(w) < 0.01)
            max_w = max(abs(w) for w in all_w)
            print(f"    Epoch {epoch:>3}  |  Loss: {mse:.4f}  |  "
                  f"Max |w|: {max_w:.4f}  |  Near-zero weights: {near_zero}/{len(all_w)}")

    return loss_history


# ================================================================
# RUN COMPARISON
# ================================================================

print("=" * 60)
print("  L1 & L2 REGULARIZATION DEMO")
print("  Network: 2 -> 4 -> 4 -> 1")
print("=" * 60)

no_reg = train_network(reg_type="none")
l2_reg = train_network(reg_type="l2", lambda_val=0.01)
l1_reg = train_network(reg_type="l1", lambda_val=0.01)

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>20} {'No Reg':>10} {'L2':>10} {'L1':>10}")
print(f"  {'-' * 50}")
print(f"  {'Final Loss':>20} {no_reg[-1]:>10.4f} {l2_reg[-1]:>10.4f} {l1_reg[-1]:>10.4f}")
print(f"{'=' * 60}")
print("\n  L2: Weights stay SMALL (weight decay)")
print("  L1: Some weights go to ZERO (sparsity / feature selection)")
