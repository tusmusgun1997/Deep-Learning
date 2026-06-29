# ============================================================
# DROPOUT — From Scratch
# ============================================================
# Dropout randomly disables neurons during training to prevent
# overfitting. This file demonstrates the difference between
# training WITH and WITHOUT dropout.
#
# Uses "Inverted Dropout":
#   Training:  Zero out neurons with prob p, scale up survivors by 1/(1-p)
#   Inference: Use all neurons, no scaling needed
# ============================================================

import math
import random

random.seed(42)

# ---- Dataset (same as MLP project) ----
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# ---- Activation ----
def leaky_relu(z):
    return z if z > 0 else 0.01 * z

def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01

# ---- Weight Init (He) ----
def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))


# ================================================================
# DROPOUT IMPLEMENTATION
# ================================================================

def apply_dropout(activations, drop_rate, training=True):
    """Inverted Dropout.

    During TRAINING:
      - Each neuron is dropped (zeroed) with probability = drop_rate
      - Surviving neurons are scaled up by 1/(1-drop_rate)
      - This keeps the expected output value the same

    During INFERENCE:
      - All neurons are active, no scaling needed

    Example (drop_rate=0.5):
      Input:   [0.8,  0.3,  0.5,  0.9]
      Mask:    [ 1,    0,    1,    0 ]     (random)
      Output:  [1.6,  0.0,  1.0,  0.0]    (survivors scaled by 2x)

    Args:
        activations: list of neuron outputs
        drop_rate: probability of dropping each neuron (0.0 to 1.0)
        training: if False, skip dropout entirely
    Returns:
        (output_activations, mask)
    """
    if not training or drop_rate == 0:
        return activations[:], [1] * len(activations)

    scale = 1.0 / (1.0 - drop_rate)
    output = []
    mask = []

    for a in activations:
        if random.random() > drop_rate:
            # KEEP this neuron (scaled up)
            output.append(a * scale)
            mask.append(1)
        else:
            # DROP this neuron
            output.append(0.0)
            mask.append(0)

    return output, mask


# ================================================================
# NETWORK: 2 inputs -> 8 hidden -> 8 hidden -> 1 output
# (Deliberately large to show overfitting)
# ================================================================

H1, H2 = 8, 8
EPOCHS = 100
LR = 0.01


def init_weights():
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]
    return W1, b1, W2, b2, W3, b3


def train_network(drop_rate=0.0, label=""):
    """Train with a given dropout rate and return loss history."""
    random.seed(42)
    W1, b1, W2, b2, W3, b3 = init_weights()

    print(f"\n  Training with dropout = {drop_rate} ({label})")
    print(f"  {'-' * 50}")
    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for i in range(n):
            x1, x2, y = X[i][0], X[i][1], Y[i]

            # ---- FORWARD PASS ----
            # Hidden 1
            z1 = [sum(W1[j][k] * [x1, x2][k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            a1, mask1 = apply_dropout(a1, drop_rate, training=True)

            # Hidden 2
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            a2, mask2 = apply_dropout(a2, drop_rate, training=True)

            # Output (linear)
            y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]

            error = y_hat - y
            total_loss += error ** 2

            # ---- BACKWARD PASS ----
            dL = (2.0 / n) * error

            # Output gradients
            dW3 = [dL * a2[k] for k in range(H2)]
            db3 = dL

            # Hidden 2 gradients
            dL_a2 = [dL * W3[0][k] for k in range(H2)]
            # Apply dropout mask (no gradient through dropped neurons)
            dL_a2 = [dL_a2[k] * mask2[k] * (1.0 / (1.0 - drop_rate) if drop_rate > 0 else 1.0)
                      for k in range(H2)]
            dL_z2 = [dL_a2[j] * leaky_relu_derivative(z2[j]) for j in range(H2)]

            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            db2_g = dL_z2[:]

            # Hidden 1 gradients
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_a1 = [dL_a1[k] * mask1[k] * (1.0 / (1.0 - drop_rate) if drop_rate > 0 else 1.0)
                      for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]

            dW1 = [[dL_z1[j] * [x1, x2][k] for k in range(2)] for j in range(H1)]
            db1_g = dL_z1[:]

            # ---- UPDATE WEIGHTS ----
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
            b3[0] -= LR * db3

        mse = total_loss / n
        loss_history.append(mse)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")

    # Final predictions (inference mode -- no dropout)
    print(f"\n    Final Predictions (dropout OFF during inference):")
    print(f"    {'CGPA':>6} {'Prof':>5} {'Actual':>7} {'Pred':>8} {'Error':>8}")
    final_mse = 0.0
    for i in range(n):
        x1, x2, y = X[i][0], X[i][1], Y[i]
        z1 = [sum(W1[j][k] * [x1, x2][k] for k in range(2)) + b1[j] for j in range(H1)]
        a1 = [leaky_relu(z) for z in z1]
        # NO dropout during inference
        z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
        a2 = [leaky_relu(z) for z in z2]
        y_hat = sum(W3[0][k] * a2[k] for k in range(H2)) + b3[0]
        err = y_hat - y
        final_mse += err ** 2
        print(f"    {x1:>6} {x2:>5} {y:>7} {y_hat:>8.2f} {err:>+8.2f}")
    print(f"    Final MSE: {final_mse / n:.4f}")

    return loss_history


# ================================================================
# RUN COMPARISON
# ================================================================

print("=" * 60)
print("  DROPOUT DEMO -- With vs Without")
print("  Network: 2 -> 8 -> 8 -> 1 (deliberately overparameterized)")
print("=" * 60)

no_drop = train_network(drop_rate=0.0, label="NO Dropout")
with_drop = train_network(drop_rate=0.3, label="Dropout = 0.3")

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>25} {'No Dropout':>12} {'Dropout 0.3':>12}")
print(f"  {'-' * 50}")
print(f"  {'Final MSE':>25} {no_drop[-1]:>12.4f} {with_drop[-1]:>12.4f}")
print(f"  {'Best MSE':>25} {min(no_drop):>12.4f} {min(with_drop):>12.4f}")
print(f"{'=' * 60}")
print("\n  Key takeaway: Dropout forces the network to learn")
print("  redundant representations, making it more robust.")
print("  The effect is most visible with a separate test set.")
