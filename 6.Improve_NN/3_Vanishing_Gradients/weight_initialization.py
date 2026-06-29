# ============================================================
# WEIGHT INITIALIZATION — From Scratch
# ============================================================
# Compares Random, Xavier, and He initialization by training
# the same network with each and observing convergence speed.
# ============================================================

import math
import random

X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z):
    return z if z > 0 else 0.01 * z
def leaky_relu_derivative(z):
    return 1.0 if z > 0 else 0.01


# ================================================================
# INITIALIZATION METHODS
# ================================================================

def random_init(fan_in, fan_out):
    """Random Uniform [-1, 1]. No consideration for layer size.
    Problem: variance explodes or vanishes across layers."""
    return random.uniform(-1, 1)

def xavier_init(fan_in, fan_out):
    """Xavier/Glorot: W ~ N(0, sqrt(2 / (fan_in + fan_out)))
    Designed for Sigmoid/Tanh.
    Keeps Var(activation) ~= Var(input) by balancing fan_in and fan_out."""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return random.gauss(0, std)

def he_init(fan_in, fan_out):
    """He/Kaiming: W ~ N(0, sqrt(2 / fan_in))
    Designed for ReLU (which kills half the neurons, so needs 2x variance).
    Only considers fan_in since ReLU zeroes half the outputs."""
    std = math.sqrt(2.0 / fan_in)
    return random.gauss(0, std)


# ================================================================
# SHOW INITIAL WEIGHT STATISTICS
# ================================================================

print("=" * 60)
print("  WEIGHT INITIALIZATION DEMO")
print("=" * 60)

fan_in, fan_out = 100, 50
num_samples = 1000

for name, init_fn in [("Random [-1,1]", random_init),
                       ("Xavier", xavier_init),
                       ("He", he_init)]:
    random.seed(42)
    weights = [init_fn(fan_in, fan_out) for _ in range(num_samples)]
    mean = sum(weights) / len(weights)
    var = sum((w - mean) ** 2 for w in weights) / len(weights)
    std = math.sqrt(var)
    min_w = min(weights)
    max_w = max(weights)

    print(f"\n  {name} (fan_in={fan_in}, fan_out={fan_out}):")
    print(f"    Mean: {mean:+.4f}  Std: {std:.4f}  Range: [{min_w:.3f}, {max_w:.3f}]")


# ================================================================
# ACTIVATION PROPAGATION: Watch variance across layers
# ================================================================

print(f"\n{'=' * 60}")
print("  FORWARD PROPAGATION: Activation variance across 10 layers")
print("  (256 neurons per layer, input = random N(0,1))")
print(f"{'=' * 60}")

num_layers = 10
layer_size = 256

for name, init_fn in [("Random [-1,1]", random_init),
                       ("Xavier", xavier_init),
                       ("He", he_init)]:
    random.seed(42)
    # Start with random input
    activations = [random.gauss(0, 1) for _ in range(layer_size)]

    print(f"\n  {name}:")
    print(f"    {'Layer':>7}  {'Mean':>10}  {'Variance':>10}  {'Status':>15}")
    print(f"    {'-' * 45}")

    for layer in range(num_layers):
        # Generate weights for this layer
        weights = [[init_fn(layer_size, layer_size) for _ in range(layer_size)]
                    for _ in range(layer_size)]

        # Forward: z = W * a, then activation
        new_activations = []
        for j in range(layer_size):
            z = sum(weights[j][k] * activations[k] for k in range(layer_size))
            new_activations.append(leaky_relu(z))

        activations = new_activations
        mean = sum(activations) / len(activations)
        var = sum((a - mean) ** 2 for a in activations) / len(activations)

        status = ""
        if var < 0.001:
            status = "VANISHING!"
        elif var > 100:
            status = "EXPLODING!"
        else:
            status = "Healthy"

        print(f"    {layer+1:>7}  {mean:>+10.4f}  {var:>10.4f}  {status:>15}")


# ================================================================
# TRAINING COMPARISON
# ================================================================

H1, H2 = 4, 4
EPOCHS = 100
LR = 0.01


def train_with_init(init_fn, name):
    random.seed(42)
    W1 = [[init_fn(2, H1) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[init_fn(H1, H2) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[init_fn(H2, 1) for _ in range(H2)]]
    b3 = [0.0]

    print(f"\n  {name}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
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
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_derivative(z1[j]) for j in range(H1)]

            for j in range(H1):
                for k in range(2): W1[j][k] -= LR * dL_z1[j] * x[k]
                b1[j] -= LR * dL_z1[j]
            for j in range(H2):
                for k in range(H1): W2[j][k] -= LR * dL_z2[j] * a1[k]
                b2[j] -= LR * dL_z2[j]
            for k in range(H2): W3[0][k] -= LR * dW3[k]
            b3[0] -= LR * dL

        mse = total_loss / n
        losses.append(mse)
        if epoch % 25 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")
    return losses


print(f"\n{'=' * 60}")
print("  TRAINING COMPARISON -- Same network, different init")
print(f"{'=' * 60}")

rand_loss = train_with_init(random_init, "Random Init")
xavier_loss = train_with_init(xavier_init, "Xavier Init")
he_loss = train_with_init(he_init, "He Init")

print(f"\n{'=' * 60}")
print("  FINAL MSE")
print(f"{'=' * 60}")
print(f"  Random:  {rand_loss[-1]:.4f}")
print(f"  Xavier:  {xavier_loss[-1]:.4f}")
print(f"  He:      {he_loss[-1]:.4f}")
print(f"{'=' * 60}")
