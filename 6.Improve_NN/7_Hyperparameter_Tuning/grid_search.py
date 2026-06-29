# ============================================================
# GRID SEARCH — From Scratch
# ============================================================
# Tries multiple hyperparameter combinations (learning rates and
# hidden layer sizes) to find the one that minimizes MSE.
# Uses a 1-hidden-layer neural network trained on the CGPA dataset.
# ============================================================

import math, random
random.seed(42)

# Dataset: CGPA, Profile Score -> Package (in LPA)
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z): return z if z > 0 else 0.01 * z
def leaky_relu_deriv(z): return 1.0 if z > 0 else 0.01
def he_init(fan_in): return random.gauss(0, math.sqrt(2.0 / fan_in))

# Hyperparameter space to search
GRID_LR = [0.005, 0.01, 0.02]
GRID_HIDDEN = [2, 4, 6]
EPOCHS = 100

def train_and_evaluate(lr, hidden_dim):
    """Trains a 1-hidden-layer network and returns the final MSE."""
    random.seed(42)
    # Initialize parameters
    # W1: shape (hidden_dim, 2)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(hidden_dim)]
    b1 = [0.0] * hidden_dim
    # W2: shape (1, hidden_dim)
    W2 = [he_init(hidden_dim) for _ in range(hidden_dim)]
    b2 = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Accumulate gradients over the batch (full-batch gradient descent)
        dW1 = [[0.0 for _ in range(2)] for _ in range(hidden_dim)]
        db1 = [0.0] * hidden_dim
        dW2 = [0.0] * hidden_dim
        db2_val = 0.0
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]
            
            # Forward pass
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(hidden_dim)]
            a1 = [leaky_relu(z) for z in z1]
            yh = sum(W2[j] * a1[j] for j in range(hidden_dim)) + b2
            total_loss += (yh - y) ** 2

            # Backward pass
            dL = (2.0 / n) * (yh - y)
            
            # Gradients for W2 and b2
            for j in range(hidden_dim):
                dW2[j] += dL * a1[j]
            db2_val += dL

            # Gradients for W1 and b1
            dL_a1 = [dL * W2[j] for j in range(hidden_dim)]
            dL_z1 = [dL_a1[j] * leaky_relu_deriv(z1[j]) for j in range(hidden_dim)]
            for j in range(hidden_dim):
                for k in range(2):
                    dW1[j][k] += dL_z1[j] * x[k]
                db1[j] += dL_z1[j]

        # Parameter updates
        for j in range(hidden_dim):
            for k in range(2):
                W1[j][k] -= lr * dW1[j][k]
            b1[j] -= lr * db1[j]
            W2[j] -= lr * dW2[j]
        b2 -= lr * db2_val

    # Return final MSE
    return total_loss / n


print("=" * 60)
print("  GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 60)

results = []
best_mse = float('inf')
best_config = None

# Search grid
for lr in GRID_LR:
    for hidden in GRID_HIDDEN:
        mse = train_and_evaluate(lr, hidden)
        results.append((lr, hidden, mse))
        
        # Check if we have a new best configuration
        # Make sure it didn't diverge (not NaN)
        if not math.isnan(mse) and mse < best_mse:
            best_mse = mse
            best_config = (lr, hidden)

# Display results table
print(f"{'Learning Rate':<15} | {'Hidden Neurons':<15} | {'Final MSE':<12}")
print("-" * 50)
for lr, hidden, mse in results:
    mse_str = f"{mse:.4f}" if not math.isnan(mse) else "NaN (Diverged)"
    print(f"{lr:<15} | {hidden:<15} | {mse_str:<12}")

print("=" * 60)
if best_config:
    print(f"  BEST CONFIGURATION:")
    print(f"  Learning Rate:  {best_config[0]}")
    print(f"  Hidden Neurons: {best_config[1]}")
    print(f"  Lowest MSE:     {best_mse:.4f}")
else:
    print("  All configurations diverged.")
print("=" * 60)
