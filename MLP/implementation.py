# ============================================================
# Multi-Layer Perceptron from Scratch — Pure Python
# Regression: Predict student package (LPA) from CGPA & Profile
# No libraries used (only math.exp)
# ============================================================

from math import exp

# ────────────────────────────────────────────────────────────
# STEP 1: Define the Dataset (4 rows)
# ────────────────────────────────────────────────────────────
# Each row: [CGPA (0-10), Profile Score (0-5)]
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
# Target: Package in Lakh Rupees
Y = [15, 7, 12, 4]

n = len(Y)  # number of data points = 4

# ────────────────────────────────────────────────────────────
# STEP 2: Initialize all 9 Parameters (6 weights + 3 biases)
# ────────────────────────────────────────────────────────────
# Weights
w1 = 0.1   # x1 (CGPA)    → h1
w2 = 0.2   # x2 (Profile)  → h1
w3 = 0.3   # a_h1          → h2
w4 = 0.15  # x2 (Profile)  → h2
w5 = 0.25  # a_h2          → output
w6 = 0.05  # a_h1          → output

# Biases
b1 = 0.1   # bias for hidden neuron h1
b2 = 0.1   # bias for hidden neuron h2
b3 = 0.1   # bias for output neuron

# Hyperparameters
learning_rate = 0.01
epochs = 5

# ────────────────────────────────────────────────────────────
# STEP 3: Activation Function — Sigmoid (for hidden layers)
# ────────────────────────────────────────────────────────────
def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
    # Clamp z to avoid overflow in exp()
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + exp(-z))

def sigmoid_derivative(z):
    """Derivative of sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
    s = sigmoid(z)
    return s * (1.0 - s)


# ────────────────────────────────────────────────────────────
# STEP 4-8: Training Loop — 5 Epochs
# ────────────────────────────────────────────────────────────

print("=" * 70)
print("  MLP FROM SCRATCH -- TRAINING LOG")
print("  Architecture: 2 inputs -> h1 (sigmoid) -> h2 (sigmoid) -> output (linear)")
print("  Loss: Mean Squared Error | Optimizer: Gradient Descent")
print("  Learning Rate:", learning_rate, "| Epochs:", epochs)
print("=" * 70)

for epoch in range(1, epochs + 1):

    # ── Gradient accumulators (reset every epoch) ──
    dL_dw1 = 0.0
    dL_dw2 = 0.0
    dL_dw3 = 0.0
    dL_dw4 = 0.0
    dL_dw5 = 0.0
    dL_dw6 = 0.0
    dL_db1 = 0.0
    dL_db2 = 0.0
    dL_db3 = 0.0

    total_loss = 0.0  # to accumulate MSE

    # ── Loop over all 4 data points ──
    for i in range(n):
        x1 = X[i][0]  # CGPA
        x2 = X[i][1]  # Profile Score
        y  = Y[i]     # Actual Package

        # ────────────────────────────────────────────
        # FORWARD PASS
        # ────────────────────────────────────────────

        # Hidden Layer 1
        z_h1 = w1 * x1 + w2 * x2 + b1
        a_h1 = sigmoid(z_h1)

        # Hidden Layer 2
        z_h2 = w3 * a_h1 + w4 * x2 + b2
        a_h2 = sigmoid(z_h2)

        # Output Layer (linear activation)
        z_out = w5 * a_h2 + w6 * a_h1 + b3
        y_hat = z_out  # linear: ŷ = z_out

        # ────────────────────────────────────────────
        # LOSS (contribution of this data point)
        # ────────────────────────────────────────────
        error = y_hat - y
        loss_i = error ** 2
        total_loss += loss_i

        # ────────────────────────────────────────────
        # BACKWARD PASS (Backpropagation)
        # ────────────────────────────────────────────

        # ── 6a. Derivative of loss w.r.t. output ──
        # MSE = (1/n) * Σ(ŷ - y)²
        # dL/dŷ for this sample = 2/n * (ŷ - y)
        dL_dy_hat = (2.0 / n) * error

        # ── 6b. Output Layer Gradients ──
        # ŷ = z_out (linear), so dŷ/dz_out = 1
        # dL/dw5 = dL/dŷ * a_h2
        # dL/dw6 = dL/dŷ * a_h1
        # dL/db3 = dL/dŷ
        dL_dw5 += dL_dy_hat * a_h2
        dL_dw6 += dL_dy_hat * a_h1
        dL_db3 += dL_dy_hat

        # ── 6c. Hidden Layer 2 Gradients ──
        # dL/dz_h2 = dL/dŷ * w5 * σ'(z_h2)
        dL_dz_h2 = dL_dy_hat * w5 * sigmoid_derivative(z_h2)

        # dL/dw3 = dL/dz_h2 * a_h1
        # dL/dw4 = dL/dz_h2 * x2
        # dL/db2 = dL/dz_h2
        dL_dw3 += dL_dz_h2 * a_h1
        dL_dw4 += dL_dz_h2 * x2
        dL_db2 += dL_dz_h2

        # ── 6d. Hidden Layer 1 Gradients ──
        # h1 feeds into BOTH h2 (via w3) and output (via w6)
        # dL/da_h1 = dL/dŷ * w6 + dL/dz_h2 * w3
        dL_da_h1 = dL_dy_hat * w6 + dL_dz_h2 * w3

        # dL/dz_h1 = dL/da_h1 * σ'(z_h1)
        dL_dz_h1 = dL_da_h1 * sigmoid_derivative(z_h1)

        # dL/dw1 = dL/dz_h1 * x1
        # dL/dw2 = dL/dz_h1 * x2
        # dL/db1 = dL/dz_h1
        dL_dw1 += dL_dz_h1 * x1
        dL_dw2 += dL_dz_h1 * x2
        dL_db1 += dL_dz_h1

    # ────────────────────────────────────────────────────
    # STEP 7: Gradient Descent — Update all 9 parameters
    # parameter = parameter - learning_rate * ∂L/∂parameter
    # ────────────────────────────────────────────────────
    w1 = w1 - learning_rate * dL_dw1
    w2 = w2 - learning_rate * dL_dw2
    w3 = w3 - learning_rate * dL_dw3
    w4 = w4 - learning_rate * dL_dw4
    w5 = w5 - learning_rate * dL_dw5
    w6 = w6 - learning_rate * dL_dw6
    b1 = b1 - learning_rate * dL_db1
    b2 = b2 - learning_rate * dL_db2
    b3 = b3 - learning_rate * dL_db3

    # ── Compute MSE for this epoch ──
    mse = total_loss / n

    # ────────────────────────────────────────────
    # Print epoch summary
    # ────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print(f"  EPOCH {epoch}/{epochs}")
    print(f"{'-' * 70}")
    print(f"  MSE Loss: {mse:.6f}")
    print(f"\n  Updated Parameters:")
    print(f"    Weights: w1={w1:.6f}  w2={w2:.6f}  w3={w3:.6f}")
    print(f"             w4={w4:.6f}  w5={w5:.6f}  w6={w6:.6f}")
    print(f"    Biases:  b1={b1:.6f}  b2={b2:.6f}  b3={b3:.6f}")
    print(f"\n  Gradients applied this epoch:")
    print(f"    dL/dw1={dL_dw1:+.6f}  dL/dw2={dL_dw2:+.6f}  dL/dw3={dL_dw3:+.6f}")
    print(f"    dL/dw4={dL_dw4:+.6f}  dL/dw5={dL_dw5:+.6f}  dL/dw6={dL_dw6:+.6f}")
    print(f"    dL/db1={dL_db1:+.6f}  dL/db2={dL_db2:+.6f}  dL/db3={dL_db3:+.6f}")


# ────────────────────────────────────────────────────────────
# STEP 9: Final Results — Predictions vs Actual
# ────────────────────────────────────────────────────────────

print(f"\n{'=' * 70}")
print("  TRAINING COMPLETE -- FINAL RESULTS")
print(f"{'=' * 70}")

print(f"\n  Final Parameters:")
print(f"    w1 = {w1:.6f}    w2 = {w2:.6f}    w3 = {w3:.6f}")
print(f"    w4 = {w4:.6f}    w5 = {w5:.6f}    w6 = {w6:.6f}")
print(f"    b1 = {b1:.6f}    b2 = {b2:.6f}    b3 = {b3:.6f}")

print(f"\n  {'CGPA':>6}  {'Profile':>8}  {'Actual (y)':>10}  {'Predicted (y^)':>14}  {'Error':>8}")
print(f"  {'-' * 52}")

final_total_loss = 0.0
for i in range(n):
    x1 = X[i][0]
    x2 = X[i][1]
    y  = Y[i]

    # Forward pass with final weights
    z_h1 = w1 * x1 + w2 * x2 + b1
    a_h1 = sigmoid(z_h1)

    z_h2 = w3 * a_h1 + w4 * x2 + b2
    a_h2 = sigmoid(z_h2)

    z_out = w5 * a_h2 + w6 * a_h1 + b3
    y_hat = z_out

    error = y_hat - y
    final_total_loss += error ** 2

    print(f"  {x1:>6}  {x2:>8}  {y:>10}  {y_hat:>14.6f}  {error:>+8.4f}")

final_mse = final_total_loss / n
print(f"\n  Final MSE Loss: {final_mse:.6f}")
print(f"{'=' * 70}")
