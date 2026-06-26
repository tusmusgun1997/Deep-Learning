# ============================================================
# SOFTMAX — From Scratch Implementation
# ============================================================
# a_i = e^z_i / sum(e^z_j)
#
# Trains a 2-hidden-layer classification MLP using ReLU in
# hidden layers and Softmax in the output layer, using
# Categorical Cross-Entropy loss.
# ============================================================

import math
import random
random.seed(42)

# ---- Dataset: CGPA, Profile Score → Grade (Class 0, 1, 2) ----
# Class 0: Low package, Class 1: Medium package, Class 2: High package
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [
    [0.0, 0.0, 1.0],  # Class 2 (High)
    [0.0, 1.0, 0.0],  # Class 1 (Medium)
    [0.0, 1.0, 0.0],  # Class 1 (Medium)
    [1.0, 0.0, 0.0]   # Class 0 (Low)
]
n = len(Y)
NUM_CLASSES = 3

# ============================================================
# SOFTMAX FUNCTION (Numerically Stable)
# ============================================================

def softmax(logits):
    """Computes softmax probabilities for a list of logits.
    Uses subtraction of max logit to prevent overflow.
    """
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    sum_exps = sum(exps)
    return [e / sum_exps for e in exps]


# ============================================================
# VALUE & STABILITY DEMO
# ============================================================

print("=" * 60)
print("  SOFTMAX — Value & Probability Table")
print("=" * 60)
test_cases = [
    [1.0, 2.0, 3.0],
    [-1.0, 0.0, 1.0],
    [10.0, 10.0, 10.0],
    [1000.0, 1001.0, 1002.0]  # Standard softmax would overflow, stable doesn't!
]

for logits in test_cases:
    probs = softmax(logits)
    probs_str = ", ".join([f"{p:.4f}" for p in probs])
    print(f"  Logits: {str(logits):<24} -> Probs: [{probs_str}]  |  Sum = {sum(probs):.1f}")

print(f"\n  NOTE: Stable Softmax successfully handled [1000, 1001, 1002]")
print(f"  without raising OverflowError, and the outputs always sum to 1.0.")


# ============================================================
# TRAINING: Multi-Class MLP with Softmax + Cross-Entropy
# ============================================================

H1, H2 = 4, 4
EPOCHS = 1000
LR = 0.1

def he_init(fan_in):
    return random.gauss(0, math.sqrt(2.0 / fan_in))

def leaky_relu(z, alpha=0.01):
    return z if z > 0.0 else alpha * z

def leaky_relu_deriv(z, alpha=0.01):
    return 1.0 if z > 0.0 else alpha


def train_softmax():
    random.seed(42)
    # Initialize weights with standard Xavier initialization
    def xavier_init(fan_in, fan_out):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return random.uniform(-limit, limit)

    W1 = [[xavier_init(2, H1) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[xavier_init(H1, H2) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    # Output weights for 3 classes
    W3 = [[xavier_init(H2, NUM_CLASSES) for _ in range(H2)] for _ in range(NUM_CLASSES)]
    b3 = [0.0] * NUM_CLASSES

    print(f"\n{'=' * 60}")
    print(f"  TRAINING CLASSIFICATION MLP WITH SOFTMAX (LR={LR}, {EPOCHS} epochs)")
    print(f"{'=' * 60}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]

            # Forward pass: Hidden Layer 1 (Leaky ReLU)
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]

            # Forward pass: Hidden Layer 2 (Leaky ReLU)
            z2 = [sum(W2[j][k] * a1[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]

            # Forward pass: Output Layer logits (Linear)
            logits = [sum(W3[c][k] * a2[k] for k in range(H2)) + b3[c] for c in range(NUM_CLASSES)]
            probs = softmax(logits)

            # Categorical Cross-Entropy Loss: L = -sum(y_c * log(probs_c))
            loss = -sum(y[c] * math.log(max(1e-15, probs[c])) for c in range(NUM_CLASSES))
            total_loss += loss

            # Combined Softmax + Cross Entropy gradients: dL/dlogits_c = probs_c - y_c
            dlogits = [probs[c] - y[c] for c in range(NUM_CLASSES)]

            # Backward pass: output weights
            dW3 = [[dlogits[c] * a2[k] for k in range(H2)] for c in range(NUM_CLASSES)]

            # Gradient reaching hidden layer 2 activations
            dL_a2 = [sum(dlogits[c] * W3[c][k] for c in range(NUM_CLASSES)) for k in range(H2)]
            dL_z2 = [dL_a2[j] * leaky_relu_deriv(z2[j]) for j in range(H2)]

            # Gradient reaching hidden layer 1 activations
            dW2 = [[dL_z2[j] * a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j] * W2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j] * leaky_relu_deriv(z1[j]) for j in range(H1)]

            # Update output parameters
            for c in range(NUM_CLASSES):
                for k in range(H2):
                    W3[c][k] -= LR * dW3[c][k]
                b3[c] -= LR * dlogits[c]

            # Update layer 2 parameters
            for j in range(H2):
                for k in range(H1):
                    W2[j][k] -= LR * dW2[j][k]
                b2[j] -= LR * dL_z2[j]

            # Update layer 1 parameters
            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * dL_z1[j] * x[k]
                b1[j] -= LR * dL_z1[j]

        avg_loss = total_loss / n
        losses.append(avg_loss)

        if epoch % 200 == 0 or epoch == 1:
            # Let's check a prediction for class [8, 4]
            x_test = X[0]
            z1_test = [sum(W1[j][k] * x_test[k] for k in range(2)) + b1[j] for j in range(H1)]
            a1_test = [leaky_relu(z) for z in z1_test]
            z2_test = [sum(W2[j][k] * a1_test[k] for k in range(H1)) + b2[j] for j in range(H2)]
            a2_test = [leaky_relu(z) for z in z2_test]
            logits_test = [sum(W3[c][k] * a2_test[k] for k in range(H2)) + b3[c] for c in range(NUM_CLASSES)]
            probs_test = softmax(logits_test)
            probs_test_str = ", ".join([f"{p:.2f}" for p in probs_test])

            print(f"  Epoch {epoch:>3}  |  Cross-Entropy Loss: {avg_loss:.4f}  |  P([8,4]): [{probs_test_str}]")

    return losses


losses = train_softmax()

print(f"\n{'=' * 60}")
print(f"  RESULTS")
print(f"{'=' * 60}")
print(f"  Final Cross-Entropy Loss: {losses[-1]:.4f}")
print(f"\n  Observation: The MLP correctly trains using Softmax and output probabilities")
print(f"  sum to 1. The model becomes confident in class 2 (index 2) for the high CGPA input.")
print(f"{'=' * 60}")
