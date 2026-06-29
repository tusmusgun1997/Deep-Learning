# ============================================================
# BATCH NORMALIZATION — From Scratch
# ============================================================
# Normalizes activations WITHIN the network at each layer,
# not just the inputs. Applied between linear computation
# and activation function.
#
# z = W*x + b
# z_norm = (z - batch_mean) / sqrt(batch_var + eps)
# y = gamma * z_norm + beta    <-- gamma, beta are LEARNABLE
# a = activation(y)
# ============================================================

import math
import random

random.seed(42)

# ---- Dataset ----
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
# BATCH NORMALIZATION CLASS
# ================================================================

class BatchNorm:
    """Batch Normalization for a layer with 'size' neurons.

    Steps (during training, per mini-batch):
        1. Compute mean of z across batch:    mu = mean(z)
        2. Compute variance:                  var = mean((z - mu)^2)
        3. Normalize:                         z_hat = (z - mu) / sqrt(var + eps)
        4. Scale and shift:                   y = gamma * z_hat + beta

    gamma (scale) and beta (shift) are learned via backprop.
    They allow the network to "undo" normalization if needed.

    During inference:
        Use running averages of mean and variance computed during training.
    """

    def __init__(self, size):
        self.size = size
        self.gamma = [1.0] * size       # Learnable scale (init=1)
        self.beta = [0.0] * size        # Learnable shift (init=0)
        self.eps = 1e-8

        # Running stats for inference
        self.running_mean = [0.0] * size
        self.running_var = [1.0] * size
        self.momentum = 0.9   # For running average update

        # Cache for backward pass
        self.z_hat = None
        self.mean = None
        self.var = None
        self.z_input = None

    def forward(self, z_batch, training=True):
        """Forward pass. z_batch is a list of lists (batch x features).

        Returns: normalized and scaled batch.
        """
        batch_size = len(z_batch)
        self.z_input = z_batch

        if training:
            # Step 1: Compute batch mean per feature
            self.mean = [0.0] * self.size
            for sample in z_batch:
                for j in range(self.size):
                    self.mean[j] += sample[j]
            self.mean = [m / batch_size for m in self.mean]

            # Step 2: Compute batch variance per feature
            self.var = [0.0] * self.size
            for sample in z_batch:
                for j in range(self.size):
                    self.var[j] += (sample[j] - self.mean[j]) ** 2
            self.var = [v / batch_size for v in self.var]

            # Update running stats (for inference later)
            for j in range(self.size):
                self.running_mean[j] = (self.momentum * self.running_mean[j]
                                        + (1 - self.momentum) * self.mean[j])
                self.running_var[j] = (self.momentum * self.running_var[j]
                                       + (1 - self.momentum) * self.var[j])
        else:
            # Use running stats during inference
            self.mean = self.running_mean[:]
            self.var = self.running_var[:]

        # Step 3: Normalize
        self.z_hat = []
        output = []
        for sample in z_batch:
            z_hat_sample = []
            out_sample = []
            for j in range(self.size):
                zh = (sample[j] - self.mean[j]) / math.sqrt(self.var[j] + self.eps)
                z_hat_sample.append(zh)
                # Step 4: Scale and shift
                out_sample.append(self.gamma[j] * zh + self.beta[j])
            self.z_hat.append(z_hat_sample)
            output.append(out_sample)

        return output

    def backward(self, dout_batch, lr=0.01):
        """Backward pass through batch norm.

        dout_batch: gradient flowing in from next layer (batch x features)
        Returns: gradient to pass to previous layer
        """
        batch_size = len(dout_batch)
        dinput = [[0.0] * self.size for _ in range(batch_size)]

        for j in range(self.size):
            # Gradients for gamma and beta
            dgamma = sum(dout_batch[b][j] * self.z_hat[b][j] for b in range(batch_size))
            dbeta = sum(dout_batch[b][j] for b in range(batch_size))

            # Gradient through normalization
            std_inv = 1.0 / math.sqrt(self.var[j] + self.eps)

            dz_hat = [dout_batch[b][j] * self.gamma[j] for b in range(batch_size)]

            dvar = sum(dz_hat[b] * (self.z_input[b][j] - self.mean[j]) * (-0.5)
                       * (self.var[j] + self.eps) ** (-1.5) for b in range(batch_size))

            dmean = (sum(-dz_hat[b] * std_inv for b in range(batch_size))
                     + dvar * sum(-2.0 * (self.z_input[b][j] - self.mean[j])
                                 for b in range(batch_size)) / batch_size)

            for b in range(batch_size):
                dinput[b][j] = (dz_hat[b] * std_inv
                                + dvar * 2.0 * (self.z_input[b][j] - self.mean[j]) / batch_size
                                + dmean / batch_size)

            # Update gamma and beta
            self.gamma[j] -= lr * dgamma
            self.beta[j] -= lr * dbeta

        return dinput


# ================================================================
# DEMO: Training with vs without Batch Norm
# ================================================================

H1 = 4
EPOCHS = 100
LR = 0.01


def train_network(use_batchnorm=False):
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)]]
    b2 = [0.0]

    bn = BatchNorm(H1) if use_batchnorm else None

    label = "WITH Batch Norm" if use_batchnorm else "WITHOUT Batch Norm"
    print(f"\n  Training: {label}")
    print(f"  {'-' * 50}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        # ---- Full batch forward ----
        z1_batch = []
        for i in range(n):
            x = X[i]
            z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(H1)]
            z1_batch.append(z1)

        if bn:
            bn_out = bn.forward(z1_batch, training=True)
            a1_batch = [[leaky_relu(bn_out[i][j]) for j in range(H1)] for i in range(n)]
        else:
            a1_batch = [[leaky_relu(z1_batch[i][j]) for j in range(H1)] for i in range(n)]

        # Output layer + loss
        total_loss = 0.0
        dL_a1_batch = [[0.0] * H1 for _ in range(n)]

        for i in range(n):
            y_hat = sum(W2[0][k] * a1_batch[i][k] for k in range(H1)) + b2[0]
            error = y_hat - Y[i]
            total_loss += error ** 2
            dL = (2.0 / n) * error

            # Output gradients
            for k in range(H1):
                W2[0][k] -= LR * dL * a1_batch[i][k]
            b2[0] -= LR * dL

            # Gradient to hidden
            for k in range(H1):
                dL_a1_batch[i][k] = dL * W2[0][k]

        # Hidden layer backward
        if bn:
            # Through activation
            dL_bn = [[dL_a1_batch[i][j] * leaky_relu_derivative(
                          bn_out[i][j] if bn else z1_batch[i][j])
                       for j in range(H1)] for i in range(n)]
            # Through batch norm
            dL_z1_batch = bn.backward(dL_bn, LR)
        else:
            dL_z1_batch = [[dL_a1_batch[i][j] * leaky_relu_derivative(z1_batch[i][j])
                            for j in range(H1)] for i in range(n)]

        # Update W1, b1
        for i in range(n):
            x = X[i]
            for j in range(H1):
                for k in range(2):
                    W1[j][k] -= LR * dL_z1_batch[i][j] * x[k]
                b1[j] -= LR * dL_z1_batch[i][j]

        mse = total_loss / n
        losses.append(mse)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")

    return losses


print("=" * 60)
print("  BATCH NORMALIZATION DEMO")
print("  Network: 2 -> 4 (BN) -> 1")
print("=" * 60)

no_bn = train_network(use_batchnorm=False)
with_bn = train_network(use_batchnorm=True)

print(f"\n{'=' * 60}")
print("  SUMMARY")
print(f"{'=' * 60}")
print(f"  {'':>20} {'No BN':>10} {'With BN':>10}")
print(f"  {'-' * 40}")
print(f"  {'Final MSE':>20} {no_bn[-1]:>10.4f} {with_bn[-1]:>10.4f}")
print(f"  {'Best MSE':>20} {min(no_bn):>10.4f} {min(with_bn):>10.4f}")
print(f"{'=' * 60}")
