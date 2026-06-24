# ============================================================
# Improving Neural Networks -- All Solutions Implemented
# Pure Python (only math and random modules)
# ============================================================
# This file implements ALL 7 categories of improvement:
#
#   1. OVERFITTING
#      1.1 Dropout
#      1.2 L1 & L2 Regularization
#      1.3 Early Stopping
#
#   2. NORMALIZATION
#      2.1 Normalizing Inputs (Min-Max, Z-Score)
#      2.2 Batch Normalization
#
#   3. VANISHING GRADIENTS
#      3.1 Activation Functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
#      3.2 Weight Initialization (Xavier, He)
#
#   4. GRADIENT CHECKING & CLIPPING
#      4.1 Gradient Checking (Numerical Verification)
#      4.2 Gradient Clipping (by value, by norm)
#
#   5. OPTIMIZERS
#      5.1 SGD with Momentum
#      5.2 Adagrad
#      5.3 RMSprop
#      5.4 Adam
#
#   6. LEARNING RATE SCHEDULING
#      6.1 Step Decay
#      6.2 Exponential Decay
#      6.3 Cosine Annealing
#
#   7. HYPERPARAMETER TUNING
#      7.1 Grid Search over layers, nodes, batch size
#
# Dataset: CGPA, Profile Score -> Package (LPA)
# Architecture: 2 -> Hidden1 -> Hidden2 -> 1
# ============================================================

import math
import random


# ================================================================
# SECTION 1: OVERFITTING SOLUTIONS
# ================================================================

# ---- 1.1 Dropout ----

def apply_dropout(activations, drop_rate=0.3, training=True):
    """Inverted Dropout: During training, randomly zero out neurons
    and scale up remaining by 1/(1-p) so expected values stay the same.
    During inference, use all neurons without scaling.

    Args:
        activations: list of neuron values
        drop_rate: probability of dropping a neuron (e.g., 0.3 = 30%)
        training: True during training, False during inference
    Returns:
        (output, mask)
    """
    if not training or drop_rate == 0:
        return activations, [1] * len(activations)

    mask = []
    output = []
    scale = 1.0 / (1.0 - drop_rate)

    for a in activations:
        if random.random() > drop_rate:  # Keep this neuron
            mask.append(1)
            output.append(a * scale)     # Scale up to compensate
        else:                             # Drop this neuron
            mask.append(0)
            output.append(0.0)

    return output, mask


# ---- 1.2 L1 & L2 Regularization ----

def l2_penalty(weights, lambda_val):
    """L2 Regularization penalty: lambda * sum(w^2)"""
    return lambda_val * sum(w ** 2 for w in weights)

def l2_gradient(weight, lambda_val):
    """L2 gradient for a single weight: 2 * lambda * w"""
    return 2 * lambda_val * weight

def l1_penalty(weights, lambda_val):
    """L1 Regularization penalty: lambda * sum(|w|)
    Drives weights to exactly 0 (sparsity)"""
    return lambda_val * sum(abs(w) for w in weights)

def l1_gradient(weight, lambda_val):
    """L1 gradient for a single weight: lambda * sign(w)"""
    if weight > 0:
        return lambda_val
    elif weight < 0:
        return -lambda_val
    else:
        return 0.0


# ---- 1.3 Early Stopping ----

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Tracks best validation loss and counts epochs without improvement.
    When patience is exceeded, signals to stop training and can restore
    the best weights seen so far.

    Args:
        patience: epochs to wait for improvement before stopping
        min_delta: minimum change to count as improvement
    """

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def check(self, val_loss, current_weights, epoch):
        """Check if training should stop.

        Args:
            val_loss: current validation loss
            current_weights: flat list of all current parameters
            epoch: current epoch number
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found -- reset counter, save weights
            self.best_loss = val_loss
            self.best_weights = current_weights[:]  # Deep copy
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True  # STOP!
            return False


# ================================================================
# SECTION 2: NORMALIZATION
# ================================================================

# ---- 2.1 Normalizing Inputs ----

def min_max_normalize(data, feature_idx):
    """Min-Max Normalization: scales feature to [0, 1].
    x_norm = (x - min) / (max - min)

    Args:
        data: list of samples, each sample is a list of features
        feature_idx: which feature to normalize
    Returns:
        (normalized_data, min_val, max_val) -- save min/max for test data!
    """
    values = [sample[feature_idx] for sample in data]
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    if range_val == 0:
        range_val = 1  # Avoid division by zero

    normalized = []
    for sample in data:
        new_sample = sample[:]
        new_sample[feature_idx] = (sample[feature_idx] - min_val) / range_val
        normalized.append(new_sample)

    return normalized, min_val, max_val


def z_score_normalize(data, feature_idx):
    """Z-Score Normalization: scales to mean=0, std=1.
    x_norm = (x - mean) / std

    Args:
        data: list of samples
        feature_idx: which feature to normalize
    Returns:
        (normalized_data, mean, std)
    """
    values = [sample[feature_idx] for sample in data]
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0

    normalized = []
    for sample in data:
        new_sample = sample[:]
        new_sample[feature_idx] = (sample[feature_idx] - mean) / std
        normalized.append(new_sample)

    return normalized, mean, std


def normalize_all_features(data, method="zscore"):
    """Normalize all features in dataset.

    Args:
        data: list of samples
        method: 'zscore' or 'minmax'
    Returns:
        (normalized_data, stats) -- stats needed to normalize test data
    """
    num_features = len(data[0])
    stats = []
    result = [sample[:] for sample in data]  # Deep copy

    for f in range(num_features):
        if method == "minmax":
            result, min_v, max_v = min_max_normalize(result, f)
            stats.append(("minmax", min_v, max_v))
        else:
            result, mean, std = z_score_normalize(result, f)
            stats.append(("zscore", mean, std))

    return result, stats


# ---- 2.2 Batch Normalization ----

class BatchNorm:
    """Batch Normalization layer.
    Normalizes inputs to mean=0, std=1, then applies learnable gamma and beta.

    Formula:
        z_norm = (z - batch_mean) / sqrt(batch_var + epsilon)
        output = gamma * z_norm + beta
    """

    def __init__(self, size):
        self.gamma = [1.0] * size  # Learnable scale
        self.beta = [0.0] * size   # Learnable shift
        self.epsilon = 1e-8
        self.z_norm = None
        self.mean = None
        self.var = None

    def forward(self, z_batch):
        """Normalize a batch of feature vectors."""
        batch_size = len(z_batch)
        feature_size = len(z_batch[0])

        # Step 1: Batch mean
        self.mean = [0.0] * feature_size
        for sample in z_batch:
            for j in range(feature_size):
                self.mean[j] += sample[j]
        self.mean = [m / batch_size for m in self.mean]

        # Step 2: Batch variance
        self.var = [0.0] * feature_size
        for sample in z_batch:
            for j in range(feature_size):
                self.var[j] += (sample[j] - self.mean[j]) ** 2
        self.var = [v / batch_size for v in self.var]

        # Step 3: Normalize + scale/shift
        self.z_norm = []
        output = []
        for sample in z_batch:
            norm_sample = []
            out_sample = []
            for j in range(feature_size):
                zn = (sample[j] - self.mean[j]) / math.sqrt(self.var[j] + self.epsilon)
                norm_sample.append(zn)
                out_sample.append(self.gamma[j] * zn + self.beta[j])
            self.z_norm.append(norm_sample)
            output.append(out_sample)

        return output


# ================================================================
# SECTION 3: VANISHING GRADIENTS SOLUTIONS
# ================================================================

# ---- 3.1 Activation Functions ----

def sigmoid(z):
    """Sigmoid: output in (0, 1). Causes vanishing gradients!
    Max derivative = 0.25 at z=0"""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def tanh_activation(z):
    """Tanh: output in (-1, 1). Better than sigmoid, can still vanish."""
    z = max(-500, min(500, z))
    return math.tanh(z)

def tanh_derivative(z):
    t = math.tanh(z)
    return 1.0 - t * t

def relu(z):
    """ReLU: f(z) = max(0, z). Solves vanishing gradient for z > 0.
    Problem: 'Dying ReLU' -- neuron dies if z always < 0."""
    return max(0.0, z)

def relu_derivative(z):
    return 1.0 if z > 0 else 0.0

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: allows small gradient for negative z.
    Fixes the dying ReLU problem."""
    return z if z > 0 else alpha * z

def leaky_relu_derivative(z, alpha=0.01):
    return 1.0 if z > 0 else alpha


# ---- 3.2 Weight Initialization ----

def random_init(fan_in, fan_out):
    """Naive random init -- causes vanishing/exploding gradients"""
    return random.uniform(-1, 1)

def xavier_init(fan_in, fan_out):
    """Xavier/Glorot Init -- best for Sigmoid/Tanh.
    Var(W) = 2 / (fan_in + fan_out)"""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return random.gauss(0, std)

def he_init(fan_in, fan_out):
    """He Init -- best for ReLU.
    Var(W) = 2 / fan_in"""
    std = math.sqrt(2.0 / fan_in)
    return random.gauss(0, std)


# ================================================================
# SECTION 4: GRADIENT CHECKING & CLIPPING
# ================================================================

# ---- 4.1 Gradient Checking (Numerical Verification) ----

def numerical_gradient(loss_fn, params, param_idx, epsilon=1e-7):
    """Compute numerical gradient for parameter at param_idx using
    two-sided finite difference:

        dL/dW = (L(W + eps) - L(W - eps)) / (2 * eps)

    This is SLOW (2 forward passes per parameter) but accurate.
    Used ONLY for debugging backprop, never in production.
    """
    original_val = params[param_idx]

    # L(W + epsilon)
    params[param_idx] = original_val + epsilon
    loss_plus = loss_fn(params)

    # L(W - epsilon)
    params[param_idx] = original_val - epsilon
    loss_minus = loss_fn(params)

    # Restore original
    params[param_idx] = original_val

    return (loss_plus - loss_minus) / (2 * epsilon)


def gradient_check(loss_fn, params, analytical_gradients, epsilon=1e-7):
    """Compare analytical gradients (from backprop) with numerical gradients.

    Returns:
        list of (param_idx, analytical, numerical, relative_error)
    """
    results = []
    params_copy = params[:]

    for i in range(len(params)):
        num_grad = numerical_gradient(loss_fn, params_copy, i, epsilon)
        ana_grad = analytical_gradients[i]

        # Relative error
        denom = abs(ana_grad) + abs(num_grad) + 1e-10
        rel_error = abs(ana_grad - num_grad) / denom

        results.append((i, ana_grad, num_grad, rel_error))

    return results


# ---- 4.2 Gradient Clipping ----

def clip_by_value(gradient, max_val):
    """Clip individual gradient to [-max_val, max_val]."""
    return max(-max_val, min(max_val, gradient))

def clip_by_norm(gradients, max_norm):
    """Clip gradient vector by L2 norm.
    If ||g|| > max_norm, scale g down to have norm = max_norm.
    Preserves gradient DIRECTION while limiting magnitude."""
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    return gradients, total_norm


# ================================================================
# SECTION 5: OPTIMIZERS
# ================================================================

# ---- 5.1 SGD with Momentum ----

class MomentumOptimizer:
    """SGD with Momentum.

    v_t = beta * v_(t-1) + (1 - beta) * gradient
    param = param - lr * v_t

    Accelerates in consistent gradient directions,
    dampens oscillations in inconsistent directions.
    """

    def __init__(self, num_params, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = [0.0] * num_params  # Velocity

    def step(self, params, gradients):
        new_params = []
        for i in range(len(params)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * gradients[i]
            new_params.append(params[i] - self.lr * self.v[i])
        return new_params


# ---- 5.2 Adagrad ----

class AdagradOptimizer:
    """Adagrad: Adaptive learning rate per parameter.

    s_t = s_(t-1) + gradient^2
    param = param - lr * gradient / (sqrt(s_t) + epsilon)

    Pros: Great for sparse features, adapts automatically.
    Cons: Learning rate monotonically decreases -- can stop learning!
    """

    def __init__(self, num_params, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.epsilon = epsilon
        self.s = [0.0] * num_params  # Accumulated squared gradients

    def step(self, params, gradients):
        new_params = []
        for i in range(len(params)):
            self.s[i] += gradients[i] ** 2
            new_params.append(
                params[i] - self.lr * gradients[i] / (math.sqrt(self.s[i]) + self.epsilon)
            )
        return new_params


# ---- 5.3 RMSprop ----

class RMSpropOptimizer:
    """RMSprop: Fixes Adagrad's decaying learning rate.

    s_t = beta * s_(t-1) + (1 - beta) * gradient^2
    param = param - lr * gradient / (sqrt(s_t) + epsilon)

    Uses exponential moving average instead of accumulating all squared gradients.
    """

    def __init__(self, num_params, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = [0.0] * num_params

    def step(self, params, gradients):
        new_params = []
        for i in range(len(params)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * gradients[i] ** 2
            new_params.append(
                params[i] - self.lr * gradients[i] / (math.sqrt(self.s[i]) + self.epsilon)
            )
        return new_params


# ---- 5.4 Adam ----

class AdamOptimizer:
    """Adam: Adaptive Moment Estimation.
    Combines Momentum (1st moment) + RMSprop (2nd moment) + bias correction.

    m_t = beta1 * m_(t-1) + (1-beta1) * gradient        # Momentum
    v_t = beta2 * v_(t-1) + (1-beta2) * gradient^2      # RMSprop
    m_hat = m_t / (1 - beta1^t)                          # Bias correction
    v_hat = v_t / (1 - beta2^t)                          # Bias correction
    param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self, num_params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0.0] * num_params
        self.v = [0.0] * num_params
        self.t = 0

    def step(self, params, gradients):
        self.t += 1
        new_params = []
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_params.append(params[i] - self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon))
        return new_params


# ================================================================
# SECTION 6: LEARNING RATE SCHEDULING
# ================================================================

def step_decay(initial_lr, epoch, drop_factor=0.5, drop_every=20):
    """Step Decay: Reduce LR by factor every N epochs."""
    return initial_lr * (drop_factor ** (epoch // drop_every))

def exponential_decay(initial_lr, epoch, decay_rate=0.95):
    """Exponential Decay: lr = lr_0 * decay_rate^epoch"""
    return initial_lr * (decay_rate ** epoch)

def cosine_annealing(initial_lr, epoch, total_epochs, lr_min=0.0001):
    """Cosine Annealing: Smooth cosine curve from lr_max to lr_min"""
    return lr_min + 0.5 * (initial_lr - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))


# ================================================================
# SECTION 7: HYPERPARAMETER TUNING
# ================================================================

def grid_search(X, Y, configs, epochs=50):
    """Simple Grid Search over network configurations.

    Args:
        X: input data
        Y: target data
        configs: list of dicts with keys 'hidden1_size', 'hidden2_size', 'lr', 'batch_size'
        epochs: epochs per config
    Returns:
        list of (config, final_mse) sorted by MSE
    """
    results = []

    for cfg in configs:
        h1 = cfg['hidden1_size']
        h2 = cfg['hidden2_size']
        lr = cfg['lr']
        batch = cfg.get('batch_size', len(X))

        random.seed(42)

        # Initialize weights
        W1 = [[he_init(2, h1) for _ in range(2)] for _ in range(h1)]
        b1 = [0.0] * h1
        W2 = [[he_init(h1, h2) for _ in range(h1)] for _ in range(h2)]
        b2 = [0.0] * h2
        W3 = [[he_init(h2, 1) for _ in range(h2)]]
        b3 = [0.0]

        n = len(Y)

        for epoch in range(epochs):
            # Mini-batch: process 'batch' samples at a time
            indices = list(range(n))
            random.shuffle(indices)

            for start in range(0, n, batch):
                batch_idx = indices[start:start + batch]
                m = len(batch_idx)

                # Accumulate gradients for this mini-batch
                dW1 = [[0.0] * 2 for _ in range(h1)]
                db1 = [0.0] * h1
                dW2 = [[0.0] * h1 for _ in range(h2)]
                db2 = [0.0] * h2
                dW3 = [[0.0] * h2]
                db3 = [0.0]

                for idx in batch_idx:
                    x = X[idx]
                    y = Y[idx]

                    # Forward
                    z1 = [sum(W1[j][k] * x[k] for k in range(2)) + b1[j] for j in range(h1)]
                    a1 = [leaky_relu(z) for z in z1]

                    z2 = [sum(W2[j][k] * a1[k] for k in range(h1)) + b2[j] for j in range(h2)]
                    a2 = [leaky_relu(z) for z in z2]

                    z3 = sum(W3[0][k] * a2[k] for k in range(h2)) + b3[0]
                    y_hat = z3

                    # Backward
                    dL_dy = (2.0 / m) * (y_hat - y)

                    for k in range(h2):
                        dW3[0][k] += dL_dy * a2[k]
                    db3[0] += dL_dy

                    dL_da2 = [dL_dy * W3[0][k] for k in range(h2)]
                    dL_dz2 = [dL_da2[j] * leaky_relu_derivative(z2[j]) for j in range(h2)]

                    for j in range(h2):
                        for k in range(h1):
                            dW2[j][k] += dL_dz2[j] * a1[k]
                        db2[j] += dL_dz2[j]

                    dL_da1 = [sum(dL_dz2[j] * W2[j][k] for j in range(h2)) for k in range(h1)]
                    dL_dz1 = [dL_da1[j] * leaky_relu_derivative(z1[j]) for j in range(h1)]

                    for j in range(h1):
                        for k in range(2):
                            dW1[j][k] += dL_dz1[j] * x[k]
                        db1[j] += dL_dz1[j]

                # Update weights
                for j in range(h1):
                    for k in range(2):
                        W1[j][k] -= lr * dW1[j][k]
                    b1[j] -= lr * db1[j]
                for j in range(h2):
                    for k in range(h1):
                        W2[j][k] -= lr * dW2[j][k]
                    b2[j] -= lr * db2[j]
                for k in range(h2):
                    W3[0][k] -= lr * dW3[0][k]
                b3[0] -= lr * db3[0]

        # Evaluate final MSE
        total_loss = 0
        for i in range(n):
            z1 = [sum(W1[j][k] * X[i][k] for k in range(2)) + b1[j] for j in range(h1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(W2[j][k] * a1[k] for k in range(h1)) + b2[j] for j in range(h2)]
            a2 = [leaky_relu(z) for z in z2]
            y_hat = sum(W3[0][k] * a2[k] for k in range(h2)) + b3[0]
            total_loss += (y_hat - Y[i]) ** 2

        mse = total_loss / n
        results.append((cfg, mse))

    results.sort(key=lambda x: x[1])
    return results


# ================================================================
# ================================================================
#                    FULL TRAINING DEMO
#           Comparing Vanilla vs Improved Network
# ================================================================
# ================================================================

# Dataset
X_raw = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# Architecture
INPUT_SIZE = 2
HIDDEN1_SIZE = 4
HIDDEN2_SIZE = 4
OUTPUT_SIZE = 1

EPOCHS = 100
INITIAL_LR = 0.01
L2_LAMBDA = 0.001
DROPOUT_RATE = 0.2
GRAD_CLIP_MAX = 5.0
EARLY_STOP_PATIENCE = 15


def init_network(init_fn, h1_size, h2_size):
    """Initialize all weights and biases."""
    net = {}
    net['W1'] = [[init_fn(INPUT_SIZE, h1_size) for _ in range(INPUT_SIZE)] for _ in range(h1_size)]
    net['b1'] = [0.0] * h1_size
    net['W2'] = [[init_fn(h1_size, h2_size) for _ in range(h1_size)] for _ in range(h2_size)]
    net['b2'] = [0.0] * h2_size
    net['W3'] = [[init_fn(h2_size, OUTPUT_SIZE) for _ in range(h2_size)]]
    net['b3'] = [0.0]
    return net


def get_all_weights(net):
    """Flatten all weights into a single list."""
    weights = []
    for key in ['W1', 'W2', 'W3']:
        for row in net[key]:
            weights.extend(row)
    return weights


def flatten_params(net):
    """Flatten all parameters into a single list."""
    flat = []
    for key in ['W1', 'W2', 'W3']:
        for row in net[key]:
            flat.extend(row)
    for key in ['b1', 'b2', 'b3']:
        flat.extend(net[key])
    return flat


def unflatten_params(flat, net):
    """Write flat parameters back into network."""
    idx = 0
    for key in ['W1', 'W2', 'W3']:
        for i, row in enumerate(net[key]):
            for j in range(len(row)):
                net[key][i][j] = flat[idx]
                idx += 1
    for key in ['b1', 'b2', 'b3']:
        for i in range(len(net[key])):
            net[key][i] = flat[idx]
            idx += 1
    return net


def forward_pass(net, x, act_fn, h1_size, h2_size, dropout_rate=0.0, training=True):
    """Forward pass with optional dropout."""
    cache = {}

    # Hidden Layer 1
    z1 = []
    for j in range(h1_size):
        z = sum(net['W1'][j][k] * x[k] for k in range(INPUT_SIZE)) + net['b1'][j]
        z1.append(z)
    cache['z1'] = z1
    a1 = [act_fn(z) for z in z1]
    a1, mask1 = apply_dropout(a1, dropout_rate, training)
    cache['a1'] = a1
    cache['mask1'] = mask1

    # Hidden Layer 2
    z2 = []
    for j in range(h2_size):
        z = sum(net['W2'][j][k] * a1[k] for k in range(h1_size)) + net['b2'][j]
        z2.append(z)
    cache['z2'] = z2
    a2 = [act_fn(z) for z in z2]
    a2, mask2 = apply_dropout(a2, dropout_rate, training)
    cache['a2'] = a2
    cache['mask2'] = mask2

    # Output Layer (linear)
    z3 = sum(net['W3'][0][k] * a2[k] for k in range(h2_size)) + net['b3'][0]
    y_hat = z3
    cache['z3'] = z3
    cache['y_hat'] = y_hat

    return y_hat, cache


def backward_pass(net, x, y, cache, act_deriv, h1_size, h2_size, l2_lambda=0.0, dropout_rate=0.0):
    """Backward pass with L2 regularization and dropout."""
    grads = {}
    y_hat = cache['y_hat']
    error = y_hat - y
    dL_dy = (2.0 / n) * error

    # Output gradients
    grads['dW3'] = [[dL_dy * cache['a2'][k] for k in range(h2_size)]]
    grads['db3'] = [dL_dy]

    # Hidden Layer 2
    dL_da2 = [dL_dy * net['W3'][0][k] for k in range(h2_size)]
    dL_da2 = [dL_da2[k] * cache['mask2'][k] for k in range(h2_size)]
    if dropout_rate > 0:
        scale = 1.0 / (1.0 - dropout_rate)
        dL_da2 = [d * scale for d in dL_da2]
    dL_dz2 = [dL_da2[j] * act_deriv(cache['z2'][j]) for j in range(h2_size)]

    grads['dW2'] = [[dL_dz2[j] * cache['a1'][k] for k in range(h1_size)] for j in range(h2_size)]
    grads['db2'] = dL_dz2[:]

    # Hidden Layer 1
    dL_da1 = [0.0] * h1_size
    for k in range(h1_size):
        for j in range(h2_size):
            dL_da1[k] += dL_dz2[j] * net['W2'][j][k]
    dL_da1 = [dL_da1[k] * cache['mask1'][k] for k in range(h1_size)]
    if dropout_rate > 0:
        scale = 1.0 / (1.0 - dropout_rate)
        dL_da1 = [d * scale for d in dL_da1]
    dL_dz1 = [dL_da1[j] * act_deriv(cache['z1'][j]) for j in range(h1_size)]

    grads['dW1'] = [[dL_dz1[j] * x[k] for k in range(INPUT_SIZE)] for j in range(h1_size)]
    grads['db1'] = dL_dz1[:]

    # Add L2 regularization gradients
    if l2_lambda > 0:
        for j in range(h1_size):
            for k in range(INPUT_SIZE):
                grads['dW1'][j][k] += l2_gradient(net['W1'][j][k], l2_lambda)
        for j in range(h2_size):
            for k in range(h1_size):
                grads['dW2'][j][k] += l2_gradient(net['W2'][j][k], l2_lambda)
        for k in range(h2_size):
            grads['dW3'][0][k] += l2_gradient(net['W3'][0][k], l2_lambda)

    return grads


def flatten_gradients(grads):
    flat = []
    for key in ['dW1', 'dW2', 'dW3']:
        for row in grads[key]:
            flat.extend(row)
    for key in ['db1', 'db2', 'db3']:
        flat.extend(grads[key])
    return flat


# ================================================================
# TRAIN FUNCTION
# ================================================================

def train(mode="improved"):
    """Train the network. mode = 'vanilla' or 'improved'."""

    if mode == "vanilla":
        init_fn = random_init
        act_fn = sigmoid
        act_deriv = sigmoid_derivative
        optimizer_name = "SGD"
        use_lr_schedule = False
        use_l2 = False
        use_dropout = False
        use_grad_clip = False
        use_early_stop = False
        use_normalization = False
        label = "VANILLA (Sigmoid + Random Init + SGD)"
    else:
        init_fn = he_init
        act_fn = leaky_relu
        act_deriv = leaky_relu_derivative
        optimizer_name = "Adam"
        use_lr_schedule = True
        use_l2 = True
        use_dropout = True
        use_grad_clip = True
        use_early_stop = True
        use_normalization = True
        label = "IMPROVED (All 7 techniques applied)"

    dropout_rate = DROPOUT_RATE if use_dropout else 0.0
    l2_lambda = L2_LAMBDA if use_l2 else 0.0

    # Normalize inputs
    if use_normalization:
        X, norm_stats = normalize_all_features(X_raw, method="zscore")
    else:
        X = [row[:] for row in X_raw]

    random.seed(42)
    net = init_network(init_fn, HIDDEN1_SIZE, HIDDEN2_SIZE)
    total_params = len(flatten_params(net))

    # Setup optimizer
    if optimizer_name == "Adam":
        optimizer = AdamOptimizer(total_params, lr=INITIAL_LR)
    elif optimizer_name == "Momentum":
        optimizer = MomentumOptimizer(total_params, lr=INITIAL_LR)
    elif optimizer_name == "RMSprop":
        optimizer = RMSpropOptimizer(total_params, lr=INITIAL_LR)
    elif optimizer_name == "Adagrad":
        optimizer = AdagradOptimizer(total_params, lr=INITIAL_LR)
    else:
        optimizer = None  # Vanilla SGD

    # Early stopping
    early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE) if use_early_stop else None

    print(f"\n{'=' * 70}")
    print(f"  MODE: {label}")
    print(f"  Parameters: {total_params} | Optimizer: {optimizer_name}")
    if use_normalization:
        print(f"  Input Normalization: Z-Score")
    if use_l2:
        print(f"  L2 Lambda: {l2_lambda}")
    if use_dropout:
        print(f"  Dropout: {dropout_rate}")
    if use_grad_clip:
        print(f"  Gradient Clipping: max_norm={GRAD_CLIP_MAX}")
    if use_early_stop:
        print(f"  Early Stopping: patience={EARLY_STOP_PATIENCE}")
    print(f"{'=' * 70}")

    loss_history = []
    stopped_early = False

    for epoch in range(1, EPOCHS + 1):
        # Learning rate schedule
        if use_lr_schedule:
            lr = cosine_annealing(INITIAL_LR, epoch, EPOCHS)
        else:
            lr = INITIAL_LR

        # Accumulate gradients
        acc_grads = None
        total_loss = 0.0

        for i in range(n):
            y_hat, cache = forward_pass(net, X[i], act_fn, HIDDEN1_SIZE, HIDDEN2_SIZE,
                                        dropout_rate, training=True)
            total_loss += (y_hat - Y[i]) ** 2

            grads = backward_pass(net, X[i], Y[i], cache, act_deriv,
                                  HIDDEN1_SIZE, HIDDEN2_SIZE, l2_lambda, dropout_rate)

            if acc_grads is None:
                acc_grads = grads
            else:
                for key in grads:
                    if isinstance(grads[key][0], list):
                        for j in range(len(grads[key])):
                            for k in range(len(grads[key][j])):
                                acc_grads[key][j][k] += grads[key][j][k]
                    else:
                        for j in range(len(grads[key])):
                            acc_grads[key][j] += grads[key][j]

        mse = total_loss / n
        loss_history.append(mse)

        # Gradient clipping
        flat_grads = flatten_gradients(acc_grads)
        if use_grad_clip:
            flat_grads, grad_norm = clip_by_norm(flat_grads, GRAD_CLIP_MAX)
        else:
            grad_norm = math.sqrt(sum(g ** 2 for g in flat_grads))

        # Update parameters
        flat_params = flatten_params(net)

        if optimizer is not None:
            if hasattr(optimizer, 'lr'):
                optimizer.lr = lr
            flat_params = optimizer.step(flat_params, flat_grads)
        else:
            flat_params = [p - lr * g for p, g in zip(flat_params, flat_grads)]

        net = unflatten_params(flat_params, net)

        # Early stopping check (using training loss as proxy since dataset is tiny)
        if early_stopper is not None:
            if early_stopper.check(mse, flatten_params(net), epoch):
                print(f"  ** Early stopping at epoch {epoch}! "
                      f"Best loss: {early_stopper.best_loss:.4f} **")
                # Restore best weights
                net = unflatten_params(early_stopper.best_weights, net)
                stopped_early = True
                break

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  |  MSE: {mse:>10.4f}  |  "
                  f"LR: {lr:.6f}  |  Grad Norm: {grad_norm:.4f}")

    # Final predictions
    print(f"\n  {'- ' * 35}")
    print(f"  FINAL PREDICTIONS:")
    print(f"  {'CGPA':>6}  {'Profile':>8}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    print(f"  {'-' * 46}")

    final_loss = 0.0
    for i in range(n):
        y_hat, _ = forward_pass(net, X[i], act_fn, HIDDEN1_SIZE, HIDDEN2_SIZE,
                                dropout_rate=0.0, training=False)
        error = y_hat - Y[i]
        final_loss += error ** 2
        print(f"  {X_raw[i][0]:>6}  {X_raw[i][1]:>8}  {Y[i]:>8}  {y_hat:>10.4f}  {error:>+8.4f}")

    final_mse = final_loss / n
    print(f"\n  Final MSE: {final_mse:.4f}")
    if stopped_early:
        print(f"  (Stopped early at epoch {early_stopper.stopped_epoch})")
    print(f"{'=' * 70}")

    return loss_history


# ================================================================
# RUN COMPARISON
# ================================================================

print("\n" + "#" * 70)
print("#" + " " * 16 + "NEURAL NETWORK IMPROVEMENT DEMO" + " " * 21 + "#")
print("#" + " " * 15 + "Vanilla vs Improved Comparison" + " " * 23 + "#")
print("#" * 70)

vanilla_losses = train(mode="vanilla")
improved_losses = train(mode="improved")

# Summary
print(f"\n{'=' * 70}")
print("  COMPARISON SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Metric':<30} {'Vanilla':>15} {'Improved':>15}")
print(f"  {'-' * 60}")
print(f"  {'Initial MSE (Epoch 1)':<30} {vanilla_losses[0]:>15.4f} {improved_losses[0]:>15.4f}")
print(f"  {'Final MSE':<30} {vanilla_losses[-1]:>15.4f} {improved_losses[-1]:>15.4f}")
print(f"  {'Best MSE':<30} {min(vanilla_losses):>15.4f} {min(improved_losses):>15.4f}")
imp = (vanilla_losses[-1] - improved_losses[-1]) / vanilla_losses[-1] * 100
print(f"  {'Improvement':<30} {'':>15} {imp:>14.1f}%")
print(f"{'=' * 70}")

print("\n  Techniques used in IMPROVED mode:")
print("    1. OVERFITTING:")
print("       [x] Dropout (rate=0.2)")
print("       [x] L2 Regularization (lambda=0.001)")
print("       [x] Early Stopping (patience=15)")
print("    2. NORMALIZATION:")
print("       [x] Z-Score Input Normalization")
print("       [x] Batch Normalization (implemented, not used in demo)")
print("    3. VANISHING GRADIENTS:")
print("       [x] Leaky ReLU Activation (instead of sigmoid)")
print("       [x] He Weight Initialization (instead of random)")
print("    4. GRADIENT CHECKING & CLIPPING:")
print("       [x] Gradient Clipping by Norm (max=5.0)")
print("       [x] Gradient Checking (implemented for debugging)")
print("    5. OPTIMIZERS:")
print("       [x] Adam (instead of vanilla SGD)")
print("       [x] Momentum, Adagrad, RMSprop (all implemented)")
print("    6. LEARNING RATE:")
print("       [x] Cosine Annealing LR Schedule")
print("    7. HYPERPARAMETER TUNING:")
print("       [x] Grid Search (implemented)")


# ================================================================
# DEMO: GRADIENT CHECKING
# ================================================================

print(f"\n\n{'=' * 70}")
print("  BONUS DEMO: GRADIENT CHECKING")
print(f"{'=' * 70}")

random.seed(42)
net_check = init_network(he_init, HIDDEN1_SIZE, HIDDEN2_SIZE)
X_norm, _ = normalize_all_features(X_raw, method="zscore")

# Compute analytical gradients for first sample
_, cache = forward_pass(net_check, X_norm[0], leaky_relu, HIDDEN1_SIZE, HIDDEN2_SIZE,
                        dropout_rate=0.0, training=False)
grads = backward_pass(net_check, X_norm[0], Y[0], cache, leaky_relu_derivative,
                      HIDDEN1_SIZE, HIDDEN2_SIZE, l2_lambda=0.0, dropout_rate=0.0)
analytical = flatten_gradients(grads)

# Define loss function for numerical gradient
def loss_fn(flat_params):
    temp_net = init_network(he_init, HIDDEN1_SIZE, HIDDEN2_SIZE)
    temp_net = unflatten_params(flat_params, temp_net)
    y_hat, _ = forward_pass(temp_net, X_norm[0], leaky_relu, HIDDEN1_SIZE, HIDDEN2_SIZE,
                            dropout_rate=0.0, training=False)
    return (1.0 / n) * (y_hat - Y[0]) ** 2

flat_params = flatten_params(net_check)
results = gradient_check(loss_fn, flat_params, analytical)

print(f"\n  Checking first 10 parameters (of {len(results)}):")
print(f"  {'Param':>6}  {'Analytical':>12}  {'Numerical':>12}  {'Rel Error':>12}  {'Status':>8}")
print(f"  {'-' * 56}")
for i, (idx, ana, num, err) in enumerate(results[:10]):
    status = "OK" if err < 1e-5 else "CHECK!" if err < 1e-3 else "BUG!"
    print(f"  {idx:>6}  {ana:>12.8f}  {num:>12.8f}  {err:>12.2e}  {status:>8}")

max_err = max(r[3] for r in results)
print(f"\n  Max relative error across all {len(results)} params: {max_err:.2e}")
if max_err < 1e-5:
    print("  Backpropagation is CORRECT!")
elif max_err < 1e-3:
    print("  Backpropagation looks OK but check edge cases.")
else:
    print("  WARNING: Backpropagation may have a bug!")


# ================================================================
# DEMO: HYPERPARAMETER GRID SEARCH
# ================================================================

print(f"\n\n{'=' * 70}")
print("  BONUS DEMO: HYPERPARAMETER GRID SEARCH")
print(f"{'=' * 70}")

configs = [
    {'hidden1_size': 2, 'hidden2_size': 2, 'lr': 0.01},
    {'hidden1_size': 4, 'hidden2_size': 4, 'lr': 0.01},
    {'hidden1_size': 8, 'hidden2_size': 4, 'lr': 0.01},
    {'hidden1_size': 4, 'hidden2_size': 4, 'lr': 0.001},
    {'hidden1_size': 4, 'hidden2_size': 4, 'lr': 0.05},
    {'hidden1_size': 8, 'hidden2_size': 8, 'lr': 0.01},
]

X_norm, _ = normalize_all_features(X_raw, method="zscore")
print(f"\n  Testing {len(configs)} configurations (50 epochs each)...")
results = grid_search(X_norm, Y, configs, epochs=50)

print(f"\n  {'Rank':>4}  {'H1':>4}  {'H2':>4}  {'LR':>8}  {'MSE':>10}")
print(f"  {'-' * 36}")
for rank, (cfg, mse) in enumerate(results, 1):
    print(f"  {rank:>4}  {cfg['hidden1_size']:>4}  {cfg['hidden2_size']:>4}  "
          f"{cfg['lr']:>8.4f}  {mse:>10.4f}")
print(f"\n  Best config: H1={results[0][0]['hidden1_size']}, "
      f"H2={results[0][0]['hidden2_size']}, LR={results[0][0]['lr']}")
print(f"{'=' * 70}")
