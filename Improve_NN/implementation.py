# ============================================================
# Improving Neural Networks -- All Solutions Implemented
# Pure Python (only math and random modules)
# ============================================================
# This file implements ALL the improvement techniques:
#   1. Weight Initialization (Xavier & He)
#   2. Activation Functions (ReLU, Leaky ReLU, Sigmoid, Tanh)
#   3. Batch Normalization
#   4. Gradient Clipping
#   5. Adam Optimizer
#   6. Learning Rate Scheduler (Step Decay)
#   7. L1 & L2 Regularization
#   8. Dropout
# ============================================================
# We use the SAME dataset from MLP (CGPA, Profile -> Package)
# but with a deeper network to show the improvements clearly.
#
# Architecture: 2 inputs -> Hidden1 (4 neurons) -> Hidden2 (4 neurons) -> 1 output
# ============================================================

import math
import random

# Set seed for reproducibility
random.seed(42)


# ================================================================
# SECTION 1: WEIGHT INITIALIZATION
# ================================================================

def random_init(fan_in, fan_out):
    """Naive random init -- causes vanishing/exploding gradients"""
    return random.uniform(-1, 1)

def xavier_init(fan_in, fan_out):
    """Xavier/Glorot Init -- best for Sigmoid/Tanh activations.
    Keeps variance stable: Var(W) = 2 / (fan_in + fan_out)"""
    std = math.sqrt(2.0 / (fan_in + fan_out))
    return random.gauss(0, std)

def he_init(fan_in, fan_out):
    """He Init -- best for ReLU activations.
    Accounts for ReLU killing half the neurons: Var(W) = 2 / fan_in"""
    std = math.sqrt(2.0 / fan_in)
    return random.gauss(0, std)


# ================================================================
# SECTION 2: ACTIVATION FUNCTIONS
# ================================================================

def sigmoid(z):
    """Sigmoid: output in (0, 1). Causes vanishing gradients."""
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)  # Max = 0.25 at z=0 -- gradient vanishes!

def tanh_activation(z):
    """Tanh: output in (-1, 1). Better than sigmoid, still can vanish."""
    z = max(-500, min(500, z))
    return math.tanh(z)

def tanh_derivative(z):
    t = math.tanh(z)
    return 1.0 - t * t  # Max = 1 at z=0

def relu(z):
    """ReLU: f(z) = max(0, z). Solves vanishing gradient for positive z.
    Problem: 'Dying ReLU' -- neuron outputs 0 forever if z < 0."""
    return max(0.0, z)

def relu_derivative(z):
    return 1.0 if z > 0 else 0.0

def leaky_relu(z, alpha=0.01):
    """Leaky ReLU: allows small gradient for negative z.
    Fixes the 'dying ReLU' problem."""
    return z if z > 0 else alpha * z

def leaky_relu_derivative(z, alpha=0.01):
    return 1.0 if z > 0 else alpha


# ================================================================
# SECTION 3: BATCH NORMALIZATION
# ================================================================

class BatchNorm:
    """Batch Normalization layer.
    Normalizes inputs to mean=0, std=1, then applies learnable scale (gamma) and shift (beta).

    Formula:
        z_norm = (z - mean) / sqrt(variance + epsilon)
        output = gamma * z_norm + beta
    """

    def __init__(self, size):
        self.gamma = [1.0] * size  # Learnable scale (init to 1)
        self.beta = [0.0] * size   # Learnable shift (init to 0)
        self.epsilon = 1e-8

        # For storing batch stats (used in backward pass)
        self.z_norm = None
        self.mean = None
        self.var = None
        self.z_input = None

    def forward(self, z_batch):
        """
        z_batch: list of lists [[z1_sample1, z2_sample1, ...], [z1_sample2, z2_sample2, ...]]
        Returns: normalized and scaled batch
        """
        batch_size = len(z_batch)
        feature_size = len(z_batch[0])

        # Step 1: Compute mean per feature
        self.mean = [0.0] * feature_size
        for sample in z_batch:
            for j in range(feature_size):
                self.mean[j] += sample[j]
        self.mean = [m / batch_size for m in self.mean]

        # Step 2: Compute variance per feature
        self.var = [0.0] * feature_size
        for sample in z_batch:
            for j in range(feature_size):
                self.var[j] += (sample[j] - self.mean[j]) ** 2
        self.var = [v / batch_size for v in self.var]

        # Step 3: Normalize
        self.z_input = z_batch
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

    def forward_single(self, z_values):
        """For single sample -- uses stored batch stats (simplified)."""
        feature_size = len(z_values)
        output = []
        z_norm_single = []
        for j in range(feature_size):
            if self.var is not None:
                zn = (z_values[j] - self.mean[j]) / math.sqrt(self.var[j] + self.epsilon)
            else:
                zn = z_values[j]
            z_norm_single.append(zn)
            output.append(self.gamma[j] * zn + self.beta[j])
        return output, z_norm_single


# ================================================================
# SECTION 4: GRADIENT CLIPPING
# ================================================================

def clip_by_value(gradient, max_val):
    """Clip individual gradient to [-max_val, max_val]."""
    return max(-max_val, min(max_val, gradient))

def clip_by_norm(gradients, max_norm):
    """Clip gradient vector by L2 norm.
    If ||g|| > max_norm, scale g down to have norm = max_norm."""
    total_norm = math.sqrt(sum(g ** 2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        gradients = [g * scale for g in gradients]
    return gradients, total_norm


# ================================================================
# SECTION 5: ADAM OPTIMIZER
# ================================================================

class AdamOptimizer:
    """Adam: Adaptive Moment Estimation.
    Combines momentum (1st moment) + RMSProp (2nd moment).

    m_t = beta1 * m_(t-1) + (1-beta1) * gradient
    v_t = beta2 * v_(t-1) + (1-beta2) * gradient^2
    m_hat = m_t / (1 - beta1^t)       (bias correction)
    v_hat = v_t / (1 - beta2^t)       (bias correction)
    param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
    """

    def __init__(self, num_params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0.0] * num_params  # 1st moment (mean of gradients)
        self.v = [0.0] * num_params  # 2nd moment (mean of squared gradients)
        self.t = 0                   # Time step

    def step(self, params, gradients):
        """Update parameters using Adam."""
        self.t += 1
        new_params = []
        for i in range(len(params)):
            # Update biased first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            # Update biased second moment
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradients[i] ** 2
            # Bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Update parameter
            new_params.append(params[i] - self.lr * m_hat / (math.sqrt(v_hat) + self.epsilon))
        return new_params


# ================================================================
# SECTION 6: LEARNING RATE SCHEDULER
# ================================================================

def step_decay(initial_lr, epoch, drop_factor=0.5, drop_every=20):
    """Step Decay: Reduce LR by factor every N epochs.
    Example: 0.01 -> 0.005 -> 0.0025 -> ..."""
    return initial_lr * (drop_factor ** (epoch // drop_every))

def exponential_decay(initial_lr, epoch, decay_rate=0.95):
    """Exponential Decay: lr = lr_0 * decay_rate^epoch"""
    return initial_lr * (decay_rate ** epoch)

def cosine_annealing(initial_lr, epoch, total_epochs, lr_min=0.0001):
    """Cosine Annealing: Smooth cosine curve from lr_max to lr_min"""
    return lr_min + 0.5 * (initial_lr - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))


# ================================================================
# SECTION 7: L1 AND L2 REGULARIZATION
# ================================================================

def l2_penalty(weights, lambda_val):
    """L2 Regularization penalty: lambda * sum(w^2)
    Gradient contribution: 2 * lambda * w (added to each weight's gradient)"""
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


# ================================================================
# SECTION 8: DROPOUT
# ================================================================

def apply_dropout(activations, drop_rate=0.3, training=True):
    """Inverted Dropout: During training, randomly zero out neurons
    and scale up remaining by 1/(1-p) so expected values stay the same.

    During inference, use all neurons without scaling.

    Args:
        activations: list of neuron values
        drop_rate: probability of dropping a neuron (e.g., 0.3 = 30%)
        training: True during training, False during inference
    Returns:
        (output, mask) -- output is the dropped-out activations, mask shows which survived
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


# ================================================================
# SECTION 9: FULL TRAINING DEMO -- Comparing Vanilla vs Improved
# ================================================================

# Dataset (same as MLP project)
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

# Architecture: 2 -> 4 -> 4 -> 1
INPUT_SIZE = 2
HIDDEN1_SIZE = 4
HIDDEN2_SIZE = 4
OUTPUT_SIZE = 1

EPOCHS = 100
INITIAL_LR = 0.01
L2_LAMBDA = 0.001       # L2 regularization strength
DROPOUT_RATE = 0.2       # 20% dropout in hidden layers
GRAD_CLIP_MAX = 5.0      # Gradient clipping threshold


def init_network(init_fn):
    """Initialize all weights and biases for the network."""
    net = {}

    # Hidden Layer 1: 2 inputs -> 4 neurons = 8 weights + 4 biases
    net['W1'] = [[init_fn(INPUT_SIZE, HIDDEN1_SIZE) for _ in range(INPUT_SIZE)] for _ in range(HIDDEN1_SIZE)]
    net['b1'] = [0.0] * HIDDEN1_SIZE

    # Hidden Layer 2: 4 inputs -> 4 neurons = 16 weights + 4 biases
    net['W2'] = [[init_fn(HIDDEN1_SIZE, HIDDEN2_SIZE) for _ in range(HIDDEN1_SIZE)] for _ in range(HIDDEN2_SIZE)]
    net['b2'] = [0.0] * HIDDEN2_SIZE

    # Output Layer: 4 inputs -> 1 neuron = 4 weights + 1 bias
    net['W3'] = [[init_fn(HIDDEN2_SIZE, OUTPUT_SIZE) for _ in range(HIDDEN2_SIZE)]]
    net['b3'] = [0.0]

    return net


def get_all_weights(net):
    """Flatten all weights into a single list (for regularization)."""
    weights = []
    for key in ['W1', 'W2', 'W3']:
        for row in net[key]:
            weights.extend(row)
    return weights


def forward_pass(net, x, activation_fn, act_derivative, training=True):
    """Forward pass through the network with dropout."""
    cache = {}

    # -- Hidden Layer 1 --
    z1 = []
    for j in range(HIDDEN1_SIZE):
        z = sum(net['W1'][j][k] * x[k] for k in range(INPUT_SIZE)) + net['b1'][j]
        z1.append(z)
    cache['z1'] = z1

    a1 = [activation_fn(z) for z in z1]
    cache['a1_pre_dropout'] = a1[:]

    # Apply Dropout to Hidden Layer 1
    a1, mask1 = apply_dropout(a1, DROPOUT_RATE, training)
    cache['a1'] = a1
    cache['mask1'] = mask1

    # -- Hidden Layer 2 --
    z2 = []
    for j in range(HIDDEN2_SIZE):
        z = sum(net['W2'][j][k] * a1[k] for k in range(HIDDEN1_SIZE)) + net['b2'][j]
        z2.append(z)
    cache['z2'] = z2

    a2 = [activation_fn(z) for z in z2]
    cache['a2_pre_dropout'] = a2[:]

    # Apply Dropout to Hidden Layer 2
    a2, mask2 = apply_dropout(a2, DROPOUT_RATE, training)
    cache['a2'] = a2
    cache['mask2'] = mask2

    # -- Output Layer (Linear) --
    z3 = sum(net['W3'][0][k] * a2[k] for k in range(HIDDEN2_SIZE)) + net['b3'][0]
    y_hat = z3  # Linear activation for regression
    cache['z3'] = z3
    cache['y_hat'] = y_hat

    return y_hat, cache


def backward_pass(net, x, y, cache, act_derivative):
    """Backward pass computing all gradients."""
    grads = {}
    y_hat = cache['y_hat']
    error = y_hat - y

    # dL/dy_hat = 2/n * (y_hat - y)
    dL_dy = (2.0 / n) * error

    # -- Output Layer Gradients --
    # dL/dW3[0][k] = dL/dy * a2[k]
    grads['dW3'] = [[dL_dy * cache['a2'][k] for k in range(HIDDEN2_SIZE)]]
    grads['db3'] = [dL_dy]

    # -- Hidden Layer 2 Gradients --
    # dL/da2[k] = dL/dy * W3[0][k]
    dL_da2 = [dL_dy * net['W3'][0][k] for k in range(HIDDEN2_SIZE)]

    # Apply dropout mask (gradients don't flow through dropped neurons)
    dL_da2 = [dL_da2[k] * cache['mask2'][k] for k in range(HIDDEN2_SIZE)]
    # Scale for inverted dropout
    if DROPOUT_RATE > 0:
        scale = 1.0 / (1.0 - DROPOUT_RATE)
        dL_da2 = [d * scale for d in dL_da2]

    dL_dz2 = [dL_da2[j] * act_derivative(cache['z2'][j]) for j in range(HIDDEN2_SIZE)]

    grads['dW2'] = [[dL_dz2[j] * cache['a1'][k] for k in range(HIDDEN1_SIZE)] for j in range(HIDDEN2_SIZE)]
    grads['db2'] = dL_dz2[:]

    # -- Hidden Layer 1 Gradients --
    dL_da1 = [0.0] * HIDDEN1_SIZE
    for k in range(HIDDEN1_SIZE):
        for j in range(HIDDEN2_SIZE):
            dL_da1[k] += dL_dz2[j] * net['W2'][j][k]

    # Apply dropout mask
    dL_da1 = [dL_da1[k] * cache['mask1'][k] for k in range(HIDDEN1_SIZE)]
    if DROPOUT_RATE > 0:
        scale = 1.0 / (1.0 - DROPOUT_RATE)
        dL_da1 = [d * scale for d in dL_da1]

    dL_dz1 = [dL_da1[j] * act_derivative(cache['z1'][j]) for j in range(HIDDEN1_SIZE)]

    grads['dW1'] = [[dL_dz1[j] * x[k] for k in range(INPUT_SIZE)] for j in range(HIDDEN1_SIZE)]
    grads['db1'] = dL_dz1[:]

    # -- Add L2 Regularization Gradients --
    for j in range(HIDDEN1_SIZE):
        for k in range(INPUT_SIZE):
            grads['dW1'][j][k] += l2_gradient(net['W1'][j][k], L2_LAMBDA)
    for j in range(HIDDEN2_SIZE):
        for k in range(HIDDEN1_SIZE):
            grads['dW2'][j][k] += l2_gradient(net['W2'][j][k], L2_LAMBDA)
    for k in range(HIDDEN2_SIZE):
        grads['dW3'][0][k] += l2_gradient(net['W3'][0][k], L2_LAMBDA)

    return grads


def flatten_gradients(grads):
    """Flatten all gradients into a single list for clipping."""
    flat = []
    for key in ['dW1', 'dW2', 'dW3']:
        for row in grads[key]:
            flat.extend(row)
    for key in ['db1', 'db2', 'db3']:
        flat.extend(grads[key])
    return flat


def unflatten_gradients(flat, grads_template):
    """Reconstruct gradient structure from flat list."""
    grads = {}
    idx = 0

    for key in ['dW1', 'dW2', 'dW3']:
        grads[key] = []
        for row in grads_template[key]:
            new_row = flat[idx:idx + len(row)]
            grads[key].append(new_row)
            idx += len(new_row)

    for key in ['db1', 'db2', 'db3']:
        size = len(grads_template[key])
        grads[key] = flat[idx:idx + size]
        idx += size

    return grads


def flatten_params(net):
    """Flatten network parameters into a single list."""
    flat = []
    for key in ['W1', 'W2', 'W3']:
        for row in net[key]:
            flat.extend(row)
    for key in ['b1', 'b2', 'b3']:
        flat.extend(net[key])
    return flat


def unflatten_params(flat, net):
    """Write flat parameters back into network structure."""
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


# ================================================================
# TRAINING FUNCTION
# ================================================================

def train(mode="improved"):
    """Train the network. mode = 'vanilla' or 'improved'."""

    if mode == "vanilla":
        init_fn = random_init
        act_fn = sigmoid
        act_deriv = sigmoid_derivative
        use_adam = False
        use_lr_schedule = False
        use_l2 = False
        use_dropout = False
        use_grad_clip = False
        label = "VANILLA (Sigmoid + Random Init + SGD)"
    else:
        init_fn = he_init
        act_fn = leaky_relu
        act_deriv = leaky_relu_derivative
        use_adam = True
        use_lr_schedule = True
        use_l2 = True
        use_dropout = True
        use_grad_clip = True
        label = "IMPROVED (LeakyReLU + He Init + Adam + LR Sched + L2 + Dropout + Grad Clip)"

    # Temporarily override globals for vanilla mode
    global DROPOUT_RATE, L2_LAMBDA
    orig_dropout = DROPOUT_RATE
    orig_l2 = L2_LAMBDA
    if not use_dropout:
        DROPOUT_RATE = 0.0
    if not use_l2:
        L2_LAMBDA = 0.0

    random.seed(42)
    net = init_network(init_fn)

    total_params = len(flatten_params(net))

    # Adam optimizer
    if use_adam:
        adam = AdamOptimizer(total_params, lr=INITIAL_LR)

    print(f"\n{'=' * 70}")
    print(f"  MODE: {label}")
    print(f"  Total parameters: {total_params}")
    print(f"{'=' * 70}")

    loss_history = []

    for epoch in range(1, EPOCHS + 1):
        # Learning rate schedule
        if use_lr_schedule:
            lr = cosine_annealing(INITIAL_LR, epoch, EPOCHS)
        else:
            lr = INITIAL_LR

        # Accumulate gradients over all samples
        acc_grads = None
        total_loss = 0.0

        for i in range(n):
            x = X[i]
            y = Y[i]

            # Forward pass
            y_hat, cache = forward_pass(net, x, act_fn, act_deriv, training=use_dropout)

            # Loss
            error = y_hat - y
            total_loss += error ** 2

            # Backward pass
            grads = backward_pass(net, x, y, cache, act_deriv)

            # Accumulate gradients
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

        # L2 regularization loss
        if use_l2:
            all_weights = get_all_weights(net)
            reg_loss = l2_penalty(all_weights, L2_LAMBDA)
            total_loss_with_reg = mse + reg_loss
        else:
            total_loss_with_reg = mse

        # Flatten gradients for clipping
        flat_grads = flatten_gradients(acc_grads)

        # Gradient Clipping
        if use_grad_clip:
            flat_grads, grad_norm = clip_by_norm(flat_grads, GRAD_CLIP_MAX)
        else:
            grad_norm = math.sqrt(sum(g ** 2 for g in flat_grads))

        # Update parameters
        flat_params = flatten_params(net)

        if use_adam:
            adam.lr = lr  # Update Adam's LR with scheduler
            flat_params = adam.step(flat_params, flat_grads)
        else:
            # Vanilla SGD
            flat_params = [p - lr * g for p, g in zip(flat_params, flat_grads)]

        net = unflatten_params(flat_params, net)
        loss_history.append(mse)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  |  MSE: {mse:>10.4f}  |  LR: {lr:.6f}  |  Grad Norm: {grad_norm:.4f}")

    # ---- Final predictions ----
    print(f"\n  {'- ' * 35}")
    print(f"  FINAL PREDICTIONS:")
    print(f"  {'CGPA':>6}  {'Profile':>8}  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    print(f"  {'-' * 46}")

    final_loss = 0.0
    for i in range(n):
        y_hat, _ = forward_pass(net, X[i], act_fn, act_deriv, training=False)
        error = y_hat - Y[i]
        final_loss += error ** 2
        print(f"  {X[i][0]:>6}  {X[i][1]:>8}  {Y[i]:>8}  {y_hat:>10.4f}  {error:>+8.4f}")

    print(f"\n  Final MSE: {final_loss / n:.4f}")
    print(f"{'=' * 70}")

    # Restore globals
    DROPOUT_RATE = orig_dropout
    L2_LAMBDA = orig_l2

    return loss_history


# ================================================================
# RUN COMPARISON
# ================================================================

print("\n" + "#" * 70)
print("#" + " " * 20 + "NEURAL NETWORK IMPROVEMENT" + " " * 22 + "#")
print("#" + " " * 15 + "Vanilla vs Improved Comparison" + " " * 23 + "#")
print("#" * 70)

# Run vanilla training
vanilla_losses = train(mode="vanilla")

# Run improved training
improved_losses = train(mode="improved")

# ---- Summary ----
print(f"\n{'=' * 70}")
print("  COMPARISON SUMMARY")
print(f"{'=' * 70}")
print(f"  {'Metric':<30} {'Vanilla':>15} {'Improved':>15}")
print(f"  {'-' * 60}")
print(f"  {'Initial MSE (Epoch 1)':<30} {vanilla_losses[0]:>15.4f} {improved_losses[0]:>15.4f}")
print(f"  {'Final MSE (Epoch 100)':<30} {vanilla_losses[-1]:>15.4f} {improved_losses[-1]:>15.4f}")
print(f"  {'Best MSE':<30} {min(vanilla_losses):>15.4f} {min(improved_losses):>15.4f}")
print(f"  {'Improvement':<30} {'':>15} {((vanilla_losses[-1] - improved_losses[-1]) / vanilla_losses[-1] * 100):>14.1f}%")
print(f"{'=' * 70}")

print("\n  Techniques used in IMPROVED mode:")
print("    [x] He Weight Initialization (instead of random)")
print("    [x] Leaky ReLU Activation (instead of sigmoid)")
print("    [x] Adam Optimizer (instead of vanilla SGD)")
print("    [x] Cosine Annealing LR Schedule")
print("    [x] L2 Regularization (lambda=0.001)")
print("    [x] Dropout (rate=0.2)")
print("    [x] Gradient Clipping (max_norm=5.0)")
print()
