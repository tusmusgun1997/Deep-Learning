# Improving Neural Network Performance

A comprehensive guide covering **7 key areas** to improve neural network training, convergence, and generalization.

---

## Overview

```
                    IMPROVING NEURAL NETWORK PERFORMANCE
                                    |
    ┌──────────┬──────────┬─────────┴──┬──────────┬──────────┬──────────────┐
    |          |          |            |          |          |              |
 1.Over-   2.Normal-  3.Vanishing  4.Gradient 5.Opti-  6.Learning  7.Hyper-
  fitting   ization    Gradients    Checking   mizers   Rate        parameter
    |          |          |          & Clipping   |      Scheduling   Tuning
    |          |          |            |          |          |              |
 -Dropout  -Normaliz- -Activation  -Gradient  -Momentum  -Step        -Num layers
 -L1/L2     ing        Functions    Checking   -Adagrad   Decay       -Nodes/layer
  Reg.      Inputs    -Weight      -Gradient   -RMSprop  -Exponential -Batch size
 -Early    -Batch      Init.       Clipping    -Adam      Decay
  Stopping  Norm                                         -Cosine
                                                          Annealing
```

---

## 1. Overfitting

**What it is:** The model memorizes training data (including noise) instead of learning general patterns. It performs great on training data but terribly on unseen data.

```
Training Loss:    0.001  (amazing!)
Validation Loss:  2.500  (terrible!)
                  ^^^^^
               OVERFITTING -- the model memorized noise
```

**Signs:**
- Training accuracy >> Validation accuracy
- Training loss keeps decreasing while validation loss increases
- Model is very confident but wrong on new data

---

### 1.1 Dropout Layers

**Concept:** During training, randomly "turn off" neurons with probability `p` (typically 0.2 to 0.5). During inference, use all neurons.

```
Training (dropout = 0.5):

Layer input:  [0.8,  0.3,  0.5,  0.9,  0.2]
Random mask:  [ 1,    0,    1,    0,    1 ]    (50% dropped)
Layer output: [0.8,  0.0,  0.5,  0.0,  0.2]   (dropped neurons output 0)
```

**Inverted Dropout (more common):**
```
Training:  Scale UP active neurons by 1/(1-p) so expected value stays the same
Inference: Use all neurons without any scaling
```

**Why it works:**
- **Prevents co-adaptation:** Neurons can't rely on specific other neurons being present
- **Ensemble effect:** Each training step uses a different random sub-network. The final model behaves like an ensemble of 2^n networks
- **Adds noise:** Acts as regularization, forces redundancy

**Where to apply:** Between hidden layers, NOT on the output layer. Higher dropout in larger layers.

---

### 1.2 Regularization (L1 & L2)

**Concept:** Add a penalty term to the loss function that punishes large weights, forcing the model to find simpler solutions.

**L2 Regularization (Ridge / Weight Decay):**
```
Loss = Original_Loss + lambda * sum(W_i^2)

Gradient update becomes:
W = W - lr * (dL/dW + 2 * lambda * W)
W = W * (1 - 2 * lr * lambda) - lr * dL/dW
                ^^^^^^^^^^^^^^^
                "weight decay" -- weights shrink every step
```

**L1 Regularization (Lasso):**
```
Loss = Original_Loss + lambda * sum(|W_i|)

Gradient update becomes:
W = W - lr * (dL/dW + lambda * sign(W))
```

**Comparison:**

| Property | L1 | L2 |
|----------|----|----|
| Penalty | Sum of absolute weights | Sum of squared weights |
| Effect | Drives weights to exactly 0 (sparse) | Drives weights close to 0 (small) |
| Feature Selection | Yes (zeros out useless features) | No (keeps all, just shrinks) |
| Best for | High-dimensional, many irrelevant features | General use, smooth solutions |

**Typical lambda values:** 0.0001 to 0.01

---

### 1.3 Early Stopping

**Concept:** Monitor the **validation loss** during training. When it stops improving (or starts increasing), stop training — even if training loss is still decreasing.

```
Epoch  | Train Loss | Val Loss  | Action
-------|------------|-----------|--------
  1    |   5.000    |  4.800    | Improving...
  10   |   2.000    |  1.900    | Improving...
  20   |   0.500    |  0.800    | Val loss increasing!
  25   |   0.200    |  1.200    | STOP! Overfitting started at epoch ~15
  30   |   0.050    |  2.000    | (would have gotten worse)
```

**How it works:**
1. Split data into **training set** and **validation set**
2. After each epoch, evaluate loss on validation set
3. Track the **best validation loss** seen so far
4. If validation loss hasn't improved for `patience` epochs → STOP
5. Restore weights from the epoch with the best validation loss

**Parameters:**
- **patience:** Number of epochs to wait for improvement (typically 5-20)
- **min_delta:** Minimum change to qualify as an improvement (e.g., 0.001)

**Why it works:**
- The point where validation loss starts rising is where the model transitions from learning real patterns to memorizing noise
- It's a form of regularization — limits model capacity by limiting training time
- **Free to implement** — no additional hyperparameters to tune beyond patience

```python
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate_validation()

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_weights = copy_weights()  # Save best model
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping! Restoring best weights.")
        restore_weights(best_weights)
        break
```

---

## 2. Normalization

### 2.1 Normalizing Inputs

**What it is:** Scale input features so they're on a similar range before feeding them to the network.

**The Problem Without Normalization:**
Consider our dataset:
```
CGPA:    [3, 5, 7, 8]      Range: 0-10
Profile: [1, 2, 3, 4]      Range: 0-5
Package: [4, 7, 12, 15]    Range: 0-20+
```

If CGPA values are much larger than Profile values, the gradients for CGPA-connected weights will dominate, causing:
- Elongated loss surface (elliptical contours instead of circular)
- Gradient descent oscillates in steep dimensions, crawls in flat ones
- Much slower convergence

**Methods:**

**a) Min-Max Normalization (scales to [0, 1]):**
```
x_norm = (x - x_min) / (x_max - x_min)

Example: CGPA=7, min=3, max=8
x_norm = (7 - 3) / (8 - 3) = 0.8
```

**b) Z-Score Normalization / Standardization (scales to mean=0, std=1):**
```
x_norm = (x - mean) / std

Example: CGPA=7, mean=5.75, std=1.92
x_norm = (7 - 5.75) / 1.92 = 0.65
```

**c) Max-Abs Normalization (scales to [-1, 1]):**
```
x_norm = x / max(|x|)
```

**Which to use:**
| Method | When to Use |
|--------|------------|
| Min-Max | When you know fixed bounds (e.g., pixel values 0-255) |
| Z-Score | General purpose, handles outliers better, most common |
| Max-Abs | When data is already centered around 0 |

**Critical Rule:** Compute mean/std from **training data only**, then apply the same transform to validation and test data. Never leak test statistics!

---

### 2.2 Batch Normalization

**What it does:** Normalizes the input to each layer within a mini-batch so it has **mean~0 and std~1**, then scales and shifts with learnable parameters.

```
For a mini-batch of values at a hidden layer:

1. Compute batch mean:     mu = (1/m) * sum(z_i)
2. Compute batch variance:  sigma^2 = (1/m) * sum((z_i - mu)^2)
3. Normalize:              z_norm = (z_i - mu) / sqrt(sigma^2 + epsilon)
4. Scale and shift:        y_i = gamma * z_norm + beta

Where gamma and beta are LEARNABLE parameters.
```

**Why it works:**
- Prevents **internal covariate shift** (distribution of inputs to layers keeps changing during training)
- Smooths the loss landscape → faster, more stable training
- Acts as mild regularization (noise from batch statistics)
- Allows using higher learning rates

**Where to apply:** Between the linear computation (z = Wx + b) and the activation function.

**At inference time:** Use running averages of mean and variance (computed during training) instead of batch statistics.

---

## 3. Vanishing Gradients

### 3.1 Activation Functions

**Why Sigmoid Causes Vanishing Gradients:**
```
Sigmoid:        sigma(z) = 1 / (1 + e^(-z))
Derivative:     sigma'(z) = sigma(z) * (1 - sigma(z))
Max derivative: 0.25 (at z=0)
```

After 10 layers: `0.25^10 = 0.00000095` — gradient is essentially **dead**.

**Better Alternatives:**

| Activation | Formula | Derivative | Solves Vanishing? |
|-----------|---------|-----------|-------------------|
| **ReLU** | `max(0, z)` | `1 if z > 0, else 0` | Yes! Gradient is 1 for positive z |
| **Leaky ReLU** | `z if z > 0, else 0.01*z` | `1 if z > 0, else 0.01` | Yes, and avoids "dying ReLU" |
| **ELU** | `z if z > 0, else alpha*(e^z - 1)` | `1 if z > 0, else f(z) + alpha` | Yes, smooth & zero-centered |
| **Tanh** | `(e^z - e^-z) / (e^z + e^-z)` | `1 - tanh(z)^2`, max=1 | Better than sigmoid, still can vanish |

**ReLU is the default choice for hidden layers in modern networks.**

**Dying ReLU Problem:** If a neuron's input is always negative, its gradient is always 0, and it never updates. Leaky ReLU fixes this by allowing a small gradient (0.01) for negative inputs.

---

### 3.2 Weight Initialization

**The Problem with Random Init:**
If weights are initialized too large -> exploding gradients. Too small -> vanishing gradients.

**Xavier/Glorot Initialization (for sigmoid/tanh):**
```
W ~ Normal(mean=0, std = sqrt(2 / (fan_in + fan_out)))
```
Where `fan_in` = number of input neurons, `fan_out` = number of output neurons.

**He Initialization (for ReLU):**
```
W ~ Normal(mean=0, std = sqrt(2 / fan_in))
```

**Why it works:** Keeps the variance of activations roughly constant across layers, so gradients neither grow nor shrink as they flow backward.

| Initialization | Best With | Formula |
|---------------|-----------|---------|
| Xavier/Glorot | Sigmoid, Tanh | `std = sqrt(2 / (fan_in + fan_out))` |
| He | ReLU, Leaky ReLU | `std = sqrt(2 / fan_in)` |
| LeCun | SELU | `std = sqrt(1 / fan_in)` |

---

## 4. Gradient Checking and Clipping

### 4.1 Gradient Checking (Numerical Gradient Verification)

**What it is:** A debugging technique to verify that your backpropagation implementation is correct by comparing analytical gradients (from backprop) with numerical gradients (from finite differences).

**How it works:**
For each parameter theta, compute the numerical gradient using the **two-sided finite difference**:

```
                     L(theta + epsilon) - L(theta - epsilon)
numerical_grad  =  -----------------------------------------
                                2 * epsilon

Where epsilon is a tiny value (e.g., 1e-7)
```

**Then compare with your backprop gradient:**
```
                     |analytical_grad - numerical_grad|
relative_error  =  ------------------------------------------
                   |analytical_grad| + |numerical_grad| + eps

If relative_error < 1e-5  -> Backprop is CORRECT
If relative_error < 1e-3  -> Might be OK, check carefully
If relative_error > 1e-3  -> Backprop has a BUG!
```

**Important notes:**
- **Only use for debugging** — it's extremely slow (2 forward passes per parameter)
- Turn off dropout and batch norm during gradient checking
- Use double precision (float64) for accuracy
- Check a few parameters, not all (for large networks)

---

### 4.2 Gradient Clipping

**What it does:** If the gradient magnitude exceeds a threshold, scale it down to prevent exploding gradients.

**Clip by Value:**
```
Each gradient element is clipped to [-max_val, max_val]
gradient = max(-max_val, min(max_val, gradient))
```

**Clip by Norm (more common):**
```
If ||gradient_vector|| > threshold:
    gradient_vector = gradient_vector * (threshold / ||gradient_vector||)
```

This preserves the **direction** of the gradient while limiting its magnitude.

**When to use:** Essential for RNNs/LSTMs, recommended for deep networks. Typical threshold: 1.0 to 5.0.

---

## 5. Optimizers

All optimizers solve the same problem: **update weights to minimize the loss function**. They differ in how smartly they do it.

### 5.1 Momentum

**Problem with vanilla SGD:** Gets stuck in ravines (narrow valleys), oscillates back and forth.

**Solution:** Accumulate a "velocity" from past gradients — like a ball rolling downhill gains momentum.

```
v_t = beta * v_(t-1) + (1 - beta) * gradient_t
W = W - learning_rate * v_t

Typical beta = 0.9 (remembers ~10 past gradients)
```

**Why it works:**
- Accelerates in consistent gradient directions (down the valley)
- Dampens oscillations in inconsistent directions (across the valley)
- Helps escape shallow local minima

```
Without Momentum:     /\/\/\/\___/\___  (oscillates, slow)
With Momentum:        ──────\_________  (smooth, fast)
```

---

### 5.2 Adagrad (Adaptive Gradient)

**Problem:** Different parameters need different learning rates. Rare features need larger updates; frequent features need smaller ones.

**Solution:** Accumulate squared gradients per parameter, use them to scale learning rate.

```
s_t = s_(t-1) + gradient_t^2             (accumulate squared gradients)
W = W - learning_rate * gradient_t / (sqrt(s_t) + epsilon)
```

**Pros:**
- Automatically adapts learning rate per parameter
- Great for sparse data (NLP, recommendation systems)

**Cons:**
- Learning rate **monotonically decreases** (s_t only grows)
- Eventually learning rate becomes infinitesimally small and training stops
- This is why RMSprop was invented

---

### 5.3 RMSprop (Root Mean Square Propagation)

**Problem with Adagrad:** The accumulated squared gradients grow forever, killing the learning rate.

**Solution:** Use an **exponential moving average** of squared gradients instead of accumulating all of them.

```
s_t = beta * s_(t-1) + (1 - beta) * gradient_t^2      (moving average)
W = W - learning_rate * gradient_t / (sqrt(s_t) + epsilon)

Typical beta = 0.9
```

**Why it's better than Adagrad:**
- The moving average "forgets" old gradients, so the learning rate doesn't decay to zero
- Adapts to the recent landscape of the loss function

---

### 5.4 Adam (Adaptive Moment Estimation)

**The best of both worlds:** Combines Momentum (1st moment) + RMSprop (2nd moment).

```
m_t = beta1 * m_(t-1) + (1 - beta1) * gradient          # 1st moment (mean)
v_t = beta2 * v_(t-1) + (1 - beta2) * gradient^2         # 2nd moment (variance)

m_hat = m_t / (1 - beta1^t)                              # Bias correction
v_hat = v_t / (1 - beta2^t)                              # Bias correction

W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
```

**Default hyperparameters:** `beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001`

**Why Adam is the default:**
- Adapts learning rate per parameter (from RMSprop)
- Has momentum to smooth updates (from Momentum)
- Bias correction handles early training instability
- Works well out-of-the-box

**Optimizer Comparison:**

| Optimizer | Adapts LR? | Has Momentum? | Best For |
|-----------|-----------|--------------|----------|
| SGD | No | No | Simple problems |
| Momentum | No | Yes | General training |
| Adagrad | Yes | No | Sparse data (NLP) |
| RMSprop | Yes | No | RNNs, non-stationary |
| **Adam** | **Yes** | **Yes** | **Default for most tasks** |

---

## 6. Learning Rate Scheduling

**The Problem:** A fixed learning rate is suboptimal:
- Too high -> overshoots, never converges
- Too low -> takes forever, gets stuck in local minima
- **Ideal:** Start high (explore broadly), gradually decrease (fine-tune)

### 6.1 Step Decay

```
Reduce LR by a factor every N epochs.
Example: Start at 0.01, multiply by 0.1 every 30 epochs

  Epoch 1-30:   lr = 0.01
  Epoch 31-60:  lr = 0.001
  Epoch 61-90:  lr = 0.0001
```

### 6.2 Exponential Decay

```
lr_t = lr_0 * decay_rate^epoch
Example: lr_0=0.01, decay_rate=0.95
  Epoch 1:   0.0095
  Epoch 10:  0.0060
  Epoch 50:  0.0008
```

### 6.3 Cosine Annealing

```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

Smoothly decreases from max to min following a cosine curve. Very popular in modern training.

### 6.4 Warm-up + Decay

```
Phase 1 (warmup):  Linearly increase LR from 0 to lr_max over first N steps
Phase 2 (decay):   Decrease LR using cosine/step/exponential
```

Used in Transformers — prevents early training instability when weights are still random.

### 6.5 ReduceLROnPlateau

```
If validation loss hasn't improved for N epochs -> reduce LR by factor
```

Adaptive — responds to actual training dynamics. Combines well with Early Stopping.

---

## 7. Hyperparameter Tuning

Hyperparameters are settings you choose **before** training — they're not learned by the network. The wrong choice can make or break performance.

### 7.1 Number of Hidden Layers

| Depth | Capability | Risk |
|-------|-----------|------|
| 0 hidden layers | Linear model only | Underfitting |
| 1 hidden layer | Can approximate any function (universal approximation) | May need many neurons |
| 2-3 hidden layers | Good for most problems | Balance of power and trainability |
| 5+ hidden layers | Complex hierarchical features | Vanishing gradients, overfitting |
| 50+ layers | Only with ResNet/skip connections | Requires special architectures |

**Rule of thumb:** Start with 1-2 hidden layers. Add more only if underfitting.

### 7.2 Nodes per Layer

**Too few nodes:** Model can't learn complex patterns (underfitting)
**Too many nodes:** Model memorizes training data (overfitting), slow training

**Common Strategies:**
- **Funnel shape:** Decreasing neurons per layer (e.g., 128 -> 64 -> 32)
- **Constant width:** Same neurons per layer (e.g., 64 -> 64 -> 64)
- **Rule of thumb:** Start with `2 * num_input_features` and adjust

```
Underfitting?  -> Increase nodes/layers
Overfitting?   -> Decrease nodes/layers (or add dropout/regularization)
```

### 7.3 Batch Size

**What it is:** Number of samples processed before one weight update.

| Batch Size | Name | Pros | Cons |
|-----------|------|------|------|
| 1 | Stochastic GD | Noisy updates, good for escaping local minima | Very slow, unstable |
| 16-64 | Mini-batch GD | Best trade-off: stable + fast | Need to tune |
| Full dataset | Batch GD | Smooth, stable gradients | Slow, memory-heavy, poor generalization |

**Effects of batch size:**
- **Small batch (16-32):** More noise -> better generalization, slower per epoch
- **Large batch (256-512):** Less noise -> faster per epoch, may overfit, sharp minima
- **Sweet spot:** Usually 32-128 for most tasks

### 7.4 Hyperparameter Search Methods

**a) Grid Search:** Try all combinations of predefined values
```
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [32, 64, 128]
-> Try all 9 combinations
```

**b) Random Search:** Sample random combinations — often better than grid search because it explores more of the space.

**c) Bayesian Optimization:** Uses past results to intelligently pick next combination to try. Most efficient but complex.

---

## Summary Table — Quick Reference

| # | Category | Technique | Key Idea | When to Use |
|---|----------|-----------|----------|-------------|
| 1a | Overfitting | Dropout | Randomly disable neurons | Dense layers, large networks |
| 1b | Overfitting | L1/L2 Regularization | Penalize large weights | When model is too complex |
| 1c | Overfitting | Early Stopping | Stop when val loss rises | Always -- free regularization |
| 2a | Normalization | Input Normalization | Scale inputs to similar range | Always -- costs nothing |
| 2b | Normalization | Batch Normalization | Normalize layer inputs | Deep networks (> 3 layers) |
| 3a | Vanishing Grad | Activation Functions (ReLU) | Gradient = 1 for positive z | Default for hidden layers |
| 3b | Vanishing Grad | Weight Init (Xavier/He) | Keep variance stable | Always -- costs nothing |
| 4a | Gradient Debug | Gradient Checking | Verify backprop with finite differences | Debugging only |
| 4b | Gradient Clip | Gradient Clipping | Cap gradient magnitude | RNNs, very deep networks |
| 5a | Optimizer | Momentum | Accumulate past gradients | General training speedup |
| 5b | Optimizer | Adagrad | Adapt LR per parameter | Sparse data |
| 5c | Optimizer | RMSprop | Moving avg of squared grads | RNNs, non-stationary |
| 5d | Optimizer | Adam | Momentum + RMSprop | Default optimizer |
| 6 | LR Schedule | Step/Cosine/Exponential | High LR early, low LR later | Always improves results |
| 7 | Hyperparams | Tuning (layers, nodes, batch) | Search for best config | After initial model works |
