# MLP From Scratch — Regression with Backpropagation

Build a **single-file Python program** that implements a Multi-Layer Perceptron from scratch — no NumPy, no TensorFlow, no libraries — using only raw Python math. The network will predict student **placement packages** (in lakhs) from **CGPA** and **Profile Score**.

---

## 1. Network Architecture

```
Input Layer (2 neurons)              Hidden Layer (2 neurons)              Output Layer (1 neuron)
┌──────────┐                         ┌──────────┐
│  x1      │──── w1 ───────────────▶│  h1      │
│  (CGPA)  │                         │          │──── w5 ──────────┐
│          │──── w2 ──────┐   ┌────▶│          │                  │      ┌──────────┐
└──────────┘              │   │      └──────────┘                  ├─────▶│  y_hat   │
                          │   │          ▲ b1                      │      │ (Package) │
                          ▼   │                                    │      └──────────┘
┌──────────┐              │   │      ┌──────────┐                  │          ▲ b3
│  x2      │──── w4 ──────┼───┘     │  h2      │                  │
│ (Profile)│              └────────▶│          │──── w6 ──────────┘
│          │──── w3 ───────────────▶│          │
└──────────┘                         └──────────┘
                                         ▲ b2
```

> **NOTE:**
> There is **1 hidden layer with 2 neurons** (h1 and h2). Both inputs (x1, x2) connect to both hidden neurons. Both hidden neurons connect to the single output neuron. This gives exactly **9 learnable parameters** (6 weights + 3 biases).

### Connection Table (6 weights)

| Weight | From → To |
|--------|-----------|
| `w1` | x1 (CGPA) → h1 |
| `w2` | x2 (Profile) → h1 |
| `w3` | x1 (CGPA) → h2 |
| `w4` | x2 (Profile) → h2 |
| `w5` | h1 → output |
| `w6` | h2 → output |

| Bias | Applied At |
|------|------------|
| `b1` | Hidden neuron h1 |
| `b2` | Hidden neuron h2 |
| `b3` | Output neuron |

**Total: 6 weights + 3 biases = 9 parameters** ✅

---

## 2. Dataset (4 Rows)

| CGPA (x1) | Profile Score (x2) | Package in LPA (y) |
|-----------|--------------------|--------------------|
| 8 | 4 | 15 |
| 5 | 2 | 7 |
| 7 | 3 | 12 |
| 3 | 1 | 4 |

---

## 3. Step-by-Step Implementation Plan

### Step 1 — Define the Dataset

Store the 4 data points as simple Python lists. No pandas, no CSV.

```python
X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
```

---

### Step 2 — Initialize All 9 Parameters

Hardcode initial values for all weights and biases (small random-ish values).

```python
w1, w2, w3, w4, w5, w6 = 0.1, 0.2, 0.3, 0.15, 0.25, 0.05
b1, b2, b3 = 0.1, 0.1, 0.1
```

---

### Step 3 — Define the Activation Function

Since this is a **regression** problem, we will use **Sigmoid** for the hidden layer and **linear (identity)** for the output layer.

```python
def sigmoid(z):
    return 1 / (1 + exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

> **NOTE:**
> We use sigmoid in the hidden layer to introduce non-linearity. The output layer is **linear** (no activation) since we're predicting a continuous value (package in lakhs).

---

### Step 4 — Forward Pass (for one data point)

Compute the predicted output step by step:

```
Hidden Neuron h1:
    z_h1 = w1*x1 + w2*x2 + b1
    a_h1 = sigmoid(z_h1)

Hidden Neuron h2:
    z_h2 = w3*x1 + w4*x2 + b2
    a_h2 = sigmoid(z_h2)

Output Neuron:
    z_out = w5*a_h1 + w6*a_h2 + b3
    y_hat = z_out  (linear activation)
```

---

### Step 5 — Loss Function (Mean Squared Error)

For all 4 data points:

```
MSE = (1/n) * Σ (yᵢ - ŷᵢ)²
```

Where `n = 4`.

---

### Step 6 — Backward Pass (Backpropagation Math)

This is the core. We compute the **partial derivative of the loss with respect to each of the 9 parameters** using the chain rule.

#### 6a. Derivative of Loss w.r.t. Output

For a single data point:

```
dL/dy_hat = (2/n) * (y_hat - y)
```

#### 6b. Output Layer Gradients

Since output activation is linear (`y_hat = z_out`), `dy_hat/dz_out = 1`:

```
dL/dw5 = dL/dy_hat * a_h1
dL/dw6 = dL/dy_hat * a_h2
dL/db3 = dL/dy_hat
```

#### 6c. Hidden Neuron h1 Gradients

```
dL/dz_h1 = dL/dy_hat * w5 * sigmoid_derivative(z_h1)

dL/dw1 = dL/dz_h1 * x1
dL/dw2 = dL/dz_h1 * x2
dL/db1 = dL/dz_h1
```

#### 6d. Hidden Neuron h2 Gradients

```
dL/dz_h2 = dL/dy_hat * w6 * sigmoid_derivative(z_h2)

dL/dw3 = dL/dz_h2 * x1
dL/dw4 = dL/dz_h2 * x2
dL/db2 = dL/dz_h2
```

---

### Step 7 — Gradient Descent Update Rule

For each of the 9 parameters, apply:

```
parameter = parameter - learning_rate * (dL/d_parameter)
```

We accumulate gradients across all 4 data points, then update once per epoch.

**Hyperparameters:**
- `learning_rate = 0.01`
- `epochs = 5`

---

### Step 8 — Training Loop (5 Epochs)

```
For epoch 1 to 5:
    Initialize all 9 gradient accumulators to 0
    For each data point (x1, x2, y):
        1. Forward pass -> compute y_hat
        2. Compute loss contribution
        3. Backward pass -> compute all 9 partial derivatives
        4. Accumulate gradients
    Update all 9 parameters using gradient descent
    Print epoch number, MSE loss, and all 9 parameter values
```

---

### Step 9 — Print Results After Training

After 5 epochs, print:
- Final values of all 9 parameters
- Final MSE loss
- Predictions vs actual values for all 4 data points

---

## 4. Summary of All 9 Gradient Formulas

| # | Parameter | Gradient Formula |
|---|-----------|-----------------|
| 1 | `w1` | `dL/dz_h1 * x1` |
| 2 | `w2` | `dL/dz_h1 * x2` |
| 3 | `b1` | `dL/dz_h1` |
| 4 | `w3` | `dL/dz_h2 * x1` |
| 5 | `w4` | `dL/dz_h2 * x2` |
| 6 | `b2` | `dL/dz_h2` |
| 7 | `w5` | `dL/dy_hat * a_h1` |
| 8 | `w6` | `dL/dy_hat * a_h2` |
| 9 | `b3` | `dL/dy_hat` |

---

## 5. File Structure

```
implementation.py    <- Single file, zero imports (except math.exp)
```

---

## 6. Verification Plan

### Manual Verification
- Run `python implementation.py`
- Confirm the loss **decreases** across the 5 epochs
- Confirm all 9 parameters are printed and updated each epoch
- Confirm predictions are printed alongside actual values after training

> **TIP:**
> If the loss doesn't decrease, the learning rate may need tuning. We'll start with `0.01` and adjust if needed.
