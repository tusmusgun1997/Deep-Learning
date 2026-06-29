# ============================================================
# Multi-Layer Perceptron FROM SCRATCH  --  with Plots & Validation
# ------------------------------------------------------------
# Regression: predict student package (LPA) from CGPA & Profile.
#
# The neural network (forward pass, backprop, gradient descent) is
# written in PURE PYTHON -- no NumPy, no PyTorch. Matplotlib is used
# ONLY to draw the graphs.
#
# >>> Running in Google Colab:
#     This is a pure-Python implementation, so it runs on the CPU.
#     Selecting a GPU runtime will NOT make it faster (there is no
#     GPU code here) -- but it's lightweight and finishes in seconds.
#     Just paste this file into a Colab cell and run it.
#
# What you'll see:
#   1. A "learning progress" panel -> predicted vs actual at several
#      training checkpoints (watch the points snap onto the diagonal).
#   2. Training Loss vs Validation Loss over epochs.
#   3. A final results graph that clearly shows how well the model did.
# ============================================================

import math
import random
import matplotlib.pyplot as plt

random.seed(42)  # reproducible data, shuffling, and weight init

# ----------------------------------------------------------------
# STEP 1: Generate a LARGER synthetic dataset (from scratch)
# ----------------------------------------------------------------
# "True" (hidden) relationship the network must discover:
#     package = 1.4*CGPA + 1.1*Profile + 0.15*(CGPA*Profile) + noise
# The CGPA*Profile term adds mild non-linearity so the hidden layer
# actually has something useful to learn (beyond a straight line).
NUM_SAMPLES = 400
NOISE_STD   = 0.5   # measurement noise (sets the best achievable error)

def generate_dataset(num):
    data_X, data_Y = [], []
    for _ in range(num):
        cgpa    = random.uniform(4.0, 10.0)   # CGPA between 4 and 10
        profile = random.uniform(1.0, 5.0)    # Profile score between 1 and 5
        noise   = random.gauss(0.0, NOISE_STD)
        package = 1.4 * cgpa + 1.1 * profile + 0.15 * cgpa * profile + noise
        data_X.append([cgpa, profile])
        data_Y.append(package)
    return data_X, data_Y

X_all, Y_all = generate_dataset(NUM_SAMPLES)

# Shuffle, then split 80% train / 20% validation
combined = list(zip(X_all, Y_all))
random.shuffle(combined)
X_all, Y_all = map(list, zip(*combined))

split = int(0.8 * NUM_SAMPLES)
X_train, Y_train = X_all[:split], Y_all[:split]
X_val,   Y_val   = X_all[split:], Y_all[split:]
n_train, n_val = len(Y_train), len(Y_val)

# ----------------------------------------------------------------
# STEP 2: Standardize features AND target  (from scratch)
# ----------------------------------------------------------------
# Neural nets train far better when inputs are on a similar scale.
# CRITICAL RULE: compute statistics on TRAINING data only, then apply
# the SAME transform to validation data -- never peek at val/test stats.

def feature_stats(rows):
    num_features = len(rows[0])
    means, stds = [], []
    for f in range(num_features):
        vals = [r[f] for r in rows]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means.append(m)
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    return means, stds

def apply_feature_norm(rows, means, stds):
    return [[(r[f] - means[f]) / stds[f] for f in range(len(r))] for r in rows]

feat_means, feat_stds = feature_stats(X_train)
Xtr = apply_feature_norm(X_train, feat_means, feat_stds)
Xva = apply_feature_norm(X_val,   feat_means, feat_stds)

# Standardize the target too (predict in "standardized space", then
# convert predictions back to real LPA units for reporting/plots).
y_mean = sum(Y_train) / n_train
y_std  = math.sqrt(sum((y - y_mean) ** 2 for y in Y_train) / n_train)
Ytr = [(y - y_mean) / y_std for y in Y_train]
Yva = [(y - y_mean) / y_std for y in Y_val]

def to_lpa(std_value):
    """Convert a standardized prediction back to real LPA units."""
    return std_value * y_std + y_mean

# ----------------------------------------------------------------
# STEP 3: Network architecture  (2 -> HIDDEN sigmoid -> 1 linear)
# ----------------------------------------------------------------
# Same idea as the original 2->2->1 net, just generalized to HIDDEN
# neurons (still pure Python). Sigmoid hidden activation, linear output.
INPUT  = 2
HIDDEN = 16
LR     = 0.30
EPOCHS = 1000

def xavier_init(fan_in, fan_out):
    """Good initial spread for sigmoid/tanh layers."""
    return random.gauss(0.0, math.sqrt(2.0 / (fan_in + fan_out)))

# Layer 1 (input -> hidden): W1[j][k], b1[j]
W1 = [[xavier_init(INPUT, HIDDEN) for _ in range(INPUT)] for _ in range(HIDDEN)]
b1 = [0.0] * HIDDEN
# Layer 2 (hidden -> output): W2[j], b2
W2 = [xavier_init(HIDDEN, 1) for _ in range(HIDDEN)]
b2 = 0.0

def sigmoid(z):
    z = max(-500.0, min(500.0, z))   # clamp to avoid math.exp overflow
    return 1.0 / (1.0 + math.exp(-z))

def forward(x):
    """Return (hidden activations, prediction) for one input row."""
    a_h = []
    for j in range(HIDDEN):
        z = b1[j] + sum(W1[j][k] * x[k] for k in range(INPUT))
        a_h.append(sigmoid(z))
    y_hat = b2 + sum(W2[j] * a_h[j] for j in range(HIDDEN))   # linear output
    return a_h, y_hat

def dataset_mse(Xn, Yn):
    total = 0.0
    for i in range(len(Yn)):
        _, y_hat = forward(Xn[i])
        total += (y_hat - Yn[i]) ** 2
    return total / len(Yn)

def r2_score(actual, pred):
    mean_a = sum(actual) / len(actual)
    ss_tot = sum((a - mean_a) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, pred))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def val_predictions_lpa():
    """Current validation predictions, converted back to real LPA units."""
    return [to_lpa(forward(Xva[i])[1]) for i in range(n_val)]

# ----------------------------------------------------------------
# STEP 4: Training loop (full-batch gradient descent) + record history
# ----------------------------------------------------------------
train_losses, val_losses = [], []

# Snapshot the model's validation predictions at these checkpoints so we
# can SEE the learning progress (points moving onto the diagonal).
checkpoints = sorted(set([1, EPOCHS // 10, EPOCHS // 4, EPOCHS // 2, EPOCHS]))
snapshots = {}   # epoch -> list of predictions (LPA)

print("=" * 64)
print("  MLP FROM SCRATCH  --  Regression with Validation")
print(f"  Data: {NUM_SAMPLES} samples ({n_train} train / {n_val} val)")
print(f"  Network: {INPUT} -> {HIDDEN} (sigmoid) -> 1 (linear)")
print(f"  LR: {LR} | Epochs: {EPOCHS} | Full-batch gradient descent")
print("=" * 64)
print(f"  {'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12}")
print(f"  {'-' * 38}")

for epoch in range(1, EPOCHS + 1):
    # ---- gradient accumulators (reset every epoch) ----
    gW1 = [[0.0] * INPUT for _ in range(HIDDEN)]
    gb1 = [0.0] * HIDDEN
    gW2 = [0.0] * HIDDEN
    gb2 = 0.0
    total_loss = 0.0

    # ---- accumulate gradients over the whole training set ----
    for i in range(n_train):
        x, y = Xtr[i], Ytr[i]

        # forward
        a_h, y_hat = forward(x)
        error = y_hat - y
        total_loss += error ** 2

        # backward (chain rule)
        dL_dyhat = (2.0 / n_train) * error          # dL/d(y_hat)
        # output layer (linear): dy_hat/dz_out = 1
        for j in range(HIDDEN):
            gW2[j] += dL_dyhat * a_h[j]
        gb2 += dL_dyhat
        # hidden layer (sigmoid): a'(z) = a*(1-a)
        for j in range(HIDDEN):
            dL_dz = dL_dyhat * W2[j] * (a_h[j] * (1.0 - a_h[j]))
            for k in range(INPUT):
                gW1[j][k] += dL_dz * x[k]
            gb1[j] += dL_dz

    # ---- gradient-descent update (once per epoch) ----
    for j in range(HIDDEN):
        for k in range(INPUT):
            W1[j][k] -= LR * gW1[j][k]
        b1[j] -= LR * gb1[j]
        W2[j] -= LR * gW2[j]
    b2 -= LR * gb2

    # ---- record losses ----
    train_losses.append(total_loss / n_train)
    val_losses.append(dataset_mse(Xva, Yva))

    if epoch in checkpoints:
        snapshots[epoch] = val_predictions_lpa()

    if epoch == 1 or epoch % 100 == 0:
        print(f"  {epoch:>6} | {train_losses[-1]:>12.6f} | {val_losses[-1]:>12.6f}")

# ----------------------------------------------------------------
# STEP 5: Final metrics (in real LPA units)
# ----------------------------------------------------------------
val_pred = val_predictions_lpa()
rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(Y_val, val_pred)) / n_val)
mae  = sum(abs(a - p) for a, p in zip(Y_val, val_pred)) / n_val
r2   = r2_score(Y_val, val_pred)

print(f"\n{'=' * 64}")
print("  TRAINING COMPLETE -- VALIDATION RESULTS (real LPA units)")
print(f"{'=' * 64}")
print(f"  RMSE : {rmse:.3f} LPA   (noise floor ~ {NOISE_STD} LPA)")
print(f"  MAE  : {mae:.3f} LPA")
print(f"  R^2  : {r2:.4f}   (1.0 = perfect)")
print(f"{'=' * 64}")

# ================================================================
# PLOTS
# ================================================================

# ---- Plot 1: LEARNING PROGRESS at each checkpoint ----
# Predicted vs Actual on the validation set, captured at several epochs.
# Early on the cloud is shapeless; later it collapses onto the y=x line.
cps = sorted(snapshots.keys())
fig, axes = plt.subplots(1, len(cps), figsize=(3.4 * len(cps), 3.6))
lo = min(min(Y_val), min(min(snapshots[e]) for e in cps))
hi = max(max(Y_val), max(max(snapshots[e]) for e in cps))
for ax, ep in zip(axes, cps):
    preds = snapshots[ep]
    ax.scatter(Y_val, preds, alpha=0.45, s=14, color="#2563eb", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.4)          # ideal y = x
    ax.set_title(f"Epoch {ep}\nR² = {r2_score(Y_val, preds):.3f}", fontsize=10)
    ax.set_xlabel("Actual (LPA)", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.25)
axes[0].set_ylabel("Predicted (LPA)", fontsize=9)
fig.suptitle("Learning Progress: Predicted vs Actual at Training Checkpoints",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# ---- Plot 2: TRAINING LOSS vs VALIDATION LOSS ----
plt.figure(figsize=(9, 5))
epochs_axis = range(1, EPOCHS + 1)
plt.plot(epochs_axis, train_losses, label="Training Loss", color="#2563eb", linewidth=2)
plt.plot(epochs_axis, val_losses,   label="Validation Loss", color="#f59e0b", linewidth=2)
plt.yscale("log")  # log scale makes the train/val gap easy to see
plt.xlabel("Epoch")
plt.ylabel("MSE (standardized target, log scale)")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Plot 3: FINAL RESULT (the clear, headline graph) ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))

# (Left) Predicted vs Actual on the full validation set
axL.scatter(Y_val, val_pred, alpha=0.55, s=28, color="#16a34a", edgecolors="white", linewidths=0.4)
axL.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect prediction (y = x)")
axL.set_xlabel("Actual Package (LPA)")
axL.set_ylabel("Predicted Package (LPA)")
axL.set_title("Final: Predicted vs Actual (Validation Set)", fontweight="bold")
axL.legend(loc="upper left")
axL.grid(True, alpha=0.3)
axL.text(0.97, 0.05, f"R² = {r2:.4f}\nRMSE = {rmse:.2f} LPA\nMAE = {mae:.2f} LPA",
         transform=axL.transAxes, ha="right", va="bottom", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#16a34a"))

# (Right) Actual vs Predicted for the first 30 validation samples, sorted by
# actual value -- an intuitive "does the prediction track the truth?" view.
order = sorted(range(n_val), key=lambda i: Y_val[i])[:30]
xs = range(len(order))
axR.plot(xs, [Y_val[i] for i in order], "o-", color="#0f172a", label="Actual", markersize=5)
axR.plot(xs, [val_pred[i] for i in order], "s--", color="#16a34a", label="Predicted", markersize=5)
axR.set_xlabel("Validation sample (sorted by actual package)")
axR.set_ylabel("Package (LPA)")
axR.set_title("Predictions Track the Real Values", fontweight="bold")
axR.legend()
axR.grid(True, alpha=0.3)

fig.suptitle("MLP From Scratch -- Final Regression Result", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
