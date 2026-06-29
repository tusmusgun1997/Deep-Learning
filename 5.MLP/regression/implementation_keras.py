# ============================================================
# MLP REGRESSION -- Keras / TensorFlow version
# ------------------------------------------------------------
# Same problem as implementation.py (predict package/LPA from CGPA &
# Profile), but built with the high-level Keras API instead of hand-
# written backprop. Compare the two files side by side:
#
#   implementation.py        ->  forward/backward/gradient-descent by hand
#   implementation_keras.py  ->  model.fit() does all of that for you
#
# >>> Google Colab + GPU:
#     Unlike the pure-Python file, TensorFlow DOES use the GPU. If you
#     pick a GPU runtime, the matrix math runs on the GPU automatically
#     (no code change needed). The script prints whether a GPU is found.
#
# Plots (same as the from-scratch file):
#   1. Learning progress: predicted vs actual at training checkpoints.
#   2. Training Loss vs Validation Loss.
#   3. Final result: predicted vs actual + sorted comparison.
# ============================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # quieten TF startup logs

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
print("=" * 64)
print(f"  TensorFlow {tf.__version__}  |  GPU available: {bool(gpus)}")
if gpus:
    print(f"  Running on GPU: {gpus[0].name}")
else:
    print("  Running on CPU (select a GPU runtime in Colab to use the GPU)")
print("=" * 64)

# ----------------------------------------------------------------
# STEP 1: Generate the SAME dataset as the from-scratch file
# ----------------------------------------------------------------
# True relationship:  package = 1.4*CGPA + 1.1*Profile + 0.15*CGPA*Profile + noise
NUM_SAMPLES = 400
NOISE_STD = 0.5

cgpa    = np.random.uniform(4.0, 10.0, NUM_SAMPLES)
profile = np.random.uniform(1.0, 5.0, NUM_SAMPLES)
noise   = np.random.normal(0.0, NOISE_STD, NUM_SAMPLES)
package = 1.4 * cgpa + 1.1 * profile + 0.15 * cgpa * profile + noise

X = np.column_stack([cgpa, profile]).astype("float32")
y = package.astype("float32")

# Shuffle + 80/20 train/validation split
perm = np.random.permutation(NUM_SAMPLES)
X, y = X[perm], y[perm]
split = int(0.8 * NUM_SAMPLES)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ----------------------------------------------------------------
# STEP 2: Standardize features AND target (train statistics only)
# ----------------------------------------------------------------
x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
Xtr = (X_train - x_mean) / x_std
Xva = (X_val   - x_mean) / x_std

y_mean, y_std = y_train.mean(), y_train.std()
ytr = (y_train - y_mean) / y_std
yva = (y_val   - y_mean) / y_std

def to_lpa(std_pred):
    """Convert a standardized prediction back to real LPA units."""
    return std_pred * y_std + y_mean

# ----------------------------------------------------------------
# STEP 3: Build the model  (2 -> 16 -> 8 -> 1)
# ----------------------------------------------------------------
# The from-scratch file used one sigmoid hidden layer; here we use the
# modern default (ReLU) and let Keras + Adam handle the optimization.
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(2,)),
    keras.layers.Dense(8,  activation="relu"),
    keras.layers.Dense(1),                       # linear output (regression)
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
model.summary()

# ----------------------------------------------------------------
# STEP 4: Train  (model.fit handles forward + backprop + updates)
# ----------------------------------------------------------------
EPOCHS = 200
CHECKPOINTS = [1, 5, 20, 60, EPOCHS]
snapshots = {}   # epoch -> validation predictions (LPA)

class ProgressSnapshots(keras.callbacks.Callback):
    """Capture validation predictions at chosen epochs (for the progress plot)."""
    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e in CHECKPOINTS:
            pred_std = self.model.predict(Xva, verbose=0).flatten()
            snapshots[e] = to_lpa(pred_std)

print(f"\n  Training for {EPOCHS} epochs (batch_size=32)...")
history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=EPOCHS, batch_size=32, verbose=0,
    callbacks=[ProgressSnapshots()],
)
print("  Done.")

# ----------------------------------------------------------------
# STEP 5: Final metrics (real LPA units)
# ----------------------------------------------------------------
val_pred = to_lpa(model.predict(Xva, verbose=0).flatten())
rmse = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
mae  = float(np.mean(np.abs(y_val - val_pred)))
ss_res = float(np.sum((y_val - val_pred) ** 2))
ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
r2 = 1.0 - ss_res / ss_tot

print(f"\n{'=' * 64}")
print("  VALIDATION RESULTS (real LPA units)")
print(f"{'=' * 64}")
print(f"  RMSE : {rmse:.3f} LPA   (noise floor ~ {NOISE_STD} LPA)")
print(f"  MAE  : {mae:.3f} LPA")
print(f"  R^2  : {r2:.4f}")
print(f"{'=' * 64}")

# ================================================================
# PLOTS
# ================================================================

# ---- Plot 1: LEARNING PROGRESS ----
cps = sorted(snapshots.keys())
fig, axes = plt.subplots(1, len(cps), figsize=(3.4 * len(cps), 3.6))
lo = min(y_val.min(), min(snapshots[e].min() for e in cps))
hi = max(y_val.max(), max(snapshots[e].max() for e in cps))
for ax, ep in zip(axes, cps):
    preds = snapshots[ep]
    sr = float(np.sum((y_val - preds) ** 2)); st = float(np.sum((y_val - y_val.mean()) ** 2))
    ax.scatter(y_val, preds, alpha=0.45, s=14, color="#2563eb", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.4)
    ax.set_title(f"Epoch {ep}\nR² = {1 - sr / st:.3f}", fontsize=10)
    ax.set_xlabel("Actual (LPA)", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.grid(True, alpha=0.25)
axes[0].set_ylabel("Predicted (LPA)", fontsize=9)
fig.suptitle("Learning Progress: Predicted vs Actual (Keras)", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# ---- Plot 2: TRAINING vs VALIDATION LOSS ----
plt.figure(figsize=(9, 5))
plt.plot(history.history["loss"],     label="Training Loss",   color="#2563eb", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch"); plt.ylabel("MSE (standardized target, log scale)")
plt.title("Training Loss vs Validation Loss (Keras)")
plt.legend(); plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Plot 3: FINAL RESULT ----
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
axL.scatter(y_val, val_pred, alpha=0.55, s=28, color="#16a34a", edgecolors="white", linewidths=0.4)
axL.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect prediction (y = x)")
axL.set_xlabel("Actual Package (LPA)"); axL.set_ylabel("Predicted Package (LPA)")
axL.set_title("Final: Predicted vs Actual (Validation)", fontweight="bold")
axL.legend(loc="upper left"); axL.grid(True, alpha=0.3)
axL.text(0.97, 0.05, f"R² = {r2:.4f}\nRMSE = {rmse:.2f} LPA\nMAE = {mae:.2f} LPA",
         transform=axL.transAxes, ha="right", va="bottom", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#16a34a"))

order = np.argsort(y_val)[:30]
xs = range(len(order))
axR.plot(xs, y_val[order],   "o-", color="#0f172a", label="Actual",    markersize=5)
axR.plot(xs, val_pred[order], "s--", color="#16a34a", label="Predicted", markersize=5)
axR.set_xlabel("Validation sample (sorted by actual)"); axR.set_ylabel("Package (LPA)")
axR.set_title("Predictions Track the Real Values", fontweight="bold")
axR.legend(); axR.grid(True, alpha=0.3)
fig.suptitle("MLP Regression (Keras/TensorFlow) -- Final Result", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
