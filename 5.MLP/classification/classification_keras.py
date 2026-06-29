# ============================================================
# MLP BINARY CLASSIFICATION -- Keras / TensorFlow version
# ------------------------------------------------------------
# Same problem as classification.py (predict PLACED / NOT PLACED from
# CGPA & Profile, with a non-linear decision boundary), but built with
# the high-level Keras API instead of hand-written backprop.
#
#   classification.py        ->  forward/backward + mini-batch SGD by hand
#   classification_keras.py  ->  model.fit() does all of that for you
#
# >>> Google Colab + GPU:
#     TensorFlow uses the GPU automatically if you select a GPU runtime
#     (no code change needed). The script prints whether a GPU is found.
#
# Plots (same as the from-scratch file):
#   1. The raw data (placed vs not placed).
#   2. The decision boundary evolving epoch by epoch.
#   3. Accuracy and Loss curves (training vs validation).
#   4. Final result: learned boundary + confusion matrix.
# ============================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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
# STEP 1: Generate the SAME non-linear dataset as the from-scratch file
# ----------------------------------------------------------------
# "Placed" = inside a curved sweet-spot ellipse of the CGPA-Profile plane.
# A straight line cannot separate this -> we need a hidden layer.
NUM_SAMPLES = 500
LABEL_NOISE = 0.04

cgpa    = np.random.uniform(4.0, 10.0, NUM_SAMPLES)
profile = np.random.uniform(1.0, 5.0, NUM_SAMPLES)
radius  = ((cgpa - 7.0) / 3.0) ** 2 + ((profile - 3.0) / 2.0) ** 2
y = (radius < 0.6).astype("float32")                      # inside ellipse => placed
flip = np.random.random(NUM_SAMPLES) < LABEL_NOISE        # random label noise
y = np.where(flip, 1.0 - y, y).astype("float32")
X = np.column_stack([cgpa, profile]).astype("float32")

# Shuffle + 80/20 split
perm = np.random.permutation(NUM_SAMPLES)
X, y = X[perm], y[perm]
split = int(0.8 * NUM_SAMPLES)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ----------------------------------------------------------------
# STEP 2: Standardize the FEATURES (train statistics only)
# ----------------------------------------------------------------
x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
Xtr = (X_train - x_mean) / x_std
Xva = (X_val   - x_mean) / x_std

# ----------------------------------------------------------------
# STEP 3: Build the model  (2 -> 10 tanh -> 1 sigmoid)
# ----------------------------------------------------------------
model = keras.Sequential([
    keras.layers.Dense(10, activation="tanh", input_shape=(2,)),
    keras.layers.Dense(1,  activation="sigmoid"),    # probability of "placed"
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# A mesh over the ORIGINAL feature space, for drawing the decision boundary.
GRID = 70
cg = np.linspace(4.0, 10.0, GRID)
pr = np.linspace(1.0, 5.0, GRID)
CG, PR = np.meshgrid(cg, pr)
grid_std = (np.column_stack([CG.ravel(), PR.ravel()]).astype("float32") - x_mean) / x_std

# ----------------------------------------------------------------
# STEP 4: Train, capturing the decision boundary at checkpoints
# ----------------------------------------------------------------
EPOCHS = 150
CHECKPOINTS = [1, 5, 15, 40, 80, EPOCHS]
boundary_snapshots = {}

class BoundarySnapshots(keras.callbacks.Callback):
    """Capture P(placed) over the whole plane at chosen epochs."""
    def on_epoch_end(self, epoch, logs=None):
        e = epoch + 1
        if e in CHECKPOINTS:
            Z = self.model.predict(grid_std, verbose=0).reshape(GRID, GRID)
            boundary_snapshots[e] = Z

print(f"\n  Training for {EPOCHS} epochs (batch_size=48)...")
history = model.fit(
    Xtr, y_train,
    validation_data=(Xva, y_val),
    epochs=EPOCHS, batch_size=48, verbose=0,
    callbacks=[BoundarySnapshots()],
)
print("  Done.")

# ----------------------------------------------------------------
# STEP 5: Final metrics + confusion matrix (validation set)
# ----------------------------------------------------------------
val_prob = model.predict(Xva, verbose=0).flatten()
val_pred = (val_prob >= 0.5).astype("float32")
TP = int(np.sum((val_pred == 1) & (y_val == 1)))
TN = int(np.sum((val_pred == 0) & (y_val == 0)))
FP = int(np.sum((val_pred == 1) & (y_val == 0)))
FN = int(np.sum((val_pred == 0) & (y_val == 1)))
acc = (TP + TN) / len(y_val)
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0

print(f"\n{'=' * 64}")
print("  VALIDATION RESULTS")
print(f"{'=' * 64}")
print(f"  Accuracy : {acc:.3f}")
print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}")
print(f"  Confusion matrix (val):  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"{'=' * 64}")

# ================================================================
# PLOTS
# ================================================================
PLACED_C, NOTPLACED_C = "#16a34a", "#dc2626"

def scatter_classes(ax, Xpts, Ypts, s=18, alpha=0.8):
    m1 = Ypts == 1
    ax.scatter(Xpts[m1, 0], Xpts[m1, 1], c=PLACED_C, s=s, alpha=alpha,
               edgecolors="white", linewidths=0.4, label="Placed")
    ax.scatter(Xpts[~m1, 0], Xpts[~m1, 1], c=NOTPLACED_C, s=s, alpha=alpha,
               edgecolors="white", linewidths=0.4, label="Not placed")

# ---- Plot 1: HOW THE DATA LOOKS ----
plt.figure(figsize=(7.5, 6))
scatter_classes(plt.gca(), X, y, s=22)
plt.xlabel("CGPA"); plt.ylabel("Profile Score")
plt.title("The Data: Placed vs Not Placed\n(curved, non-linear separation)", fontweight="bold")
plt.legend(loc="upper right"); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Plot 2: DECISION BOUNDARY EVOLVING OVER EPOCHS ----
cps = [e for e in CHECKPOINTS if e in boundary_snapshots]
val_acc_hist = history.history["val_accuracy"]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for ax, ep in zip(axes, cps):
    Z = boundary_snapshots[ep]
    ax.contourf(CG, PR, Z, levels=20, cmap="RdYlGn", alpha=0.75, vmin=0, vmax=1)
    ax.contour(CG, PR, Z, levels=[0.5], colors="black", linewidths=1.8)
    scatter_classes(ax, X_train, y_train, s=10, alpha=0.6)
    ax.set_title(f"Epoch {ep}   |   Val Acc = {val_acc_hist[ep - 1]:.2f}", fontsize=11)
    ax.set_xlabel("CGPA"); ax.set_ylabel("Profile")
fig.suptitle("Decision Boundary Forming Over Training (Keras)  "
             "[black line = P(placed)=0.5]", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---- Plot 3: LOSS & ACCURACY CURVES ----
fig, (axL, axA) = plt.subplots(1, 2, figsize=(13, 5))
axL.plot(history.history["loss"],     label="Training Loss",   color="#2563eb", linewidth=2)
axL.plot(history.history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2)
axL.set_xlabel("Epoch"); axL.set_ylabel("BCE Loss")
axL.set_title("Training vs Validation Loss", fontweight="bold")
axL.legend(); axL.grid(True, alpha=0.3)

axA.plot(history.history["accuracy"],     label="Training Accuracy",   color="#2563eb", linewidth=2)
axA.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#f59e0b", linewidth=2)
axA.set_xlabel("Epoch"); axA.set_ylabel("Accuracy"); axA.set_ylim(0.3, 1.02)
axA.set_title("Training vs Validation Accuracy", fontweight="bold")
axA.legend(); axA.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# ---- Plot 4: FINAL RESULT ----
fig, (axB, axC) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})
Z = boundary_snapshots[EPOCHS]
axB.contourf(CG, PR, Z, levels=20, cmap="RdYlGn", alpha=0.7, vmin=0, vmax=1)
axB.contour(CG, PR, Z, levels=[0.5], colors="black", linewidths=2)
scatter_classes(axB, X_val, y_val, s=42)
wrong = val_pred != y_val
axB.scatter(X_val[wrong, 0], X_val[wrong, 1], marker="x", c="black", s=90,
            linewidths=2, label="Misclassified")
axB.set_xlabel("CGPA"); axB.set_ylabel("Profile Score")
axB.set_title(f"Final Learned Boundary on Validation Data  (Acc = {acc:.2f})", fontweight="bold")
axB.legend(loc="upper right")

cm = [[TN, FP], [FN, TP]]
axC.imshow(cm, cmap="Blues")
axC.set_xticks([0, 1]); axC.set_xticklabels(["Pred: Not", "Pred: Placed"])
axC.set_yticks([0, 1]); axC.set_yticklabels(["Actual: Not", "Actual: Placed"])
axC.set_title("Confusion Matrix (Validation)", fontweight="bold")
mx = max(max(r) for r in cm)
for r in range(2):
    for c in range(2):
        axC.text(c, r, str(cm[r][c]), ha="center", va="center", fontsize=20,
                 color="white" if cm[r][c] > mx / 2 else "black")
fig.suptitle("MLP Classification (Keras/TensorFlow) -- Final Result", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
