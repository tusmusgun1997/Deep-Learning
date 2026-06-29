# ============================================================
# MLP FROM SCRATCH -- BINARY CLASSIFICATION  (with Plots)
# ------------------------------------------------------------
# Same inputs as the regression file (CGPA, Profile Score), but now
# we predict a YES/NO outcome:  did the student get PLACED (1) or not (0)?
#
# The placement pattern is deliberately NON-LINEAR: a straight line
# cannot separate the two classes -- only a network with a hidden layer
# can carve the curved boundary. (The data is synthetic, chosen so the
# non-linearity is obvious; see generate_dataset below.)
#
# Everything (forward pass, backprop, mini-batch SGD) is PURE PYTHON
# -- no NumPy, no PyTorch. Matplotlib is used ONLY for the graphs.
#
# >>> Google Colab: pure-Python => runs on CPU. A GPU runtime won't
#     speed it up, but it finishes in a few seconds. Just paste & run.
#
# What you'll see:
#   1. The raw data  -> how "placed" vs "not placed" looks in 2D.
#   2. The DECISION BOUNDARY evolving epoch by epoch (the "output curve"
#      wrapping around the placed region as the network learns).
#   3. Accuracy and Loss curves (training vs validation).
#   4. A final results graph: the learned boundary + accuracy + confusion matrix.
# ============================================================

import math
import random
import matplotlib.pyplot as plt

random.seed(42)  # reproducible data, shuffling, and weight init

# ----------------------------------------------------------------
# STEP 1: Generate a NON-LINEAR binary dataset (from scratch)
# ----------------------------------------------------------------
# Rule: a student is "placed" if they land inside a curved sweet-spot
# region of the CGPA-Profile space (an ellipse), else "not placed".
# This boundary is a CLOSED CURVE -- impossible for a single straight
# line to separate, which is exactly why we need a hidden layer.
# A few labels are randomly flipped to add realistic noise.
NUM_SAMPLES = 500
LABEL_NOISE = 0.04   # fraction of labels randomly flipped

def generate_dataset(num):
    data_X, data_Y = [], []
    for _ in range(num):
        cgpa    = random.uniform(4.0, 10.0)
        profile = random.uniform(1.0, 5.0)
        # normalized squared distance from the "sweet spot" centre (7, 3)
        radius = ((cgpa - 7.0) / 3.0) ** 2 + ((profile - 3.0) / 2.0) ** 2
        label = 1 if radius < 0.6 else 0          # inside the curve => placed
        if random.random() < LABEL_NOISE:
            label = 1 - label                     # flip (noise)
        data_X.append([cgpa, profile])
        data_Y.append(label)
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
# STEP 2: Standardize the FEATURES (train statistics only)
# ----------------------------------------------------------------
# (The target is already 0/1, so it is NOT standardized.)
def feature_stats(rows):
    means, stds = [], []
    for f in range(len(rows[0])):
        vals = [r[f] for r in rows]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        means.append(m)
        stds.append(math.sqrt(var) if var > 0 else 1.0)
    return means, stds

def apply_norm(rows, means, stds):
    return [[(r[f] - means[f]) / stds[f] for f in range(len(r))] for r in rows]

feat_means, feat_stds = feature_stats(X_train)
Xtr = apply_norm(X_train, feat_means, feat_stds)
Xva = apply_norm(X_val,   feat_means, feat_stds)

# ----------------------------------------------------------------
# STEP 3: Network  (2 -> HIDDEN tanh -> 1 sigmoid)  +  BCE loss
# ----------------------------------------------------------------
# tanh in the hidden layer (strong gradients, zero-centred), sigmoid on
# the output to produce a probability P(placed). Loss = Binary Cross-Entropy.
INPUT  = 2
HIDDEN = 10
LR     = 0.30
EPOCHS = 150
BATCH  = 48

def xavier_init(fan_in, fan_out):
    return random.gauss(0.0, math.sqrt(2.0 / (fan_in + fan_out)))

W1 = [[xavier_init(INPUT, HIDDEN) for _ in range(INPUT)] for _ in range(HIDDEN)]
b1 = [0.0] * HIDDEN
W2 = [xavier_init(HIDDEN, 1) for _ in range(HIDDEN)]
b2 = 0.0

def sigmoid(z):
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))

def tanh(z):
    z = max(-500.0, min(500.0, z))
    return math.tanh(z)

def forward(x):
    """Return (hidden activations, probability of 'placed')."""
    a_h = [tanh(b1[j] + sum(W1[j][k] * x[k] for k in range(INPUT))) for j in range(HIDDEN)]
    p = sigmoid(b2 + sum(W2[j] * a_h[j] for j in range(HIDDEN)))
    return a_h, p

def bce_loss(Xn, Yn):
    """Binary Cross-Entropy over a dataset."""
    eps, total = 1e-9, 0.0
    for i in range(len(Yn)):
        p = forward(Xn[i])[1]
        p = min(1 - eps, max(eps, p))
        total += -(Yn[i] * math.log(p) + (1 - Yn[i]) * math.log(1 - p))
    return total / len(Yn)

def accuracy(Xn, Yn):
    correct = sum(1 for i in range(len(Yn)) if (forward(Xn[i])[1] >= 0.5) == bool(Yn[i]))
    return correct / len(Yn)

# A fixed mesh over the ORIGINAL feature space, used to draw the decision
# boundary (we standardize each grid point before feeding it to the model).
GRID = 70
cgpa_axis    = [4.0 + 6.0 * i / (GRID - 1) for i in range(GRID)]
profile_axis = [1.0 + 4.0 * j / (GRID - 1) for j in range(GRID)]

def probability_grid():
    """P(placed) over the whole CGPA x Profile plane (for contour plots)."""
    Z = []
    for pv in profile_axis:
        row = []
        for cv in cgpa_axis:
            xs = [(cv - feat_means[0]) / feat_stds[0], (pv - feat_means[1]) / feat_stds[1]]
            row.append(forward(xs)[1])
        Z.append(row)
    return Z

# ----------------------------------------------------------------
# STEP 4: Train with MINI-BATCH SGD + record history & snapshots
# ----------------------------------------------------------------
train_losses, val_losses, train_accs, val_accs = [], [], [], []

# Capture the decision boundary at these epochs to watch it form.
checkpoints = [1, 5, 15, 40, 80, EPOCHS]
boundary_snapshots = {}   # epoch -> probability grid

order = list(range(n_train))

print("=" * 64)
print("  MLP FROM SCRATCH  --  Binary Classification (Placed / Not)")
print(f"  Data: {NUM_SAMPLES} samples ({n_train} train / {n_val} val)")
print(f"  Network: {INPUT} -> {HIDDEN} (tanh) -> 1 (sigmoid) | Loss: BCE")
print(f"  LR: {LR} | Epochs: {EPOCHS} | Mini-batch size: {BATCH}")
print("=" * 64)
print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>9} | {'Train Acc':>9} | {'Val Acc':>8}")
print(f"  {'-' * 56}")

for epoch in range(1, EPOCHS + 1):
    random.shuffle(order)

    # ---- iterate over mini-batches ----
    for start in range(0, n_train, BATCH):
        batch = order[start:start + BATCH]
        m = len(batch)

        gW1 = [[0.0] * INPUT for _ in range(HIDDEN)]
        gb1 = [0.0] * HIDDEN
        gW2 = [0.0] * HIDDEN
        gb2 = 0.0

        for i in batch:
            x, y = Xtr[i], Y_train[i]
            a_h, p = forward(x)

            # BCE + sigmoid output -> clean gradient: dL/dz_out = (p - y)
            d_out = (p - y) / m
            for j in range(HIDDEN):
                gW2[j] += d_out * a_h[j]
            gb2 += d_out
            # tanh hidden: a'(z) = 1 - a^2
            for j in range(HIDDEN):
                d_h = d_out * W2[j] * (1.0 - a_h[j] * a_h[j])
                for k in range(INPUT):
                    gW1[j][k] += d_h * x[k]
                gb1[j] += d_h

        # ---- update parameters (once per mini-batch) ----
        for j in range(HIDDEN):
            for k in range(INPUT):
                W1[j][k] -= LR * gW1[j][k]
            b1[j] -= LR * gb1[j]
            W2[j] -= LR * gW2[j]
        b2 -= LR * gb2

    # ---- record full-dataset metrics at the end of the epoch ----
    train_losses.append(bce_loss(Xtr, Y_train))
    val_losses.append(bce_loss(Xva, Y_val))
    train_accs.append(accuracy(Xtr, Y_train))
    val_accs.append(accuracy(Xva, Y_val))

    if epoch in checkpoints:
        boundary_snapshots[epoch] = probability_grid()

    if epoch == 1 or epoch % 25 == 0:
        print(f"  {epoch:>6} | {train_losses[-1]:>10.4f} | {val_losses[-1]:>9.4f} | "
              f"{train_accs[-1]:>9.3f} | {val_accs[-1]:>8.3f}")

# ----------------------------------------------------------------
# STEP 5: Final metrics + confusion matrix (validation set)
# ----------------------------------------------------------------
TP = TN = FP = FN = 0
for i in range(n_val):
    pred = 1 if forward(Xva[i])[1] >= 0.5 else 0
    actual = Y_val[i]
    if   pred == 1 and actual == 1: TP += 1
    elif pred == 0 and actual == 0: TN += 1
    elif pred == 1 and actual == 0: FP += 1
    else:                           FN += 1
final_acc = (TP + TN) / n_val
precision = TP / (TP + FP) if (TP + FP) else 0.0
recall    = TP / (TP + FN) if (TP + FN) else 0.0

print(f"\n{'=' * 64}")
print("  TRAINING COMPLETE -- VALIDATION RESULTS")
print(f"{'=' * 64}")
print(f"  Accuracy : {final_acc:.3f}")
print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}")
print(f"  Confusion matrix (val):  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
print(f"{'=' * 64}")

# ================================================================
# PLOTS
# ================================================================
PLACED_C, NOTPLACED_C = "#16a34a", "#dc2626"   # green = placed, red = not

def scatter_classes(ax, Xpts, Ypts, s=18, alpha=0.8):
    px = [Xpts[i][0] for i in range(len(Ypts)) if Ypts[i] == 1]
    py = [Xpts[i][1] for i in range(len(Ypts)) if Ypts[i] == 1]
    nx = [Xpts[i][0] for i in range(len(Ypts)) if Ypts[i] == 0]
    ny = [Xpts[i][1] for i in range(len(Ypts)) if Ypts[i] == 0]
    ax.scatter(px, py, c=PLACED_C, s=s, alpha=alpha, edgecolors="white",
               linewidths=0.4, label="Placed")
    ax.scatter(nx, ny, c=NOTPLACED_C, s=s, alpha=alpha, edgecolors="white",
               linewidths=0.4, label="Not placed")

# ---- Plot 1: HOW THE DATA LOOKS ----
plt.figure(figsize=(7.5, 6))
scatter_classes(plt.gca(), X_all, Y_all, s=22)
plt.xlabel("CGPA"); plt.ylabel("Profile Score")
plt.title("The Data: Placed vs Not Placed\n(notice the curved, non-linear separation)",
          fontweight="bold")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---- Plot 2: DECISION BOUNDARY EVOLVING OVER EPOCHS ----
cps = [e for e in checkpoints if e in boundary_snapshots]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
for ax, ep in zip(axes, cps):
    Z = boundary_snapshots[ep]
    ax.contourf(cgpa_axis, profile_axis, Z, levels=20, cmap="RdYlGn", alpha=0.75, vmin=0, vmax=1)
    ax.contour(cgpa_axis, profile_axis, Z, levels=[0.5], colors="black", linewidths=1.8)
    scatter_classes(ax, X_train, Y_train, s=10, alpha=0.6)
    # accuracy at this checkpoint (index epoch-1 in the history)
    ax.set_title(f"Epoch {ep}   |   Val Acc = {val_accs[ep - 1]:.2f}", fontsize=11)
    ax.set_xlabel("CGPA"); ax.set_ylabel("Profile")
fig.suptitle("Decision Boundary Forming Over Training "
             "(black line = where P(placed) = 0.5)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ---- Plot 3: LOSS & ACCURACY CURVES (train vs validation) ----
fig, (axL, axA) = plt.subplots(1, 2, figsize=(13, 5))
ep_axis = range(1, EPOCHS + 1)
axL.plot(ep_axis, train_losses, label="Training Loss", color="#2563eb", linewidth=2)
axL.plot(ep_axis, val_losses,   label="Validation Loss", color="#f59e0b", linewidth=2)
axL.set_xlabel("Epoch"); axL.set_ylabel("BCE Loss")
axL.set_title("Training vs Validation Loss", fontweight="bold")
axL.legend(); axL.grid(True, alpha=0.3)

axA.plot(ep_axis, train_accs, label="Training Accuracy", color="#2563eb", linewidth=2)
axA.plot(ep_axis, val_accs,   label="Validation Accuracy", color="#f59e0b", linewidth=2)
axA.set_xlabel("Epoch"); axA.set_ylabel("Accuracy")
axA.set_title("Training vs Validation Accuracy", fontweight="bold")
axA.set_ylim(0.3, 1.02)
axA.legend(); axA.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# ---- Plot 4: FINAL RESULT (learned boundary + validation points + metrics) ----
fig, (axB, axC) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1]})

# (Left) final decision boundary with validation points; mark mistakes
Z = boundary_snapshots[EPOCHS]
axB.contourf(cgpa_axis, profile_axis, Z, levels=20, cmap="RdYlGn", alpha=0.7, vmin=0, vmax=1)
axB.contour(cgpa_axis, profile_axis, Z, levels=[0.5], colors="black", linewidths=2)
scatter_classes(axB, X_val, Y_val, s=42)
wrong_x, wrong_y = [], []
for i in range(n_val):
    pred = 1 if forward(Xva[i])[1] >= 0.5 else 0
    if pred != Y_val[i]:
        wrong_x.append(X_val[i][0]); wrong_y.append(X_val[i][1])
axB.scatter(wrong_x, wrong_y, marker="x", c="black", s=90, linewidths=2,
            label="Misclassified")
axB.set_xlabel("CGPA"); axB.set_ylabel("Profile Score")
axB.set_title(f"Final Learned Boundary on Validation Data  (Acc = {final_acc:.2f})",
              fontweight="bold")
axB.legend(loc="upper right")

# (Right) confusion matrix as a 2x2 heatmap
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
fig.suptitle("MLP From Scratch -- Final Classification Result",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
