# ============================================================
# THE PERCEPTRON -- From Scratch
# ------------------------------------------------------------
# The simplest "neural network": ONE neuron that learns a single
# straight line to separate two classes. (Rosenblatt, 1958.)
#
#   Model:        z = w1*x1 + w2*x2 + b
#   Prediction:   1 if z >= 0 else 0           (a hard step)
#
#   Learning rule (the "basic formula"), applied for each point:
#       error = y_true - y_pred                (one of +1, 0, -1)
#       w1 <- w1 + lr * error * x1
#       w2 <- w2 + lr * error * x2
#       b  <- b  + lr * error
#
# We START FROM A RANDOM LINE and nudge it, point by point, until it
# separates the data. Then we show WHY it fails on data that no single
# straight line can separate -- its famous limitation.
#
# Pure Python for the algorithm; matplotlib only for the plots.
# Runs anywhere (Colab-ready). It's pure Python, so it runs on CPU.
# ============================================================

import random
import matplotlib.pyplot as plt

random.seed(42)

# ----------------------------------------------------------------
# The perceptron: model + learning rule
# ----------------------------------------------------------------
def predict(x1, x2, w1, w2, b):
    """Hard step activation: fire (1) if the weighted sum is >= 0, else 0."""
    return 1 if (w1 * x1 + w2 * x2 + b) >= 0 else 0

def train_perceptron(X, Y, epochs, lr=1.0):
    # START WITH A RANDOM LINE  (random weights + bias)
    w1, w2, b = random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)
    snapshots = [(w1, w2, b)]      # the line after each epoch (epoch 0 = random start)
    error_history = []

    for epoch in range(1, epochs + 1):
        errors = 0
        for (x1, x2), y in zip(X, Y):
            y_pred = predict(x1, x2, w1, w2, b)
            error = y - y_pred                     # +1, 0, or -1
            if error != 0:                         # only move the line for MISTAKES
                w1 += lr * error * x1              # <-- the basic perceptron formula
                w2 += lr * error * x2
                b  += lr * error
                errors += 1
        error_history.append(errors)
        snapshots.append((w1, w2, b))
        if errors == 0:                            # perfectly separated -> stop early
            break
    return (w1, w2, b), snapshots, error_history

# ----------------------------------------------------------------
# Datasets (1000 points, 2 columns each)
# ----------------------------------------------------------------
def make_separable(n):
    """1000 LINEARLY SEPARABLE points: labelled by a true straight line,
    with a small margin gap so the two classes don't touch."""
    X, Y = [], []
    while len(Y) < n:
        x1, x2 = random.uniform(-5, 5), random.uniform(-5, 5)
        s = 1.0 * x1 - 1.2 * x2 + 0.3              # the hidden "true" boundary
        if abs(s) < 0.6:                            # skip the margin -> clean gap
            continue
        X.append((x1, x2))
        Y.append(1 if s > 0 else 0)
    return X, Y

def make_circle(n):
    """1000 NON-linearly-separable points: inside a circle = 1, outside = 0.
    No straight line can separate the inside of a circle from the outside."""
    X, Y = [], []
    for _ in range(n):
        x1, x2 = random.uniform(-5, 5), random.uniform(-5, 5)
        X.append((x1, x2))
        Y.append(1 if (x1 * x1 + x2 * x2) < 9.0 else 0)   # radius 3
    return X, Y

# ----------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------
def scatter_classes(ax, X, Y):
    c1 = [p for p, l in zip(X, Y) if l == 1]
    c0 = [p for p, l in zip(X, Y) if l == 0]
    ax.scatter([p[0] for p in c1], [p[1] for p in c1], s=8, c="#16a34a", alpha=0.45, label="class 1")
    ax.scatter([p[0] for p in c0], [p[1] for p in c0], s=8, c="#dc2626", alpha=0.45, label="class 0")

def line_xy(w1, w2, b, x_min, x_max):
    """Points of the boundary w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1 + b)/w2."""
    if abs(w2) < 1e-9:
        w2 = 1e-9
    return [x_min, x_max], [-(w1 * x_min + b) / w2, -(w1 * x_max + b) / w2]

# ================================================================
# PART 1 -- Linearly separable: the line MOVES until it separates
# ================================================================
N = 1000
Xs, Ys = make_separable(N)
(w1, w2, b), snaps, errs = train_perceptron(Xs, Ys, epochs=50, lr=1.0)

print("=" * 60)
print("  PART 1: PERCEPTRON ON LINEARLY SEPARABLE DATA")
print(f"  {N} points | started from a random line")
print("=" * 60)
for e, n_err in enumerate(errs, 1):
    print(f"  Epoch {e:>2}: {n_err:>4} misclassified")
if errs[-1] == 0:
    print(f"\n  CONVERGED in {len(errs)} epochs -> found a separating line.")
print(f"  Final line:  {w1:.3f}*x1 + {w2:.3f}*x2 + {b:.3f} = 0")

LIM = (-5.5, 5.5)

# Plot 1: the line moving from random start to final separator
fig, ax = plt.subplots(figsize=(7, 7))
scatter_classes(ax, Xs, Ys)
xi, yi = line_xy(*snaps[0], *LIM)
ax.plot(xi, yi, color="#9ca3af", linestyle=":", linewidth=2.2, label="initial random line")
for k in {max(1, len(snaps) // 4), len(snaps) // 2, max(1, len(snaps) - 2)}:
    xm, ym = line_xy(*snaps[k], *LIM)
    ax.plot(xm, ym, color="#60a5fa", linewidth=1.0, alpha=0.7)
xf, yf = line_xy(*snaps[-1], *LIM)
ax.plot(xf, yf, color="#1d4ed8", linewidth=2.8, label="final learned line")
ax.set_xlim(*LIM); ax.set_ylim(*LIM)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title("Perceptron: a random line nudged until it separates the data", fontweight="bold")
ax.legend(loc="upper right"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# Plot 2: misclassifications per epoch -> falls to 0
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(errs) + 1), errs, "o-", color="#1d4ed8", linewidth=2)
plt.xlabel("epoch (one pass over all 1000 points)")
plt.ylabel("# misclassified")
plt.title("Separable data: errors fall to 0 (the perceptron CONVERGES)", fontweight="bold")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ================================================================
# PART 2 -- The limitation: a circle (no line can separate it)
# ================================================================
Xc, Yc = make_circle(N)
(_, _, _), snaps_c, errs_c = train_perceptron(Xc, Yc, epochs=50, lr=1.0)

print(f"\n{'=' * 60}")
print("  PART 2: PERCEPTRON ON NON-SEPARABLE DATA (a circle)")
print("=" * 60)
print(f"  Best (lowest) error over {len(errs_c)} epochs: {min(errs_c)} / {N}")
print(f"  Final error: {errs_c[-1]} / {N}  -> NEVER reaches 0 (no line can do it).")

# Plot 3: the circle data + the perceptron's hopeless final line
fig, ax = plt.subplots(figsize=(7, 7))
scatter_classes(ax, Xc, Yc)
xf, yf = line_xy(*snaps_c[-1], *LIM)
ax.plot(xf, yf, color="#1d4ed8", linewidth=2.8, label="perceptron's best line")
ax.set_xlim(*LIM); ax.set_ylim(*LIM)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title("Limitation: a straight line CANNOT separate a circle", fontweight="bold")
ax.legend(loc="upper right"); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()

# Plot 4: errors per epoch -> never reaches 0, just oscillates
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, len(errs_c) + 1), errs_c, "o-", color="#dc2626", linewidth=2)
plt.xlabel("epoch (one pass over all 1000 points)")
plt.ylabel("# misclassified")
plt.title("Non-separable data: errors NEVER reach 0 (no convergence)", fontweight="bold")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
