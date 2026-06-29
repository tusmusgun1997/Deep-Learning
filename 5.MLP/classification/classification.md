# Binary Classification MLP — From Scratch (Full Deep Dive)

This document explains **everything** in [classification.py](classification.py): what the code does, the **math** behind every step, and — importantly — **the design journey**: what we tried first, why it *failed*, and why we ended up with the choices in the final file. Read it top-to-bottom and you'll understand not just *what* the code is, but *why* it is that way.

---

## 0. TL;DR

We predict a **yes/no** outcome — *did a student get placed?* — from two inputs (**CGPA**, **Profile Score**). The catch: the placement pattern is **non-linear** (a curved region), so a straight line cannot separate the two classes. Only a network with a **hidden layer** can.

Final recipe (after several failed attempts):
- **Network:** `2 → 10 tanh → 1 sigmoid`, trained with **Binary Cross-Entropy** loss.
- **Optimizer:** **mini-batch SGD** (batch 48, learning rate 0.30, 150 epochs).
- **Result:** ~**92% validation accuracy** with clean generalization (a linear model gets only ~50%).

---

## 1. The Problem

| | Regression sibling ([../regression](../regression/implementation.py)) | This file (classification) |
|---|---|---|
| Inputs | CGPA, Profile | CGPA, Profile (**same**) |
| Output | A **number** (package in LPA) | A **class**: placed (1) / not placed (0) |
| Output activation | Linear | **Sigmoid** (a probability) |
| Loss | MSE | **Binary Cross-Entropy** |
| "Correct?" measure | RMSE / R² | **Accuracy**, precision, recall |

Same inputs, completely different *kind* of output — so almost everything after the hidden layer changes.

---

## 2. The Data — and Why It's Non-Linear

We **synthesize** 500 students. Each has a CGPA (4–10) and a Profile score (1–5). The placement rule is a **curved sweet-spot**:

```
radius² = ((CGPA - 7) / 3)²  +  ((Profile - 3) / 2)²
placed  = 1  if  radius² < 0.6   else  0
```

That's the inside of an **ellipse** centered at (CGPA 7, Profile 3). We also **flip 4% of labels at random** (`LABEL_NOISE = 0.04`) so the data is realistically noisy (no model can hit 100%).

> ⚠️ **Honesty note:** this is a *deliberately synthetic* pattern chosen to make the non-linearity obvious for teaching. It is **not** a claim about how real placements work (a real rule wouldn't reject a 10-CGPA / 5-profile student). The point is the **shape** of the boundary, not the story.

### Why a straight line *cannot* do this

A linear classifier computes one number, `w₁·CGPA + w₂·Profile + b`, and splits the plane with a **single straight line** (everything on one side = placed). But our "placed" region is a **closed blob** in the middle:

```
   Profile
     ^         . . . . . . . . .          A straight line can only
     |       .  o o o o o o o  .          cut the plane in two halves.
     |      .  o o (placed) o o .         No single line can carve out
     |      .  o o o o o o o o  .         a CLOSED region in the middle.
     |       .  o o o o o o o  .          --> a line is the WRONG tool.
     |         . . . . . . . . .
     +-------------------------------> CGPA
         (everything outside = not placed)
```

No half-plane equals "the inside of an ellipse." So a line is stuck at roughly **50% accuracy** (a coin flip, since the classes are ~48% placed / 52% not). **We measured exactly this: a linear model scored 0.44–0.50 in testing.** A hidden layer, by combining several tanh units, *can* bend a closed boundary around the blob — that's the whole reason this file exists.

**Plot 1** in the code (`The Data`) shows this: a green "placed" cloud in the center, red "not placed" all around it.

---

## 3. The Design Journey — What Failed and Why

This is the part most tutorials hide. Getting a clean result took **five iterations**. Here's each one with the *actual measured numbers*.

### Attempt 1 — Sigmoid hidden + full-batch GD → ❌ the MLP tied the linear model

The first instinct was to copy the regression file's style: **sigmoid** hidden units and **full-batch** gradient descent. We tested it on four candidate datasets. The MLP **never beat** a plain linear classifier:

| Dataset rule | placed % | Linear acc | MLP acc | Gap |
|---|---|---|---|---|
| product `CGPA·Profile > 24` | 37% | 0.975 | 0.975 | **+0.000** |
| corner ellipse | 66% | 0.808 | 0.808 | **+0.000** |
| sine boundary | 27% | 0.850 | 0.850 | **+0.000** |
| middle band | 66% | 0.900 | 0.900 | **+0.000** |

A gap of **+0.000 everywhere** is a red flag. The MLP was **underfitting** — collapsing to a basically-linear function. Two causes:
1. **Sigmoid hidden units** squash gradients (max slope 0.25 — see [vanishing gradients](../../Improve_NN/3_Vanishing_Gradients/vanishing_gradients.md)), so the hidden layer learned almost nothing.
2. Some of those "non-linear" datasets weren't actually non-linear enough (the *band* rule's upper bound was almost never hit, making it effectively linear).

### Fix 1 — Switch hidden activation to **tanh** + train longer → ✅ the MLP comes alive

Swapping sigmoid → **tanh** (zero-centered, slope up to **1.0** at the origin, so gradients flow) and giving it more epochs, the MLP finally learned curved boundaries:

| Dataset rule | placed % | Linear acc | MLP acc | Gap |
|---|---|---|---|---|
| **circles** (ellipse blob) | 44% | **0.442** | **0.858** | **+0.417** |
| AND-corner | 34% | 0.783 | 0.867 | +0.083 |
| sine boundary | 31% | 0.792 | 0.883 | +0.092 |

The **circles** dataset gave by far the biggest gap: a line gets 0.44 (worse than guessing!), the MLP 0.86. That huge gap is the clearest possible demonstration that *the non-linearity matters* — so we chose circles.

### Attempt 2 — Full-batch GD on circles → ❌ too slow (timed out)

Full-batch gradient descent (one update per epoch over all 400 training points) needed **thousands** of epochs to carve the circle, and pure-Python is slow — the tuning run **timed out after 2 minutes**.

### Fix 2 — **Mini-batch SGD** → ✅ fast *and* accurate

Switching to **mini-batch SGD** (update after every 32–48 samples, so ~8–13 updates per epoch instead of 1) converged in a fraction of the time:

| Config | Val acc | Wall time |
|---|---|---|
| H=16, 150 epochs, batch 32 | 0.840 | 0.9 s |
| H=16, 250 epochs, batch 32 | 0.910 | **1.4 s** |

From a 2-minute timeout to **under 2 seconds**. (Section 7 explains *why* mini-batch is so much faster.)

### Attempt 3 — H=16, 300 epochs → ❌ overfitting

Running the full file with 16 hidden units for 300 epochs, the model started **memorizing the 4% noisy labels**:

```
Validation loss bottomed out (~0.36) around epoch 75, then ROSE to 0.44 by epoch 300.
Validation accuracy peaked at 0.92, then DROPPED to 0.86.
Meanwhile training accuracy kept climbing to 0.945.
```

That widening train↔val gap is the classic **overfitting** signature (see [overfitting](../../Improve_NN/1_Overfitting/overfitting.md)). The final model was *worse* than one stopped halfway.

### Fix 3 — Smaller network + fewer epochs → ✅ clean generalization

We swept capacity / epochs / batch size, looking for a config where **validation tracks training** (small gap, no val-loss U-turn):

| H | epochs | lr | batch | noise | train | val | gap | val-loss rise |
|---|---|---|---|---|---|---|---|---|
| 8 | 150 | 0.3 | 48 | 0.04 | 0.922 | 0.870 | +0.052 | +0.023 |
| **10** | **150** | **0.3** | **48** | **0.04** | **0.925** | **0.930** | **−0.005** | **+0.000** ✅ |
| 8 | 120 | 0.25 | 48 | 0.03 | 0.910 | 0.910 | +0.000 | +0.000 |
| 6 | 160 | 0.3 | 48 | 0.04 | 0.905 | 0.900 | +0.005 | +0.019 |

**`H=10, 150 epochs, lr=0.30, batch=48, noise=0.04`** won: ~0.93 validation accuracy, essentially **zero** train/val gap, and **no** validation-loss rise. That's the final configuration in the file.

### Lesson of the journey

> Three different failures, three different lessons: **activation choice** (tanh beat sigmoid for hidden layers), **optimizer choice** (mini-batch beat full-batch for speed), and **model capacity** (smaller + shorter beat bigger + longer, to avoid overfitting). None of these were obvious up front — they came from *measuring* and reacting.

---

## 4. The Architecture

```
   INPUT          HIDDEN LAYER              OUTPUT
  (2 features)    (10 tanh units)        (1 sigmoid unit)

   CGPA  ─┐      ┌─[ tanh ]─┐
          ├────► │   ...    ├───────►  z_out ──► sigmoid ──► p = P(placed) ∈ (0,1)
 Profile ─┘      └─[ tanh ]─┘                                    │
                  (10 of them)                                   ▼
                                                       predict "placed" if p ≥ 0.5
```

**Parameter count:** `W1` (10×2 = 20) + `b1` (10) + `W2` (10) + `b2` (1) = **41 learnable parameters**.

---

## 5. The Math, Step by Step

### 5.1 Feature standardization (and the no-leakage rule)

CGPA (4–10) and Profile (1–5) live on different scales, which skews gradients. We standardize each feature to mean 0, std 1:

```
x_norm = (x - mean) / std
```

**Critical:** `mean` and `std` are computed from the **training set only**, then applied to the validation set. Using validation stats would **leak** information and inflate the score. (More in [normalization](../../Improve_NN/2_Normalization/normalization.md).) The target is already 0/1, so it is **not** standardized.

### 5.2 Forward pass

For a standardized input `x = [x₁, x₂]`:

```
Hidden unit j (j = 1..10):
    z_h[j] = b1[j] + W1[j][1]·x₁ + W1[j][2]·x₂
    a_h[j] = tanh(z_h[j])                         ← non-linearity

Output:
    z_out  = b2 + Σ_j W2[j]·a_h[j]
    p      = sigmoid(z_out) = 1 / (1 + e^(−z_out))   ← probability of "placed"
```

### 5.3 Why **tanh** in the hidden layer (not sigmoid)

```
tanh(z) = (eᶻ − e⁻ᶻ)/(eᶻ + e⁻ᶻ)      range (−1, 1),  max slope = 1.0  (at z=0)
sigmoid(z) = 1/(1 + e⁻ᶻ)              range  (0, 1),  max slope = 0.25 (at z=0)
```

Two reasons tanh wins for hidden layers (confirmed by Fix 1 above):
1. **Stronger gradients** — slope up to 1.0 vs sigmoid's 0.25, so the hidden layer actually learns.
2. **Zero-centered** outputs (−1..1) keep the next layer's gradients balanced, speeding convergence.

### 5.4 Why **sigmoid + Binary Cross-Entropy** at the output (not MSE)

The output must be a **probability** in (0, 1) → sigmoid. The loss must measure "how wrong is this probability?" → **Binary Cross-Entropy (BCE)**:

```
BCE (one sample):  L = −[ y·log(p) + (1−y)·log(1−p) ]
```

- If `y = 1`: `L = −log(p)`. Predict p=0.99 → loss 0.01 (tiny). Predict p=0.01 → loss 4.6 (huge).
- If `y = 0`: `L = −log(1−p)`. Symmetric.

Why not MSE? With a sigmoid output, MSE produces a gradient containing an extra `sigmoid'(z)` factor that **vanishes** when the model is confidently wrong (slow learning), and the surface is non-convex. BCE is the proper maximum-likelihood loss for yes/no outcomes and gives the clean gradient below. (Full comparison in [Binary Cross-Entropy](../../LossFunction/3_BinaryCrossEntropy/bce.md).)

### 5.5 The beautiful gradient: `dL/dz_out = p − y`

This is the key simplification that makes sigmoid+BCE the standard pairing. Chain rule:

```
L = −[y·log(p) + (1−y)·log(1−p)],   p = sigmoid(z_out)

dL/dp     = −y/p + (1−y)/(1−p) = (p − y) / [p(1−p)]
dp/dz_out = p(1−p)                                   ← sigmoid derivative

dL/dz_out = dL/dp · dp/dz_out = (p − y)/[p(1−p)] · p(1−p) = (p − y)
```

The messy `p(1−p)` terms **cancel exactly**. So the output-layer error signal is simply:

```
δ_out = p − y         (prediction minus truth — that's it!)
```

### 5.6 Backpropagation (the rest of the chain)

With `δ_out = p − y` and tanh's derivative `tanh'(z) = 1 − tanh²(z) = 1 − a²`:

```
Output layer:
    dL/dW2[j] = δ_out · a_h[j]
    dL/db2    = δ_out

Hidden layer (per unit j):
    δ_h[j]      = δ_out · W2[j] · (1 − a_h[j]²)        ← tanh derivative
    dL/dW1[j][k] = δ_h[j] · x[k]
    dL/db1[j]    = δ_h[j]
```

### 5.7 Mini-batch SGD update

For each mini-batch of `m` samples, we **average** the gradients (so we divide `δ_out` by `m`) and step every parameter downhill:

```
parameter ← parameter − learning_rate · (average gradient)
```

with `learning_rate = 0.30`. One update per mini-batch → many updates per epoch.

---

## 6. A Full Numerical Dry Run

Let's do **one complete forward + backward + update** by hand. To keep it readable we use a tiny **2-hidden-unit** version (the real file has 10 — identical math, just more units).

### Setup
```
Standardized input:  x = [1.0, −0.5]      True label:  y = 1  (placed)

W1 = [[ 0.5, −0.3],          b1 = [ 0.1, −0.1]
      [ 0.2,  0.4]]
W2 = [ 0.6, −0.7]            b2 = 0.2
learning_rate = 0.3
```

### Step 1 — Forward: hidden layer
```
z_h1 = 0.1 + 0.5·1.0 + (−0.3)·(−0.5) = 0.1 + 0.5 + 0.15 = 0.75
a_h1 = tanh(0.75)  = 0.6351

z_h2 = −0.1 + 0.2·1.0 + 0.4·(−0.5)   = −0.1 + 0.2 − 0.20 = −0.10
a_h2 = tanh(−0.10) = −0.0997
```

### Step 2 — Forward: output + probability
```
z_out = 0.2 + 0.6·0.6351 + (−0.7)·(−0.0997)
      = 0.2 + 0.3811 + 0.0698 = 0.6509
p     = sigmoid(0.6509) = 1/(1 + e^(−0.6509)) = 1/1.5216 = 0.6572
```

### Step 3 — Loss (BCE, y = 1)
```
L = −log(p) = −log(0.6572) = 0.4198
```
The model says **66% placed**; the truth is "placed", so there's room to improve.

### Step 4 — Backward: output error
```
δ_out = p − y = 0.6572 − 1 = −0.3428

dL/dW2[1] = δ_out · a_h1 = −0.3428 · 0.6351 = −0.2177
dL/dW2[2] = δ_out · a_h2 = −0.3428 · (−0.0997) = +0.0342
dL/db2    = δ_out = −0.3428
```

### Step 5 — Backward: hidden error  (tanh': 1 − a²)
```
δ_h1 = δ_out · W2[1] · (1 − a_h1²) = −0.3428 · 0.6  · (1 − 0.6351²) = −0.1227
δ_h2 = δ_out · W2[2] · (1 − a_h2²) = −0.3428 · (−0.7)·(1 − 0.0997²) = +0.2376

dL/dW1[1] = [δ_h1·x₁, δ_h1·x₂] = [−0.1227, +0.0614]
dL/dW1[2] = [δ_h2·x₁, δ_h2·x₂] = [+0.2376, −0.1188]
dL/db1    = [δ_h1, δ_h2]        = [−0.1227, +0.2376]
```

### Step 6 — Update (lr = 0.3)
```
W2[1] ← 0.6  − 0.3·(−0.2177) = 0.6653
W2[2] ← −0.7 − 0.3·( 0.0342) = −0.7102
b2    ← 0.2  − 0.3·(−0.3428) = 0.3028
W1[1] ← [0.5 + 0.0368, −0.3 − 0.0184] = [0.5368, −0.3184]
W1[2] ← [0.2 − 0.0713,  0.4 + 0.0356] = [0.1287,  0.4356]
b1    ← [0.1 + 0.0368, −0.1 − 0.0713] = [0.1368, −0.1713]
```

### Step 7 — Did it work? Re-run the forward pass
```
z_h1 = 0.8328 → a_h1 = 0.6820
z_h2 = −0.2604 → a_h2 = −0.2546
z_out = 0.9374 → p = sigmoid(0.9374) = 0.7186

New loss = −log(0.7186) = 0.3305   (was 0.4198)
```

✅ The probability moved **0.6572 → 0.7186** (toward 1, the correct answer) and the loss **dropped 0.4198 → 0.3305** in a single step. Repeat this over mini-batches for 150 epochs and the network carves out the whole curved boundary.

---

## 7. Why Mini-Batch SGD (the speed math)

```
Full-batch GD:  1 weight update per epoch   (average over all 400 training points)
Mini-batch SGD: ⌈400 / 48⌉ ≈ 9 updates per epoch
```

So in the **same** 150 epochs, mini-batch makes **~9× more updates** (~1,350 vs 150). Each update also uses a slightly **noisy** gradient (from a random subset), and that noise actually *helps* — it knocks the model out of flat spots and forms sharp non-linear boundaries faster. This is exactly why Attempt 2 (full-batch) timed out while mini-batch finished in ~2 seconds. (More on optimizers in [optimizers](../../Improve_NN/5_Optimizers/optimizers.md).)

We shuffle the training order every epoch (`random.shuffle(order)`) so the mini-batches differ each pass.

---

## 8. The Four Plots (and how to read them)

| # | Plot | What it shows | What to look for |
|---|------|---------------|------------------|
| 1 | **The Data** | Green (placed) vs red (not) in CGPA–Profile space | The curved, closed separation — clearly not a straight line |
| 2 | **Decision boundary over epochs** | 6 snapshots (epochs 1, 5, 15, 40, 80, 150); filled = P(placed), black line = the 0.5 boundary | Watch a flat field bend into a closed loop wrapping the green blob |
| 3 | **Loss & Accuracy curves** | Train vs validation BCE loss, and train vs validation accuracy | Both curves track each other (healthy). A rising val-loss would mean overfitting |
| 4 | **Final result** | Learned boundary on validation points (✕ = misclassified) + confusion matrix | Most errors sit right on the boundary; the matrix shows TP/FP/FN/TN |

The **decision boundary** in plots 2 & 4 is drawn by evaluating the trained network on a 70×70 grid of (CGPA, Profile) points and contouring `P(placed)` — it's literally a picture of the function the network learned.

---

## 9. Reading the Final Results

The file prints, on the 100-sample validation set:

```
Accuracy : 0.920
Precision: 0.952   Recall: 0.870
Confusion matrix (val):  TP=40  FP=2  FN=6  TN=52
```

The **confusion matrix** breaks down predictions vs truth:

```
                  Predicted: Not   Predicted: Placed
Actual: Not            TN = 52          FP = 2
Actual: Placed         FN = 6           TP = 40
```

- **Accuracy** = (TP + TN) / total = (40 + 52) / 100 = **0.92** — overall correctness.
- **Precision** = TP / (TP + FP) = 40 / 42 = **0.95** — of those we *called* placed, 95% truly were.
- **Recall** = TP / (TP + FN) = 40 / 46 = **0.87** — of those who *actually* got placed, we caught 87%.

Accuracy alone can lie when classes are imbalanced; precision and recall tell you *which kind* of mistake the model makes. (Why accuracy isn't enough is discussed in [model selection](../../ModelSelection/choosing_the_right_model.md).)

---

## 10. Key Takeaways

1. **Non-linear data needs a hidden layer.** A line scored ~0.50 on the elliptical blob; the MLP reached 0.92. That gap *is* the lesson.
2. **Use tanh (or ReLU), not sigmoid, for hidden layers.** Sigmoid's tiny gradients made the network underfit until we switched.
3. **Classification ⇒ sigmoid output + BCE loss**, which gives the clean `δ = p − y` gradient. MSE is the wrong tool here.
4. **Mini-batch SGD** trains far faster than full-batch (more updates/epoch) and its noise sharpens non-linear boundaries.
5. **Bigger/longer is not better.** 16 units × 300 epochs overfit the noise; 10 units × 150 epochs generalized cleanly. Watch the **validation** curve to choose.
6. **Standardize features (train stats only)** and judge with **precision/recall + a confusion matrix**, not just accuracy.

### Related reading in this repo
- [Binary Cross-Entropy loss](../../LossFunction/3_BinaryCrossEntropy/bce.md) · [Sigmoid](../../ActivationFunction/1_Sigmoid/sigmoid.md) · [Tanh](../../ActivationFunction/2_Tanh/tanh.md)
- [Overfitting & early stopping](../../Improve_NN/1_Overfitting/overfitting.md) · [Optimizers](../../Improve_NN/5_Optimizers/optimizers.md) · [Normalization](../../Improve_NN/2_Normalization/normalization.md)
- The regression sibling: [../regression/implementation.py](../regression/implementation.py)
