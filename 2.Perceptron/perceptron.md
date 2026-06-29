# The Perceptron — From Scratch

The perceptron is the **grandfather of every neural network**: a single neuron, invented by Frank Rosenblatt in **1958**. Understanding it makes everything else (your MLP, activations, backprop) click, because an MLP is literally just *many* of these wired together with smooth activations.

This note explains exactly what [perceptron.py](perceptron.py) does, the math behind it, a worked update by hand, and — crucially — **why it fails** on the kind of data that forced the field to invent hidden layers.

---

## 1. What a perceptron is

One neuron that draws **one straight line** to split two classes.

```
Inputs        Weights         Neuron                    Output
 x1 ───w1──┐
           ├──►  z = w1·x1 + w2·x2 + b  ──►  step(z)  ──►  0 or 1
 x2 ───w2──┘                                  ▲
                                              b (bias)

step(z) = 1  if z ≥ 0      (the neuron "fires")
        = 0  if z <  0      (it stays quiet)
```

That's the whole model. Two weights, one bias, and a hard on/off switch.

---

## 2. The geometry: a perceptron *is* a line

The decision boundary is the set of points where `z = 0`:

```
w1·x1 + w2·x2 + b = 0
```

In 2D that's a **straight line**. On one side `z ≥ 0` → class 1; on the other `z < 0` → class 0. So:

- the **weights** `(w1, w2)` set the line's tilt/orientation,
- the **bias** `b` slides the line away from the origin.

Learning a perceptron = finding the weights and bias that put the line *between* the two classes.

---

## 3. The learning rule (the "basic formula")

We can't know the right line in advance, so we **start from a random line** and fix it one mistake at a time. For each data point:

```
y_pred = step(w1·x1 + w2·x2 + b)
error  = y_true − y_pred              (one of +1, 0, or −1)

w1 ← w1 + lr · error · x1
w2 ← w2 + lr · error · x2
b  ← b  + lr · error
```

Three cases:
- **Correct** (`error = 0`): change nothing — leave the line where it is.
- **Said 0, should be 1** (`error = +1`): **add** the point to the weights → rotate the line *toward* the point so it lands on the firing side.
- **Said 1, should be 0** (`error = −1`): **subtract** the point → push the line *away* so the point falls on the quiet side.

That's the entire algorithm. Sweep over all the points, nudging the line for every mistake; repeat until there are no mistakes left.

---

## 4. A worked update by hand

```
Start:  w1 = −0.2,  w2 = 0.3,  b = −0.1,   lr = 1
Point:  (x1, x2) = (2, 1),  true label y = 1

Forward:
  z = (−0.2)(2) + (0.3)(1) + (−0.1) = −0.4 + 0.3 − 0.1 = −0.2
  step(−0.2) = 0          → predicted 0, but truth is 1  → MISTAKE
  error = 1 − 0 = +1

Update (add the point, because error = +1):
  w1 ← −0.2 + 1·(+1)·2 = 1.8
  w2 ←  0.3 + 1·(+1)·1 = 1.3
  b  ← −0.1 + 1·(+1)   = 0.9

Re-check the same point:
  z = (1.8)(2) + (1.3)(1) + 0.9 = 3.6 + 1.3 + 0.9 = 5.8
  step(5.8) = 1          → now correct ✓
```

One nudge moved the line so this point flipped to the right side. Do this for every mistake, pass after pass, and the line walks into place.

---

## 5. What the code does (Part 1)

[perceptron.py](perceptron.py) generates **1000 linearly separable points** (labelled by a hidden true line, with a small margin gap so the classes don't touch), starts from a **random line**, and applies the rule above.

```
PART 1: PERCEPTRON ON LINEARLY SEPARABLE DATA
  Epoch  1:   13 misclassified
  Epoch  2:    0 misclassified
  CONVERGED in 2 epochs -> found a separating line.
```

The plots show:
1. The **initial random line** (dotted) and the **final learned line** (solid) that cleanly separates the two classes — the line literally moved into the gap.
2. **Misclassifications per epoch** dropping to **0**.

### Why it's guaranteed to work here — the Perceptron Convergence Theorem
> If the data is linearly separable, the perceptron rule is **guaranteed** to find a separating line in a **finite** number of steps.

That's a real theorem (Rosenblatt / Novikoff). It's why Part 1 always converges — here, in just 2 passes.

---

## 6. The limitations (Part 2 — the important part)

### Limitation 1 — it can only draw a straight line

The perceptron's boundary is *always* a single straight line. So it **cannot** separate classes that aren't linearly separable. Part 2 of the code shows the classic case: points **inside a circle** are class 1, outside are class 0.

```
PART 2: PERCEPTRON ON NON-SEPARABLE DATA (a circle)
  Best (lowest) error over 50 epochs: 435 / 1000
  Final error: 447 / 1000  -> NEVER reaches 0 (no line can do it).
```

The plot shows the perceptron's "best" line slashing through the middle — separating essentially nothing. No straight line can wrap around the inside of a circle.

### Limitation 2 — it never converges on non-separable data

On separable data, errors hit 0 and we stop. On the circle, the error count just **oscillates forever** (the line keeps getting shoved back and forth by points it can never all satisfy). The convergence theorem only promises termination when a separating line *exists*.

### The famous example: XOR (4 points, 2 columns)

You don't even need 1000 points to break it — 4 will do:

```
x1  x2 | y
 0   0 | 0
 0   1 | 1
 1   0 | 1
 1   1 | 0     (output is 1 when the inputs differ)
```

There is no `w1, w2, b` that satisfies all four (try it: the four inequalities contradict each other). In **1969**, Minsky & Papert proved this in their book *Perceptrons*, and the realization that a single perceptron can't even do XOR triggered an "AI winter" that stalled neural-network research for years.

### Limitation 3 — a hard step, so no probabilities and no gradient

`step(z)` outputs a flat 0 or 1. Its slope is 0 almost everywhere, so you **can't take a useful derivative** of it — which means you can't train it with gradient descent / backpropagation. The perceptron rule is a clever hand-crafted fix for exactly this; it doesn't generalize to deep networks.

### Limitation 4 — it finds *a* line, not the *best* line

The moment errors hit 0 it stops, even if the line sits right against the points. It has no concept of a safety **margin** (that idea came later, with Support Vector Machines).

---

## 7. How these limits were fixed → the MLP

Each limitation maps directly to something you already built:

| Perceptron limitation | The fix | Where in this repo |
|---|---|---|
| Only a straight line | **Stack neurons in hidden layers** + non-linear activations → bends straight lines into curves | [MLP/classification](../MLP/classification/classification.md) |
| Hard step (no gradient) | **Smooth activations** (sigmoid, tanh, ReLU) → differentiable → trainable by backprop | [Sigmoid](../ActivationFunction/1_Sigmoid/sigmoid.md) · [ReLU](../ActivationFunction/3_ReLU/relu.md) |
| Can't do XOR / circle | A 2-layer network carves curved / closed regions | the circle in [classification.py](../MLP/classification/classification.py) |
| No probabilities | Sigmoid output + a proper loss | [Binary Cross-Entropy](../LossFunction/3_BinaryCrossEntropy/bce.md) |

A modern neuron is just a perceptron with the hard step swapped for a smooth activation — and an MLP is a grid of them. Everything you learned in the `MLP` folder grows directly out of this one neuron.

---

## 8. TL;DR

- A **perceptron** is one neuron: `z = w·x + b`, then a hard step → 0/1. Geometrically, it's a single straight line splitting the plane.
- It learns by the **basic rule** `w ← w + lr·(y − ŷ)·x`: start from a random line, and for every misclassified point, shove the line toward the right side. Repeat until no mistakes.
- On **linearly separable** data it is *guaranteed* to converge (here, in 2 epochs on 1000 points).
- Its **limits**: only straight-line boundaries (fails on XOR and circles), never converges on non-separable data, a non-differentiable step (no gradient descent), and no margin.
- The cure is **hidden layers + smooth activations + backprop** — i.e. the MLP.
