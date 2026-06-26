# ============================================================
# RMSPROP — Root Mean Square Propagation (From Scratch)
# ============================================================
# s_t = beta * s_(t-1) + (1 - beta) * gradient^2
# w = w - lr * gradient / (sqrt(s_t) + eps)
#
# Fixes Adagrad's problem: uses MOVING AVERAGE of squared
# gradients instead of accumulating all of them.
# The LR adapts but doesn't decay to zero.
# ============================================================

import math, random
random.seed(42)

X = [[8, 4], [5, 2], [7, 3], [3, 1]]
Y = [15, 7, 12, 4]
n = len(Y)

def leaky_relu(z): return z if z > 0 else 0.01 * z
def leaky_relu_deriv(z): return 1.0 if z > 0 else 0.01
def he_init(fan_in): return random.gauss(0, math.sqrt(2.0 / fan_in))

H1, H2 = 4, 4
EPOCHS = 100


class RMSprop:
    """RMSprop: Adagrad with a moving average.

    s = beta * s + (1-beta) * grad^2     (exponential moving average)
    w = w - lr * grad / (sqrt(s) + eps)

    beta=0.9 means: remember ~10 recent squared gradients.
    Old gradients fade away -> LR doesn't decay to zero.
    """
    def __init__(self, num_params, lr=0.001, beta=0.9, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = epsilon
        self.s = [0.0] * num_params

    def step(self, params, grads):
        new = []
        for i in range(len(params)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grads[i] ** 2
            new.append(params[i] - self.lr * grads[i] / (math.sqrt(self.s[i]) + self.eps))
        return new


class Adagrad:
    def __init__(self, num_params, lr=0.01, epsilon=1e-8):
        self.lr = lr
        self.eps = epsilon
        self.s = [0.0] * num_params
    def step(self, params, grads):
        new = []
        for i in range(len(params)):
            self.s[i] += grads[i] ** 2
            new.append(params[i] - self.lr * grads[i] / (math.sqrt(self.s[i]) + self.eps))
        return new


def train(opt_class, lr, label, **kwargs):
    random.seed(42)
    W1 = [[he_init(2) for _ in range(2)] for _ in range(H1)]
    b1 = [0.0] * H1
    W2 = [[he_init(H1) for _ in range(H1)] for _ in range(H2)]
    b2 = [0.0] * H2
    W3 = [[he_init(H2) for _ in range(H2)]]
    b3 = [0.0]

    flat = []
    for r in W1: flat.extend(r)
    for r in W2: flat.extend(r)
    flat.extend(W3[0]); flat.extend(b1); flat.extend(b2); flat.extend(b3)

    opt = opt_class(len(flat), lr=lr, **kwargs)
    print(f"\n  {label}")
    losses = []

    for epoch in range(1, EPOCHS + 1):
        grads = [0.0] * len(flat)
        total_loss = 0.0

        for i in range(n):
            x, y = X[i], Y[i]
            idx = 0
            w1 = [[flat[idx+j*2+k] for k in range(2)] for j in range(H1)]; idx += H1*2
            w2 = [[flat[idx+j*H1+k] for k in range(H1)] for j in range(H2)]; idx += H2*H1
            w3 = [flat[idx:idx+H2]]; idx += H2
            b_1 = flat[idx:idx+H1]; idx += H1
            b_2 = flat[idx:idx+H2]; idx += H2
            b_3 = flat[idx:idx+1]

            z1 = [sum(w1[j][k]*x[k] for k in range(2))+b_1[j] for j in range(H1)]
            a1 = [leaky_relu(z) for z in z1]
            z2 = [sum(w2[j][k]*a1[k] for k in range(H1))+b_2[j] for j in range(H2)]
            a2 = [leaky_relu(z) for z in z2]
            yh = sum(w3[0][k]*a2[k] for k in range(H2)) + b_3[0]
            total_loss += (yh - y)**2
            dL = (2.0/n) * (yh - y)

            dW3 = [dL*a2[k] for k in range(H2)]
            dL_z2 = [dL*w3[0][j]*leaky_relu_deriv(z2[j]) for j in range(H2)]
            dW2 = [[dL_z2[j]*a1[k] for k in range(H1)] for j in range(H2)]
            dL_a1 = [sum(dL_z2[j]*w2[j][k] for j in range(H2)) for k in range(H1)]
            dL_z1 = [dL_a1[j]*leaky_relu_deriv(z1[j]) for j in range(H1)]
            dW1 = [[dL_z1[j]*x[k] for k in range(2)] for j in range(H1)]

            g = []
            for r in dW1: g.extend(r)
            for r in dW2: g.extend(r)
            g.extend(dW3); g.extend(dL_z1); g.extend(dL_z2); g.append(dL)
            for gi in range(len(grads)): grads[gi] += g[gi]

        flat = opt.step(flat, grads)
        mse = total_loss / n
        losses.append(mse)

        eff_lr = lr / (math.sqrt(opt.s[0]) + opt.eps) if opt.s[0] > 0 else lr

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}  |  Eff LR[0]: {eff_lr:.6f}")

    return losses


print("=" * 60)
print("  RMSPROP vs ADAGRAD")
print("  Showing how RMSprop fixes Adagrad's LR decay")
print("=" * 60)

ada_loss = train(Adagrad, lr=0.1, label="Adagrad (lr=0.1)")
rms_loss = train(RMSprop, lr=0.01, label="RMSprop (lr=0.01, beta=0.9)")

print(f"\n{'=' * 60}")
print(f"  Adagrad final MSE:  {ada_loss[-1]:.4f}")
print(f"  RMSprop final MSE:  {rms_loss[-1]:.4f}")
print(f"\n  RMSprop's moving average prevents LR from dying!")
print(f"{'=' * 60}")
