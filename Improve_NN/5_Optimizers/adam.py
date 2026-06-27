# ============================================================
# ADAM — Adaptive Moment Estimation (From Scratch)
# ============================================================
# m_t = beta1 * m_(t-1) + (1 - beta1) * gradient
# v_t = beta2 * v_(t-1) + (1 - beta2) * gradient^2
#
# Bias Correction:
# m_hat = m_t / (1 - beta1^t)
# v_hat = v_t / (1 - beta2^t)
#
# Update Rule:
# w = w - lr * m_hat / (sqrt(v_hat) + eps)
#
# Combines Momentum (first moment) and RMSprop (second moment)
# with bias correction for early training steps.
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

class Adam:
    """Adam Optimizer.
    Combines momentum (exponential moving average of gradients) and
    RMSprop (exponential moving average of squared gradients) with bias correction.

    Canonical defaults: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8.
    NOTE: Adam's usual LR (0.001) is ~10x SMALLER than SGD's (0.01-0.1) -- do not
    copy an SGD learning rate into Adam. This demo passes lr=0.01 explicitly only
    so the SGD-vs-Adam comparison below is a fair head-to-head at the same LR.
    """
    def __init__(self, num_params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = [0.0] * num_params
        self.v = [0.0] * num_params
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        new = []
        for i in range(len(params)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grads[i] ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)

            # Update parameters
            new.append(params[i] - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
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

    opt = opt_class(len(flat), lr=lr, **kwargs) if opt_class else None
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

        if opt:
            flat = opt.step(flat, grads)
        else:
            flat = [p - lr * g for p, g in zip(flat, grads)]

        mse = total_loss / n
        losses.append(mse)
        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:>3}  |  MSE: {mse:.4f}")
    return losses


print("=" * 60)
print("  SGD vs ADAM")
print("=" * 60)

sgd = train(None, lr=0.01, label="Vanilla SGD")
adam = train(Adam, lr=0.01, label="Adam (beta1=0.9, beta2=0.999)")

print(f"\n{'=' * 60}")
print(f"  Vanilla SGD final MSE:  {sgd[-1]:.4f}")
print(f"  Adam final MSE:         {adam[-1]:.4f}")
print(f"{'=' * 60}")
