"""Chapter 4, §4.2 — Adam on the Fisher ladder (entry 9).

Purpose: place Adam on the Fisher ladder — show it behaves like a diagonal
natural-gradient method in disguise, on an ill-conditioned logistic
regression (20 features, scales spanning 10^3, condition number ~10^6).

Compares plain GD, Adam (numpy), diagonal-Fisher-preconditioned GD, and
exact natural gradient (full Fisher = X^T S X / n, S = diag(p(1-p))).

Expected result (acceptance): Adam and diagonal-Fisher curves nearly
coincide; exact NGD fastest; plain GD slowest by orders of magnitude in
steps (loss vs step, log-y).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import get_arg_parser, init, savefig

EPS = 1e-8


def make_data(n=5000, d=20):
    scales = np.logspace(0, 3, d)
    X = np.random.randn(n, d) * scales
    w_true = np.random.randn(d) / scales  # keeps logits O(1)
    p = 1 / (1 + np.exp(-X @ w_true))
    y = (np.random.rand(n) < p).astype(float)
    return X, y, scales


def loss_grad(w, X, y):
    z = X @ w
    p = 1 / (1 + np.exp(-z))
    L = np.mean(np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0) - y * z)
    g = X.T @ (p - y) / len(y)
    return L, g, p


def train(method, X, y, steps, lr):
    d = X.shape[1]
    w = np.zeros(d)
    m = np.zeros(d); v = np.zeros(d)  # adam state
    losses = []
    for t in range(1, steps + 1):
        L, g, p = loss_grad(w, X, y)
        losses.append(L)
        if method == "gd":
            step = g
        elif method == "adam":
            b1, b2 = 0.9, 0.999
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g ** 2
            step = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + EPS)
        elif method == "diag_fisher":
            s = p * (1 - p)
            fdiag = (X ** 2 * s[:, None]).mean(0)
            step = g / (fdiag + 1e-6)
        elif method == "full_fisher":
            s = p * (1 - p)
            F = X.T @ (X * s[:, None]) / len(y)
            F += 1e-8 * np.trace(F) / d * np.eye(d)
            step = np.linalg.solve(F, g)
        w = w - lr * step
    return np.array(losses)


def main():
    args = get_arg_parser("preconditioning ladder: GD, Adam, diag-F, full-F").parse_args()
    outdir = init(args, "04_02_preconditioning_comparison")
    steps = 300 if args.quick else 2000

    X, y, scales = make_data()
    # Hessian condition number at w=0 (S = 1/4 I): cond(X^T X)
    H = X.T @ X / len(y)
    cond = np.linalg.cond(H)
    print(f"feature scales: {scales[0]:.0f} .. {scales[-1]:.0f}   "
          f"condition number ~ {cond:.2e}")

    methods = [
        ("plain GD", "gd", 2.0 / np.linalg.eigvalsh(H).max() * 4),
        ("Adam", "adam", 0.01),
        ("diag-Fisher GD", "diag_fisher", 0.05),
        ("exact NGD (full Fisher)", "full_fisher", 1.0),
    ]
    curves = {name: train(kind, X, y, steps, lr) for name, kind, lr in methods}

    # reference optimum from a long full-Fisher (Newton-like) run
    L_star = train("full_fisher", X, y, 200, 1.0)[-1]
    L_star = min(L_star, min(c.min() for c in curves.values()))

    print(f"\noptimal loss L* = {L_star:.6f}")
    print(f"{'method':26s} {'final loss':>12s} {'L - L*':>12s}")
    for name, _, lr in methods:
        Lf = curves[name][-1]
        print(f"{name:26s} {Lf:12.6f} {Lf - L_star:12.2e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, _, _ in methods:
        ax.semilogy(np.maximum(curves[name] - L_star, 1e-12), label=name)
    ax.set_xlabel("step"); ax.set_ylabel("loss - L*  (log scale)")
    ax.set_title("preconditioning ladder on ill-conditioned logistic regression")
    ax.legend(); ax.grid(alpha=0.3)
    savefig(fig, outdir, "loss_curves.png", show=args.show)

    # Adam vs diag-Fisher similarity
    a, dfc = curves["Adam"], curves["diag-Fisher GD"]
    print(f"\nAdam vs diag-Fisher: final losses {a[-1]:.5f} vs {dfc[-1]:.5f}")
    order = sorted(curves, key=lambda k: curves[k][-1])
    print(f"ranking (best final loss first): {order}")


if __name__ == "__main__":
    main()
