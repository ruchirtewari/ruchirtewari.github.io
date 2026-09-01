"""Chapter 2, §2.1 — KL divergence is locally a quadratic bowl.

Compare exact D_KL(p_theta || p_{theta+delta}) with the quadratic
(1/2) F(theta) delta^2 for Bernoulli (F = 1/(theta(1-theta))) and for a
Gaussian mean (F = 1/sigma^2, where KL = delta^2 / (2 sigma^2) exactly).

Expected result: the quadratic matches KL to <1% for |delta| small relative
to the bowl width (exactly for the Gaussian mean); the third-order mismatch
grows with delta, and sooner near the edge (theta = 0.95 shows ~12% already
at |delta| = 0.01). Bowls narrow visibly as theta -> edge and sigma -> small.
"""

import matplotlib
import numpy as np

from common import get_arg_parser, init, savefig


def kl_bernoulli(t, d):
    """D_KL(Bern(t) || Bern(t+d)) in nats."""
    q = t + d
    return t * np.log(t / q) + (1 - t) * np.log((1 - t) / (1 - q))


def kl_gauss_mean(delta, sigma):
    """D_KL(N(mu, s^2) || N(mu+delta, s^2)) in nats — exactly quadratic."""
    return delta**2 / (2 * sigma**2)


def main():
    args = get_arg_parser("KL bowl vs (1/2) Fisher delta^2").parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = init(args, "02_01_kl_bowl_and_fisher")

    n = 101 if args.quick else 401
    thetas = [0.5, 0.8, 0.95]
    sigmas = [0.5, 1.0, 2.0]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    print(f"{'family':<14} {'param':>7} {'Fisher F':>10} {'max rel err (|d|<0.01)':>24}")
    print("-" * 60)

    for ax, t in zip(axes[0], thetas):
        F = 1 / (t * (1 - t))
        dmax = 0.8 * min(t, 1 - t)  # keep theta+delta inside (0,1)
        d = np.linspace(-dmax, dmax, n)
        kl = kl_bernoulli(t, d)
        quad = 0.5 * F * d**2
        ax.plot(d, kl, label="exact KL")
        ax.plot(d, quad, "--", label=r"$\frac{1}{2}F\delta^2$")
        ax.set_title(rf"Bernoulli $\theta$={t},  F={F:.2f}")
        ax.set_xlabel(r"$\delta$")
        ax.legend(fontsize=8)
        small = np.abs(d) < 0.01
        small &= d != 0
        err = np.max(np.abs(quad[small] - kl[small]) / kl[small])
        print(f"{'Bernoulli':<14} {t:>7} {F:>10.2f} {err:>24.2e}")

    for ax, s in zip(axes[1], sigmas):
        F = 1 / s**2
        d = np.linspace(-2, 2, n)
        kl = kl_gauss_mean(d, s)
        quad = 0.5 * F * d**2
        ax.plot(d, kl, label="exact KL")
        ax.plot(d, quad, "--", label=r"$\frac{1}{2}F\delta^2$")
        ax.set_title(rf"Gaussian mean $\sigma$={s},  F={F:.2f}")
        ax.set_xlabel(r"$\delta$")
        ax.legend(fontsize=8)
        small = (np.abs(d) < 0.01) & (d != 0)
        # for the Gaussian mean the match is exact — error is machine epsilon
        err = 0.0 if not small.any() else \
            np.max(np.abs(quad[small] - kl[small]) / np.maximum(kl[small], 1e-300))
        print(f"{'Gaussian mean':<14} {s:>7} {F:>10.2f} {err:>24.2e}")

    axes[0][0].set_ylabel("KL (nats)")
    axes[1][0].set_ylabel("KL (nats)")
    fig.suptitle("KL divergence is locally a quadratic bowl with curvature = Fisher information")
    fig.tight_layout()
    savefig(fig, outdir, "kl_bowls.png", show=args.show)


if __name__ == "__main__":
    main()
