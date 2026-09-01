"""Chapter 4, §4.3 — noise selects flat minima (entry 10).

Purpose: demonstrate noise-selects-flat-basins and its generalization payoff
on a constructed 2-D loss with two equal-depth global minima — one wide
basin, one narrow. Noisy GD from many random starts at ~8 noise scales;
"test loss" is the same surface shifted by a small offset.

Expected result (acceptance): fraction of runs ending in the wide basin
rises toward 1 with noise scale; the wide basin's test loss is lower under
the shift.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import get_arg_parser, init, savefig

WIDE_C = np.array([-2.0, 0.0]);  W_WIDE = 2.0
NARR_C = np.array([2.0, 0.0]);   W_NARR = 0.25
SHIFT = np.array([0.35, 0.25])
CONF = 0.02  # weak symmetric confining bowl so far starts still descend


def loss(p, shift=None):
    """p: (..., 2). Equal-depth double-Gaussian well + tiny confinement."""
    wc, nc = WIDE_C, NARR_C
    if shift is not None:
        wc, nc = wc + shift, nc + shift
    dw = ((p - wc) ** 2).sum(-1)
    dn = ((p - nc) ** 2).sum(-1)
    return (1.0 - np.exp(-dw / W_WIDE) - np.exp(-dn / W_NARR)
            + CONF * (p ** 2).sum(-1))


def grad(p):
    dw = p - WIDE_C
    dn = p - NARR_C
    g = (2 / W_WIDE) * np.exp(-(dw ** 2).sum(-1, keepdims=True) / W_WIDE) * dw
    g += (2 / W_NARR) * np.exp(-(dn ** 2).sum(-1, keepdims=True) / W_NARR) * dn
    return g + 2 * CONF * p


def run_batch(n_starts, noise, steps, lr=0.05):
    p = np.random.uniform(-4, 4, size=(n_starts, 2))
    anneal_from = int(0.7 * steps)  # last 30%: anneal noise to 0, settle
    for t in range(steps):
        eta = noise if t < anneal_from else noise * (steps - t) / (steps - anneal_from)
        p = p - lr * grad(p) + eta * np.sqrt(lr) * np.random.randn(n_starts, 2)
    for _ in range(800):  # deterministic settle: every run ends in a basin
        p = p - lr * grad(p)
    return p


def main():
    args = get_arg_parser("noisy GD selects the wide basin").parse_args()
    outdir = init(args, "04_03_flat_minima_and_noise")
    n_starts = 100 if args.quick else 500
    steps = 200 if args.quick else 600
    noises = np.array([0.0, 0.04, 0.08, 0.15, 0.25, 0.35, 0.5, 0.65])

    frac_wide, test_by_basin = [], []
    for nz in noises:
        p = run_batch(n_starts, nz, steps)
        in_wide = ((p - WIDE_C) ** 2).sum(-1) < ((p - NARR_C) ** 2).sum(-1)
        frac_wide.append(in_wide.mean())
        test = loss(p, shift=SHIFT)
        test_by_basin.append((nz, test[in_wide], test[~in_wide]))
        print(f"noise {nz:5.2f}: wide-basin fraction {in_wide.mean():5.2f}   "
              f"test loss wide {test[in_wide].mean() if in_wide.any() else float('nan'):6.3f}"
              f"  narrow {test[~in_wide].mean() if (~in_wide).any() else float('nan'):6.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # surface
    g = np.linspace(-4.5, 4.5, 300)
    xx, yy = np.meshgrid(g, g)
    Z = loss(np.stack([xx, yy], -1))
    ax = axes[0]
    cs = ax.contourf(xx, yy, Z, levels=40, cmap="viridis")
    fig.colorbar(cs, ax=ax)
    ax.plot(*WIDE_C, "w*", ms=12); ax.text(*WIDE_C + [0.2, 0.2], "wide", color="w")
    ax.plot(*NARR_C, "w*", ms=12); ax.text(*NARR_C + [0.2, 0.2], "narrow", color="w")
    ax.set_title("loss surface (equal-depth minima)")

    # fraction in wide basin vs noise
    ax = axes[1]
    ax.plot(noises, frac_wide, "o-")
    ax.set_xlabel("noise scale"); ax.set_ylabel("fraction ending in wide basin")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    ax.set_title("noise selects the flat basin")

    # test loss scatter by basin
    ax = axes[2]
    for i, (nz, tw, tn) in enumerate(test_by_basin):
        jitter = lambda k: 0.008 * np.random.randn(k)
        if len(tw):
            ax.scatter(np.full(len(tw), nz) + jitter(len(tw)), tw, s=8,
                       color="#1f77b4", alpha=0.4,
                       label="wide basin" if i == 0 else None)
        if len(tn):
            ax.scatter(np.full(len(tn), nz) + jitter(len(tn)), tn, s=8,
                       color="#d62728", alpha=0.4,
                       label="narrow basin" if i == 0 else None)
    ax.set_xlabel("noise scale"); ax.set_ylabel("test loss (shifted surface)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("generalization payoff of the wide basin")
    savefig(fig, outdir, "flat_minima_and_noise.png", show=args.show)

    all_w = np.concatenate([t[1] for t in test_by_basin if len(t[1])])
    all_n = np.concatenate([t[2] for t in test_by_basin if len(t[2])])
    print(f"\nmean test loss: wide {all_w.mean():.3f}  narrow {all_n.mean():.3f}")
    print(f"wide-basin fraction: {frac_wide[0]:.2f} (no noise) -> "
          f"{frac_wide[-1]:.2f} (max noise)")
    print(f"acceptance: fraction rises with noise: {frac_wide[-1] > frac_wide[0]}; "
          f"wide test loss lower: {all_w.mean() < all_n.mean()}")


if __name__ == "__main__":
    main()
