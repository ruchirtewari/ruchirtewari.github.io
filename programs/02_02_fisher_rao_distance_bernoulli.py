"""Chapter 2, §2.2 — the Euclidean ruler vs the Fisher-Rao ruler.

Compare |p-q| with d_FR(p,q) = 2 arccos(sqrt(pq) + sqrt((1-p)(1-q))) over
the Bernoulli family, then verify empirically with a sequential
likelihood-ratio test: flips needed to distinguish (0.50, 0.51) vs
(0.98, 0.99) at 95% confidence.

Expected result: flips-to-distinguish tracks the Fisher-Rao ordering, not
the Euclidean one — (0.98, 0.99) needs far fewer flips than (0.50, 0.51)
despite identical |p-q| = 0.01.
"""

import matplotlib
import numpy as np

from common import get_arg_parser, init, savefig


def d_fisher_rao(p, q):
    arg = np.sqrt(p * q) + np.sqrt((1 - p) * (1 - q))
    return 2 * np.arccos(np.clip(arg, -1, 1))


def flips_to_distinguish(pa, pb, n_trials, rng, chunk=32768):
    """Median N for a sequential LLR test (truth = Bernoulli(pa)) to cross
    |LLR| > log(19), i.e. 95% posterior odds between the two hypotheses."""
    thresh = np.log(19.0)
    llr1 = np.log(pa / pb)          # LLR increment for a heads
    llr0 = np.log((1 - pa) / (1 - pb))  # ... for a tails
    ns = np.empty(n_trials)
    for t in range(n_trials):
        total, n_done = 0.0, 0
        while True:
            x = rng.random(chunk) < pa
            llr = total + np.cumsum(np.where(x, llr1, llr0))
            hit = np.nonzero(np.abs(llr) > thresh)[0]
            if hit.size:
                ns[t] = n_done + hit[0] + 1
                break
            total = llr[-1]
            n_done += chunk
    return np.median(ns)


def main():
    args = get_arg_parser("Fisher-Rao vs Euclidean distance on Bernoulli").parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = init(args, "02_02_fisher_rao_distance_bernoulli")
    rng = np.random.default_rng(args.seed)

    # ---- heatmaps over (p, q) -------------------------------------------
    n = 97 if args.quick else 193
    grid = np.linspace(0.02, 0.98, n)
    P, Q = np.meshgrid(grid, grid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, title in [(axes[0], np.abs(P - Q), "Euclidean  |p - q|"),
                         (axes[1], d_fisher_rao(P, Q), "Fisher-Rao  d_FR(p, q)")]:
        im = ax.pcolormesh(P, Q, Z, shading="auto", cmap="viridis")
        ax.set_xlabel("p")
        ax.set_ylabel("q")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    savefig(fig, outdir, "heatmaps.png", show=args.show)

    # ---- slice from p = 0.5 ---------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, np.abs(0.5 - grid), label="|0.5 - q|")
    ax.plot(grid, d_fisher_rao(0.5, grid), label="d_FR(0.5, q)")
    ax.set_xlabel("q")
    ax.set_ylabel("distance")
    ax.set_title("Both rulers, measured from p = 0.5")
    ax.legend()
    savefig(fig, outdir, "slice_from_half.png", show=args.show)

    # ---- sequential likelihood-ratio distinguishability test ------------
    n_trials = 100 if args.quick else 1000
    pairs = [(0.50, 0.51), (0.98, 0.99)]
    print(f"\nsequential LLR test, |LLR| > log(19) (95%), {n_trials} trials each\n")
    print(f"{'pair':<14} {'d_Euclid':>9} {'d_FR':>8} {'median flips':>13}")
    print("-" * 48)
    for pa, pb in pairs:
        de = abs(pa - pb)
        dfr = d_fisher_rao(pa, pb)
        med = flips_to_distinguish(pa, pb, n_trials, rng)
        print(f"({pa:.2f}, {pb:.2f})  {de:>9.3f} {dfr:>8.4f} {med:>13.0f}")
    print("\nSame Euclidean gap, very different work: N tracks d_FR, not |p-q|.")


if __name__ == "__main__":
    main()
