"""Chapter 1, §1.3 — conditional entropy vs context length.

Estimate H(next char | k previous chars) for k = 0..8 by counting on the
Shakespeare corpus, and count distinct observed k-contexts against the
possible 27^k.

Expected result: entropy falls with k, then the estimate turns down toward
zero unreliably precisely where observed contexts stop repeating — the
state-explosion wall (27^k outruns the corpus around k ~ 4-5).
"""

import math

import matplotlib
import numpy as np

from common import get_arg_parser, init, load_shakespeare, savefig

LOG2_27 = math.log2(27)


def conditional_entropy(text, k):
    """Empirical H(X | k-context) with add-one smoothing inside each context.

    H = sum_ctx p(ctx) * H(next | ctx), where p(ctx) is the empirical context
    frequency and each within-context distribution is (count+1)/(n+27).
    Returns (bits/char, number of distinct observed contexts).
    """
    counts = {}
    for i in range(k, len(text)):
        d = counts.setdefault(text[i - k:i], {})
        d[text[i]] = d.get(text[i], 0) + 1
    total = len(text) - k
    h = 0.0
    for d in counts.values():
        n = sum(d.values())
        h_ctx = 0.0
        for c in d.values():
            p = (c + 1) / (n + 27)
            h_ctx -= p * math.log2(p)
        # the (27 - len(d)) unseen symbols each get probability 1/(n+27)
        p0 = 1 / (n + 27)
        h_ctx -= (27 - len(d)) * p0 * math.log2(p0)
        h += (n / total) * h_ctx
    return h, len(counts)


def main():
    args = get_arg_parser("Conditional entropy vs context length k").parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir = init(args, "01_03_entropy_vs_context_length")

    text = load_shakespeare()
    kmax = 6 if args.quick else 8
    if args.quick:
        text = text[:500_000]
    print(f"corpus: {len(text):,} chars, k = 0..{kmax}\n")

    ks = list(range(kmax + 1))
    ents, n_ctx = [], []
    print(f"{'k':>2} {'H (bits/char)':>14} {'observed ctx':>14} {'possible 27^k':>16}")
    for k in ks:
        h, nc = conditional_entropy(text, k)
        ents.append(h)
        n_ctx.append(nc)
        print(f"{k:>2} {h:>14.3f} {nc:>14,} {27**k:>16,}")

    # ---- plot (a): entropy vs k with Shannon's reference points ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, ents, "o-", color="C0", label="empirical estimate")
    for val, lab in [(LOG2_27, "uniform 4.75"), (4.1, "Shannon order-0 ~4.1"),
                     (3.1, "Shannon order-1 ~3.1"), (1.0, "Shannon true ~1.0")]:
        ax.axhline(val, ls="--", lw=0.8, color="gray")
        ax.text(kmax, val, lab, fontsize=8, va="bottom", ha="right", color="gray")
    ax.set_xlabel("context length k (chars)")
    ax.set_ylabel("H(next char | context)  [bits]")
    ax.set_title("Conditional entropy vs context length")
    ax.legend()
    savefig(fig, outdir, "entropy_vs_k.png", show=args.show)

    # ---- plot (b): the sparsity wall ------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(ks, [max(n, 1) for n in n_ctx], "o-", label="distinct observed contexts")
    ax.semilogy(ks, [27.0**k for k in ks], "s--", label=r"possible contexts $27^k$")
    ax.axhline(len(text), ls=":", color="gray")
    ax.text(0, len(text), " corpus length", fontsize=8, va="bottom", color="gray")
    ax.set_xlabel("context length k (chars)")
    ax.set_ylabel("number of contexts (log scale)")
    ax.set_title("The state-explosion wall")
    ax.legend()
    savefig(fig, outdir, "context_explosion.png", show=args.show)


if __name__ == "__main__":
    main()
