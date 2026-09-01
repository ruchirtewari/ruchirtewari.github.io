"""Chapter 8, §8.3 - interpretability_case_study.py (capstone, catalog entry 22).

Purpose: a runnable SCAFFOLD for the full interpretability loop --
quantify -> hypothesize -> probe -> intervene -> report -- demonstrated end to
end on one concrete default behavior of GPT-2 (its strong prior to continue
"The capital of France is" with " Paris", and how a steering direction built
from country/capital contrasts moves it). Stages 1, 3, 4, 5 actually run; stage
2 is where you write competing hypotheses in your own words.

Expected result (acceptance): the script emits a filled-in markdown report to
the output directory with measured effect sizes and confidence intervals, plus
the figure behind them -- the deliverable that makes it science, not a demo.
The point is the scaffold: swap DEFAULT_* for your own behavior and rerun.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (add_vector_hook, get_arg_parser, init, load_causal_lm,
                    remove_hooks, savefig, ResidualCapture)

# --- the default behavior under study (edit these to study your own) ---------
DEFAULT_PROMPTS = [
    "The capital of France is",
    "France's capital city is called",
    "If you visit France, the capital you'll land in is",
    "The seat of the French government is in the city of",
    "Bonjour from the capital of France, the city of",
]
DEFAULT_TARGET = " Paris"
# contrast pairs for a "French capital" direction: (positive, negative) prompts
CONTRAST_POS = ["The capital of France is", "Paris is the capital of France."]
CONTRAST_NEG = ["The capital of Japan is", "Tokyo is the capital of Japan."]
STEER_LAYER = 6

# Stage 2 -- YOU author these (they are printed and saved verbatim):
HYPOTHESES = [
    "H1: the ' Paris' completion is driven by a mid-layer 'French capital' "
    "direction in the residual stream; adding it should strengthen ' Paris', "
    "subtracting it should weaken it.",
    "H2: the completion is lexical/positional (the token 'France' late in the "
    "prompt), not a steerable concept direction; a country-contrast vector "
    "should then have little effect.",
]


def target_logprob(model, tok, device, prompt, target, hooks_fn=None):
    ids = tok(prompt, return_tensors="pt").to(device)
    tgt = tok(target, add_special_tokens=False)["input_ids"][0]
    handles = hooks_fn() if hooks_fn else []
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]
    remove_hooks(handles)
    return torch.log_softmax(logits, -1)[tgt].item()


def bootstrap_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_direction(model, tok, device):
    """Difference-of-means residual direction at STEER_LAYER from contrast pairs."""
    def mean_act(prompts):
        vs = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").to(device)
            with ResidualCapture(model, layers=[STEER_LAYER]) as cap:
                model(**ids)
            vs.append(cap.acts[STEER_LAYER][0, -1])
        return torch.stack(vs).mean(0)
    return mean_act(CONTRAST_POS) - mean_act(CONTRAST_NEG)


def main():
    args = get_arg_parser("interpretability capstone scaffold").parse_args()
    outdir = init(args, "08_03_interpretability_case_study")

    try:
        model, tok, device = load_causal_lm()  # gpt2 by default
    except Exception as e:
        print(f"[skip] could not load base model: {e}")
        return

    seeds = [0, 1] if args.quick else [0, 1, 2]

    # --- Stage 1: quantify prevalence over prompt variants x seeds -----------
    print("=== Stage 1: quantify ===")
    base_lps = []
    for s in seeds:
        torch.manual_seed(s)
        for p in DEFAULT_PROMPTS:
            base_lps.append(target_logprob(model, tok, device, p, DEFAULT_TARGET))
    b_mean, b_lo, b_hi = bootstrap_ci(base_lps)
    print(f"baseline log P('{DEFAULT_TARGET}') = {b_mean:.3f} [{b_lo:.3f}, {b_hi:.3f}] "
          f"over {len(base_lps)} prompt x seed cells")

    # --- Stage 2: hypotheses (author-written) --------------------------------
    print("\n=== Stage 2: competing hypotheses (edit HYPOTHESES in the file) ===")
    for h in HYPOTHESES:
        print("  " + h)

    # --- Stage 3: probe (logit-lens: is the target already promoted mid-stack?)
    print("\n=== Stage 3: probe (logit lens across layers) ===")
    ids = tok(DEFAULT_PROMPTS[0], return_tensors="pt").to(device)
    with ResidualCapture(model) as cap:
        model(**ids)
    tgt = tok(DEFAULT_TARGET, add_special_tokens=False)["input_ids"][0]
    W_U = model.get_output_embeddings().weight
    ln_f = getattr(model, "transformer", model).ln_f if hasattr(getattr(model, "transformer", model), "ln_f") else None
    lens = []
    for i in sorted(cap.acts):
        h = cap.acts[i][0, -1]
        if ln_f is not None:
            h = ln_f(h)
        lp = torch.log_softmax(W_U @ h, -1)[tgt].item()
        lens.append((i, lp))
        print(f"  layer {i:2d}: logit-lens log P(target) = {lp:.3f}")

    # --- Stage 4: intervene (add / subtract the concept direction) -----------
    print("\n=== Stage 4: intervene (steer the 'French capital' direction) ===")
    direction = build_direction(model, tok, device)
    results = {}
    for alpha in (-8.0, -4.0, 0.0, 4.0, 8.0):
        lps = []
        for p in DEFAULT_PROMPTS:
            fn = None if alpha == 0 else (
                lambda a=alpha: [add_vector_hook(model, STEER_LAYER, direction, a)])
            lps.append(target_logprob(model, tok, device, p, DEFAULT_TARGET, fn))
        m, lo, hi = bootstrap_ci(lps)
        results[alpha] = (m, lo, hi)
        print(f"  alpha {alpha:+5.1f}: log P(target) = {m:.3f} [{lo:.3f}, {hi:.3f}]")

    # --- figure --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ls, lps = zip(*lens)
    axes[0].plot(ls, lps, "o-"); axes[0].set_xlabel("layer")
    axes[0].set_ylabel("logit-lens log P(target)")
    axes[0].set_title("Stage 3: where the completion is decided")
    alphas = sorted(results)
    ms = [results[a][0] for a in alphas]
    err = [[results[a][0] - results[a][1] for a in alphas],
           [results[a][2] - results[a][0] for a in alphas]]
    axes[1].errorbar(alphas, ms, yerr=err, fmt="o-", capsize=4)
    axes[1].set_xlabel("steering coefficient alpha")
    axes[1].set_ylabel("log P(target)")
    axes[1].set_title("Stage 4: effect of the concept direction")
    savefig(fig, outdir, "case_study.png", show=args.show)

    # --- Stage 5: report -----------------------------------------------------
    effect = results[8.0][0] - results[-8.0][0]
    report = outdir / "report.md"
    with open(report, "w") as f:
        f.write("# Interpretability Case Study\n\n")
        f.write(f"**Behavior:** completion of prompts like "
                f"`{DEFAULT_PROMPTS[0]!r}` with `{DEFAULT_TARGET!r}`.\n\n")
        f.write("## Stage 1 - Quantify\n")
        f.write(f"- Baseline log P(target): **{b_mean:.3f}** "
                f"(95% CI [{b_lo:.3f}, {b_hi:.3f}]) over {len(base_lps)} "
                f"prompt x seed cells.\n\n")
        f.write("## Stage 2 - Hypotheses\n")
        for h in HYPOTHESES:
            f.write(f"- {h}\n")
        f.write("\n## Stage 3 - Probe (logit lens)\n")
        f.write(f"- Target first strongly promoted around layer "
                f"{max(lens, key=lambda t: t[1])[0]} of {ls[-1]}.\n\n")
        f.write("## Stage 4 - Intervene\n")
        f.write(f"- Steering effect (alpha +8 minus -8): **{effect:+.3f}** log-prob.\n")
        f.write(f"- Monotone in alpha: "
                f"**{all(ms[i] <= ms[i+1] for i in range(len(ms)-1))}**.\n\n")
        f.write("## Stage 5 - Verdict\n")
        verdict = ("H1 supported: a mid-layer concept direction causally moves the "
                   "completion." if effect > 0.5 else
                   "Effect weak; H2 (lexical/positional) not ruled out.")
        f.write(f"- {verdict}\n")
        f.write("- **Would overturn this:** a norm-matched random direction at the "
                "same layer producing the same effect (run as a control).\n")
        f.write("- **Unexplained:** whether the direction is France-specific or a "
                "generic 'capital city' direction (test other countries).\n")
    print(f"\n[report] {report}")
    print(f"acceptance: report written with steering effect {effect:+.3f} log-prob")


if __name__ == "__main__":
    main()
