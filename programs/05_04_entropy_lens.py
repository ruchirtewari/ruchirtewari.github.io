"""Chapter 5, §5.6 - entropy_lens.py (catalog entry 23).

Purpose: apply Chapter 1's entropy to the *inside* of the machine. Read the
residual stream after every layer through the model's own unembedding (the
logit lens), turn each intermediate state into a next-token distribution, and
measure its Shannon entropy in bits. The result is the entropy of the model's
prediction as a function of depth: Chapter 1's y-axis, plotted layer by layer
inside a single forward pass.

How to use:
    python 05_04_entropy_lens.py            # full: 8 prompts
    python 05_04_entropy_lens.py --quick    # 4 prompts, shorter
    ./run_tests.sh 05_04 --quick            # via the runner (sets up venv)

Input: built-in factual/prose prompts; GPT-2 small (config hf_base_model).
Output: out/05_04_entropy_lens/entropy_lens.png with two panels:
  (a) mean next-token entropy (bits) vs layer, with per-prompt spread - the
      collapse of uncertainty across depth;
  (b) a layer x position entropy heatmap for one prompt - where in the
      sentence the collapse happens early vs late.

Expected result (acceptance): the naive expectation - entropy falls
monotonically with depth - is FALSE, and seeing why is the point: early
logit-lens readouts are often confidently WRONG (low entropy about garbage,
the lens bias of ch5 §5.6), so entropy alone cannot distinguish
confident-right from confident-wrong. The paired measurement that is
monotone-ish is KL(final || layer-l readout): the divergence of each layer's
readout from the model's actual final prediction, which collapses with depth
as the prediction forms. Acceptance: late-layer KL-to-final is well below
early-layer KL-to-final. The entropy curve is plotted alongside as the
cautionary exhibit.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import get_arg_parser, init, savefig, load_causal_lm, ResidualCapture

PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The first president of the United States was George",
    "Water is made of hydrogen and",
    "On Monday she flew from New York to",
    "The opposite of hot is",
    "In 1969 humans first landed on the",
    "The chemical symbol for gold is",
]


def entropy_bits(logp):
    """Shannon entropy (bits) from log-probs along the last dim."""
    return (-(logp.exp() * logp).sum(-1) / np.log(2.0)).cpu().numpy()


def main():
    args = get_arg_parser("entropy and KL-to-final of the prediction, layer by layer").parse_args()
    outdir = init(args, "05_04_entropy_lens")

    model, tok, device = load_causal_lm()
    prompts = PROMPTS[:4] if args.quick else PROMPTS
    ln_f, head = model.transformer.ln_f, model.lm_head
    n_layers = len(model.transformer.h)

    ent_curves, kl_curves = [], []   # (n_prompts, n_layers) at last position
    heat = None; heat_tokens = None  # KL heatmap (layers x positions), prompt 0
    for pi, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad(), ResidualCapture(model) as cap:
            model(**enc)
            final_logp = torch.log_softmax(
                head(ln_f(cap.acts[n_layers - 1]))[0].float(), dim=-1)
            ents, kls = [], []
            for l in range(n_layers):
                logp = torch.log_softmax(head(ln_f(cap.acts[l]))[0].float(), dim=-1)
                ents.append(entropy_bits(logp))                       # (seq,)
                kl = (final_logp.exp() * (final_logp - logp)).sum(-1) / np.log(2.0)
                kls.append(kl.cpu().numpy())                          # (seq,)
        ents, kls = np.stack(ents), np.stack(kls)                     # (layers, seq)
        ent_curves.append(ents[:, -1]); kl_curves.append(kls[:, -1])
        if pi == 0:
            heat = kls
            heat_tokens = [tok.decode([t]) for t in enc["input_ids"][0]]
        print(f"  {prompt!r:55s} KL-to-final layer0 {kls[0,-1]:6.2f} -> "
              f"layer{n_layers-2} {kls[-2,-1]:6.2f} bits; "
              f"entropy layer0 {ents[0,-1]:4.2f} vs final {ents[-1,-1]:4.2f}")
    ent_curves = np.stack(ent_curves)                                 # (prompts, layers)
    kl_curves = np.stack(kl_curves)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(n_layers)
    axes[0].plot(x, kl_curves.mean(0), marker="o", color="navy")
    axes[0].fill_between(x, kl_curves.min(0), kl_curves.max(0), alpha=0.2, color="navy")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("KL(final || layer) (bits)")
    axes[0].set_title("the prediction forms: readouts converge to final")

    axes[1].plot(x, ent_curves.mean(0), marker="o", color="darkred")
    axes[1].fill_between(x, ent_curves.min(0), ent_curves.max(0), alpha=0.15, color="darkred")
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("next-token entropy (bits)")
    axes[1].set_title("entropy is NOT monotone: early layers\ncan be confidently wrong (lens bias)")
    axes[1].axhline(np.log2(head.out_features), ls=":", c="gray", lw=1)

    im = axes[2].imshow(heat, aspect="auto", cmap="viridis")
    axes[2].set_xlabel("position"); axes[2].set_ylabel("layer")
    axes[2].set_xticks(range(len(heat_tokens)))
    axes[2].set_xticklabels(heat_tokens, rotation=60, fontsize=7)
    axes[2].set_title(f"KL-to-final per (layer, position)\n{prompts[0]!r}")
    fig.colorbar(im, ax=axes[2], label="bits")
    fig.tight_layout()
    savefig(fig, outdir, "entropy_lens.png", show=args.show)

    third = max(1, n_layers // 3)
    kl_early = kl_curves[:, :third].mean()
    kl_late = kl_curves[:, -third - 1:-1].mean()      # exclude final (KL=0 by construction)
    ent0, entf = ent_curves[:, 0].mean(), ent_curves[:, -1].mean()
    print(f"\nmean KL-to-final: first {third} layers = {kl_early:.2f} bits, "
          f"last {third} (pre-final) = {kl_late:.2f} bits")
    print(f"mean entropy: layer 0 = {ent0:.2f} bits, final = {entf:.2f} bits "
          f"- note non-monotonicity: entropy alone cannot separate "
          f"confident-right from confident-wrong (ch5 §5.6 lens bias).")
    ok = kl_late < 0.5 * kl_early
    print(f"acceptance: layer readouts converge to the final prediction "
          f"(late KL < half of early KL): {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
