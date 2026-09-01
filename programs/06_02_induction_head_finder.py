"""induction_head_finder.py — Chapter 6, Section 6.2

Finds GPT-2's induction heads and causally verifies them by ablation. Builds 100
synthetic sequences of 50 random tokens repeated twice. For every (layer, head)
it computes a prefix-matching score: mean attention from a second-half position t
back to the position right after token_t's first occurrence. Heads are ranked in
a 12x12 heatmap. The top-k heads are then zero-ablated (their attention output
patched to zero) and the loss is measured on second-half vs first-half tokens
with and without ablation.
Acceptance: a few heads (typically layers 5-7) score far above the rest, and
ablating them erases most of the second-half loss advantage.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from common import get_arg_parser, init, savefig, load_causal_lm, get_blocks


def make_sequences(tok, n_seq, half_len, rng, device):
    """n_seq sequences of `half_len` random tokens repeated twice."""
    vocab = tok.vocab_size
    specials = set(tok.all_special_ids)
    seqs = []
    for _ in range(n_seq):
        toks = []
        while len(toks) < half_len:
            t = int(rng.integers(0, vocab))
            if t not in specials:
                toks.append(t)
        seqs.append(toks + toks)  # repeat twice
    return torch.tensor(seqs, dtype=torch.long, device=device)  # (n_seq, 2*half)


@torch.no_grad()
def prefix_matching_scores(model, tok, seqs, n_layers, n_heads, half_len):
    """Return (n_layers, n_heads) mean prefix-matching attention.

    For second-half query position t (t >= half_len), the induction target is
    position (t - half_len) + 1 = first occurrence of token_t, plus one. We read
    attention[query=t, key=target].
    """
    scores = np.zeros((n_layers, n_heads))
    count = 0
    for s in range(seqs.shape[0]):
        ids = seqs[s:s + 1]  # (1, T)
        out = model(input_ids=ids, output_attentions=True)
        attns = out.attentions  # tuple of (1, n_heads, T, T)
        T = ids.shape[1]
        # query positions in second half; target = t-half_len+1
        qpos = np.arange(half_len, T)
        tgt = qpos - half_len + 1
        valid = tgt < T
        qpos, tgt = qpos[valid], tgt[valid]
        for L in range(n_layers):
            a = attns[L][0]  # (n_heads, T, T)
            # gather attn[head, q, target]
            vals = a[:, qpos, tgt]  # (n_heads, len(qpos))
            scores[L] += vals.mean(dim=1).cpu().numpy()
        count += 1
    return scores / max(count, 1)


def head_ablation_hook(head_idx, n_heads):
    """Zero one head's contribution at the attention output (c_proj input).

    GPT-2 attn output before c_proj is (B, T, n_heads*head_dim); we zero the
    slice for head_idx. We hook the .attn module's forward to rewrite the merged
    head output. Simpler: hook c_proj's input via forward_pre_hook.
    """
    def pre_hook(module, args):
        x = args[0]  # (B, T, d_model), concatenated heads
        d_model = x.shape[-1]
        head_dim = d_model // n_heads
        x = x.clone()
        x[..., head_idx * head_dim:(head_idx + 1) * head_dim] = 0.0
        return (x,) + args[1:]
    return pre_hook


@torch.no_grad()
def halves_loss(model, seqs, half_len):
    """Mean next-token CE loss on first-half vs second-half target positions."""
    import torch.nn.functional as F
    first, second = [], []
    for s in range(seqs.shape[0]):
        ids = seqs[s:s + 1]
        out = model(input_ids=ids)
        logits = out.logits[0, :-1]        # predict token t+1
        targets = ids[0, 1:]
        loss = F.cross_entropy(logits, targets, reduction="none")  # (T-1,)
        # position i predicts token i+1; second half targets: i+1 >= half_len
        idx = np.arange(loss.shape[0])
        tgt_pos = idx + 1
        first.append(loss[tgt_pos < half_len].mean().item())
        second.append(loss[tgt_pos >= half_len].mean().item())
    return float(np.mean(first)), float(np.mean(second))


def main():
    args = get_arg_parser("Find and ablate GPT-2 induction heads").parse_args()
    outdir = init(args, "06_02_induction_head_finder")
    rng = np.random.default_rng(args.seed)

    model, tok, device = load_causal_lm("gpt2", attn_eager=True)
    n_layers = len(get_blocks(model))
    n_heads = model.config.n_head
    half_len = 50
    n_seq = 20 if args.quick else 100
    print(f"[model] gpt2: {n_layers} layers x {n_heads} heads")

    seqs = make_sequences(tok, n_seq, half_len, rng, device)
    print(f"[data] {n_seq} sequences of length {2 * half_len}")

    scores = prefix_matching_scores(model, tok, seqs, n_layers, n_heads, half_len)

    # Rank heads.
    flat = [(scores[L, H], L, H) for L in range(n_layers) for H in range(n_heads)]
    flat.sort(reverse=True)
    print("\ntop prefix-matching heads:")
    for sc, L, H in flat[:10]:
        print(f"  L{L:2d}H{H:2d}  score={sc:.3f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(scores, cmap="viridis", aspect="auto")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Prefix-matching score per head")
    fig.colorbar(im, ax=ax)
    savefig(fig, outdir, "prefix_matching_heatmap.png", show=args.show)

    # Ablate top-k heads (zero attention output slice).
    topk = 3
    top_heads = [(L, H) for _, L, H in flat[:topk]]
    print(f"\n[ablate] zeroing heads {top_heads}")

    base_first, base_second = halves_loss(model, seqs, half_len)
    blocks = get_blocks(model)
    handles = []
    for (L, H) in top_heads:
        cproj = blocks[L].attn.c_proj
        handles.append(cproj.register_forward_pre_hook(
            head_ablation_hook(H, n_heads)))
    try:
        abl_first, abl_second = halves_loss(model, seqs, half_len)
    finally:
        for h in handles:
            h.remove()

    print("\ncondition     first-half loss   second-half loss   advantage")
    print(f"baseline      {base_first:14.3f}   {base_second:15.3f}   "
          f"{base_first - base_second:9.3f}")
    print(f"ablated       {abl_first:14.3f}   {abl_second:15.3f}   "
          f"{abl_first - abl_second:9.3f}")

    fig2, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, [base_first, base_second], w, label="baseline")
    ax.bar(x + w / 2, [abl_first, abl_second], w, label="top-heads ablated")
    ax.set_xticks(x)
    ax.set_xticklabels(["first half", "second half"])
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Induction-head ablation erases second-half advantage")
    ax.legend()
    savefig(fig2, outdir, "ablation_loss.png", show=args.show)

    base_adv = base_first - base_second
    abl_adv = abl_first - abl_second
    print(f"\n[check] second-half advantage {base_adv:.3f} -> {abl_adv:.3f} "
          f"after ablation ({'PASS' if abl_adv < base_adv else 'CHECK'})")


if __name__ == "__main__":
    main()
