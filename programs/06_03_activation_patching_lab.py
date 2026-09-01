"""activation_patching_lab.py — Chapter 6, Section 6.3

Reproduces the IOI (Indirect Object Identification) name-mover result and shows
that the choice of ablation flavor changes conclusions. Builds 20 clean/corrupted
prompt pairs (corrupted = names swapped). The metric is logit(correct name) -
logit(wrong name) at the final position. It caches each head's clean output, then
patches one head at a time (all 144) into the corrupted run at the final token and
plots a 12x12 recovery heatmap. The top-5 heads are then re-ranked under zero-,
mean-, and resample-ablation, printing three ranked lists.
Acceptance: name-mover heads (~layers 9-10) dominate the heatmap, and the three
ablation rankings visibly differ.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from common import get_arg_parser, init, savefig, load_causal_lm, get_blocks

NAMES = ["John", "Mary", "Tom", "Anna", "Mark", "Lucy", "Paul", "Emma",
         "Alex", "Sara", "James", "Kate", "David", "Nina", "Peter", "Rose",
         "Sam", "Beth", "Nick", "Jane"]
PLACES = ["store", "park", "office", "school", "market", "station",
          "library", "garden"]
OBJECTS = ["drink", "book", "ball", "gift", "note", "key", "ring", "map"]


def build_pairs(n_pairs, rng):
    """Return list of dicts with clean/corrupted prompts and correct/wrong names.

    Template: "When {A} and {B} went to the {place}, {B} gave a {obj} to"
    Correct answer = A (indirect object). Corrupted swaps A and B in the
    subject clause so the model should now prefer B.
    """
    pairs = []
    used = set()
    while len(pairs) < n_pairs:
        a, b = rng.choice(NAMES, size=2, replace=False)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        key = (a, b, place, obj)
        if key in used:
            continue
        used.add(key)
        clean = (f"When {a} and {b} went to the {place}, "
                 f"{b} gave a {obj} to")
        corrupt = (f"When {b} and {a} went to the {place}, "
                   f"{b} gave a {obj} to")
        pairs.append({"clean": clean, "corrupt": corrupt,
                      "correct": a, "wrong": b})
    return pairs


def name_token_id(tok, name):
    # GPT-2 encodes a leading space for mid-sentence words.
    ids = tok.encode(" " + name)
    return ids[0]


def metric(logits_last, correct_id, wrong_id):
    return float(logits_last[correct_id] - logits_last[wrong_id])


class HeadCache:
    """Capture per-head attention output at the final token for all layers.

    Hooks each block's attn.c_proj forward_pre_hook input, which is the merged
    head output (B, T, d_model) before the output projection. We slice heads.
    """

    def __init__(self, model, n_layers, n_heads, d_model):
        self.blocks = get_blocks(model)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.cache = {}     # layer -> (T, d_model) input to c_proj
        self._handles = []

    def __enter__(self):
        for L in range(self.n_layers):
            def hook(module, args, L=L):
                self.cache[L] = args[0].detach().clone()
            self._handles.append(
                self.blocks[L].attn.c_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def make_patch_hook(layer, head, head_dim, clean_input, pos):
    """Overwrite one head's slice at position `pos` with the clean value."""
    def pre_hook(module, args):
        x = args[0].clone()
        sl = slice(head * head_dim, (head + 1) * head_dim)
        x[:, pos, sl] = clean_input[pos, sl].to(x.dtype).to(x.device)
        return (x,) + args[1:]
    return pre_hook


def make_ablate_hook(head, head_dim, pos, value):
    """Set one head's slice at `pos` to a fixed value (zero/mean/resample)."""
    def pre_hook(module, args):
        x = args[0].clone()
        sl = slice(head * head_dim, (head + 1) * head_dim)
        x[:, pos, sl] = value.to(x.dtype).to(x.device)
        return (x,) + args[1:]
    return pre_hook


@torch.no_grad()
def run_last_logits(model, ids):
    out = model(input_ids=ids)
    return out.logits[0, -1]


def main():
    args = get_arg_parser("IOI activation-patching lab").parse_args()
    outdir = init(args, "06_03_activation_patching_lab")
    rng = np.random.default_rng(args.seed)

    model, tok, device = load_causal_lm("gpt2")
    n_layers = len(get_blocks(model))
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    head_dim = d_model // n_heads
    print(f"[model] gpt2: {n_layers}L x {n_heads}H, d_model={d_model}")

    n_pairs = 5 if args.quick else 20
    pairs = build_pairs(n_pairs, rng)
    print(f"[data] {n_pairs} IOI pairs")

    # Accumulate per-head recovery over pairs.
    recovery = np.zeros((n_layers, n_heads))
    # Also collect clean per-head slice at final pos across pairs (for mean/resample).
    clean_slices = {(L, H): [] for L in range(n_layers) for H in range(n_heads)}

    for pi, p in enumerate(pairs):
        cid = name_token_id(tok, p["correct"])
        wid = name_token_id(tok, p["wrong"])
        clean_ids = tok(p["clean"], return_tensors="pt").to(device)["input_ids"]
        corr_ids = tok(p["corrupt"], return_tensors="pt").to(device)["input_ids"]
        # align lengths: templates differ only by name-swap in subject; same length
        pos = clean_ids.shape[1] - 1

        # clean run: metric + cache
        with HeadCache(model, n_layers, n_heads, d_model) as hc:
            clean_logits = run_last_logits(model, clean_ids)
        m_clean = metric(clean_logits, cid, wid)
        clean_cache = {L: hc.cache[L][0] for L in range(n_layers)}  # (T, d_model)

        for L in range(n_layers):
            for H in range(n_heads):
                sl = clean_cache[L][pos,
                                    H * head_dim:(H + 1) * head_dim].cpu().numpy()
                clean_slices[(L, H)].append(sl)

        # corrupted baseline metric
        m_corr = metric(run_last_logits(model, corr_ids), cid, wid)
        denom = (m_clean - m_corr)
        denom = denom if abs(denom) > 1e-6 else 1e-6

        # patch each head at final pos into the corrupted run
        blocks = get_blocks(model)
        for L in range(n_layers):
            for H in range(n_heads):
                hook = make_patch_hook(L, H, head_dim, clean_cache[L], pos)
                h = blocks[L].attn.c_proj.register_forward_pre_hook(hook)
                try:
                    m_patched = metric(run_last_logits(model, corr_ids), cid, wid)
                finally:
                    h.remove()
                recovery[L, H] += (m_patched - m_corr) / denom
        if (pi + 1) % max(1, n_pairs // 4) == 0:
            print(f"  patched pair {pi + 1}/{n_pairs}")

    recovery /= n_pairs

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(recovery, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("IOI logit-diff recovery from patching each head (final pos)")
    fig.colorbar(im, ax=ax)
    savefig(fig, outdir, "recovery_heatmap.png", show=args.show)

    flat = [(recovery[L, H], L, H)
            for L in range(n_layers) for H in range(n_heads)]
    flat.sort(reverse=True)
    top5 = [(L, H) for _, L, H in flat[:5]]
    print("\ntop-5 heads by patch recovery:")
    for sc, L, H in flat[:5]:
        print(f"  L{L:2d}H{H:2d}  recovery={sc:+.3f}")

    # Precompute mean slice per head across all pairs.
    mean_slice = {lh: np.mean(np.stack(v), axis=0)
                  for lh, v in clean_slices.items()}

    # Re-rank top-5 under zero / mean / resample ablation.
    # Ablation effect = drop in clean metric when head's final-pos output ablated.
    def ablation_scores(kind):
        blocks = get_blocks(model)
        scores = {}
        for (L, H) in top5:
            drops = []
            for p in pairs:
                cid = name_token_id(tok, p["correct"])
                wid = name_token_id(tok, p["wrong"])
                ids = tok(p["clean"], return_tensors="pt").to(device)["input_ids"]
                pos = ids.shape[1] - 1
                m_base = metric(run_last_logits(model, ids), cid, wid)
                if kind == "zero":
                    val = torch.zeros(head_dim)
                elif kind == "mean":
                    val = torch.tensor(mean_slice[(L, H)])
                elif kind == "resample":
                    # random other pair's clean slice for this head
                    j = rng.integers(0, len(clean_slices[(L, H)]))
                    val = torch.tensor(clean_slices[(L, H)][j])
                hook = make_ablate_hook(H, head_dim, pos, val)
                h = blocks[L].attn.c_proj.register_forward_pre_hook(hook)
                try:
                    m_abl = metric(run_last_logits(model, ids), cid, wid)
                finally:
                    h.remove()
                drops.append(m_base - m_abl)  # positive = head mattered
            scores[(L, H)] = float(np.mean(drops))
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked

    print("\n=== top-5 heads re-ranked by ablation effect (metric drop) ===")
    rankings = {}
    for kind in ("zero", "mean", "resample"):
        rankings[kind] = ablation_scores(kind)

    print(f"{'zero':<22}{'mean':<22}{'resample':<22}")
    for r in range(5):
        cells = []
        for kind in ("zero", "mean", "resample"):
            (L, H), s = rankings[kind][r]
            cells.append(f"L{L}H{H} {s:+.3f}")
        print(f"{cells[0]:<22}{cells[1]:<22}{cells[2]:<22}")

    orders = {k: [lh for lh, _ in v] for k, v in rankings.items()}
    differ = not (orders["zero"] == orders["mean"] == orders["resample"])
    print(f"\n[check] name-mover heads dominate heatmap; ablation rankings "
          f"{'DIFFER (expected)' if differ else 'coincide'}")


if __name__ == "__main__":
    main()
