"""concept_direction_probe_and_steer.py — Chapter 5, Section 5.2

Builds one sentiment concept direction (difference-of-means) per GPT-2 layer and
evaluates it two ways: correlationally with a held-out projection-threshold
probe, and causally by steering generation (add +alpha*v_hat at a layer) and
scoring continuations with an embedded sentiment lexicon.
Acceptance: probe accuracy climbs and stays high from the middle layers on, the
steering effect peaks mid-network, and the two peaks do NOT coincide.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from common import (get_arg_parser, init, savefig, load_causal_lm,
                    add_vector_hook)

POS_TEMPLATES = [
    "I absolutely loved the {}, it was wonderful.",
    "The {} made me so happy and grateful.",
    "What a delightful {}, truly excellent.",
    "This {} is fantastic and brilliant.",
    "I am thrilled with the {}, superb work.",
    "A joyful and heartwarming {}.",
    "The {} was amazing, I felt great.",
    "Such a beautiful {}, pure delight.",
    "I adore this {}, it is perfect.",
    "The best {} I have ever experienced.",
]
NEG_TEMPLATES = [
    "I absolutely hated the {}, it was terrible.",
    "The {} made me so sad and angry.",
    "What a dreadful {}, truly awful.",
    "This {} is horrible and disgusting.",
    "I am furious with the {}, dreadful work.",
    "A miserable and depressing {}.",
    "The {} was appalling, I felt sick.",
    "Such an ugly {}, pure misery.",
    "I loathe this {}, it is broken.",
    "The worst {} I have ever experienced.",
]
NOUNS = ["movie", "meal", "trip", "gift", "book", "concert", "day", "game",
         "painting", "song"]

NEUTRAL_PROMPTS = [
    "Yesterday the weather", "The city council announced", "In the morning she",
    "The report described", "After the meeting they", "The new device",
    "During the walk he", "The teacher explained", "On the table there",
    "The train arrived",
] * 2  # 20 prompts

POS_WORDS = set("""good great wonderful excellent happy love loved loving joy joyful
delight delightful amazing fantastic brilliant superb perfect beautiful adore
best glad grateful pleasant nice lovely bright hope hopeful warm kind gentle
success successful win winning smile laughter cheerful thrilled pleased
gorgeous marvelous splendid terrific fabulous positive enjoy enjoyed enjoyable
fun charming pretty sweet calm peaceful safe healthy strong clever wise
generous friendly comfort comforting bliss blissful radiant vibrant fresh
proud triumph elegant graceful sincere trust honest fair rich abundant
flourish thrive uplifting inspiring soothing tender fond cherish gleeful""".split())
NEG_WORDS = set("""bad terrible awful horrible hate hated sad angry furious
miserable depressing appalling sick ugly misery loathe broken worst dreadful
disgusting fear afraid pain painful cruel dark cold harsh grief mourn weep
cry failure fail losing lose lost weak stupid foolish selfish rude enemy
suffer suffering agony despair hopeless bitter poison rot decay filthy
disgust vile wretched grim bleak gloomy dread panic terror violent brutal
savage vicious nasty mean spite hostile toxic rage wrath scorn shame guilt
disaster tragic doom ruin wreck collapse rotten corrupt evil sinister
frightened anxious worried tense stressed sorrow ache wounded hurt""".split())


def build_dataset():
    pos, neg = [], []
    for t in POS_TEMPLATES:
        for n in NOUNS:
            pos.append(t.format(n))
    for t in NEG_TEMPLATES:
        for n in NOUNS:
            neg.append(t.format(n))
    return pos, neg  # 100 each; we subsample to 50/50 in main


@torch.no_grad()
def last_token_acts(model, tok, device, sentences, n_layers, batch_size=16):
    """Return dict layer -> (N, d_model) last-token residual activations."""
    from common import ResidualCapture
    layers = list(range(n_layers))
    per = {L: [] for L in layers}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=32)
        enc = {k: v.to(device) for k, v in enc.items()}
        lengths = enc["attention_mask"].sum(dim=1) - 1  # last real token idx
        with ResidualCapture(model, layers=layers) as cap:
            model(**enc)
        for L in layers:
            h = cap.acts[L]                       # (B, T, d)
            sel = h[torch.arange(h.size(0)), lengths]  # (B, d)
            per[L].append(sel.float().cpu().numpy())
    return {L: np.concatenate(v, axis=0) for L, v in per.items()}


def probe_accuracy(train_pos, train_neg, test_pos, test_neg, v):
    """Threshold the projection onto v; threshold = midpoint of class means."""
    v = v / (np.linalg.norm(v) + 1e-12)
    tp = train_pos @ v
    tn = train_neg @ v
    thresh = 0.5 * (tp.mean() + tn.mean())
    sign = 1.0 if tp.mean() >= tn.mean() else -1.0
    ep = test_pos @ v
    en = test_neg @ v
    pred_pos = sign * (ep - thresh) > 0
    pred_neg = sign * (en - thresh) <= 0
    return (pred_pos.sum() + pred_neg.sum()) / (len(ep) + len(en))


def lexicon_score(text):
    words = [w.strip(".,!?;:\"'()").lower() for w in text.split()]
    p = sum(w in POS_WORDS for w in words)
    n = sum(w in NEG_WORDS for w in words)
    return p - n


@torch.no_grad()
def gen_scores(model, tok, device, prompts, layer=None, vec=None, alpha=0.0,
               max_new=20):
    """Mean lexicon score of greedy continuations, optionally steered."""
    handle = None
    if layer is not None:
        vt = torch.tensor(vec, dtype=torch.float32, device=device)
        handle = add_vector_hook(model, layer, vt, alpha=alpha)
    scores = []
    try:
        for prompt in prompts:
            enc = tok(prompt, return_tensors="pt").to(device)
            out = model.generate(**enc, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            scores.append(lexicon_score(txt))
    finally:
        if handle is not None:
            handle.remove()
    return float(np.mean(scores))


def main():
    args = get_arg_parser(
        "GPT-2 sentiment concept direction: probe vs steer").parse_args()
    outdir = init(args, "05_02_concept_direction_probe_and_steer")
    rng = np.random.default_rng(args.seed)

    pos_all, neg_all = build_dataset()
    rng.shuffle(pos_all)
    rng.shuffle(neg_all)
    n = 50
    pos, neg = pos_all[:n], neg_all[:n]

    model, tok, device = load_causal_lm("gpt2")
    n_layers = len(model.transformer.h)
    print(f"[model] gpt2 on {device}, {n_layers} blocks")

    acts_pos = last_token_acts(model, tok, device, pos, n_layers)
    acts_neg = last_token_acts(model, tok, device, neg, n_layers)

    # 60/40 train/test split within each class
    ntr = int(n * 0.6)
    idx_p = rng.permutation(n)
    idx_n = rng.permutation(n)

    layers = list(range(n_layers))
    probe_acc, directions = [], {}
    for L in layers:
        P, N = acts_pos[L], acts_neg[L]
        ptr, pte = P[idx_p[:ntr]], P[idx_p[ntr:]]
        ntr_, nte = N[idx_n[:ntr]], N[idx_n[ntr:]]
        v = ptr.mean(0) - ntr_.mean(0)  # diff-of-means from train only
        directions[L] = v
        probe_acc.append(probe_accuracy(ptr, ntr_, pte, nte, v))

    print("\nlayer   probe_acc")
    for L in layers:
        print(f"{L:5d}   {probe_acc[L]:.3f}")

    # Steering: sweep layers x alpha; effect = steered - unsteered baseline.
    alphas = [8.0] if args.quick else [4.0, 8.0]
    steer_prompts = NEUTRAL_PROMPTS[:5] if args.quick else NEUTRAL_PROMPTS
    steer_layers = layers if not args.quick else layers[::2]
    base_score = gen_scores(model, tok, device, steer_prompts)  # cached once
    steer_eff = {}
    print(f"\n[steer] baseline lexicon score {base_score:+.2f}; "
          f"sweeping layers x alpha (this is the slow part)")
    for L in steer_layers:
        vhat = directions[L] / (np.linalg.norm(directions[L]) + 1e-12)
        effects = [gen_scores(model, tok, device, steer_prompts, L, vhat, a)
                   - base_score for a in alphas]
        best = max(effects, key=abs)  # largest-magnitude effect over alpha
        steer_eff[L] = best
        print(f"layer {L:2d}   steer_effect={best:+.2f}")

    steer_series = [steer_eff.get(L, np.nan) for L in layers]

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(layers, probe_acc, "o-", color="C0", label="probe accuracy")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("probe accuracy", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0.4, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(layers, steer_series, "s--", color="C3", label="steering effect")
    ax2.set_ylabel("steering effect (lexicon delta)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("Sentiment direction: correlational probe vs causal steer")
    fig.tight_layout()
    savefig(fig, outdir, "concept_direction_probe_and_steer.png", show=args.show)

    probe_peak = int(np.nanargmax(probe_acc))
    steer_peak = layers[int(np.nanargmax(np.abs(np.array(steer_series,
                                                         dtype=float))))]
    print(f"\n[check] probe peak layer {probe_peak}, steer peak layer "
          f"{steer_peak}; peaks {'DIFFER (expected)' if probe_peak != steer_peak else 'coincide'}")


if __name__ == "__main__":
    main()
