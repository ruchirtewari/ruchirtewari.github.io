"""intrinsic_dimension_profile.py — Chapter 5, Section 5.1

Measures the "hunchback" of intrinsic dimension (ID) per layer of GPT-2 small.
Runs ~2000 English sentences through gpt2, mean-pools the residual stream to one
vector per sentence per layer, and estimates ID with the TwoNN estimator
(mu_i = r2/r1, d_hat = N / sum(log mu_i)). Plots ID vs layer for natural and
word-shuffled text, plus an ID-vs-sample-count saturation curve at layer 6.
Acceptance: a rise-then-fall profile with ID far below the ambient 768.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from common import (get_arg_parser, init, savefig, load_causal_lm,
                    load_shakespeare, ResidualCapture)


def make_sentences(n):
    """Split cached Shakespeare into sentence-like chunks of words."""
    # load_shakespeare(letters_only=False) keeps punctuation for splitting.
    text = load_shakespeare(letters_only=False)
    # Split on sentence-ending punctuation.
    import re
    parts = re.split(r"[.!?]+", text)
    sents = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        w = p.split()
        if 5 <= len(w) <= 40:
            sents.append(" ".join(w))
        if len(sents) >= n:
            break
    return sents[:n]


def shuffle_words(sentences, rng):
    out = []
    for s in sentences:
        w = s.split()
        rng.shuffle(w)
        out.append(" ".join(w))
    return out


@torch.no_grad()
def embed_sentences(model, tok, device, sentences, layers, batch_size=16):
    """Return dict profile-index -> (N, d_model) mean-pooled residual vectors.

    Profile index 0 is the embedding output; block L maps to index L+1.
    ResidualCapture hooks block outputs, so we also grab the embedding layer
    from output_hidden_states for a full profile.
    """
    per_layer = {L: [] for L in [0] + [L + 1 for L in layers]}
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=64)
        enc = {k: v.to(device) for k, v in enc.items()}
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)              # (B, 1)
        with ResidualCapture(model, layers=layers) as cap:
            out = model(**enc, output_hidden_states=True)
        # embedding layer (index 0 of hidden_states)
        emb = out.hidden_states[0]
        pooled = (emb * mask).sum(dim=1) / denom
        per_layer[0].append(pooled.float().cpu().numpy())
        # block outputs L=0..n_layers-1 map to profile index L+1
        for L in layers:
            h = cap.acts[L]
            pooled = (h * mask).sum(dim=1) / denom
            per_layer[L + 1].append(pooled.float().cpu().numpy())
    return {L: np.concatenate(v, axis=0) for L, v in per_layer.items()}


def twonn(X, discard_fraction=0.1):
    """TwoNN intrinsic dimension estimate.

    For each point, mu = (2nd NN dist) / (1st NN dist). The cdf of mu follows
    F(mu) = 1 - mu^-d, so d_hat = N / sum(log mu), robustified by discarding the
    largest-mu tail before the MLE.
    """
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    # pairwise distances
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    d2 = np.maximum(d2, 0.0)
    dist = np.sqrt(d2)
    dist.sort(axis=1)
    r1 = dist[:, 0]
    r2 = dist[:, 1]
    good = r1 > 1e-12
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0 + 1e-12]
    mu = np.sort(mu)
    keep = int(len(mu) * (1.0 - discard_fraction))
    mu = mu[:keep]
    if len(mu) < 5:
        return float("nan")
    d_hat = len(mu) / np.sum(np.log(mu))
    return float(d_hat)


def main():
    args = get_arg_parser("GPT-2 intrinsic-dimension-per-layer profile").parse_args()
    outdir = init(args, "05_01_intrinsic_dimension_profile")
    rng = np.random.default_rng(args.seed)

    n_sent = 200 if args.quick else 2000
    print(f"[data] building {n_sent} sentences")
    sentences = make_sentences(n_sent)
    print(f"[data] got {len(sentences)} sentences")
    shuffled = shuffle_words(sentences, rng)

    model, tok, device = load_causal_lm("gpt2")
    n_layers = len(model.transformer.h)  # 12
    block_layers = [0, 3, 6, 9, 11] if args.quick else list(range(n_layers))
    print(f"[model] gpt2 on {device}, capturing blocks {block_layers}")

    print("[embed] natural text")
    emb_nat = embed_sentences(model, tok, device, sentences, block_layers)
    print("[embed] shuffled text")
    emb_shuf = embed_sentences(model, tok, device, shuffled, block_layers)

    profile_layers = [0] + [L + 1 for L in block_layers]  # 0 = embeddings
    id_nat = [twonn(emb_nat[L]) for L in profile_layers]
    id_shuf = [twonn(emb_shuf[L]) for L in profile_layers]

    print("\nlayer   ID(natural)   ID(shuffled)")
    for i, L in enumerate(profile_layers):
        print(f"{L:5d}   {id_nat[i]:11.2f}   {id_shuf[i]:11.2f}")

    # Extension: ID at layer 6 (block output -> profile index 7) vs N.
    ext_layer = 7
    counts = [100, 200, 500] if args.quick else [100, 200, 500, 1000, 1500, 2000]
    counts = [c for c in counts if c <= len(sentences)]
    id_vs_n = []
    Xfull = emb_nat[ext_layer]
    for c in counts:
        idx = rng.choice(Xfull.shape[0], size=c, replace=False)
        id_vs_n.append(twonn(Xfull[idx]))
    print("\nsample-count   ID(layer6)")
    for c, v in zip(counts, id_vs_n):
        print(f"{c:11d}   {v:9.2f}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(profile_layers, id_nat, "o-", label="natural")
    ax[0].plot(profile_layers, id_shuf, "s--", label="word-shuffled")
    ax[0].axhline(768, color="gray", ls=":", label="ambient d=768")
    ax[0].set_xlabel("layer (0 = embeddings)")
    ax[0].set_ylabel("intrinsic dimension (TwoNN)")
    ax[0].set_title("ID hunchback across GPT-2 layers")
    ax[0].legend()

    ax[1].plot(counts, id_vs_n, "o-")
    ax[1].set_xlabel("sample count N")
    ax[1].set_ylabel("ID at layer 6")
    ax[1].set_title("TwoNN saturation vs N")
    fig.tight_layout()
    savefig(fig, outdir, "intrinsic_dimension_profile.png", show=args.show)

    peak = profile_layers[int(np.nanargmax(np.array(id_nat)))]
    print(f"\n[check] peak ID at layer {peak}; "
          f"max ID {np.nanmax(id_nat):.1f} << 768 -> "
          f"{'PASS' if np.nanmax(id_nat) < 768 else 'FAIL'}")


if __name__ == "__main__":
    main()
