"""Chapter 3, §3.1 — n-gram vs tiny transformer (catalog entry 5).

Purpose: extend the ch1 entropy ladder by one learned rung; locate where the
learned state beats the counted one. Trains a 2-layer, 2-head, d_model=128
character transformer and order-2..5 add-one-smoothed n-gram models on the
same Shakespeare train split, compares bits/char on a shared held-out slice.

Expected result (acceptance): transformer beats the order-5 n-gram in
bits/char; the top disagreement positions (transformer better than best
n-gram) show visibly long-range structure (repeated names, quotes).
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import get_arg_parser, init, load_shakespeare, savefig

CTX = 64
D_MODEL = 128
LN2 = math.log(2.0)


class TinyTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=2, dim_feedforward=4 * D_MODEL,
            dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(D_MODEL, vocab)
        mask = torch.triu(torch.full((CTX, CTX), float("-inf")), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):  # x: (B, T)
        t = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(t, device=x.device))
        h = self.enc(h, mask=self.mask[:t, :t])
        return self.head(h)


def ngram_bits(train_ids, test_ids, order, vocab):
    """Per-position bits/char of an order-k n-gram (k chars of context),
    add-one smoothing, on test_ids[order:]."""
    base = vocab ** np.arange(order - 1, -1, -1) if order else np.array([], dtype=np.int64)

    def contexts(ids):
        if order == 0:
            return np.zeros(len(ids), dtype=np.int64)
        w = np.lib.stride_tricks.sliding_window_view(ids, order)[:-1]
        return w @ base  # context codes for positions order..end

    counts = {}
    ctx = contexts(train_ids)
    nxt = train_ids[order:]
    for c, n in zip(ctx, nxt):
        row = counts.setdefault(c, np.zeros(vocab, dtype=np.int64))
        row[n] += 1
    bits = np.empty(len(test_ids) - order)
    tctx, tnxt = contexts(test_ids), test_ids[order:]
    for i, (c, n) in enumerate(zip(tctx, tnxt)):
        row = counts.get(c)
        num = (1 if row is None else row[n] + 1)
        den = (vocab if row is None else row.sum() + vocab)
        bits[i] = -math.log2(num / den)
    return bits


def transformer_bits(model, test_ids, device, batch=256):
    """Bits/char at every test position >= CTX (full context each)."""
    model.eval()
    windows = np.lib.stride_tricks.sliding_window_view(test_ids, CTX + 1)
    out = np.empty(len(windows))
    with torch.no_grad():
        for s in range(0, len(windows), batch):
            w = torch.tensor(windows[s:s + batch], device=device)
            logits = model(w[:, :-1])[:, -1, :]
            lp = torch.log_softmax(logits, dim=-1)
            nll = -lp[torch.arange(len(w)), w[:, -1]]
            out[s:s + batch] = (nll / LN2).cpu().numpy()
    return out  # bits for predicting test_ids[CTX + i]


def main():
    args = get_arg_parser("n-gram vs tiny character transformer").parse_args()
    outdir = init(args, "03_01_ngram_vs_tiny_transformer")
    device = "cpu"

    text = load_shakespeare()
    if args.quick:
        text, steps = text[:200_000], 300
    else:
        text, steps = text[:2_000_000], 4000
    chars = sorted(set(text))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(ids))
    train_ids, test_ids = ids[:split], ids[split:]
    print(f"vocab={vocab} train={len(train_ids)} test={len(test_ids)} steps={steps}")

    # --- transformer training ---
    model = TinyTransformer(vocab).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"transformer params: {nparam/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bs = 64
    for step in range(steps):
        idx = np.random.randint(0, len(train_ids) - CTX - 1, size=bs)
        x = torch.tensor(np.stack([train_ids[i:i + CTX] for i in idx]), device=device)
        y = torch.tensor(np.stack([train_ids[i + 1:i + CTX + 1] for i in idx]), device=device)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            print(f"step {step:5d}  train loss {loss.item()/LN2:.3f} bits/char")

    # --- evaluation on shared positions (index >= CTX in test slice) ---
    max_eval = 20_000 if args.quick else 100_000
    ev = test_ids[:min(len(test_ids), max_eval + CTX)]
    tf_bits = transformer_bits(model, ev, device)  # positions CTX..len(ev)-1
    orders = [2, 3, 4, 5]
    ng = {k: ngram_bits(train_ids, ev, k, vocab)[CTX - k:] for k in orders}
    n = len(tf_bits)
    results = {f"{k}-gram": ng[k][:n].mean() for k in orders}
    results["transformer"] = tf_bits.mean()
    print("\nmodel        bits/char")
    for name, b in results.items():
        print(f"{name:12s} {b:8.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(results)
    ax.bar(names, [results[m] for m in names],
           color=["#888"] * len(orders) + ["#d62728"])
    ax.set_ylabel("bits/char (held-out)")
    ax.set_title("n-gram vs tiny transformer")
    savefig(fig, outdir, "bits_per_char.png", show=args.show)

    # --- top-10 disagreement positions vs best n-gram ---
    best_k = min(orders, key=lambda k: ng[k][:n].mean())
    gap = ng[best_k][:n] - tf_bits
    top = np.argsort(gap)[::-1][:10]
    print(f"\ntop-10 positions where transformer beats best n-gram ({best_k}-gram):")
    for r, i in enumerate(top, 1):
        pos = CTX + i  # index into ev
        ctx_str = "".join(chars[c] for c in ev[max(0, pos - 80):pos])
        tgt = chars[ev[pos]]
        print(f"{r:2d}. gap {gap[i]:5.2f} bits  ...{ctx_str}[{tgt}]")

    best = min(results, key=results.get)
    print(f"\nbest model: {best}; transformer beats {best_k}-gram: "
          f"{results['transformer'] < results[f'{best_k}-gram']}")


if __name__ == "__main__":
    main()
