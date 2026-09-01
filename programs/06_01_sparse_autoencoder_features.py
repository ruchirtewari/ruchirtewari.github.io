"""sparse_autoencoder_features.py — Chapter 6, Section 6.1

Trains a small TopK sparse autoencoder on GPT-2 layer-6 residual-stream
activations and browses the recovered features. Activations are cached from a
text sample (HuggingFace 'stas/openwebtext-10k' if available, else the cached
Shakespeare corpus). The SAE is 768 -> 8192 with TopK (k=32), unit-norm decoder
columns, and an MSE objective. A CLI feature browser prints the top-20
activating contexts for a latent with its trigger token marked; an optional
extension auto-labels 100 features via the modular LLM.
Acceptance: many latents show coherent themes while some are messy — both are
the lesson.
"""

import numpy as np
import torch
import torch.nn as nn

from common import (get_arg_parser, init, savefig, load_causal_lm,
                    load_shakespeare, ResidualCapture, LLM)

LAYER = 6
D_MODEL = 768
N_LATENTS = 8192
TOPK = 32


def get_texts(n_docs, quick):
    """Text chunks: openwebtext-10k if datasets available, else Shakespeare."""
    try:
        from datasets import load_dataset
        ds = load_dataset("stas/openwebtext-10k", split="train")
        texts = []
        for i in range(min(n_docs, len(ds))):
            t = ds[i]["text"].strip()
            if len(t) > 50:
                texts.append(t[:2000])
        if texts:
            print(f"[data] openwebtext-10k: {len(texts)} docs")
            return texts
    except Exception as e:
        print(f"[data] openwebtext unavailable ({e}); using Shakespeare")
    text = load_shakespeare(letters_only=False)
    words = text.split()
    chunk = 200
    texts = [" ".join(words[i:i + chunk])
             for i in range(0, len(words), chunk)]
    return texts[:n_docs]


@torch.no_grad()
def cache_activations(model, tok, device, texts, max_tokens, seq_len=128):
    """Return (acts float32 (M, 768), token_ids list, doc/pos index list).

    Keeps token ids and their originating doc + position so the browser can
    reconstruct context windows.
    """
    acts, tok_ids, meta = [], [], []
    total = 0
    for di, text in enumerate(texts):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)
        ids = enc["input_ids"].to(device)
        with ResidualCapture(model, layers=[LAYER]) as cap:
            model(input_ids=ids)
        h = cap.acts[LAYER][0].float().cpu().numpy()  # (T, 768)
        T = h.shape[0]
        acts.append(h)
        row_ids = ids[0].cpu().tolist()
        for p in range(T):
            tok_ids.append(row_ids[p])
            meta.append((di, p))
        total += T
        if total >= max_tokens:
            break
    A = np.concatenate(acts, axis=0)[:max_tokens]
    tok_ids = tok_ids[:max_tokens]
    meta = meta[:max_tokens]
    print(f"[cache] {A.shape[0]} activation vectors of dim {A.shape[1]}")
    return A.astype(np.float32), tok_ids, meta


class TopKSAE(nn.Module):
    def __init__(self, d_in=D_MODEL, n_latents=N_LATENTS, k=TOPK):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(d_in, n_latents, bias=True)
        self.dec = nn.Linear(n_latents, d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.normalize_decoder()

    def normalize_decoder(self):
        w = self.dec.weight  # (d_in, n_latents): columns are decoder atoms
        w.div_(w.norm(dim=0, keepdim=True) + 1e-8)

    def encode(self, x):
        z = self.enc(x - self.pre_bias)
        z = torch.relu(z)
        # TopK per row
        topv, topi = z.topk(self.k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, topi, topv)
        return out

    def forward(self, x):
        z = self.encode(x)
        xhat = self.dec(z) + self.pre_bias
        return xhat, z


def train_sae(A, steps, batch, lr, device, quick):
    n_latents = 1024 if quick else N_LATENTS
    k = TOPK
    sae = TopKSAE(n_latents=n_latents, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    A_t = torch.from_numpy(A).to(device)
    n = A_t.shape[0]
    losses, l0s = [], []
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        x = A_t[idx]
        xhat, z = sae(x)
        loss = ((xhat - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.normalize_decoder()
        if step % max(1, steps // 50) == 0:
            l0 = (z > 0).float().sum(-1).mean().item()
            losses.append(loss.item())
            l0s.append(l0)
            if step % max(1, steps // 10) == 0:
                print(f"step {step:5d}  mse={loss.item():.4f}  L0={l0:.1f}")
    return sae, losses, l0s


@torch.no_grad()
def all_latent_acts(sae, A, device, batch=4096):
    A_t = torch.from_numpy(A).to(device)
    outs = []
    for i in range(0, A_t.shape[0], batch):
        _, z = sae(A_t[i:i + batch])
        outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)  # (M, n_latents)


def context_string(tok, texts, meta, tok_ids, row, window=8):
    """Reconstruct a context window around one activation position."""
    di, pos = meta[row]
    # collect ids for the same doc
    start = row
    while start > 0 and meta[start - 1][0] == di and meta[start - 1][1] == meta[start][1] - 1:
        start -= 1
        if pos - meta[start][1] > window:
            break
    end = row
    while end + 1 < len(meta) and meta[end + 1][0] == di and meta[end + 1][1] == meta[end][1] + 1:
        end += 1
        if meta[end][1] - pos > window:
            break
    pieces = []
    for r in range(start, end + 1):
        t = tok.decode([tok_ids[r]])
        pieces.append(f"[[{t}]]" if r == row else t)
    return "".join(pieces)


def browse_feature(sae, Z, tok, texts, meta, tok_ids, latent, topn=20):
    col = Z[:, latent]
    order = np.argsort(-col)[:topn]
    print(f"\n=== feature {latent}: top {topn} activating contexts ===")
    for rank, row in enumerate(order):
        if col[row] <= 0:
            break
        ctx = context_string(tok, texts, meta, tok_ids, row)
        print(f"{rank + 1:2d}. act={col[row]:.2f}  {ctx}")


def auto_label(sae, Z, tok, texts, meta, tok_ids, latents):
    llm = LLM()
    if not llm.available():
        print("[label] LLM backend unavailable — skipping auto-labeling")
        return
    print(f"\n[label] auto-labeling {len(latents)} features via LLM")
    for latent in latents:
        col = Z[:, latent]
        order = np.argsort(-col)[:8]
        ctxs = [context_string(tok, texts, meta, tok_ids, r)
                for r in order if col[r] > 0]
        if not ctxs:
            continue
        prompt = ("These text snippets each mark a token in [[...]] where a "
                  "neural feature fired. In 6 words or fewer, name the common "
                  "pattern.\n\n" + "\n".join(ctxs[:8]))
        try:
            label = llm.chat(prompt, max_tokens=24, temperature=0.0).strip()
        except Exception as e:
            label = f"(llm error: {e})"
        print(f"feature {latent:5d}: {label}")


def main():
    args = get_arg_parser("TopK SAE on GPT-2 layer-6 activations").parse_args()
    outdir = init(args, "sparse_autoencoder_features")

    model, tok, device = load_causal_lm("gpt2")
    print(f"[model] gpt2 on {device}")

    if args.quick:
        n_docs, max_tokens, steps, batch = 20, 4000, 400, 256
    else:
        n_docs, max_tokens, steps, batch = 400, 200000, 4000, 1024

    texts = get_texts(n_docs, args.quick)
    A, tok_ids, meta = cache_activations(model, tok, device, texts, max_tokens)

    sae, losses, l0s = train_sae(A, steps, batch, 1e-4, device, args.quick)

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(losses))
    ax1.plot(x, losses, "-", color="C0", label="MSE")
    ax1.set_xlabel("checkpoint")
    ax1.set_ylabel("MSE", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(x, l0s, "-", color="C1", label="L0")
    ax2.set_ylabel("L0 (active latents)", color="C1")
    ax1.set_title("SAE training: reconstruction MSE and sparsity")
    fig.tight_layout()
    savefig(fig, outdir, "sae_training.png", show=args.show)

    Z = all_latent_acts(sae, A, device)

    # Pick a few live features (nonzero variance) to browse.
    activity = (Z > 0).sum(axis=0)
    live = np.where(activity > 3)[0]
    print(f"[feat] {len(live)} live features out of {Z.shape[1]}")
    show_feats = live[np.argsort(-activity[live])][:3] if len(live) else []
    for lat in show_feats:
        browse_feature(sae, Z, tok, texts, meta, tok_ids, int(lat))

    # Extension: auto-label up to 100 random live features.
    if len(live):
        rng = np.random.default_rng(args.seed)
        sample = rng.choice(live, size=min(100, len(live)), replace=False)
        auto_label(sae, Z, tok, texts, meta, tok_ids,
                   [int(x) for x in sample[:5 if args.quick else 100]])

    print(f"\n[check] trained SAE, {len(live)} live features; browse output "
          f"above shows coherent + messy latents (the lesson)")


if __name__ == "__main__":
    main()
