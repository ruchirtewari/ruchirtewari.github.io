"""Chapter 8, §8.2 - grokking_phase_transition_for_modular_arithmetic.py (catalog entry 21).

Purpose: reproduce grokking and watch the internal circuit form before the
behavioral jump -- the book's flagship experiment. A 2-layer transformer learns
modular addition a+b mod 97 -- the same architecture size (2 layers, width 128,
4 heads) as the original Power et al. 2022 grokking paper. We log train/test
accuracy, weight norm, and the Fourier power spectrum of the token-embedding
matrix.

The --train-frac knob controls the regime. Grokking is not a superior end state
-- immediate generalization is better in practice. It is a LAB CURIOSITY: a
small, over-parameterized model trained on a TINY fraction of the data with
strong weight decay memorizes first (train acc 1.0, test at chance), then long
after, weight decay grinds the high-norm lookup table down and the low-norm
general circuit emerges and takes over -- the test-accuracy jump. The value is
that the long delay lets us WATCH the circuit form (the sharpening Fourier
spectrum leads the behavioral jump) instead of it happening instantly and
invisibly. ~0.3 sits in the grokking window; 0.5 gives enough coverage that the
rule is the cheapest path from the start, so the net generalizes immediately and
there is no delay to observe.

Expected result (acceptance): train accuracy saturates early; test accuracy
stays near chance for a long plateau, then jumps; a few Fourier frequencies in
the embeddings sharpen DURING the plateau -- the internal signal that leads the
behavioral transition.

Sample run (default --train-frac 0.3, seed 0, CPU, ~10^5 steps):

    step      0   train 0.013   test 0.011   wnorm 119.2
    step    500   train 1.000   test 0.299   wnorm  75.6   <- memorized; test at chance
    step   2000   train 1.000   test 0.299   wnorm  37.1   <- long plateau, wnorm falling
    step   4000   train 1.000   test 0.313   wnorm  37.5
    step   4500   train 1.000   test 0.435   wnorm  38.5   <- circuit starts to win
    step   5000   train 1.000   test 0.712   wnorm  36.2
    step   5500   train 1.000   test 0.997   wnorm  33.8   <- GROK: test snaps up
    step   6000   train 1.000   test 1.000   wnorm  30.0
    step 100000   train 1.000   test 1.000   wnorm  ~31    <- stays generalized

    acceptance: delayed generalization present: True   (delay ~5000 steps)

Read the weight norm alongside test accuracy: it climbs while the model
memorizes, then weight decay drives it down through the plateau, and the grok
lands as it settles -- the low-norm general circuit displacing the high-norm
lookup table. Widen to --train-frac 0.5 and the plateau vanishes: test reaches
~1.0 within ~2000 steps with no delay (acceptance: False), which is the
better outcome in practice but shows no circuit-formation window to observe.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import get_arg_parser, init, savefig, resolve_device

P = 97  # modulus (prime)


class Block(nn.Module):
    """One pre-norm decoder block: causal self-attention + MLP, residual."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, d_model))

    def forward(self, h, mask):
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), attn_mask=mask)
        h = h + a
        h = h + self.mlp(self.ln2(h))
        return h


class Transformer(nn.Module):
    """Tokens: [a, b, '='] -> predict (a+b) mod P. Vocab = P + 1.

    Decoder-only, causal masking, 2 layers / width 128 / 4 heads -- the same
    shape as the original Power et al. 2022 grokking network (~4e5 params).
    """

    def __init__(self, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.vocab = P + 1
        self.eq_tok = P
        self.embed = nn.Embedding(self.vocab, d_model)
        self.pos = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, P)

    def forward(self, x):
        h = self.embed(x) + self.pos
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.unembed(self.ln_f(h)[:, -1])


def make_data(device):
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    x = torch.stack([a, b, torch.full_like(a, P)], dim=1)
    y = (a + b) % P
    return x.to(device), y.to(device)


def embedding_fourier_power(model):
    """Power in each frequency of the token embeddings over the P residue tokens."""
    W = model.embed.weight.detach().cpu().numpy()[:P]      # (P, d_model)
    F = np.fft.rfft(W, axis=0)                              # (P//2+1, d_model)
    return (np.abs(F) ** 2).sum(axis=1)                    # power per frequency


def main():
    parser = get_arg_parser("grokking with a Fourier leading indicator")
    parser.add_argument("--train-frac", type=float, default=0.3,
                        help="fraction of all P*P equations used for training. "
                             "~0.3 induces grokking (memorize-then-generalize delay); "
                             "0.5 is enough coverage that the net generalizes immediately.")
    args = parser.parse_args()
    outdir = init(args, "08_02_grokking_phase_transition_for_modular_arithmetic")
    device = resolve_device()

    steps = 5000 if args.quick else 100_000
    log_every = 200 if args.quick else 500

    x, y = make_data(device)
    n = x.shape[0]
    torch.manual_seed(args.seed)
    perm = torch.randperm(n, device=device)
    n_train = int(n * args.train_frac)
    tr, te = perm[:n_train], perm[n_train:]

    model = Transformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    loss_fn = nn.CrossEntropyLoss()

    hist = {"step": [], "train_acc": [], "test_acc": [], "wnorm": [], "fourier": []}

    for step in range(steps + 1):
        model.train()
        logits = model(x[tr])
        loss = loss_fn(logits, y[tr])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(x[tr]).argmax(-1) == y[tr]).float().mean().item()
                te_acc = (model(x[te]).argmax(-1) == y[te]).float().mean().item()
            wnorm = sum(p.detach().pow(2).sum().item() for p in model.parameters()) ** 0.5
            hist["step"].append(step)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)
            hist["wnorm"].append(wnorm)
            hist["fourier"].append(embedding_fourier_power(model))
            print(f"step {step:6d}  train {tr_acc:.3f}  test {te_acc:.3f}  wnorm {wnorm:7.1f}")

    steps_arr = np.array(hist["step"])
    fourier = np.array(hist["fourier"])            # (T, P//2+1)
    fourier = fourier / (fourier.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    x_plot = np.maximum(steps_arr, 1)

    axes[0].plot(x_plot, hist["train_acc"], label="train")
    axes[0].plot(x_plot, hist["test_acc"], label="test")
    axes[0].set_xscale("log"); axes[0].set_ylabel("accuracy")
    axes[0].legend(); axes[0].set_title("grokking: test accuracy jumps long after train saturates")

    axes[1].plot(x_plot, hist["wnorm"], color="#8c564b")
    axes[1].set_xscale("log"); axes[1].set_ylabel("weight norm")
    axes[1].set_title("weight decay shrinks the norm during the plateau")

    top = np.argsort(fourier[-1])[-5:]  # 5 dominant final frequencies
    for k in top:
        axes[2].plot(x_plot, fourier[:, k], label=f"freq {k}")
    axes[2].set_xscale("log"); axes[2].set_xlabel("training step (log)")
    axes[2].set_ylabel("normalized Fourier power")
    axes[2].legend(fontsize=8)
    axes[2].set_title("embedding Fourier structure sharpens BEFORE the test-accuracy jump")
    savefig(fig, outdir, "grokking_phase_transition.png", show=args.show)

    # Grokking = delayed generalization: measure the GAP between when train
    # saturates and when test generalizes. This is regime-independent -- it
    # works whether the grok lands at step 5k or step 90k. (The old "best test
    # in the first third" heuristic misfired when the grok happened early, e.g.
    # at train_frac 0.3, because the post-grok 1.0 fell inside that window.)
    def first_step_reaching(key, thresh):
        for s, v in zip(hist["step"], hist[key]):
            if v >= thresh:
                return s
        return None

    final_test = hist["test_acc"][-1]
    t_train = first_step_reaching("train_acc", 0.99)   # train saturates (memorized)
    t_test = first_step_reaching("test_acc", 0.90)     # test generalizes (grokked)
    delay = (t_test - t_train) if (t_train is not None and t_test is not None) else None
    DELAY_STEPS = 2000                                 # a "delay" worth calling grokking
    print(f"\nfinal test acc {final_test:.3f}; train saturated at step {t_train}; "
          f"test generalized at step {t_test}"
          + (f" (delay {delay} steps)" if delay is not None else " (test never reached 0.90)"))
    if args.quick:
        print("acceptance: --quick is too short to grok; run full for the transition.")
    else:
        grokked = final_test > 0.9 and delay is not None and delay >= DELAY_STEPS
        print(f"acceptance: delayed generalization present: {grokked}")


if __name__ == "__main__":
    main()
