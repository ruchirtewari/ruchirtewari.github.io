"""Chapter 8, §8.2 - fisher_spectrum_inspector.py (catalog entry 24).

Purpose: cash Chapter 8's promissory note that internal observables (Fisher
spectra) reorganize before behavior changes. Re-run the grokking experiment
(same model, data, and regime as 08_02) and, every few hundred steps, estimate
the top-k eigenvalues of the true Fisher information matrix of the model on
its training inputs. Plot the spectrum's trajectory alongside test accuracy:
does the Fisher spectrum move before the grok?

Method (matches ch2 §2.1 'measuring it in a billion-dial machine'):
  - Fisher = GGN for softmax + log-loss: F = E[J^T G J], G = diag(p) - p p^T.
  - Fisher-vector products, never the matrix: Jv by central finite difference
    of the logits in direction v (robust; forward-mode AD has coverage gaps on
    nn.MultiheadAttention); J^T u by exact autograd VJP through one retained
    graph per probe.
  - Lanczos with full reorthogonalization on those products; eigenvalues of
    the small tridiagonal give the top of the spectrum.

How to use:
    python 08_04_fisher_spectrum_inspector.py             # full: ~12k steps, CPU ~30-60 min
    python 08_04_fisher_spectrum_inspector.py --quick     # 1200 steps, no grok, smoke only
    python 08_04_fisher_spectrum_inspector.py --steps 20000 --probe-every 250
    ./run_tests.sh 08_04 --quick

Input: all pairs a+b mod 97 (imported from the 08_02 program; train-frac 0.3,
weight decay 1.0, seed 0 - the regime that grokked at step ~5000-5500).
Output: out/08_04_fisher_spectrum_inspector/fisher_spectrum.png with three
panels: (a) train/test accuracy vs step; (b) top-k Fisher eigenvalues vs step
(log y); (c) top eigenvalue and embedding Fourier concentration overlaid.

Expected result (acceptance, FULL run): the run grokks (delayed test-acc jump)
AND the tracked top eigenvalue is finite with a dynamic range > 5x across the
run - the spectrum visibly reorganizes. The lead/lag between the spectrum's
largest movement and the behavioral jump is printed as a measurement, not
asserted: either sign is a finding. --quick only verifies the estimator
(finite, positive eigenvalues on a briefly trained model).
"""

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn

from common import get_arg_parser, init, savefig, resolve_device

# ---- reuse the 08_02 model/data/fourier code (same experiment, new probe) ---
_spec = importlib.util.spec_from_file_location(
    "grok_mod", Path(__file__).parent /
    "08_02_grokking_phase_transition_for_modular_arithmetic.py")
_grok = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grok)
Transformer, make_data, P = _grok.Transformer, _grok.make_data, _grok.P
embedding_fourier_power = _grok.embedding_fourier_power


# --------------------------------------------------------------- Fisher ops --
def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflat(vec, like):
    out, i = [], 0
    for t in like:
        n = t.numel()
        out.append(vec[i:i + n].view_as(t))
        i += n
    return out


class FisherOperator:
    """Fv products for the true Fisher (=GGN) on a fixed input batch.

    One retained autograd graph per instance serves every VJP; each product
    costs two extra forwards (finite-difference Jv) and one backward.
    """

    def __init__(self, model, x, fd_eps=1e-3):
        self.model, self.x, self.eps = model, x, fd_eps
        self.names = [n for n, _ in model.named_parameters()]
        self.params = [p for _, p in model.named_parameters()]
        self.base = {n: p.detach() for n, p in model.named_parameters()}
        with torch.no_grad():
            self.probs = torch.softmax(model(x), dim=-1)          # (B, P)
        self.graph_logits = model(x)                              # retained graph

    @torch.no_grad()
    def _jv(self, vflat):
        # central difference via functional_call: live parameters never mutate,
        # so the retained VJP graph stays valid.
        from torch.func import functional_call
        eps = self.eps / (vflat.norm() + 1e-12)
        v = _unflat(vflat, self.params)
        plus = {n: self.base[n] + eps * d for n, d in zip(self.names, v)}
        minus = {n: self.base[n] - eps * d for n, d in zip(self.names, v)}
        lp = functional_call(self.model, plus, (self.x,))
        lm = functional_call(self.model, minus, (self.x,))
        return (lp - lm) / (2 * eps)                              # (B, P)

    def matvec(self, vflat):
        Jv = self._jv(vflat)
        p = self.probs
        Gu = (p * Jv - p * (p * Jv).sum(-1, keepdim=True)) / self.x.shape[0]
        grads = torch.autograd.grad(self.graph_logits, self.params,
                                    grad_outputs=Gu, retain_graph=True)
        return _flat([g.detach() for g in grads])


def lanczos_top_eigs(op, dim, m=16, k=8, seed=0, device="cpu"):
    """Top-k eigenvalue estimates of a symmetric PSD operator via m Lanczos
    iterations with full reorthogonalization."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g).to(device)
    q /= q.norm()
    Q, alphas, betas = [q], [], []
    for j in range(m):
        w = op.matvec(Q[-1])
        a = torch.dot(w, Q[-1]).item()
        alphas.append(a)
        w = w - a * Q[-1] - (betas[-1] * Q[-2] if j > 0 else 0)
        for qi in Q:                                   # full reorthogonalization
            w = w - torch.dot(w, qi) * qi
        b = w.norm().item()
        if b < 1e-10:
            break
        betas.append(b)
        Q.append(w / b)
    T = np.diag(alphas)
    for i, b in enumerate(betas[:len(alphas) - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    eigs = np.linalg.eigvalsh(T)[::-1]
    return eigs[:k]


# ------------------------------------------------------------------- main ----
def main():
    parser = get_arg_parser("Fisher spectrum tracked through a grokking run")
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=None,
                        help="training steps (default 12000 full / 1200 quick)")
    parser.add_argument("--probe-every", type=int, default=None,
                        help="steps between Fisher probes (default 400 / 400)")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    outdir = init(args, "08_04_fisher_spectrum_inspector")
    device = resolve_device()

    steps = args.steps or (1200 if args.quick else 12000)
    probe_every = args.probe_every or 400
    lanczos_m = 8 if args.quick else 16
    subset = 128 if args.quick else 512

    x, y = make_data(device)
    n = x.shape[0]
    torch.manual_seed(args.seed)
    perm = torch.randperm(n, device=device)
    n_train = int(n * args.train_frac)
    tr, te = perm[:n_train], perm[n_train:]
    probe_x = x[tr[:subset]]

    model = Transformer().to(device)
    dim = sum(p.numel() for p in model.parameters())
    print(f"model dim {dim}; probing top-{args.topk} Fisher eigenvalues "
          f"every {probe_every} steps (Lanczos m={lanczos_m}, batch {subset})")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0,
                            betas=(0.9, 0.98))
    loss_fn = torch.nn.CrossEntropyLoss()

    hist = {"step": [], "train_acc": [], "test_acc": []}
    probes = {"step": [], "eigs": [], "fourier_share": []}

    for step in range(steps + 1):
        model.train()
        loss = loss_fn(model(x[tr]), y[tr])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % probe_every == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = (model(x[tr]).argmax(-1) == y[tr]).float().mean().item()
                te_acc = (model(x[te]).argmax(-1) == y[te]).float().mean().item()
            hist["step"].append(step)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)

            op = FisherOperator(model, probe_x)
            eigs = lanczos_top_eigs(op, dim, m=lanczos_m, k=args.topk,
                                    seed=args.seed + step, device=device)
            del op
            fp = embedding_fourier_power(model)
            share = float(fp.max() / (fp.sum() + 1e-12))
            probes["step"].append(step)
            probes["eigs"].append(eigs)
            probes["fourier_share"].append(share)
            print(f"step {step:6d}  train {tr_acc:.3f}  test {te_acc:.3f}  "
                  f"top-eig {eigs[0]:9.3e}  fourier-share {share:.3f}")

    eigs = np.array(probes["eigs"])                      # (probes, k)
    psteps = np.array(probes["step"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(hist["step"], hist["train_acc"], label="train")
    axes[0].plot(hist["step"], hist["test_acc"], label="test")
    axes[0].set_ylabel("accuracy"); axes[0].legend()
    axes[0].set_title("behavior vs the Fisher spectrum, same run")
    for j in range(eigs.shape[1]):
        axes[1].plot(psteps, eigs[:, j], lw=1.2 if j == 0 else 0.7,
                     color="navy", alpha=1.0 if j == 0 else 0.45)
    axes[1].set_yscale("log"); axes[1].set_ylabel("top Fisher eigenvalues")
    ax2 = axes[2].twinx()
    axes[2].plot(psteps, eigs[:, 0], color="navy", label="top Fisher eig")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("top Fisher eigenvalue", color="navy")
    ax2.plot(psteps, probes["fourier_share"], color="darkorange",
             label="embedding Fourier share")
    ax2.set_ylabel("max-frequency share", color="darkorange")
    axes[2].set_xlabel("training step")
    fig.tight_layout()
    savefig(fig, outdir, "fisher_spectrum.png", show=args.show)

    top = eigs[:, 0]
    finite = bool(np.isfinite(eigs).all() and (top > 0).all())
    dyn = float(top.max() / max(top.min(), 1e-30))
    print(f"\ntop-eigenvalue dynamic range across run: {dyn:.1f}x; finite/positive: {finite}")

    if args.quick:
        print("acceptance: estimator sound (finite positive spectrum): "
              f"{finite}  (--quick is too short to grok; run full for the lead/lag result)")
        return 0 if finite else 1

    # grok step (test >= 0.9) and largest spectral movement step
    t_test = next((s for s, a in zip(hist["step"], hist["test_acc"]) if a >= 0.9), None)
    rel = np.abs(np.diff(np.log(top + 1e-30)))
    t_spec = int(psteps[1 + int(np.argmax(rel))]) if len(rel) else None
    if t_test is not None and t_spec is not None:
        lead = t_test - t_spec
        print(f"grok (test>=0.9) at step {t_test}; largest top-eig movement at "
              f"step {t_spec}; spectrum leads behavior by {lead} steps"
              if lead >= 0 else
              f"grok at step {t_test}; largest top-eig movement at step {t_spec}; "
              f"spectrum LAGS behavior by {-lead} steps")
    else:
        print(f"grok step: {t_test}; run longer (--steps) if None.")
    ok = finite and dyn > 5.0 and t_test is not None
    print(f"acceptance: Fisher spectrum tracked through a grokking run: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
