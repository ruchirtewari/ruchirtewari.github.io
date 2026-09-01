"""superposition_phase_transition.py — Chapter 5, Section 5.3

Reproduces the Elhage et al. toy model of superposition. Features x in R^20 are
individually active with probability (1-S); a linear map W in R^{5x20} squeezes
them into 5 dimensions and x_hat = ReLU(W^T W x + b) reconstructs them under an
importance-weighted MSE. Sweeping sparsity S in {0,.5,.9,.97,.99,.999} shows the
phase transition into superposition.
Acceptance: <=5 features represented at S=0, more than 5 at high sparsity, and
antipodal/polygonal off-diagonal structure visible in the W^T W heatmaps.
"""

import numpy as np
import torch
import torch.nn as nn

from common import get_arg_parser, init, savefig

N_FEATURES = 20
N_HIDDEN = 5


class ToyModel(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_hidden=N_HIDDEN):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        h = x @ self.W.T                    # (B, n_hidden)
        out = h @ self.W                    # (B, n_features)
        return torch.relu(out + self.b)


def sample_batch(batch, sparsity, rng, n_features=N_FEATURES):
    """Each coordinate active with prob (1-sparsity); active values ~ U[0,1]."""
    active = (rng.random((batch, n_features)) > sparsity).astype(np.float32)
    mag = rng.random((batch, n_features)).astype(np.float32)
    return torch.from_numpy(active * mag)


def train_one(sparsity, importance, steps, batch, lr, rng, device):
    model = ToyModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    imp = torch.tensor(importance, dtype=torch.float32, device=device)
    for step in range(steps):
        x = sample_batch(batch, sparsity, rng).to(device)
        xhat = model(x)
        loss = (imp * (x - xhat) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, float(loss.item())


def main():
    args = get_arg_parser("Elhage toy model of superposition").parse_args()
    outdir = init(args, "superposition_phase_transition")
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    sparsities = [0.0, 0.5, 0.9, 0.97, 0.99, 0.999]
    importance = 0.9 ** np.arange(N_FEATURES)  # geometric decay
    steps = 800 if args.quick else 5000
    batch = 512 if args.quick else 1024
    lr = 1e-2

    results = {}
    n_represented = []
    print("sparsity   loss      features_represented(||W_i||>0.5)")
    for S in sparsities:
        model, loss = train_one(S, importance, steps, batch, lr, rng, device)
        W = model.W.detach().cpu().numpy()          # (5, 20)
        norms = np.linalg.norm(W, axis=0)           # per-feature column norm
        WtW = W.T @ W                                # (20, 20)
        nrep = int((norms > 0.5).sum())
        results[S] = (W, WtW, norms)
        n_represented.append(nrep)
        print(f"{S:8.3f}   {loss:.5f}   {nrep}")

    import matplotlib.pyplot as plt
    fig1, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(sparsities)), n_represented, "o-")
    ax.axhline(N_HIDDEN, color="gray", ls=":", label=f"n_hidden={N_HIDDEN}")
    ax.set_xticks(range(len(sparsities)))
    ax.set_xticklabels([str(s) for s in sparsities])
    ax.set_xlabel("sparsity S")
    ax.set_ylabel("features represented (||W_i|| > 0.5)")
    ax.set_title("Superposition phase transition")
    ax.legend()
    fig1.tight_layout()
    savefig(fig1, outdir, "features_vs_sparsity.png", show=args.show)

    fig2, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, S in zip(axes.ravel(), sparsities):
        WtW = results[S][1]
        im = ax.imshow(WtW, cmap="RdBu", vmin=-1, vmax=1)
        ax.set_title(f"W^T W, S={S}")
        fig2.colorbar(im, ax=ax, fraction=0.046)
    fig2.suptitle("Gram matrix W^T W: off-diagonals reveal superposition")
    fig2.tight_layout()
    savefig(fig2, outdir, "WtW_heatmaps.png", show=args.show)

    nrep_low = n_represented[0]
    nrep_high = n_represented[-1]
    print(f"\n[check] S=0 -> {nrep_low} features (<= {N_HIDDEN}); "
          f"S=0.999 -> {nrep_high} features; "
          f"{'PASS' if nrep_low <= N_HIDDEN and nrep_high > N_HIDDEN else 'CHECK'}")


if __name__ == "__main__":
    main()
