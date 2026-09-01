"""Chapter 3, §3.2 — normalizing flow warps 2D Gaussian to two moons (entry 6).

Purpose: watch global distribution reshaping — a Gaussian warped into two
moons through 4 invertible RealNVP affine coupling layers trained by exact
NLL via change of variables.

Expected result (acceptance): samples land on the two moons; learned
log-density high on the moons and low elsewhere; grid deformation smooth
and invertible (no folds).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import get_arg_parser, init, savefig


def two_moons(n, noise=0.08):
    t = np.random.rand(n) * np.pi
    top = n // 2
    x = np.empty((n, 2))
    x[:top] = np.stack([np.cos(t[:top]), np.sin(t[:top])], 1)
    x[top:] = np.stack([1 - np.cos(t[top:]), 0.5 - np.sin(t[top:])], 1)
    x += noise * np.random.randn(n, 2)
    x[:, 0] -= 0.5
    x[:, 1] -= 0.25
    return torch.tensor(x, dtype=torch.float32)


class Coupling(nn.Module):
    """Affine coupling: keep masked coord, transform the other."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 4))  # 2 scale + 2 shift

    def _st(self, x_masked):
        h = self.net(x_masked)
        s, t = h[:, :2], h[:, 2:]
        s = 2.0 * torch.tanh(s)  # bounded scale
        return s * (1 - self.mask), t * (1 - self.mask)

    def forward(self, z):  # base -> data
        s, t = self._st(z * self.mask)
        x = z * self.mask + (1 - self.mask) * (z * torch.exp(s) + t)
        return x, s.sum(1)

    def inverse(self, x):  # data -> base
        s, t = self._st(x * self.mask)
        z = x * self.mask + (1 - self.mask) * ((x - t) * torch.exp(-s))
        return z, -s.sum(1)


class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [Coupling([1.0, 0.0]), Coupling([0.0, 1.0])] * 2)

    def log_prob(self, x):
        logdet = torch.zeros(len(x))
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            logdet = logdet + ld
        base = -0.5 * (x ** 2).sum(1) - np.log(2 * np.pi)
        return base + logdet

    def push(self, z, upto):
        """Push base points through layers 0..upto-1."""
        with torch.no_grad():
            for layer in self.layers[:upto]:
                z, _ = layer(z)
        return z


def main():
    args = get_arg_parser("RealNVP: Gaussian warped to two moons").parse_args()
    outdir = init(args, "03_02_flow_warp_2d")
    steps = 400 if args.quick else 4000

    data = two_moons(5000)
    model = RealNVP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(steps):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -model.log_prob(batch).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            print(f"step {step:5d}  NLL {loss.item():.3f}")

    final_nll = -model.log_prob(data).mean().item()
    print(f"\nfinal NLL on all data: {final_nll:.3f} nats")

    # figure: base + after each layer (samples + deformed grid), then density
    z = torch.randn(1000, 2)
    gl = np.linspace(-2.5, 2.5, 21)
    lines = []  # each line = (n,2) points
    for g in gl:
        lines.append(np.stack([np.full(120, g), np.linspace(-2.5, 2.5, 120)], 1))
        lines.append(np.stack([np.linspace(-2.5, 2.5, 120), np.full(120, g)], 1))
    grid = torch.tensor(np.concatenate(lines), dtype=torch.float32)

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    for k in range(5):
        ax = axes[k]
        gz = model.push(grid.clone(), k).numpy()
        for i in range(len(lines)):
            seg = gz[i * 120:(i + 1) * 120]
            ax.plot(seg[:, 0], seg[:, 1], color="#bbb", lw=0.5, zorder=1)
        sz = model.push(z.clone(), k).numpy()
        ax.scatter(sz[:, 0], sz[:, 1], s=3, color="#d62728", zorder=2)
        ax.set_title("base N(0,I)" if k == 0 else f"after layer {k}")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)

    ax = axes[5]
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 150), np.linspace(-2.5, 2.5, 150))
    pts = torch.tensor(np.stack([xx.ravel(), yy.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        lp = model.log_prob(pts).reshape(150, 150).numpy()
    cs = ax.contourf(xx, yy, lp, levels=30, cmap="viridis", vmin=-8)
    fig.colorbar(cs, ax=ax, label="log density")
    ax.scatter(data[:800, 0], data[:800, 1], s=2, color="w", alpha=0.4)
    ax.set_title("learned log-density")
    savefig(fig, outdir, "flow_warp.png", show=args.show)

    on = model.log_prob(data[:1000]).mean().item()
    off = model.log_prob(torch.tensor([[2.0, 2.0], [-2.0, -2.0], [0.0, 1.5]])).mean().item()
    print(f"mean log-density on moons: {on:.2f}   off moons: {off:.2f}")
    print(f"density concentrated on moons: {on > off}")


if __name__ == "__main__":
    main()
