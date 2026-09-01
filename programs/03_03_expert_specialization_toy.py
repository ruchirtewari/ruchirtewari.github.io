"""Chapter 3, §3.3 — mixture-of-experts specialization and collapse (entry 7).

Purpose: watch mixture-of-experts specialization emerge on a four-quadrant
regression task with a soft learned gate — and its collapse failure mode
under hard top-1 routing with no load balancing.

Expected result (acceptance): soft-gate run converges to roughly one expert
per quadrant (gate argmax map shows four colored regions); hard-routing run
exhibits expert collapse (one expert takes nearly all traffic).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from common import get_arg_parser, init, savefig


def target(xy):
    x, y = xy[:, 0], xy[:, 1]
    out = torch.empty(len(xy))
    q1 = (x >= 0) & (y >= 0); out[q1] = torch.sin(3 * np.pi * x[q1])       # sin
    q2 = (x < 0) & (y >= 0);  out[q2] = x[q2] * y[q2] * 4                  # saddle
    q3 = (x < 0) & (y < 0);   out[q3] = x[q3] + y[q3]                      # plane
    q4 = (x >= 0) & (y < 0)
    out[q4] = torch.exp(-8 * ((x[q4] - 0.5) ** 2 + (y[q4] + 0.5) ** 2))    # bump
    return out


def quadrant(xy):
    return (xy[:, 0] < 0).long() * 1 + (xy[:, 1] < 0).long() * 2  # 0..3


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
            for _ in range(4))
        self.gate = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 4))

    def forward(self, x, hard=False):
        probs = torch.softmax(self.gate(x), dim=-1)          # (B,4)
        outs = torch.cat([e(x) for e in self.experts], 1)    # (B,4)
        if hard:
            idx = probs.argmax(1)
            onehot = nn.functional.one_hot(idx, 4).float()
            w = onehot + probs - probs.detach()              # straight-through
        else:
            w = probs
        return (w * outs).sum(1), probs


def run(hard, steps, snaps, seed):
    torch.manual_seed(seed)
    model = MoE()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    snapshots, usage = [], []
    snap_at = [int(s) for s in np.linspace(0, steps - 1, snaps)]
    for step in range(steps):
        x = torch.rand(512, 2) * 2 - 1
        y = target(x)
        pred, probs = model(x, hard=hard)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        usage.append(probs.argmax(1).bincount(minlength=4).float().numpy() / 512)
        if step in snap_at:
            snapshots.append((step, snapshot(model), loss.item()))
    return model, snapshots, np.array(usage)


def snapshot(model):
    n = 80
    g = torch.linspace(-1, 1, n)
    xx, yy = torch.meshgrid(g, g, indexing="xy")
    pts = torch.stack([xx.ravel(), yy.ravel()], 1)
    with torch.no_grad():
        _, probs = model(pts)
        gmap = probs.argmax(1).reshape(n, n).numpy()
        outs = torch.cat([e(pts) for e in model.experts], 1)
        tgt = target(pts)
        q = quadrant(pts)
        err = np.zeros((4, 4))  # expert x quadrant MSE
        for e in range(4):
            for qi in range(4):
                m = q == qi
                err[e, qi] = ((outs[m, e] - tgt[m]) ** 2).mean().item()
    return gmap, err


def main():
    args = get_arg_parser("MoE specialization vs collapse").parse_args()
    outdir = init(args, "03_03_expert_specialization_toy")
    steps = 400 if args.quick else 3000
    snaps = 5

    # --- run 1: soft gating ---
    model, shots, soft_usage = run(hard=False, steps=steps, snaps=snaps, seed=args.seed)
    fig, axes = plt.subplots(2, snaps, figsize=(4 * snaps, 8))
    for j, (step, (gmap, err), loss) in enumerate(shots):
        ax = axes[0, j]
        ax.imshow(gmap, origin="lower", extent=[-1, 1, -1, 1], cmap="tab10",
                  vmin=0, vmax=9)
        ax.set_title(f"gate argmax, step {step}\nloss {loss:.3f}")
        ax = axes[1, j]
        im = ax.imshow(np.log10(err + 1e-6), cmap="magma")
        ax.set_xlabel("quadrant"); ax.set_ylabel("expert")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_title("log10 MSE expert x quadrant")
    fig.colorbar(im, ax=axes[1, -1])
    fig.suptitle("Soft-gate MoE: specialization emerging")
    savefig(fig, outdir, "soft_gate_specialization.png", show=args.show)

    gmap, err = shots[-1][1]
    assign = err.argmin(0)  # best expert per quadrant
    print("final per-expert-per-quadrant MSE (soft gate):")
    print(np.array2string(err, precision=4))
    print(f"best expert per quadrant: {assign}  "
          f"(distinct experts used: {len(set(assign.tolist()))}/4)")

    # --- run 2: hard top-1 routing ---
    _, hard_shots, hard_usage = run(hard=True, steps=steps, snaps=snaps,
                                    seed=args.seed + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for e in range(4):
        axes[0].plot(soft_usage[:, e], label=f"expert {e}")
        axes[1].plot(hard_usage[:, e], label=f"expert {e}")
    axes[0].set_title("soft gate: expert usage over training")
    axes[1].set_title("hard top-1 routing: collapse")
    for ax in axes:
        ax.set_xlabel("step"); ax.set_ylabel("fraction of batch routed")
        ax.set_ylim(0, 1); ax.legend()
    savefig(fig, outdir, "expert_usage.png", show=args.show)

    final_hard = hard_usage[-50:].mean(0)
    final_soft = soft_usage[-50:].mean(0)
    print("\nfinal expert usage (mean of last 50 steps):")
    print(f"soft: {np.array2string(final_soft, precision=3)}")
    print(f"hard: {np.array2string(final_hard, precision=3)}")
    print(f"hard-routing dominant expert takes {final_hard.max():.0%} of traffic "
          f"(collapse: {final_hard.max() > 0.6})")


if __name__ == "__main__":
    main()
