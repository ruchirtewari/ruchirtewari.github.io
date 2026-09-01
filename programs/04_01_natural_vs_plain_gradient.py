"""Chapter 4, §4.1 — natural vs plain gradient on a Gaussian fit (entry 8).

Purpose: the chapter's core picture — plain gradient zigzags/stalls while
natural gradient follows the information geometry; plus an invariance
demonstration under reparametrization.

Fit (mu, log sigma) of a Gaussian to 500 samples from N(3, 0.3^2) by NLL
descent from (mu=-2, sigma=3). Fisher in (mu, log sigma) is
F = diag(1/sigma^2, 2) (the 2/sigma^2 curvature in sigma times
(d sigma/d log sigma)^2 = sigma^2). In (mu, v=sigma^2), F = diag(1/v, 1/(2 v^2)).

Expected result (acceptance): GD oscillates/stalls as sigma shrinks; NGD path
smooth; after reparametrizing sigma -> sigma^2 the two NGD trajectories mapped
into (mu, sigma) space coincide while the GD trajectories differ.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import get_arg_parser, init, savefig

MU0, SIG0 = -2.0, 3.0


def nll(mu, sig, x):
    return np.log(sig) + ((x - mu) ** 2).mean() / (2 * sig ** 2) \
        + 0.5 * np.log(2 * np.pi)


def descend(x, lr, steps, natural, param):
    """param 'logsig': theta = (mu, log sigma); 'var': theta = (mu, sigma^2).
    Returns trajectory in (mu, sigma) space."""
    if param == "logsig":
        th = np.array([MU0, np.log(SIG0)])
    else:
        th = np.array([MU0, SIG0 ** 2])
    traj = []
    for _ in range(steps):
        mu = th[0]
        sig = np.exp(th[1]) if param == "logsig" else np.sqrt(max(th[1], 1e-6))
        traj.append((mu, sig))
        m1 = (mu - x).mean()
        m2 = ((x - mu) ** 2).mean()
        if param == "logsig":
            g = np.array([m1 / sig ** 2, 1.0 - m2 / sig ** 2])
            if natural:
                g = g / np.array([1.0 / sig ** 2, 2.0])  # F^-1 g
        else:
            v = sig ** 2
            g = np.array([m1 / v, 0.5 / v - m2 / (2 * v ** 2)])
            if natural:
                g = g / np.array([1.0 / v, 1.0 / (2 * v ** 2)])
        th = th - lr * g
        if param == "var":
            th[1] = max(th[1], 1e-4)
    return np.array(traj)


def main():
    args = get_arg_parser("plain vs natural gradient, Gaussian NLL").parse_args()
    outdir = init(args, "04_01_natural_vs_plain_gradient")
    x = 3.0 + 0.3 * np.random.randn(500)
    steps = 150 if args.quick else 400
    lr_gd, lr_ngd = 0.12, 0.1

    gd_ls = descend(x, lr_gd, steps, natural=False, param="logsig")
    ngd_ls = descend(x, lr_ngd, steps, natural=True, param="logsig")
    gd_v = descend(x, 2e-3, steps, natural=False, param="var")
    ngd_v = descend(x, lr_ngd, steps, natural=True, param="var")

    # --- figure 1: trajectories over contours + loss curves ---
    mus = np.linspace(-3, 5, 200)
    sigs = np.linspace(0.05, 4, 200)
    MU, SG = np.meshgrid(mus, sigs)
    Z = np.log(SG) + ((x[:, None, None] - MU) ** 2).mean(0) / (2 * SG ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.contour(MU, SG, Z, levels=40, colors="#ccc", linewidths=0.6)
    ax.plot(gd_ls[:, 0], gd_ls[:, 1], ".-", ms=3, label="plain GD")
    ax.plot(ngd_ls[:, 0], ngd_ls[:, 1], ".-", ms=3, label="natural GD")
    ax.plot([x.mean()], [x.std()], "k*", ms=14, label="optimum")
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\sigma$")
    ax.set_title(r"trajectories in $(\mu,\log\sigma)$ parametrization")
    ax.legend()
    ax = axes[1]
    for tr, name in [(gd_ls, "plain GD"), (ngd_ls, "natural GD")]:
        losses = [nll(m, s, x) for m, s in tr]
        ax.semilogy(np.array(losses) - nll(x.mean(), x.std(), x) + 1e-12,
                    label=name)
    ax.set_xlabel("step"); ax.set_ylabel("NLL - NLL*")
    ax.set_title("loss vs step"); ax.legend()
    savefig(fig, outdir, "trajectories.png", show=args.show)

    # --- figure 2: reparametrization invariance ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contour(MU, SG, Z, levels=40, colors="#ddd", linewidths=0.6)
    ax.plot(ngd_ls[:, 0], ngd_ls[:, 1], "-", lw=3, label=r"NGD in $(\mu,\log\sigma)$")
    ax.plot(ngd_v[:, 0], ngd_v[:, 1], "--", lw=2, label=r"NGD in $(\mu,\sigma^2)$")
    ax.plot(gd_ls[:, 0], gd_ls[:, 1], "-", lw=1.5, label=r"GD in $(\mu,\log\sigma)$")
    ax.plot(gd_v[:, 0], gd_v[:, 1], "-", lw=1.5, label=r"GD in $(\mu,\sigma^2)$")
    ax.plot([x.mean()], [x.std()], "k*", ms=14)
    ax.set_xlabel(r"$\mu$"); ax.set_ylabel(r"$\sigma$")
    ax.set_title("NGD is (near-)invariant to reparametrization; GD is not")
    ax.legend()
    savefig(fig, outdir, "reparam_invariance.png", show=args.show)

    n = min(len(ngd_ls), len(ngd_v))
    ngd_dev = np.abs(ngd_ls[:n] - ngd_v[:n]).max()
    gd_dev = np.abs(gd_ls[:n] - gd_v[:n]).max()
    print(f"target: mu*={x.mean():.3f} sigma*={x.std():.3f}")
    print("\nfinal (mu, sigma):")
    for name, tr in [("GD (logsig)", gd_ls), ("NGD (logsig)", ngd_ls),
                     ("GD (var)", gd_v), ("NGD (var)", ngd_v)]:
        print(f"  {name:14s} mu={tr[-1, 0]:7.3f}  sigma={tr[-1, 1]:7.3f}")
    print(f"\nmax trajectory deviation between parametrizations:")
    print(f"  NGD: {ngd_dev:.4f}   GD: {gd_dev:.4f}")
    print(f"NGD invariant (deviation << GD's): {ngd_dev < 0.1 * gd_dev}")


if __name__ == "__main__":
    main()
