# Strange Attractors — Hénon & Lorenz

**Chapter 1 · Sections §1.12, §1.13**
**Concept:** Attractors, sensitive dependence, Lyapunov exponents
**File:** `programs/ch1_henon_lorenz.py`
**Dependencies:** numpy, matplotlib, scipy

## What it shows

Side-by-side plots: Hénon map attractor (2D) and Lorenz flow attractor (3D). Overlays two nearby initial conditions diverging exponentially to illustrate sensitivity. Computes and displays the leading Lyapunov exponent numerically.

## Parameters

  - `henon_a`: 1.4 (classic)
  - `henon_b`: -0.3 (classic)
  - `lorenz_sigma`: 10
  - `lorenz_rho`: 28
  - `lorenz_beta`: 8/3
  - `n_points`: 100000 for Hénon, 50000 steps for Lorenz
  - `delta_x0`: 1e-8 (perturbation for Lyapunov)

## Algorithm

```
Hénon: iterate (x,y)→(a-x²+by, x) from (0,0).
Lorenz: RK4 on dx/dt=σ(y-x), dy/dt=x(ρ-z)-y, dz/dt=xy-βz.
Lyapunov: run two orbits; λ = (1/N)Σ log(|δ_n|/|δ_0|) with renormalization every k steps.
```

## Expected output

2-panel figure: Hénon attractor scatter (2D) + Lorenz butterfly (3D). Inset: divergence-of-nearby-orbits plot with λ annotation.
