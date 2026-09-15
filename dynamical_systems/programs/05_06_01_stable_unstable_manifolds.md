# Stable & Unstable Manifolds — Hyperbolic Fixed Point

**Chapter 5 · Sections §5.6, §5.8**
**Concept:** Hyperbolic sets, stable/unstable manifolds, homoclinic points
**File:** `programs/ch5_manifolds.py`
**Dependencies:** numpy, matplotlib, scipy

## What it shows

For the Hénon map (or Arnold's cat map on ℝ²), computes the stable manifold W^s(p) and unstable manifold W^u(p) of a hyperbolic fixed point p by iterating a small segment of the linearized (un)stable direction forward and backward respectively. Displays the transverse homoclinic intersection, the hallmark of chaos.

## Parameters

  - `map`: henon | catmap
  - `a`: 1.4
  - `b`: -0.3
  - `n_manifold_pts`: 5000
  - `n_iter`: 20 (folding iterations)

## Algorithm

```
Find fixed point p by Newton iteration on (f(x)-x=0). Linearize: compute Df(p); eigenvectors give E^s, E^u directions. W^u: iterate small segment along E^u direction forward n times. W^s: iterate small segment along E^s direction backward n times.
```

## Expected output

Phase-plane plot: W^s in blue, W^u in red, fixed point marked, homoclinic tangle visible after enough iterations.
