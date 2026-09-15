# Newton's Method Basins — Rational Map on ℂ̄

**Chapter 8 · Sections §8.4, §8.5**
**Concept:** Fatou/Julia decomposition, immediate basins, rational maps
**File:** `programs/ch8_newton.py`
**Dependencies:** numpy, matplotlib

## What it shows

Newton's method for p(z)=zⁿ-1 (roots are n-th roots of unity) gives a rational map R(z)=z - p(z)/p'(z) with n Fatou components (basins of attracting fixed points). The Julia set J(R) is the boundary between basins — a fractal. Color each z₀ by which root it converges to. For n=3 the fractal is especially striking.

## Parameters

  - `degree`: 3 (default), 4, 5
  - `resolution`: 1000×1000
  - `max_iter`: 100
  - `tol`: 1e-6

## Algorithm

```
For each z₀ in grid: iterate z ← z - p(z)/p'(z) until |p(z)|<tol or max_iter. Identify which root (arg(z) mod 2π/n → index 0..n-1). Color by (root_index, convergence_speed).
```

## Expected output

Full-screen fractal: n-colored basins with Julia boundary visible.
