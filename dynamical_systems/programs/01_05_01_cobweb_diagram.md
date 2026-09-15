# Cobweb Diagram — Quadratic Family & Gauss Map

**Chapter 1 · Sections §1.5, §1.6**
**Concept:** Orbits, fixed points, periodic orbits, iteration
**File:** `programs/ch1_cobweb.py`
**Dependencies:** numpy, matplotlib

## What it shows

Interactive cobweb plot for f(x) = x²+c (or any user map). Starting from x₀ the cobweb zig-zags between y=f(x) and y=x, making convergence to fixed/periodic orbits or chaos visually obvious. Include Gauss map G(x) = {1/x} (fractional part) as a preset.

## Parameters

  - `map`: quadratic | gauss | logistic | user-defined
  - `c`: parameter for f(x)=x²+c, default -1.0
  - `x0`: initial condition, default 0.3
  - `n_iter`: number of iterations, default 80

## Algorithm

```
1. Build (x, y=f(x)) curve on [0,1] or [-2,2].
2. From (x0, 0) draw vertical to (x0, f(x0)), then horizontal to    (f(x0), f(x0)), repeat n_iter times.
3. Color orbit segments by iteration index (early=blue, late=red).
```

## Expected output

Single matplotlib figure: y=x line, y=f(x) curve, cobweb path.
