# Ergodicity of Cat Map — Equidistribution Along Foliations

**Chapter 6 · Sections §6.3**
**Concept:** Ergodicity, Hopf argument, stable/unstable foliations
**File:** `programs/ch6_ergodicity.py`
**Dependencies:** numpy, matplotlib

## What it shows

Illustrates Hopf's argument numerically for Arnold's cat map. Takes a smooth test function φ on T². Computes forward time averages A^+_N φ(x) = (1/N)Σ φ(f^k x) and backward averages A^-_N φ(x). Shows A^+_N is approximately constant along W^s leaves and A^-_N along W^u leaves; both converge to ∫φ dµ as N grows, confirming ergodicity.

## Parameters

  - `phi`: sin(2πx)cos(2πy) | indicator of quadrant
  - `N`: 500 iterations
  - `grid`: 64×64 evaluation points on T²

## Algorithm

```
64×64 grid of initial points. For each, iterate cat map N times. Compute A^+_N and A^-_N. Display as heatmaps. Also show variance of A^+_N along stable leaf segments → 0.
```

## Expected output

4-panel: φ | A^+_50 | A^+_500 | A^-_500, all as heatmaps on T². Should see A^+ flattening to uniform gray = ∫φ.
