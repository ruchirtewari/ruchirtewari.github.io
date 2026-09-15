# Birkhoff Ergodic Theorem — Time Average vs Space Average

**Chapter 4 · Sections §4.3, §4.5, §4.7**
**Concept:** Ergodicity, time averages, equidistribution
**File:** `programs/ch4_birkhoff.py`
**Dependencies:** numpy, matplotlib

## What it shows

For irrational rotation Rα (ergodic) and rational rotation Rp/q (non-ergodic), plots (1/N)Σf(R^n_α(x)) vs N for a test function f. Ergodic case converges to ∫f dθ regardless of x; non-ergodic case converges to orbit average, which depends on x. Side panel shows equidistribution: scatter of {nα mod 1} for 1000 steps.

## Parameters

  - `alpha`: irrational rotation angle (default: (√5-1)/2 = golden ratio)
  - `alpha2`: rational angle p/q for comparison
  - `f`: observable: sin(2πx) | indicator([0,0.5]) | x²
  - `N_max`: 10000 iterations

## Algorithm

```
Orbit: x_n = x_0 + nα mod 1. Running average A_N = (1/N)Σ_{n=0}^{N-1} f(x_n). Space average: numerical integration of f over [0,1].
```

## Expected output

2-panel: left=running average converging to space mean (2 curves: ergodic vs non-ergodic); right=equidistribution scatter.
