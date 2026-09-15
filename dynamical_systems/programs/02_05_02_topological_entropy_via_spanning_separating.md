# Topological Entropy via Spanning/Separating Sets

**Chapter 2 · Sections §2.5, §2.6**
**Concept:** Topological entropy, exponential orbit complexity
**File:** `programs/ch2_entropy.py`
**Dependencies:** numpy, matplotlib, scipy

## What it shows

Computes h(f) numerically for the logistic map f_r(x)=rx(1-x) across r ∈ [0,4]. At each r, counts minimal (n,ε)-spanning sets for increasing n and fits log(S(n,ε))/n to estimate h(f_r). Plots h(f_r) vs r alongside the bifurcation diagram so the entropy-chaos connection is visually direct.

## Parameters

  - `r_range`: [0, 4], 400 values
  - `epsilon`: 0.01
  - `n_max`: 15 (orbit length)
  - `n_sample`: 2000 initial conditions

## Algorithm

```
For each r and n, generate all n-step orbits from a grid. Use greedy set cover to estimate spanning set size S(n,ε). Fit line to log(S(n,ε)) vs n; slope ≈ h(f).
Bifurcation diagram: discard transient, plot last 100 iterates.
```

## Expected output

2-panel figure: top=bifurcation diagram, bottom=h(f_r) vs r with log(2) marked at r=4 (full chaos).
