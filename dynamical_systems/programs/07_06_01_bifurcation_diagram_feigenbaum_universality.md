# Bifurcation Diagram & Feigenbaum Universality

**Chapter 7 · Sections §7.6, §7.7, §7.8**
**Concept:** Period doubling, bifurcations, Feigenbaum constant δ≈4.669
**File:** `programs/ch7_bifurcation.py`
**Dependencies:** numpy, matplotlib

## What it shows

High-resolution bifurcation diagram for f_r(x)=rx(1-x), r∈[0,4]. Zooms into successive period-doubling bifurcations to measure δ_n = (r_{n+1}-r_n)/(r_{n+2}-r_{n+1}) → 4.669… Shows self-similar structure (the diagram is asymptotically its own rescaling by δ horizontally and 2.5 vertically).

## Parameters

  - `r_range`: [2.4, 4.0], 3000 pts
  - `n_transient`: 1000
  - `n_plot`: 500
  - `zoom_sequence`: auto-detect bifurcation points

## Algorithm

```
Bifurcation: for each r, iterate 1000 steps (transient), plot 500. Bifurcation points: r₁≈3.0, r₂≈3.449, r₃≈3.544, r₄≈3.5644, …
Feigenbaum: δ_n = (r_n - r_{n-1})/(r_{n+1} - r_n). Inset: zoom boxes showing self-similarity.
```

## Expected output

Main bifurcation diagram + 3 zoom insets + table of δ_n converging to 4.6692… displayed as annotation.
