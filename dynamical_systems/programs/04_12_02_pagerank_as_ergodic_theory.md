# PageRank as Ergodic Theory — Stationary Measure of a Random Walk

**Chapter 4 · Sections §4.12, §4.6**
**Concept:** Invariant measures, unique ergodicity, PageRank
**File:** `programs/ch4_pagerank.py`
**Dependencies:** numpy, matplotlib, networkx

## What it shows

Builds a small random web graph, constructs the Google matrix G = (1-d)·(1/n)·11ᵀ + d·A (d=0.85), and computes the stationary distribution by power iteration. Animates rank convergence and displays the final ranked graph with node size ∝ PageRank.

## Parameters

  - `n_pages`: 20 (default)
  - `edge_prob`: 0.3
  - `damping`: 0.85
  - `n_iter`: 100 power iterations

## Algorithm

```
Build random directed graph. Form column-stochastic A. G = d·A + (1-d)/n·1. Power iteration: v_{k+1} = G·v_k until ||Δv||<1e-8. Draw with networkx spring layout, node size ∝ rank.
```

## Expected output

Animated convergence plot + final ranked graph visualization.
