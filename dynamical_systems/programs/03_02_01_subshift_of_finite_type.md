# Subshift of Finite Type — Transition Graph & Perron-Frobenius

**Chapter 3 · Sections §3.2, §3.3, §3.4**
**Concept:** SFT, transition matrix, Perron-Frobenius, periodic orbit counting
**File:** `programs/ch3_sft.py`
**Dependencies:** numpy, matplotlib, networkx

## What it shows

Interactive: user enters a 0/1 transition matrix A. Program draws the transition graph (digraph), computes spectral radius ρ(A) = e^h, shows the Perron eigenvector as stationary probabilities on edges, and plots the number of periodic orbits of period n vs n (should grow like ρ^n).

## Parameters

  - `A`: transition matrix (default: [[1,1],[1,0]] Fibonacci shift)
  - `n_max`: max period for orbit count, default 20

## Algorithm

```
1. Compute eigenvalues of A; ρ = max |λ|.
2. #fixed pts of σ^n = Tr(A^n); plot vs n on semi-log axes.
3. Draw graph with networkx; node size ∝ stationary prob.
4. Display Perron eigenvector as bar chart.
```

## Expected output

3-panel: transition graph | periodic orbit counts (log scale) | Perron eigenvector bars.
