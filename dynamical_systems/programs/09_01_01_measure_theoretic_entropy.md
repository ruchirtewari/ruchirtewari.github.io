# Measure-Theoretic Entropy — Partition Refinement

**Chapter 9 · Sections §9.1, §9.3, §9.4**
**Concept:** Entropy of partitions, Kolmogorov-Sinai theorem, information
**File:** `programs/ch9_entropy.py`
**Dependencies:** numpy, matplotlib

## What it shows

For the doubling map T(x)=2x mod 1 with the natural partition ξ={[0,0.5),[0.5,1)}, computes H(ξ^n) = H(ξ ∨ T⁻¹ξ ∨ … ∨ T^{-(n-1)}ξ) for n=1..15. This grows linearly: H(ξ^n) = n·log2 (binary entropy). The slope is the metric entropy h(T,ξ) = log2. Compares with the irrational rotation (h=0): H(ξ^n) grows like log(n) not linearly.

## Parameters

  - `map`: doubling | rotation_alpha
  - `alpha`: (√5-1)/2
  - `n_max`: 20 (partition levels)
  - `n_points`: 100000 (Monte Carlo for measure)

## Algorithm

```
Generate n_points uniform samples. For each n, compute the joined partition ξ^n (2^n intervals for doubling map). Estimate µ(C_i) by counting samples. H = -Σ µ_i log µ_i. Plot H(ξ^n) vs n; fit line to extract slope = h.
```

## Expected output

Plot of H(ξ^n) vs n for both maps: doubling (linear, slope=log2≈0.693) and rotation (sublinear, slope→0). Annotated with h values.
