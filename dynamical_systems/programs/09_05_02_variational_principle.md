# Variational Principle — h(f) = sup_µ h_µ(f)

**Chapter 9 · Sections §9.5**
**Concept:** Variational principle, measure-theoretic vs topological entropy
**File:** `programs/ch9_variational.py`
**Dependencies:** numpy, matplotlib

## What it shows

For the logistic map f_r at various r, compares: (a) topological entropy h(f_r) estimated via spanning sets (ch2), (b) metric entropy h_µ(f_r) for the natural SRB measure µ, estimated via the Shannon-McMillan-Breiman theorem: -(1/n)log µ(ξ^n(x)) → h_µ(T) a.s. Variational principle: the two should agree at the measure of maximal entropy. Plots both curves on the same axes.

## Parameters

  - `r_values`: np.linspace(3.5, 4.0, 100)
  - `n_smb`: 500 (SMB averaging length)
  - `n_orbit`: 50000

## Algorithm

```
SMB: generate long orbit, track partition element at each step, accumulate log-measure, divide by n. Topological entropy: spanning-set estimate from ch2 program. Both plotted vs r with h=log2 reference line at r=4.
```

## Expected output

Single plot: h_top(r) and h_µ(r) vs r, with h=log2 horizontal ref.
