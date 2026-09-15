# Poincaré Recurrence — Return Times Distribution

**Chapter 4 · Sections §4.2**
**Concept:** Recurrence, Kac's lemma, return-time distribution
**File:** `programs/ch4_poincare.py`
**Dependencies:** numpy, matplotlib

## What it shows

For the irrational rotation Rα and the doubling map E₂(x)=2x mod 1, picks a small interval A of measure µ(A)=ε and histograms the return times τ_A(x) for many x₀∈A. Kac's lemma predicts ⟨τ_A⟩=1/µ(A). For E₂ (mixing) the distribution is approximately geometric; for Rα (equidistributed) it is concentrated near 1/ε.

## Parameters

  - `alpha`: (√5-1)/2
  - `epsilon`: 0.02 (interval length = measure)
  - `n_points`: 5000 starting points
  - `max_return`: 10000 iterations

## Algorithm

```
For each x₀ ∈ A=[0,ε], iterate until x_n ∈ A; record n. Histogram return times. Draw vertical line at 1/ε (Kac prediction).
```

## Expected output

2-panel histogram: Rα return times | E₂ return times, both with Kac mean line.
