# Rotation Number of a Circle Map

**Chapter 7 · Sections §7.1, §7.2**
**Concept:** Rotation number, Poincaré classification, Arnold tongues
**File:** `programs/ch7_rotation.py`
**Dependencies:** numpy, matplotlib

## What it shows

For the Arnold standard family f_{K,Ω}(θ) = θ + Ω - (K/2π)sin(2πθ), plots ρ(K, Ω) as a color map over (K,Ω)∈[0,1]×[0,1]. The Arnold tongues (resonance regions where ρ is rational) appear as colored wedges emanating from rational points on the Ω-axis. For K=0 the tongue width is 0 (pure rotation); at K=1 they overlap (chaos).

## Parameters

  - `K_range`: [0, 1], 200 pts
  - `Omega_range`: [0, 1], 200 pts
  - `N`: 1000 iterations for rotation number

## Algorithm

```
ρ(f) = lim_{n→∞} F^n(x)/n where F is a lift of f. For each (K,Ω), iterate 1000 steps, ρ ≈ (F^1000(x₀)-x₀)/1000. Color by rational value (nearest p/q with q≤10) or irrational.
```

## Expected output

Arnold tongue diagram: 200×200 color heatmap with tongue labels (0/1, 1/2, 1/3, …).
