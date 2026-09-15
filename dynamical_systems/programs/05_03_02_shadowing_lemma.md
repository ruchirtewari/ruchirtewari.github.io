# Shadowing Lemma — Pseudo-orbit vs True Orbit

**Chapter 5 · Sections §5.3**
**Concept:** ε-orbits, pseudo-orbits, shadowing, numerical reliability
**File:** `programs/ch5_shadowing.py`
**Dependencies:** numpy, matplotlib

## What it shows

Generates a δ-pseudo-orbit of the cat map (apply exact map, then add noise δ at each step). Finds the true orbit that shadows it (using the theoretical shadowing constant). Plots both orbits on T² to show they stay close. Also shows what happens when δ is too large (no shadow exists).

## Parameters

  - `delta`: perturbation per step: [1e-4, 1e-3, 1e-2, 0.1]
  - `n_steps`: 50
  - `n_trials`: 5 pseudo-orbits

## Algorithm

```
Cat map A=[[2,1],[1,1]] on [0,1)². Pseudo-orbit: x_{n+1} = A·x_n mod 1 + η (||η||≤δ). Shadow: solve least-squares system for x_0 such that true orbit stays within C·δ of pseudo-orbit. Display on torus.
```

## Expected output

2-panel: small δ (orbits indistinguishable) vs large δ (diverge).
