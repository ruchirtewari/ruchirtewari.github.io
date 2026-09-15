# Substitution Systems — Fibonacci & Thue-Morse

**Chapter 3 · Sections §3.6**
**Concept:** Substitution dynamics, self-similarity, power spectrum
**File:** `programs/ch3_substitution.py`
**Dependencies:** numpy, matplotlib

## What it shows

Iterates substitution rules (e.g., Fibonacci: a→ab, b→a; Thue-Morse: a→ab, b→ba) to large length. Plots the sequence as a 1D color strip and its Fourier power spectrum. Fibonacci spectrum has discrete lines at golden-ratio-spaced frequencies (quasicrystal); Thue-Morse has singular continuous spectrum.

## Parameters

  - `rule`: fibonacci | thue_morse | user
  - `n_levels`: number of substitution steps, default 14

## Algorithm

```
Apply substitution n_levels times starting from 'a'. Map letters to {0,1} or {-1,+1}. Compute FFT; plot |FFT|² on log-log axes.
```

## Expected output

2-panel: sequence strip + power spectrum (linear and log-log).
