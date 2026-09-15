# Omega-Limit Sets & Non-Wandering Set

**Chapter 2 · Sections §2.1**
**Concept:** ω-limit sets, recurrence, non-wandering set
**File:** `programs/ch2_omega_limit.py`
**Dependencies:** numpy, matplotlib, scipy

## What it shows

For a 2D flow (e.g., Duffing or Van der Pol), scatter-plots the ω-limit set of many initial conditions using color to distinguish basins. Overlays the non-wandering set NW(f) detected by tracking recurrence times. Makes the distinction between limit sets and transient behavior vivid.

## Parameters

  - `system`: van_der_pol | duffing | user_ode
  - `mu`: Van der Pol damping, default 1.0
  - `n_orbits`: 50 random initial conditions
  - `T`: integration time, default 200

## Algorithm

```
Integrate ODE with scipy.integrate.solve_ivp. After transient (first half of T), collect remaining orbit points. Cluster by color using starting basin. Highlight fixed/periodic pts.
```

## Expected output

Phase portrait with colored ω-limit sets, limit cycles marked.
