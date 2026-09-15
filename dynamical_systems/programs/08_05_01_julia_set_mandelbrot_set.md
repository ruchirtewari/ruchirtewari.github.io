# Julia Set & Mandelbrot Set — Complex Quadratic Family

**Chapter 8 · Sections §8.5, §8.6**
**Concept:** Julia set, Fatou set, Mandelbrot set, connectedness
**File:** `programs/ch8_julia_mandelbrot.py`
**Dependencies:** numpy, matplotlib

## What it shows

GPU-friendly (pure numpy vectorized) renderer for f_c(z) = z²+c. Mandelbrot set: color each c by escape time of orbit of 0. Julia set: for fixed c, color each z₀ by escape time. Clicking on a point in the Mandelbrot set updates the Julia set in real time (matplotlib event loop). Demonstrates: c∈M ↔ J(f_c) connected; c∉M ↔ J(f_c) Cantor dust.

## Parameters

  - `resolution`: 800×800 default
  - `max_iter`: 256
  - `escape_radius`: 2.0
  - `colormap`: inferno | hot | custom smooth bands

## Algorithm

```
Vectorized: Z = grid of z₀, C = constant. Iterate Z ← Z²+C up to max_iter; record escape iteration n. Smooth coloring: n + 1 - log(log|Z|)/log2 for anti-banding. Mandelbrot: C = grid, Z starts at 0.
```

## Expected output

Interactive 2-panel matplotlib figure: Mandelbrot (left), Julia for clicked c (right). Click anywhere in Mandelbrot to update.
