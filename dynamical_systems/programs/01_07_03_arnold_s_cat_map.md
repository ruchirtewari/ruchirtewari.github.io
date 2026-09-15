# Arnold's Cat Map — Hyperbolic Toral Automorphism

**Chapter 1 · Sections §1.7**
**Concept:** Hyperbolic toral automorphism, mixing, image stretching
**File:** `programs/ch1_cat_map.py`
**Dependencies:** numpy, matplotlib, pillow

## What it shows

Animates the action of A = [[2,1],[1,1]] on T² = [0,1)²: a pixel image of a cat (or test image) gets stretched, sheared, and wrapped until it appears random; after exactly as many steps as the period it returns. Shows exponential mixing visually.

## Parameters

  - `matrix`: [[2,1],[1,1]] default (cat map)
  - `n_steps`: animation frames, default 20
  - `image_size`: 64×64 to 256×256

## Algorithm

```
For each pixel (i,j) in NxN grid, apply (x,y)→A(x,y) mod 1 iteratively, sampling pixel colors. Display as animation or n-panel grid showing t=0,1,2,...,T.
```

## Expected output

Animated GIF or multi-panel figure of image evolution on torus.
