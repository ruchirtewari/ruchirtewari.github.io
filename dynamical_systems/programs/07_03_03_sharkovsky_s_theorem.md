# Sharkovsky's Theorem — Period Implication Cascade

**Chapter 7 · Sections §7.3**
**Concept:** Period forcing, Sharkovsky ordering, period-3 implies chaos
**File:** `programs/ch7_sharkovsky.py`
**Dependencies:** numpy, matplotlib

## What it shows

For the tent map T_r(x) = r·min(x, 1-x), sweeps r∈[1,2] and detects which periods exist at each r. Displays as a 'period existence' heat-strip: once period 3 appears at r≈1.9, all periods are present (Sharkovsky). Also shows explicit period-3 orbit and proves its existence via the intermediate value theorem argument.

## Parameters

  - `r_range`: [1.0, 2.0], 500 pts
  - `max_period`: 16
  - `n_detect`: 10000 transient + 2000 orbit per r

## Algorithm

```
For each r: iterate tent map, collect unique orbit points (rounded). Detect periods by checking when x_n ≈ x_0. Build period-existence matrix, display as imshow.
```

## Expected output

Heat strip: x-axis=r, y-axis=period p∈{1..16}, color=exists.
