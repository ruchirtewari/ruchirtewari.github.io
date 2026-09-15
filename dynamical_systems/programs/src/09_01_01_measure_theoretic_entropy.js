/**
 * @file 09_01_01_measure_theoretic_entropy.js
 * @chapter 9 — Entropy
 * @sections §9.1, §9.3, §9.4
 * @concept Entropy of a partition, Kolmogorov–Sinai theorem, information growth
 *
 * The entropy of a finite partition ξ = {C₁,…,C_m} with respect to an
 * invariant measure µ is
 *
 *     H(ξ) = − Σ_i µ(C_i) · log µ(C_i)          (log base e → nats).
 *
 * Iterating a map T refines ξ into the JOIN
 *
 *     ξⁿ = ξ ∨ T⁻¹ξ ∨ … ∨ T^{-(n-1)}ξ,
 *
 * whose atoms are the sets of points sharing the first n symbols of their
 * itinerary. The Kolmogorov–Sinai entropy rate is  h(T,ξ) = limₙ H(ξⁿ)/n.
 *
 * DOUBLING MAP  T(x) = 2x mod 1, natural partition ξ = {[0,½),[½,1)}. The
 * atoms of ξⁿ are exactly the dyadic intervals [k/2ⁿ,(k+1)/2ⁿ), all of measure
 * 2⁻ⁿ, so H(ξⁿ) = 2ⁿ·(2⁻ⁿ·log 2ⁿ) = n·log 2 — LINEAR growth, rate log 2.
 *
 * IRRATIONAL ROTATION  R_α(x) = x + α mod 1. The boundaries of ξⁿ are the
 * 2n points { −kα, ½ − kα : k = 0,…,n−1 } (mod 1). By the THREE-DISTANCE
 * THEOREM these cut the circle into intervals taking only 2 or 3 distinct
 * lengths, so H(ξⁿ) grows only like O(log n): the rate is h = 0. The rotation
 * is deterministic in the entropy sense even though its orbits are dense.
 *
 * Left: H(ξⁿ) vs n for both maps against the slope-log 2 guide. Right: the
 * actual atoms of ξⁿ drawn as coloured intervals on [0,1], n = 1..8.
 */

var MeasureEntropy = (() => {
  const { Palette, Canvas, Controls, Math2D } = DSUtils;

  const meta = {
    id: '09_01_01',
    title: 'Measure-Theoretic Entropy — Partition Refinement',
    chapter: 'Chapter 9 — Entropy',
    sections: ['9.1', '9.3', '9.4'],
    concept: 'H(ξⁿ) growth: linear (rate log 2) for the doubling map, logarithmic (rate 0) for rotation',
    description:
      'Plots the partition entropy H(ξⁿ) for the doubling map and an irrational ' +
      'rotation, and draws the refined partitions atom-by-atom, illustrating the ' +
      'Kolmogorov–Sinai entropy rate via the slope of H(ξⁿ).',
  };

  const LN2 = Math.LN2;
  const mod1 = x => ((x % 1) + 1) % 1;

  const DEFAULTS = { alpha: (Math.sqrt(5) - 1) / 2, n_max: 15, show_rotation: true, show_partition_visual: true };

  // Atoms of ξⁿ for the doubling map: dyadic intervals.
  function doublingAtoms(n) {
    const N = 1 << n, atoms = [];
    for (let k = 0; k < N; k++) atoms.push([k / N, (k + 1) / N]);
    return atoms;
  }

  // Boundary points of ξⁿ for rotation R_α, then the atoms between them.
  function rotationAtoms(n, alpha) {
    const bset = new Set();
    for (let k = 0; k < n; k++) {
      bset.add(+mod1(-k * alpha).toFixed(12));
      bset.add(+mod1(0.5 - k * alpha).toFixed(12));
    }
    bset.add(0);
    const b = Array.from(bset).sort((p, q) => p - q);
    const atoms = [];
    for (let i = 0; i < b.length; i++) {
      const s = b[i], e = (i + 1 < b.length) ? b[i + 1] : 1;
      if (e - s > 1e-12) atoms.push([s, e]);
    }
    return atoms;
  }

  function entropyOfAtoms(atoms) {
    let H = 0;
    for (const [s, e] of atoms) { const p = e - s; if (p > 0) H -= p * Math.log(p); }
    return H;
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 400;
    const { ctx } = Canvas.create(container, W, H);

    // Left plot region.
    const PX0 = 46, PX1 = 300, PY0 = 40, PY1 = 336;
    // Right partition region.
    const RX0 = 322, RX1 = 552, RY0 = 40, RY1 = 380;

    function draw() {
      Canvas.clear(ctx, W, H);
      const nMax = state.n_max | 0;

      // ── compute H(ξⁿ) series ──────────────────────────────────────────
      const Hdoub = [], Hrot = [];
      for (let n = 1; n <= nMax; n++) {
        Hdoub.push(n * LN2);
        Hrot.push(entropyOfAtoms(rotationAtoms(n, state.alpha)));
      }
      const maxH = nMax * LN2 * 1.05;

      const mapX = n => PX0 + (nMax <= 1 ? 0 : (n - 1) / (nMax - 1)) * (PX1 - PX0);
      const mapY = h => PY1 - (h / maxH) * (PY1 - PY0);

      // Axes.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(PX0, PY0); ctx.lineTo(PX0, PY1); ctx.lineTo(PX1, PY1); ctx.stroke();
      // y ticks every log2.
      for (let k = 0; k * LN2 <= maxH; k++) {
        const y = mapY(k * LN2);
        ctx.strokeStyle = Palette.surface;
        ctx.beginPath(); ctx.moveTo(PX0, y); ctx.lineTo(PX1, y); ctx.stroke();
        Canvas.label(ctx, `${k}·log2`, PX0 - 4, y + 3, { color: Palette.muted, font: '8px monospace', align: 'right' });
      }
      // x ticks.
      for (let n = 1; n <= nMax; n += Math.ceil(nMax / 8)) {
        Canvas.label(ctx, String(n), mapX(n), PY1 + 12, { color: Palette.muted, font: '8px monospace', align: 'center' });
      }

      // Slope-log2 guide (dashed gold): the maximal entropy rate.
      ctx.strokeStyle = Palette.gold; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(mapX(1), mapY(1 * LN2)); ctx.lineTo(mapX(nMax), mapY(nMax * LN2)); ctx.stroke();
      ctx.setLineDash([]);

      // Doubling curve (accent) — lies on the guide (rate = log 2).
      plotSeries(Hdoub, mapX, mapY, Palette.accent);
      // Rotation curve (blue) — sub-linear (rate = 0).
      if (state.show_rotation) plotSeries(Hrot, mapX, mapY, Palette.blue);

      // Labels & slopes.
      Canvas.label(ctx, 'H(ξⁿ) vs n', PX0, PY0 - 8, { color: Palette.text, font: '11px monospace' });
      Canvas.label(ctx, 'doubling: slope = log2 ≈ 0.693', mapX(1) + 4, mapY(nMax * LN2) + 14, { color: Palette.accent, font: '9px monospace' });
      if (state.show_rotation)
        Canvas.label(ctx, 'rotation: slope → 0 (h = 0)', mapX(Math.max(2, nMax * 0.4)), mapY(Hrot[Math.min(Hrot.length - 1, (nMax * 0.4) | 0)]) - 6, { color: Palette.blue, font: '9px monospace' });

      // ── right: partition atom visualisation ────────────────────────────
      if (state.show_partition_visual) {
        const nShow = Math.min(8, nMax);
        Canvas.label(ctx, 'Atoms of ξⁿ on [0,1]', RX0, RY0 - 8, { color: Palette.text, font: '11px monospace' });
        const rows = state.show_rotation ? nShow * 2 + 1 : nShow;
        const rowH = Math.min(18, (RY1 - RY0) / rows);
        let y = RY0;
        Canvas.label(ctx, 'doubling', RX0, y + rowH - 4, { color: Palette.accent, font: '9px monospace' });
        y += rowH * 0.7;
        for (let n = 1; n <= nShow; n++) {
          drawAtoms(doublingAtoms(n), RX0, y, RX1 - RX0, rowH - 3, (i, tot) => Palette.heat((i % 8) / 8), n);
          y += rowH;
        }
        if (state.show_rotation) {
          y += 4;
          Canvas.label(ctx, 'rotation', RX0, y + rowH - 4, { color: Palette.blue, font: '9px monospace' });
          y += rowH * 0.7;
          for (let n = 1; n <= nShow; n++) {
            const atoms = rotationAtoms(n, state.alpha);
            // colour by interval length class (three-distance theorem).
            const sizes = atoms.map(a => a[1] - a[0]);
            const smax = Math.max(...sizes), smin = Math.min(...sizes);
            drawAtoms(atoms, RX0, y, RX1 - RX0, rowH - 3,
              (i) => Palette.heat(smax > smin ? (sizes[i] - smin) / (smax - smin) : 0.5), n);
            y += rowH;
          }
        }
      }

      // Header.
      Canvas.label(ctx, meta.title, 8, 16, { color: Palette.blue, font: '12px monospace' });
      Canvas.label(ctx, `α = ${state.alpha.toFixed(4)}   n_max = ${nMax}`, 8, 30, { color: Palette.gold, font: '10px monospace' });
    }

    function plotSeries(H, mapX, mapY, color) {
      const pts = H.map((h, i) => [mapX(i + 1), mapY(h)]);
      Canvas.polyline(ctx, pts, color, 1.8);
      pts.forEach(p => Canvas.dot(ctx, p[0], p[1], 2, color));
    }

    function drawAtoms(atoms, x0, y, w, h, colorFn, n) {
      for (let i = 0; i < atoms.length; i++) {
        const [s, e] = atoms[i];
        const ax = x0 + s * w, aw = Math.max(0.5, (e - s) * w);
        ctx.fillStyle = colorFn(i, atoms.length);
        ctx.fillRect(ax, y, aw, h);
      }
      // hairline boundaries only when atoms are few enough to see.
      if (atoms.length <= 64) {
        ctx.strokeStyle = Palette.bg; ctx.lineWidth = 0.5;
        for (const [s] of atoms) {
          const ax = x0 + s * w;
          ctx.beginPath(); ctx.moveTo(ax, y); ctx.lineTo(ax, y + h); ctx.stroke();
        }
      }
      Canvas.label(ctx, `n=${n}`, x0 + w + 2, y + h, { color: Palette.muted, font: '8px monospace' });
    }

    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'alpha', label: 'α', min: 0.01, max: 0.99, step: 0.001, value: state.alpha, format: v => v.toFixed(3) },
      { type: 'slider', key: 'n_max', label: 'n max', min: 5, max: 22, step: 1, value: state.n_max, format: v => v | 0 },
      { type: 'checkbox', key: 'show_rotation', label: 'rotation', value: state.show_rotation },
      { type: 'checkbox', key: 'show_partition_visual', label: 'partitions', value: state.show_partition_visual },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = (key === 'n_max') ? (val | 0) : val;
      draw();
    });

    function reset() {
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      draw();
    }

    draw();

    return {
      destroy() { container.innerHTML = ''; },
      reset,
      setParam(k, v) { state[k] = (k === 'n_max') ? (v | 0) : v; draw(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = MeasureEntropy;
