/**
 * 02_05_02_topological_entropy.js
 * ─────────────────────────────────────────────────────────────────────────
 * Topological Entropy & Bifurcation Diagram  (Brin–Stuck §2.5–§2.6)
 *
 * The logistic map  f_r(x) = r·x·(1−x)  on [0,1] is the canonical example
 * that ties together three notions the chapter develops:
 *
 *   • its ASYMPTOTIC ORBIT STRUCTURE (fixed point → 2-cycle → 4-cycle → … →
 *     chaos), visualised by the bifurcation diagram, and
 *   • its TOPOLOGICAL ENTROPY  h(f_r), the exponential growth rate of the
 *     number of orbit segments distinguishable at resolution ε (the size of
 *     minimal (n,ε)-spanning / (n,ε)-separating sets).
 *
 * For the logistic family h(f_r) = 0 on the periodic-window side of the
 * Feigenbaum point r_∞ ≈ 3.5699 (the accumulation of period doublings), and
 * then rises monotonically to its maximum  h(f_4) = log 2  at r = 4, where
 * f_4 is topologically conjugate to the full one-sided 2-shift and every
 * binary itinerary is realised.
 *
 * TOP panel  (~60% height): bifurcation diagram. For each r we discard 800
 *   transient iterates and scatter the next 300, colouring the column by the
 *   detected period (1 = blue, 2 = cyan, 4 = gold, 8 = orange, chaos = red).
 *   Rendered progressively (column batches per animation frame).
 *
 * BOTTOM panel (~40% height): the analytic entropy curve.  We plot the
 *   normalised approximation
 *        h(r) = log 2 · ln(r / r_∞) / ln(4 / r_∞)   for r ≥ r_∞,   else 0,
 *   which honours the log(r/r_∞) growth form while reaching exactly
 *   h = log 2 at r = 4.  A horizontal reference line marks log 2 = h(f₄),
 *   and the first four period-doubling thresholds r₁..r₄ are dashed.
 * ─────────────────────────────────────────────────────────────────────────
 */

var TopologicalEntropy = (() => {
  const { Palette, Canvas, Controls } = DSUtils;

  const meta = {
    id: '02_05_02_topological_entropy',
    title: 'Bifurcation Diagram & Topological Entropy',
    chapter: 'Chapter 2 · §2.5–§2.6',
    sections: ['2.5', '2.6'],
    concept: 'Topological entropy, exponential orbit complexity',
    description:
      'Logistic-map bifurcation diagram paired with the analytic ' +
      'topological-entropy curve h(f_r), rising from 0 at the Feigenbaum ' +
      'point to log 2 at r = 4.',
  };

  // ── Mathematical constants ────────────────────────────────────────────
  const R_CHAOS = 3.5699456;          // Feigenbaum accumulation point r_∞
  const LOG2 = Math.LN2;              // = h(f_4)
  // First four period-doubling bifurcation thresholds.
  const DOUBLINGS = [
    { r: 3.0,    label: 'r₁' },
    { r: 3.449,  label: 'r₂' },
    { r: 3.544,  label: 'r₃' },
    { r: 3.5644, label: 'r₄' },
  ];

  // ── Canvas geometry ───────────────────────────────────────────────────
  const W = 560, H = 500;
  const TITLE_H = 28;
  //  Top (orbit) panel occupies ~64% of the plotting band, bottom ~36%.
  const TOP = { x0: 46, y0: 38, w: 502, h: 250 };   // bifurcation
  const SEP_Y = 300;                                // 1px divider
  const BOT = { x0: 46, y0: 312, w: 502, h: 140 };  // entropy
  const Y_MAX = 0.75;                               // entropy axis top (> log2)

  /** Normalised analytic entropy approximation (see header). */
  function entropyOf(r) {
    if (r <= R_CHAOS) return 0;
    return LOG2 * Math.log(r / R_CHAOS) / Math.log(4 / R_CHAOS);
  }

  /**
   * Detect the period of a settled orbit by matching values p apart.
   * Returns 0 when no period ≤ 16 fits (treated as chaotic / high period).
   */
  function detectPeriod(orbit) {
    const n = orbit.length;
    const tol = 1e-4;
    for (let p = 1; p <= 16; p++) {
      let ok = true;
      const span = Math.min(p * 4, n - p);
      for (let k = 0; k < span; k++) {
        if (Math.abs(orbit[n - 1 - k] - orbit[n - 1 - k - p]) > tol) { ok = false; break; }
      }
      if (ok) return p;
    }
    return 0;
  }

  /** Map a detected period to a colour. */
  function periodColor(p) {
    switch (p) {
      case 1: return Palette.blue;      // fixed point
      case 2: return '#37e0c8';         // 2-cycle (cyan)
      case 4: return Palette.gold;      // 4-cycle
      case 8: return '#ff8c42';         // 8-cycle (orange)
      case 0: return Palette.accent;    // chaotic band (red)
      default: return Palette.heat(Math.min(1, 0.45 + Math.log2(p) / 6));
    }
  }

  function init(container, params = {}) {
    const { canvas, ctx, width, height } = Canvas.create(container, W, H);

    // Independent state (kept in sync with the sliders).
    const state = {
      rMin: params.rMin != null ? params.rMin : 2.5,
      rMax: params.rMax != null ? params.rMax : 4.0,
      nPix: params.nPix != null ? params.nPix : 500,
    };

    let rafId = null;             // progressive-render handle
    let rangeInputs = null;       // [rMin, rMax, resolution] slider elements

    // Shared x mapping (both panels share the same r-range).
    const rToX = r => BOT.x0 + ((r - state.rMin) / (state.rMax - state.rMin)) * BOT.w;
    const hToY = h => BOT.y0 + (1 - h / Y_MAX) * BOT.h;

    // ── Static layers (title, frames, entropy panel) ────────────────────
    function drawStatic() {
      Canvas.clear(ctx, width, height);

      // Title
      ctx.textAlign = 'center';
      ctx.font = 'bold 14px monospace';
      ctx.fillStyle = Palette.text;
      ctx.fillText('Bifurcation Diagram & Topological Entropy', W / 2, 19);
      ctx.textAlign = 'left';

      // Panel frames
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.strokeRect(TOP.x0 + 0.5, TOP.y0 + 0.5, TOP.w, TOP.h);
      ctx.strokeRect(BOT.x0 + 0.5, BOT.y0 + 0.5, BOT.w, BOT.h);

      // Divider
      ctx.strokeStyle = Palette.muted;
      ctx.beginPath();
      ctx.moveTo(0, SEP_Y + 0.5); ctx.lineTo(W, SEP_Y + 0.5); ctx.stroke();

      // Panel labels
      ctx.font = '11px monospace';
      ctx.fillStyle = Palette.muted;
      Canvas.label(ctx, 'Orbit', TOP.x0 + 6, TOP.y0 + 16, { color: Palette.muted });
      Canvas.label(ctx, 'h(f)', BOT.x0 + 6, BOT.y0 + 16, { color: Palette.muted });

      drawEntropyPanel();
      drawAxesLabels();
    }

    function drawEntropyPanel() {
      // Period-doubling markers (dashed verticals) — only if in view.
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = Palette.muted;
      ctx.font = '10px monospace';
      for (const d of DOUBLINGS) {
        if (d.r < state.rMin || d.r > state.rMax) continue;
        const x = rToX(d.r);
        ctx.beginPath();
        ctx.moveTo(x, BOT.y0); ctx.lineTo(x, BOT.y0 + BOT.h); ctx.stroke();
        ctx.fillStyle = Palette.muted;
        ctx.fillText(d.label, x + 2, BOT.y0 + BOT.h - 4);
      }
      ctx.setLineDash([]);

      // Reference line at log 2.
      const yRef = hToY(LOG2);
      ctx.strokeStyle = Palette.gold;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(BOT.x0, yRef); ctx.lineTo(BOT.x0 + BOT.w, yRef); ctx.stroke();
      ctx.setLineDash([]);
      Canvas.label(ctx, 'log 2 = h(f₄)', BOT.x0 + BOT.w - 96, yRef - 3, { color: Palette.gold });

      // Entropy curve h(r).
      const pts = [];
      const N = 300;
      for (let i = 0; i <= N; i++) {
        const r = state.rMin + (i / N) * (state.rMax - state.rMin);
        pts.push([rToX(r), hToY(entropyOf(r))]);
      }
      Canvas.polyline(ctx, pts, Palette.blue, 2);
    }

    function drawAxesLabels() {
      ctx.font = '10px monospace';
      ctx.fillStyle = Palette.muted;
      ctx.textAlign = 'center';
      // x ticks (r) beneath the bottom panel
      for (let t = 0; t <= 4; t++) {
        const r = state.rMin + (t / 4) * (state.rMax - state.rMin);
        ctx.fillText('r=' + r.toFixed(3), rToX(r), BOT.y0 + BOT.h + 14);
      }
      // y ticks for entropy panel
      ctx.textAlign = 'right';
      for (const h of [0, LOG2]) {
        ctx.fillText(h.toFixed(2), BOT.x0 - 4, hToY(h) + 3);
      }
      // y ticks for orbit panel (x in [0,1])
      for (const xv of [0, 0.5, 1]) {
        ctx.fillText(xv.toFixed(1), TOP.x0 - 4, TOP.y0 + (1 - xv) * TOP.h + 3);
      }
      ctx.textAlign = 'left';
    }

    // ── Progressive bifurcation render ──────────────────────────────────
    function startBifurcation() {
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      const nPix = Math.round(state.nPix);
      const batch = Math.max(4, Math.ceil(nPix / 90));
      let col = 0;

      const step = () => {
        for (let b = 0; b < batch && col < nPix; b++, col++) {
          const r = state.rMin + (col / (nPix - 1)) * (state.rMax - state.rMin);
          const x = rToX(r);

          // Iterate: discard transient, collect orbit.
          let xv = 0.5;
          for (let i = 0; i < 800; i++) xv = r * xv * (1 - xv);
          const orbit = new Array(300);
          for (let i = 0; i < 300; i++) { xv = r * xv * (1 - xv); orbit[i] = xv; }

          // One colour per column (same r ⇒ same period).
          ctx.fillStyle = periodColor(detectPeriod(orbit));
          for (let i = 0; i < 300; i++) {
            const cy = TOP.y0 + (1 - orbit[i]) * TOP.h;
            ctx.fillRect(x, cy, 1, 1);
          }
        }
        if (col < nPix) {
          rafId = requestAnimationFrame(step);
        } else {
          rafId = null;
        }
      };
      rafId = requestAnimationFrame(step);
    }

    function redraw() {
      // keep rMin < rMax
      if (state.rMin >= state.rMax) state.rMin = state.rMax - 0.01;
      drawStatic();
      startBifurcation();
    }

    // ── Controls ────────────────────────────────────────────────────────
    // Set a slider's value and fire its input event (syncs label + state).
    function setSlider(i, v) {
      const inp = rangeInputs[i];
      inp.value = v;
      inp.dispatchEvent(new Event('input'));
    }

    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'rMin', label: 'r min', min: 0, max: 3.5, step: 0.01, value: state.rMin, format: v => v.toFixed(2) },
      { type: 'slider', key: 'rMax', label: 'r max', min: 3.5, max: 4, step: 0.005, value: state.rMax, format: v => v.toFixed(3) },
      { type: 'slider', key: 'nPix', label: 'resolution', min: 200, max: 800, step: 20, value: state.nPix, format: v => v + 'px' },
      { type: 'button', label: 'Zoom in',  action: () => { setSlider(0, 3.4); setSlider(1, 4); } },
      { type: 'button', label: 'Zoom out', action: () => { setSlider(0, 0);   setSlider(1, 4); } },
      { type: 'button', label: 'Reset',    action: () => { setSlider(2, 500); setSlider(0, 2.5); setSlider(1, 4); } },
    ]);

    // Grab the actual <input type=range> elements (order: rMin, rMax, nPix).
    const panel = container.lastChild;
    rangeInputs = panel.querySelectorAll('input[type=range]');

    ctrl.onChange((key, val) => {
      state[key] = val;
      redraw();
    });

    // Initial paint
    redraw();

    return {
      destroy() {
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
        container.innerHTML = '';
      },
      reset() {
        setSlider(2, 500); setSlider(0, 2.5); setSlider(1, 4);
      },
      setParam(key, val) {
        const idx = { rMin: 0, rMax: 1, nPix: 2 }[key];
        if (idx != null) setSlider(idx, val);
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = TopologicalEntropy;
