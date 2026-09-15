/**
 * @file 07_06_01_bifurcation_diagram.js
 * @chapter 7 — One-Dimensional Dynamics
 * @sections §7.6, §7.7, §7.8
 * @concept Period doubling, bifurcations, Feigenbaum universality (δ ≈ 4.669)
 *
 * Bifurcation diagram of the logistic family
 *
 *     f_r(x) = r·x(1 − x),      r ∈ [0, 4],  x ∈ [0, 1].
 *
 * For each parameter r we discard a long transient and then plot the tail of
 * a single orbit. The set of accumulation points of that orbit — the ω-limit
 * set / attractor — appears as a vertical slice of the diagram:
 *
 *   • r < 1            → 0 is globally attracting (extinction).
 *   • 1 < r < 3        → one attracting fixed point x* = 1 − 1/r.
 *   • 3 < r < 3.449…   → attracting 2-cycle (first period doubling).
 *   • 3.449 < r < 3.544→ 4-cycle, then 8, 16, … doubling ever faster.
 *   • r ≈ 3.5699…      → Feigenbaum point r_∞, onset of chaos.
 *   • r > r_∞          → chaos interlaced with periodic windows (e.g. the
 *                        period-3 window near r ≈ 3.828).
 *
 * FEIGENBAUM UNIVERSALITY. Let r_n be the parameter of the n-th period
 * doubling (period 2^n born). The ratios
 *
 *     δ_n = (r_{n+1} − r_n) / (r_{n+2} − r_{n+1})
 *
 * converge to the Feigenbaum constant δ = 4.669201609… — a number that is the
 * SAME for every smooth unimodal map with a quadratic maximum, not just the
 * logistic one. The diagram is asymptotically self-similar: rescaling r by δ
 * and x by the second Feigenbaum constant α ≈ 2.5029 reproduces the picture.
 *
 * Rendering is done with an accumulation buffer (orbit-point density per
 * pixel) painted through an ImageData for speed, exactly as a long-exposure
 * photograph of the attractor would look: dense = bright.
 */

var BifurcationDiagram = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  // ── meta ───────────────────────────────────────────────────────────────
  const meta = {
    id: '07_06_01',
    title: 'Bifurcation Diagram & Feigenbaum Universality',
    chapter: 'Chapter 7 — One-Dimensional Dynamics',
    sections: ['7.6', '7.7', '7.8'],
    concept: 'Period-doubling cascade of r·x(1−x) and the universal constant δ ≈ 4.669',
    description:
      'High-resolution bifurcation diagram of the logistic map with a live zoom ' +
      'window, Feigenbaum bifurcation markers, the measured δ-ratio table, and an ' +
      'animated r-sweep whose current orbit is drawn in a strip below.',
  };

  // Feigenbaum period-doubling thresholds r_n (period 2^n born at r_n).
  const FEIGENBAUM_R = [3.0, 3.449490, 3.544090, 3.564407];
  const R_INFINITY   = 3.569946;   // accumulation point (onset of chaos)
  const DELTA        = 4.669201609;

  const DEFAULTS = {
    r_min: 2.5, r_max: 4.0,
    zoom_min: 3.4, zoom_max: 4.0,
    resolution: 800,
    transient: 1000, plot: 300,
    feigenbaum: true,
  };

  // Build a 256-entry lookup table mapping heat value → [r,g,b] once, so the
  // per-pixel painting loop never has to parse a CSS colour string.
  const HEAT_LUT = (() => {
    const lut = new Uint8ClampedArray(256 * 3);
    for (let i = 0; i < 256; i++) {
      const css = Palette.heat(i / 255);              // "rgb(r,g,b)"
      const m = /(\d+)\D+(\d+)\D+(\d+)/.exec(css);
      lut[i * 3]     = +m[1];
      lut[i * 3 + 1] = +m[2];
      lut[i * 3 + 2] = +m[3];
    }
    return lut;
  })();

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 480;
    // Vertical layout: main diagram, zoom diagram, orbit strip.
    const MAIN_Y = 0,   MAIN_H = 300;
    const ZOOM_Y = 312, ZOOM_H = 100;
    const STRIP_Y = 424, STRIP_H = 56;

    const { canvas, ctx, width, height } = Canvas.create(container, W, H);

    // Offscreen buffers for the two pixel-rendered panels (ImageData painting
    // bypasses the dpr transform on the main ctx, so we render at CSS size on
    // an offscreen canvas and drawImage it back).
    const mainBuf = makeBuffer(W, MAIN_H);
    const zoomBuf = makeBuffer(W, ZOOM_H);

    function makeBuffer(w, h) {
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      return { canvas: c, ctx: c.getContext('2d'), img: c.getContext('2d').createImageData(w, h) };
    }

    // ── core: render one bifurcation panel into a buffer ──────────────────
    // Density-accumulate 'plot' orbit points per column across [rMin,rMax],
    // then paint with a gamma-compressed heat map (per-column normalisation
    // gives good contrast whether the slice is a point or a chaotic band).
    function renderPanel(buf, w, h, rMin, rMax, cols) {
      const density = new Float32Array(cols * h);
      const colMax  = new Float32Array(cols);
      const transient = state.transient | 0;
      const plot = state.plot | 0;

      for (let c = 0; c < cols; c++) {
        const r = rMin + (rMax - rMin) * (c / (cols - 1));
        let x = 0.5;
        for (let i = 0; i < transient; i++) x = r * x * (1 - x);
        let mx = 0;
        const base = c * h;
        for (let i = 0; i < plot; i++) {
          x = r * x * (1 - x);
          // x∈[0,1] → row, with y increasing upward in math but downward in px.
          let row = (h - 1) - ((x * (h - 1)) | 0);
          if (row < 0) row = 0; else if (row >= h) row = h - 1;
          const d = ++density[base + row];
          if (d > mx) mx = d;
        }
        colMax[c] = mx || 1;
      }

      // Paint: each screen pixel column x samples the nearest data column.
      const data = buf.img.data;
      for (let px = 0; px < w; px++) {
        const c = Math.round((px / (w - 1)) * (cols - 1));
        const cm = colMax[c];
        const base = c * h;
        for (let py = 0; py < h; py++) {
          const idx = (py * w + px) * 4;
          const d = density[base + py];
          if (d <= 0) {
            // background
            data[idx] = 26; data[idx + 1] = 26; data[idx + 2] = 46; data[idx + 3] = 255;
          } else {
            // gamma-compress so faint tails remain visible.
            const t = Math.pow(d / cm, 0.35);
            const li = Math.min(255, (t * 255) | 0);
            data[idx]     = HEAT_LUT[li * 3];
            data[idx + 1] = HEAT_LUT[li * 3 + 1];
            data[idx + 2] = HEAT_LUT[li * 3 + 2];
            data[idx + 3] = 255;
          }
        }
      }
      buf.ctx.putImageData(buf.img, 0, 0);
    }

    // Recompute flags: we only recompute the (expensive) panels when needed.
    function computeMain() { renderPanel(mainBuf, W, MAIN_H, state.r_min, state.r_max, state.resolution | 0); }
    function computeZoom() { renderPanel(zoomBuf, W, ZOOM_H, state.zoom_min, state.zoom_max, state.resolution | 0); }

    // ── sweep animation state ─────────────────────────────────────────────
    let sweepR = null;                 // current cursor r during sweep, or null

    // ── composite draw (cheap; called every frame / interaction) ──────────
    function draw() {
      Canvas.clear(ctx, W, H);

      // Main panel.
      ctx.drawImage(mainBuf.canvas, 0, MAIN_Y, W, MAIN_H);
      // Zoom panel.
      ctx.drawImage(zoomBuf.canvas, 0, ZOOM_Y, W, ZOOM_H);

      // Frame the panels.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(0.5, MAIN_Y + 0.5, W - 1, MAIN_H - 1);
      ctx.strokeRect(0.5, ZOOM_Y + 0.5, W - 1, ZOOM_H - 1);
      ctx.strokeRect(0.5, STRIP_Y + 0.5, W - 1, STRIP_H - 1);

      // Feigenbaum markers on both panels.
      if (state.feigenbaum) {
        drawFeigenbaum(state.r_min, state.r_max, MAIN_Y, MAIN_H);
        drawFeigenbaum(state.zoom_min, state.zoom_max, ZOOM_Y, ZOOM_H);
      }

      // Zoom-window rectangle indicated on the main panel.
      if (state.zoom_min >= state.r_min && state.zoom_max <= state.r_max) {
        const xa = mapR(state.zoom_min, state.r_min, state.r_max);
        const xb = mapR(state.zoom_max, state.r_min, state.r_max);
        ctx.strokeStyle = Palette.blue; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
        ctx.strokeRect(xa, MAIN_Y + 1, Math.max(1, xb - xa), MAIN_H - 2);
        ctx.setLineDash([]);
      }

      // Sweep cursor + orbit strip.
      drawOrbitStrip();
      if (sweepR !== null) {
        if (sweepR >= state.r_min && sweepR <= state.r_max) {
          const x = mapR(sweepR, state.r_min, state.r_max);
          vline(x, MAIN_Y, MAIN_H, Palette.white, 1);
        }
        if (sweepR >= state.zoom_min && sweepR <= state.zoom_max) {
          const x = mapR(sweepR, state.zoom_min, state.zoom_max);
          vline(x, ZOOM_Y, ZOOM_H, Palette.white, 1);
        }
      }

      // Labels & δ table.
      Canvas.label(ctx, meta.title, 10, 18, { color: Palette.blue, font: '12px monospace' });
      Canvas.label(ctx, `f_r(x)=r·x(1−x)   r∈[${state.r_min.toFixed(3)}, ${state.r_max.toFixed(3)}]`,
        10, MAIN_H - 8, { color: Palette.text, font: '11px monospace' });
      Canvas.label(ctx, `zoom  r∈[${state.zoom_min.toFixed(4)}, ${state.zoom_max.toFixed(4)}]`,
        10, ZOOM_Y + 14, { color: Palette.gold, font: '11px monospace' });
      Canvas.label(ctx, sweepR !== null ? `orbit  r = ${sweepR.toFixed(4)}` : 'orbit strip (sweep or drag main panel to zoom)',
        10, STRIP_Y + 14, { color: Palette.muted, font: '10px monospace' });

      drawDeltaTable();
    }

    function drawFeigenbaum(rMin, rMax, y0, h) {
      ctx.setLineDash([4, 4]);
      FEIGENBAUM_R.forEach((r, i) => {
        if (r < rMin || r > rMax) return;
        const x = mapR(r, rMin, rMax);
        ctx.strokeStyle = Palette.gold; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); ctx.stroke();
        if (y0 === MAIN_Y) Canvas.label(ctx, `r${i + 1}`, x + 2, y0 + 16 + i * 12, { color: Palette.gold, font: '9px monospace' });
      });
      // r_∞ marker.
      if (R_INFINITY >= rMin && R_INFINITY <= rMax) {
        const x = mapR(R_INFINITY, rMin, rMax);
        ctx.strokeStyle = Palette.accent; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    function drawDeltaTable() {
      // δ_n = (r_{n+1}−r_n)/(r_{n+2}−r_{n+1}); with 4 points we get 2 ratios.
      const lines = ['δ measured →  4.669…'];
      for (let i = 0; i + 2 < FEIGENBAUM_R.length; i++) {
        const d = (FEIGENBAUM_R[i + 1] - FEIGENBAUM_R[i]) /
                  (FEIGENBAUM_R[i + 2] - FEIGENBAUM_R[i + 1]);
        lines.push(`δ${i + 1} = ${d.toFixed(3)}`);
      }
      const x = W - 8;
      lines.forEach((ln, i) =>
        Canvas.label(ctx, ln, x, 18 + i * 13, { color: i === 0 ? Palette.gold : Palette.text, font: '10px monospace', align: 'right' }));
    }

    function drawOrbitStrip() {
      // Draw the tail of the orbit at the current r (sweep cursor, else the
      // centre of the zoom window) as a connected time series across the strip.
      const r = sweepR !== null ? sweepR : 0.5 * (state.zoom_min + state.zoom_max);
      let x = 0.5;
      for (let i = 0; i < 500; i++) x = r * x * (1 - x);
      const N = 220;
      const pts = [];
      for (let i = 0; i < N; i++) {
        x = r * x * (1 - x);
        const px = (i / (N - 1)) * (W - 4) + 2;
        const py = STRIP_Y + STRIP_H - 3 - x * (STRIP_H - 6);
        pts.push([px, py]);
      }
      Canvas.polyline(ctx, pts, Palette.blue, 1);
      for (let i = 0; i < N; i += 4) Canvas.dot(ctx, pts[i][0], pts[i][1], 0.9, Palette.gold);
    }

    function mapR(r, rMin, rMax) { return ((r - rMin) / (rMax - rMin)) * W; }
    function vline(x, y0, h, color, lw) {
      ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, y0 + h); ctx.stroke();
    }

    // ── drag-to-select zoom window on the main panel ──────────────────────
    let dragging = false, dragStartX = null;
    function evX(e) {
      const rect = canvas.getBoundingClientRect();
      return Math.max(0, Math.min(W, e.clientX - rect.left));
    }
    function evY(e) {
      const rect = canvas.getBoundingClientRect();
      return e.clientY - rect.top;
    }
    const onDown = e => {
      if (evY(e) < MAIN_Y || evY(e) > MAIN_Y + MAIN_H) return;
      dragging = true; dragStartX = evX(e);
    };
    const onMove = e => {
      if (!dragging) return;
      const x0 = dragStartX, x1 = evX(e);
      draw();
      ctx.fillStyle = 'rgba(83,216,251,0.15)';
      ctx.fillRect(Math.min(x0, x1), MAIN_Y, Math.abs(x1 - x0), MAIN_H);
    };
    const onUp = e => {
      if (!dragging) return;
      dragging = false;
      const x1 = evX(e);
      let ra = state.r_min + (Math.min(dragStartX, x1) / W) * (state.r_max - state.r_min);
      let rb = state.r_min + (Math.max(dragStartX, x1) / W) * (state.r_max - state.r_min);
      if (rb - ra < 1e-3) { draw(); return; }              // ignore trivial drags
      state.zoom_min = ra; state.zoom_max = rb;
      ctrl.set('zoom_min', ra); ctrl.set('zoom_max', rb);
      computeZoom(); draw();
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);

    // ── sweep animation ───────────────────────────────────────────────────
    const anim = Anim.loop(dt => {
      if (sweepR === null) sweepR = state.r_min;
      sweepR += (state.r_max - state.r_min) * dt * 0.15;       // ~7 s per full pass
      if (sweepR > state.r_max) sweepR = state.r_min;
      draw();
    });

    // ── controls ──────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'r_min', label: 'r min', min: 0, max: 3, step: 0.01, value: state.r_min, format: v => v.toFixed(2) },
      { type: 'slider', key: 'r_max', label: 'r max', min: 3, max: 4, step: 0.005, value: state.r_max, format: v => v.toFixed(3) },
      { type: 'slider', key: 'zoom_min', label: 'zoom min', min: 3.0, max: 4.0, step: 0.001, value: state.zoom_min, format: v => v.toFixed(3) },
      { type: 'slider', key: 'zoom_max', label: 'zoom max', min: 3.0, max: 4.0, step: 0.001, value: state.zoom_max, format: v => v.toFixed(3) },
      {
        type: 'select', key: 'resolution', label: 'columns', value: String(state.resolution),
        options: [{ value: '400', label: '400' }, { value: '800', label: '800' }, { value: '1200', label: '1200' }],
      },
      { type: 'checkbox', key: 'feigenbaum', label: 'Feigenbaum', value: state.feigenbaum },
      { type: 'button', label: 'Recompute', action: () => { computeMain(); computeZoom(); draw(); } },
      { type: 'button', label: 'Sweep r', action: () => { if (anim.isRunning()) { anim.stop(); sweepR = null; } else anim.start(); draw(); } },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      if (key === 'resolution') { state.resolution = parseInt(val, 10); computeMain(); computeZoom(); }
      else if (key === 'feigenbaum') { state.feigenbaum = val; }
      else {
        state[key] = val;
        if (key === 'r_min' || key === 'r_max') computeMain();
        if (key === 'zoom_min' || key === 'zoom_max') computeZoom();
      }
      draw();
    });

    function reset() {
      anim.stop(); sweepR = null;
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      computeMain(); computeZoom(); draw();
    }

    // Initial render.
    computeMain(); computeZoom(); draw();

    return {
      destroy() {
        anim.stop();
        canvas.removeEventListener('mousedown', onDown);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        container.innerHTML = '';
      },
      reset,
      setParam(k, v) {
        state[k] = (k === 'resolution') ? parseInt(v, 10) : v;
        computeMain(); computeZoom(); draw();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = BifurcationDiagram;
