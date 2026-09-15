/**
 * @file 01_05_01_cobweb_diagram.js
 * @chapter 1 — Examples and Basic Concepts
 * @sections §1.5, §1.6
 * @concept Orbits, fixed points, periodic orbits, iteration of 1D maps
 *
 * Cobweb (staircase) diagram for a one-dimensional map x_{n+1} = f(x_n).
 *
 * The orbit of x₀ is traced geometrically: from the point (xₙ, xₙ) on the
 * diagonal y = x, move VERTICALLY to the graph y = f(x), landing at
 * (xₙ, f(xₙ)) = (xₙ, x_{n+1}); then move HORIZONTALLY back to the diagonal,
 * landing at (x_{n+1}, x_{n+1}). Repeating produces a "cobweb" whose geometry
 * exposes the orbit's fate:
 *   - staircase into a corner  → convergence to an attracting fixed point
 *   - closed box               → an attracting 2-cycle
 *   - space-filling tangle     → chaos
 *
 * Fixed points are the intersections of y = f(x) with the diagonal y = x;
 * their stability is governed by |f'(x*)| (< 1 attracting, > 1 repelling).
 */

var CobwebDiagram = (() => {
  const { Palette, Canvas, Controls } = DSUtils;

  // ── meta ───────────────────────────────────────────────────────────────
  const meta = {
    id: '01_05_01',
    title: 'Cobweb Diagram — Iterating a 1D Map',
    chapter: 'Chapter 1 — Examples and Basic Concepts',
    sections: ['1.5', '1.6'],
    concept: 'Orbits, fixed points and periodic orbits seen through cobweb iteration',
    description:
      'Interactive cobweb plot for f(x)=x²+c, the logistic map r·x(1-x), and the ' +
      'Gauss map {1/x}. The staircase between y=f(x) and y=x makes convergence to ' +
      'fixed / periodic orbits — or descent into chaos — visually obvious.',
  };

  // ── map definitions ─────────────────────────────────────────────────────
  // Each map exposes its function f(x, p), a domain, a LaTeX-free label, and
  // which slider parameter (if any) drives it.
  const MAPS = {
    quad: {
      label: 'f(x) = x² + c',
      f: (x, p) => x * x + p.c,
      xMin: -2, xMax: 2, yMin: -2, yMax: 2,
      param: 'c',
      eqn: p => `f(x) = x² + ${p.c.toFixed(3)}`,
    },
    logistic: {
      label: 'f(x) = r·x(1−x)',
      f: (x, p) => p.r * x * (1 - x),
      xMin: 0, xMax: 1, yMin: 0, yMax: 1,
      param: 'r',
      eqn: p => `f(x) = ${p.r.toFixed(3)}·x(1−x)`,
    },
    gauss: {
      // Gauss map: fractional part of 1/x, with G(0) := 0 by convention.
      label: 'f(x) = {1/x}',
      f: (x) => (x <= 0 ? 0 : (1 / x) - Math.floor(1 / x)),
      xMin: 0, xMax: 1, yMin: 0, yMax: 1,
      param: null,
      eqn: () => 'f(x) = frac(1/x)',
    },
  };

  const DEFAULTS = { map: 'quad', c: -1.0, r: 3.2, x0: 0.3, n_iter: 80 };

  // ── main init ───────────────────────────────────────────────────────────
  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 440;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Build controls. The parameter slider is generic (c/r); we relabel it as
    // the selected map dictates but keep both values in state independently.
    const ctrl = Controls.build(container, [
      {
        type: 'select', key: 'map', label: 'map', value: state.map,
        options: [
          { value: 'quad', label: 'x² + c' },
          { value: 'logistic', label: 'r·x(1−x)' },
          { value: 'gauss', label: '{1/x} (Gauss)' },
        ],
      },
      { type: 'slider', key: 'c', label: 'c', min: -2, max: 0.25, step: 0.005, value: state.c, format: v => v.toFixed(3) },
      { type: 'slider', key: 'r', label: 'r', min: 0, max: 4, step: 0.005, value: state.r, format: v => v.toFixed(3) },
      { type: 'slider', key: 'x0', label: 'x₀', min: 0.001, max: 0.999, step: 0.001, value: state.x0, format: v => v.toFixed(3) },
      { type: 'slider', key: 'n_iter', label: 'iterations', min: 20, max: 200, step: 1, value: state.n_iter, format: v => v | 0 },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      // Keep x₀ inside the active map's domain when switching maps.
      if (key === 'map') clampX0();
      draw();
    });

    function clampX0() {
      const m = MAPS[state.map];
      state.x0 = Math.max(m.xMin + 1e-4, Math.min(m.xMax - 1e-4, state.x0));
      ctrl.set('x0', state.x0);
    }

    // ── drawing ─────────────────────────────────────────────────────────
    function draw() {
      const m = MAPS[state.map];
      const vp = Canvas.viewport(width, height, m.xMin, m.xMax, m.yMin, m.yMax);
      const p = { c: state.c, r: state.r };

      Canvas.clear(ctx, width, height);
      Canvas.axes(ctx, width, height, m.xMin, m.xMax, m.yMin, m.yMax);

      // Diagonal y = x (dashed, muted).
      ctx.strokeStyle = Palette.muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      const lo = Math.max(m.xMin, m.yMin), hi = Math.min(m.xMax, m.yMax);
      let [dx0, dy0] = vp.toCanvas(lo, lo);
      let [dx1, dy1] = vp.toCanvas(hi, hi);
      ctx.moveTo(dx0, dy0); ctx.lineTo(dx1, dy1); ctx.stroke();
      ctx.setLineDash([]);

      // Graph y = f(x): sample densely; break the polyline where the map
      // jumps (Gauss map has infinitely many discontinuities near 0).
      const N = 1200;
      let seg = [];
      for (let i = 0; i <= N; i++) {
        const x = m.xMin + (i / N) * (m.xMax - m.xMin);
        const y = m.f(x, p);
        if (!isFinite(y)) { flush(seg); seg = []; continue; }
        // Detect a large vertical jump between consecutive samples.
        if (seg.length) {
          const prevY = m.yMin + (1 - seg[seg.length - 1][1] / height) * (m.yMax - m.yMin);
          if (Math.abs(y - prevY) > (m.yMax - m.yMin) * 0.5) { flush(seg); seg = []; }
        }
        seg.push(vp.toCanvas(x, Math.max(m.yMin, Math.min(m.yMax, y))));
      }
      flush(seg);
      function flush(s) { if (s.length > 1) Canvas.polyline(ctx, s, Palette.blue, 2); }

      // Cobweb path. Segment i is coloured by heat(i / n_iter): early = cyan,
      // late = red, so you can watch the orbit "age" toward its limit set.
      let x = state.x0;
      let [px, py] = vp.toCanvas(x, 0);            // start on the x-axis
      const n = state.n_iter | 0;
      for (let i = 0; i < n; i++) {
        const fx = m.f(x, p);
        const col = Palette.heat(i / n);
        // vertical to the graph
        const [vxA, vyA] = vp.toCanvas(x, fx);
        drawSeg(px, py, vxA, vyA, col);
        // horizontal to the diagonal
        const [vxB, vyB] = vp.toCanvas(fx, fx);
        drawSeg(vxA, vyA, vxB, vyB, col);
        px = vxB; py = vyB;
        x = fx;
        if (!isFinite(x)) break;
      }

      function drawSeg(ax, ay, bx, by, col) {
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      }

      // Highlight fixed points: scan the domain for sign changes of f(x)-x.
      let prev = null;
      for (let i = 0; i <= N; i++) {
        const xx = m.xMin + (i / N) * (m.xMax - m.xMin);
        const g = m.f(xx, p) - xx;
        if (!isFinite(g)) { prev = null; continue; }
        if (prev !== null && prev.g * g <= 0 && Math.abs(g - prev.g) < 1) {
          // linear interpolate the crossing
          const t = prev.g / (prev.g - g);
          const xr = prev.x + t * (xx - prev.x);
          const [cx, cy] = vp.toCanvas(xr, xr);
          Canvas.dot(ctx, cx, cy, 4.5, Palette.accent);
        }
        prev = { x: xx, g };
      }

      // x₀ marker on the x-axis (gold).
      const [gx, gy] = vp.toCanvas(state.x0, 0);
      Canvas.dot(ctx, gx, gy, 5, Palette.gold);

      // Canvas labels.
      Canvas.label(ctx, meta.title, 12, 22, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx, m.eqn(p), 12, 42, { color: Palette.text });
      Canvas.label(ctx, `x₀ = ${state.x0.toFixed(3)}   n = ${n}`, 12, 62, { color: Palette.gold });
      Canvas.label(ctx, '● fixed point', width - 12, 22, { color: Palette.accent, align: 'right' });
    }

    function reset() {
      Object.assign(state, DEFAULTS);
      ['map', 'c', 'r', 'x0', 'n_iter'].forEach(k => ctrl.set(k, state[k]));
      // Controls.build does not re-render on set(); rebuild slider labels by
      // dispatching a redraw. (Values persist correctly in state.)
      draw();
    }

    clampX0();
    draw();

    return {
      destroy() { container.innerHTML = ''; },
      reset,
      setParam(k, v) { state[k] = v; if (k === 'map') clampX0(); draw(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = CobwebDiagram;
