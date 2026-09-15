/**
 * @file 05_06_01_stable_unstable_manifolds.js
 * @chapter 5 — Hyperbolicity
 * @sections §5.6, §5.8
 * @concept Hyperbolic fixed points, stable/unstable manifolds, homoclinic tangle
 *
 * Stable & Unstable Manifolds of the Hénon map
 *   f(x, y) = (a − x² + b·y,  x)          (default a = 1.4, b = −0.3)
 *
 * MATH
 * ----
 * 1. FIXED POINT.  A fixed point satisfies x = a − x² + b·x and y = x, i.e.
 *        x² + (1 − b)x − a = 0   ⇒   x* = (−(1 − b) + √((1−b)² + 4a)) / 2 .
 *    (The quadratic has a closed form; Newton's method on
 *     F(x,y) = (a − x² + by − x, x − y) converges to the same root. We use the
 *     closed form because it is exact.)  Then p = (x*, x*).
 *
 * 2. LINEARIZATION.  The Jacobian is
 *        Df(x,y) = [ −2x  b ]
 *                  [  1   0 ].
 *    At p its eigenvalues solve λ² + 2x*·λ − b = 0, giving
 *        λ = −x* ± √(x*² + b).
 *    The eigenvector for λ is v = (λ, 1)ᵀ.  |λ| > 1 ⇒ unstable direction E^u;
 *    |λ| < 1 ⇒ stable direction E^s.
 *
 * 3. UNSTABLE MANIFOLD W^u(p).  Seed a short segment through p along E^u and
 *    iterate it FORWARD.  Forward iteration stretches the segment along W^u
 *    while its transverse (stable) error contracts, so the iterates converge
 *    onto the true unstable manifold, folding into ever longer curves.
 *
 * 4. STABLE MANIFOLD W^s(p).  W^s is the unstable manifold of the INVERSE map.
 *    The inverse of (X,Y) = (a − x² + b·y, x) is
 *        f⁻¹(X,Y) = ( Y,  (X − a + Y²)/b ).
 *    Seed along E^s and iterate f⁻¹ (i.e. run the dynamics backwards).
 *
 * 5. The transverse intersections of W^u and W^s are homoclinic points — the
 *    signature of chaos.  The Hénon attractor (grey) is the closure of W^u(p).
 */

var StableUnstableManifolds = (() => {
  const { Palette, Canvas, Controls, Anim, Math2D } = DSUtils;

  const meta = {
    id: '05_06_01',
    title: 'Stable & Unstable Manifolds — Hénon Map',
    chapter: 'Chapter 5 — Hyperbolicity',
    sections: ['5.6', '5.8'],
    concept: 'Stable/unstable manifolds of a hyperbolic fixed point; homoclinic tangle',
    description:
      'Unstable manifold W^u (red) and stable manifold W^s (cyan) of the ' +
      'hyperbolic fixed point of the Hénon map, obtained by iterating short ' +
      'eigen-segments forward and backward. The grey cloud is the attractor.',
  };

  const DEFAULTS = { a: 1.4, b: -0.3, iters: 12 };

  const SEED_N   = 900;    // points along each eigen-segment
  const SEED_L   = 0.55;   // half-length of the seed segment (math units)
  const ATTR_N   = 50000;  // attractor sample size
  const ATTR_SKIP = 500;   // transient discarded before sampling
  const DIVERGE  = 1e3;    // magnitude beyond which an orbit is deemed escaped

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 440;
    const xMin = -2, xMax = 2, yMin = -1.5, yMax = 1.5;
    const { ctx, width, height } = Canvas.create(container, W, H);
    const vp = Canvas.viewport(W, H, xMin, xMax, yMin, yMax);

    // Derived quantities, recomputed whenever a or b changes.
    let geom = null;      // { px, py, lu, ls, vu, vs, ok, msg }
    let attractor = [];   // [x, y] samples

    // ── forward / inverse Hénon maps ─────────────────────────────────────
    const fwd = (x, y) => [state.a - x * x + state.b * y, x];
    const inv = (x, y) => [y, (x - state.a + y * y) / state.b];

    // ── fixed point + eigenstructure ─────────────────────────────────────
    function computeGeometry() {
      const a = state.a, b = state.b;
      const disc = (1 - b) * (1 - b) + 4 * a;
      if (disc < 0) { geom = { ok: false, msg: 'no real fixed point' }; return; }
      const px = (-(1 - b) + Math.sqrt(disc)) / 2;
      const py = px;                                   // fixed point lies on y = x

      const rad = px * px + b;                         // discriminant of eigen-equation
      if (rad < 0) { geom = { ok: false, msg: 'fixed point is a spiral (complex λ)' }; return; }
      const s = Math.sqrt(rad);
      const l1 = -px + s, l2 = -px - s;

      // Sort into unstable (|λ|>1) and stable (|λ|<1).
      let lu, ls;
      if (Math.abs(l1) >= Math.abs(l2)) { lu = l1; ls = l2; } else { lu = l2; ls = l1; }
      if (Math.abs(lu) <= 1 || Math.abs(ls) >= 1) {
        geom = { ok: false, msg: 'fixed point not hyperbolic' }; return;
      }
      const vu = norm([lu, 1]);
      const vs = norm([ls, 1]);
      geom = { ok: true, px, py, lu, ls, vu, vs };
    }

    function norm(v) { const m = Math.hypot(v[0], v[1]); return [v[0] / m, v[1] / m]; }

    // ── attractor sample (closure of W^u) ────────────────────────────────
    function computeAttractor() {
      attractor = [];
      let x = 0, y = 0;
      for (let i = 0; i < ATTR_N + ATTR_SKIP; i++) {
        const [nx, ny] = fwd(x, y);
        x = nx; y = ny;
        if (!isFinite(x) || Math.abs(x) > DIVERGE) break;
        if (i >= ATTR_SKIP) attractor.push([x, y]);
      }
    }

    function recompute() { computeGeometry(); computeAttractor(); }

    // ── seed a segment of SEED_N points through p along direction v ───────
    function seed(dir) {
      const { px, py } = geom;
      const pts = new Array(SEED_N);
      for (let i = 0; i < SEED_N; i++) {
        const t = -SEED_L + (2 * SEED_L) * i / (SEED_N - 1);
        pts[i] = [px + t * dir[0], py + t * dir[1]];
      }
      return pts;
    }

    // Draw a polyline of math points, breaking wherever a point leaves the
    // viewport or the orbit has escaped to infinity.
    function drawBroken(pts, color, lw) {
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let pen = false;
      const inView = (x, y) =>
        isFinite(x) && isFinite(y) &&
        x > xMin - 0.5 && x < xMax + 0.5 && y > yMin - 0.5 && y < yMax + 0.5;
      for (const [x, y] of pts) {
        if (!inView(x, y)) { pen = false; continue; }
        const [cx, cy] = vp.toCanvas(x, y);
        if (!pen) { ctx.moveTo(cx, cy); pen = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }

    // Iterate a seed segment `step`-times, drawing the folded curve each time.
    function drawManifold(dir, step, color) {
      let cur = seed(dir);
      for (let k = 0; k <= state.iters; k++) {
        // Fade earliest generations so the newest (longest) curves dominate.
        ctx.globalAlpha = 0.25 + 0.75 * (k / state.iters);
        drawBroken(cur, color, k === 0 ? 2.2 : 1.1);
        cur = cur.map(([x, y]) => step(x, y));
      }
      ctx.globalAlpha = 1;
    }

    // ── main draw ────────────────────────────────────────────────────────
    function draw() {
      Canvas.clear(ctx, width, height);
      Canvas.axes(ctx, width, height, xMin, xMax, yMin, yMax);

      // Attractor context cloud.
      ctx.fillStyle = 'rgba(136,146,164,0.35)';
      for (const [x, y] of attractor) {
        const [cx, cy] = vp.toCanvas(x, y);
        ctx.fillRect(cx, cy, 1, 1);
      }

      if (!geom || !geom.ok) {
        Canvas.label(ctx, geom ? geom.msg : '…', 16, 26,
          { color: Palette.accent, font: '13px monospace' });
        return;
      }

      // Manifolds: unstable forward (red), stable backward (cyan).
      drawManifold(geom.vs, inv, Palette.blue);
      drawManifold(geom.vu, fwd, Palette.accent);

      // Fixed point.
      const [fx, fy] = vp.toCanvas(geom.px, geom.py);
      Canvas.dot(ctx, fx, fy, 5, Palette.gold);
      Canvas.dot(ctx, fx, fy, 2.2, Palette.bg);

      // Manifold labels near the fixed point, offset along each direction.
      const [ux, uy] = vp.toCanvas(geom.px + 0.9 * geom.vu[0], geom.py + 0.9 * geom.vu[1]);
      const [sx, sy] = vp.toCanvas(geom.px + 0.9 * geom.vs[0], geom.py + 0.9 * geom.vs[1]);
      Canvas.label(ctx, 'Wᵘ', ux, uy, { color: Palette.accent, font: 'bold 13px monospace' });
      Canvas.label(ctx, 'Wˢ', sx, sy, { color: Palette.blue, font: 'bold 13px monospace' });

      // Textual readout.
      Canvas.label(ctx, meta.title, 12, 20, { color: Palette.text, font: '13px monospace' });
      Canvas.label(ctx, `p = (${geom.px.toFixed(3)}, ${geom.py.toFixed(3)})`,
        12, 40, { color: Palette.gold });
      Canvas.label(ctx, `λᵘ = ${geom.lu.toFixed(3)}   λˢ = ${geom.ls.toFixed(3)}`,
        12, 58, { color: Palette.muted });
    }

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'a', label: 'a', min: 0.5, max: 2, step: 0.01,
        value: state.a, format: v => v.toFixed(2) },
      { type: 'slider', key: 'b', label: 'b', min: -1, max: 1, step: 0.01,
        value: state.b, format: v => v.toFixed(2) },
      { type: 'slider', key: 'iters', label: 'manifold iters', min: 5, max: 18, step: 1,
        value: state.iters, format: v => String(v | 0) },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      if (key === 'a' || key === 'b') recompute();   // fixed point / attractor change
      draw();
    });

    function reset() {
      Object.assign(state, DEFAULTS);
      ['a', 'b', 'iters'].forEach(k => ctrl.set(k, state[k]));
      recompute();
      draw();
    }

    recompute();
    draw();

    return {
      destroy() { container.innerHTML = ''; },
      reset,
      setParam(k, v) {
        state[k] = v;
        if (k === 'a' || k === 'b') recompute();
        draw();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = StableUnstableManifolds;
