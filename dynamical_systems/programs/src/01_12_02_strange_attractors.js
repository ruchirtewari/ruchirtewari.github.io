/**
 * @file 01_12_02_strange_attractors.js
 * @chapter 1 — Examples and Basic Concepts
 * @sections §1.12, §1.13
 * @concept Strange attractors, sensitive dependence on initial conditions
 *
 * Side-by-side gallery of two archetypal strange attractors.
 *
 * LEFT — Hénon map (discrete, 2D):
 *     (x, y) → (a − x² + b·y,  x)
 *   Classic parameters a = 1.4, b = −0.3 produce a fractal attractor: an
 *   invariant set with the local structure of a Cantor set × interval.
 *
 * RIGHT — Lorenz system (continuous, 3D), projected to the (x, z) plane:
 *     ẋ = σ(y − x),  ẏ = x(ρ − z) − y,  ż = xy − βz
 *   with σ = 10, ρ = 28, β = 8/3 — the original "butterfly". We integrate
 *   with RK4 and overlay a SECOND orbit started a distance ε = 1e-5 away to
 *   make sensitive dependence (positive Lyapunov exponent) visible: the two
 *   trajectories track together, then peel apart onto different wings.
 */

var StrangeAttractors = (() => {
  const { Palette, Canvas, Controls, Anim, Math2D } = DSUtils;

  // ── meta ───────────────────────────────────────────────────────────────
  const meta = {
    id: '01_12_02',
    title: 'Strange Attractors — Hénon & Lorenz',
    chapter: 'Chapter 1 — Examples and Basic Concepts',
    sections: ['1.12', '1.13'],
    concept: 'Fractal attractors and sensitive dependence on initial conditions',
    description:
      'The Hénon map attractor (left) and the Lorenz butterfly (right, projected ' +
      'to the x–z plane). A second Lorenz orbit displaced by ε=1e-5 diverges ' +
      'exponentially, illustrating chaos.',
  };

  // ── defaults ────────────────────────────────────────────────────────────
  const DEFAULTS = {
    a: 1.4, b: -0.3,               // Hénon
    sigma: 10, rho: 28, beta: 8 / 3, // Lorenz
    animate: false,
  };

  const HENON_N = 50000, HENON_TRANSIENT = 1000;
  const LORENZ_STEPS = 15000, LORENZ_H = 0.01;
  const EPS = 1e-5;               // perturbation for the twin Lorenz orbit

  // Lorenz vector field for Math2D.rk4 (t is unused — autonomous system).
  function lorenzField(p) {
    return (t, s) => {
      const [x, y, z] = s;
      return [p.sigma * (y - x), x * (p.rho - z) - y, x * y - p.beta * z];
    };
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 420;
    const MID = W / 2;                 // vertical divider between panels
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Precomputed orbits (recomputed when params change).
    let henonPts = [];                 // [x, y]
    let lorenzA = [], lorenzB = [];    // [x, y, z]
    let animIdx = 0;                   // how many Lorenz steps to reveal

    // ── orbit generation ─────────────────────────────────────────────────
    function computeHenon() {
      henonPts = [];
      let x = 0, y = 0;
      const a = state.a, b = state.b;
      for (let i = 0; i < HENON_N + HENON_TRANSIENT; i++) {
        const xn = a - x * x + b * y;
        const yn = x;
        x = xn; y = yn;
        // Diverged (params outside the bounded regime) — bail out cleanly.
        if (!isFinite(x) || Math.abs(x) > 1e6) break;
        if (i >= HENON_TRANSIENT) henonPts.push([x, y]);
      }
    }

    function computeLorenz() {
      lorenzA = []; lorenzB = [];
      const f = lorenzField(state);
      let sA = [1, 1, 1];
      let sB = [1 + EPS, 1, 1];        // ε-displaced twin
      lorenzA.push(sA); lorenzB.push(sB);
      for (let i = 0; i < LORENZ_STEPS; i++) {
        sA = Math2D.rk4(f, 0, sA, LORENZ_H);
        sB = Math2D.rk4(f, 0, sB, LORENZ_H);
        lorenzA.push(sA); lorenzB.push(sB);
        if (!isFinite(sA[0])) break;
      }
      animIdx = state.animate ? 0 : lorenzA.length;
    }

    function recompute() {
      computeHenon();
      computeLorenz();
    }

    // ── drawing ──────────────────────────────────────────────────────────
    function draw() {
      Canvas.clear(ctx, width, height);

      drawHenon();
      drawLorenz();

      // Divider.
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(MID, 0); ctx.lineTo(MID, height); ctx.stroke();

      // Panel titles & params.
      Canvas.label(ctx, 'Hénon map', 12, 20, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx, `(x,y)→(a−x²+by, x)`, 12, 38, { color: Palette.muted });
      Canvas.label(ctx, `a=${state.a.toFixed(2)}  b=${state.b.toFixed(2)}`, 12, 56, { color: Palette.gold });

      Canvas.label(ctx, 'Lorenz (x–z)', MID + 12, 20, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx, `ρ=${state.rho.toFixed(1)}  σ=${state.sigma.toFixed(0)}  β=8/3`, MID + 12, 38, { color: Palette.gold });
      Canvas.label(ctx, `twin ε=${EPS}`, MID + 12, 56, { color: Palette.accent });
    }

    function drawHenon() {
      if (!henonPts.length) return;
      // Auto-fit bounds with a small margin.
      let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
      for (const [x, y] of henonPts) {
        if (x < xMin) xMin = x; if (x > xMax) xMax = x;
        if (y < yMin) yMin = y; if (y > yMax) yMax = y;
      }
      const mx = (xMax - xMin) * 0.08 + 1e-6, my = (yMax - yMin) * 0.08 + 1e-6;
      xMin -= mx; xMax += mx; yMin -= my; yMax += my;

      const panelW = MID;
      const sx = x => ((x - xMin) / (xMax - xMin)) * panelW;
      const sy = y => height - ((y - yMin) / (yMax - yMin)) * height;

      const N = henonPts.length;
      for (let i = 0; i < N; i++) {
        const [x, y] = henonPts[i];
        ctx.fillStyle = Palette.heat(i / N);
        ctx.fillRect(sx(x), sy(y), 1, 1);
      }
    }

    function drawLorenz() {
      if (!lorenzA.length) return;
      // Fixed viewport tuned for the classic butterfly in the x–z plane.
      const xMin = -25, xMax = 25, zMin = 0, zMax = 52;
      const panelW = W - MID;
      const sx = x => MID + ((x - xMin) / (xMax - xMin)) * panelW;
      const sz = z => height - ((z - zMin) / (zMax - zMin)) * height;

      const shown = Math.min(animIdx, lorenzA.length);

      // Primary orbit, coloured by time (cyan → red).
      ctx.lineWidth = 0.8;
      for (let i = 1; i < shown; i++) {
        const p0 = lorenzA[i - 1], p1 = lorenzA[i];
        ctx.strokeStyle = Palette.heat(i / lorenzA.length);
        ctx.beginPath();
        ctx.moveTo(sx(p0[0]), sz(p0[2]));
        ctx.lineTo(sx(p1[0]), sz(p1[2]));
        ctx.stroke();
      }

      // Twin orbit in accent red — thin overlay showing divergence.
      ctx.strokeStyle = Palette.accent;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < shown; i++) {
        const p = lorenzB[i];
        const cx = sx(p[0]), cy = sz(p[2]);
        if (!started) { ctx.moveTo(cx, cy); started = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Leading heads.
      if (shown > 0) {
        const ha = lorenzA[shown - 1], hb = lorenzB[shown - 1];
        Canvas.dot(ctx, sx(ha[0]), sz(ha[2]), 3, Palette.white);
        Canvas.dot(ctx, sx(hb[0]), sz(hb[2]), 3, Palette.accent);
      }
    }

    // ── animation ────────────────────────────────────────────────────────
    const anim = Anim.loop(() => {
      if (animIdx < lorenzA.length) {
        animIdx = Math.min(lorenzA.length, animIdx + 120); // ~reveal speed
        draw();
      } else {
        anim.stop();
      }
    });

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'a', label: 'Hénon a', min: -2, max: 2, step: 0.01, value: state.a, format: v => v.toFixed(2) },
      { type: 'slider', key: 'b', label: 'Hénon b', min: -1, max: 1, step: 0.01, value: state.b, format: v => v.toFixed(2) },
      { type: 'slider', key: 'rho', label: 'Lorenz ρ', min: 0.1, max: 50, step: 0.1, value: state.rho, format: v => v.toFixed(1) },
      { type: 'checkbox', key: 'animate', label: 'animate Lorenz', value: state.animate },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      if (key === 'animate') {
        if (val) { animIdx = 0; anim.start(); }
        else { anim.stop(); animIdx = lorenzA.length; draw(); }
        return;
      }
      // A param changed → recompute the relevant orbit(s).
      recompute();
      if (state.animate) anim.start(); else draw();
    });

    function reset() {
      anim.stop();
      Object.assign(state, DEFAULTS);
      ['a', 'b', 'rho', 'animate'].forEach(k => ctrl.set(k, state[k]));
      recompute();
      draw();
    }

    recompute();
    if (state.animate) anim.start(); else draw();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset,
      setParam(k, v) {
        state[k] = v;
        if (k === 'animate') { v ? (animIdx = 0, anim.start()) : (anim.stop(), animIdx = lorenzA.length); }
        else recompute();
        draw();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = StrangeAttractors;
