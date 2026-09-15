/**
 * @file 04_03_01_birkhoff_ergodic_theorem.js
 * @chapter 4 — Ergodic Theory
 * @sections §4.3
 * @concept Birkhoff's ergodic theorem; equidistribution (Weyl)
 *
 * BIRKHOFF ERGODIC THEOREM — TIME AVERAGE vs SPACE AVERAGE
 *
 * For a measure-preserving map T and an integrable observable f, Birkhoff's
 * pointwise ergodic theorem guarantees that the TIME AVERAGE
 *
 *     A_n(x) = (1/n) Σ_{i=0}^{n-1} f(Tⁱ x)
 *
 * converges for μ-almost-every x to a limit f*(x). When the system is ERGODIC,
 * f* is constant and equals the SPACE AVERAGE:
 *
 *     lim_{n→∞} A_n(x) = ∫ f dμ         (for a.e. x, if T is ergodic).
 *
 * The circle rotation R_α(x) = x + α (mod 1) is the cleanest laboratory:
 *
 *   • α IRRATIONAL  ⇒  R_α is UNIQUELY ERGODIC (Weyl equidistribution). Every
 *     starting point gives the SAME limit — the flat space average. All the
 *     time-average curves collapse onto the dashed red line ∫f dλ.
 *
 *   • α = p/q RATIONAL  ⇒  R_α is NOT ergodic. Every orbit is periodic with
 *     period q, so A_n(x) → (1/q)Σ f(x + kp/q): the limit DEPENDS on x. The
 *     curves fan out to different plateaus — one per orbit.
 *
 * Observables and their Lebesgue space averages on [0,1):
 *     f(x) = sin(2πx)        ⇒  ∫f dλ = 0
 *     f(x) = x²              ⇒  ∫f dλ = 1/3
 *     f(x) = 1_[0,1/2)(x)    ⇒  ∫f dλ = 1/2
 *
 * The RIGHT panel shows Weyl equidistribution directly: the points {nα} for
 * n = 1..N. For irrational α they fill [0,1) uniformly; for rational α only q
 * distinct heights appear — the geometric reason ergodicity fails.
 */

var BirkhoffErgodic = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '04_03_01',
    title: 'Birkhoff Ergodic Theorem — Time Average vs Space Average',
    chapter: 'Chapter 4 — Ergodic Theory',
    sections: ['4.3'],
    concept: 'Time averages converge to the space average iff the system is ergodic',
    description:
      'Running time averages A_n(x) for several orbits of a circle rotation. For ' +
      'irrational α every curve converges to the flat space average ∫f dλ; for a ' +
      'rational overlay they split into orbit-dependent plateaus. A side scatter of ' +
      '{nα} shows Weyl equidistribution vs finitely many rational points.',
  };

  // Observables with their exact Lebesgue averages and a sensible plot range.
  const OBS = {
    sin: { f: x => Math.sin(2 * Math.PI * x), avg: 0,   yMin: -0.7, yMax: 0.7, label: 'sin(2πx)' },
    sq:  { f: x => x * x,                     avg: 1/3, yMin: 0,    yMax: 1,   label: 'x²'       },
    ind: { f: x => (x < 0.5 ? 1 : 0),         avg: 0.5, yMin: 0,    yMax: 1,   label: '1_[0,½)'  },
  };

  // Non-ergodic rotation numbers p/q → orbits are periodic with period q.
  const RATS = { '1/2': [1, 2], '1/3': [1, 3], '2/5': [2, 5] };

  const DEFAULTS = {
    alpha: (Math.sqrt(5) - 1) / 2,   // golden ratio conjugate — "most irrational"
    pq: '1/3',
    obs: 'sin',
    n_max: 3000,
    n_orbits: 5,
    show_nonergodic: false,
  };

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 420;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Left panel (time averages) and right panel (equidistribution scatter).
    const L = { px: 45,  py: 55, pw: 290, ph: 320 };
    const R = { px: 380, py: 55, pw: 155, ph: 320 };
    const N_SCATTER_CAP = 1000;

    // ── simulation state (advanced incrementally by the animation) ────────
    let ergOrbits, ratOrbits, scErg, scRat, nCur, logNmax, obs;

    function x0Of(j, m) { return (j + 1) / (m + 1); }        // spread of initial conditions

    function makeOrbit(x0) { return { x: x0, sum: 0, lastPx: -1e9, pts: [] }; }

    function resetSim() {
      obs = OBS[state.obs];
      const m = state.n_orbits | 0;
      ergOrbits = Array.from({ length: m }, (_, j) => makeOrbit(x0Of(j, m)));
      ratOrbits = Array.from({ length: m }, (_, j) => makeOrbit(x0Of(j, m)));
      scErg = [];               // {nα} for irrational α
      scRat = [];               // {n·p/q} for the rational overlay
      nCur = 0;
      logNmax = Math.log(state.n_max | 0);
    }

    // Advance every orbit by one Birkhoff step at global index n.
    // A_n uses f over x_0..x_{n-1}, so we accumulate f(x) BEFORE rotating.
    function advanceOne(n, alpha, rat) {
      const px = L.px + (Math.log(n) / logNmax) * L.pw;   // log-x pixel for this n

      for (const o of ergOrbits) {
        o.sum += obs.f(o.x);
        const A = o.sum / n;
        o.x += alpha; if (o.x >= 1) o.x -= 1;             // R_α
        if (n === 1 || px - o.lastPx >= 1) { o.pts.push([px, A]); o.lastPx = px; }
      }
      if (state.show_nonergodic) {
        const rr = rat[0] / rat[1];
        for (const o of ratOrbits) {
          o.sum += obs.f(o.x);
          const A = o.sum / n;
          o.x += rr; if (o.x >= 1) o.x -= 1;              // R_{p/q}
          if (n === 1 || px - o.lastPx >= 1) { o.pts.push([px, A]); o.lastPx = px; }
        }
      }

      // Equidistribution scatter (capped for legibility).
      if (n <= N_SCATTER_CAP) {
        scErg.push([n, (n * alpha) % 1]);
        if (state.show_nonergodic) scRat.push([n, (n * (rat[0] / rat[1])) % 1]);
      }
    }

    // ── drawing ──────────────────────────────────────────────────────────
    function yPix(A) { return L.py + L.ph - ((A - obs.yMin) / (obs.yMax - obs.yMin)) * L.ph; }

    function drawLeft() {
      // frame + axes
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(L.px, L.py, L.pw, L.ph);

      // space average: dashed red horizontal line at ∫f dλ
      const ay = yPix(obs.avg);
      ctx.strokeStyle = Palette.accent; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(L.px, ay); ctx.lineTo(L.px + L.pw, ay); ctx.stroke();
      ctx.setLineDash([]);
      Canvas.label(ctx, `∫f dλ = ${fmtAvg(obs.avg)}`, L.px + L.pw - 4, ay - 4,
        { color: Palette.accent, font: '10px monospace', align: 'right' });

      // non-ergodic (rational) curves first, muted gold dashed
      if (state.show_nonergodic) {
        ctx.setLineDash([3, 3]);
        for (const o of ratOrbits) Canvas.polyline(ctx, o.pts, Palette.gold, 1);
        ctx.setLineDash([]);
      }

      // ergodic curves in a heat gradient
      const m = ergOrbits.length;
      ergOrbits.forEach((o, j) => {
        const col = Palette.heat(m > 1 ? j / (m - 1) : 0.5);
        Canvas.polyline(ctx, o.pts, col, 1.5);
      });

      // label final values at the right edge (thinned to avoid clutter)
      const stride = Math.ceil(m / 5);
      ergOrbits.forEach((o, j) => {
        if (!o.pts.length || j % stride !== 0) return;
        const last = o.pts[o.pts.length - 1];
        Canvas.label(ctx, last[1].toFixed(3), L.px + L.pw + 2, yPix(last[1]) + 3,
          { color: Palette.heat(m > 1 ? j / (m - 1) : 0.5), font: '9px monospace' });
      });

      // axis annotations
      Canvas.label(ctx, 'A_n(x)  (time average)', L.px, L.py - 8,
        { color: Palette.blue, font: '11px monospace' });
      Canvas.label(ctx, obs.yMax.toFixed(2), L.px - 4, L.py + 8,
        { color: Palette.muted, font: '9px monospace', align: 'right' });
      Canvas.label(ctx, obs.yMin.toFixed(2), L.px - 4, L.py + L.ph,
        { color: Palette.muted, font: '9px monospace', align: 'right' });
      Canvas.label(ctx, 'n = 1', L.px, L.py + L.ph + 13,
        { color: Palette.muted, font: '9px monospace' });
      Canvas.label(ctx, `n = ${state.n_max | 0}  (log)`, L.px + L.pw, L.py + L.ph + 13,
        { color: Palette.muted, font: '9px monospace', align: 'right' });
    }

    function drawRight() {
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(R.px, R.py, R.pw, R.ph);

      const cap = Math.min(state.n_max | 0, N_SCATTER_CAP);
      const sx = n => R.px + (Math.min(n, cap) / cap) * R.pw;
      const sy = v => R.py + R.ph - v * R.ph;

      // rational scatter (few distinct heights) under the ergodic one
      if (state.show_nonergodic) {
        ctx.fillStyle = Palette.accent;
        for (const [n, v] of scRat) { ctx.globalAlpha = 0.55; Canvas.dot(ctx, sx(n), sy(v), 1.6, Palette.accent); }
        ctx.globalAlpha = 1;
      }
      // irrational scatter — fills [0,1) uniformly (Weyl)
      for (const [n, v] of scErg) { Canvas.dot(ctx, sx(n), sy(v), 1.1, Palette.blue); }

      Canvas.label(ctx, '{nα} equidistribution', R.px, R.py - 8,
        { color: Palette.blue, font: '11px monospace' });
      Canvas.label(ctx, '1', R.px - 4, R.py + 8, { color: Palette.muted, font: '9px monospace', align: 'right' });
      Canvas.label(ctx, '0', R.px - 4, R.py + R.ph, { color: Palette.muted, font: '9px monospace', align: 'right' });
      Canvas.label(ctx, `n → ${cap}`, R.px + R.pw, R.py + R.ph + 13,
        { color: Palette.muted, font: '9px monospace', align: 'right' });
    }

    function fmtAvg(v) {
      if (Math.abs(v - 1 / 3) < 1e-9) return '1/3';
      if (Math.abs(v - 0.5) < 1e-9) return '1/2';
      return v.toFixed(3);
    }

    function draw() {
      Canvas.clear(ctx, width, height);
      Canvas.label(ctx, meta.title, 16, 24, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx,
        `f(x) = ${obs.label}     α = ${state.alpha.toFixed(6)}…` +
        (state.show_nonergodic ? `     p/q = ${state.pq}` : ''),
        16, 42, { color: Palette.text, font: '11px monospace' });

      drawLeft();
      drawRight();

      const status = nCur >= (state.n_max | 0) ? `converged (n = ${state.n_max | 0})` : `n = ${nCur}`;
      Canvas.label(ctx, status, width - 16, 24,
        { color: Palette.muted, font: '11px monospace', align: 'right' });
    }

    // ── animation: reveal convergence step by step ───────────────────────
    const anim = Anim.loop(() => {
      const nmax = state.n_max | 0;
      if (nCur >= nmax) { anim.stop(); return; }
      const rat = RATS[state.pq];
      const batch = Math.max(1, Math.ceil(nmax / 240));
      const target = Math.min(nmax, nCur + batch);
      for (let n = nCur + 1; n <= target; n++) advanceOne(n, state.alpha, rat);
      nCur = target;
      draw();
      if (nCur >= nmax) anim.stop();
    });

    function recompute() {
      resetSim();
      draw();
      anim.stop();
      anim.start();
    }

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'alpha', label: 'α (ergodic)', min: 0.01, max: 0.99, step: 0.0001,
        value: state.alpha, format: v => v.toFixed(5) },
      { type: 'select', key: 'pq', label: 'p/q (non-erg.)', value: state.pq,
        options: Object.keys(RATS).map(k => ({ value: k, label: k })) },
      { type: 'select', key: 'obs', label: 'f(x)', value: state.obs,
        options: [
          { value: 'sin', label: 'sin(2πx)' },
          { value: 'sq',  label: 'x²' },
          { value: 'ind', label: '1_[0,½)' },
        ] },
      { type: 'slider', key: 'n_max', label: 'n_max', min: 100, max: 10000, step: 100,
        value: state.n_max, format: v => v | 0 },
      { type: 'slider', key: 'n_orbits', label: 'orbits', min: 3, max: 10, step: 1,
        value: state.n_orbits, format: v => v | 0 },
      { type: 'checkbox', key: 'show_nonergodic', label: 'show non-ergodic (rational)',
        value: state.show_nonergodic },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      recompute();
    });

    recompute();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset() { recompute(); },
      setParam(k, v) { state[k] = v; recompute(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = BirkhoffErgodic;
