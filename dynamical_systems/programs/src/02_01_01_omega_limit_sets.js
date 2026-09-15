/**
 * @file 02_01_01_omega_limit_sets.js
 * @chapter 2 — Topological Dynamics
 * @sections §2.1
 * @concept ω-limit sets, recurrence, non-wandering set
 *
 * The ω-limit set of a point x under a flow φ_t is
 *     ω(x) = { y : φ_{t_n}(x) → y for some t_n → ∞ } ,
 * i.e. everywhere the forward orbit accumulates. This visualization integrates
 * many trajectories of a planar flow and shows the TRANSIENT (faint) fading
 * into the ω-limit set (bright) — a fixed point, a limit cycle, or several.
 *
 * Systems (autonomous ẋ = F(x)):
 *   • Van der Pol : ẋ = y,  ẏ = μ(1−x²)y − x         → one stable limit cycle
 *   • Duffing     : ẋ = y,  ẏ = −δy − x(β + αx²)      → competing attractors
 *   • Linear      : ẋ = ax+by, ẏ = cx+dy              → classifiable fixed point
 *
 * Numerically we integrate with RK4 (h = 0.02) for T time units from 20-ish
 * random seeds in the disk of radius 3, drop the first 20% as transient, and
 * mark the last stretch of each orbit — an empirical ω-limit set.
 */

var OmegaLimitSets = (() => {
  const { Palette, Canvas, Controls, Anim, Math2D } = DSUtils;

  // ── meta ───────────────────────────────────────────────────────────────
  const meta = {
    id: '02_01_01',
    title: 'ω-Limit Sets of a Planar Flow',
    chapter: 'Chapter 2 — Topological Dynamics',
    sections: ['2.1'],
    concept: 'Where trajectories accumulate as t→∞: limit cycles and fixed points',
    description:
      'Twenty-ish trajectories of Van der Pol, Duffing, or a linear flow are ' +
      'integrated with RK4. The transient fades (muted) into the ω-limit set ' +
      '(bright dots); for Van der Pol every orbit is drawn onto the same glowing ' +
      'blue limit cycle.',
  };

  // Vector fields, each a function of the parameters → F(t, [x,y]).
  const SYSTEMS = {
    vdp: {
      label: 'Van der Pol',
      field: p => (t, s) => [s[1], p.mu * (1 - s[0] * s[0]) * s[1] - s[0]],
      view: [-4, 4, -6, 6],
      eqn: p => `ẋ=y,  ẏ=${p.mu.toFixed(2)}(1−x²)y−x`,
    },
    duffing: {
      // Unforced Duffing: ẏ = −δy − x(β + αx²). With β<0, α>0 → double well.
      label: 'Duffing',
      field: p => (t, s) => [s[1], -p.delta * s[1] - s[0] * (-1 + 1 * s[0] * s[0])],
      view: [-3, 3, -3, 3],
      eqn: p => `ẋ=y,  ẏ=−${p.delta.toFixed(2)}y−x(x²−1)`,
    },
    linear: {
      // A gentle spiral sink: ẋ = −0.3x − y, ẏ = x − 0.3y.
      label: 'Linear (spiral sink)',
      field: () => (t, s) => [-0.3 * s[0] - s[1], s[0] - 0.3 * s[1]],
      view: [-4, 4, -4, 4],
      eqn: () => `ẋ=−0.3x−y,  ẏ=x−0.3y`,
    },
  };

  const H = 0.02;                    // RK4 step
  const DEFAULTS = { system: 'vdp', mu: 1.5, delta: 0.3, n_orbits: 20, T: 60, playing: true };

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, HH = 440;
    const { ctx, width, height } = Canvas.create(container, W, HH);

    let orbits = [];                 // [{ pts:[[x,y]...], hue }]
    let maxSteps = 0;
    let reveal = 0;                  // how many steps currently shown

    // Deterministic-enough pseudo-random seeds in the disk of radius 3.
    function seeds(n) {
      const out = [];
      for (let k = 0; k < n; k++) {
        const r = 3 * Math.sqrt(Math.random());   // √ for uniform area coverage
        const th = Math.random() * 2 * Math.PI;
        out.push([r * Math.cos(th), r * Math.sin(th)]);
      }
      return out;
    }

    function computeOrbits() {
      const sys = SYSTEMS[state.system];
      const p = { mu: state.mu, delta: state.delta };
      const f = sys.field(p);
      maxSteps = Math.round(state.T / H);
      const n = state.n_orbits | 0;
      orbits = seeds(n).map((s0, idx) => {
        const pts = [s0.slice()];
        let s = s0;
        for (let i = 0; i < maxSteps; i++) {
          s = Math2D.rk4(f, 0, s, H);
          // Guard against blow-up for wild parameter choices.
          if (!isFinite(s[0]) || Math.abs(s[0]) > 1e4) { s = pts[pts.length - 1].slice(); }
          pts.push([s[0], s[1]]);
        }
        return { pts, hue: (idx / n) * 360 };
      });
      reveal = state.playing ? 0 : maxSteps;
    }

    function draw() {
      const sys = SYSTEMS[state.system];
      const [xMin, xMax, yMin, yMax] = sys.view;
      const vp = Canvas.viewport(width, height, xMin, xMax, yMin, yMax);

      Canvas.clear(ctx, width, height);
      Canvas.axes(ctx, width, height, xMin, xMax, yMin, yMax);

      const shown = Math.min(reveal, maxSteps);
      const transientEnd = Math.floor(maxSteps * 0.2);   // first 20% is transient
      const limitStart = Math.max(0, shown - 200);        // last 200 pts ≈ ω-limit

      for (const orb of orbits) {
        const pts = orb.pts;
        // Transient (muted).
        drawRange(pts, 0, Math.min(shown, transientEnd), vp, Palette.muted, 0.9, 0.5);
        // Post-transient body (basin hue).
        const bodyColor = `hsl(${orb.hue}, 70%, 55%)`;
        drawRange(pts, transientEnd, shown, vp, bodyColor, 1.0, 0.9);
      }

      // ω-limit set dots. For Van der Pol they all land on the shared blue
      // limit cycle; otherwise colour by each orbit's basin hue.
      for (const orb of orbits) {
        const pts = orb.pts;
        const col = state.system === 'vdp' ? Palette.blue : `hsl(${orb.hue}, 90%, 65%)`;
        for (let i = limitStart; i < shown; i += 2) {
          const [cx, cy] = vp.toCanvas(pts[i][0], pts[i][1]);
          if (state.system === 'vdp') {
            // Glow: additive translucent dots build a bright cycle.
            ctx.globalAlpha = 0.5;
            Canvas.dot(ctx, cx, cy, 2.2, col);
            ctx.globalAlpha = 1;
          } else {
            Canvas.dot(ctx, cx, cy, 1.6, col);
          }
        }
      }

      // Seed markers (gold) so you can see the disk of initial conditions.
      for (const orb of orbits) {
        const [sx, sy] = vp.toCanvas(orb.pts[0][0], orb.pts[0][1]);
        Canvas.dot(ctx, sx, sy, 2.5, Palette.gold);
      }

      // Labels.
      Canvas.label(ctx, meta.title, 12, 22, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx, `${sys.label}:  ${sys.eqn({ mu: state.mu, delta: state.delta })}`, 12, 42, { color: Palette.text });
      const pct = maxSteps ? Math.round((shown / maxSteps) * 100) : 100;
      Canvas.label(ctx, `orbits ${orbits.length}   T=${state.T}   t: ${pct}%`, 12, 62, { color: Palette.gold });
      Canvas.label(ctx, '● seed   ● ω-limit', width - 12, 22, { color: Palette.muted, align: 'right' });
    }

    // Stroke pts[a..b] as a single polyline in math coords.
    function drawRange(pts, a, b, vp, color, lw, alpha) {
      if (b - a < 2) return;
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.beginPath();
      let [x0, y0] = vp.toCanvas(pts[a][0], pts[a][1]);
      ctx.moveTo(x0, y0);
      for (let i = a + 1; i < b; i++) {
        const [x, y] = vp.toCanvas(pts[i][0], pts[i][1]);
        ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // ── animation: reveal ~500 integration steps per second ───────────────
    const anim = Anim.loop(dt => {
      if (reveal < maxSteps) {
        reveal = Math.min(maxSteps, reveal + Math.max(1, Math.round(500 * dt)));
        draw();
      } else {
        anim.stop();
      }
    });

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      {
        type: 'select', key: 'system', label: 'system', value: state.system,
        options: [
          { value: 'vdp', label: 'Van der Pol' },
          { value: 'duffing', label: 'Duffing' },
          { value: 'linear', label: 'Linear' },
        ],
      },
      { type: 'slider', key: 'mu', label: 'μ (VdP)', min: 0.1, max: 4, step: 0.05, value: state.mu, format: v => v.toFixed(2) },
      { type: 'slider', key: 'delta', label: 'δ (Duffing)', min: 0, max: 1, step: 0.01, value: state.delta, format: v => v.toFixed(2) },
      { type: 'slider', key: 'n_orbits', label: 'orbits', min: 5, max: 30, step: 1, value: state.n_orbits, format: v => v | 0 },
      { type: 'slider', key: 'T', label: 'T', min: 20, max: 150, step: 1, value: state.T, format: v => v | 0 },
      { type: 'button', label: 'Play / Pause', action: () => anim.toggle() },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = (key === 'system') ? val : val;
      computeOrbits();
      if (state.playing) anim.start(); else draw();
    });

    function reset() {
      anim.stop();
      Object.assign(state, DEFAULTS);
      ['system', 'mu', 'delta', 'n_orbits', 'T'].forEach(k => ctrl.set(k, state[k]));
      computeOrbits();
      draw();
      anim.start();
    }

    computeOrbits();
    draw();
    if (state.playing) anim.start();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset,
      setParam(k, v) { state[k] = v; computeOrbits(); draw(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = OmegaLimitSets;
