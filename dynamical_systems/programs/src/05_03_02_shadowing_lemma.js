/**
 * @file 05_03_02_shadowing_lemma.js
 * @chapter 5 — Hyperbolicity
 * @sections §5.3
 * @concept δ-pseudo-orbits, ε-shadowing, numerical reliability of chaos
 *
 * Shadowing Lemma — Pseudo-orbit vs True Orbit (Arnold's cat map)
 *
 * THE THEOREM.  In a hyperbolic (Anosov) system, for every ε > 0 there is a
 * δ > 0 such that every δ-pseudo-orbit is ε-shadowed by a genuine orbit:
 * a small per-step error never accumulates into a large global error, because
 * the expanding and contracting directions each absorb their share of the
 * perturbation. This is why numerically computed chaotic orbits — full of
 * round-off "noise" — still reflect true dynamics.
 *
 * THE MAP.  Arnold's cat map on the torus T² = [0,1)²:
 *        f(x, y) = (2x + y,  x + y)  (mod 1),   A = [[2,1],[1,1]].
 * Its eigenvalues are λ± = (3 ± √5)/2, with λ⁺ ≈ 2.618 (expanding) and
 * λ⁻ ≈ 0.382 (contracting) — hyperbolic, so shadowing applies.
 *
 * WHAT WE DO.
 *   • δ-pseudo-orbit:  x_{n+1} = (A·x_n mod 1) + η_n,  η_n ∈ [−δ, δ]² uniform.
 *   • true (shadowing) orbit:  the exact orbit iterated from the same x_0.
 *   • three columns δ ∈ {0.001, 0.01, 0.1} shown together, each with a torus
 *     plot (true = solid cyan, pseudo = dashed grey, and red when it escapes
 *     the ε-tube) and a plot of ‖x_n^pseudo − x_n^true‖ below.
 *
 * THE CONSTANT.  The shadowing error is bounded by ‖error‖ ≤ C·δ with a
 * constant governed by the eigenvalues: the expanding error is summed by a
 * geometric series with ratio 1/λ⁺ and the contracting one with ratio λ⁻,
 * giving  C = 1/(λ⁺ − 1) + 1/(1 − λ⁻) = √5 ≈ 2.236.
 */

var ShadowingLemma = (() => {
  const { Palette, Canvas, Controls } = DSUtils;

  const meta = {
    id: '05_03_02',
    title: 'Shadowing Lemma — Pseudo-orbit vs True Orbit',
    chapter: 'Chapter 5 — Hyperbolicity',
    sections: ['5.3'],
    concept: 'δ-pseudo-orbits are ε-shadowed by true orbits in hyperbolic systems',
    description:
      'A δ-pseudo-orbit of the cat map (dashed grey) and the true orbit that ' +
      'shadows it (cyan), for δ = 0.001, 0.01, 0.1. Below each: the shadowing ' +
      'error ‖xₙᵖ − xₙᵗ‖ against n, bounded by C·δ with C = √5.',
  };

  const DELTAS = [0.001, 0.01, 0.1];
  const DEFAULTS = { n_steps: 40 };

  // Cat-map eigenvalues and shadowing constant.
  const LP = (3 + Math.sqrt(5)) / 2;         // λ⁺ ≈ 2.618
  const LM = (3 - Math.sqrt(5)) / 2;         // λ⁻ ≈ 0.382
  const C_SHADOW = 1 / (LP - 1) + 1 / (1 - LM);   // = √5 ≈ 2.236

  const X0 = [0.30, 0.42];                   // shared initial condition

  const cat = (x, y) => [
    Math2Dmod(2 * x + y, 1),
    Math2Dmod(x + y, 1),
  ];
  function Math2Dmod(a, n) { return ((a % n) + n) % n; }

  // Toroidal distance between two points of T².
  function torusDist(p, q) {
    const dx = Math.abs(p[0] - q[0]), dy = Math.abs(p[1] - q[1]);
    const ex = Math.min(dx, 1 - dx), ey = Math.min(dy, 1 - dy);
    return Math.hypot(ex, ey);
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 420;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Per-column noise, regenerated on demand. noise[c][n] = [ηx, ηy].
    let noise = [];
    function regenNoise() {
      noise = DELTAS.map(() => {
        const seq = [];
        for (let n = 0; n < 200; n++) seq.push([Math.random() * 2 - 1, Math.random() * 2 - 1]);
        return seq;
      });
    }

    // Build the true and pseudo orbits for a given column.
    function orbits(col) {
      const delta = DELTAS[col], N = state.n_steps;
      const trueO = [X0.slice()], pseudo = [X0.slice()];
      for (let n = 0; n < N; n++) {
        // true orbit — exact iteration
        trueO.push(cat(trueO[n][0], trueO[n][1]));
        // pseudo orbit — iterate then add δ-noise, wrap back to the torus
        const step = cat(pseudo[n][0], pseudo[n][1]);
        const eta = noise[col][n];
        pseudo.push([
          Math2Dmod(step[0] + delta * eta[0], 1),
          Math2Dmod(step[1] + delta * eta[1], 1),
        ]);
      }
      const err = trueO.map((t, n) => torusDist(t, pseudo[n]));
      return { trueO, pseudo, err, delta };
    }

    // Draw a torus trajectory, splitting the polyline at boundary wraps.
    function drawTorus(pts, ox, oy, size, color, dashed) {
      const sx = x => ox + x * size;
      const sy = y => oy + (1 - y) * size;
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 1 : 1.4;
      ctx.setLineDash(dashed ? [3, 3] : []);
      ctx.beginPath();
      let pen = false;
      for (let i = 0; i < pts.length; i++) {
        if (i > 0) {
          // A jump > 0.5 in either coordinate means the orbit wrapped — break.
          const dx = Math.abs(pts[i][0] - pts[i - 1][0]);
          const dy = Math.abs(pts[i][1] - pts[i - 1][1]);
          if (dx > 0.5 || dy > 0.5) pen = false;
        }
        const cx = sx(pts[i][0]), cy = sy(pts[i][1]);
        if (!pen) { ctx.moveTo(cx, cy); pen = true; } else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    function draw() {
      Canvas.clear(ctx, width, height);

      const colW = W / 3;
      const torusSize = 130;
      const torusTop = 46;
      const plotTop = torusTop + torusSize + 34;
      const plotH = 96;

      for (let col = 0; col < 3; col++) {
        const { trueO, pseudo, err, delta } = orbits(col);
        const ox = col * colW + (colW - torusSize) / 2;

        // ε-tube status: has the pseudo-orbit strayed beyond the shadow bound?
        const bound = C_SHADOW * delta;
        const escaped = err.some(e => e > Math.max(bound, 0.05));

        // Column header.
        Canvas.label(ctx, `δ = ${delta}`, col * colW + 10, 20,
          { color: Palette.text, font: 'bold 12px monospace' });
        Canvas.label(ctx, `‖err‖ ≤ ${bound.toFixed(3)}`, col * colW + 10, 38,
          { color: Palette.muted, font: '10px monospace' });

        // Torus panel background + border.
        ctx.fillStyle = Palette.surface;
        ctx.fillRect(ox, torusTop, torusSize, torusSize);
        ctx.strokeStyle = '#2a3a5c';
        ctx.lineWidth = 1;
        ctx.strokeRect(ox, torusTop, torusSize, torusSize);

        // True orbit (cyan solid), pseudo-orbit (grey dashed, or red if escaped).
        drawTorus(trueO, ox, torusTop, torusSize, Palette.blue, false);
        drawTorus(pseudo, ox, torusTop, torusSize,
          escaped ? Palette.accent : Palette.muted, true);
        // start point marker
        Canvas.dot(ctx, ox + X0[0] * torusSize, torusTop + (1 - X0[1]) * torusSize, 3, Palette.gold);

        // Error plot below.
        drawErrorPlot(err, bound, ox, plotTop, torusSize, plotH,
          escaped ? Palette.accent : Palette.blue);
      }

      // Footer note.
      Canvas.label(ctx,
        `cat map  λ⁺=${LP.toFixed(3)}  λ⁻=${LM.toFixed(3)}   ⇒   shadowing C = √5 ≈ ${C_SHADOW.toFixed(3)}`,
        10, H - 8, { color: Palette.gold, font: '11px monospace' });
    }

    function drawErrorPlot(err, bound, ox, oy, w, h, color) {
      // Auto-scale to the larger of the observed error and the theoretical bound.
      const maxE = Math.max(bound * 1.3, ...err, 1e-6);
      const sx = n => ox + (n / (err.length - 1)) * w;
      const sy = e => oy + h - (e / maxE) * h;

      // axes box
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.strokeRect(ox, oy, w, h);

      // theoretical bound C·δ (dashed gold horizontal line)
      ctx.strokeStyle = Palette.gold;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(ox, sy(bound)); ctx.lineTo(ox + w, sy(bound));
      ctx.stroke();
      ctx.setLineDash([]);

      // error curve
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      err.forEach((e, n) => {
        const cx = sx(n), cy = sy(e);
        if (n === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      });
      ctx.stroke();

      Canvas.label(ctx, '‖err‖ vs n', ox + 3, oy + 12,
        { color: Palette.muted, font: '9px monospace' });
    }

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'n_steps', label: 'n steps', min: 20, max: 80, step: 1,
        value: state.n_steps, format: v => String(v | 0) },
      { type: 'button', label: 'New pseudo-orbit', action: () => { regenNoise(); draw(); } },
    ]);

    ctrl.onChange((key, val) => { state[key] = val; draw(); });

    function reset() {
      Object.assign(state, DEFAULTS);
      ctrl.set('n_steps', state.n_steps);
      regenNoise();
      draw();
    }

    regenNoise();
    draw();

    return {
      destroy() { container.innerHTML = ''; },
      reset,
      setParam(k, v) { state[k] = v; draw(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = ShadowingLemma;
