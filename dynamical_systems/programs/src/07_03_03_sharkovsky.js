/**
 * @file 07_03_03_sharkovsky.js
 * @chapter 7 — One-Dimensional Dynamics
 * @sections §7.3
 * @concept Sharkovsky's theorem, period forcing, "period 3 implies chaos"
 *
 * SHARKOVSKY'S THEOREM orders the natural numbers by
 *
 *   3 ≻ 5 ≻ 7 ≻ … ≻ 2·3 ≻ 2·5 ≻ … ≻ 2²·3 ≻ … ≻ 2³ ≻ 2² ≻ 2 ≻ 1
 *   (odds > 1, then 2×odds, then 4×odds, …, then the powers of two descending)
 *
 * and states: if a continuous map of an interval has a periodic point of
 * period m, then it has a periodic point of every period n with m ≻ n. The
 * maximal element is 3, so a period-3 point FORCES points of every period —
 * the Li–Yorke "period three implies chaos".
 *
 * We visualise this with the tent family  T_r(x) = r·min(x, 1−x),  r ∈ [1, 2].
 * For each r we detect which least-periods p exist by COUNTING periodic points
 * with the intermediate-value argument: the graph of T_r^p is piecewise linear
 * with slope ±r^p, and the fixed points of T_r^p are the sign changes of
 * g_p(x) = T_r^p(x) − x. Writing Fix(d) = #{x : T_r^d(x)=x}, the number of
 * points of LEAST period p is the Möbius sum
 *
 *     LP(p) = Σ_{d | p} μ(p/d) · Fix(d),
 *
 * and period p "exists" when LP(p) > 0. Sweeping r shows periods switching on
 * exactly in Sharkovsky order: once period 3 lights up (near r ≈ 1.9), every
 * period below it in the ordering is already present.
 */

var Sharkovsky = (() => {
  const { Palette, Canvas, Controls } = DSUtils;

  const meta = {
    id: '07_03_03',
    title: "Sharkovsky's Theorem — Period Cascade",
    chapter: 'Chapter 7 — One-Dimensional Dynamics',
    sections: ['7.3'],
    concept: 'Which periods exist for the tent map, and how period 3 forces all the rest',
    description:
      'A period-existence heat strip for the tent map T_r(x)=r·min(x,1−x) over ' +
      'r∈[1,2], computed by counting fixed points of T_r^p via sign changes, plus a ' +
      'live cobweb of the selected orbit. Marks the period-3 onset that forces chaos.',
  };

  const DEFAULTS = { r: 1.95, period_max: 16, cols: 260, ngrid: 1400 };

  // Möbius μ(n).
  function mobius(n) {
    if (n === 1) return 1;
    let m = n, primes = 0;
    for (let i = 2; i * i <= m; i++) {
      if (m % i === 0) {
        m /= i;
        if (m % i === 0) return 0;    // square factor
        primes++;
      }
    }
    if (m > 1) primes++;
    return primes % 2 === 0 ? 1 : -1;
  }

  const tent = (x, r) => r * Math.min(x, 1 - x);

  // Row colour by period (spec: 1 blue, 2 gold, 3 accent, 4 cyan, else dim).
  function periodColor(p) {
    if (p === 1) return Palette.blue;
    if (p === 2) return Palette.gold;
    if (p === 3) return Palette.accent;
    if (p === 4) return '#00e5c0';
    // higher periods: dim muted, slightly varied so bands are distinguishable.
    const l = 30 + (p % 3) * 6;
    return `hsl(220, 25%, ${l}%)`;
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 420;
    const LM = 40;                       // left margin for period-axis labels
    const STRIP_Y = 26, STRIP_H = 226;   // heat strip
    const COB_Y = 288, COB_H = 120;      // cobweb panel
    const R_MIN = 1, R_MAX = 2;

    const { ctx, width, height } = Canvas.create(container, W, H);

    // exists[c][p] = does least-period p exist at column c's r-value?
    let exists = null;
    let period3OnsetR = null;

    // ── heavy computation: period existence over the r-sweep ──────────────
    function compute() {
      const cols = state.cols | 0;
      const pmax = state.period_max | 0;
      const ng = state.ngrid | 0;
      exists = [];
      period3OnsetR = null;

      for (let c = 0; c < cols; c++) {
        const r = R_MIN + (R_MAX - R_MIN) * (c / (cols - 1));
        // Count Fix(d) = #fixed points of T^d via sign changes of T^d(x)−x.
        const fix = new Float64Array(pmax + 1);
        const prevSign = new Int8Array(pmax + 1);
        let first = true;
        for (let g = 0; g <= ng; g++) {
          const x = (g + 0.5) / (ng + 1);       // sample interior of (0,1)
          // Iterate, recording g_d = T^d(x) − x at each depth d.
          let v = x;
          for (let d = 1; d <= pmax; d++) {
            v = tent(v, r);
            const gd = v - x;
            const s = gd >= 0 ? 1 : -1;
            if (!first && s !== prevSign[d]) fix[d] += 1;
            prevSign[d] = s;
          }
          first = false;
        }
        // Möbius-invert to least-period counts.
        const col = new Uint8Array(pmax + 1);
        for (let p = 1; p <= pmax; p++) {
          let lp = 0;
          for (let d = 1; d <= p; d++) if (p % d === 0) lp += mobius(p / d) * fix[d];
          col[p] = lp > 0.5 ? 1 : 0;
        }
        exists.push(col);
        if (period3OnsetR === null && pmax >= 3 && col[3]) period3OnsetR = r;
      }
    }

    // ── drawing ───────────────────────────────────────────────────────────
    function draw() {
      Canvas.clear(ctx, W, H);
      const pmax = state.period_max | 0;
      const cols = exists.length;
      const rowH = STRIP_H / pmax;
      const stripW = W - LM;

      // Period-existence cells.
      for (let c = 0; c < cols; c++) {
        const x0 = LM + (c / cols) * stripW;
        const cw = Math.ceil(stripW / cols) + 1;
        const col = exists[c];
        for (let p = 1; p <= pmax; p++) {
          if (!col[p]) continue;
          const y0 = STRIP_Y + STRIP_H - p * rowH;   // p=1 at bottom
          ctx.fillStyle = periodColor(p);
          ctx.fillRect(x0, y0, cw, Math.ceil(rowH) + 0.5);
        }
      }

      // Period-axis labels (a subset to avoid clutter).
      ctx.fillStyle = Palette.muted; ctx.font = '9px monospace'; ctx.textAlign = 'right';
      for (let p = 1; p <= pmax; p++) {
        if (pmax > 8 && p > 4 && p % 2 !== 0 && p !== pmax) continue;
        const y = STRIP_Y + STRIP_H - (p - 0.5) * rowH + 3;
        ctx.fillText(String(p), LM - 4, y);
      }
      ctx.textAlign = 'left';

      // Frame + r-axis ticks.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(LM + 0.5, STRIP_Y + 0.5, stripW - 1, STRIP_H - 1);
      for (let k = 0; k <= 5; k++) {
        const r = R_MIN + (R_MAX - R_MIN) * k / 5;
        const x = LM + (stripW) * k / 5;
        Canvas.label(ctx, r.toFixed(1), x, STRIP_Y + STRIP_H + 14, { color: Palette.muted, font: '9px monospace', align: 'center' });
      }

      // Period-3 onset line + label ("once period 3 appears, all periods do").
      if (period3OnsetR !== null) {
        const x = LM + ((period3OnsetR - R_MIN) / (R_MAX - R_MIN)) * stripW;
        ctx.strokeStyle = Palette.accent; ctx.lineWidth = 1.5; ctx.setLineDash([5, 3]);
        ctx.beginPath(); ctx.moveTo(x, STRIP_Y); ctx.lineTo(x, STRIP_Y + STRIP_H); ctx.stroke();
        ctx.setLineDash([]);
        // arrow + label
        Canvas.label(ctx, `Period-3 onset  r≈${period3OnsetR.toFixed(3)}`, Math.min(x + 6, W - 150), STRIP_Y + 12,
          { color: Palette.accent, font: '10px monospace' });
      }

      // r-cursor for the cobweb.
      const xc = LM + ((state.r - R_MIN) / (R_MAX - R_MIN)) * stripW;
      ctx.strokeStyle = Palette.white; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(xc, STRIP_Y); ctx.lineTo(xc, STRIP_Y + STRIP_H); ctx.stroke();
      ctx.setLineDash([]);

      drawCobweb();

      // Titles & Sharkovsky ordering annotation.
      Canvas.label(ctx, meta.title, 10, 16, { color: Palette.blue, font: '12px monospace' });
      Canvas.label(ctx, '3 ≻ 5 ≻ 7 ≻ … ≻ 2·3 ≻ 2·5 ≻ … ≻ 4 ≻ 2 ≻ 1', LM, STRIP_Y + STRIP_H + 30,
        { color: Palette.gold, font: '10px monospace' });
    }

    // Cobweb of T_r on [0,1] inside the bottom panel.
    function drawCobweb() {
      const r = state.r;
      const px = x => LM + x * (W - LM - 6);
      const py = y => COB_Y + COB_H - 4 - y * (COB_H - 8);

      // panel bg + frame
      ctx.fillStyle = Palette.surface;
      ctx.fillRect(LM, COB_Y, W - LM - 6, COB_H);
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(LM + 0.5, COB_Y + 0.5, W - LM - 7, COB_H - 1);

      // diagonal y=x
      ctx.strokeStyle = Palette.muted; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(1), py(1)); ctx.stroke();
      ctx.setLineDash([]);

      // graph of T_r (two line segments meeting at x=0.5)
      Canvas.polyline(ctx, [[px(0), py(0)], [px(0.5), py(r * 0.5)], [px(1), py(0)]], Palette.blue, 2);

      // fixed point x* = r/(r+1)
      const xs = r / (r + 1);
      Canvas.dot(ctx, px(xs), py(xs), 3.5, Palette.accent);

      // cobweb orbit
      let x = 0.15;
      let cx = px(x), cy = py(0);
      for (let i = 0; i < 50; i++) {
        const fx = tent(x, r);
        const col = Palette.heat(i / 50);
        ctx.strokeStyle = col; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px(x), py(fx)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(px(x), py(fx)); ctx.lineTo(px(fx), py(fx)); ctx.stroke();
        cx = px(fx); cy = py(fx); x = fx;
      }

      Canvas.label(ctx, `T_r(x)=r·min(x,1−x)   r=${r.toFixed(3)}   x*=r/(r+1)=${xs.toFixed(3)}`,
        LM + 2, COB_Y + 14, { color: Palette.text, font: '10px monospace' });
    }

    // ── controls ──────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'r', label: 'r (cobweb)', min: R_MIN, max: R_MAX, step: 0.002, value: state.r, format: v => v.toFixed(3) },
      { type: 'slider', key: 'period_max', label: 'max period', min: 4, max: 16, step: 1, value: state.period_max, format: v => v | 0 },
      { type: 'button', label: 'Recompute', action: () => { compute(); draw(); } },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      if (key === 'period_max') { state.period_max = val | 0; compute(); }
      else state[key] = val;
      draw();
    });

    function reset() {
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      compute(); draw();
    }

    compute(); draw();

    return {
      destroy() { container.innerHTML = ''; },
      reset,
      setParam(k, v) {
        state[k] = (k === 'period_max') ? (v | 0) : v;
        if (k === 'period_max') compute();
        draw();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = Sharkovsky;
