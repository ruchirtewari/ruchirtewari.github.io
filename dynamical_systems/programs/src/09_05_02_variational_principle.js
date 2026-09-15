/**
 * @file 09_05_02_variational_principle.js
 * @chapter 9 — Entropy
 * @sections §9.5
 * @concept Variational principle: h_top(f) = sup_µ h_µ(f)
 *
 * The VARIATIONAL PRINCIPLE relates the two notions of entropy: for a
 * continuous map of a compact space,
 *
 *     h_top(f) = sup_µ h_µ(f),
 *
 * the supremum over all f-invariant Borel probability measures. A measure
 * attaining the sup is a "measure of maximal entropy". We test this on the
 * logistic family f_r(x) = r·x(1 − x), r ∈ [3.5, 4].
 *
 * TOPOLOGICAL ENTROPY  (blue) is estimated by LAP COUNTING (Milnor–Thurston):
 * if ℓ_n = number of monotone laps of f_r^n (one plus its count of turning
 * points), then h_top = limₙ (1/n) log ℓ_n. We evaluate f_r^n on a fine grid
 * and count direction reversals. This is exact in spirit: at r = 4 the map has
 * 2ⁿ laps so h_top = log 2, and below the Feigenbaum point r_∞ the laps grow
 * sub-exponentially so h_top → 0.
 *
 * METRIC ENTROPY  (red) of the natural (SRB) measure is estimated by the
 * Shannon–McMillan–Breiman theorem. Coding each orbit point by the binary
 * partition L/R of ½, the empirical frequency p_w of each length-k word gives
 *
 *     h_µ ≈ H(ξ^k)/k = −(1/k) Σ_w p_w log p_w,
 *
 * averaged over several k for accuracy. Since the SRB measure of a unimodal map
 * is not generally the measure of maximal entropy, h_µ ≤ h_top, with equality
 * at r = 4 (both equal log 2) — the variational principle made visible.
 *
 * A slow computation, so it runs incrementally with a live progress bar.
 */

var VariationalPrinciple = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '09_05_02',
    title: 'Variational Principle — h(f) = sup_µ h_µ(f)',
    chapter: 'Chapter 9 — Entropy',
    sections: ['9.5'],
    concept: 'Topological entropy (lap counting) vs metric entropy of the SRB measure (SMB estimate)',
    description:
      'Compares the topological entropy of the logistic map (estimated by counting ' +
      'monotone laps of fⁿ) with the metric entropy of its natural measure (estimated ' +
      'via word-frequency / Shannon–McMillan–Breiman), illustrating h_top = sup_µ h_µ.',
  };

  const LN2 = Math.LN2;
  const FEIGENBAUM_R = [3.0, 3.449490, 3.544090];

  const DEFAULTS = { r_min: 3.5, r_max: 4.0, N_orbit: 50000, k_word: 10, cols: 90 };

  // Topological entropy via lap counting of f_r^n on a fine grid.
  function lapEntropy(r, nLap, grid) {
    let prevY = null, prevDir = 0, reversals = 0;
    for (let g = 0; g <= grid; g++) {
      let x = g / grid;
      for (let i = 0; i < nLap; i++) x = r * x * (1 - x);
      if (prevY !== null) {
        const dir = x > prevY ? 1 : (x < prevY ? -1 : prevDir);
        if (dir !== 0 && prevDir !== 0 && dir !== prevDir) reversals++;
        if (dir !== 0) prevDir = dir;
      }
      prevY = x;
    }
    const laps = reversals + 1;
    return Math.min(LN2, Math.log(laps) / nLap);
  }

  // Metric entropy of the SRB measure via k-gram word frequencies (SMB).
  function metricEntropy(r, N, ks) {
    // Build the binary itinerary of one long orbit (symbol = x ≥ 0.5).
    let x = 0.3;
    for (let i = 0; i < 1000; i++) x = r * x * (1 - x);        // transient
    const sym = new Uint8Array(N);
    for (let i = 0; i < N; i++) { x = r * x * (1 - x); sym[i] = x >= 0.5 ? 1 : 0; }

    let sum = 0, cnt = 0;
    for (const k of ks) {
      if (k < 1 || k > 16) continue;
      const size = 1 << k, counts = new Float64Array(size);
      let idx = 0;
      const mask = size - 1;
      // prime the first (k-1) bits
      for (let i = 0; i < k - 1; i++) idx = ((idx << 1) | sym[i]) & mask;
      let total = 0;
      for (let i = k - 1; i < N; i++) {
        idx = ((idx << 1) | sym[i]) & mask;
        counts[idx]++; total++;
      }
      let Hk = 0;
      for (let w = 0; w < size; w++) {
        if (counts[w] > 0) { const p = counts[w] / total; Hk -= p * Math.log(p); }
      }
      sum += Hk / k; cnt++;
    }
    return cnt ? sum / cnt : 0;
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 380;
    const { ctx } = Canvas.create(container, W, H);

    const PX0 = 46, PX1 = 548, PY0 = 44, PY1 = 316;

    // Computation state (filled incrementally).
    let rVals = [], htop = [], hmet = [], cursor = 0, computing = false;

    const anim = Anim.loop(() => {
      const CHUNK = 2;                        // r-columns per frame
      const nLap = 11, grid = 20000;
      const k = state.k_word | 0;
      const ks = [Math.max(4, k - 4), Math.max(4, k - 2), k];
      for (let c = 0; c < CHUNK && cursor < rVals.length; c++, cursor++) {
        const r = rVals[cursor];
        htop[cursor] = lapEntropy(r, nLap, grid);
        hmet[cursor] = metricEntropy(r, state.N_orbit | 0, ks);
      }
      draw();
      if (cursor >= rVals.length) { computing = false; anim.stop(); draw(); }
    });

    function startCompute() {
      const cols = state.cols | 0;
      rVals = []; htop = new Array(cols).fill(NaN); hmet = new Array(cols).fill(NaN);
      for (let c = 0; c < cols; c++) rVals.push(state.r_min + (state.r_max - state.r_min) * (c / (cols - 1)));
      cursor = 0; computing = true;
      anim.start();
    }

    function mapX(r) { return PX0 + ((r - state.r_min) / (state.r_max - state.r_min)) * (PX1 - PX0); }
    function mapY(h) { return PY1 - (h / (LN2 * 1.08)) * (PY1 - PY0); }

    function draw() {
      Canvas.clear(ctx, W, H);

      // Axes.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(PX0, PY0); ctx.lineTo(PX0, PY1); ctx.lineTo(PX1, PY1); ctx.stroke();
      // y ticks.
      for (let k = 0; k <= 4; k++) {
        const v = (LN2 * k) / 4, y = mapY(v);
        ctx.strokeStyle = Palette.surface; ctx.beginPath(); ctx.moveTo(PX0, y); ctx.lineTo(PX1, y); ctx.stroke();
        Canvas.label(ctx, v.toFixed(2), PX0 - 4, y + 3, { color: Palette.muted, font: '8px monospace', align: 'right' });
      }
      // x ticks.
      for (let t = 0; t <= 5; t++) {
        const r = state.r_min + (state.r_max - state.r_min) * t / 5;
        Canvas.label(ctx, r.toFixed(2), mapX(r), PY1 + 12, { color: Palette.muted, font: '8px monospace', align: 'center' });
      }

      // h = log2 reference (dashed gold).
      ctx.strokeStyle = Palette.gold; ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
      const yl = mapY(LN2);
      ctx.beginPath(); ctx.moveTo(PX0, yl); ctx.lineTo(PX1, yl); ctx.stroke();
      ctx.setLineDash([]);
      Canvas.label(ctx, 'h = log 2', PX1 - 4, yl - 4, { color: Palette.gold, font: '9px monospace', align: 'right' });

      // Bifurcation markers r₁,r₂,r₃.
      FEIGENBAUM_R.forEach((r, i) => {
        if (r < state.r_min || r > state.r_max) return;
        const x = mapX(r);
        ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(x, PY0); ctx.lineTo(x, PY1); ctx.stroke();
        ctx.setLineDash([]);
        Canvas.label(ctx, `r${i + 1}`, x + 2, PY0 + 10, { color: Palette.muted, font: '8px monospace' });
      });

      // Curves (only the computed prefix).
      const topPts = [], metPts = [];
      for (let c = 0; c < rVals.length; c++) {
        if (!isNaN(htop[c])) topPts.push([mapX(rVals[c]), mapY(htop[c])]);
        if (!isNaN(hmet[c])) metPts.push([mapX(rVals[c]), mapY(hmet[c])]);
      }
      Canvas.polyline(ctx, topPts, Palette.blue, 1.8);
      Canvas.polyline(ctx, metPts, Palette.accent, 1.8);

      // Legend.
      Canvas.label(ctx, '■ h_top  (lap counting)', PX0 + 6, PY0 + 12, { color: Palette.blue, font: '10px monospace' });
      Canvas.label(ctx, '■ h_µ  (SRB, SMB estimate)', PX0 + 6, PY0 + 26, { color: Palette.accent, font: '10px monospace' });

      // Header + principle annotation.
      Canvas.label(ctx, meta.title, 8, 16, { color: Palette.blue, font: '12px monospace' });
      Canvas.label(ctx, 'Variational Principle:  h_top = sup_µ h_µ', W - 8, 16,
        { color: Palette.gold, font: '10px monospace', align: 'right' });
      Canvas.label(ctx, `N = ${state.N_orbit | 0}   k = ${state.k_word | 0}`, W - 8, 30,
        { color: Palette.muted, font: '9px monospace', align: 'right' });

      // Progress bar.
      if (computing) {
        const frac = rVals.length ? cursor / rVals.length : 0;
        ctx.fillStyle = Palette.card; ctx.fillRect(PX0, PY1 + 20, PX1 - PX0, 6);
        ctx.fillStyle = Palette.blue; ctx.fillRect(PX0, PY1 + 20, (PX1 - PX0) * frac, 6);
        Canvas.label(ctx, `computing… ${(frac * 100) | 0}%`, PX0, PY1 + 44, { color: Palette.blue, font: '10px monospace' });
      } else {
        Canvas.label(ctx, 'Curves meet at r=4 (both = log 2): SRB measure is maximal there.',
          PX0, PY1 + 40, { color: Palette.text, font: '10px monospace' });
      }
    }

    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'r_min', label: 'r min', min: 3.0, max: 3.8, step: 0.01, value: state.r_min, format: v => v.toFixed(2) },
      { type: 'slider', key: 'r_max', label: 'r max', min: 3.8, max: 4.0, step: 0.005, value: state.r_max, format: v => v.toFixed(3) },
      { type: 'slider', key: 'N_orbit', label: 'N orbit', min: 10000, max: 100000, step: 5000, value: state.N_orbit, format: v => (v | 0) },
      { type: 'slider', key: 'k_word', label: 'k word', min: 4, max: 12, step: 1, value: state.k_word, format: v => v | 0 },
      { type: 'button', label: 'Recompute', action: () => startCompute() },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = (key === 'N_orbit' || key === 'k_word') ? (val | 0) : val;
      // parameter changes require a recompute to take effect; reflect ranges live.
      draw();
    });

    function reset() {
      anim.stop();
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      startCompute();
    }

    startCompute();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset,
      setParam(k, v) { state[k] = (k === 'N_orbit' || k === 'k_word' || k === 'cols') ? (v | 0) : v; startCompute(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = VariationalPrinciple;
