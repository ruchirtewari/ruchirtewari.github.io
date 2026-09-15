/**
 * @file 04_02_03_poincare_recurrence.js
 * @chapter 4 — Ergodic Theory
 * @sections §4.2, §4.3
 * @concept Poincaré recurrence and Kac's lemma
 *
 * POINCARÉ RECURRENCE — RETURN TIME DISTRIBUTIONS
 *
 * The Poincaré recurrence theorem says that for a measure-preserving map T of a
 * probability space, almost every point of a positive-measure set A returns to A
 * infinitely often. Kac's lemma sharpens this into a quantitative statement about
 * the *first-return time*
 *
 *     τ_A(x) = min { n ≥ 1 : Tⁿ(x) ∈ A } :
 *
 *     ∫_A τ_A dμ = 1      ⇒      E_A[ τ_A ] = 1 / μ(A).                      (Kac)
 *
 * So the mean return time to a set of measure ε is exactly 1/ε — regardless of the
 * map's finer dynamics. What *does* depend on the dynamics is the *shape* of the
 * return-time distribution:
 *
 *   • Irrational rotation  R_α(x) = x + α (mod 1)   — quasi-periodic / rigid.
 *     By the three-gap (Steinhaus) theorem the return times to [0,ε) take only a
 *     few distinct values clustered tightly around 1/ε. The histogram is spiky.
 *
 *   • Doubling map  E₂(x) = 2x (mod 1)              — mixing / chaotic.
 *     Successive visits are essentially independent, so τ_A is approximately
 *     GEOMETRIC with parameter ε: P(τ = n) ≈ ε(1−ε)^{n−1}, mean 1/ε, long tail.
 *
 * Both share the SAME mean 1/ε (dashed red Kac line) but have very different
 * spreads — the visual signature of "rigid" vs "mixing" dynamics.
 *
 * Numerical note: iterating E₂ in floating point suffers "precision death" (the
 * mantissa runs out of random bits after ~52 doublings and collapses to 0). We
 * avoid this by simulating E₂ exactly on a random real: keep a 53-bit window
 * x_{n} = 0.b_n b_{n+1}… and update x_{n+1} = 2·x_n − b_n while shifting a FRESH
 * random low-order bit into slot 2⁻⁵³, so the orbit never loses entropy.
 */

var PoincareRecurrence = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '04_02_03',
    title: 'Poincaré Recurrence — Return Time Distributions',
    chapter: 'Chapter 4 — Ergodic Theory',
    sections: ['4.2', '4.3'],
    concept: "Kac's lemma: the mean first-return time to A equals 1/μ(A)",
    description:
      'Compares first-return-time statistics to a small interval A = [0, ε) for a ' +
      'rigid irrational rotation (spiky, three-gap) and a mixing doubling map ' +
      '(geometric tail). Both means land on the Kac prediction 1/ε.',
  };

  // Irrational rotation numbers (all in (0,1); their orbits equidistribute).
  const ALPHAS = {
    golden: { value: (Math.sqrt(5) - 1) / 2, label: '(√5−1)/2' },
    sqrt2:  { value: Math.SQRT2 - 1,         label: '√2 − 1'   },
    einv:   { value: (1 / Math.E) % 1,       label: '1/e mod 1' },
  };

  const NBINS = 40;
  const DEFAULTS = { eps: 0.05, alpha: 'golden', n_particles: 2000, max_iter: 20000 };

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 400;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Panel geometry: two 240-wide boxes with an 80px gap (spec layout).
    const PL = { px: 25,  py: 100, pw: 205, ph: 235 };  // left  (rotation)
    const PR = { px: 330, py: 100, pw: 205, ph: 235 };  // right (doubling)

    // ── computation state ────────────────────────────────────────────────
    let histL, histR, idx, done;
    let sumL, sumR, cntL, cntR, censL, censR;  // running mean + censored counts
    let xHi, kac;                              // histogram x-range and 1/ε

    function alphaVal() { return ALPHAS[state.alpha].value; }

    function resetComputation() {
      histL = new Float64Array(NBINS);
      histR = new Float64Array(NBINS);
      idx = 0; done = false;
      sumL = sumR = 0; cntL = cntR = 0; censL = censR = 0;
      kac = 1 / state.eps;                       // Kac's predicted mean
      xHi = Math.max(30, Math.ceil(3 * kac));    // show ~3× the mean on the x-axis
    }

    // First-return time of an irrational rotation R_α to A = [0, ε).
    // Doubles are numerically safe here (values stay O(1), no precision decay).
    function returnRot(x0, eps, alpha, maxIter) {
      let x = x0;                                 // x0 ∈ [0, ε) ⊂ A
      for (let n = 1; n <= maxIter; n++) {
        x += alpha; if (x >= 1) x -= 1;           // R_α(x) = (x + α) mod 1
        if (x < eps) return n;                    // returned to A
      }
      return -1;                                  // censored (did not return)
    }

    // First-return time of the doubling map E₂ via an EXACT random-bit stream,
    // maintaining a full 53-bit window so the orbit never suffers precision death.
    const P53 = Math.pow(2, -53);
    function returnDbl(x0, eps, maxIter) {
      let v = x0;                                 // v = x_n as a real in [0,1)
      for (let n = 1; n <= maxIter; n++) {
        const b = v >= 0.5 ? 1 : 0;               // leading bit leaving the window
        // x_{n+1} = 2·x_n − b, refilling a fresh random bit at position 2⁻⁵³:
        v = 2 * v - b + (Math.random() < 0.5 ? P53 : 0);
        if (v < eps) return n;                    // E₂ⁿ(x) ∈ A
      }
      return -1;
    }

    // Process `batch` particles: each contributes one return time to each map.
    function step(batch) {
      const eps = state.eps, alpha = alphaVal(), maxIter = state.max_iter | 0;
      const end = Math.min(idx + batch, state.n_particles | 0);
      const bw = xHi / NBINS;                     // bin width in units of return time
      for (let i = idx; i < end; i++) {
        const r1 = returnRot(Math.random() * eps, eps, alpha, maxIter);
        if (r1 > 0) { histL[Math.min(NBINS - 1, (r1 / bw) | 0)]++; sumL += r1; cntL++; }
        else censL++;

        const r2 = returnDbl(Math.random() * eps, eps, maxIter);
        if (r2 > 0) { histR[Math.min(NBINS - 1, (r2 / bw) | 0)]++; sumR += r2; cntR++; }
        else censR++;
      }
      idx = end;
      if (idx >= (state.n_particles | 0)) done = true;
    }

    // ── drawing ──────────────────────────────────────────────────────────
    function roundRect(x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y,     x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x,     y + h, r);
      ctx.arcTo(x,     y + h, x,     y,     r);
      ctx.arcTo(x,     y,     x + w, y,     r);
      ctx.closePath();
    }

    function drawPanel(P, hist, color, title, eqn, meanObs, cens) {
      const { px, py, pw, ph } = P;

      // panel card
      roundRect(px - 10, py - 30, pw + 20, ph + 78, 6);
      ctx.fillStyle = Palette.surface; ctx.fill();
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1; ctx.stroke();

      // axes (L-shape)
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.lineTo(px + pw, py + ph);
      ctx.stroke();

      // bars, normalized to the tallest bin
      let maxc = 1; for (const c of hist) if (c > maxc) maxc = c;
      const bw = pw / NBINS;
      for (let i = 0; i < NBINS; i++) {
        if (hist[i] <= 0) continue;
        const hh = (hist[i] / maxc) * ph;
        ctx.fillStyle = color;
        ctx.fillRect(px + i * bw + 0.5, py + ph - hh, Math.max(1, bw - 1), hh);
      }

      // Kac prediction: vertical dashed red line at τ = 1/ε
      const kx = px + Math.min(1, kac / xHi) * pw;
      ctx.strokeStyle = Palette.accent; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(kx, py); ctx.lineTo(kx, py + ph); ctx.stroke();
      ctx.setLineDash([]);
      Canvas.label(ctx, '1/ε', kx + 3, py + 12, { color: Palette.accent, font: '10px monospace' });

      // titles + equations
      Canvas.label(ctx, title, px, py - 14, { color, font: '12px monospace' });
      Canvas.label(ctx, eqn,   px, py - 32 + 32 + ph + 18, { color: Palette.muted, font: '11px monospace' });

      // x-axis endpoints
      Canvas.label(ctx, '0',           px,      py + ph + 14, { color: Palette.muted, font: '10px monospace' });
      Canvas.label(ctx, String(xHi),   px + pw, py + ph + 14, { color: Palette.muted, font: '10px monospace', align: 'right' });
      Canvas.label(ctx, 'return time τ', px + pw / 2, py + ph + 14, { color: Palette.muted, font: '10px monospace', align: 'center' });

      // measured mean vs Kac target
      const meanTxt = cntStatValid(meanObs) ? meanObs.toFixed(1) : '—';
      Canvas.label(ctx, `mean τ ≈ ${meanTxt}  (Kac 1/ε = ${kac.toFixed(1)})`,
        px, py + ph + 32, { color: Palette.gold, font: '10px monospace' });
      if (cens > 0)
        Canvas.label(ctx, `censored: ${cens}`, px, py + ph + 46,
          { color: Palette.muted, font: '10px monospace' });
    }

    function cntStatValid(v) { return isFinite(v) && v > 0; }

    function draw() {
      Canvas.clear(ctx, width, height);

      // title + Kac statement
      Canvas.label(ctx, meta.title, 16, 24, { color: Palette.blue, font: '13px monospace' });
      Canvas.label(ctx, "Kac's lemma:  E[τ_A] = 1/μ(A) = 1/ε",
        16, 44, { color: Palette.text, font: '11px monospace' });

      const progress = done
        ? `done — ${state.n_particles | 0} particles`
        : `computing… ${idx}/${state.n_particles | 0}`;
      Canvas.label(ctx, progress, width - 16, 24,
        { color: Palette.muted, font: '11px monospace', align: 'right' });

      drawPanel(PL, histL, Palette.blue,
        'Rotation R_α  (rigid)',
        `R_α(x) = (x + ${alphaVal().toFixed(4)}) mod 1`,
        sumL / cntL, censL);

      drawPanel(PR, histR, Palette.gold,
        'Doubling E₂  (mixing)',
        'E₂(x) = 2x mod 1',
        sumR / cntR, censR);
    }

    // ── animation of the computation ─────────────────────────────────────
    const anim = Anim.loop(() => {
      if (!done) {
        step(100);                 // batch of 100 particles per frame
        draw();
        if (done) anim.stop();
      }
    });

    function recompute() {
      resetComputation();
      draw();
      anim.stop();
      anim.start();
    }

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'eps', label: 'ε (μ(A))', min: 0.01, max: 0.2, step: 0.005,
        value: state.eps, format: v => v.toFixed(3) },
      { type: 'select', key: 'alpha', label: 'α (rotation)', value: state.alpha,
        options: Object.keys(ALPHAS).map(k => ({ value: k, label: ALPHAS[k].label })) },
      { type: 'slider', key: 'n_particles', label: 'particles', min: 500, max: 5000, step: 100,
        value: state.n_particles, format: v => v | 0 },
      { type: 'slider', key: 'max_iter', label: 'max iter', min: 1000, max: 50000, step: 1000,
        value: state.max_iter, format: v => v | 0 },
      { type: 'button', label: 'Recompute', action: () => recompute() },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      recompute();
    });

    resetComputation();
    recompute();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset() { recompute(); },
      setParam(k, v) { state[k] = v; recompute(); },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = PoincareRecurrence;
