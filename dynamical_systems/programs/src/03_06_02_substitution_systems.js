/**
 * 03_06_02_substitution_systems.js
 * ─────────────────────────────────────────────────────────────────────────
 * Substitution Systems — Fibonacci & Thue–Morse Sequences  (Brin–Stuck §3.6)
 *
 * A substitution (or morphism) replaces each symbol of a finite alphabet by a
 * fixed word.  Iterating it from a seed letter generates a self-similar
 * infinite sequence — the fixed point of the substitution — whose orbit
 * closure under the shift is the associated substitution subshift.  These
 * systems bridge symbolic dynamics and spectral theory:
 *
 *   • FIBONACCI  a→ab, b→a.  Length after n steps is Fib(n+2); the sequence
 *     is Sturmian (a cut sequence of an irrational-slope line).  Its diffraction
 *     / power spectrum is PURE POINT — sharp Bragg peaks at golden-ratio-spaced
 *     frequencies — the 1-D quasicrystal.
 *   • THUE–MORSE  a→ab, b→ba.  Length 2ⁿ.  Its spectrum is SINGULAR
 *     CONTINUOUS: no isolated peaks and no absolutely continuous part, a
 *     self-similar hierarchy of humps instead.
 *   • TRIBONACCI  a→ab, b→ac, c→a  (3-symbol Pisot analogue of Fibonacci).
 *   • RUDIN–SHAPIRO  a→ac, b→dc, c→ab, d→db, projected {a,b}→+1, {c,d}→−1,
 *     whose spectrum is (essentially) absolutely continuous — flat like noise.
 *
 * We map symbols to numeric values, take the DFT with a hand-written iterative
 * Cooley–Tukey FFT (no libraries), and plot the power spectrum |X(k)|².
 *
 * Panels (canvas 560×380):
 *   TOP    (560×80)  — colour-coded strip of the first ~400 symbols.
 *   BOTTOM (560×280) — power spectrum, linear or log-y (toggle).
 * ─────────────────────────────────────────────────────────────────────────
 */

var SubstitutionSystems = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '03_06_02_substitution_systems',
    title: 'Substitution Systems — Fibonacci & Thue–Morse Sequences',
    chapter: 'Chapter 3 · §3.6',
    sections: ['3.6'],
    concept: 'Substitution dynamics, self-similarity, power spectrum',
    description:
      'Generates the fixed point of a substitution, shows it as a colour ' +
      'strip, and plots the FFT power spectrum: pure-point (Fibonacci), ' +
      'singular-continuous (Thue–Morse), or absolutely continuous ' +
      '(Rudin–Shapiro).',
  };

  // ── Substitution rules, symbol→value maps, and per-symbol colours ─────
  const SYM_COLOR = { a: Palette.blue, b: Palette.gold, c: Palette.accent, d: Palette.green };
  const RULES = {
    fibonacci: {
      name: 'Fibonacci', rule: { a: 'ab', b: 'a' },
      sym: { a: 1, b: -1 }, text: 'a→ab,  b→a',
    },
    thue_morse: {
      name: 'Thue–Morse', rule: { a: 'ab', b: 'ba' },
      sym: { a: 1, b: -1 }, text: 'a→ab,  b→ba',
    },
    tribonacci: {
      name: 'Tribonacci', rule: { a: 'ab', b: 'ac', c: 'a' },
      sym: { a: 1, b: -1, c: 0 }, text: 'a→ab,  b→ac,  c→a',
    },
    rudin_shapiro: {
      name: 'Rudin–Shapiro', rule: { a: 'ac', b: 'dc', c: 'ab', d: 'db' },
      sym: { a: 1, b: 1, c: -1, d: -1 }, text: 'a→ac, b→dc, c→ab, d→db',
    },
  };

  const W = 560, H = 380;
  const STRIP = { x0: 0, y0: 24, w: 560, h: 80 };
  const SPEC = { x0: 44, y0: 128, w: 500, h: 220 };
  const MAX_LEN = 1 << 18;   // safety cap on generated sequence length

  // ── Sequence generation ───────────────────────────────────────────────
  function generate(ruleKey, nLevels) {
    const { rule } = RULES[ruleKey];
    let s = 'a';
    for (let i = 0; i < nLevels; i++) {
      let out = '';
      for (let k = 0; k < s.length; k++) out += rule[s[k]];
      s = out;
      if (s.length > MAX_LEN) break;
    }
    return s;
  }

  // ── Iterative Cooley–Tukey FFT (in place, radix-2) ────────────────────
  // Transforms complex arrays re[], im[] of length N (a power of two).
  function fft(re, im) {
    const n = re.length;
    // bit-reversal permutation
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    // butterflies
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < half; k++) {
          const p = i + k, q = i + k + half;
          const vr = re[q] * cr - im[q] * ci;
          const vi = re[q] * ci + im[q] * cr;
          re[q] = re[p] - vr; im[q] = im[p] - vi;
          re[p] += vr;        im[p] += vi;
          const ncr = cr * wr - ci * wi;      // advance twiddle factor
          ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  /** Largest power of two ≤ x. */
  function pow2Floor(x) { let p = 1; while (p * 2 <= x) p *= 2; return p; }

  function init(container, params = {}) {
    const { canvas, ctx, width, height } = Canvas.create(container, W, H);

    const state = {
      rule: params.rule || 'fibonacci',
      nLevels: params.nLevels != null ? params.nLevels : 14,
      logY: params.logY != null ? params.logY : true,
    };

    let seq = '', numeric = null, power = null;   // cached derived data
    let anim = null, animAccum = 0;
    let levelInput = null;                         // n_levels <input> ref

    // Recompute sequence, numeric mapping, and power spectrum.
    function recompute() {
      const R = RULES[state.rule];
      seq = generate(state.rule, Math.round(state.nLevels));
      numeric = new Float64Array(seq.length);
      for (let i = 0; i < seq.length; i++) numeric[i] = R.sym[seq[i]];

      const N = pow2Floor(seq.length);
      if (N < 2) { power = null; return; }
      const re = new Float64Array(N), im = new Float64Array(N);
      // Subtract mean so the k=0 (DC) term does not dominate.
      let mean = 0; for (let i = 0; i < N; i++) mean += numeric[i]; mean /= N;
      for (let i = 0; i < N; i++) re[i] = numeric[i] - mean;
      fft(re, im);
      const half = N >> 1;
      power = new Float64Array(half + 1);
      for (let k = 0; k <= half; k++) power[k] = (re[k] * re[k] + im[k] * im[k]) / N;
    }

    // ── Drawing ─────────────────────────────────────────────────────────
    function drawAll() {
      Canvas.clear(ctx, width, height);
      const R = RULES[state.rule];

      // Title + info line
      ctx.textAlign = 'left';
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = Palette.text;
      ctx.fillText(`Substitution: ${R.name}`, 6, 16);
      ctx.font = '10px monospace';
      ctx.fillStyle = Palette.muted;
      ctx.textAlign = 'right';
      ctx.fillText(`rule ${R.text}   n=${Math.round(state.nLevels)}   length=${seq.length}`, W - 6, 16);
      ctx.textAlign = 'left';

      drawStrip(R);
      drawSpectrum();
    }

    function drawStrip(R) {
      const shown = Math.min(seq.length, 400);
      const bw = STRIP.w / shown;
      for (let i = 0; i < shown; i++) {
        ctx.fillStyle = SYM_COLOR[seq[i]] || Palette.muted;
        ctx.fillRect(STRIP.x0 + i * bw, STRIP.y0, Math.ceil(bw) + 0.5, STRIP.h);
      }
      // frame + caption
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1;
      ctx.strokeRect(STRIP.x0 + 0.5, STRIP.y0 + 0.5, STRIP.w - 1, STRIP.h);
      Canvas.label(ctx, `first ${shown} symbols`, STRIP.x0 + 6, STRIP.y0 + STRIP.h - 6, { color: Palette.text });

      // legend
      let lx = STRIP.x0 + 120;
      ctx.font = '10px monospace';
      for (const sym of Object.keys(R.sym)) {
        ctx.fillStyle = SYM_COLOR[sym];
        ctx.fillRect(lx, STRIP.y0 + STRIP.h - 16, 10, 10);
        ctx.fillStyle = Palette.text;
        ctx.fillText(`${sym}=${R.sym[sym] >= 0 ? '+' : ''}${R.sym[sym]}`, lx + 13, STRIP.y0 + STRIP.h - 7);
        lx += 58;
      }
    }

    function drawSpectrum() {
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1;
      ctx.strokeRect(SPEC.x0 + 0.5, SPEC.y0 + 0.5, SPEC.w, SPEC.h);
      Canvas.label(ctx, `Power spectrum |X(k)|²  (${state.logY ? 'log' : 'linear'} y)`,
        SPEC.x0 + 6, SPEC.y0 + 16, { color: Palette.muted });

      if (!power) {
        Canvas.label(ctx, 'sequence too short', SPEC.x0 + SPEC.w / 2 - 60, SPEC.y0 + SPEC.h / 2, { color: Palette.muted });
        return;
      }

      const nHalf = power.length - 1;                  // k = 0..N/2
      // Peak-preserving downsample to pixel columns.
      const cols = Math.min(SPEC.w, nHalf);
      const colMax = new Float64Array(cols);
      let gMax = 0;
      for (let k = 1; k <= nHalf; k++) {               // skip DC (k=0)
        const c = Math.min(cols - 1, ((k / nHalf) * (cols - 1)) | 0);
        if (power[k] > colMax[c]) colMax[c] = power[k];
        if (power[k] > gMax) gMax = power[k];
      }
      if (gMax <= 0) gMax = 1;

      const gx0 = SPEC.x0 + 2, gy0 = SPEC.y0 + SPEC.h - 20, gyTop = SPEC.y0 + 26;
      const plotH = gy0 - gyTop;
      const colW = SPEC.w / cols;

      // y-axis mapping (linear or log)
      const floor = gMax * 1e-6;
      const toY = v => {
        let t;
        if (state.logY) {
          const lv = Math.log10(Math.max(v, floor));
          const lmax = Math.log10(gMax), lmin = Math.log10(floor);
          t = (lv - lmin) / (lmax - lmin);
        } else {
          t = v / gMax;
        }
        return gy0 - Math.max(0, Math.min(1, t)) * plotH;
      };

      // impulses
      ctx.strokeStyle = Palette.blue; ctx.lineWidth = Math.max(1, colW * 0.9);
      ctx.beginPath();
      for (let c = 0; c < cols; c++) {
        if (colMax[c] <= 0) continue;
        const x = gx0 + c * colW + colW / 2;
        ctx.moveTo(x, gy0); ctx.lineTo(x, toY(colMax[c]));
      }
      ctx.stroke();

      // baseline
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(gx0, gy0); ctx.lineTo(gx0 + SPEC.w - 4, gy0); ctx.stroke();

      // axis labels
      ctx.font = '9px monospace'; ctx.fillStyle = Palette.muted; ctx.textAlign = 'center';
      ctx.fillText('k=0', gx0 + 4, gy0 + 12);
      ctx.fillText('N/2', gx0 + SPEC.w - 12, gy0 + 12);
      ctx.textAlign = 'right';
      ctx.save(); ctx.translate(SPEC.x0 - 6, SPEC.y0 + SPEC.h / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.fillText(state.logY ? 'log |X(k)|²' : '|X(k)|²', 0, 0); ctx.restore();
      ctx.textAlign = 'left';
    }

    // ── Growth animation (n_levels +1 per second, wraps at 16) ───────────
    function tick(dt) {
      animAccum += dt;
      if (animAccum >= 1) {
        animAccum = 0;
        let next = Math.round(state.nLevels) + 1;
        if (next > 16) next = 4;
        setLevel(next);            // updates slider + redraws
      }
    }

    function setLevel(v) {
      if (levelInput) { levelInput.value = v; levelInput.dispatchEvent(new Event('input')); }
      else { state.nLevels = v; recompute(); drawAll(); }
    }

    function setAnimating(on) {
      if (on) { animAccum = 0; if (!anim) anim = Anim.loop(tick); anim.start(); }
      else if (anim) anim.stop();
    }

    // ── Controls ────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'select', key: 'rule', label: 'Substitution', value: state.rule, options: [
        { value: 'fibonacci',     label: 'Fibonacci (a→ab, b→a)' },
        { value: 'thue_morse',    label: 'Thue–Morse (a→ab, b→ba)' },
        { value: 'tribonacci',    label: 'Tribonacci (3-symbol)' },
        { value: 'rudin_shapiro', label: 'Rudin–Shapiro (4-symbol)' },
      ] },
      { type: 'slider', key: 'nLevels', label: 'n levels', min: 4, max: 16, step: 1, value: state.nLevels, format: v => v | 0 },
      { type: 'checkbox', key: 'logY', label: 'log y-axis', value: state.logY },
      { type: 'checkbox', key: 'animate', label: 'Animate growth', value: false },
    ]);

    const panel = container.lastChild;
    levelInput = panel.querySelector('input[type=range]');

    ctrl.onChange((key, val) => {
      if (key === 'animate') { setAnimating(val); return; }
      state[key] = val;
      if (key === 'rule' || key === 'nLevels') recompute();
      drawAll();
    });

    // Initial paint
    recompute();
    drawAll();

    return {
      destroy() {
        setAnimating(false);
        container.innerHTML = '';
      },
      reset() {
        setAnimating(false);
        ctrl.set('animate', false);
        state.rule = 'fibonacci'; state.logY = true;
        setLevel(14);
      },
      setParam(key, val) {
        if (key === 'nLevels') setLevel(val);
        else if (key in state) { state[key] = val; recompute(); drawAll(); }
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = SubstitutionSystems;
