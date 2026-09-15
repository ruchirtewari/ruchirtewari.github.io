/**
 * @file 07_01_02_rotation_number.js
 * @chapter 7 — Circle Maps
 * @sections §7.1, §7.2
 * @concept Rotation number, Arnold tongues, the devil's staircase
 *
 * Arnold Tongues — rotation number of the standard circle map
 *
 * THE FAMILY.  The Arnold standard family of circle maps is
 *        f_{K,ω}(θ) = θ + ω − (K/2π) sin(2πθ)   (mod 1).
 * Its rotation number
 *        ρ(K, ω) = lim_{n→∞} (Fⁿ(θ) − θ) / n
 * (computed on the lift F, θ arbitrary) measures the average rotation per
 * iterate. Poincaré's theorem: ρ is well defined and independent of θ.
 *
 * ARNOLD TONGUES.  For K > 0, ρ locks onto rational values p/q over whole
 * open regions of (ω, K)-space — the tongues, or resonance horns — rooted at
 * ω = p/q on the K = 0 axis. Between them ρ varies through irrationals. As
 * K → 1 the tongues widen and begin to overlap (onset of chaos, non-invertible
 * maps for K > 1).
 *
 * LEFT PANEL.  ρ(K, ω) over (ω, K) ∈ [0,1]² on a 200×200 grid. Rational
 * plateaus with q ≤ 8 are given solid tongue colours (the widest few are
 * labelled p/q); irrational values get a grey ramp.
 *
 * RIGHT STRIP.  For the K chosen by the slider, ρ(ω) at 500-point resolution —
 * the devil's staircase: flat steps at every rational, rising through a Cantor
 * set of irrational values.
 *
 * The grid is computed in RAF chunks so the UI stays responsive.
 */

var RotationNumber = (() => {
  const { Palette, Controls, Canvas } = DSUtils;

  const meta = {
    id: '07_01_02',
    title: 'Arnold Tongues — Rotation Number of the Standard Circle Map',
    chapter: 'Chapter 7 — Circle Maps',
    sections: ['7.1', '7.2'],
    concept: 'Rotation number, mode-locking (Arnold tongues), the devil\'s staircase',
    description:
      'The rotation number ρ(K, ω) of the Arnold standard circle map over ' +
      '(ω, K) ∈ [0,1]²: coloured Arnold tongues where ρ is rational, grey where ' +
      'irrational. The right strip is the devil\'s staircase ρ(ω) at fixed K.',
  };

  const DEFAULTS = { K: 0.6, N_iter: 800, showLabels: true };

  const GX = 200, GY = 200;          // rotation-number grid resolution
  const TAU = 2 * Math.PI;

  // ── Farey rationals p/q with q ≤ 8 (tongue centres) ────────────────────
  function gcd(a, b) { while (b) { [a, b] = [b, a % b]; } return a; }
  const RATIONALS = (() => {
    const list = [];
    for (let q = 1; q <= 8; q++)
      for (let p = 0; p <= q; p++)
        if (gcd(p, q) === 1) list.push({ p, q, val: p / q, key: `${p}/${q}` });
    return list;
  })();

  // Fixed colours for the most prominent tongues; others derive a hue.
  const MAIN = {
    '0/1': [83, 216, 251],   // blue
    '1/1': [83, 216, 251],   // blue (ρ = 1 ≡ 0)
    '1/4': [76, 175, 80],    // green
    '1/3': [245, 166, 35],   // gold
    '1/2': [233, 69, 96],    // red
    '2/3': [255, 140, 0],    // orange
    '3/4': [150, 90, 210],   // purple
  };
  const MAIN_KEYS = ['1/4', '1/3', '1/2', '2/3', '3/4', '0/1'];   // labelled tongues

  function hslToRgb(h, s, l) {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hk = h / 360;
    const f = t => {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return [f(hk + 1 / 3) * 255, f(hk) * 255, f(hk - 1 / 3) * 255];
  }

  // ── rotation number via the lift ───────────────────────────────────────
  function rotationNumber(K, omega, N) {
    let F = 0;                                   // lift, starting from θ = 0
    for (let n = 0; n < N; n++) {
      F = F + omega - (K / TAU) * Math.sin(TAU * F);
    }
    return F / N;                                // ≈ ρ
  }

  // Classify a ρ value: nearest rational p/q (q ≤ 8) within tolerance, else null.
  const TOL = 0.02;
  function classify(rho) {
    let best = null, bestErr = TOL;
    for (const r of RATIONALS) {
      const e = Math.abs(rho - r.val);
      if (e < bestErr) { bestErr = e; best = r; }
    }
    return best;
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 380;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Left colour-map geometry.
    const MAP_X = 10, MAP_Y = 10, MAP_W = 400, MAP_H = 360;
    // Right staircase-strip geometry.
    const STR_X = 420, STR_Y = 10, STR_W = 130, STR_H = 360;

    // Offscreen grid canvas (GX×GY), blitted scaled into the map region.
    const off = document.createElement('canvas');
    off.width = GX; off.height = GY;
    const offCtx = off.getContext('2d');
    const img = offCtx.createImageData(GX, GY);

    // Label centroids accumulated during computation.
    let centroids = {};        // key -> { sx, sy, n }
    let rafId = null;          // chunked-render handle
    let computing = false;

    // ── chunked computation of the ρ grid ────────────────────────────────
    function renderMap() {
      cancelRender();
      computing = true;
      centroids = {};
      let gy = 0;
      const N = state.N_iter | 0;
      const ROWS_PER_FRAME = 6;

      const step = () => {
        const end = Math.min(GY, gy + ROWS_PER_FRAME);
        for (; gy < end; gy++) {
          const K = 1 - gy / (GY - 1);                 // top row = K = 1
          for (let gx = 0; gx < GX; gx++) {
            const omega = gx / (GX - 1);
            const rho = rotationNumber(K, omega, N);
            const cls = classify(rho);
            let rgb;
            if (cls) {
              rgb = MAIN[cls.key] || hslToRgb((cls.val * 300) % 360, 0.55, 0.5);
              // accumulate centroid for labelled tongues
              if (MAIN[cls.key]) {
                const c = centroids[cls.key] || (centroids[cls.key] = { sx: 0, sy: 0, n: 0 });
                c.sx += gx; c.sy += gy; c.n++;
              }
            } else {
              const t = Math.max(0, Math.min(1, rho));   // grey ramp for irrationals
              const g = 40 + t * 120;
              rgb = [g, g, g + 20];
            }
            const p = (gy * GX + gx) * 4;
            img.data[p] = rgb[0]; img.data[p + 1] = rgb[1];
            img.data[p + 2] = rgb[2]; img.data[p + 3] = 255;
          }
        }
        offCtx.putImageData(img, 0, 0);
        blitAll();
        if (gy < GY) {
          rafId = requestAnimationFrame(step);
        } else {
          computing = false; rafId = null;
          blitAll();
        }
      };
      rafId = requestAnimationFrame(step);
    }

    function cancelRender() {
      if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      computing = false;
    }

    // ── compose the full frame (map + staircase + overlays) ──────────────
    function blitAll() {
      ctx.fillStyle = Palette.bg;
      ctx.fillRect(0, 0, W, H);

      // colour map
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, MAP_X, MAP_Y, MAP_W, MAP_H);
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.strokeRect(MAP_X, MAP_Y, MAP_W, MAP_H);

      // axes labels for the map
      Canvas.label(ctx, 'ω →', MAP_X + MAP_W - 34, MAP_Y + MAP_H - 6,
        { color: Palette.muted, font: '10px monospace' });
      Canvas.label(ctx, 'K ↑', MAP_X + 4, MAP_Y + 14,
        { color: Palette.muted, font: '10px monospace' });

      // line marking the K currently shown in the staircase
      const kY = MAP_Y + (1 - state.K) * MAP_H;
      ctx.strokeStyle = Palette.white;
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(MAP_X, kY); ctx.lineTo(MAP_X + MAP_W, kY); ctx.stroke();
      ctx.setLineDash([]);

      // tongue labels at centroids
      if (state.showLabels && !computing) {
        for (const key of MAIN_KEYS) {
          const c = centroids[key];
          if (!c || c.n < 30) continue;                // skip near-invisible tongues
          const cx = MAP_X + (c.sx / c.n / (GX - 1)) * MAP_W;
          const cy = MAP_Y + (c.sy / c.n / (GY - 1)) * MAP_H;
          Canvas.label(ctx, key, cx, cy,
            { color: Palette.text, font: 'bold 11px monospace', align: 'center' });
        }
      }

      if (computing) {
        Canvas.label(ctx, 'computing…', MAP_X + 8, MAP_Y + MAP_H - 8,
          { color: Palette.gold, font: '11px monospace' });
      }

      drawStaircase();

      // titles
      Canvas.label(ctx, `devil's staircase`, STR_X, STR_Y + 12,
        { color: Palette.text, font: '11px monospace' });
      Canvas.label(ctx, `K = ${state.K.toFixed(2)}`, STR_X, H - 8,
        { color: Palette.gold, font: '11px monospace' });
    }

    // ── devil's staircase ρ(ω) at fixed K ────────────────────────────────
    function drawStaircase() {
      const K = state.K;
      // Cap iterations for the interactive staircase so the slider stays snappy.
      const N = Math.min(state.N_iter | 0, 900);
      const M = 500;

      ctx.fillStyle = Palette.surface;
      ctx.fillRect(STR_X, STR_Y + 18, STR_W, STR_H - 18);
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.strokeRect(STR_X, STR_Y + 18, STR_W, STR_H - 18);

      const y0 = STR_Y + 18, hh = STR_H - 18;
      const sx = omega => STR_X + omega * STR_W;            // ω across the strip
      const sy = rho => y0 + hh - Math.max(0, Math.min(1, rho)) * hh;  // ρ up

      ctx.strokeStyle = Palette.blue;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let i = 0; i < M; i++) {
        const omega = i / (M - 1);
        const rho = rotationNumber(K, omega, N);
        const cx = sx(omega), cy = sy(rho);
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      // axis ticks
      Canvas.label(ctx, 'ρ', STR_X + 2, y0 + 12, { color: Palette.muted, font: '9px monospace' });
      Canvas.label(ctx, 'ω', STR_X + STR_W - 12, y0 + hh - 3,
        { color: Palette.muted, font: '9px monospace' });
    }

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'K', label: 'K', min: 0, max: 1, step: 0.01,
        value: state.K, format: v => v.toFixed(2) },
      { type: 'slider', key: 'N_iter', label: 'N iter', min: 200, max: 2000, step: 50,
        value: state.N_iter, format: v => String(v | 0) },
      { type: 'button', label: 'Recompute (slow)', action: () => renderMap() },
      { type: 'checkbox', key: 'showLabels', label: 'tongue labels', value: state.showLabels },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      // K and label toggles are cheap → just recompose; the map is recomputed
      // only via the Recompute button (it is the heavy operation).
      if (!computing) blitAll();
    });

    function reset() {
      cancelRender();
      Object.assign(state, DEFAULTS);
      ['K', 'N_iter', 'showLabels'].forEach(k => ctrl.set(k, state[k]));
      renderMap();
    }

    renderMap();

    return {
      destroy() { cancelRender(); container.innerHTML = ''; },
      reset,
      setParam(k, v) {
        state[k] = v;
        if (k === 'N_iter') renderMap(); else if (!computing) blitAll();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = RotationNumber;
