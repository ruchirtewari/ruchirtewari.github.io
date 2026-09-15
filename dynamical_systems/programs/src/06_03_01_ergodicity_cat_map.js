/**
 * @file 06_03_01_ergodicity_cat_map.js
 * @chapter 6 — Ergodic Theory
 * @sections §6.3
 * @concept Ergodicity, Birkhoff time averages, Hopf's argument
 *
 * Ergodicity of the Cat Map — Hopf-argument visualization
 *
 * A measure-preserving map f is ERGODIC when every time average equals the
 * space average for almost every starting point:
 *        A_N⁺φ(x) = (1/N) Σ_{k=0}^{N-1} φ(fᵏ x)   ⟶   ∫ φ dµ   as N → ∞.
 *
 * Arnold's cat map  f(x,y) = (2x + y, x + y) (mod 1)  is ergodic (in fact
 * mixing). We SEE this: on a 64×64 grid of initial points we colour each by
 * A_N⁺φ. As N grows the wildly-structured picture of φ homogenises toward the
 * single constant value ∫ φ dµ — for our test functions, 0 (uniform dark).
 *
 * HOPF'S ARGUMENT.  The forward average A_N⁺φ is (asymptotically) constant
 * along stable leaves W^s (points on the same leaf share a future), while the
 * backward average is constant along unstable leaves; since the two foliations
 * are transverse and the map is ergodic the common value is the space average.
 * The small panel at the bottom tracks Var of A_N⁺φ sampled along one stable
 * leaf — it collapses to 0, the quantitative face of Hopf's argument.
 *
 * TEST FUNCTIONS (all with the chosen space-average subtracted for colouring):
 *   sin(2πx)cos(2πy)   ∫ = 0
 *   sin(2πx)           ∫ = 0
 *   x·y − 0.25         ∫ = 0
 *   1_{x>0.5}          ∫ = 0.5
 *
 * COLOUR.  Diverging map centred at 0: negative → red (accent), 0 → black,
 * positive → cyan (blue), interpolated in RGB.
 */

var ErgodicityCatMap = (() => {
  const { Palette, Controls, Canvas, Anim, Math2D } = DSUtils;
  const TAU = Math2D.TAU;

  const meta = {
    id: '06_03_01',
    title: 'Ergodicity of the Cat Map — Hopf Argument',
    chapter: 'Chapter 6 — Ergodic Theory',
    sections: ['6.3'],
    concept: 'Birkhoff time averages converge to the space average (ergodicity)',
    description:
      'Time averages A_N⁺φ of a test function under the cat map, on a 64×64 grid ' +
      'of initial points, for N = 1, 10, 50 and an adjustable N₄. As N grows the ' +
      'picture flattens to the space average, illustrating ergodicity; the bottom ' +
      'plot shows the variance along a stable leaf collapsing to 0.',
  };

  const GRID = 64;
  const ACCENT_RGB = [233, 69, 96];
  const BLUE_RGB   = [83, 216, 251];

  // Test functions and their space averages.
  const PHIS = {
    sincos: { f: (x, y) => Math.sin(TAU * x) * Math.cos(TAU * y), mean: 0,   label: 'sin(2πx)cos(2πy)' },
    sinx:   { f: (x, y) => Math.sin(TAU * x),                     mean: 0,   label: 'sin(2πx)' },
    xy:     { f: (x, y) => x * y - 0.25,                          mean: 0,   label: 'x·y − 0.25' },
    ind:    { f: (x, y) => (x > 0.5 ? 1 : 0),                     mean: 0.5, label: '1(x>0.5)' },
  };

  const DEFAULTS = { phi: 'sincos', N4: 200, animate: false };
  const PANEL_NS = [1, 10, 50];   // N for the first three panels

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 540;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Offscreen 64×64 canvas used to upscale value grids with nearest-neighbour.
    const off = document.createElement('canvas');
    off.width = GRID; off.height = GRID;
    const offCtx = off.getContext('2d');

    // Colour scale is fixed from the N=1 grid so higher-N panels visibly fade.
    let colourScale = 1;

    // Compute A_N⁺φ on the whole grid; returns Float64Array of length GRID².
    function averageGrid(N) {
      const phi = PHIS[state.phi];
      const out = new Float64Array(GRID * GRID);
      for (let j = 0; j < GRID; j++) {
        for (let i = 0; i < GRID; i++) {
          let x = i / GRID, y = j / GRID, acc = 0;
          for (let k = 0; k < N; k++) {
            acc += phi.f(x, y);
            const nx = (2 * x + y) % 1, ny = (x + y) % 1;   // cat map (already in [0,1))
            x = nx; y = ny;
          }
          out[j * GRID + i] = acc / N - phi.mean;   // centre on the space average
        }
      }
      return out;
    }

    // Render a value grid into the offscreen canvas, then blit scaled.
    function drawGrid(values, dx, dy, size) {
      const img = offCtx.createImageData(GRID, GRID);
      const d = img.data;
      for (let idx = 0; idx < values.length; idx++) {
        const v = values[idx] / colourScale;             // normalise
        let r, g, b;
        if (v < 0) {
          const t = Math.min(1, -v);
          r = ACCENT_RGB[0] * t; g = ACCENT_RGB[1] * t; b = ACCENT_RGB[2] * t;
        } else {
          const t = Math.min(1, v);
          r = BLUE_RGB[0] * t; g = BLUE_RGB[1] * t; b = BLUE_RGB[2] * t;
        }
        const p = idx * 4;
        d[p] = r; d[p + 1] = g; d[p + 2] = b; d[p + 3] = 255;
      }
      offCtx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, dx, dy, size, size);
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.strokeRect(dx, dy, size, size);
    }

    // Variance of A_N⁺φ sampled along one stable leaf, for N = 1..Nmax.
    // Stable eigenvector direction of A = [[2,1],[1,1]] is (−1/φ, 1) up to scale;
    // we walk 20 points along that direction and measure how their forward
    // averages agree as N grows (Hopf: they must coincide → variance → 0).
    function convergenceCurve(Nmax) {
      const phi = PHIS[state.phi];
      const M = 20;
      const dir = [-1 / Math2D.PHI, 1];
      const dlen = Math.hypot(dir[0], dir[1]);
      const ux = dir[0] / dlen, uy = dir[1] / dlen;
      const base = [0.15, 0.15];

      // Seed the M leaf points.
      const pts = [];
      for (let m = 0; m < M; m++) {
        const s = (m / M) * 0.8;
        pts.push([Math2D.mod(base[0] + s * ux, 1), Math2D.mod(base[1] + s * uy, 1)]);
      }
      // Running accumulators so we can read off the average at every N.
      const cur = pts.map(p => p.slice());
      const acc = new Float64Array(M);
      const vars = new Float64Array(Nmax);
      for (let N = 1; N <= Nmax; N++) {
        for (let m = 0; m < M; m++) {
          acc[m] += phi.f(cur[m][0], cur[m][1]);
          const x = cur[m][0], y = cur[m][1];
          cur[m][0] = (2 * x + y) % 1; cur[m][1] = (x + y) % 1;
        }
        // variance of the current averages acc/N across the M leaf points
        let mean = 0;
        for (let m = 0; m < M; m++) mean += acc[m] / N;
        mean /= M;
        let v = 0;
        for (let m = 0; m < M; m++) { const d = acc[m] / N - mean; v += d * d; }
        vars[N - 1] = v / M;
      }
      return vars;
    }

    function drawConvergence(dx, dy, w, h) {
      const Nmax = Math.max(50, state.N4 | 0);
      const vars = convergenceCurve(Nmax);
      const maxV = Math.max(...vars, 1e-9);

      ctx.fillStyle = Palette.surface;
      ctx.fillRect(dx, dy, w, h);
      ctx.strokeStyle = '#2a3a5c';
      ctx.lineWidth = 1;
      ctx.strokeRect(dx, dy, w, h);

      ctx.strokeStyle = Palette.blue;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      for (let n = 0; n < Nmax; n++) {
        const cx = dx + (n / (Nmax - 1)) * w;
        const cy = dy + h - (vars[n] / maxV) * (h - 4) - 2;
        if (n === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.stroke();

      Canvas.label(ctx, 'Var of A_N⁺φ along a stable leaf  →  0',
        dx + 4, dy + 13, { color: Palette.muted, font: '10px monospace' });
    }

    function draw() {
      // (re)establish the colour scale from the N=1 grid
      const g1 = averageGrid(1);
      let maxAbs = 0;
      for (const v of g1) if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
      colourScale = Math.max(maxAbs, 0.05);

      // clear
      ctx.fillStyle = Palette.bg;
      ctx.fillRect(0, 0, W, H);

      Canvas.label(ctx, `φ = ${PHIS[state.phi].label}`, 12, 18,
        { color: Palette.text, font: '13px monospace' });

      // 2×2 grid of panels
      const size = 210, gap = 14;
      const x0 = (W - (2 * size + gap)) / 2;
      const y0 = 30;
      const Ns = [PANEL_NS[0], PANEL_NS[1], PANEL_NS[2], Math.max(1, state.N4 | 0)];
      const positions = [
        [x0, y0], [x0 + size + gap, y0],
        [x0, y0 + size + gap], [x0 + size + gap, y0 + size + gap],
      ];
      for (let k = 0; k < 4; k++) {
        const grid = k === 0 ? g1 : averageGrid(Ns[k]);
        const [px, py] = positions[k];
        drawGrid(grid, px, py, size);
        Canvas.label(ctx, `N = ${Ns[k]}`, px + 6, py + 16,
          { color: Palette.gold, font: 'bold 12px monospace' });
      }

      // convergence panel at the bottom
      const convY = y0 + 2 * size + gap + 12;
      drawConvergence(x0, convY, 2 * size + gap, H - convY - 10);
    }

    // ── animation: sweep N₄ upward to watch the 4th panel flatten ──────────
    const anim = Anim.loop(() => {
      state.N4 = Math.min(500, state.N4 + 6);
      ctrl.set('N4', state.N4);
      draw();
      if (state.N4 >= 500) anim.stop();
    });

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'select', key: 'phi', label: 'φ', value: state.phi, options: [
        { value: 'sincos', label: 'sin(2πx)cos(2πy)' },
        { value: 'sinx',   label: 'sin(2πx)' },
        { value: 'xy',     label: 'x·y − 0.25' },
        { value: 'ind',    label: '1(x>0.5)' },
      ] },
      { type: 'slider', key: 'N4', label: 'N₄ (panel 4)', min: 50, max: 500, step: 1,
        value: state.N4, format: v => String(v | 0) },
      { type: 'button', label: 'Recompute', action: () => draw() },
      { type: 'checkbox', key: 'animate', label: 'animate N₄', value: state.animate },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      if (key === 'animate') {
        if (val) { state.N4 = 50; ctrl.set('N4', 50); anim.start(); }
        else anim.stop();
        return;
      }
      draw();
    });

    function reset() {
      anim.stop();
      Object.assign(state, DEFAULTS);
      ['phi', 'N4', 'animate'].forEach(k => ctrl.set(k, state[k]));
      draw();
    }

    draw();
    if (state.animate) anim.start();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset,
      setParam(k, v) {
        state[k] = v;
        if (k === 'animate') { v ? anim.start() : anim.stop(); }
        else draw();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = ErgodicityCatMap;
