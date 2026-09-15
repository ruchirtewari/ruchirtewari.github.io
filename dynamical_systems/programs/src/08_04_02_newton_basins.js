/**
 * @file 08_04_02_newton_basins.js
 * @chapter 8 — Complex Dynamics
 * @sections §8.4, §8.5
 * @concept Newton's method as a rational map; Fatou basins and the Julia set
 *
 * Newton's method for a polynomial p(z) is the rational map
 *
 *     N(z) = z − p(z)/p′(z),
 *
 * whose attracting fixed points are exactly the simple roots of p. Each root
 * ω_k has a Fatou basin  B_k = { z₀ : N^n(z₀) → ω_k }; the COMMON boundary of
 * all the basins is the Julia set J(N) — a fractal, because near it arbitrarily
 * close starting points fall into different roots.
 *
 * For p(z) = zⁿ − 1 (roots = the n-th roots of unity) Newton's map simplifies:
 *
 *     N(z) = z − (zⁿ − 1)/(n·zⁿ⁻¹) = ((n−1)zⁿ + 1) / (n·zⁿ⁻¹).
 *
 * We colour each starting pixel z₀ by which root it converges to (n distinct
 * hues) and modulate brightness by convergence speed, so fast basins glow and
 * the intricate Julia boundary appears where colours interleave. The Newton
 * map for p(z) = z³ − 2z + 2 additionally has an ATTRACTING 2-CYCLE, producing
 * black "escape" regions that never reach any root — a Fatou component that is
 * not a basin of a root.
 *
 * Domain [-2,2]² by default; scroll to zoom, drag to pan. Rendering proceeds
 * in 8 strips (one per frame) with a progress bar.
 */

var NewtonBasins = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '08_04_02',
    title: "Newton's Method Basins — Rational Map on ℂ̄",
    chapter: 'Chapter 8 — Complex Dynamics',
    sections: ['8.4', '8.5'],
    concept: 'Basins of attraction of Newton iteration and the fractal Julia boundary',
    description:
      'Colours the plane by which root of p(z) Newton\'s method converges to, with ' +
      'brightness set by convergence speed. Supports zⁿ−1, zⁿ−z and z³−2z+2, degree ' +
      '2–6, zoom/pan, and chunked strip rendering.',
  };

  const STRIPS = 8;
  const HEADER = 28, PANEL_H = 452;

  const DEFAULTS = { n: 3, maxIter: 60, poly: 'zn1', tol: 1e-4 };

  // Hard-coded roots of z³−2z+2 (one real, two complex, sum = 0, product = −2).
  const CUBIC_ROOTS = [
    [-1.769292354, 0],
    [0.884646177, 0.589742805],
    [0.884646177, -0.589742805],
  ];

  // Complex helpers.
  function cmul(ar, ai, br, bi) { return [ar * br - ai * bi, ar * bi + ai * br]; }
  function cpow(zr, zi, k) {
    let rr = 1, ri = 0;
    for (let i = 0; i < k; i++) { const t = cmul(rr, ri, zr, zi); rr = t[0]; ri = t[1]; }
    return [rr, ri];
  }
  function cdiv(ar, ai, br, bi) {
    const d = br * br + bi * bi;
    return [(ar * br + ai * bi) / d, (ai * br - ar * bi) / d];
  }

  // Roots of the currently selected polynomial.
  function rootsFor(poly, n) {
    if (poly === 'cubic') return CUBIC_ROOTS;
    if (poly === 'zn1') {
      const r = [];
      for (let k = 0; k < n; k++) r.push([Math.cos(2 * Math.PI * k / n), Math.sin(2 * Math.PI * k / n)]);
      return r;
    }
    // znz: p = zⁿ − z = z(zⁿ⁻¹ − 1): root 0 plus the (n−1)-th roots of unity.
    const r = [[0, 0]];
    const m = n - 1;
    for (let k = 0; k < m; k++) r.push([Math.cos(2 * Math.PI * k / m), Math.sin(2 * Math.PI * k / m)]);
    return r;
  }

  // p(z) and p′(z) for the selected polynomial.
  function pAndDeriv(poly, n, zr, zi) {
    if (poly === 'cubic') {
      // p = z³ − 2z + 2 ; p′ = 3z² − 2
      const [z3r, z3i] = cpow(zr, zi, 3);
      const [z2r, z2i] = cpow(zr, zi, 2);
      return [z3r - 2 * zr + 2, z3i - 2 * zi, 3 * z2r - 2, 3 * z2i];
    }
    const [znr, zni] = cpow(zr, zi, n);
    const [zn1r, zn1i] = cpow(zr, zi, n - 1);
    if (poly === 'zn1') {
      // p = zⁿ − 1 ; p′ = n·zⁿ⁻¹
      return [znr - 1, zni, n * zn1r, n * zn1i];
    }
    // znz: p = zⁿ − z ; p′ = n·zⁿ⁻¹ − 1
    return [znr - zr, zni - zi, n * zn1r - 1, n * zn1i];
  }

  function formulaLabel(poly, n) {
    if (poly === 'zn1') return `p(z)=z^${n}−1   N(z)=((${n - 1})z^${n}+1)/(${n}·z^${n - 1})`;
    if (poly === 'znz') return `p(z)=z^${n}−z   N(z)=z−(z^${n}−z)/(${n}z^${n - 1}−1)`;
    return `p(z)=z³−2z+2   N(z)=z−(z³−2z+2)/(3z²−2)`;
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 480;
    const { canvas, ctx } = Canvas.create(container, W, H);

    // Offscreen render buffer.
    const buf = document.createElement('canvas');
    buf.width = W; buf.height = PANEL_H;
    const bctx = buf.getContext('2d');
    let img = bctx.createImageData(W, PANEL_H);

    const view = { cx: 0, cy: 0, halfW: 2, baseHalfW: 2 };
    let roots = rootsFor(state.poly, state.n);
    let rootRGB = roots.map((_, k) => Palette.basinRGB(k, roots.length));

    let nextStrip = STRIPS;

    function pixToC(px, py) {
      const halfH = view.halfW * (PANEL_H / W);
      return [view.cx + (px / W - 0.5) * 2 * view.halfW,
              view.cy + (0.5 - py / PANEL_H) * 2 * halfH];
    }

    function schedule() { nextStrip = 0; ensureLoop(); }

    // Render one strip: Newton-iterate each pixel, classify by nearest root.
    function renderStrip() {
      const strip = nextStrip;
      const y0 = Math.floor(strip * PANEL_H / STRIPS);
      const y1 = Math.floor((strip + 1) * PANEL_H / STRIPS);
      const data = img.data;
      const maxIter = state.maxIter | 0;
      const poly = state.poly, n = state.n | 0, tol2 = state.tol * state.tol;
      const nr = roots.length;

      for (let py = y0; py < y1; py++) {
        for (let px = 0; px < W; px++) {
          let [zr, zi] = pixToC(px, py);
          let root = -1, iters = maxIter;
          for (let it = 0; it < maxIter; it++) {
            const [pr, pi, dr, di] = pAndDeriv(poly, n, zr, zi);
            if (dr === 0 && di === 0) break;                 // critical point: stuck
            const [qr, qi] = cdiv(pr, pi, dr, di);
            zr -= qr; zi -= qi;
            // Converged to a root?
            for (let k = 0; k < nr; k++) {
              const ddr = zr - roots[k][0], ddi = zi - roots[k][1];
              if (ddr * ddr + ddi * ddi < tol2) { root = k; iters = it; break; }
            }
            if (root >= 0) break;
          }
          const idx = (py * W + px) * 4;
          if (root < 0) {
            // Did not converge to any root (e.g. attracted to a cycle): black.
            data[idx] = 0; data[idx + 1] = 0; data[idx + 2] = 0; data[idx + 3] = 255;
          } else {
            // Brightness ∝ convergence speed (faster = brighter).
            const b = 0.35 + 0.65 * (1 - iters / maxIter);
            const c = rootRGB[root];
            data[idx] = c[0] * b; data[idx + 1] = c[1] * b; data[idx + 2] = c[2] * b; data[idx + 3] = 255;
          }
        }
      }
      bctx.putImageData(img, 0, 0, 0, y0, W, y1 - y0);
      nextStrip++;
    }

    let loopRunning = false;
    const anim = Anim.loop(() => {
      if (nextStrip < STRIPS) { renderStrip(); composite(); }
      else { anim.stop(); loopRunning = false; composite(); }
    });
    function ensureLoop() { if (!loopRunning) { loopRunning = true; anim.start(); } }

    // ── composite ─────────────────────────────────────────────────────────
    let panDX = 0, panDY = 0, panning = false;
    function composite() {
      Canvas.clear(ctx, W, H);
      ctx.drawImage(buf, panning ? panDX : 0, HEADER + (panning ? panDY : 0), W, PANEL_H);

      // Root markers with labels ω_k ≈ (cos, sin).
      const halfH = view.halfW * (PANEL_H / W);
      roots.forEach((rt, k) => {
        const rx = (rt[0] - view.cx) / (2 * view.halfW) * W + W / 2;
        const ry = PANEL_H / 2 - (rt[1] - view.cy) / (2 * halfH) * PANEL_H;
        if (rx < 0 || rx > W || ry < 0 || ry > PANEL_H) return;
        const cy = HEADER + ry;
        Canvas.dot(ctx, rx, cy, 4, Palette.white);
        Canvas.dot(ctx, rx, cy, 2.5, `rgb(${rootRGB[k].join(',')})`);
        Canvas.label(ctx, `ω${k}≈(${rt[0].toFixed(2)},${rt[1].toFixed(2)})`, rx + 6, cy - 2,
          { color: Palette.text, font: '9px monospace' });
      });

      // Frame + progress bar.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(0.5, HEADER + 0.5, W - 1, PANEL_H - 1);
      if (nextStrip < STRIPS) {
        const frac = nextStrip / STRIPS;
        ctx.fillStyle = Palette.card; ctx.fillRect(4, HEADER + 4, W - 8, 4);
        ctx.fillStyle = Palette.blue; ctx.fillRect(4, HEADER + 4, (W - 8) * frac, 4);
      }

      // Header labels.
      Canvas.label(ctx, meta.title, 8, 12, { color: Palette.blue, font: '11px monospace' });
      Canvas.label(ctx, formulaLabel(state.poly, state.n), 8, 25, { color: Palette.gold, font: '10px monospace' });
      Canvas.label(ctx, `zoom ${(view.baseHalfW / view.halfW).toFixed(1)}×   ${roots.length} roots`,
        W - 8, 25, { color: Palette.muted, font: '10px monospace', align: 'right' });
    }

    // ── pointer interaction (zoom + pan) ──────────────────────────────────
    function local(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top - HEADER;
      if (y < 0 || y > PANEL_H || x < 0 || x > W) return null;
      return { x, y };
    }
    let down = null;
    const onDown = e => { const l = local(e); if (l) { down = l; panning = true; panDX = 0; panDY = 0; } };
    const onMove = e => {
      if (!down) return;
      const rect = canvas.getBoundingClientRect();
      panDX = (e.clientX - rect.left) - down.x;
      panDY = (e.clientY - rect.top - HEADER) - down.y;
      composite();
    };
    const onUp = () => {
      if (!down) return;
      const halfH = view.halfW * (PANEL_H / W);
      view.cx -= panDX / W * 2 * view.halfW;
      view.cy += panDY / PANEL_H * 2 * halfH;
      panning = false; panDX = 0; panDY = 0; down = null;
      schedule();
    };
    const onWheel = e => {
      const l = local(e);
      if (!l) return;
      e.preventDefault();
      const [ax, ay] = pixToC(l.x, l.y);
      const factor = e.deltaY < 0 ? 0.8 : 1.25;
      view.cx = ax + (view.cx - ax) * factor;
      view.cy = ay + (view.cy - ay) * factor;
      view.halfW *= factor;
      schedule();
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    // ── controls ──────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      {
        type: 'select', key: 'n', label: 'degree n', value: String(state.n),
        options: [2, 3, 4, 5, 6].map(v => ({ value: String(v), label: String(v) })),
      },
      { type: 'slider', key: 'maxIter', label: 'max iter', min: 20, max: 200, step: 5, value: state.maxIter, format: v => v | 0 },
      {
        type: 'select', key: 'poly', label: 'polynomial', value: state.poly,
        options: [
          { value: 'zn1', label: 'zⁿ − 1' },
          { value: 'znz', label: 'zⁿ − z' },
          { value: 'cubic', label: 'z³ − 2z + 2' },
        ],
      },
      { type: 'button', label: 'Reset zoom', action: () => { view.cx = 0; view.cy = 0; view.halfW = 2; schedule(); } },
      { type: 'button', label: 'Reset', action: () => reset() },
    ]);

    function rebuildRoots() {
      roots = rootsFor(state.poly, state.n);
      rootRGB = roots.map((_, k) => Palette.basinRGB(k, roots.length));
    }

    ctrl.onChange((key, val) => {
      if (key === 'n') { state.n = parseInt(val, 10); rebuildRoots(); }
      else if (key === 'poly') { state.poly = val; rebuildRoots(); }
      else if (key === 'maxIter') { state.maxIter = val | 0; }
      schedule();
    });

    function reset() {
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      view.cx = 0; view.cy = 0; view.halfW = 2;
      rebuildRoots(); schedule();
    }

    schedule(); composite();

    return {
      destroy() {
        anim.stop();
        canvas.removeEventListener('mousedown', onDown);
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
        canvas.removeEventListener('wheel', onWheel);
        container.innerHTML = '';
      },
      reset,
      setParam(k, v) {
        state[k] = (k === 'n' || k === 'maxIter') ? parseInt(v, 10) : v;
        if (k === 'n' || k === 'poly') rebuildRoots();
        schedule();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = NewtonBasins;
