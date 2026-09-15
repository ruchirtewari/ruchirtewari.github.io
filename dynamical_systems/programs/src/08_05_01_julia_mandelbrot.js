/**
 * @file 08_05_01_julia_mandelbrot.js
 * @chapter 8 — Complex Dynamics
 * @sections §8.5, §8.6
 * @concept Julia set, Fatou set, Mandelbrot set, connectedness dichotomy
 *
 * The quadratic family  f_c(z) = z² + c  on the Riemann sphere.
 *
 * MANDELBROT SET  M = { c ∈ ℂ : the orbit of 0 under f_c stays bounded }.
 * Equivalently c ∈ M ⇔ |f_c^n(0)| ≤ 2 for all n. We colour c ∉ M by its
 * escape time (how many iterations until |z| > 2), smoothed to remove the
 * banding, and paint c ∈ M black.
 *
 * JULIA SET  J(f_c) = ∂{ z₀ : orbit of z₀ is bounded }. For a fixed c we sweep
 * the INITIAL point z₀ over the plane and colour by its escape time. The
 * Fatou/Julia dichotomy (Chapter 8):
 *
 *     c ∈ M  ⇔  J(f_c) is connected;
 *     c ∉ M  ⇔  J(f_c) is a totally disconnected Cantor dust.
 *
 * Click anywhere in the Mandelbrot panel to set c and watch the Julia set on
 * the right morph across this dichotomy in real time.
 *
 * The escape iteration is the classic optimised form (squares reused):
 *     re² + im² > 4  is the escape test; z ↦ z² + c is
 *     im ← 2·re·im + c_im ;  re ← re² − im² + c_re.
 *
 * Both panels render in 8 horizontal strips, one strip per animation frame,
 * so large max-iteration counts never freeze the UI; a progress bar tracks it.
 */

var JuliaMandelbrot = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '08_05_01',
    title: 'Julia Set & Mandelbrot Set — Interactive Explorer',
    chapter: 'Chapter 8 — Complex Dynamics',
    sections: ['8.5', '8.6'],
    concept: 'Escape-time portraits of z²+c: the Mandelbrot set and its Julia sets',
    description:
      'Side-by-side Mandelbrot and Julia explorers with click-to-pick c, scroll-to-zoom, ' +
      'drag-to-pan, smooth escape colouring (inferno / hsl / gold-blue), and chunked ' +
      'strip rendering with a progress bar.',
  };

  const STRIPS = 8;
  const LW = 270, RW = 270, PANEL_H = 400, PANEL_Y = 32, DIV_X = 270, R_X = 290;

  const DEFAULTS = { maxIter: 200, colormap: 'inferno', showJulia: true, c_re: -0.4, c_im: 0.6 };

  // Optimised escape-time; returns [iterationCount, |z|²atEscape].
  function escape(zr, zi, cr, ci, maxIter) {
    let re = zr, im = zi;
    for (let n = 0; n < maxIter; n++) {
      const re2 = re * re, im2 = im * im;
      if (re2 + im2 > 4) return [n, re2 + im2];
      im = 2 * re * im + ci;
      re = re2 - im2 + cr;
    }
    return [maxIter, re * re + im * im];
  }

  function hsl2rgb(h, s, l) {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const hk = ((h % 360) + 360) % 360 / 360;
    const f = t => {
      t = (t + 1) % 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    return [(f(hk + 1 / 3) * 255) | 0, (f(hk) * 255) | 0, (f(hk - 1 / 3) * 255) | 0];
  }

  // Map escape data → [r,g,b] under the chosen colormap.
  function colorFor(n, z2, maxIter, mode) {
    if (n >= maxIter) return [0, 0, 0];                     // interior: bounded
    const smooth = n + 1 - Math.log(Math.log(Math.sqrt(z2))) / Math.LN2;
    const t = Math.max(0, Math.min(1, smooth / maxIter));
    if (mode === 'inferno') return Palette.fractalSmooth(n, maxIter, z2);
    if (mode === 'hsl') return hsl2rgb(360 * t, 0.85, 0.55);
    // gold-blue gradient
    const B = [83, 216, 251], G = [245, 166, 35];
    const tt = Math.pow(t, 0.5);
    return [B[0] + (G[0] - B[0]) * tt | 0, B[1] + (G[1] - B[1]) * tt | 0, B[2] + (G[2] - B[2]) * tt | 0];
  }

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 440;
    const { canvas, ctx } = Canvas.create(container, W, H);

    // Per-panel offscreen buffer + complex-plane view.
    function makePanel(w, h, cx, cy, halfW) {
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      const cctx = c.getContext('2d');
      return {
        canvas: c, ctx: cctx, w, h,
        img: cctx.createImageData(w, h),
        view: { cx, cy, halfW, baseHalfW: halfW },
        nextStrip: STRIPS,          // STRIPS = done; < STRIPS = pending
      };
    }
    const mandel = makePanel(LW, PANEL_H, -0.5, 0, 1.6);
    const julia  = makePanel(RW, PANEL_H, 0, 0, 1.6);

    // Complex coordinate of a panel pixel.
    function pixToC(panel, px, py) {
      const halfH = panel.view.halfW * (panel.h / panel.w);
      return [
        panel.view.cx + (px / panel.w - 0.5) * 2 * panel.view.halfW,
        panel.view.cy + (0.5 - py / panel.h) * 2 * halfH,
      ];
    }

    function scheduleRender(panel) { panel.nextStrip = 0; ensureLoop(); }
    function scheduleBoth() { scheduleRender(mandel); if (state.showJulia) scheduleRender(julia); }

    // Render the panel's next pending strip into its ImageData.
    function renderStrip(panel, isMandel) {
      const strip = panel.nextStrip;
      const y0 = Math.floor(strip * panel.h / STRIPS);
      const y1 = Math.floor((strip + 1) * panel.h / STRIPS);
      const data = panel.img.data;
      const maxIter = state.maxIter | 0;
      const mode = state.colormap;
      const cr = state.c_re, ci = state.c_im;
      for (let py = y0; py < y1; py++) {
        for (let px = 0; px < panel.w; px++) {
          const [a, b] = pixToC(panel, px, py);
          let n, z2;
          if (isMandel) ([n, z2] = escape(0, 0, a, b, maxIter));
          else          ([n, z2] = escape(a, b, cr, ci, maxIter));
          const [r, g, bl] = colorFor(n, z2, maxIter, mode);
          const idx = (py * panel.w + px) * 4;
          data[idx] = r; data[idx + 1] = g; data[idx + 2] = bl; data[idx + 3] = 255;
        }
      }
      panel.ctx.putImageData(panel.img, 0, 0, 0, y0, panel.w, y1 - y0);
      panel.nextStrip++;
    }

    // ── render loop: one strip of pending work per frame ──────────────────
    let loopRunning = false;
    const anim = Anim.loop(() => {
      let worked = false;
      if (mandel.nextStrip < STRIPS) { renderStrip(mandel, true); worked = true; }
      else if (state.showJulia && julia.nextStrip < STRIPS) { renderStrip(julia, false); worked = true; }
      composite();
      if (!worked && mandel.nextStrip >= STRIPS && julia.nextStrip >= STRIPS) { anim.stop(); loopRunning = false; }
    });
    function ensureLoop() { if (!loopRunning) { loopRunning = true; anim.start(); } }

    // ── composite offscreen buffers + chrome onto the visible canvas ──────
    let panDX = 0, panDY = 0, panPanel = null;    // live-drag pixel offset
    function composite() {
      Canvas.clear(ctx, W, H);

      // Mandelbrot (left).
      ctx.drawImage(mandel.canvas, (panPanel === mandel ? panDX : 0), PANEL_Y + (panPanel === mandel ? panDY : 0), LW, PANEL_H);
      // Julia (right).
      if (state.showJulia) {
        ctx.drawImage(julia.canvas, R_X + (panPanel === julia ? panDX : 0), PANEL_Y + (panPanel === julia ? panDY : 0), RW, PANEL_H);
      } else {
        ctx.fillStyle = Palette.surface; ctx.fillRect(R_X, PANEL_Y, RW, PANEL_H);
        Canvas.label(ctx, 'Julia set hidden', R_X + RW / 2, PANEL_Y + PANEL_H / 2, { color: Palette.muted, align: 'center' });
      }

      // Panel frames + divider.
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(0.5, PANEL_Y + 0.5, LW - 1, PANEL_H - 1);
      ctx.strokeRect(R_X + 0.5, PANEL_Y + 0.5, RW - 1, PANEL_H - 1);
      ctx.strokeStyle = Palette.card; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(DIV_X + 10, PANEL_Y); ctx.lineTo(DIV_X + 10, PANEL_Y + PANEL_H); ctx.stroke();

      // Marker of c inside the Mandelbrot panel.
      const halfH = mandel.view.halfW * (mandel.h / mandel.w);
      const mpx = (state.c_re - mandel.view.cx) / (2 * mandel.view.halfW) * mandel.w + mandel.w / 2;
      const mpy = mandel.h / 2 - (state.c_im - mandel.view.cy) / (2 * halfH) * mandel.h;
      if (mpx >= 0 && mpx <= LW && mpy >= 0 && mpy <= PANEL_H) {
        Canvas.dot(ctx, mpx, PANEL_Y + mpy, 3.5, Palette.accent);
        ctx.strokeStyle = Palette.white; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(mpx, PANEL_Y + mpy, 6, 0, Math.PI * 2); ctx.stroke();
      }

      // Progress bars while rendering.
      progressBar(mandel, 0);
      if (state.showJulia) progressBar(julia, R_X);

      // Labels.
      Canvas.label(ctx, 'Mandelbrot Set', 8, 20, { color: Palette.blue, font: '12px monospace' });
      Canvas.label(ctx, `Julia Set: c = (${state.c_re.toFixed(3)}, ${state.c_im.toFixed(3)})`, R_X + 4, 20, { color: Palette.gold, font: '12px monospace' });
      Canvas.label(ctx, `zoom ${(mandel.view.baseHalfW / mandel.view.halfW).toFixed(1)}×`, 8, PANEL_Y + PANEL_H - 6, { color: Palette.muted, font: '10px monospace' });
      if (state.showJulia)
        Canvas.label(ctx, `zoom ${(julia.view.baseHalfW / julia.view.halfW).toFixed(1)}×`, R_X + 4, PANEL_Y + PANEL_H - 6, { color: Palette.muted, font: '10px monospace' });
    }

    function progressBar(panel, xOff) {
      if (panel.nextStrip >= STRIPS) return;
      const frac = panel.nextStrip / STRIPS;
      ctx.fillStyle = Palette.card; ctx.fillRect(xOff + 4, PANEL_Y + 4, panel.w - 8, 4);
      ctx.fillStyle = Palette.blue; ctx.fillRect(xOff + 4, PANEL_Y + 4, (panel.w - 8) * frac, 4);
    }

    // ── pointer interaction ───────────────────────────────────────────────
    function locate(e) {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const ly = y - PANEL_Y;
      if (ly < 0 || ly > PANEL_H) return null;
      if (x >= 0 && x <= LW) return { panel: mandel, isMandel: true, lx: x, ly };
      if (x >= R_X && x <= R_X + RW && state.showJulia) return { panel: julia, isMandel: false, lx: x - R_X, ly };
      return null;
    }

    let down = null, moved = false;
    const onDown = e => {
      const hit = locate(e);
      if (!hit) return;
      down = hit; moved = false; panDX = 0; panDY = 0; panPanel = hit.panel;
    };
    const onMove = e => {
      if (!down) return;
      const rect = canvas.getBoundingClientRect();
      const nx = e.clientX - rect.left, ny = e.clientY - rect.top;
      panDX = nx - (down.lx + (down.isMandel ? 0 : R_X));
      panDY = ny - (down.ly + PANEL_Y);
      if (Math.abs(panDX) > 3 || Math.abs(panDY) > 3) moved = true;
      composite();
    };
    const onUp = e => {
      if (!down) return;
      const panel = down.panel;
      if (moved) {
        // Apply pan: shift centre opposite to the pixel drag.
        const halfH = panel.view.halfW * (panel.h / panel.w);
        panel.view.cx -= panDX / panel.w * 2 * panel.view.halfW;
        panel.view.cy += panDY / panel.h * 2 * halfH;
        panDX = 0; panDY = 0; panPanel = null;
        scheduleRender(panel);
      } else {
        // A click: on the Mandelbrot panel this picks c for the Julia set.
        panPanel = null;
        if (down.isMandel) {
          const [a, b] = pixToC(mandel, down.lx, down.ly);
          state.c_re = a; state.c_im = b;
          ctrl.set('c_re', a); ctrl.set('c_im', b);
          if (state.showJulia) scheduleRender(julia);
          composite();
        }
      }
      down = null;
    };
    const onWheel = e => {
      const hit = locate(e);
      if (!hit) return;
      e.preventDefault();
      const panel = hit.panel;
      const [ax, ay] = pixToC(panel, hit.lx, hit.ly);   // fixed point of the zoom
      const factor = e.deltaY < 0 ? 0.8 : 1.25;
      panel.view.cx = ax + (panel.view.cx - ax) * factor;
      panel.view.cy = ay + (panel.view.cy - ay) * factor;
      panel.view.halfW *= factor;
      scheduleRender(panel);
    };
    canvas.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    // ── controls ──────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'maxIter', label: 'max iter', min: 50, max: 500, step: 10, value: state.maxIter, format: v => v | 0 },
      {
        type: 'select', key: 'colormap', label: 'colormap', value: state.colormap,
        options: [{ value: 'inferno', label: 'inferno' }, { value: 'hsl', label: 'hsl' }, { value: 'goldblue', label: 'gold-blue' }],
      },
      { type: 'slider', key: 'c_re', label: 'c re', min: -2, max: 1, step: 0.005, value: state.c_re, format: v => v.toFixed(3) },
      { type: 'slider', key: 'c_im', label: 'c im', min: -1.5, max: 1.5, step: 0.005, value: state.c_im, format: v => v.toFixed(3) },
      { type: 'checkbox', key: 'showJulia', label: 'Show Julia', value: state.showJulia },
      { type: 'button', label: 'Reset M zoom', action: () => { resetView(mandel, -0.5, 0, 1.6); } },
      { type: 'button', label: 'Reset J zoom', action: () => { resetView(julia, 0, 0, 1.6); } },
    ]);

    ctrl.onChange((key, val) => {
      if (key === 'colormap') { state.colormap = val; scheduleBoth(); }
      else if (key === 'showJulia') { state.showJulia = val; if (val) scheduleRender(julia); composite(); }
      else if (key === 'maxIter') { state.maxIter = val | 0; scheduleBoth(); }
      else if (key === 'c_re' || key === 'c_im') { state[key] = val; if (state.showJulia) scheduleRender(julia); composite(); }
    });

    function resetView(panel, cx, cy, halfW) {
      panel.view.cx = cx; panel.view.cy = cy; panel.view.halfW = halfW;
      scheduleRender(panel);
    }

    function reset() {
      Object.assign(state, DEFAULTS);
      Object.keys(DEFAULTS).forEach(k => ctrl.set(k, DEFAULTS[k]));
      resetView(mandel, -0.5, 0, 1.6);
      resetView(julia, 0, 0, 1.6);
    }

    // Kick off the first render.
    scheduleBoth();
    composite();

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
        state[k] = (k === 'maxIter') ? (v | 0) : v;
        scheduleBoth(); composite();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = JuliaMandelbrot;
