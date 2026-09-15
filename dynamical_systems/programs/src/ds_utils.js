/**
 * ds_utils.js — shared utilities for Brin-Stuck dynamical systems visualizations
 *
 * Every program in programs/src/ imports from this module via:
 *   const { Palette, Canvas, Controls, Anim, Math2D } = DSUtils;
 *
 * Design goals:
 *   - No external dependencies (pure vanilla JS)
 *   - Each visualization is a self-contained module: { meta, init(container, params) }
 *   - init() returns { destroy(), setParam(key, val), reset() }
 *   - Dark-theme palette matches the book's HTML reader
 */

var DSUtils = (() => {

  // ── Colour palette ────────────────────────────────────────────────────────
  const Palette = {
    bg:      '#1a1a2e',
    surface: '#16213e',
    card:    '#0f3460',
    accent:  '#e94560',   // red — fixed points, attractors
    blue:    '#53d8fb',   // cyan — stable manifolds, ergodic averages
    gold:    '#f5a623',   // gold — unstable manifolds, entropy curves
    green:   '#4caf50',
    muted:   '#8892a4',
    text:    '#e0e0e0',
    white:   '#ffffff',

    /** Return a CSS colour from a 0..1 "heat" value (blue→cyan→gold→red). */
    heat(t) {
      t = Math.max(0, Math.min(1, t));
      if (t < 0.33) {
        const s = t / 0.33;
        return `rgb(${lerp(83,245,s)|0},${lerp(216,166,s)|0},${lerp(251,35,s)|0})`;
      } else if (t < 0.66) {
        const s = (t - 0.33) / 0.33;
        return `rgb(${lerp(245,233,s)|0},${lerp(166,69,s)|0},${lerp(35,96,s)|0})`;
      } else {
        const s = (t - 0.66) / 0.34;
        return `rgb(${lerp(233,255,s)|0},${lerp(69,255,s)|0},${lerp(96,255,s)|0})`;
      }
    },

    /** Smooth gradient from cold (blue) to hot (red) for fractal colouring. */
    fractal(n, maxN) {
      if (n >= maxN) return '#000000';
      const t = n / maxN;
      const h = (0.67 - t * 0.67) * 360;   // hue: 240° (blue) → 0° (red)
      return `hsl(${h}, 80%, 55%)`;
    },

    /** Smooth colouring for escape-time fractals (anti-banding). */
    fractalSmooth(n, maxN, z2) {
      if (n >= maxN) return [0, 0, 0];
      // Renormalize: subtract log of log of |z|
      const smooth = n + 1 - Math.log(Math.log(Math.sqrt(z2))) / Math.LN2;
      const t = Math.max(0, Math.min(1, smooth / maxN));
      // Inferno-ish palette
      const r = Math.min(255, (t * 3.0 * 255) | 0);
      const g = Math.min(255, (Math.max(0, t * 3 - 1) * 255) | 0);
      const b = Math.min(255, (Math.max(0, t * 3 - 2) * 255) | 0);
      return [r, g, b];
    },

    /** N distinct hues for basin colouring (Newton's method etc.). */
    basin(k, n) {
      const h = (k / n) * 360;
      return `hsl(${h}, 70%, 45%)`;
    },

    basinRGB(k, n) {
      const h = (k / n) * 360;
      // Convert hsl to rgb
      const s = 0.70, l = 0.45;
      const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      const p = 2 * l - q;
      const hk = h / 360;
      function hue2rgb(t) {
        if (t < 0) t += 1; if (t > 1) t -= 1;
        if (t < 1/6) return p + (q - p) * 6 * t;
        if (t < 1/2) return q;
        if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
        return p;
      }
      return [
        (hue2rgb(hk + 1/3) * 255) | 0,
        (hue2rgb(hk) * 255) | 0,
        (hue2rgb(hk - 1/3) * 255) | 0,
      ];
    },
  };

  function lerp(a, b, t) { return a + (b - a) * t; }

  // ── Canvas helpers ────────────────────────────────────────────────────────
  const Canvas = {
    /**
     * Create a <canvas> inside container, return { canvas, ctx }.
     * Handles devicePixelRatio for crisp rendering on retina screens.
     */
    create(container, width, height) {
      const canvas = document.createElement('canvas');
      const dpr = window.devicePixelRatio || 1;
      canvas.width  = width  * dpr;
      canvas.height = height * dpr;
      canvas.style.width  = width  + 'px';
      canvas.style.height = height + 'px';
      canvas.style.display = 'block';
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);
      container.appendChild(canvas);
      return { canvas, ctx, width, height, dpr };
    },

    /** Fill with background colour. */
    clear(ctx, width, height, color = Palette.bg) {
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, width, height);
    },

    /** Draw labelled axes on a math canvas that maps [xMin,xMax]×[yMin,yMax]. */
    axes(ctx, width, height, xMin, xMax, yMin, yMax, opts = {}) {
      const {
        color     = Palette.muted,
        tickColor = Palette.surface,
        label     = true,
        font      = '11px monospace',
      } = opts;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([]);

      // Map math coords → canvas coords
      const mx = x => ((x - xMin) / (xMax - xMin)) * width;
      const my = y => height - ((y - yMin) / (yMax - yMin)) * height;

      // x-axis (y=0 if in range)
      const y0 = my(0);
      if (y0 >= 0 && y0 <= height) {
        ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(width, y0); ctx.stroke();
      }
      // y-axis
      const x0 = mx(0);
      if (x0 >= 0 && x0 <= width) {
        ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, height); ctx.stroke();
      }
    },

    /**
     * Draw a labelled plot inside a math viewport.
     * Returns {toCanvas(x,y), toMath(cx,cy)} coordinate converters.
     */
    viewport(width, height, xMin, xMax, yMin, yMax) {
      return {
        toCanvas: (x, y) => [
          ((x - xMin) / (xMax - xMin)) * width,
          height - ((y - yMin) / (yMax - yMin)) * height,
        ],
        toMath: (cx, cy) => [
          xMin + (cx / width) * (xMax - xMin),
          yMin + (1 - cy / height) * (yMax - yMin),
        ],
      };
    },

    /** Stroke a polyline defined by array of [x,y] canvas points. */
    polyline(ctx, pts, color, lineWidth = 1.5) {
      if (pts.length < 2) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
    },

    /** Draw a filled circle. */
    dot(ctx, x, y, r, color) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.TAU || 2 * Math.PI);
      ctx.fill();
    },

    /** Render text with a semi-transparent background box for readability. */
    label(ctx, text, x, y, opts = {}) {
      const { color = Palette.text, font = '12px monospace', align = 'left' } = opts;
      ctx.font = font;
      ctx.textAlign = align;
      ctx.fillStyle = 'rgba(26,26,46,0.75)';
      const w = ctx.measureText(text).width;
      ctx.fillRect(x - (align === 'center' ? w / 2 : 0) - 2, y - 13, w + 4, 16);
      ctx.fillStyle = color;
      ctx.fillText(text, x, y);
    },
  };

  // ── Control panel ─────────────────────────────────────────────────────────
  const Controls = {
    /**
     * Build a control panel below the canvas.
     * defs: array of control descriptors:
     *   { type:'slider', key, label, min, max, step, value, format }
     *   { type:'select', key, label, options:[{value,label}], value }
     *   { type:'button', label, action }
     *   { type:'checkbox', key, label, value }
     *
     * Returns { get(key), set(key,val), onChange(fn) }
     */
    build(container, defs) {
      const state = {};
      const listeners = [];

      const panel = document.createElement('div');
      panel.style.cssText = `
        display:flex; flex-wrap:wrap; gap:10px; padding:10px 0;
        font-family:monospace; font-size:12px; color:${Palette.text};
        align-items:flex-end;
      `;

      defs.forEach(def => {
        if (def.type === 'slider') {
          state[def.key] = def.value;
          const wrap = document.createElement('div');
          wrap.style.cssText = 'display:flex;flex-direction:column;gap:3px;min-width:120px';

          const valSpan = document.createElement('span');
          const fmt = def.format || (v => +v.toFixed(4));
          valSpan.textContent = `${def.label}: ${fmt(def.value)}`;
          valSpan.style.color = Palette.blue;

          const slider = document.createElement('input');
          slider.type = 'range';
          slider.min   = def.min;
          slider.max   = def.max;
          slider.step  = def.step || (def.max - def.min) / 200;
          slider.value = def.value;
          slider.style.cssText = `width:100%;accent-color:${Palette.blue}`;

          slider.addEventListener('input', () => {
            const v = parseFloat(slider.value);
            state[def.key] = v;
            valSpan.textContent = `${def.label}: ${fmt(v)}`;
            listeners.forEach(fn => fn(def.key, v));
          });

          wrap.append(valSpan, slider);
          panel.appendChild(wrap);

        } else if (def.type === 'select') {
          state[def.key] = def.value;
          const wrap = document.createElement('div');
          wrap.style.cssText = 'display:flex;flex-direction:column;gap:3px';

          const lbl = document.createElement('span');
          lbl.textContent = def.label;
          lbl.style.color = Palette.muted;

          const sel = document.createElement('select');
          sel.style.cssText = `background:${Palette.card};color:${Palette.text};border:1px solid ${Palette.muted};padding:3px 6px;font-family:monospace;font-size:12px;border-radius:3px`;
          (def.options || []).forEach(opt => {
            const o = document.createElement('option');
            o.value = opt.value; o.textContent = opt.label;
            if (opt.value === def.value) o.selected = true;
            sel.appendChild(o);
          });
          sel.addEventListener('change', () => {
            state[def.key] = sel.value;
            listeners.forEach(fn => fn(def.key, sel.value));
          });

          wrap.append(lbl, sel);
          panel.appendChild(wrap);

        } else if (def.type === 'button') {
          const btn = document.createElement('button');
          btn.textContent = def.label;
          btn.style.cssText = `
            padding:5px 14px; background:${Palette.card}; color:${Palette.blue};
            border:1px solid ${Palette.blue}; border-radius:3px; cursor:pointer;
            font-family:monospace; font-size:12px;
          `;
          btn.addEventListener('click', def.action);
          panel.appendChild(btn);

        } else if (def.type === 'checkbox') {
          state[def.key] = def.value;
          const wrap = document.createElement('label');
          wrap.style.cssText = 'display:flex;align-items:center;gap:5px;cursor:pointer';

          const cb = document.createElement('input');
          cb.type = 'checkbox'; cb.checked = def.value;
          cb.style.accentColor = Palette.blue;
          cb.addEventListener('change', () => {
            state[def.key] = cb.checked;
            listeners.forEach(fn => fn(def.key, cb.checked));
          });

          const lbl = document.createElement('span');
          lbl.textContent = def.label;

          wrap.append(cb, lbl);
          panel.appendChild(wrap);
        }
      });

      container.appendChild(panel);

      return {
        get: k => state[k],
        set: (k, v) => { state[k] = v; },
        onChange: fn => listeners.push(fn),
      };
    },
  };

  // ── Animation loop ────────────────────────────────────────────────────────
  const Anim = {
    /**
     * Run fn(dt) on each frame. Returns { stop, start, toggle }.
     * dt = time since last frame in seconds.
     */
    loop(fn) {
      let rafId = null, running = false, last = 0;
      const tick = (ts) => {
        const dt = Math.min((ts - last) / 1000, 0.1);
        last = ts;
        fn(dt);
        if (running) rafId = requestAnimationFrame(tick);
      };
      return {
        start() { if (!running) { running = true; last = performance.now(); rafId = requestAnimationFrame(tick); } },
        stop()  { running = false; if (rafId) { cancelAnimationFrame(rafId); rafId = null; } },
        toggle() { running ? this.stop() : this.start(); },
        isRunning: () => running,
      };
    },

    /** Run fn once asynchronously (for heavy one-shot renders). */
    defer(fn) { setTimeout(fn, 0); },
  };

  // ── Numerical / math utilities ────────────────────────────────────────────
  const Math2D = {
    /** 4th-order Runge-Kutta step for ODE dy/dt = f(t, y). y is an array. */
    rk4(f, t, y, h) {
      const k1 = f(t,          y);
      const k2 = f(t + h / 2, y.map((yi, i) => yi + h / 2 * k1[i]));
      const k3 = f(t + h / 2, y.map((yi, i) => yi + h / 2 * k2[i]));
      const k4 = f(t + h,     y.map((yi, i) => yi + h       * k3[i]));
      return y.map((yi, i) => yi + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    },

    /** Complex multiply: (a+bi)(c+di) → [ac-bd, ad+bc] */
    cmul(a, b, c, d) { return [a * c - b * d, a * d + b * c]; },

    /** Complex add */
    cadd(a, b, c, d) { return [a + c, b + d]; },

    /** |z|² for z = [re, im] */
    cabs2(re, im) { return re * re + im * im; },

    /** Lerp */
    lerp,

    /** Clamp */
    clamp: (v, lo, hi) => Math.max(lo, Math.min(hi, v)),

    /** Modulo that always returns a non-negative result (true mathematical mod). */
    mod: (a, n) => ((a % n) + n) % n,

    /** Golden ratio */
    PHI: (1 + Math.sqrt(5)) / 2,

    /** 2π */
    TAU: 2 * Math.PI,

    /** Spectral radius of a real square matrix (dominant eigenvalue magnitude). */
    spectralRadius(A) {
      const n = A.length;
      // Power iteration
      let v = Array.from({ length: n }, () => Math.random());
      let norm = Math.hypot(...v);
      v = v.map(x => x / norm);
      let rho = 0;
      for (let iter = 0; iter < 200; iter++) {
        const Av = A.map(row => row.reduce((s, aij, j) => s + aij * v[j], 0));
        rho = Math.max(...Av.map(Math.abs));
        if (rho === 0) break;
        v = Av.map(x => x / rho);
      }
      return rho;
    },

    /** Histogram: count values in n_bins bins over [lo, hi]. */
    histogram(data, nBins, lo, hi) {
      const bins = new Float64Array(nBins);
      const scale = nBins / (hi - lo);
      for (const x of data) {
        const b = ((x - lo) * scale) | 0;
        if (b >= 0 && b < nBins) bins[b]++;
      }
      return bins;
    },
  };

  // ── Module host (optional iframe embedding) ───────────────────────────────
  /**
   * Wrap a visualization init() call inside a styled card.
   * Use this in the HTML reader to embed any program:
   *   DSUtils.embed(document.getElementById('target'), CobwebDiagram);
   */
  function embed(container, module, params = {}) {
    const card = document.createElement('div');
    card.style.cssText = `
      background:${Palette.surface}; border:1px solid #2a3a5c;
      border-radius:6px; padding:14px; margin:16px 0;
    `;

    const header = document.createElement('div');
    header.style.cssText = `
      display:flex; justify-content:space-between; align-items:center;
      margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid #2a3a5c;
    `;
    const title = document.createElement('span');
    title.textContent = module.meta?.title || 'Visualization';
    title.style.cssText = `color:${Palette.blue}; font-family:monospace; font-size:13px; font-weight:bold`;

    const chTag = document.createElement('span');
    chTag.textContent = module.meta?.chapter || '';
    chTag.style.cssText = `color:${Palette.muted}; font-family:monospace; font-size:11px`;

    header.append(title, chTag);
    card.appendChild(header);

    const handle = module.init(card, params);
    container.appendChild(card);
    return handle;
  }

  // Public API
  return { Palette, Canvas, Controls, Anim, Math2D, lerp, embed };
})();

// Make available as both module export and global
if (typeof module !== 'undefined') module.exports = DSUtils;
