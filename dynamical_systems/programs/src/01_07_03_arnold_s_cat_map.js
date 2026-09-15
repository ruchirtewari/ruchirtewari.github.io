/**
 * @file 01_07_03_arnold_s_cat_map.js
 * @chapter 1 — Examples and Basic Concepts
 * @sections §1.7, §1.8
 * @concept Hyperbolic toral automorphism — mixing and Poincaré recurrence on T²
 *
 * Arnold's cat map on the 2-torus T² = [0,1)²:
 *
 *   A = [[2, 1],     f(x,y) = A·(x,y)ᵀ mod 1.
 *        [1, 1]]
 *
 * A has eigenvalues φ² ≈ 2.618 (unstable) and φ⁻² ≈ 0.382 (stable), where
 * φ = (1+√5)/2.  The map is area-preserving (det A = 1), hyperbolic, and
 * Anosov — ergodic and mixing with entropy h = log(φ²) ≈ 0.962 nats/step.
 *
 * Implementation strategy
 * ───────────────────────
 * We render a procedural cat face on a 64×64 pixel grid, then precompute all
 * 48 frames of the full recurrence cycle by applying A once per step (iterative,
 * no matrix-power computation).  Animation steps through the precomputed frames
 * via setInterval — completely independent of DSUtils Anim timing details.
 * The face scrambles visibly by step 3–5 and returns EXACTLY at step 48.
 */

var ArnoldCatMap = (() => {
  const { Palette, Canvas, Controls } = DSUtils;

  const meta = {
    id: '01_07_03',
    title: "Arnold's Cat Map — Mixing then Poincaré Recurrence",
    chapter: 'Chapter 1 — Examples and Basic Concepts',
    sections: ['1.7', '1.8'],
    concept: 'A hyperbolic Anosov map scrambles an image, then Poincaré recurrence restores it',
    description:
      'A cat face is drawn on a 64×64 pixel grid.  Applying A=[[2,1],[1,1]] mod 64 ' +
      'scrambles it exponentially (h = log φ² ≈ 0.962 nats/step), yet the image returns ' +
      'exactly at step 48 via Poincaré recurrence.  ' +
      'Stable (cyan) and unstable (red) eigenvector directions are overlaid.',
  };

  const PHI      = (1 + Math.sqrt(5)) / 2;
  const LAM_PLUS = PHI * PHI;        // unstable eigenvalue ≈ 2.618
  const LAM_MIN  = 1 / LAM_PLUS;    // stable eigenvalue  ≈ 0.382
  const ENTROPY  = Math.log(LAM_PLUS); // ≈ 0.962 nats/step

  // Eigenvector angles in canvas coords (y-axis points down):
  //   unstable direction (φ, 1) → atan2(1, φ)
  //   stable   direction (1,−φ) → atan2(−φ, 1)
  const EU_ANG = Math.atan2(1,    PHI);
  const ES_ANG = Math.atan2(-PHI, 1  );

  const N      = 64;    // pixel grid dimension; period = 48 for N=64
  const PERIOD = 48;    // known period of A on Z_64²
  const CELL   = 7;     // display pixels per grid cell (64 × 7 = 448)
  const DISP   = N * CELL;  // 448

  // ── Procedural cat face ───────────────────────────────────────────────────
  function makeCatFace() {
    const oc = document.createElement('canvas');
    oc.width = N; oc.height = N;
    const c = oc.getContext('2d');

    const cx = N * 0.5, cy = N * 0.57, r = N * 0.36;

    c.fillStyle = '#1a1a2e';
    c.fillRect(0, 0, N, N);

    // Outer ears
    c.fillStyle = '#b86228';
    c.beginPath();
    c.moveTo(cx - r*0.85, cy - r*0.50);
    c.lineTo(cx - r*0.32, cy - r*1.22);
    c.lineTo(cx - r*0.05, cy - r*0.48);
    c.closePath(); c.fill();
    c.beginPath();
    c.moveTo(cx + r*0.85, cy - r*0.50);
    c.lineTo(cx + r*0.32, cy - r*1.22);
    c.lineTo(cx + r*0.05, cy - r*0.48);
    c.closePath(); c.fill();

    // Inner ears
    c.fillStyle = '#dd7898';
    c.beginPath();
    c.moveTo(cx - r*0.72, cy - r*0.54);
    c.lineTo(cx - r*0.34, cy - r*1.02);
    c.lineTo(cx - r*0.08, cy - r*0.52);
    c.closePath(); c.fill();
    c.beginPath();
    c.moveTo(cx + r*0.72, cy - r*0.54);
    c.lineTo(cx + r*0.34, cy - r*1.02);
    c.lineTo(cx + r*0.08, cy - r*0.52);
    c.closePath(); c.fill();

    // Face with radial gradient
    const grd = c.createRadialGradient(cx - r*0.1, cy - r*0.1, r*0.05, cx, cy, r);
    grd.addColorStop(0, '#e88844'); grd.addColorStop(1, '#bf5e20');
    c.fillStyle = grd;
    c.beginPath(); c.ellipse(cx, cy, r, r*0.88, 0, 0, Math.PI*2); c.fill();

    // Muzzle
    c.fillStyle = '#f2c484';
    c.beginPath(); c.ellipse(cx, cy + r*0.26, r*0.40, r*0.30, 0, 0, Math.PI*2); c.fill();

    // Forehead stripes
    c.strokeStyle = '#984818'; c.lineWidth = Math.max(1.2, N*0.014); c.lineCap = 'round';
    [-0.22, 0, 0.22].forEach(sx => {
      c.beginPath();
      c.moveTo(cx + sx*r, cy - r*0.32);
      c.lineTo(cx + sx*r*0.82, cy - r*0.72);
      c.stroke();
    });

    // Eyes
    const eyeX = r*0.32, eyeY = cy - r*0.14, eyeR = r*0.18;
    [-1, 1].forEach(s => {
      c.fillStyle = '#c8a010';
      c.beginPath(); c.ellipse(cx + s*eyeX, eyeY, eyeR, eyeR*0.85, 0, 0, Math.PI*2); c.fill();
      c.fillStyle = '#060606';
      c.beginPath(); c.ellipse(cx + s*eyeX, eyeY, eyeR*0.24, eyeR*0.70, 0, 0, Math.PI*2); c.fill();
      c.fillStyle = 'rgba(255,255,255,0.88)';
      c.beginPath(); c.arc(cx + s*eyeX - eyeR*0.22, eyeY - eyeR*0.25, eyeR*0.13, 0, Math.PI*2); c.fill();
    });

    // Nose
    c.fillStyle = '#d85868';
    c.beginPath();
    c.moveTo(cx, cy + r*0.08);
    c.lineTo(cx - r*0.10, cy - r*0.02);
    c.lineTo(cx + r*0.10, cy - r*0.02);
    c.closePath(); c.fill();

    // Mouth
    c.strokeStyle = '#884050'; c.lineWidth = Math.max(0.8, N*0.010); c.lineCap = 'round';
    c.beginPath(); c.moveTo(cx, cy + r*0.08); c.lineTo(cx, cy + r*0.19); c.stroke();
    c.beginPath();
    c.moveTo(cx - r*0.23, cy + r*0.21);
    c.quadraticCurveTo(cx - r*0.09, cy + r*0.30, cx, cy + r*0.19);
    c.quadraticCurveTo(cx + r*0.09, cy + r*0.30, cx + r*0.23, cy + r*0.21);
    c.stroke();

    // Whiskers
    c.strokeStyle = 'rgba(255,250,210,0.85)'; c.lineWidth = Math.max(0.6, N*0.008); c.lineCap = 'butt';
    [-r*0.10, 0, r*0.10].forEach(dy => {
      c.beginPath(); c.moveTo(cx - r*0.28, cy + dy + r*0.06); c.lineTo(cx - r*0.97, cy + dy - dy*0.20 + r*0.06); c.stroke();
      c.beginPath(); c.moveTo(cx + r*0.28, cy + dy + r*0.06); c.lineTo(cx + r*0.97, cy + dy - dy*0.20 + r*0.06); c.stroke();
    });

    return c.getImageData(0, 0, N, N).data;  // return raw Uint8ClampedArray
  }

  // ── Precompute all PERIOD frames ──────────────────────────────────────────
  // Applies A = [[2,1],[1,1]] mod N iteratively; avoids matrix-power computation.
  function buildFrames(src) {
    const frames = [src];  // frame 0 = original
    let cur = src;
    for (let k = 1; k < PERIOD; k++) {
      const next = new Uint8ClampedArray(N * N * 4);
      for (let j = 0; j < N; j++) {
        for (let i = 0; i < N; i++) {
          const di  = (2 * i + j) % N;
          const dj  = (i + j) % N;
          const si  = (j * N + i)  * 4;
          const di4 = (dj * N + di) * 4;
          next[di4]   = cur[si];
          next[di4+1] = cur[si+1];
          next[di4+2] = cur[si+2];
          next[di4+3] = cur[si+3];
        }
      }
      frames.push(next);
      cur = next;
    }
    return frames;
  }

  // ── init ─────────────────────────────────────────────────────────────────
  function init(container, params = {}) {
    const W = 560, H = 516;
    const IX = (W - DISP) / 2;  // 56 — left edge of image
    const IY = 36;               // top edge of image

    const { canvas, ctx } = Canvas.create(container, W, H);

    // Offscreen canvas: 64×64, used to blit pixel data → main canvas via drawImage
    const oc  = document.createElement('canvas');
    oc.width  = N; oc.height = N;
    const oct = oc.getContext('2d');

    // Build all frames once at startup
    const srcPixels = makeCatFace();
    const frames    = buildFrames(srcPixels);

    // Reusable ImageData to avoid repeated allocation in draw()
    const imgData = new ImageData(N, N);

    let step      = 0;
    let showEigen = true;
    let totalSteps = 0; // separate counter so "returned" triggers only after at least one full cycle

    // ── Arrow helper ──────────────────────────────────────────────────────
    function drawArrow(x1, y1, x2, y2, color, lbl) {
      const dx = x2-x1, dy = y2-y1, len = Math.hypot(dx, dy);
      const nx = dx/len, ny = dy/len, ah = 9, aw = 4;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]); ctx.globalAlpha = 0.72;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - nx*ah - ny*aw, y2 - ny*ah + nx*aw);
      ctx.lineTo(x2 - nx*ah + ny*aw, y2 - ny*ah - nx*aw);
      ctx.closePath(); ctx.fill();
      if (lbl) {
        ctx.font = '9px monospace'; ctx.fillStyle = color;
        ctx.textAlign = nx >= 0 ? 'left' : 'right';
        ctx.fillText(lbl, x2 + nx*4, y2 + ny*4 + 3);
      }
      ctx.restore();
    }

    // ── Main draw ─────────────────────────────────────────────────────────
    function draw() {
      Canvas.clear(ctx, W, H);

      const returned = (step === 0 && totalSteps > 0);

      // Blit current frame → offscreen canvas → main canvas (nearest-neighbour scale)
      imgData.data.set(frames[step]);
      oct.putImageData(imgData, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(oc, IX, IY, DISP, DISP);
      ctx.imageSmoothingEnabled = true;

      // Image border
      ctx.strokeStyle = returned ? Palette.blue : '#2a3a5c';
      ctx.lineWidth   = returned ? 2 : 1;
      ctx.strokeRect(IX, IY, DISP, DISP);

      // Eigenvector overlays
      if (showEigen) {
        const mx = IX + DISP/2, my = IY + DISP/2, alen = DISP * 0.40;
        const ux = Math.cos(EU_ANG), uy = Math.sin(EU_ANG);
        const sx = Math.cos(ES_ANG), sy = Math.sin(ES_ANG);
        drawArrow(mx - ux*alen, my - uy*alen, mx + ux*alen, my + uy*alen,
                  Palette.accent, 'λ⁺≈2.618 unstable');
        drawArrow(mx - sx*alen, my - sy*alen, mx + sx*alen, my + sy*alen,
                  Palette.blue, 'λ⁻≈0.382 stable');
      }

      // Return flash
      if (returned) {
        ctx.fillStyle = 'rgba(83,216,251,0.13)';
        ctx.fillRect(IX, IY, DISP, DISP);
        ctx.save();
        ctx.font = 'bold 24px monospace'; ctx.fillStyle = Palette.blue;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('✓ IMAGE RESTORED', IX + DISP/2, IY + DISP/2);
        ctx.restore();
      }

      // Header
      ctx.fillStyle = '#16213e'; ctx.fillRect(0, 0, W, IY);
      Canvas.label(ctx, meta.title, 10, 14, { color: Palette.blue, font: '11px monospace' });
      const toReturn = step === 0 ? PERIOD : PERIOD - step;
      Canvas.label(ctx,
        'step ' + step + ' / ' + PERIOD + '   ·   ' + toReturn + ' step' + (toReturn === 1 ? '' : 's') + ' to return',
        10, 28, { color: Palette.gold, font: '11px monospace' });

      // Progress bar
      ctx.fillStyle = '#0f3460'; ctx.fillRect(0, IY-4, W, 4);
      ctx.fillStyle = returned ? Palette.blue : Palette.accent;
      ctx.fillRect(0, IY-4, W * (step / PERIOD), 4);

      // Footer
      const FY = IY + DISP + 6;
      ctx.fillStyle = '#16213e'; ctx.fillRect(0, FY, W, H - FY);
      Canvas.label(ctx,
        'A=[[2,1],[1,1]]   λ⁺=φ²≈2.618   λ⁻=φ⁻²≈0.382   h=log(φ²)≈0.962 nats   N=64  period=48',
        10, FY + 17, { color: Palette.muted, font: '10px monospace' });

      // Original thumbnail — blit frame 0
      imgData.data.set(frames[0]);
      oct.putImageData(imgData, 0, 0);
      const TH = 54, TX = W - TH - 8, TY = FY + 5;
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(oc, TX, TY, TH, TH);
      ctx.imageSmoothingEnabled = true;
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1;
      ctx.strokeRect(TX, TY, TH, TH);
      Canvas.label(ctx, 'original', TX + TH/2, TY + TH + 12,
        { color: Palette.muted, font: '8px monospace', align: 'center' });
    }

    // ── Animation via setInterval — no rAF delta accumulation ────────────
    let intervalId = null;

    function startPlay() {
      if (intervalId) return;
      intervalId = setInterval(() => {
        step = (step + 1) % PERIOD;
        totalSteps++;
        ctrl.set('step', step);
        draw();
      }, 500);
    }

    function stopPlay() {
      if (intervalId) { clearInterval(intervalId); intervalId = null; }
    }

    // ── Controls ──────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'step', label: 'step', min: 0, max: PERIOD - 1, step: 1,
        value: step, format: v => String(v | 0) },
      { type: 'checkbox', key: 'showEigen', label: 'eigenvectors', value: showEigen },
      { type: 'button', label: 'Play / Pause', action: () => {
        if (intervalId) stopPlay(); else startPlay();
      }},
      { type: 'button', label: '+ Step', action: () => {
        stopPlay();
        step = (step + 1) % PERIOD;
        totalSteps++;
        ctrl.set('step', step);
        draw();
      }},
      { type: 'button', label: 'Reset', action: () => {
        stopPlay(); step = 0; totalSteps = 0;
        ctrl.set('step', 0); draw();
      }},
    ]);

    ctrl.onChange((key, val) => {
      if (key === 'step') { step = val | 0; draw(); }
      else if (key === 'showEigen') { showEigen = val; draw(); }
    });

    // Start playing and show initial frame
    draw();
    startPlay();

    return {
      destroy() { stopPlay(); container.innerHTML = ''; },
      reset() { stopPlay(); step = 0; totalSteps = 0; ctrl.set('step', 0); draw(); },
      setParam(k, v) {
        if (k === 'step') { step = +v % PERIOD; draw(); }
        else if (k === 'showEigen') { showEigen = !!v; draw(); }
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = ArnoldCatMap;
