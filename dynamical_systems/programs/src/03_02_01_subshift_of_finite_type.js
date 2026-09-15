/**
 * 03_02_01_subshift_of_finite_type.js
 * ─────────────────────────────────────────────────────────────────────────
 * Subshift of Finite Type — Transition Graph & Perron–Frobenius (§3.2–§3.4)
 *
 * A subshift of finite type (SFT) is the set of bi-infinite (or one-sided)
 * sequences over an alphabet {1..n} in which a symbol i may be followed by j
 * only when the 0/1 transition matrix A has A[i][j] = 1.  Equivalently it is
 * the set of infinite walks on the directed graph whose adjacency matrix is A.
 *
 * The Perron–Frobenius theorem: for an irreducible non-negative A there is a
 * dominant eigenvalue ρ = ρ(A) > 0 (the spectral radius) with a strictly
 * positive right eigenvector v (the Perron eigenvector).  Two consequences the
 * chapter draws out are visualised here:
 *
 *   • TOPOLOGICAL ENTROPY.  h(σ_A) = log ρ(A).  The number of period-n points
 *     of the shift equals the number of length-n cycles in the graph, i.e.
 *     Tr(Aⁿ) = Σ λᵢⁿ ≈ ρⁿ.  Hence log Tr(Aⁿ) is asymptotically linear in n
 *     with slope log ρ — the right-hand panel fits exactly this line.
 *   • STATIONARY STRUCTURE.  The (normalised) Perron eigenvector gives the
 *     relative weights of the states; node sizes in the graph and the middle
 *     bar chart both display it.
 *
 * Panels (canvas 560×380):
 *   LEFT   — directed transition graph; node area ∝ Perron entry; an animated
 *            random walk (blinking dot) hops along admissible edges.
 *   MIDDLE — bar chart of the Perron eigenvector (stationary distribution).
 *   RIGHT  — log Tr(Aⁿ) vs n (n = 1..15) with the fitted slope-log ρ line.
 * ─────────────────────────────────────────────────────────────────────────
 */

var SubshiftFiniteType = (() => {
  const { Palette, Canvas, Controls, Anim, Math2D } = DSUtils;

  const meta = {
    id: '03_02_01_subshift_of_finite_type',
    title: 'Subshift of Finite Type — Transition Graph & Perron–Frobenius',
    chapter: 'Chapter 3 · §3.2–§3.4',
    sections: ['3.2', '3.3', '3.4'],
    concept: 'SFT, transition matrix, Perron–Frobenius, periodic-orbit counting',
    description:
      'Directed transition graph of an SFT with Perron eigenvector node ' +
      'weights, stationary bar chart, and exponential periodic-orbit growth ' +
      'Tr(Aⁿ) ~ ρⁿ on a log axis.',
  };

  // ── Preset transition matrices ────────────────────────────────────────
  const PRESETS = {
    fibonacci:   [[1, 1], [1, 0]],
    golden_mean: [[1, 1, 0], [0, 0, 1], [1, 0, 0]],
    full3:       [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
  };

  const W = 560, H = 380;
  const P1 = { x0: 8,   y0: 40, w: 232, h: 328 };  // graph
  const P2 = { x0: 248, y0: 40, w: 150, h: 328 };  // Perron bars
  const P3 = { x0: 406, y0: 40, w: 146, h: 328 };  // log Tr(Aⁿ)

  // ── Linear algebra helpers ────────────────────────────────────────────
  function identity(n) {
    return Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  }
  function matMul(X, Y) {
    const n = X.length;
    const R = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++)
      for (let k = 0; k < n; k++) {
        if (!X[i][k]) continue;
        for (let j = 0; j < n; j++) R[i][j] += X[i][k] * Y[k][j];
      }
    return R;
  }
  function matPow(X, p) {
    let R = identity(X.length), B = X;
    while (p > 0) { if (p & 1) R = matMul(R, B); B = matMul(B, B); p >>= 1; }
    return R;
  }
  function trace(X) { let s = 0; for (let i = 0; i < X.length; i++) s += X[i][i]; return s; }

  /** Perron right eigenvector by power iteration, normalised to sum 1. */
  function perronVector(A) {
    const n = A.length;
    let v = new Array(n).fill(1 / Math.sqrt(n));
    for (let it = 0; it < 600; it++) {
      const Av = A.map(row => row.reduce((s, a, j) => s + a * v[j], 0));
      const norm = Math.hypot(...Av);
      if (norm === 0) break;
      v = Av.map(x => x / norm);
    }
    v = v.map(Math.abs);
    const s = v.reduce((a, b) => a + b, 0) || 1;
    return v.map(x => x / s);
  }

  /** Parse a whitespace/comma matrix (must be square, non-negative). */
  function parseMatrix(text) {
    const rows = text.trim().split(/\n+/).map(line =>
      line.trim().split(/[\s,]+/).filter(s => s.length).map(Number));
    if (!rows.length) return null;
    const n = rows.length;
    for (const r of rows) {
      if (r.length !== n) return null;
      for (const x of r) if (!Number.isFinite(x) || x < 0) return null;
    }
    return rows;
  }

  function matToText(A) { return A.map(r => r.join(' ')).join('\n'); }

  function init(container, params = {}) {
    const { canvas, ctx, width, height } = Canvas.create(container, W, H);

    let A = params.matrix || PRESETS.fibonacci.map(r => r.slice());
    let rho, hEntropy, perron, positions, traces;

    // Random-walk animation state.
    const walk = { cur: 0, nxt: 0, t: 0, blink: 0 };
    let anim = null, animating = false;

    // ── Derived quantities ──────────────────────────────────────────────
    function recompute() {
      const n = A.length;
      rho = Math2D.spectralRadius(A);
      hEntropy = rho > 0 ? Math.log(rho) : 0;
      perron = perronVector(A);
      positions = layout(n);
      traces = [];
      let P = identity(n);
      for (let k = 1; k <= 15; k++) { P = matMul(P, A); traces.push({ n: k, tr: trace(P) }); }
      walk.cur = 0; walk.nxt = chooseNext(0); walk.t = 0;
    }

    // Node positions: circle for n ≤ 4, else a small spring layout.
    function layout(n) {
      const cx = P1.x0 + P1.w / 2, cy = P1.y0 + P1.h / 2;
      const R = Math.min(P1.w, P1.h) / 2 - 40;
      if (n <= 4) {
        return Array.from({ length: n }, (_, i) => {
          const a = -Math.PI / 2 + (2 * Math.PI * i) / n;
          return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
        });
      }
      // Spring layout (Fruchterman–Reingold-ish), 10 iterations.
      let pos = Array.from({ length: n }, (_, i) => {
        const a = (2 * Math.PI * i) / n;
        return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
      });
      const k = R / Math.sqrt(n);
      for (let it = 0; it < 10; it++) {
        const disp = pos.map(() => [0, 0]);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          if (i === j) continue;
          let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
          let d = Math.hypot(dx, dy) || 0.01;
          const rep = (k * k) / d;                       // repulsion
          disp[i][0] += (dx / d) * rep; disp[i][1] += (dy / d) * rep;
          if (A[i][j] || A[j][i]) {                      // attraction on edges
            const att = (d * d) / k;
            disp[i][0] -= (dx / d) * att * 0.5; disp[i][1] -= (dy / d) * att * 0.5;
          }
        }
        for (let i = 0; i < n; i++) {
          const dl = Math.hypot(...disp[i]) || 0.01;
          pos[i][0] += (disp[i][0] / dl) * Math.min(dl, 8);
          pos[i][1] += (disp[i][1] / dl) * Math.min(dl, 8);
          pos[i][0] = Math.max(P1.x0 + 24, Math.min(P1.x0 + P1.w - 24, pos[i][0]));
          pos[i][1] = Math.max(P1.y0 + 24, Math.min(P1.y0 + P1.h - 24, pos[i][1]));
        }
      }
      return pos;
    }

    function chooseNext(i) {
      const outs = [];
      for (let j = 0; j < A.length; j++) if (A[i][j] > 0) outs.push(j);
      if (!outs.length) return i;                        // dead state: stay
      return outs[(Math.random() * outs.length) | 0];
    }

    // Node radius from Perron weight (area ∝ weight).
    function nodeRadius(i) {
      return 9 + 26 * Math.sqrt(perron[i]);
    }

    // ── Drawing ─────────────────────────────────────────────────────────
    function drawAll() {
      Canvas.clear(ctx, width, height);

      // Title + entropy readout
      ctx.textAlign = 'left';
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = Palette.text;
      ctx.fillText('Subshift of Finite Type — Perron–Frobenius', 8, 16);
      ctx.font = '11px monospace';
      ctx.fillStyle = Palette.gold;
      ctx.fillText(`ρ(A) = ${rho.toFixed(4)}    h = log ρ = ${hEntropy.toFixed(4)}`, 8, 32);

      drawGraph();
      drawBars();
      drawTraces();
    }

    function drawGraph() {
      ctx.save();
      ctx.beginPath();
      ctx.rect(P1.x0, P1.y0, P1.w, P1.h);
      ctx.clip();
      // panel background wipe (for animation frames)
      ctx.fillStyle = Palette.bg;
      ctx.fillRect(P1.x0, P1.y0, P1.w, P1.h);

      const n = A.length;
      // Edges first (behind nodes).
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (!A[i][j]) continue;
        if (i === j) drawSelfLoop(positions[i], nodeRadius(i));
        else drawEdge(positions[i], positions[j], nodeRadius(i), nodeRadius(j), A[j][i] > 0);
      }

      // Random-walk dot: interpolate along the straight segment cur→nxt.
      const [ax, ay] = positions[walk.cur], [bx, by] = positions[walk.nxt];
      const dx = ax + (bx - ax) * walk.t, dy = ay + (by - ay) * walk.t;
      const alpha = 0.55 + 0.45 * Math.sin(walk.blink * 6);
      ctx.globalAlpha = alpha;
      Canvas.dot(ctx, dx, dy, 6, Palette.accent);
      ctx.globalAlpha = 1;

      // Nodes on top.
      ctx.font = '11px monospace';
      for (let i = 0; i < n; i++) {
        const r = nodeRadius(i);
        Canvas.dot(ctx, positions[i][0], positions[i][1], r, Palette.card);
        ctx.strokeStyle = Palette.blue; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(positions[i][0], positions[i][1], r, 0, 2 * Math.PI); ctx.stroke();
        ctx.fillStyle = Palette.text; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), positions[i][0], positions[i][1]);
      }
      ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
      ctx.restore();

      // Panel caption
      Canvas.label(ctx, 'Transition graph', P1.x0 + 4, P1.y0 + P1.h - 6, { color: Palette.muted });
    }

    // Directed edge with arrowhead; curved slightly if reciprocal.
    function drawEdge([ax, ay], [bx, by], ra, rb, reciprocal) {
      let dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      // trim to node borders
      const sx = ax + ux * ra, sy = ay + uy * ra;
      let ex = bx - ux * rb, ey = by - uy * rb;
      // perpendicular bow for reciprocal pairs
      const bow = reciprocal ? 14 : 0;
      const nx = -uy * bow, ny = ux * bow;
      const mx = (sx + ex) / 2 + nx, my = (sy + ey) / 2 + ny;

      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo(mx, my, ex, ey); ctx.stroke();

      // arrowhead pointing along the incoming tangent (from control point)
      const tx = ex - mx, ty = ey - my;
      const ta = Math.atan2(ty, tx);
      arrowHead(ex, ey, ta);

      // edge label "1"
      Canvas.label(ctx, '1', mx, my, { color: Palette.muted, font: '9px monospace' });
    }

    function drawSelfLoop([x, y], r) {
      const cx = x, cy = y - r - 12;
      ctx.strokeStyle = Palette.muted; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(cx, cy, 10, 0.2 * Math.PI, 2.6 * Math.PI); ctx.stroke();
      arrowHead(cx + 10 * Math.cos(0.2 * Math.PI), cy + 10 * Math.sin(0.2 * Math.PI), Math.PI * 0.9);
      Canvas.label(ctx, '1', cx, cy - 12, { color: Palette.muted, font: '9px monospace' });
    }

    function arrowHead(x, y, ang) {
      const s = 6;
      ctx.fillStyle = Palette.muted;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - s * Math.cos(ang - 0.4), y - s * Math.sin(ang - 0.4));
      ctx.lineTo(x - s * Math.cos(ang + 0.4), y - s * Math.sin(ang + 0.4));
      ctx.closePath(); ctx.fill();
    }

    function drawBars() {
      const n = perron.length;
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1;
      ctx.strokeRect(P2.x0 + 0.5, P2.y0 + 0.5, P2.w, P2.h);
      Canvas.label(ctx, 'Perron eigenvector', P2.x0 + 4, P2.y0 + 14, { color: Palette.muted });

      const base = P2.y0 + P2.h - 24;
      const top = P2.y0 + 28;
      const maxV = Math.max(...perron, 1e-9);
      const bw = (P2.w - 20) / n;
      for (let i = 0; i < n; i++) {
        const bh = ((perron[i] / maxV) * (base - top));
        const x = P2.x0 + 10 + i * bw;
        ctx.fillStyle = Palette.blue;
        ctx.fillRect(x + bw * 0.15, base - bh, bw * 0.7, bh);
        ctx.fillStyle = Palette.text; ctx.font = '9px monospace'; ctx.textAlign = 'center';
        ctx.fillText(perron[i].toFixed(3), x + bw / 2, base - bh - 4);
        ctx.fillStyle = Palette.muted;
        ctx.fillText(String(i + 1), x + bw / 2, base + 12);
      }
      ctx.textAlign = 'left';
    }

    function drawTraces() {
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1;
      ctx.strokeRect(P3.x0 + 0.5, P3.y0 + 0.5, P3.w, P3.h);
      Canvas.label(ctx, 'log Tr(Aⁿ)', P3.x0 + 4, P3.y0 + 14, { color: Palette.muted });

      const pts = traces.filter(d => d.tr > 0).map(d => ({ n: d.n, y: Math.log(d.tr) }));
      if (!pts.length) return;
      const nMax = 15;
      const yMin = Math.min(...pts.map(p => p.y), 0);
      const yMax = Math.max(...pts.map(p => p.y), Math.log(2));
      const pad = (yMax - yMin) * 0.1 + 1e-6;
      const lo = yMin - pad, hi = yMax + pad;

      const gx0 = P3.x0 + 26, gx1 = P3.x0 + P3.w - 8;
      const gy0 = P3.y0 + P3.h - 24, gy1 = P3.y0 + 26;
      const X = n => gx0 + ((n - 1) / (nMax - 1)) * (gx1 - gx0);
      const Y = y => gy0 + (1 - (y - lo) / (hi - lo)) * (gy1 - gy0);

      // Fitted line  y = n·log ρ + c   (c = mean residual).
      const logRho = Math.log(Math.max(rho, 1e-9));
      const c = pts.reduce((s, p) => s + (p.y - p.n * logRho), 0) / pts.length;
      Canvas.polyline(ctx, [[X(1), Y(1 * logRho + c)], [X(nMax), Y(nMax * logRho + c)]], Palette.gold, 2);

      // Data points
      for (const p of pts) Canvas.dot(ctx, X(p.n), Y(p.y), 2.5, Palette.blue);

      // Axis ticks
      ctx.font = '9px monospace'; ctx.fillStyle = Palette.muted; ctx.textAlign = 'center';
      ctx.fillText('n=1', X(1), gy0 + 12);
      ctx.fillText('15', X(15), gy0 + 12);
      ctx.textAlign = 'left';
      Canvas.label(ctx, 'slope=log ρ', gx0 + 2, gy1 + 2, { color: Palette.gold, font: '9px monospace' });
    }

    // ── Animation ───────────────────────────────────────────────────────
    function tick(dt) {
      walk.blink += dt;
      walk.t += dt / 0.6;                 // one edge per 0.6 s
      if (walk.t >= 1) { walk.t = 0; walk.cur = walk.nxt; walk.nxt = chooseNext(walk.cur); }
      drawGraph();                        // only the graph panel repaints
    }

    function setAnimating(on) {
      animating = on;
      if (on) { if (!anim) anim = Anim.loop(tick); anim.start(); }
      else if (anim) anim.stop();
    }

    // ── Controls ────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'select', key: 'preset', label: 'Preset', value: 'fibonacci', options: [
        { value: 'fibonacci',   label: 'Fibonacci (2×2)' },
        { value: 'golden_mean', label: 'Golden mean (3×3)' },
        { value: 'full3',       label: 'Full shift 3 (3×3)' },
        { value: 'user',        label: 'User matrix' },
      ] },
      { type: 'checkbox', key: 'animate', label: 'Animate random walk', value: false },
      { type: 'button', label: 'Step', action: () => {
        walk.cur = walk.nxt; walk.nxt = chooseNext(walk.cur); walk.t = 0; drawGraph();
      } },
    ]);

    // Editable matrix textarea + apply button.
    const editWrap = document.createElement('div');
    editWrap.style.cssText = 'display:flex;flex-direction:column;gap:4px;margin-top:6px;font-family:monospace;font-size:12px;color:' + Palette.text;
    const taLabel = document.createElement('span');
    taLabel.textContent = 'Transition matrix (rows of 0/1):';
    taLabel.style.color = Palette.muted;
    const ta = document.createElement('textarea');
    ta.value = matToText(A);
    ta.rows = 4;
    ta.style.cssText = `width:220px;background:${Palette.card};color:${Palette.text};border:1px solid ${Palette.muted};border-radius:3px;font-family:monospace;font-size:12px;padding:4px`;
    const applyRow = document.createElement('div');
    applyRow.style.cssText = 'display:flex;gap:8px;align-items:center';
    const applyBtn = document.createElement('button');
    applyBtn.textContent = 'Apply matrix';
    applyBtn.style.cssText = `padding:5px 14px;background:${Palette.card};color:${Palette.blue};border:1px solid ${Palette.blue};border-radius:3px;cursor:pointer;font-family:monospace;font-size:12px`;
    const status = document.createElement('span');
    status.style.color = Palette.muted;
    applyRow.append(applyBtn, status);
    editWrap.append(taLabel, ta, applyRow);
    container.appendChild(editWrap);

    function loadMatrix(M) {
      A = M.map(r => r.slice());
      ta.value = matToText(A);
      recompute();
      drawAll();
    }

    applyBtn.addEventListener('click', () => {
      const M = parseMatrix(ta.value);
      if (!M) { status.textContent = '✗ not a square non-negative matrix'; status.style.color = Palette.accent; return; }
      status.textContent = '✓ applied'; status.style.color = Palette.green;
      loadMatrix(M);
    });

    ctrl.onChange((key, val) => {
      if (key === 'preset') {
        if (val === 'user') { ta.focus(); return; }
        loadMatrix(PRESETS[val]);
      } else if (key === 'animate') {
        setAnimating(val);
      }
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
        loadMatrix(PRESETS.fibonacci);
      },
      setParam(key, val) {
        if (key === 'matrix' && Array.isArray(val)) loadMatrix(val);
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = SubshiftFiniteType;
