/**
 * @file 04_12_02_pagerank.js
 * @chapter 4 — Ergodic Theory
 * @sections §4.12
 * @concept PageRank as the invariant measure of a random walk on a graph
 *
 * PAGERANK AS AN INVARIANT MEASURE
 *
 * PageRank models a "random surfer" performing a random walk on a directed graph
 * of web pages. With probability d the surfer follows a uniformly random OUTLINK
 * of the current page; with probability 1−d it TELEPORTS to a uniformly random
 * page. The transition operator is the column-stochastic GOOGLE MATRIX
 *
 *     G = d · A  +  (1−d)/n · 𝟙𝟙ᵀ,
 *
 * where A is the column-stochastic link matrix:
 *
 *     A[i][j] = 1/outdeg(j)   if j → i is an edge,
 *             = 1/n           if j is DANGLING (no outlinks),   ← redistribute
 *             = 0             otherwise.
 *
 * PageRank is the stationary distribution v = G v, i.e. the unique INVARIANT
 * MEASURE of this Markov chain. Because G is column-stochastic, positive and
 * primitive (teleportation makes the chain irreducible + aperiodic), the
 * Perron–Frobenius theorem guarantees a unique positive eigenvector of
 * eigenvalue 1, and POWER ITERATION vₖ₊₁ = G vₖ converges to it geometrically —
 * the rate set by the second eigenvalue |λ₂| ≤ d. This is exactly the ergodic
 * "time average = space average" principle: the fraction of time the surfer
 * spends on a page converges to that page's stationary probability.
 *
 * Applying G to a distribution v (Σvⱼ = 1) simplifies to
 *
 *     v'[i] = d · Σ_{j→i} v[j]/outdeg(j)  +  (1−d)/n  +  d/n · Σ_{dangling j} v[j].
 *
 * The graph is drawn with a Fruchterman–Reingold spring layout; each node's size
 * and brightness track its current rank, and an inset shows ‖Δv‖₁ per iteration
 * on a log axis so the geometric convergence is visible.
 */

var PageRank = (() => {
  const { Palette, Canvas, Controls, Anim } = DSUtils;

  const meta = {
    id: '04_12_02',
    title: 'PageRank as Invariant Measure of a Random Walk',
    chapter: 'Chapter 4 — Ergodic Theory',
    sections: ['4.12'],
    concept: 'The stationary distribution of the Google matrix via power iteration',
    description:
      'A random directed graph and live power iteration of the Google matrix ' +
      'G = d·A + (1−d)/n·𝟙𝟙ᵀ. Node size/brightness track PageRank; an inset plots ' +
      'the ‖Δv‖₁ convergence on a log axis.',
  };

  const TAU = 2 * Math.PI;
  const DEFAULTS = { n_pages: 14, edge_prob: 0.18, damping: 0.85, animate: true };

  function init(container, params = {}) {
    const state = Object.assign({}, DEFAULTS, params);

    const W = 560, H = 440;
    const { ctx, width, height } = Canvas.create(container, W, H);

    // Graph drawing region and convergence inset (bottom-right).
    const G = { x: 18, y: 40, w: 524, h: 382 };
    const INSET = { x: W - 18 - 130, y: H - 18 - 84, w: 130, h: 84 };

    // ── graph + iteration state ──────────────────────────────────────────
    let n, adjOut, outdeg, pos, v, iter, converged, iterating, deltaHist;

    function buildGraph() {
      n = state.n_pages | 0;
      adjOut = Array.from({ length: n }, () => []);
      outdeg = new Int32Array(n);
      // Erdős–Rényi directed graph: independent edge i→j with prob edge_prob.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j && Math.random() < state.edge_prob) adjOut[i].push(j);
        }
        outdeg[i] = adjOut[i].length;   // dangling nodes keep outdeg 0
      }
      layout();
      resetRanks();
    }

    // Fruchterman–Reingold spring layout (undirected adjacency for placement).
    function layout() {
      const area = G.w * G.h;
      const k = Math.sqrt(area / Math.max(1, n));   // ideal edge length
      const cx = G.x + G.w / 2, cy = G.y + G.h / 2;

      // undirected neighbour flags for attraction
      const nbr = Array.from({ length: n }, () => new Set());
      for (let i = 0; i < n; i++) for (const j of adjOut[i]) { nbr[i].add(j); nbr[j].add(i); }

      pos = Array.from({ length: n }, () => [
        cx + (Math.random() - 0.5) * G.w * 0.6,
        cy + (Math.random() - 0.5) * G.h * 0.6,
      ]);

      let temp = G.w / 8;
      for (let step = 0; step < 60; step++) {
        const disp = Array.from({ length: n }, () => [0, 0]);
        // repulsion between all pairs: F_rep = k²/dist
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
            let dist = Math.hypot(dx, dy) || 0.01;
            const f = (k * k) / dist;
            const ux = dx / dist, uy = dy / dist;
            disp[i][0] += ux * f; disp[i][1] += uy * f;
            disp[j][0] -= ux * f; disp[j][1] -= uy * f;
          }
        }
        // attraction along edges: F_att = dist²/k
        for (let i = 0; i < n; i++) {
          for (const j of nbr[i]) {
            if (j <= i) continue;
            let dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
            let dist = Math.hypot(dx, dy) || 0.01;
            const f = (dist * dist) / k;
            const ux = dx / dist, uy = dy / dist;
            disp[i][0] -= ux * f; disp[i][1] -= uy * f;
            disp[j][0] += ux * f; disp[j][1] += uy * f;
          }
        }
        // mild gravity toward centre keeps the whole graph on-canvas
        for (let i = 0; i < n; i++) {
          disp[i][0] += (cx - pos[i][0]) * 0.03;
          disp[i][1] += (cy - pos[i][1]) * 0.03;
        }
        // apply, capped by temperature, then cool
        for (let i = 0; i < n; i++) {
          const d = Math.hypot(disp[i][0], disp[i][1]) || 0.01;
          const lim = Math.min(d, temp);
          pos[i][0] += (disp[i][0] / d) * lim;
          pos[i][1] += (disp[i][1] / d) * lim;
          // clamp inside region (with node-radius margin)
          pos[i][0] = Math.max(G.x + 24, Math.min(G.x + G.w - 24, pos[i][0]));
          pos[i][1] = Math.max(G.y + 24, Math.min(G.y + G.h - 24, pos[i][1]));
        }
        temp *= 0.95;
      }
    }

    function resetRanks() {
      v = new Float64Array(n).fill(1 / n);   // uniform prior
      iter = 0; converged = false; iterating = true;
      deltaHist = [];
    }

    // One power-iteration step v ← G v. Returns ‖Δv‖₁ and ‖Δv‖∞.
    function powerStep() {
      const d = state.damping;
      const nv = new Float64Array(n);

      // total rank sitting on dangling nodes → redistributed uniformly
      let dangling = 0;
      for (let j = 0; j < n; j++) if (outdeg[j] === 0) dangling += v[j];

      // teleport + dangling mass, uniform over all nodes
      const base = (1 - d) / n + (d * dangling) / n;
      for (let i = 0; i < n; i++) nv[i] = base;

      // link contributions: each page pushes d·v[j]/outdeg(j) to its targets
      for (let j = 0; j < n; j++) {
        if (outdeg[j] === 0) continue;
        const share = (d * v[j]) / outdeg[j];
        for (const i of adjOut[j]) nv[i] += share;
      }

      let l1 = 0, linf = 0;
      for (let i = 0; i < n; i++) {
        const dd = Math.abs(nv[i] - v[i]);
        l1 += dd; if (dd > linf) linf = dd;
      }
      v = nv;
      return { l1, linf };
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

    function arrow(x0, y0, x1, y1, rTarget, color) {
      // shorten the segment so the head sits on the target node's rim
      const dx = x1 - x0, dy = y1 - y0;
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const ex = x1 - ux * rTarget, ey = y1 - uy * rTarget;
      ctx.strokeStyle = color; ctx.lineWidth = 1;
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(x0 + ux * 6, y0 + uy * 6); ctx.lineTo(ex, ey); ctx.stroke();
      // arrowhead
      const ah = 6, aw = 3.2;
      const bx = ex - ux * ah, by = ey - uy * ah;
      const px = -uy, py = ux;   // perpendicular
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(bx + px * aw, by + py * aw);
      ctx.lineTo(bx - px * aw, by - py * aw);
      ctx.closePath(); ctx.fill();
      ctx.globalAlpha = 1;
    }

    function radiusOf(rank, maxRank) {
      // area ∝ rank feels natural → radius ∝ sqrt(rank)
      const t = maxRank > 0 ? rank / maxRank : 0;
      return 5 + 18 * Math.sqrt(t);
    }

    function drawInset() {
      roundRect(INSET.x, INSET.y, INSET.w, INSET.h, 5);
      ctx.fillStyle = 'rgba(15,52,96,0.85)'; ctx.fill();
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1; ctx.stroke();

      Canvas.label(ctx, '‖Δv‖₁  (log)', INSET.x + 6, INSET.y + 13,
        { color: Palette.blue, font: '9px monospace' });

      // log-y from 1e0 (top) to 1e-6 (bottom)
      const pad = 6, top = INSET.y + 18, bot = INSET.y + INSET.h - 8;
      const plotH = bot - top, plotW = INSET.w - 2 * pad;
      const logMin = -6, logMax = 0;
      const yOf = val => {
        const lg = Math.max(logMin, Math.min(logMax, Math.log10(Math.max(val, 1e-12))));
        return bot - ((lg - logMin) / (logMax - logMin)) * plotH;
      };
      // convergence threshold guide at 1e-6
      ctx.strokeStyle = Palette.accent; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(INSET.x + pad, yOf(1e-6)); ctx.lineTo(INSET.x + pad + plotW, yOf(1e-6)); ctx.stroke();
      ctx.setLineDash([]);

      if (deltaHist.length >= 2) {
        const maxK = Math.max(100, deltaHist.length);
        const pts = deltaHist.map((val, i) => [
          INSET.x + pad + (i / (maxK - 1)) * plotW,
          yOf(val),
        ]);
        Canvas.polyline(ctx, pts, Palette.gold, 1.5);
      }
    }

    function draw() {
      Canvas.clear(ctx, width, height);

      // title
      Canvas.label(ctx, meta.title, 16, 22, { color: Palette.blue, font: '13px monospace' });

      // graph card
      roundRect(G.x, G.y, G.w, G.h, 8);
      ctx.fillStyle = Palette.card; ctx.globalAlpha = 0.35; ctx.fill(); ctx.globalAlpha = 1;
      ctx.strokeStyle = '#2a3a5c'; ctx.lineWidth = 1; ctx.stroke();

      // rank extremes for scaling
      let maxRank = 0, argMax = 0;
      for (let i = 0; i < n; i++) if (v[i] > maxRank) { maxRank = v[i]; argMax = i; }

      // arrows first (under nodes)
      for (let j = 0; j < n; j++) {
        for (const i of adjOut[j]) {
          const rT = radiusOf(v[i], maxRank);
          arrow(pos[j][0], pos[j][1], pos[i][0], pos[i][1], rT, Palette.muted);
        }
      }

      // nodes
      for (let i = 0; i < n; i++) {
        const r = radiusOf(v[i], maxRank);
        const t = maxRank > 0 ? v[i] / maxRank : 0;
        const isTop = i === argMax;
        // fill: blue with alpha ∝ rank, largest node gold
        ctx.beginPath(); ctx.arc(pos[i][0], pos[i][1], r, 0, TAU);
        if (isTop) {
          ctx.fillStyle = Palette.gold;
        } else {
          ctx.globalAlpha = 0.3 + 0.7 * t;
          ctx.fillStyle = Palette.blue;
        }
        ctx.fill(); ctx.globalAlpha = 1;
        ctx.lineWidth = 1.2; ctx.strokeStyle = isTop ? '#fff' : Palette.blue; ctx.stroke();
        // label
        ctx.fillStyle = isTop ? '#1a1a2e' : Palette.text;
        ctx.font = '10px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(String(i), pos[i][0], pos[i][1]);
        ctx.textBaseline = 'alphabetic';
      }

      // status line
      const status = converged
        ? `converged in ${iter} steps  (‖Δv‖∞ < 1e-6)`
        : (iterating ? `iterating…  step ${iter}` : `paused  step ${iter}`);
      Canvas.label(ctx, status, 16, 38, { color: Palette.muted, font: '11px monospace' });
      Canvas.label(ctx, `d = ${state.damping.toFixed(2)}   n = ${n}   top = page ${argMax} (${(maxRank * 100).toFixed(1)}%)`,
        width - 16, 22, { color: Palette.gold, font: '11px monospace', align: 'right' });

      drawInset();
    }

    // ── animation: one power iteration per frame ─────────────────────────
    const anim = Anim.loop(() => {
      if (state.animate && iterating && !converged) {
        const { l1, linf } = powerStep();
        deltaHist.push(l1);
        iter++;
        if (linf < 1e-6) { converged = true; iterating = false; }
        if (iter >= 100) iterating = false;   // hard cap
      }
      draw();
    });

    // ── controls ─────────────────────────────────────────────────────────
    const ctrl = Controls.build(container, [
      { type: 'slider', key: 'n_pages', label: 'pages n', min: 8, max: 30, step: 1,
        value: state.n_pages, format: v => v | 0 },
      { type: 'slider', key: 'edge_prob', label: 'edge prob', min: 0.1, max: 0.5, step: 0.01,
        value: state.edge_prob, format: v => v.toFixed(2) },
      { type: 'slider', key: 'damping', label: 'damping d', min: 0.5, max: 0.99, step: 0.01,
        value: state.damping, format: v => v.toFixed(2) },
      { type: 'button', label: 'Regenerate', action: () => { buildGraph(); } },
      { type: 'button', label: 'Reset ranks', action: () => { resetRanks(); } },
      { type: 'checkbox', key: 'animate', label: 'animate', value: state.animate },
    ]);

    ctrl.onChange((key, val) => {
      state[key] = val;
      if (key === 'n_pages' || key === 'edge_prob') buildGraph();
      else if (key === 'damping') resetRanks();   // new operator → restart iteration
      // 'animate' just gates the loop; nothing else to do
    });

    buildGraph();
    anim.start();

    return {
      destroy() { anim.stop(); container.innerHTML = ''; },
      reset() { resetRanks(); },
      setParam(k, v2) {
        state[k] = v2;
        if (k === 'n_pages' || k === 'edge_prob') buildGraph();
        else if (k === 'damping') resetRanks();
      },
    };
  }

  return { meta, init };
})();

if (typeof module !== 'undefined') module.exports = PageRank;
