// MacroCut stabilisation core — pure maths, no DOM. Runs in the browser and in Node (tests).
//
// Coordinate conventions
//  - Analysis frames are grayscale Float32 (0..255), display-oriented, w×h pixels, row 0 = top.
//  - "Normalised" coords: origin at frame centre, units of frame HEIGHT, y down.
//      x_n = (x - w/2) / h,  y_n = (y - h/2) / h   → x ∈ [-A/2, A/2], y ∈ [-1/2, 1/2], A = w/h
//  - A similarity transform is [a, b, tx, ty]:  p' = [[a,-b],[b,a]]·p + t
//  - C_k ("camera") maps frame-k normalised coords → world (frame 0) normalised coords.

// ───────────────────────── similarity algebra ─────────────────────────
export const IDENT = Object.freeze([1, 0, 0, 0]);

export function compose(m1, m2) { // m1 ∘ m2
  const [a1, b1, x1, y1] = m1, [a2, b2, x2, y2] = m2;
  return [a1 * a2 - b1 * b2, a1 * b2 + b1 * a2, a1 * x2 - b1 * y2 + x1, b1 * x2 + a1 * y2 + y1];
}
export function invert(m) {
  const [a, b, x, y] = m, d = a * a + b * b;
  const ia = a / d, ib = -b / d;
  return [ia, ib, -(ia * x - ib * y), -(ib * x + ia * y)];
}
export function apply(m, x, y) { return [m[0] * x - m[1] * y + m[2], m[1] * x + m[0] * y + m[3]]; }
export function toParams(m) { return { th: Math.atan2(m[1], m[0]), ls: Math.log(Math.hypot(m[0], m[1])), x: m[2], y: m[3] }; }
export function fromParams(th, ls, x, y) { const s = Math.exp(ls); return [s * Math.cos(th), s * Math.sin(th), x, y]; }

// pixel-space similarity → normalised-space similarity (same frame size w×h on both sides)
export function pixToNorm(m, w, h) {
  const [a, b, tx, ty] = m, cx = w / 2, cy = h / 2;
  return [a, b, (a * cx - b * cy + tx - cx) / h, (b * cx + a * cy + ty - cy) / h];
}

// ───────────────────────── image pyramid ─────────────────────────
export function buildPyramid(gray, w, h, levels) {
  const pyr = [{ d: gray, w, h }];
  for (let l = 1; l < levels; l++) {
    const p = pyr[l - 1], nw = p.w >> 1, nh = p.h >> 1;
    if (nw < 16 || nh < 16) break;
    const tmp = new Float32Array(nw * p.h), out = new Float32Array(nw * nh), s = p.d, W = p.w, H = p.h;
    // separable [1 3 3 1]/8 with 2× decimation
    for (let y = 0; y < H; y++) {
      const r = y * W;
      for (let x = 0; x < nw; x++) {
        const sx = 2 * x, x0 = sx > 0 ? sx - 1 : 0, x3 = sx + 2 < W ? sx + 2 : W - 1, x2 = sx + 1 < W ? sx + 1 : W - 1;
        tmp[y * nw + x] = (s[r + x0] + 3 * s[r + sx] + 3 * s[r + x2] + s[r + x3]) * 0.125;
      }
    }
    for (let y = 0; y < nh; y++) {
      const sy = 2 * y, y0 = sy > 0 ? sy - 1 : 0, y2 = sy + 1 < H ? sy + 1 : H - 1, y3 = sy + 2 < H ? sy + 2 : H - 1;
      for (let x = 0; x < nw; x++) {
        out[y * nw + x] = (tmp[y0 * nw + x] + 3 * tmp[sy * nw + x] + 3 * tmp[y2 * nw + x] + tmp[y3 * nw + x]) * 0.125;
      }
    }
    pyr.push({ d: out, w: nw, h: nh });
  }
  return pyr;
}

function bilin(d, w, h, x, y) {
  if (x < 0) x = 0; else if (x > w - 1.001) x = w - 1.001;
  if (y < 0) y = 0; else if (y > h - 1.001) y = h - 1.001;
  const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = y0 * w + x0;
  const a = d[i], b = d[i + 1], c = d[i + w], e = d[i + w + 1];
  return a + fx * (b - a) + fy * (c - a) + fx * fy * (a - b - c + e);
}

// ───────────────────────── features (Shi–Tomasi, grid-spread) ─────────────────────────
// Expected min-eigenvalue score of pure sensor noise (5×5 window), from a robust noise estimate.
function noiseFloor(img, w, h) {
  // median |Laplacian| in quarter-units for precision, via histogram
  const med = histPercentile(add => {
    for (let y = 2; y < h - 2; y += 3) for (let x = 2; x < w - 2; x += 3) {
      const i = y * w + x;
      add(4 * Math.abs(4 * img[i] - img[i - 1] - img[i + 1] - img[i - w] - img[i + w]));
    }
  }, 0.5) / 4;
  const sigma = 1.4826 * med / Math.sqrt(20);                    // Laplacian of white noise has σ·√20
  return Math.max(12, 25 * 0.5 * sigma * sigma * 3);               // ~3× the noise-only score
}
// mask(x, y) → boolean (optional). Returns Float32Array [x0,y0,x1,y1,…] in level-0 pixels.
export function detectFeatures(img, w, h, maxN = 160, mask = null, margin = 12, rect = null) {
  const gx = new Float32Array(w * h), gy = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    gx[i] = (img[i + 1] - img[i - 1]) * 0.5;
    gy[i] = (img[i + w] - img[i - w]) * 0.5;
  }
  // 5×5 box sums of structure tensor (separable)
  const R = 2, xx = new Float32Array(w * h), yy = new Float32Array(w * h), xy = new Float32Array(w * h);
  const txx = new Float32Array(w * h), tyy = new Float32Array(w * h), txy = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let sxx = 0, syy = 0, sxy = 0; const r = y * w;
    for (let x = 0; x < w; x++) {
      const i = r + x; sxx += gx[i] * gx[i]; syy += gy[i] * gy[i]; sxy += gx[i] * gy[i];
      if (x > 2 * R) { const j = i - 2 * R - 1; sxx -= gx[j] * gx[j]; syy -= gy[j] * gy[j]; sxy -= gx[j] * gy[j]; }
      if (x >= R) { const o = i - R; txx[o] = sxx; tyy[o] = syy; txy[o] = sxy; }
    }
  }
  for (let x = 0; x < w; x++) {
    let sxx = 0, syy = 0, sxy = 0;
    for (let y = 0; y < h; y++) {
      const i = y * w + x; sxx += txx[i]; syy += tyy[i]; sxy += txy[i];
      if (y > 2 * R) { const j = i - (2 * R + 1) * w; sxx -= txx[j]; syy -= tyy[j]; sxy -= txy[j]; }
      if (y >= R) { const o = i - R * w; xx[o] = sxx; yy[o] = syy; xy[o] = sxy; }
    }
  }
  const R0 = rect || { x0: 0, y0: 0, x1: w, y1: h }, RW = R0.x1 - R0.x0, RH = R0.y1 - R0.y0;
  const cellsX = rect ? 12 : 16, cellsY = rect ? Math.max(6, Math.round(12 * RH / RW)) : Math.max(6, Math.round(16 * h / w));
  const best = [];
  let maxScore = 0;
  for (let cy = 0; cy < cellsY; cy++) for (let cx = 0; cx < cellsX; cx++) {
    const x0 = Math.max(margin, Math.floor(R0.x0 + cx * RW / cellsX)), x1 = Math.min(w - margin, Math.floor(R0.x0 + (cx + 1) * RW / cellsX));
    const y0 = Math.max(margin, Math.floor(R0.y0 + cy * RH / cellsY)), y1 = Math.min(h - margin, Math.floor(R0.y0 + (cy + 1) * RH / cellsY));
    // two best (spaced) per cell
    let s1 = 0, i1 = -1, s2 = 0, i2 = -1;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      if (mask && !mask(x, y)) continue;
      const i = y * w + x, a = xx[i], c = yy[i], b = xy[i];
      const hm = (a - c) * 0.5, s = (a + c) * 0.5 - Math.sqrt(hm * hm + b * b);
      if (s > s1) { s2 = s1; i2 = i1; s1 = s; i1 = i; }
      else if (s > s2) { s2 = s; i2 = i; }
    }
    if (i1 >= 0) { best.push([s1, i1]); if (s1 > maxScore) maxScore = s1; }
    if (i2 >= 0) {
      const dx = (i2 % w) - (i1 % w), dy = ((i2 / w) | 0) - ((i1 / w) | 0);
      if (dx * dx + dy * dy > (rect ? 9 : 36)) best.push([s2, i2]);
    }
  }
  // Robust threshold: relative to the typical cell, not the single strongest point
  // (a specular catch-light in a spider's eye would otherwise wipe out all substrate texture).
  const med = best.map(b => b[0]).sort((a, b) => a - b)[best.length >> 1] || 0;
  // Absolute floor tied to sensor noise, so low-contrast (HDR, smooth bokeh) footage still gets points.
  const floor = Math.max(noiseFloor(img, w, h), 0.1 * med, 0.002 * maxScore);
  const sel = best.filter(b => b[0] > floor).sort((a, b) => b[0] - a[0]).slice(0, maxN);
  const out = new Float32Array(sel.length * 2);
  sel.forEach(([, i], k) => { out[2 * k] = i % w; out[2 * k + 1] = (i / w) | 0; });
  return out;
}

// ───────────────────────── keyframe templates for inverse-compositional LK ─────────────────────────
export function makeTemplates(pyr, feats, half) {
  const n = feats.length / 2, L = pyr.length, ws = 2 * half + 1, np = ws * ws;
  const lv = [];
  for (let l = 0; l < L; l++) {
    const { d, w, h } = pyr[l], sc = 1 / (1 << l);
    const T = new Float32Array(n * np), GX = new Float32Array(n * np), GY = new Float32Array(n * np);
    const Hinv = new Float32Array(n * 3), mg = new Float32Array(n * 2), ok = new Uint8Array(n);
    for (let f = 0; f < n; f++) {
      const px = feats[2 * f] * sc, py = feats[2 * f + 1] * sc, base = f * np;
      let sgx = 0, sgy = 0, k = 0;
      for (let v = -half; v <= half; v++) for (let u = -half; u <= half; u++, k++) {
        const x = px + u, y = py + v;
        T[base + k] = bilin(d, w, h, x, y);
        const gxv = (bilin(d, w, h, x + 1, y) - bilin(d, w, h, x - 1, y)) * 0.5;
        const gyv = (bilin(d, w, h, x, y + 1) - bilin(d, w, h, x, y - 1)) * 0.5;
        GX[base + k] = gxv; GY[base + k] = gyv; sgx += gxv; sgy += gyv;
      }
      const mx = sgx / np, my = sgy / np;
      let a = 0, b = 0, c = 0;
      for (k = 0; k < np; k++) {
        const ax = GX[base + k] - mx, ay = GY[base + k] - my;
        a += ax * ax; b += ax * ay; c += ay * ay;
      }
      const det = a * c - b * b;
      if (det > 1e-3 * (a + c) * (a + c) && a + c > 1) {
        Hinv[3 * f] = c / det; Hinv[3 * f + 1] = -b / det; Hinv[3 * f + 2] = a / det; ok[f] = 1;
      }
      mg[2 * f] = mx; mg[2 * f + 1] = my;
    }
    lv.push({ T, GX, GY, Hinv, mg, ok });
  }
  return { lv, half, np, n };
}

// Track keyframe features into current pyramid. guess: Float32Array of predicted level-0 positions.
export function trackLK(tpl, feats, cur, guess, maxErr = 14) {
  const { lv, half, np, n } = tpl, L = Math.min(lv.length, cur.length);
  const out = new Float32Array(n * 2), status = new Uint8Array(n);
  for (let f = 0; f < n; f++) {
    let dx = (guess[2 * f] - feats[2 * f]) / (1 << (L - 1));
    let dy = (guess[2 * f + 1] - feats[2 * f + 1]) / (1 << (L - 1));
    let good = true;
    for (let l = L - 1; l >= 0; l--) {
      const T = lv[l];
      if (!T.ok[f]) { if (l === 0) good = false; if (l > 0) { dx *= 2; dy *= 2; } continue; }
      const { d, w, h } = cur[l], sc = 1 / (1 << l);
      const px = feats[2 * f] * sc, py = feats[2 * f + 1] * sc, base = f * np;
      const h0 = T.Hinv[3 * f], h1 = T.Hinv[3 * f + 1], h2 = T.Hinv[3 * f + 2], mx = T.mg[2 * f], my = T.mg[2 * f + 1];
      const iters = l === L - 1 ? 10 : 7;
      for (let it = 0; it < iters; it++) {
        let bx = 0, by = 0, k = 0;
        const ox = px + dx, oy = py + dy;
        for (let v = -half; v <= half; v++) for (let u = -half; u <= half; u++, k++) {
          const e = bilin(d, w, h, ox + u, oy + v) - T.T[base + k];
          bx += (T.GX[base + k] - mx) * e; by += (T.GY[base + k] - my) * e;
        }
        // brightness-offset invariant (zero-mean) update
        const ddx = h0 * bx + h1 * by, ddy = h1 * bx + h2 * by;
        dx -= ddx; dy -= ddy;
        if (ddx * ddx + ddy * ddy < 1e-4) break;
      }
      if (l > 0) { dx *= 2; dy *= 2; }
      else {
        // residual after convergence (offset-compensated)
        let s = 0, sa = 0, k = 0;
        const ox = px + dx, oy = py + dy, buf = [];
        for (let v = -half; v <= half; v++) for (let u = -half; u <= half; u++, k++) {
          const e = bilin(d, w, h, ox + u, oy + v) - T.T[base + k]; buf.push(e); s += e;
        }
        const m = s / np;
        for (const e of buf) sa += Math.abs(e - m);
        // contrast-relative: fine hair texture resamples with large absolute residuals but is still well tracked
        let tm = 0, tv = 0;
        for (k = 0; k < np; k++) tm += T.T[base + k];
        tm /= np;
        for (k = 0; k < np; k++) tv += Math.abs(T.T[base + k] - tm);
        if (sa / np > Math.max(maxErr, 0.5 * tv / np)) good = false;
        const nx = feats[2 * f] + dx, ny = feats[2 * f + 1] + dy;
        if (!(nx > 1 && ny > 1 && nx < w - 2 && ny < h - 2)) good = false;
      }
    }
    out[2 * f] = feats[2 * f] + dx; out[2 * f + 1] = feats[2 * f + 1] + dy;
    status[f] = good && Number.isFinite(dx) && Number.isFinite(dy) ? 1 : 0;
  }
  return { pts: out, status };
}

// ───────────────────────── robust similarity fit ─────────────────────────
function lsqSimilarity(S, D, idx) {
  let msx = 0, msy = 0, mdx = 0, mdy = 0; const n = idx.length;
  for (const i of idx) { msx += S[2 * i]; msy += S[2 * i + 1]; mdx += D[2 * i]; mdy += D[2 * i + 1]; }
  msx /= n; msy /= n; mdx /= n; mdy /= n;
  let num1 = 0, num2 = 0, den = 0;
  for (const i of idx) {
    const sx = S[2 * i] - msx, sy = S[2 * i + 1] - msy, dx = D[2 * i] - mdx, dy = D[2 * i + 1] - mdy;
    num1 += sx * dx + sy * dy; num2 += sx * dy - sy * dx; den += sx * sx + sy * sy;
  }
  if (den < 1e-9) return null;
  const a = num1 / den, b = num2 / den;
  return [a, b, mdx - (a * msx - b * msy), mdy - (b * msx + a * msy)];
}

function countInliers(M, S, D, idx, thr2, out) {
  let c = 0; out.length = 0;
  for (const i of idx) {
    const x = M[0] * S[2 * i] - M[1] * S[2 * i + 1] + M[2] - D[2 * i];
    const y = M[1] * S[2 * i] + M[0] * S[2 * i + 1] + M[3] - D[2 * i + 1];
    if (x * x + y * y < thr2) { c++; out.push(i); }
  }
  return c;
}

export function ransacSimilarity(S, D, idx, thr = 1.2, iters = 160, seed = 1, prior = null) {
  if (idx.length < 4) return null;
  let s = seed >>> 0 || 1;
  const rnd = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
  const thr2 = thr * thr, tmp = [];
  let best = null, bestC = 0, bestIn = [];
  for (let it = 0; it < iters; it++) {
    const i = idx[(rnd() * idx.length) | 0], j = idx[(rnd() * idx.length) | 0];
    if (i === j) continue;
    const ux = S[2 * j] - S[2 * i], uy = S[2 * j + 1] - S[2 * i + 1];
    const vx = D[2 * j] - D[2 * i], vy = D[2 * j + 1] - D[2 * i + 1];
    const den = ux * ux + uy * uy; if (den < 25) continue;
    const a = (vx * ux + vy * uy) / den, b = (vy * ux - vx * uy) / den;
    const sc = Math.hypot(a, b); if (sc < 0.7 || sc > 1.4) continue;
    const M = [a, b, D[2 * i] - (a * S[2 * i] - b * S[2 * i + 1]), D[2 * i + 1] - (b * S[2 * i] + a * S[2 * i + 1])];
    const c = countInliers(M, S, D, idx, thr2, tmp);
    if (c > bestC) { bestC = c; best = M; bestIn = tmp.slice(); }
  }
  // Temporal stickiness: when two motion layers compete (spider vs leaf), keep following the one the
  // prediction points at unless the other clearly dominates. Flipping layers frame-to-frame is jitter.
  if (prior) {
    const pin = []; countInliers(prior, S, D, idx, (thr * 2.5) ** 2, pin);
    if (pin.length >= 6 && pin.length >= 0.6 * bestC) {
      const Mp = lsqSimilarity(S, D, pin);
      if (Mp) { const t = []; const c = countInliers(Mp, S, D, idx, thr2, t); if (c >= 4) { best = Mp; bestIn = t; bestC = c; } }
    }
  }
  if (!best || bestC < 4) return null;
  let M = best, inl = bestIn;
  for (let r = 0; r < 3; r++) {
    const M2 = lsqSimilarity(S, D, inl); if (!M2) break;
    const t = [];
    const c = countInliers(M2, S, D, idx, thr2, t);
    if (c < 4) break;
    M = M2; inl = t;
  }
  // inlier RMS
  let e = 0;
  for (const i of inl) {
    const x = M[0] * S[2 * i] - M[1] * S[2 * i + 1] + M[2] - D[2 * i];
    const y = M[1] * S[2 * i] + M[0] * S[2 * i + 1] + M[3] - D[2 * i + 1];
    e += x * x + y * y;
  }
  return { M, inliers: inl.length, rms: Math.sqrt(e / inl.length), inlierIdx: inl };
}

// ───────────────────────── sharpness metric ─────────────────────────
// High-percentile |Laplacian| (focus measure). Histogram percentile: O(n), no sorting.
const _hist = new Uint32Array(1024);
function histPercentile(fill, q) {
  _hist.fill(0); let n = 0;
  fill(v => { _hist[v > 1023 ? 1023 : v | 0]++; n++; });
  const target = q * n; let c = 0;
  for (let i = 0; i < 1024; i++) { c += _hist[i]; if (c >= target) return i; }
  return 1023;
}
export function frameSharpness(g, w, h) {
  // Laplacian on a 2×2-binned grid: real detail survives, per-pixel sensor noise averages out.
  const W2 = w >> 1, H2 = h >> 1;
  const b = (x, y) => { const i = 2 * y * w + 2 * x; return (g[i] + g[i + 1] + g[i + w] + g[i + w + 1]) * 0.25; };
  return histPercentile(add => {
    for (let y = 2; y < H2 - 2; y += 2) for (let x = 2 + (y & 2); x < W2 - 2; x += 2) {
      add(4 * Math.abs(4 * b(x, y) - b(x - 1, y) - b(x + 1, y) - b(x, y - 1) - b(x, y + 1)));
    }
  }, 0.97) / 4;
}

// ───────────────────────── frame-to-world tracker ─────────────────────────
// Tracks every frame against a keyframe (drift-free while the keyframe stays valid),
// falls back to frame-to-frame, re-keys when the view has changed too much.
export class MotionTracker {
  constructor(w, h, opts = {}) {
    this.w = w; this.h = h;
    this.levels = opts.levels || 4;
    this.half = opts.half || 7;
    this.maxFeatures = opts.maxFeatures || 180;
    this.roi = opts.roi || null; // {x, y, r} in world-normalised coords
    this.kf = null; this.prev = null; this.frame = 0; this.seed = 7;
  }

  _mask(C) {
    if (!this.roi) return null;
    const inv = invert(C), [cx, cy] = apply(inv, this.roi.x, this.roi.y);
    const r = this.roi.r / Math.hypot(C[0], C[1]) * this.h;
    const px = cx * this.h + this.w / 2, py = cy * this.h + this.h / 2, r2 = r * r;
    const fn = (x, y) => (x - px) * (x - px) + (y - py) * (y - py) < r2;
    fn.rect = { x0: Math.max(0, px - r), y0: Math.max(0, py - r), x1: Math.min(this.w, px + r), y1: Math.min(this.h, py + r) };
    return fn;
  }

  _makeKey(pyr, C) {
    const m = this._mask(C);
    const feats = detectFeatures(pyr[0].d, this.w, this.h, this.maxFeatures, m, 12, m ? m.rect : null);
    const tpl = makeTemplates(pyr, feats, this.half);
    return { pyr, C, feats, tpl, n0: 0, Q: IDENT.slice(), sharp: frameSharpness(pyr[1].d, pyr[1].w, pyr[1].h) };
  }

  _estimate(key, pyr, guessM) {
    const n = key.feats.length / 2;
    if (n < 6) return null;
    const guess = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const [x, y] = apply(guessM, key.feats[2 * i], key.feats[2 * i + 1]);
      guess[2 * i] = x; guess[2 * i + 1] = y;
    }
    const { pts, status } = trackLK(key.tpl, key.feats, pyr, guess);
    const idx = []; for (let i = 0; i < n; i++) if (status[i]) idx.push(i);
    const fit = ransacSimilarity(key.feats, pts, idx, 1.2, 160, this.seed++, guessM);
    if (!fit) return null;
    fit.tracked = idx.length;
    // Rotation/scale are only trustworthy if the inliers are spread out. A tight cluster (e.g. only the
    // spider has texture) pins translation but not rotation/scale: keep those from the prediction instead.
    let mx = 0, my = 0;
    for (const i of fit.inlierIdx) { mx += key.feats[2 * i]; my += key.feats[2 * i + 1]; }
    mx /= fit.inliers; my /= fit.inliers;
    let sp = 0;
    for (const i of fit.inlierIdx) sp += (key.feats[2 * i] - mx) ** 2 + (key.feats[2 * i + 1] - my) ** 2;
    sp = Math.sqrt(sp / fit.inliers);
    fit.spread = sp / this.h;
    if (fit.spread < 0.14) {
      const a = guessM[0], b = guessM[1];
      let tx = 0, ty = 0;
      for (const i of fit.inlierIdx) {
        tx += pts[2 * i] - (a * key.feats[2 * i] - b * key.feats[2 * i + 1]);
        ty += pts[2 * i + 1] - (b * key.feats[2 * i] + a * key.feats[2 * i + 1]);
      }
      fit.M = [a, b, tx / fit.inliers, ty / fit.inliers];
      fit.transOnly = true;
    }
    return fit;
  }

  // Wide translation search at the coarsest pyramid level (used only to re-acquire after a failure).
  _globalGuess(key, pyr, guessM) {
    const L = Math.min(key.pyr.length, pyr.length) - 1, sc = 1 << L;
    const A = key.pyr[L], B = pyr[L], R = 14;
    const pts = [];
    for (let y = 4; y < A.h - 4; y += 2) for (let x = 4; x < A.w - 4; x += 2) {
      const [qx, qy] = apply(guessM, x * sc, y * sc);
      pts.push(x, y, qx / sc, qy / sc);
    }
    let best = Infinity, bx = 0, by = 0;
    for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) {
      let sad = 0, n = 0, sd = 0;
      for (let i = 0; i < pts.length; i += 4) {
        const qx = pts[i + 2] + dx, qy = pts[i + 3] + dy;
        if (qx < 0 || qy < 0 || qx > B.w - 1.001 || qy > B.h - 1.001) continue;
        const d = bilin(B.d, B.w, B.h, qx, qy) - A.d[pts[i + 1] * A.w + pts[i]];
        sd += d; sad += Math.abs(d); n++;
      }
      if (n < pts.length / 16) continue;
      const score = (sad - Math.abs(sd)) / n; // brightness-offset tolerant
      if (score < best) { best = score; bx = dx; by = dy; }
    }
    return [guessM[0], guessM[1], guessM[2] + bx * sc, guessM[3] + by * sc];
  }

  // gray: Float32Array w*h. Returns { C, ok, inliers, rekey }
  push(gray) {
    const pyr = buildPyramid(gray, this.w, this.h, this.levels);
    this.frame++;
    if (!this.kf) {
      this.kf = this._makeKey(pyr, IDENT.slice());
      this.kf.n0 = this.kf.feats.length / 2;
      this.prev = { pyr, C: IDENT.slice(), ok: true };
      this.fails = 0;
      return { C: IDENT.slice(), ok: true, inliers: this.kf.n0, rekey: true };
    }
    const kf = this.kf;
    let C = null, ok = false, inliers = 0, Q = null;
    const minIn = Math.max(8, Math.round(0.2 * kf.n0));
    const good = f => f && f.inliers >= minIn;
    // constant-velocity prediction (handles deliberate follow-pans)
    let pred = kf.Q;
    if (kf.Qp) pred = [kf.Q[0], kf.Q[1], 2 * kf.Q[2] - kf.Qp[2], 2 * kf.Q[3] - kf.Qp[3]];
    let fit = this._estimate(kf, pyr, pred);
    if (!good(fit) && kf.Qp) fit = this._estimate(kf, pyr, kf.Q) || fit;
    if (!good(fit) && kf.n0 >= 12) fit = this._estimate(kf, pyr, this._globalGuess(kf, pyr, kf.Q)) || fit;
    if (good(fit)) {
      Q = fit.M; inliers = fit.inliers; ok = true;
      C = compose(kf.C, invert(pixToNorm(Q, this.w, this.h)));
    } else {
      // Frame-to-frame fallback. If the previous frame was itself lost its camera is a placeholder,
      // but relative motion is still right and the gap gets re-interpolated afterwards.
      const pk = this._makeKey(this.prev.pyr, this.prev.C);
      const vel = this.prev.vel || IDENT;
      let f2 = this._estimate(pk, pyr, vel);
      if (!(f2 && f2.inliers >= 8) && pk.feats.length >= 24) f2 = this._estimate(pk, pyr, this._globalGuess(pk, pyr, vel)) || f2;
      if (f2 && f2.inliers >= Math.max(8, 0.2 * pk.feats.length / 2)) {
        C = compose(this.prev.C, invert(pixToNorm(f2.M, this.w, this.h)));
        ok = this.prev.ok; inliers = f2.inliers; this._f2 = f2.M;
        if (!ok) { this.relock = true; }
      }
    }
    // Physically implausible single-frame rotation/zoom jumps are tracking errors, not camera motion.
    if (C && this.prev.ok) {
      const d = toParams(compose(invert(this.prev.C), C));
      if (Math.abs(d.th) > 0.04 || Math.abs(d.ls) > 0.03) { C = null; Q = null; ok = false; }
    }
    let rekey = false;
    const f2ok = !Q && C !== null;   // recovered via frame-to-frame
    if (ok || f2ok) {
      if (ok) this.fails = 0;
      if (Q) {
        const p = toParams(Q);
        rekey = inliers < 0.6 * kf.n0 || Math.abs(p.th) > 0.06 || Math.abs(p.ls) > 0.05;
        kf.Qp = kf.Q; kf.Q = Q;
      } else rekey = true;
      if (!ok) this.fails++;
    } else {
      // Lost (blur, occlusion). Keep the last good keyframe so we can re-acquire against it;
      // only give up on it if it was poor or we've been lost for a while.
      this.fails++;
      C = this.prev.C.slice();
      if (kf.n0 < 12 || this.fails > 20) rekey = true;
    }
    if (rekey) {
      const cand = this._makeKey(pyr, C); cand.n0 = cand.feats.length / 2;
      // Never swap a good anchor for a blurrier/emptier frame: during focus loss we keep the
      // last sharp anchor and re-acquire against it when focus returns.
      const kfBad = kf.n0 < 12;
      const candOk = cand.n0 >= Math.max(12, 0.3 * kf.n0) && cand.sharp >= 0.65 * kf.sharp;
      if ((kfBad && cand.n0 > kf.n0) || candOk || (f2ok && cand.n0 >= 12) || (this.fails > 20 && cand.n0 >= 12)) this.kf = cand;
      else rekey = false;
    }
    // per-frame velocity in pixel space (for predicting the next frame-to-frame step)
    let vel = IDENT;
    if (C && this.prev.C) { const d = compose(invert(C), this.prev.C); vel = [1, 0, d[2] * this.h, d[3] * this.h]; }
    this.prev = { pyr, C, ok, vel };
    return { C, ok, inliers, rekey };
  }
}

// ───────────────────────── gap filling ─────────────────────────
// Replace camera estimates on failed frames by interpolating (in θ, log-scale, x, y) between good neighbours.
export function fillGaps(C, ok, times) {
  const N = C.length, out = C.map(m => m.slice());
  let i = 0;
  while (i < N) {
    if (ok[i]) { i++; continue; }
    let j = i; while (j < N && !ok[j]) j++;
    const L = i - 1, R = j < N ? j : -1;
    for (let k = i; k < j; k++) {
      if (L < 0 && R < 0) break;
      if (L < 0) { out[k] = C[R].slice(); continue; }
      if (R < 0) { out[k] = C[L].slice(); continue; }
      const u = (times[k] - times[L]) / Math.max(1e-9, times[R] - times[L]);
      const a = toParams(C[L]), b = toParams(C[R]);
      let dth = b.th - a.th; while (dth > Math.PI) dth -= 2 * Math.PI; while (dth < -Math.PI) dth += 2 * Math.PI;
      out[k] = fromParams(a.th + u * dth, a.ls + u * (b.ls - a.ls), a.x + u * (b.x - a.x), a.y + u * (b.y - a.y));
    }
    i = j;
  }
  return out;
}

// ───────────────────────── path smoothing + auto crop ─────────────────────────
function unwrap(arr) {
  for (let i = 1; i < arr.length; i++) {
    let d = arr[i] - arr[i - 1];
    while (d > Math.PI) { arr[i] -= 2 * Math.PI; d -= 2 * Math.PI; }
    while (d < -Math.PI) { arr[i] += 2 * Math.PI; d += 2 * Math.PI; }
  }
  return arr;
}

function medianStep(times) {
  const d = []; for (let i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
  d.sort((a, b) => a - b); return d.length ? Math.max(1e-4, d[d.length >> 1]) : 1 / 30;
}

// O(N) Gaussian approximation: three box passes (σ in frames), optional mask weights on the first pass.
export function boxSmooth3(vals, sigma, mask) {
  const n = vals.length;
  const r = Math.max(0, Math.round((Math.sqrt(4 * sigma * sigma + 1) - 1) / 2));
  if (!r || n < 2) return Float64Array.from(vals);
  const pv = new Float64Array(n + 1), pw = new Float64Array(n + 1);
  let cur = Float64Array.from(vals);
  for (let pass = 0; pass < 3; pass++) {
    for (let i = 0; i < n; i++) {
      const w = pass === 0 && mask ? (mask[i] ? 1 : 0) : 1;
      pv[i + 1] = pv[i] + cur[i] * w; pw[i + 1] = pw[i] + w;
    }
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = Math.max(0, i - r), b = Math.min(n, i + r + 1), ws = pw[b] - pw[a];
      out[i] = ws > 0 ? (pv[b] - pv[a]) / ws : cur[i];
    }
    cur = out;
  }
  return cur;
}

function gaussSmooth(vals, times, sigma, mask) {
  const n = vals.length, out = new Float64Array(n);
  if (sigma <= 0) { out.set(vals); return out; }
  let lo = 0;
  for (let i = 0; i < n; i++) {
    const t = times[i];
    while (times[lo] < t - 3 * sigma) lo++;
    let s = 0, ws = 0;
    for (let j = lo; j < n && times[j] <= t + 3 * sigma; j++) {
      if (mask && !mask[j]) continue;
      const d = times[j] - t, w = Math.exp(-d * d / (2 * sigma * sigma));
      s += w * vals[j]; ws += w;
    }
    out[i] = ws > 0 ? s / ws : vals[i];
  }
  return out;
}

// Given W (output-norm-independent warp, display-norm → source-norm) and framing,
// return the minimum total zoom Z such that the output rectangle stays inside the source.
// framing: { A (source aspect), Ao (output aspect), base, rot (rad), panX, panY }
export function needZoom(W, fr) {
  const [s0x, s0y] = apply(W, fr.panX, fr.panY);
  const hx = fr.A / 2, hy = 0.5;
  if (Math.abs(s0x) >= hx || Math.abs(s0y) >= hy) return Infinity;
  const c = Math.cos(fr.rot), s = Math.sin(fr.rot);
  let umax = Infinity;
  for (const [ox, oy] of [[-fr.Ao / 2, -0.5], [fr.Ao / 2, -0.5], [fr.Ao / 2, 0.5], [-fr.Ao / 2, 0.5]]) {
    const rx = fr.base * (c * ox - s * oy), ry = fr.base * (s * ox + c * oy);
    const vx = W[0] * rx - W[1] * ry, vy = W[1] * rx + W[0] * ry;
    if (vx > 1e-12) umax = Math.min(umax, (hx - s0x) / vx); else if (vx < -1e-12) umax = Math.min(umax, (hx + s0x) / -vx);
    if (vy > 1e-12) umax = Math.min(umax, (hy - s0y) / vy); else if (vy < -1e-12) umax = Math.min(umax, (hy + s0y) / -vy);
  }
  return 1 / umax;
}

// Output height in source-height units at zoom 1. quarter = extra 90° turns applied by the user.
export function framingBase(A, Ao, quarter = 0) {
  const odd = (quarter & 1) === 1, Aq = odd ? 1 / A : A, Hq = odd ? A : 1;
  return Hq * Math.min(1, Aq / Ao);
}

/**
 * opts: {
 *   C: Array<[a,b,tx,ty]> | null, times: Float64Array (s), kept: Uint8Array|null,
 *   mode: 'smooth'|'lock', sigma (s), rotation: bool, zoom: bool, strength (0..1),
 *   A, Ao, rot (rad), zoomUser (>=1), panX, panY, maxCrop (fraction, e.g. 0.2)
 * }
 * returns { M: Float64Array(N*4) output-norm → source-display-norm similarity per frame,
 *           Z, cropPct, limitedFrames }
 */
export function computeStabilisation(o) {
  const N = o.times.length;
  const q = o.quarter || 0;
  const fr = { A: o.A, Ao: o.Ao, base: framingBase(o.A, o.Ao, q), rot: (o.rot || 0) + q * Math.PI / 2, panX: o.panX || 0, panY: o.panY || 0 };
  const M = new Float64Array(N * 4);
  const zFrame = needZoom(IDENT, fr);
  const zFloor = Math.max(o.zoomUser || 1, zFrame);
  const W = new Array(N);
  const haveStab = !!(o.C && o.C.length === N);
  let Z = zFloor, adjusted = 0, touching = 0, paths = null;
  if (haveStab) {
    const C = o.C;
    const P = [unwrap(Float64Array.from(C, m => Math.atan2(m[1], m[0]))),
      Float64Array.from(C, m => Math.log(Math.hypot(m[0], m[1]))),
      Float64Array.from(C, m => m[2]), Float64Array.from(C, m => m[3])];
    const mask = o.kept && o.kept.some(v => v) ? o.kept : null;
    const dt = medianStep(o.times);
    const lock = o.mode === 'lock';
    const st = o.strength == null ? 1 : o.strength;
    const active = [o.rotation !== false, o.zoom !== false, true, true];
    // Ladder of smoothing strengths, strongest first. Where the crop budget can't cover the strongest,
    // we fall back to weaker low-pass for those moments only: slow drift is followed, jitter still removed.
    const sig = Math.max(0.05, o.sigma || 0.8);
    const ladder = lock ? [Infinity, 4, 2, 1, 0.5, 0.25, 0.12, 0.06] : [];
    if (!lock) { for (let x = sig; x > 0.05; x *= 0.6) ladder.push(x); ladder.push(0.04); }
    const mkPath = (sg) => P.map((arr, c) => {
      if (!active[c]) return Float64Array.from(arr);
      let sm;
      if (sg === Infinity) { let t = 0, n = 0; for (let i = 0; i < N; i++) if (!mask || mask[i]) { t += arr[i]; n++; } sm = new Float64Array(N).fill(t / Math.max(1, n)); }
      else sm = boxSmooth3(arr, sg / dt, mask);
      return Float64Array.from(arr, (v, i) => v + st * (sm[i] - v));
    });
    const L = [mkPath(ladder[0])];
    const warp = (i, Sx) => compose(invert(C[i]), fromParams(Sx[0][i], Sx[1][i], Sx[2][i], Sx[3][i]));
    const zMax = zFloor * (1 + Math.max(0, o.maxCrop ?? 0.2));
    let need = 0;
    for (let i = 0; i < N; i++) if (!o.kept || o.kept[i]) need = Math.max(need, needZoom(warp(i, L[0]), fr));
    Z = Math.min(Math.max(need, zFloor), zMax);
    let S = L[0].map(a => Float64Array.from(a));
    if (need > Z * 1.0001) {
      // feasible level per frame (0 = strongest smoothing)
      const nL = ladder.length, feas = new Float64Array(N);
      for (let j = 1; j < nL; j++) L.push(mkPath(ladder[j]));
      for (let i = 0; i < N; i++) {
        let j = 0;
        while (j < nL - 1 && needZoom(warp(i, L[j]), fr) > Z) j++;
        feas[i] = (o.kept && !o.kept[i]) ? 0 : j;
      }
      // spread + ease the level changes in time so the switch itself is invisible
      const win = Math.max(1, Math.round(0.4 / dt)), dil = new Float64Array(N);
      for (let i = 0; i < N; i++) { let m = 0; for (let k = Math.max(0, i - win); k <= Math.min(N - 1, i + win); k++) m = Math.max(m, feas[k]); dil[i] = m; }
      const lvS = boxSmooth3(dil, 0.2 / dt, null);
      for (let i = 0; i < N; i++) {
        const lv = Math.min(nL - 1, Math.max(lvS[i], feas[i]));
        const j0 = Math.floor(lv), j1 = Math.min(nL - 1, j0 + 1), u = lv - j0;
        for (let c = 0; c < 4; c++) S[c][i] = L[j0][c][i] + u * (L[j1][c][i] - L[j0][c][i]);
        if (lv > 1e-3) adjusted++;
      }
      // last resort: frames that still don't fit (even light smoothing) get pulled toward the raw camera
      for (let i = 0; i < N; i++) {
        if ((o.kept && !o.kept[i]) || needZoom(warp(i, S), fr) <= Z * 1.0001) continue;
        const keep = S.map(a => a[i]); let lo = 0, hi = 1;
        for (let it = 0; it < 16; it++) {
          const m = (lo + hi) / 2;
          for (let c = 0; c < 4; c++) S[c][i] = P[c][i] + m * (keep[c] - P[c][i]);
          if (needZoom(warp(i, S), fr) <= Z) lo = m; else hi = m;
        }
        for (let c = 0; c < 4; c++) S[c][i] = P[c][i] + lo * (keep[c] - P[c][i]);
        touching++;
      }
    }
    for (let i = 0; i < N; i++) W[i] = warp(i, S);
    paths = { rx: P[2], ry: P[3], sx: S[2], sy: S[3], rth: P[0], sth: S[0] };
  }
  const c = Math.cos(fr.rot), s = Math.sin(fr.rot), k = fr.base / Z;
  const F = [k * c, k * s, fr.panX, fr.panY];
  for (let i = 0; i < N; i++) {
    const m = compose(haveStab ? W[i] : IDENT, F);
    M[4 * i] = m[0]; M[4 * i + 1] = m[1]; M[4 * i + 2] = m[2]; M[4 * i + 3] = m[3];
  }
  return { M, Z, zFloor, cropPct: (1 - 1 / Z) * 100, limitedFrames: adjusted, touchingFrames: touching, paths };
}

// Clamp a pan so that the un-stabilised view fits at zoom z.
export function clampPan(A, Ao, quarter, rot, z, panX, panY) {
  const base = framingBase(A, Ao, quarter);
  const fr = { A, Ao, base, rot: rot + quarter * Math.PI / 2, panX, panY };
  if (needZoom(IDENT, fr) <= z) return [panX, panY];
  let lo = 0, hi = 1;
  for (let i = 0; i < 20; i++) {
    const m = (lo + hi) / 2; fr.panX = panX * m; fr.panY = panY * m;
    if (needZoom(IDENT, fr) <= z) lo = m; else hi = m;
  }
  return [panX * lo, panY * lo];
}

// nearest frame index for time t (times sorted ascending)
export function frameAt(times, t) {
  let lo = 0, hi = times.length - 1;
  if (hi < 0) return 0;
  while (lo < hi) { const m = (lo + hi) >> 1; if (times[m] < t) lo = m + 1; else hi = m; }
  if (lo > 0 && Math.abs(times[lo - 1] - t) <= Math.abs(times[lo] - t)) return lo - 1;
  return lo;
}

// ───────────────────────── auto-trim suggestions ─────────────────────────
// Keep the stretches that are in focus and not mid-sweep. Returns [{start, end}] (seconds).
export function suggestCuts(times, sharp, ok, C, o = {}) {
  const N = times.length; if (N < 3) return [{ start: 0, end: times[N - 1] || 0 }];
  const dt = medianStep(times);
  const sorted = Array.from(sharp).sort((a, b) => a - b), ref = sorted[Math.floor(0.85 * (N - 1))] || 1;
  const blurFrac = o.blur ?? 0.4, minKeep = o.minKeep ?? 1.2, maxGap = o.maxGap ?? 0.6;
  const speed = new Float64Array(N);
  for (let k = 1; k < N; k++) speed[k] = Math.hypot(C[k][2] - C[k - 1][2], C[k][3] - C[k - 1][3]) / Math.max(1e-3, times[k] - times[k - 1]);
  speed[0] = speed[1];
  // a "sweep" is camera motion well above this clip's normal handheld following (frame-heights per second)
  const medSpeed = Array.from(speed).sort((a, b) => a - b)[N >> 1];
  const sweep = o.sweep ?? Math.max(1.2, 3 * medSpeed);
  const spS = boxSmooth3(speed, 0.15 / dt, null);
  const bad = new Float64Array(N);
  for (let k = 0; k < N; k++) bad[k] = (!ok[k] || sharp[k] < blurFrac * ref || spS[k] > sweep) ? 1 : 0;
  const badS = boxSmooth3(bad, 0.2 / dt, null);
  let segs = [];
  for (let k = 0; k < N;) {
    if (badS[k] >= 0.5) { k++; continue; }
    let j = k; while (j < N && badS[j] < 0.5) j++;
    segs.push({ start: times[k], end: j < N ? times[j] : times[N - 1] + dt });
    k = j;
  }
  // merge tiny gaps, drop tiny keeps
  const merged = [];
  for (const s of segs) {
    const last = merged[merged.length - 1];
    if (last && s.start - last.end < maxGap) last.end = s.end; else merged.push({ ...s });
  }
  segs = merged.filter(s => s.end - s.start >= minKeep);
  return segs.length ? segs : [{ start: times[0], end: times[N - 1] + dt }];
}
