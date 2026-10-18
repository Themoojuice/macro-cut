// Node test: tracker accuracy vs ground truth on synthetic clip, then stabilisation residual.
import fs from 'node:fs';
import { MotionTracker, compose, invert, toParams, computeStabilisation } from '../js/core.js';
const base = process.argv[2];
const meta = JSON.parse(fs.readFileSync(base + '.json'));
const { W, H, fps } = meta;
const raw = fs.readFileSync(base + '.gray');
const N = raw.length / (W * H);
const tr = new MotionTracker(W, H);
const C = []; let t0 = performance.now(), fails = 0, rekeys = 0;
for (let k = 0; k < N; k++) {
  const g = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = raw[k * W * H + i];
  const r = tr.push(g); C.push(r.C); if (!r.ok) fails++; if (r.rekey) rekeys++;
}
const ms = (performance.now() - t0) / N;
// ground truth C_k = T0^-1 ∘ Tk in normalised coords
const T = meta.gt.map(([a, b, px, py]) => [a, b, px / H, py / H]);
const iT0 = invert(T[0]);
let maxPx = 0, rmsPx = 0, maxDeg = 0;
const corners = [[-W / H / 2, -0.5], [W / H / 2, -0.5], [W / H / 2, 0.5], [-W / H / 2, 0.5]];
for (let k = 0; k < N; k++) {
  const G = compose(iT0, T[k]);
  for (const [x, y] of corners) {
    const ge = [G[0] * x - G[1] * y + G[2], G[1] * x + G[0] * y + G[3]];
    const e = [C[k][0] * x - C[k][1] * y + C[k][2], C[k][1] * x + C[k][0] * y + C[k][3]];
    const d = Math.hypot(ge[0] - e[0], ge[1] - e[1]) * H;
    maxPx = Math.max(maxPx, d); rmsPx += d * d;
  }
  maxDeg = Math.max(maxDeg, Math.abs(toParams(G).th - toParams(C[k]).th) * 180 / Math.PI);
}
rmsPx = Math.sqrt(rmsPx / (4 * N));
console.log(`frames ${N}  ${ms.toFixed(1)} ms/frame  fails ${fails}  rekeys ${rekeys}`);
console.log(`corner error vs truth (analysis px): rms ${rmsPx.toFixed(3)}  max ${maxPx.toFixed(3)}  rot err max ${maxDeg.toFixed(3)}°`);
// stabilise (lock) and measure residual motion of output corners in world
const times = Float64Array.from({ length: N }, (_, i) => i / fps);
for (const mode of ['lock', 'smooth']) {
  const st = computeStabilisation({ C, times, kept: null, mode, sigma: 0.5, A: W / H, Ao: W / H, rot: 0, zoomUser: 1, maxCrop: 0.3 });
  // output point o -> source s = M o ; world (truth) = G_k(s). Shake = frame-to-frame jump of world pos of output corners.
  let jit = 0, rawJit = 0;
  let prevW = null, prevR = null;
  for (let k = 0; k < N; k++) {
    const G = compose(iT0, T[k]); const m = Array.from(st.M.slice(4 * k, 4 * k + 4));
    const Wm = compose(G, m);
    if (prevW) {
      for (const [x, y] of corners) {
        const a = [Wm[0] * x - Wm[1] * y + Wm[2], Wm[1] * x + Wm[0] * y + Wm[3]];
        const b = [prevW[0] * x - prevW[1] * y + prevW[2], prevW[1] * x + prevW[0] * y + prevW[3]];
        jit += Math.hypot(a[0] - b[0], a[1] - b[1]) * H;
        const c = [G[0] * x - G[1] * y + G[2], G[1] * x + G[0] * y + G[3]];
        const d = [prevR[0] * x - prevR[1] * y + prevR[2], prevR[1] * x + prevR[0] * y + prevR[3]];
        rawJit += Math.hypot(c[0] - d[0], c[1] - d[1]) * H;
      }
    }
    prevW = Wm; prevR = G;
  }
  console.log(`${mode}: crop ${st.cropPct.toFixed(1)}%  limited ${st.limitedFrames}  mean frame-to-frame corner motion raw ${(rawJit / (4 * (N - 1))).toFixed(2)} px -> stabilised ${(jit / (4 * (N - 1))).toFixed(2)} px`);
}
