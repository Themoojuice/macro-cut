// Score tracker + stabiliser against ground truth.
// jitter = mean |second difference| of output-corner world positions (shake), drift = mean |first difference|.
import fs from 'node:fs';
import { MotionTracker, compose, invert, computeStabilisation, fillGaps } from '../js/core.js';
import * as core from '../js/core.js';
const base = process.argv[2], modes = (process.argv[3] || 'smooth,lock').split(','), maxCrop = +(process.argv[4] || 0.2);
const meta = JSON.parse(fs.readFileSync(base + '.json')), { W, H, fps } = meta;
const raw = fs.readFileSync(base + '.gray'), N = raw.length / (W * H);
const tr = new MotionTracker(W, H); const C = [], ok = new Uint8Array(N);
const t0 = performance.now();
for (let k = 0; k < N; k++) { const r = tr.push(Float32Array.from(raw.subarray(k * W * H, (k + 1) * W * H))); C.push(r.C); ok[k] = r.ok ? 1 : 0; }
const ms = (performance.now() - t0) / N;
const T = meta.gt.map(([a, b, px, py]) => [a, b, px / H, py / H]), iT0 = invert(T[0]);
const G = T.map(t => compose(iT0, t));
const corners = [[-W / H / 2, -0.5], [W / H / 2, -0.5], [W / H / 2, 0.5], [-W / H / 2, 0.5]];
const ap = (m, x, y) => [m[0] * x - m[1] * y + m[2], m[1] * x + m[0] * y + m[3]];
let te = 0; for (let k = 0; k < N; k++) for (const [x, y] of corners) { const a = ap(G[k], x, y), b = ap(C[k], x, y); te = Math.max(te, Math.hypot(a[0] - b[0], a[1] - b[1]) * H); }
const fails = N - ok.reduce((a, b) => a + b, 0);
function path(getM) { // world position of output corners per frame
  return Array.from({ length: N }, (_, k) => corners.map(([x, y]) => ap(compose(G[k], getM(k)), x, y)));
}
function stats(P) {
  let d1 = 0, d2 = 0;
  for (let k = 1; k < N; k++) for (let c = 0; c < 4; c++) d1 += Math.hypot(P[k][c][0] - P[k - 1][c][0], P[k][c][1] - P[k - 1][c][1]);
  for (let k = 2; k < N; k++) for (let c = 0; c < 4; c++) d2 += Math.hypot(P[k][c][0] - 2 * P[k - 1][c][0] + P[k - 2][c][0], P[k][c][1] - 2 * P[k - 1][c][1] + P[k - 2][c][1]);
  return [d1 / (4 * (N - 1)) * H, d2 / (4 * (N - 2)) * H];
}
const [r1, r2] = stats(path(() => [1, 0, 0, 0]));
console.log(`${base.split('/').pop()}: ${N} frames, ${ms.toFixed(1)} ms/frame, track fails ${fails}, max track err ${te.toFixed(2)} px`);
console.log(`  raw          drift ${r1.toFixed(2)}  jitter ${r2.toFixed(2)} px`);
const times = Float64Array.from({ length: N }, (_, i) => i / fps);
const Cf = fillGaps(C, ok, times);
for (const mode of modes) {
  const st = computeStabilisation({ C: Cf, times, kept: null, mode, sigma: 0.8, A: W / H, Ao: W / H, rot: 0, zoomUser: 1, maxCrop });
  const [s1, s2] = stats(path(k => Array.from(st.M.slice(4 * k, 4 * k + 4))));
  console.log(`  ${mode.padEnd(6)} crop ${st.cropPct.toFixed(1).padStart(4)}%  adjusted ${String(st.limitedFrames).padStart(4)} touch ${String(st.touchingFrames).padStart(3)}  drift ${s1.toFixed(2)}  jitter ${s2.toFixed(2)} px  (jitter ÷${(r2 / s2).toFixed(1)})`);
}
void core;
