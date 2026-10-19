// Interpolation quality on real-ish footage: 60 fps synthetic clip, rebuild every odd frame from its neighbours.
// Compare to: holding the previous frame (current app) and a plain cross-fade.
import fs from 'node:fs';
import { denseFlow, buildPyramid, globalShift } from '../js/core.js';
const base = process.argv[2], GAP = +process.argv[3] || 2, meta = JSON.parse(fs.readFileSync(base + '.json')), { W, H } = meta;
const raw = fs.readFileSync(base + '.gray'), N = raw.length / (W * H);
const fr = k => Float32Array.from(raw.subarray(k * W * H, (k + 1) * W * H));
function bil(d, x, y) {
  x = Math.min(W - 1.001, Math.max(0, x)); y = Math.min(H - 1.001, Math.max(0, y));
  const x0 = x | 0, y0 = y | 0, ax = x - x0, ay = y - y0, i = y0 * W + x0;
  return d[i] * (1 - ax) * (1 - ay) + d[i + 1] * ax * (1 - ay) + d[i + W] * (1 - ax) * ay + d[i + W + 1] * ax * ay;
}
const psnr = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return 10 * Math.log10(255 * 255 / (s / a.length)); };
let P = { hold: 0, fade: 0, flow: 0 }, cnt = 0, ms = 0, vf = 0;
for (let k = 0; k + GAP < N; k += 6) {
  const A = fr(k), B = fr(k + GAP);
  const t0 = performance.now();
  const pa = buildPyramid(A, W, H, 4), pb = buildPyramid(B, W, H, 4);
  const F = denseFlow(A, B, W, H, { pyrA: pa, pyrB: pb, guess: globalShift(pa, pb), step: 16, half: 5 });
  ms += performance.now() - t0; vf += F.validFrac;
  const sf = (arr, x, y) => {
    const u = Math.min(F.gw - 1.001, Math.max(0, (x + 0.5) * F.gw / W - 0.5)), v = Math.min(F.gh - 1.001, Math.max(0, (y + 0.5) * F.gh / H - 0.5));
    const x0 = u | 0, y0 = v | 0, ax = u - x0, ay = v - y0, i = y0 * F.gw + x0;
    return arr[i] * (1 - ax) * (1 - ay) + arr[i + 1] * ax * (1 - ay) + arr[i + F.gw] * (1 - ax) * ay + arr[i + F.gw + 1] * ax * ay;
  };
  for (let j = 1; j < GAP; j++) {
    const t = j / GAP, T = fr(k + j), I = new Float32Array(W * H), X = new Float32Array(W * H);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, u = sf(F.fx, x, y), v = sf(F.fy, x, y);
      I[i] = (1 - t) * bil(A, x - t * u, y - t * v) + t * bil(B, x + (1 - t) * u, y + (1 - t) * v);
      X[i] = (1 - t) * A[i] + t * B[i];
    }
    P.hold += psnr(t < 0.5 ? A : B, T); P.fade += psnr(X, T); P.flow += psnr(I, T); cnt++;
  }
}
console.log(`${base.split('/').pop()}: ${cnt} rebuilt frames, ${(ms / cnt).toFixed(0)} ms per pair, reliable vectors ${(100 * vf / cnt).toFixed(0)}%`);
console.log(`  PSNR vs real missing frame (higher = closer):  repeat frame ${(P.hold / cnt).toFixed(1)} dB | cross-fade ${(P.fade / cnt).toFixed(1)} dB | motion-based ${(P.flow / cnt).toFixed(1)} dB`);
