// Run tracker + stabiliser on real footage (gray frames + times json), dump per-frame results.
import fs from 'node:fs';
import { MotionTracker, computeStabilisation, fillGaps, frameSharpness } from '../js/core.js';
const [base, out, mode = 'smooth', sigma = '0.8', maxCrop = '0.25', k0 = '0', k1 = '1e9'] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(base + '.json')), { W, H } = meta, times = Float64Array.from(meta.times);
const fd = fs.openSync(base + '.gray', 'r'), N = times.length, buf = Buffer.alloc(W * H);
const tr = new MotionTracker(W, H); const C = [], ok = new Uint8Array(N), inl = [], sharp = [], rekeys = [];
const t0 = performance.now();
for (let k = 0; k < N; k++) {
  fs.readSync(fd, buf, 0, W * H, k * W * H);
  const g = Float32Array.from(buf);
  const r = tr.push(g); C.push(r.C); ok[k] = r.ok ? 1 : 0; inl.push(r.inliers); sharp.push(frameSharpness(g, W, H)); if (r.rekey) rekeys.push(k);
}
const ms = (performance.now() - t0) / N;
const Cf = fillGaps(C, ok, times);
const st = computeStabilisation({ C: Cf, times, kept: Uint8Array.from(times, (_, i) => i >= +k0 && i < +k1 ? 1 : 0), mode, sigma: +sigma, A: W / H, Ao: W / H, rot: 0, zoomUser: 1, maxCrop: +maxCrop, rotation: true, zoom: true });
const fails = N - ok.reduce((a, b) => a + b, 0);
const q = (a, f) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(f * (s.length - 1))]; };
console.log(`${N} frames, ${ms.toFixed(1)} ms/frame, fails ${fails} (${(100 * fails / N).toFixed(1)}%), rekeys ${rekeys.length}, inliers p10/p50 ${q(inl, .1)}/${q(inl, .5)}`);
console.log(`mode ${mode} σ ${sigma}s maxCrop ${maxCrop}: crop ${st.cropPct.toFixed(1)}%, lighter-smoothing frames ${st.limitedFrames}, touching ${st.touchingFrames}`);
const failRuns = []; for (let i = 0; i < N; i++) if (!ok[i]) { let j = i; while (j < N && !ok[j]) j++; failRuns.push(`${times[i].toFixed(1)}s×${j - i}`); i = j; }
console.log('fail runs:', failRuns.slice(0, 30).join(' '));
fs.writeFileSync(out, JSON.stringify({ M: Array.from(st.M), ok: Array.from(ok), C: Cf, inl, sharp, Z: st.Z }));
