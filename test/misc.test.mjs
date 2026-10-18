import fs from 'node:fs';
import assert from 'node:assert';
import { MotionTracker, computeStabilisation, needZoom, framingBase, IDENT, invert, compose } from '../js/core.js';
// 1) quarter-turn framing: portrait output from landscape source at zoom 1 needs no extra zoom
const A = 16 / 9;
const r = computeStabilisation({ C: null, times: new Float64Array([0]), A, Ao: 1 / A, quarter: 1, rot: 0, zoomUser: 1 });
assert(Math.abs(r.Z - 1) < 1e-9, 'quarter turn zoom ' + r.Z);
assert(Math.abs(framingBase(A, 1 / A, 1) - A) < 1e-12);
// fine rotation needs zoom > 1 and needZoom agrees with geometry (5° on 16:9)
const z = needZoom(IDENT, { A, Ao: A, base: 1, rot: 5 * Math.PI / 180, panX: 0, panY: 0 });
assert(z > 1.1 && z < 1.25, 'rot zoom ' + z);
console.log('framing ok: 5° rotation needs', z.toFixed(3), 'x zoom');
// 2) ROI tracking on the synthetic clip
const base = process.argv[2], meta = JSON.parse(fs.readFileSync(base + '.json'));
const { W, H } = meta, raw = fs.readFileSync(base + '.gray'), N = raw.length / (W * H);
const tr = new MotionTracker(W, H, { roi: { x: 0.0, y: 0.0, r: 0.2 } });
const T = meta.gt.map(([a, b, px, py]) => [a, b, px / H, py / H]), iT0 = invert(T[0]);
let err = 0;
for (let k = 0; k < N; k++) {
  const g = Float32Array.from(raw.subarray(k * W * H, (k + 1) * W * H));
  const { C } = tr.push(g), G = compose(iT0, T[k]);
  err = Math.max(err, Math.hypot(C[2] - G[2], C[3] - G[3]) * H);
}
console.log('ROI tracking max centre error', err.toFixed(3), 'px');
assert(err < 2);
