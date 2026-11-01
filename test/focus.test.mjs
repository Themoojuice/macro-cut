// Node test: focus build-up geometry (canvas sizes, frame → canvas mapping, when a fresh canvas starts).
import assert from 'node:assert/strict';
import { canvasGeometry, frameToAnchor, fitsCanvas, planAnchors, decisionParams, lodForSigma, FOCUS_MARGIN, poseTracker, POSE } from '../js/focus.js';
import { compose, apply, fromParams } from '../js/core.js';

let n = 0; const ok = (name, f) => { f(); n++; console.log('ok', name); };
const near = (a, b, e = 1e-9) => assert.ok(Math.abs(a - b) <= e, `${a} vs ${b}`);

ok('canvas: 4K source to 4K export keeps full resolution, plus margin', () => {
  const g = canvasGeometry(3840, 2160, 0.9);
  near(g.scale, 1); assert.equal(g.AH, Math.round(2160 * (1 + 2 * FOCUS_MARGIN))); near(g.CW / g.CH, g.AW / g.AH, 1e-3);
  assert.ok(g.DH <= 540 * (1 + 2 * FOCUS_MARGIN) + 1 && g.DH >= 600, `decision grid ${g.DW}×${g.DH}`);
});
ok('canvas: 4K source to 720p export uses a lighter canvas', () => {
  const g = canvasGeometry(3840, 2160, 2.7);
  assert.ok(g.scale < 0.65 && g.AH < 1700, `scale ${g.scale}, ${g.AW}×${g.AH}`);
  assert.ok(decisionParams(2160, g).down > 0, 'downsampling into the canvas uses mipmaps');
});
ok('canvas: respects the GPU texture limit', () => {
  const g = canvasGeometry(7680, 4320, 1, 8192);
  assert.ok(Math.max(g.AW, g.AH) <= 8192);
});
ok('frame → anchor: a point lands where both cameras agree in the world', () => {
  const Ca = fromParams(0.03, 0.02, 0.01, -0.02), Ck = fromParams(-0.01, -0.03, 0.05, 0.04);
  const G = frameToAnchor(Ca, Ck);
  for (const [x, y] of [[0, 0], [0.7, -0.4], [-0.8, 0.5]]) {
    const w1 = apply(Ck, x, y), w2 = apply(Ca, ...apply(G, x, y));
    near(w1[0], w2[0], 1e-12); near(w1[1], w2[1], 1e-12);
  }
});
ok('fits: small shake fits, a big pan, turn or zoom does not', () => {
  const g = canvasGeometry(1920, 1080, 1);
  assert.ok(fitsCanvas([1, 0, 0.03, -0.02], g));
  assert.ok(fitsCanvas(fromParams(0.02, 0.04, 0.02, 0.02), g));
  assert.ok(!fitsCanvas([1, 0, 0.5, 0], g), 'pan by over a quarter of the frame width');
  assert.ok(!fitsCanvas(fromParams(0.3, 0, 0, 0), g), '17° turn');
  assert.ok(!fitsCanvas(fromParams(0, Math.log(1.3), 0, 0), g), '30% zoom');
});
ok('anchors: steady clip = one canvas; drift = fresh canvas, deterministic', () => {
  const g = canvasGeometry(1920, 1080, 1);
  const N = 300, frames = [...Array(N).keys()];
  const steady = frames.map(k => [1, 0, 0.01 * Math.sin(k / 7), 0.01 * Math.cos(k / 5)]);
  assert.equal(new Set(planAnchors(frames, steady, g).values()).size, 1);
  const drift = frames.map(k => [1, 0, 0.004 * k, 0]);   // slow pan: 1.2 frame-heights over the clip
  const A1 = planAnchors(frames, drift, g), A2 = planAnchors(frames, drift, g);
  const anchors = [...new Set(A1.values())];
  assert.ok(anchors.length >= 3, `anchors ${anchors}`);
  assert.deepEqual([...A1.entries()], [...A2.entries()]);
  for (const [k, a] of A1) assert.ok(a <= k && fitsCanvas(frameToAnchor(drift[a], drift[k]), g));
  // only processed frames get anchors (cut-out gaps skipped)
  const some = frames.filter(k => k % 3 === 0);
  assert.equal(planAnchors(some, steady, g).size, some.length);
});
ok('decision blur levels grow with resolution (same look at any resolution)', () => {
  const p720 = decisionParams(720, canvasGeometry(1280, 720, 1)), p4k = decisionParams(2160, canvasGeometry(3840, 2160, 1));
  near(p4k.lodW - p720.lodW, Math.log2(3), 1e-9);
  for (let i = 0; i < 6; i++) assert.ok(p4k.lodA[i] >= p720.lodA[i]);
  near(lodForSigma(0.29 * 8), 3, 1e-9);
});
ok('composition sanity (core.compose used for anchors)', () => {
  const C = fromParams(0.1, 0.05, 0.2, -0.1);
  const G = frameToAnchor(C, C);
  near(G[0], 1, 1e-12); near(G[1], 0, 1e-12); near(G[2], 0, 1e-12); near(G[3], 0, 1e-12);
  void compose;
});
// pose tracking ("sharp from the start"): frames list the regions that stopped matching ({ id: share of the
// detailed area }); like the GPU mask, "since" is the area of the union since the tracker last asked for a reset.
const run = (frames, fps = 30) => {
  const p = poseTracker(); let seen = new Map();
  return frames.map((f, k) => {
    for (const [id, a] of Object.entries(f)) seen.set(id, a);
    const o = p.push(k, k / fps, [...seen.values()].reduce((x, y) => x + y, 0));
    if (o.reset) seen = new Map();
    return o;
  });
};
const rng = (seed => () => (seed = (seed * 16807) % 2147483647) / 2147483647)(7);
const palps = () => (rng() < 0.5 ? { palpL: 0.03 } : {}) ;
ok('pose: a still spider flicking its palps never splits, and keeps snapshotting', () => {
  for (const fps of [30, 60]) {
    const out = run(Array.from({ length: 10 * fps }, palps), fps);
    assert.equal(out.filter(o => o.brk).length, 0);
    assert.ok(out.filter(o => o.snap).length >= 30, `snapshots refresh while still (${fps} fps)`);
  }
});
ok('pose: a head turn splits at the last still snapshot before it', () => {
  const f = Array.from({ length: 200 }, (_, k) => (k >= 100 && k < 106 ? { ['head' + k]: 0.08, ...palps() } : palps()));
  const out = run(f), b = out.filter(o => o.brk && o.brk.end != null);
  assert.equal(b.length, 1, 'one pose ends with a snapshot (any later splits mid-turn have none)');
  assert.ok(out.filter(o => o.brk).every(o => o.k >= 100 && o.k < 110));
  const lastSnapBefore = Math.max(...out.filter(o => o.snap && o.k < 100).map(o => o.k));
  assert.equal(b[0].brk.end, lastSnapBefore); assert.ok(100 - lastSnapBefore <= 9, `snapshot ${100 - lastSnapBefore} frames before the move`);
  assert.ok(out.some(o => o.snap && o.k > 110), 'snapshots resume once still');
});
ok('pose: a slow drift that never looks like movement in any one window does not split (behaves like Build up)', () => {
  const out = run(Array.from({ length: 240 }, (_, k) => (k >= 80 && k < 140 ? { ['drop' + k]: 0.006 } : {})));
  assert.equal(out.filter(o => o.brk).length, 0); assert.ok(out.some(o => o.snap && o.k > 140), 'still snapshotting');
});
ok('pose: a quick reposition spread over two windows carries over and splits', () => {
  const f = Array.from({ length: 200 }, (_, k) => (k >= 100 && k < 112 ? { ['m' + k]: 0.05 } : {}));
  const b = run(f).filter(o => o.brk);
  assert.ok(b.length >= 1 && b[0].brk.end != null && b[0].brk.end < 100);
});
ok('pose: walking splits without a still snapshot to keep', () => {
  const out = run(Array.from({ length: 150 }, (_, k) => ({ ['walk' + k]: 0.05 })));
  assert.ok(out.filter(o => o.brk).length >= 3 && out.every(o => !o.brk || o.brk.end == null) && out.every(o => !o.snap));
});
ok('pose: parameters sane', () => { assert.ok(POSE.quiet < POSE.brk && POSE.refresh > 0.05 && POSE.refresh < 1); });
console.log(`${n} focus tests passed`);
