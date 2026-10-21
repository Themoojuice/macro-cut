import assert from 'node:assert/strict';
import { PRESETS, NEUTRAL, KEYS, LIMITS, resolveLook, normLook, defaultLook, applyAdj, adjFor, oneSided, baseOf } from '../js/look.js';

const near = (a, b) => Math.abs(a - b) < 1e-9;
// all sliders at 0 = preset exactly
for (const p of ['natural', 'pop']) assert.deepEqual(resolveLook({ preset: p, off: false, adj: {} }), PRESETS[p]);
assert.deepEqual(resolveLook({ preset: null, off: false, adj: {} }), { ...NEUTRAL });
// Off ignores preset and tweaks, and is lossless
const L = { preset: 'pop', off: true, adj: { exposure: 0.5 } };
assert.deepEqual(resolveLook(L), { ...NEUTRAL });
L.off = false; assert(near(resolveLook(L).exposure, 1));
// ends reach the limits, both directions, from any preset
for (const k of KEYS) for (const B of [PRESETS.natural, PRESETS.pop, NEUTRAL]) {
  assert(near(applyAdj(k, B[k], 1), LIMITS[k][1]), k);
  assert(near(applyAdj(k, B[k], -1), LIMITS[k][0]), k);
  for (const a of [-0.7, -0.2, 0.3, 0.9]) if (!(a < 0 && oneSided(k, B))) assert(near(adjFor(k, B[k], applyAdj(k, B[k], a)), a), `${k} ${a}`);
}
// monotonic: more slider = more effect
for (const k of KEYS) { let prev = -Infinity; for (let a = -1; a <= 1.0001; a += 0.05) { const v = applyAdj(k, PRESETS.natural[k], a); assert(v >= prev - 1e-12); prev = v; } }
// one-sided only where the preset has none of the effect
assert(oneSided('cas', NEUTRAL) && !oneSided('cas', PRESETS.natural) && !oneSided('exposure', NEUTRAL));
// old absolute looks migrate
assert.deepEqual(normLook({ ...PRESETS.pop }), { preset: 'pop', off: false, adj: {} });
assert.deepEqual(normLook({ ...NEUTRAL }), { preset: null, off: false, adj: {} });
const old = { ...PRESETS.natural, exposure: 1 }, m = normLook(old);
assert.equal(m.preset, 'natural'); assert(near(resolveLook(m).exposure, 1));
for (const k of KEYS) assert(near(resolveLook(m)[k], old[k]), k);
assert.deepEqual(normLook(null), defaultLook());
assert.deepEqual(normLook({ preset: 'bogus', adj: { cas: 5, nope: 1 } }), { preset: null, off: false, adj: { cas: 1 } });
assert.equal(baseOf({ preset: 'pop' }), PRESETS.pop);
console.log('look model: all assertions passed');
