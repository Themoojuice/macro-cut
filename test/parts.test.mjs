import assert from 'node:assert';
import { lookAt, speedAt, segmentAt } from '../js/core.js';
import { buildTimeline } from '../js/engine.js';
const base = { exposure: 0, saturation: 0.1 }, dark = { exposure: -1, saturation: 0.1 };
const segs = [{ start: 0, end: 3 }, { start: 3, end: 4, look: dark, speed: 0.5 }, { start: 5, end: 6 }];
assert.equal(segmentAt(segs, 3.5), 1); assert.equal(segmentAt(segs, 4.5), -1);
assert.equal(lookAt(segs, base, 1).exposure, 0);
assert.equal(lookAt(segs, base, 3.5).exposure, -1);
assert.equal(lookAt(segs, base, 3).exposure, -0.5, 'midpoint of blend at split');
assert(Math.abs(lookAt(segs, base, 2.9).exposure + 0.1) < 1e-9, 'blend starts 0.125 s before split');
assert.equal(lookAt(segs, base, 2.8).exposure, 0, 'no blend before window');
assert.equal(lookAt(segs, base, 5.01).exposure, 0, 'no blend across a real cut (4→5 gap)');
assert.equal(lookAt(segs, base, 3.99).exposure, -1, 'no blend into a cut');
assert.equal(speedAt(segs, 1, 3.5), 0.5); assert.equal(speedAt(segs, 1, 1), 1);
const clip = { times: Float64Array.from({ length: 360 }, (_, i) => i / 60) };   // 6 s at 60 fps
const { ticks, duration } = buildTimeline(clip, segs, 1, 30);
assert(Math.abs(duration - (3 + 2 + 1)) < 1e-9, 'duration ' + duration);      // 1 s part at ½× → 2 s
assert.equal(ticks.length, 180);
const slow = ticks.slice(90, 150).map(t => t[0]), uniq = new Set(slow).size;
console.log(`timeline ok: ${ticks.length} frames, ${duration} s; slow part uses ${uniq} unique source frames for 60 output frames (60 fps source at ½× → all unique)`);
assert.equal(uniq, 60);
console.log('per-part look/speed: all assertions passed');
