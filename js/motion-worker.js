// Background thread for export-time motion fields (smooth slow motion, Extra clean): the main thread keeps
// decoding, drawing and encoding while this works out how every part of the picture moves between two frames.
import { buildPyramid, denseFlow, globalShift, frameSharpness } from './core.js';

// Small greyscale copy of a frame: the GPU hands over RGBA rows, the red channel is the luma (see FS_ANALYSIS).
function gray(rgba, n) {
  const g = new Float32Array(n), px = new Uint8Array(rgba);
  for (let i = 0, j = 0; j < n; i += 4, j++) g[j] = px[i];
  return g;
}

self.onmessage = e => {
  const m = e.data;
  if (m.type !== 'flow') return;
  try {
    const { w, h } = m, n = w * h;
    const pa = buildPyramid(gray(m.a, n), w, h, 4), pb = buildPyramid(gray(m.b, n), w, h, 4);
    const F = denseFlow(null, null, w, h, { pyrA: pa, pyrB: pb, guess: globalShift(pa, pb) });
    const out = {
      type: 'flow', id: m.id, gw: F.gw, gh: F.gh, fx: F.fx, fy: F.fy, bx: F.bx, by: F.by, validFrac: F.validFrac,
      sharpA: frameSharpness(pa[0].d, w, h), sharpB: frameSharpness(pb[0].d, w, h),
    };
    self.postMessage(out, [F.fx.buffer, F.fy.buffer, F.bx.buffer, F.by.buffer]);
  } catch (err) {
    self.postMessage({ type: 'flow', id: m.id, error: String(err && err.message || err) });
  }
};
self.postMessage({ type: 'ready' });
