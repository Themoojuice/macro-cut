// Background thread for motion tracking: the main thread keeps decoding while this tracks.
import { MotionTracker, frameSharpness } from './core.js';
let tracker = null, w = 0, h = 0, g = null;
self.onmessage = e => {
  const m = e.data;
  if (m.type === 'init') {
    w = m.w; h = m.h; g = new Float32Array(w * h);
    tracker = new MotionTracker(w, h, { roi: m.roi });
    self.postMessage({ type: 'ready' });
  } else if (m.type === 'frame') {
    const px = new Uint8Array(m.buf);
    for (let i = 0; i < g.length; i++) g[i] = px[i];
    const r = tracker.push(g);
    self.postMessage({ type: 'result', i: m.i, C: r.C, ok: r.ok, inliers: r.inliers, sharp: frameSharpness(g, w, h), buf: m.buf }, [m.buf]);
  }
};
