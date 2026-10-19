// Analysis (motion tracking) and export (frame-accurate WebCodecs render → MP4).
/* global Mp4Muxer */
import { decodeRange, WindowReader, hdrMode } from './media.js';
import { Downscaler, Renderer } from './gpu.js';
import { MotionTracker, frameSharpness, frameAt, fillGaps, denseFlow, buildPyramid, globalShift, chainField, grainSigma } from './core.js';

export function frameInfo(clip, f) {
  const odd = (clip.rotation / 90) & 1;
  return {
    texW: f.displayWidth, texH: f.displayHeight,
    dispW: odd ? f.displayHeight : f.displayWidth, dispH: odd ? f.displayWidth : f.displayHeight,
    rot: clip.rotation / 90,
    hdr: hdrMode(clip, f.colorSpace),
  };
}

const tick = () => new Promise(r => setTimeout(r, 0));

export function analysisSize(clip) {
  const A = clip.dispW / clip.dispH, short = 360;
  return A >= 1 ? [Math.round(short * A / 2) * 2, short] : [short, Math.round(short / A / 2) * 2];
}

// Tracks motion on a background thread (falls back to this thread if workers are unavailable).
function makeTrackerChannel(w, h, roi) {
  return new Promise(resolve => {
    let worker = null;
    try { worker = new Worker(new URL('./track-worker.js', import.meta.url), { type: 'module' }); } catch (e) { worker = null; }
    if (!worker) return resolve(makeInline());
    const handlers = { result: null };
    const free = [];
    let ready = false;
    const t = setTimeout(() => { if (!ready) { worker.terminate(); worker = null; resolve(makeInline()); } }, 4000);
    function makeInline() {
      const tr = new MotionTracker(w, h, { roi }), g = new Float32Array(w * h);
      return { inline: true, push(i, px, onResult) { for (let k = 0; k < g.length; k++) g[k] = px[k]; const r = tr.push(g); onResult({ i, C: r.C, ok: r.ok, inliers: r.inliers, sharp: frameSharpness(g, w, h) }); }, close() {} };
    }
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'ready') { ready = true; clearTimeout(t); resolve(chan); }
      else if (m.type === 'result') { free.push(m.buf); handlers.result && handlers.result(m); }
    };
    worker.onerror = e => {
      if (!ready) { clearTimeout(t); resolve(makeInline()); return; }
      chan.failed = (e && e.message) || 'motion tracker crashed';
      handlers.result && handlers.result(null);
    };
    const chan = {
      inline: false,
      push(i, px, onResult) {
        handlers.result = onResult;
        const buf = free.pop() || new ArrayBuffer(w * h);
        new Uint8Array(buf).set(px);
        worker.postMessage({ type: 'frame', i, buf }, [buf]);
      },
      close() { worker.terminate(); },
    };
    worker.postMessage({ type: 'init', w, h, roi });
  });
}

// Returns { C: Array<[a,b,tx,ty]>, ok: Uint8Array, sharp: Float32Array, inliers: Uint16Array }
export async function analyseClip(clip, { roi = null, onProgress = () => {}, signal } = {}) {
  const [w, h] = analysisSize(clip);
  const ds = new Downscaler(w, h);
  const chan = await makeTrackerChannel(w, h, roi);
  const N = clip.times.length;
  const idx = new Map();
  clip.times.forEach((t, i) => idx.set(Math.round(t * 1e6), i));
  const C = new Array(N).fill(null), ok = new Uint8Array(N), sharp = new Float32Array(N), inl = new Uint16Array(N);
  const px = new Uint8Array(w * h);
  let sent = 0, done = 0, last = performance.now(), wake = null;
  const onResult = r => {
    if (r) { C[r.i] = r.C; ok[r.i] = r.ok ? 1 : 0; inl[r.i] = r.inliers; sharp[r.i] = r.sharp; done++; }
    if (wake) { const f = wake; wake = null; f(); }
  };
  const waitFor = cond => new Promise((res, rej) => {
    const chk = () => (chan.failed ? rej(new Error(chan.failed)) : cond() ? res() : (wake = chk)); chk();
  });
  try {
    await decodeRange(clip, {
      signal,
      onFrame: async f => {
        const i = idx.get(f.timestamp);
        ds.grabBytes(f, frameInfo(clip, f), px);
        f.close();
        if (i === undefined) return;
        chan.push(i, px, onResult); sent++;
        if (sent - done > 6) await waitFor(() => sent - done <= 3);   // keep the worker fed, not flooded
        if (performance.now() - last > 120) { last = performance.now(); onProgress(done / N); if (chan.inline) await tick(); }
      },
    });
    await waitFor(() => done >= sent);
  } finally { ds.dispose(); chan.close(); }
  if (signal && signal.aborted) return null;
  // any frame the decoder never produced: hold previous camera
  for (let i = 0; i < N; i++) if (!C[i]) { C[i] = i ? C[i - 1].slice() : [1, 0, 0, 0]; ok[i] = 0; }
  // camera over blurred / lost frames is interpolated between the good frames either side
  return { C: fillGaps(C, ok, clip.times), ok, sharp, inliers: inl, w, h };
}

// Output timeline: which source frame feeds each output tick.
// ticks: [frameIndex, tsUs, u]; u > 0 means "u of the way from frameIndex to the next frame" (in-between frame).
// opts.interp: create in-between frames for parts slowed below the source frame rate.
export function buildTimeline(clip, segments, speed, fps, opts = {}) {
  const ticks = [], T = clip.times;
  let outT = 0, j = 0, interpCount = 0;
  for (const seg of segments) {
    const sp = seg.speed != null ? seg.speed : speed;
    const len = (seg.end - seg.start) / sp;
    let first = frameAt(T, seg.start); if (T[first] < seg.start - 1e-4 && first < T.length - 1) first++;
    let lastF = frameAt(T, seg.end); if (T[lastF] > seg.end + 1e-4 && lastF > 0) lastF--;
    const interp = opts.interp && sp < 0.999;
    while (j / fps < outT + len - 1e-6) {
      const src = seg.start + (j / fps - outT) * sp;
      let fi, u = 0;
      if (interp) {
        fi = frameAt(T, src); if (T[fi] > src && fi > 0) fi--;
        fi = Math.min(Math.max(fi, first), lastF);
        if (fi < lastF) {
          u = (src - T[fi]) / Math.max(1e-6, T[fi + 1] - T[fi]);
          if (u < 0.03) u = 0; else if (u > 0.97) { fi++; u = 0; }
          u = Math.min(Math.max(u, 0), 1);
        }
      } else {
        fi = Math.min(Math.max(frameAt(T, src), first), lastF);
      }
      if (u > 0) interpCount++;
      ticks.push([fi, Math.round(j * 1e6 / fps), u]);
      j++;
    }
    outT += len;
  }
  return { ticks, duration: outT, interpCount };
}

const AVC = ['avc1.640034', 'avc1.640033', 'avc1.640032', 'avc1.64002A', 'avc1.640028', 'avc1.4D0028', 'avc1.42E028'];
async function pickEncoder(w, h, fps, bitrate) {
  for (const codec of AVC) {
    const cfg = { codec, width: w, height: h, bitrate, framerate: fps, latencyMode: 'quality', avc: { format: 'avc' } };
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return { cfg, mux: 'avc' }; } catch (e) { /* next */ }
  }
  for (const codec of ['vp09.00.51.08', 'vp09.00.41.08', 'vp09.00.10.08']) {
    const cfg = { codec, width: w, height: h, bitrate, framerate: fps, latencyMode: 'quality' };
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) return { cfg, mux: 'vp9' }; } catch (e) { /* next */ }
  }
  throw new Error(`No video encoder available for ${w}×${h}. Try a lower export resolution.`);
}

/**
 * plan: { segments, speed, fps, outW, outH, M (Float64Array N*4), look, audio (bool) }
 * returns { blob, codec, frames, hasAudio }
 */
export async function exportClip(clip, plan, { onProgress = () => {}, signal } = {}) {
  const { outW, outH, fps } = plan;
  const { ticks, duration, interpCount } = buildTimeline(clip, plan.segments, plan.speed, fps, { interp: plan.interp });
  if (!ticks.length) throw new Error('Nothing to export — all segments are empty.');
  const byFrame = new Map();   // frame index → { whole: [ts], between: [[ts, u]] }
  for (const [fi, ts, u] of ticks) {
    if (!byFrame.has(fi)) byFrame.set(fi, { whole: [], between: [] });
    if (u > 0) byFrame.get(fi).between.push([ts, u]); else byFrame.get(fi).whole.push(ts);
  }
  const bitrate = Math.min(80e6, Math.max(6e6, Math.round(outW * outH * fps * 0.22)));
  const enc = await pickEncoder(outW, outH, fps, bitrate);

  const A = clip.audio;
  const allNormal = plan.segments.every(sg => Math.abs((sg.speed != null ? sg.speed : plan.speed) - 1) < 1e-6);
  const withAudio = !!(plan.audio && A && A.description && allNormal);
  const lookFor = typeof plan.look === 'function' ? plan.look : () => plan.look;
  const target = new Mp4Muxer.ArrayBufferTarget();
  const muxer = new Mp4Muxer.Muxer({
    target,
    video: { codec: enc.mux, width: outW, height: outH, frameRate: fps },
    audio: withAudio ? { codec: 'aac', numberOfChannels: A.channels, sampleRate: A.sampleRate } : undefined,
    fastStart: 'in-memory',
    firstTimestampBehavior: 'cross-track-offset',
  });
  let encErr = null, wake = null;
  const poke = () => { if (wake) { const w = wake; wake = null; w(); } };
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: e => { encErr = e; poke(); },
  });
  encoder.addEventListener('dequeue', poke);
  encoder.configure(enc.cfg);

  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(outW, outH) : Object.assign(document.createElement('canvas'), { width: outW, height: outH });
  const R = new Renderer(canvas);
  R.setSize(outW, outH);

  const fromUs = Math.round(plan.segments[0].start * 1e6) - 1, toUs = Math.round(plan.segments[plan.segments.length - 1].end * 1e6) + 1;
  const tsIndex = new Map();
  clip.times.forEach((t, i) => tsIndex.set(Math.round(t * 1e6), i));
  let emitted = 0, last = performance.now();
  const keyEvery = Math.max(1, Math.round(fps * 2));
  const emit = async ts => {
    const waitStart = performance.now();
    while (encoder.encodeQueueSize > 3 && !encErr) {
      if (performance.now() - waitStart > 20000) throw new Error('The video encoder stopped responding');
      await new Promise(r => { wake = r; setTimeout(r, 50); });
    }
    if (encErr) throw encErr;
    const vf = new VideoFrame(canvas, { timestamp: ts, duration: Math.round(1e6 / fps) });
    encoder.encode(vf, { keyFrame: emitted % keyEvery === 0 });
    vf.close();
    emitted++;
  };
  const Mi = i => plan.M.subarray(4 * i, 4 * i + 4);
  const lerpM = (a, b, u) => [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])];
  // In-between frames need a small greyscale copy of each frame to measure motion.
  const [aw, ah] = analysisSize(clip);
  const ds = (interpCount || plan.denoise) ? new Downscaler(aw, ah) : null;
  const gbytes = ds ? new Uint8Array(aw * ah) : null;
  let prev = null;   // { i, pyr } for the frame currently held in R.src
  // ── Extra clean: each frame blended with its ±2 neighbours along measured motion ──
  async function exportDenoised() {
    const D = 2;
    // Blend threshold follows the clip's real grain: only differences grain could explain get averaged.
    let H = 0.03, grainChecks = 0;
    const gcv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(192, 192) : Object.assign(document.createElement('canvas'), { width: 192, height: 192 });
    const gctx = gcv.getContext('2d', { willReadFrequently: true });
    const measureGrain = f => {
      try {
        const W = f.displayWidth, Hh = f.displayHeight, sz = 192, est = [];
        for (const [fx, fy] of [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]]) {
          const x = Math.round(fx * (W - sz)), y = Math.round(fy * (Hh - sz));
          gctx.drawImage(f, x, y, sz, sz, 0, 0, sz, sz);
          est.push(grainSigma(gctx.getImageData(0, 0, sz, sz).data, sz, sz));
        }
        est.sort((a, b) => a - b);
        const sigma = est[1];                                   // quietest areas = grain, not texture
        const sd = sigma * Math.sqrt(2 / 5);                    // spread of a 5-tap patch difference from grain alone
        const h = Math.min(0.06, Math.max(0.004, 3.5 * sd));
        H = grainChecks ? 0.7 * H + 0.3 * h : h; grainChecks++;
      } catch (e) { /* keep previous */ }
    };
    const needed = new Set();
    for (const [fi, , u] of ticks) { needed.add(fi); if (u > 0) needed.add(fi + 1); }
    const near = k => { for (let d = -D; d <= D; d++) if (needed.has(k + d)) return true; return false; };
    const raw = new Map(), fw = new Map(), bw = new Map(), pool = [], flowPool = [];
    const targets = [];
    let flip = 0, last = -2, prevC = null, lastProg = performance.now();
    const norm = F => { const d = new Float32Array(F.gw * F.gh * 2); for (let k = 0; k < F.gw * F.gh; k++) { d[2 * k] = F.x[k] / ah; d[2 * k + 1] = F.y[k] / ah; } return d; };
    const flowTex = (F) => { const t = flowPool.pop() || R.makeTex(); R.setFlowTex(t, F.gw, F.gh, norm(F)); return t; };
    const done = new Set();
    const holdPrev = async () => {
      const pb = prevC && byFrame.get(prevC.c);
      if (pb && pb.between.length && !prevC.betweenDone) {
        R.useSrc(prevC.tex, prevC.info, prevC.mip);
        for (const [ts] of pb.between) { R.render(Mi(prevC.c), lookFor(clip.times[prevC.c]), { pxScale: 1 }); await emit(ts); }
        prevC.betweenDone = true;
      }
    };
    exportDenoised.grain = () => H;
    const ready = async (c, tex, C) => {
      const pb = prevC && byFrame.get(prevC.c);
      if (pb && pb.between.length && prevC.c === c - 1 && fw.has(c - 1)) {
        R.useSrc(prevC.tex, prevC.info, prevC.mip); R.useSrcB(tex);
        R.setFlow(fw.get(c - 1).gw, fw.get(c - 1).gh, norm(fw.get(c - 1)));
        const mA = Mi(c - 1), mB = Mi(c), tA = clip.times[c - 1], tB = clip.times[c];
        for (const [ts, u] of pb.between) { R.render(lerpM(mA, mB, u), lookFor(tA + u * (tB - tA)), { pxScale: 1, interp: u }); await emit(ts); }
        prevC.betweenDone = true;
      } else await holdPrev();
      R.useSrc(tex, C.info, C.mip);
      const outs = byFrame.get(c);
      if (outs) for (const ts of outs.whole) { R.render(Mi(c), lookFor(clip.times[c]), { pxScale: 1 }); await emit(ts); }
      prevC = { c, tex, info: C.info, mip: C.mip };
    };
    const finalize = async c => {
      if (done.has(c) || !raw.has(c) || !needed.has(c)) return;
      done.add(c);
      const C = raw.get(c), neigh = [], used = [];
      // a blurrier neighbour (shake blur, focus drift) would soften the result: weight it down
      const add = (k, F) => {
        if (!raw.has(k) || !F) return;
        const ft = flowTex(F); used.push(ft);
        const r = raw.get(k).sharp / Math.max(1e-6, C.sharp);
        neigh.push({ tex: raw.get(k).tex, flowTex: ft, weight: Math.min(1, r * r) });
      };
      add(c + 1, fw.get(c));
      add(c - 1, bw.get(c));
      if (fw.get(c) && fw.get(c + 1)) add(c + 2, chainField(fw.get(c), fw.get(c + 1)));
      if (bw.get(c) && bw.get(c - 1)) add(c - 2, chainField(bw.get(c), bw.get(c - 1)));
      if (!targets.length) for (let k = 0; k < 2; k++) targets.push(R.makeTarget(C.info.texW, C.info.texH));
      const tgt = targets[flip]; flip ^= 1;
      R.denoise(C.tex, neigh, C.info, tgt, H, C.mip);
      flowPool.push(...used);
      await ready(c, tgt.t, C);
    };
    const flushAll = async () => { for (const c of [...raw.keys()].sort((a, b) => a - b)) await finalize(c); };
    const evict = keep => {
      for (const k of [...raw.keys()]) if (k < keep) { pool.push(raw.get(k).tex); raw.delete(k); fw.delete(k); bw.delete(k); }
    };
    await decodeRange(clip, {
      fromUs, toUs, signal,
      onFrame: async f => {
        const k = tsIndex.get(f.timestamp);
        if (k === undefined || !near(k)) { f.close(); return; }
        if (k !== last + 1) { await flushAll(); await holdPrev(); evict(Infinity); prevC = null; }
        last = k;
        const info = frameInfo(clip, f), m = Mi(k);
        const mip = Math.hypot(m[0], m[1]) * info.dispH / outH > 1.25;
        const tex = pool.pop() || R.makeTex();
        if (grainChecks === 0 || (k % 30 === 0 && grainChecks < 20)) measureGrain(f);
        R.uploadTex(tex, f);
        ds.grabBytes(f, info, gbytes);
        f.close();
        const pyr = buildPyramid(Float32Array.from(gbytes), aw, ah, 4);
        raw.set(k, { tex, pyr, info, mip, sharp: frameSharpness(pyr[0].d, aw, ah) });
        if (raw.has(k - 1)) {
          const F = denseFlow(null, null, aw, ah, { pyrA: raw.get(k - 1).pyr, pyrB: pyr, guess: globalShift(raw.get(k - 1).pyr, pyr) });
          fw.set(k - 1, { gw: F.gw, gh: F.gh, w: aw, h: ah, x: F.fx, y: F.fy });
          bw.set(k, { gw: F.gw, gh: F.gh, w: aw, h: ah, x: F.bx, y: F.by });
        }
        await finalize(k - D);
        evict(k - 2 * D);
        if (performance.now() - lastProg > 150) { lastProg = performance.now(); onProgress(emitted / ticks.length); await tick(); }
      },
    });
    if (!(signal && signal.aborted)) { await flushAll(); await holdPrev(); }
    console.info(`MacroCut extra clean: blend threshold ${H.toFixed(4)}`);
  }

  const renderBetween = async (iA, pyrA, pyrB, list) => {
    const F = denseFlow(null, null, aw, ah, { pyrA, pyrB, guess: globalShift(pyrA, pyrB) });
    const data = new Float32Array(F.gw * F.gh * 2);
    for (let k = 0; k < F.gw * F.gh; k++) { data[2 * k] = F.fx[k] / ah; data[2 * k + 1] = F.fy[k] / ah; }
    R.setFlow(F.gw, F.gh, data);
    const mA = Mi(iA), mB = Mi(iA + 1), tA = clip.times[iA], tB = clip.times[iA + 1];
    for (const [ts, u] of list) {
      R.render(lerpM(mA, mB, u), lookFor(tA + u * (tB - tA)), { pxScale: 1, interp: u });
      await emit(ts);
    }
  };
  try {
  if (plan.denoise) await exportDenoised();
  else await decodeRange(clip, {
    fromUs, toUs, signal,
    onFrame: async f => {
      const i = tsIndex.get(f.timestamp);
      const outs = i === undefined ? null : byFrame.get(i);
      const prevNeeds = prev && byFrame.get(prev.i) && byFrame.get(prev.i).between.length && prev.i === i - 1;
      if (!outs && !prevNeeds) {
        f.close();
        if (prev && prev.i !== i) {
          const pb = byFrame.get(prev.i);   // previous frame owed in-between frames but its neighbour is missing: hold it
          if (pb && pb.between.length) for (const [ts] of pb.between) { R.render(Mi(prev.i), lookFor(clip.times[prev.i]), { pxScale: 1 }); await emit(ts); }
          prev = null;
        }
        return;
      }
      const info = frameInfo(clip, f);
      const m = Mi(i);
      const scale = Math.hypot(m[0], m[1]) * info.dispH / outH;
      let pyr = null;
      if (ds && (prevNeeds || (outs && outs.between.length))) {
        ds.grabBytes(f, info, gbytes);
        pyr = buildPyramid(Float32Array.from(gbytes), aw, ah, 4);
      }
      R.uploadB(f, info, scale > 1.25);
      f.close();
      if (prevNeeds) await renderBetween(prev.i, prev.pyr, pyr, byFrame.get(prev.i).between);
      else if (prev && byFrame.get(prev.i) && byFrame.get(prev.i).between.length) {
        // no next frame available: hold the previous frame instead
        for (const [ts] of byFrame.get(prev.i).between) { R.render(Mi(prev.i), lookFor(clip.times[prev.i]), { pxScale: 1 }); await emit(ts); }
      }
      R.swapFrames();
      if (outs) for (const ts of outs.whole) { R.render(m, lookFor(clip.times[i]), { pxScale: 1 }); await emit(ts); }
      prev = { i, pyr };
      if (performance.now() - last > 150) { last = performance.now(); onProgress(emitted / ticks.length); await tick(); }
    },
  });
  // last held frame still owes in-between frames (end of range): hold it
  if (!plan.denoise && prev && byFrame.get(prev.i) && byFrame.get(prev.i).between.length && !(signal && signal.aborted)) {
    for (const [ts] of byFrame.get(prev.i).between) { R.render(Mi(prev.i), lookFor(clip.times[prev.i]), { pxScale: 1 }); await emit(ts); }
  }
  } catch (e) { if (encoder.state !== 'closed') encoder.close(); R.dispose(); if (ds) ds.dispose(); throw e; }
  R.dispose(); if (ds) ds.dispose();
  if (signal && signal.aborted) { encoder.close(); return null; }
  await encoder.flush();
  if (encErr) throw encErr;
  encoder.close();

  if (withAudio) {
    let outT = 0, first = true;
    const areader = new WindowReader(clip.file, 4 << 20);
    const meta = { decoderConfig: { codec: A.codec, sampleRate: A.sampleRate, numberOfChannels: A.channels, description: A.description } };
    for (const seg of plan.segments) {
      const s0 = seg.start * 1e6, s1 = seg.end * 1e6;
      let lastTs = -1;
      for (const s of A.samples) {
        if (s.ts < s0 || s.ts + s.dur > s1) continue;
        const ts = Math.round(outT * 1e6 + (s.ts - s0));
        if (ts <= lastTs) continue;
        muxer.addAudioChunkRaw((await areader.read(s.offset, s.size)).slice(), 'key', ts, s.dur, first ? meta : undefined);
        first = false; lastTs = ts;
      }
      outT += seg.end - seg.start;
    }
  }
  muxer.finalize();
  onProgress(1);
  return { blob: new Blob([target.buffer], { type: 'video/mp4' }), codec: enc.cfg.codec, frames: emitted, hasAudio: withAudio, duration, interpolated: interpCount, cleaned: !!plan.denoise };
}


// Grain level of a decoded frame (sampled at full resolution in 7 patches; second-quietest = grain, not texture).
let _gcv = null;
export function grainOfFrame(f) {
  const sz = 192;
  if (!_gcv) _gcv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(sz, sz) : Object.assign(document.createElement('canvas'), { width: sz, height: sz });
  const ctx = _gcv.getContext('2d', { willReadFrequently: true }), W = f.displayWidth, H = f.displayHeight, est = [];
  for (const [fx, fy] of [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]]) {
    ctx.drawImage(f, Math.round(fx * (W - sz)), Math.round(fy * (H - sz)), sz, sz, 0, 0, sz, sz);
    est.push(grainSigma(ctx.getImageData(0, 0, sz, sz).data, sz, sz));
  }
  est.sort((a, b) => a - b);
  return est[1];
}

/**
 * Stacked still: frames within ±radius of frame c are lined up on it and fused, keeping the sharpest version
 * of each region. Returns { blob, before, frames }. opts: { M, look, outW, outH, radius, maxFrames, onProgress, signal }
 */
export async function stackStill(clip, c, opts) {
  const { M, look, outW, outH, radius = 0.3, maxFrames = 20, onProgress = () => {}, signal } = opts;
  const T = clip.times, near = [];
  for (let i = 0; i < T.length; i++) if (Math.abs(T[i] - T[c]) <= radius + 1e-6) near.push(i);
  const step = Math.max(1, Math.ceil(near.length / maxFrames));
  const keep = new Set(near.filter(i => (i - c) % step === 0));
  const lo = Math.min(...keep), hi = Math.max(...keep);
  const tsIndex = new Map(); T.forEach((t, i) => tsIndex.set(Math.round(t * 1e6), i));
  const [aw, ah] = analysisSize(clip);
  const ds = new Downscaler(aw, ah), gb = new Uint8Array(aw * ah);
  const pyr = new Map(); let cGray = null, sigma = 0.004, sharpC = 0;
  const range = { fromUs: Math.round(T[lo] * 1e6) - 1, toUs: Math.round(T[hi] * 1e6) + 1, signal };
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(outW, outH) : Object.assign(document.createElement('canvas'), { width: outW, height: outH });
  const R = new Renderer(canvas); R.setSize(outW, outH);
  const centreTex = R.makeTex();
  // pass 1: small greyscale copies → motion of every frame relative to the centre; full-size centre kept on the GPU
  try {
    await decodeRange(clip, { ...range, onFrame: f => {
      const i = tsIndex.get(f.timestamp);
      if (keep.has(i)) {
        ds.grabBytes(f, frameInfo(clip, f), gb);
        pyr.set(i, buildPyramid(Float32Array.from(gb), aw, ah, 4));
        if (i === c) { cGray = gb.slice(); R.uploadTex(centreTex, f); R.mipTex(centreTex); try { sigma = grainOfFrame(f); } catch (e) { /* keep default */ } }
      }
      f.close();
    } });
  } catch (e) { R.dispose(); throw e; } finally { ds.dispose(); }
  if (signal && signal.aborted) { R.dispose(); return null; }
  if (!pyr.has(c)) { R.dispose(); throw new Error('Couldn\'t read the chosen frame.'); }
  sharpC = frameSharpness(pyr.get(c)[0].d, aw, ah);
  const flows = new Map(), order = [...keep].sort((a, b) => Math.abs(a - c) - Math.abs(b - c));
  let n = 0;
  for (const i of order) {
    if (!pyr.has(i)) continue;
    if (i === c) { flows.set(i, null); continue; }
    const F = denseFlow(null, null, aw, ah, { pyrA: pyr.get(c), pyrB: pyr.get(i), guess: globalShift(pyr.get(c), pyr.get(i), 16) });
    if (F.validFrac >= 0.2) flows.set(i, F);
    onProgress(0.4 * (++n) / order.length);
    await tick();
  }
  // pass 2: line every frame up on the centre and fuse
  const toBlob = () => canvas.convertToBlob ? canvas.convertToBlob({ type: 'image/png' }) : new Promise(r => canvas.toBlob(r, 'image/png'));
  let before = null, used = 0, info = null;
  try {
    const frameTex = R.makeTex(), flowTex = R.makeTex(), grayTex = R.makeTex();
    R.setGrayTex(grayTex, aw, ah, cGray); R.stackCentre(centreTex);
    const zero = new Float32Array(4);
    // grain sets the floor below which "sharpness" is just noise; tolerance for "agrees with centre" at coarse scale
    // grain's own contribution to the band-pass sharpness measure (≈0.2σ² for white noise) is subtracted
    // 8-bit footage can't have less than ~half a code value of noise (a flat patch can measure 0)
    sigma = Math.max(sigma, 0.002);
    const noise = 0.2 * sigma * sigma * (opts.noiseScale ?? 1), eps = Math.max((opts.epsScale ?? 4) * noise, opts.epsMin ?? 1e-4), tol = opts.tol ?? 0.06, pw = opts.pow ?? 8;
    let started = false;
    // two passes over the frames: (0) the sharpest agreeing view of every spot, (1) blend weighted relative to it
    for (const mode of [0, 1]) {
      if (mode === 1) R.stackSecondPass();
      used = 0;
      await decodeRange(clip, { ...range, onFrame: async f => {
        const i = tsIndex.get(f.timestamp);
        if (!flows.has(i)) { f.close(); return; }
        info = frameInfo(clip, f);
        if (!started) { R.stackBegin(info.texW, info.texH); started = true; }
        R.uploadTex(frameTex, f); f.close(); R.mipTex(frameTex);
        const F = flows.get(i);
        if (F) {
          const d = new Float32Array(F.gw * F.gh * 2);
          for (let k = 0; k < F.gw * F.gh; k++) { d[2 * k] = F.fx[k] / ah; d[2 * k + 1] = F.fy[k] / ah; }
          R.setFlowTex(flowTex, F.gw, F.gh, d);
        } else R.setFlowTex(flowTex, 2, 1, zero);
        const lod = Math.max(0, Math.log2(info.dispH / ah));
        // the centre frame gets a head start: where views are equally sharp, the unwarped original is the safest
        R.stackAdd(frameTex, flowTex, grayTex, info, { eps, tol, lod, noise, pow: pw, mode, bias: i === c ? 2 : 1 });
        used++;
        if (mode === 1 && i === c) {   // the plain centre frame, for before/after
          R.useSrc(frameTex, info, Math.hypot(M[0], M[1]) * info.dispH / outH > 1.25);
          R.render(M, look, { pxScale: 1 }); before = await toBlob();
        }
        onProgress(0.4 + 0.3 * mode + 0.3 * used / flows.size);
        await tick();
      } });
      if (signal && signal.aborted) return null;
    }
    if (signal && signal.aborted) return null;
    const mip = Math.hypot(M[0], M[1]) * info.dispH / outH > 1.25;
    const out = R.makeTarget(info.texW, info.texH);
    R.stackFinish(out, mip);
    R.useSrc(out.t, info, mip);
    R.render(M, look, { pxScale: 1 });
    const blob = await toBlob();
    onProgress(1);
    return { blob, before, frames: used, sharpC, sigma };
  } finally { R.dispose(); }
}
