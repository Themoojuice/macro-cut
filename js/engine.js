// Analysis (motion tracking) and export (frame-accurate WebCodecs render → MP4).
/* global Mp4Muxer */
import { decodeRange, WindowReader, hdrMode, activeNow } from './media.js';
import { Downscaler, Renderer, yieldNow } from './gpu.js';
import { MotionTracker, frameSharpness, frameAt, fillGaps, denseFlow, buildPyramid, globalShift, chainField, grainSigma } from './core.js';
import { FocusBuild } from './focus.js';

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
  const px = new Uint8Array(w * h), rgba = new Uint8Array(w * h * 4);
  // small copies are read back without stalling: a few frames in flight between the GPU and the tracker
  const FLY = 3, ring = [], flying = [];
  for (let k = 0; k < FLY; k++) ring.push(ds.makePackBuffer(w * h * 4));
  let sent = 0, done = 0, last = performance.now(), wake = null, slot = 0;
  const onResult = r => {
    if (r) { C[r.i] = r.C; ok[r.i] = r.ok ? 1 : 0; inl[r.i] = r.inliers; sharp[r.i] = r.sharp; done++; }
    if (wake) { const f = wake; wake = null; f(); }
  };
  const waitFor = cond => new Promise((res, rej) => {
    const chk = () => (chan.failed ? rej(new Error(chan.failed)) : cond() ? res() : (wake = chk)); chk();
  });
  const sendOldest = async () => {
    const x = flying.shift();
    await ds.waitFence(x.fence);
    ds.collect(x.pb, rgba);
    for (let a = 0, b = 0; b < px.length; a += 4, b++) px[b] = rgba[a];
    chan.push(x.i, px, onResult); sent++;
    if (sent - done > 6) await waitFor(() => sent - done <= 3);   // keep the worker fed, not flooded
  };
  try {
    await decodeRange(clip, {
      signal,
      onFrame: async f => {
        const i = idx.get(f.timestamp);
        if (i === undefined) { f.close(); return; }
        if (flying.length >= FLY) await sendOldest();
        const pb = ring[slot]; slot = (slot + 1) % FLY;
        flying.push({ i, pb, fence: ds.grabAsync(f, frameInfo(clip, f), pb) });
        f.close();
        while (flying.length && ds.fenceDone(flying[0].fence)) await sendOldest();
        if (performance.now() - last > 120) { last = performance.now(); onProgress(done / N); if (chan.inline) await tick(); }
      },
    });
    while (flying.length) await sendOldest();
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
const VP9 = ['vp09.00.51.08', 'vp09.00.41.08', 'vp09.00.10.08'];
// The phone's video chip first: a software encoder is many times slower at 4K. Only if no H.264/VP9 setting works in
// hardware do we let the browser fall back to software (still H.264 first).
async function pickEncoder(w, h, fps, bitrate) {
  const tryCfg = async (codec, hw) => {
    const cfg = { codec, width: w, height: h, bitrate, framerate: fps, latencyMode: 'quality' };
    if (codec.startsWith('avc')) cfg.avc = { format: 'avc' };
    if (hw) cfg.hardwareAcceleration = 'prefer-hardware';
    try { if ((await VideoEncoder.isConfigSupported(cfg)).supported) { delete cfg.hardwareAcceleration; return cfg; } } catch (e) { /* next */ }
    return null;
  };
  // (configured without the hint: the browser still picks the hardware it just confirmed, and can fall back to
  // software instead of failing if the chip is taken away mid-export)
  for (const hw of [true, false]) {
    for (const codec of AVC) { const cfg = await tryCfg(codec, hw); if (cfg) return { cfg, mux: 'avc', hw }; }
    for (const codec of VP9) { const cfg = await tryCfg(codec, hw); if (cfg) return { cfg, mux: 'vp9', hw }; }
  }
  throw new Error(`No video encoder available for ${w}×${h}. Try a lower export resolution.`);
}

// The MP4 as it is written. The first few hundred KB (file header + room reserved for the index, so the finished
// file starts playing straight away) stay in memory to be filled in at the end; everything after is handed to the
// browser as Blob parts, which it can keep on disk instead of in the page's memory. Nothing is copied at the end.
class MP4Sink {
  constructor() { this.head = new Uint8Array(0); this.headLen = -1; this.end = 0; this.parts = []; this.buf = []; this.bufBytes = 0; }
  write(data, pos) {
    const n = data.byteLength;
    if (this.headLen < 0) {
      if (pos + n > this.head.length) { const h = new Uint8Array(pos + n); h.set(this.head); this.head = h; }
      this.head.set(data, pos); this.end = Math.max(this.end, pos + n); return;
    }
    if (pos + n <= this.headLen) { this.head.set(data, pos); return; }
    if (pos !== this.end) throw new Error(`MP4 writer: unexpected write at ${pos} (file is ${this.end} bytes)`);
    this.buf.push(data); this.bufBytes += n; this.end += n;
    if (this.bufBytes >= 32 << 20) this.spill();
  }
  freezeHead() { this.headLen = this.end; }
  spill() { if (this.buf.length) { this.parts.push(new Blob(this.buf)); this.buf = []; this.bufBytes = 0; } }
  blob(type) { this.spill(); return new Blob([this.head, ...this.parts], { type }); }
}

// Motion fields for in-between frames and Extra clean, worked out on background threads (several at once) while
// this thread keeps decoding, drawing and encoding. Falls back to working them out here.
class MotionPool {
  constructor(w, h, n) {
    this.w = w; this.h = h; this.jobs = new Map(); this.id = 0; this.workers = []; this.failed = null;
    for (let i = 0; i < n; i++) {
      let wk = null;
      try { wk = new Worker(new URL('./motion-worker.js', import.meta.url), { type: 'module' }); } catch (e) { break; }
      const slot = { wk, busy: 0 };
      wk.onmessage = e => {
        const m = e.data; if (m.type !== 'flow') return;
        const job = this.jobs.get(m.id); if (!job) return;
        this.jobs.delete(m.id); slot.busy--;
        if (m.error) job.rej(new Error(m.error)); else job.res(m);
      };
      wk.onerror = e => { this.failed = (e && e.message) || 'motion worker crashed'; for (const j of this.jobs.values()) j.rej(new Error(this.failed)); this.jobs.clear(); };
      this.workers.push(slot);
    }
  }
  // a, b: RGBA rows (ArrayBuffer) of the two frames' small greyscale copies
  flow(a, b) {
    if (!this.workers.length || this.failed) return this.inline(a, b);
    const slot = this.workers.reduce((x, y) => (y.busy < x.busy ? y : x));
    const id = ++this.id;
    return new Promise((res, rej) => {
      this.jobs.set(id, { res, rej }); slot.busy++;
      slot.wk.postMessage({ type: 'flow', id, w: this.w, h: this.h, a, b });
    }).catch(e => { if (this.failed) return this.inline(a, b); throw e; });   // worker couldn't start or died: do it here
  }
  async inline(a, b) {
    await yieldNow();
    const { w, h } = this, g = buf => { const px = new Uint8Array(buf), out = new Float32Array(w * h); for (let i = 0, j = 0; j < out.length; i += 4, j++) out[j] = px[i]; return out; };
    const pa = buildPyramid(g(a), w, h, 4), pb = buildPyramid(g(b), w, h, 4);
    const F = denseFlow(null, null, w, h, { pyrA: pa, pyrB: pb, guess: globalShift(pa, pb) });
    return { gw: F.gw, gh: F.gh, fx: F.fx, fy: F.fy, bx: F.bx, by: F.by, validFrac: F.validFrac, sharpA: frameSharpness(pa[0].d, w, h), sharpB: frameSharpness(pb[0].d, w, h) };
  }
  close() { for (const s of this.workers) s.wk.terminate(); this.workers = []; }
}

/**
 * plan: { segments, speed, fps, outW, outH, M (Float64Array N*4), look, audio (bool),
 *         interp, denoise, focus ('build' | 'full' | null), C (camera per frame, needed for focus) }
 * returns { blob, codec, frames, hasAudio, ... }
 *
 * How a render flows (everything overlaps; nothing waits for anything it doesn't need):
 *   file → decoder (kept fed continuously) → frame uploaded to the GPU once → [small grey copy read back without
 *   waiting → motion worked out on background threads] → [Extra clean / focus build-up on the GPU] → stabilise,
 *   sharpen, grade on the GPU → converted to the encoder's own format on the GPU → read back without waiting
 *   (a few frames in flight) → hardware encoder → MP4 written as it goes.
 */
export async function exportClip(clip, plan, opts = {}) {
  try { return await exportOnce(clip, plan, opts, null); }
  catch (e) {
    // The GPU → encoder route is new; if a phone's encoder rejects its frames, render again the old way.
    if (!e || !e.retryCanvas) throw e;
    console.warn('export: GPU frame route failed at the start, using the canvas route instead', e);
    return exportOnce(clip, plan, opts, 'canvas');
  }
}

async function exportOnce(clip, plan, { onProgress = () => {}, signal } = {}, route = null) {
  const { outW, outH, fps } = plan;
  const { ticks, duration, interpCount } = buildTimeline(clip, plan.segments, plan.speed, fps, { interp: plan.interp });
  if (!ticks.length) throw new Error('Nothing to export — all segments are empty.');
  const byFrame = new Map();   // frame index → { whole: [ts], between: [[ts, u]] }
  for (const [fi, ts, u] of ticks) {
    if (!byFrame.has(fi)) byFrame.set(fi, { whole: [], between: [] });
    if (u > 0) byFrame.get(fi).between.push([ts, u]); else byFrame.get(fi).whole.push(ts);
  }
  const betweens = k => { const b = byFrame.get(k); return b && b.between.length ? b.between : null; };
  const bitrate = Math.min(80e6, Math.max(6e6, Math.round(outW * outH * fps * 0.22)));
  const enc = await pickEncoder(outW, outH, fps, bitrate);
  const tStart = activeNow();
  const T = { upload: 0, render: 0, outWait: 0, encWait: 0, flowWait: 0, srcWait: 0, aheadWait: 0, analysis: 0 };

  // ── sound: read up front (small) so it can be written alongside the video as it goes ──
  const A = clip.audio;
  const allNormal = plan.segments.every(sg => Math.abs((sg.speed != null ? sg.speed : plan.speed) - 1) < 1e-6);
  const withAudio = !!(plan.audio && A && A.description && allNormal);
  const audioQ = [];
  if (withAudio) {
    let outT = 0;
    const want = [];
    for (const seg of plan.segments) {
      const s0 = seg.start * 1e6, s1 = seg.end * 1e6;
      let lastTs = -1;
      for (const s of A.samples) {
        if (s.ts < s0 || s.ts + s.dur > s1) continue;
        const ts = Math.round(outT * 1e6 + (s.ts - s0));
        if (ts <= lastTs) continue;
        want.push(s); audioQ.push({ data: null, ts, dur: s.dur }); lastTs = ts;
      }
      outT += seg.end - seg.start;
    }
    // sound sits in runs of back-to-back samples between the video: one read per run, not a big window per sample
    const areader = new WindowReader(clip.file, 0, clip.original);
    for (let i = 0; i < want.length;) {
      let j = i + 1, end = want[i].offset + want[i].size;
      while (j < want.length && want[j].offset === end && end - want[i].offset < (4 << 20)) { end += want[j].size; j++; }
      const run = (await areader.read(want[i].offset, end - want[i].offset)).slice();
      for (let k = i, o = 0; k < j; o += want[k].size, k++) audioQ[k].data = run.subarray(o, o + want[k].size);
      i = j;
    }
  }
  const lookFor = typeof plan.look === 'function' ? plan.look : () => plan.look;
  const sink = new MP4Sink();
  const muxer = new Mp4Muxer.Muxer({
    target: new Mp4Muxer.StreamTarget({ onData: (data, pos) => sink.write(data, pos) }),
    video: { codec: enc.mux, width: outW, height: outH, frameRate: fps },
    audio: withAudio ? { codec: 'aac', numberOfChannels: A.channels, sampleRate: A.sampleRate } : undefined,
    fastStart: { expectedVideoChunks: ticks.length, expectedAudioChunks: audioQ.length },
    firstTimestampBehavior: 'cross-track-offset',
  });
  sink.freezeHead();
  let audioNext = 0, audioMeta = withAudio ? { decoderConfig: { codec: A.codec, sampleRate: A.sampleRate, numberOfChannels: A.channels, description: A.description } } : null;
  const addAudioUpTo = us => {
    while (audioNext < audioQ.length && audioQ[audioNext].ts <= us) {
      const a = audioQ[audioNext++];
      muxer.addAudioChunkRaw(a.data, 'key', a.ts, a.dur, audioMeta); audioMeta = undefined; a.data = null;
    }
  };

  // ── encoder ──
  let encErr = null, encWake = null, encoder = null, emitted = 0, encoded = 0;
  const encPoke = () => { if (encWake) { const w = encWake; encWake = null; w(); } };
  // Opened just before the first frame is encoded: an encoder left idle for minutes (the first pass of focus
  // build-up "sharp from the start" reads the whole clip without encoding) gets reclaimed by the browser.
  const openEncoder = () => {
    encoder = new VideoEncoder({
      output: (chunk, meta) => { muxer.addVideoChunk(chunk, meta); addAudioUpTo(chunk.timestamp); },
      error: e => { encErr = e; encPoke(); },
    });
    encoder.addEventListener('dequeue', encPoke);
    encoder.configure(enc.cfg);
  };
  const closeEncoder = () => { if (encoder && encoder.state !== 'closed') encoder.close(); };
  const ENCQ = 5;   // frames waiting inside the encoder: enough to keep it busy, not so many they pile up in memory
  const encodeFrame = async (vf, key) => {
    const w0 = activeNow();
    while (encoder.encodeQueueSize > ENCQ && !encErr) {
      if (activeNow() - w0 > 20000) { vf.close(); throw new Error('The video encoder stopped responding'); }
      await new Promise(r => { encWake = r; setTimeout(r, 50); });
    }
    T.encWait += activeNow() - w0;
    if (encErr) { vf.close(); throw encErr; }
    encoder.encode(vf, { keyFrame: key }); vf.close(); encoded++;
  };

  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(outW, outH) : Object.assign(document.createElement('canvas'), { width: outW, height: outH });
  const R = new Renderer(canvas);
  R.setSize(outW, outH);

  // ── finished frames → encoder. Normally as I420 straight from the GPU, a few frames in flight; the older
  // canvas route (the browser converts and reads back each frame itself, waiting for the GPU) is the fallback. ──
  let yuvOut = false;
  try {
    const mode = route || (() => { try { return localStorage.getItem('macrocut.out'); } catch (e) { return null; } })();
    if (mode !== 'canvas') { new VideoFrame(new Uint8Array(16 * 16 * 3 / 2), { format: 'I420', codedWidth: 16, codedHeight: 16, timestamp: 0 }).close(); yuvOut = true; }
  } catch (e) { yuvOut = false; }
  const Y = yuvOut ? R.useYuvOutput(true) : null;
  const OUTQ = 3, slots = [], inflight = [];
  if (Y) for (let i = 0; i < OUTQ; i++) slots.push({ pb: R.makePackBuffer(Y.bytes), buf: new Uint8Array(Y.bytes), fence: null, ts: 0, key: false });
  let slotI = 0;
  const keyEvery = Math.max(1, Math.round(fps * 2));
  const yuvInit = Y && {
    format: 'I420', codedWidth: outW, codedHeight: outH, duration: Math.round(1e6 / fps),
    layout: [{ offset: 0, stride: Y.S }, { offset: Y.S * outH, stride: Y.S / 2 }, { offset: Y.S * (outH + Y.Q), stride: Y.S / 2 }],
    colorSpace: { primaries: 'bt709', transfer: 'iec61966-2-1', matrix: 'bt709', fullRange: false },   // as the browser tags its own conversion
  };
  const finishOldest = async () => {
    const s = inflight.shift();
    const w0 = activeNow(); await R.waitFence(s.fence); s.fence = null; T.outWait += activeNow() - w0;
    R.collect(s.pb, s.buf);
    try { await encodeFrame(new VideoFrame(s.buf, { ...yuvInit, timestamp: s.ts }), s.key); }
    catch (e) { if (e && typeof e === 'object') e.fromYuv = true; throw e; }
  };
  const emit = async ts => {
    if (!encoder) openEncoder();
    const key = emitted % keyEvery === 0; emitted++;
    if (!Y) { await encodeFrame(new VideoFrame(canvas, { timestamp: ts, duration: Math.round(1e6 / fps) }), key); return; }
    if (inflight.length >= OUTQ) await finishOldest();
    const s = slots[slotI]; slotI = (slotI + 1) % OUTQ;
    s.fence = R.readYuvAsync(s.pb); s.ts = ts; s.key = key;
    inflight.push(s);
    while (inflight.length && R.fenceDone(inflight[0].fence)) await finishOldest();
  };
  const flushOut = async () => { while (inflight.length) await finishOldest(); };

  // every source frame the export reads (whole frames + the second frame of each in-between pair)
  const usedFrames = new Set();
  for (const [fi, , u] of ticks) { usedFrames.add(fi); if (u > 0) usedFrames.add(fi + 1); }
  // ── Focus build-up: parts that have been sharp stay sharp while they haven't moved (js/focus.js) ──
  let fb = null;
  if (plan.focus && plan.C && plan.C.length === clip.times.length) {
    try {
      const frames = [...usedFrames].sort((a, b) => a - b), m0 = plan.M.subarray(4 * frames[0], 4 * frames[0] + 4);
      fb = new FocusBuild(R, plan.C, clip.times, { frames, mode: plan.focus, dispW: clip.dispW, dispH: clip.dispH, srcPerOut: Math.hypot(m0[0], m0[1]) * clip.dispH / outH, outH });
    } catch (e) { R.dispose(); throw e; }
  }
  const twoPass = !!(fb && fb.mode === 'full');
  let dry = false, readCount = 0;   // dry = first pass of "sharp from the start": build the canvases, show nothing
  const fo = (k, u = 0, k2 = k) => (fb && !dry ? fb.overlay(k, u, k2) : null);
  const fromUs = Math.round(plan.segments[0].start * 1e6) - 1, toUs = Math.round(plan.segments[plan.segments.length - 1].end * 1e6) + 1;
  const tsIndex = new Map();
  clip.times.forEach((t, i) => tsIndex.set(Math.round(t * 1e6), i));
  let last = performance.now(), salvaged = null;
  const progress = () => (dry ? 0.45 * readCount / Math.max(1, usedFrames.size) : (twoPass ? 0.45 : 0) + (twoPass ? 0.55 : 1) * emitted / ticks.length);
  const maybeProgress = async () => { if (performance.now() - last > 150) { last = performance.now(); onProgress(progress()); await yieldNow(); } };
  // draw one output frame and encode it (nothing on the first, "dry" pass of focus build-up)
  const show = async (m, look, opts, ts) => { if (dry) return; const r0 = activeNow(); R.render(m, look, opts); T.render += activeNow() - r0; await emit(ts); };
  // focus build-up: add a frame to the canvas. On the dry pass nothing else paces the GPU (no frame is encoded), so
  // keep at most two frames of focus work queued: otherwise decoding races ahead and queues minutes of GPU work.
  const dryFences = [];
  const focusStep = async (k, tex, info) => {
    if (!(fb && fb.step(k, tex, info))) return;
    readCount++;
    if (dry) { dryFences.push(R.gl.fenceSync(R.gl.SYNC_GPU_COMMANDS_COMPLETE, 0)); R.gl.flush(); if (dryFences.length > 2) await R.waitFence(dryFences.shift()); }
  };
  const focusGrain = f => { if (fb && !fb.grainSet) { try { fb.setGrain(grainOfFrame(f)); } catch (e) { fb.grainSet = true; } } };
  const Mi = i => plan.M.subarray(4 * i, 4 * i + 4);
  const lerpM = (a, b, u) => [a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])];
  // In-between frames and Extra clean need a small greyscale copy of each frame to measure motion.
  const [aw, ah] = analysisSize(clip);
  const motion = (interpCount || plan.denoise) ? new MotionPool(aw, ah, Math.max(1, Math.min(3, (navigator.hardwareConcurrency || 4) - 2))) : null;
  const lumPool = [];
  // start reading frame tex's grey copy back; resolves to its RGBA rows (ArrayBuffer)
  const lumaOf = (tex, info, hasMips) => {
    const a0 = activeNow();
    const pb = lumPool.pop() || R.makePackBuffer(aw * ah * 4);
    const f = R.analysisAsync(tex, info, aw, ah, pb, hasMips);
    T.analysis += activeNow() - a0;
    return (async () => { await R.waitFence(f); const out = new Uint8Array(aw * ah * 4); R.collect(pb, out); lumPool.push(pb); return out.buffer; })();
  };
  const normFlow = (gw, gh, x, y) => { const d = new Float32Array(gw * gh * 2); for (let k = 0; k < gw * gh; k++) { d[2 * k] = x[k] / ah; d[2 * k + 1] = y[k] / ah; } return d; };
  const awaitFlow = async p => { const w0 = activeNow(); try { return await p; } finally { T.flowWait += activeNow() - w0; } };

  // ── Producer → consumer. The producer (decoder callback) uploads each needed frame and starts its motion
  // measurement; the consumer draws and encodes in order. The producer may run a few frames ahead, so the motion
  // for the next pairs is being worked out while the current ones are drawn. ──
  const texPool = [];
  const run = async (produce, consume, ahead) => {
    const queue = []; let ended = false, cErr = null, live = 0, wakeC = null, wakeP = null;
    const pokeC = () => { if (wakeC) { const w = wakeC; wakeC = null; w(); } };
    const pokeP = () => { if (wakeP) { const w = wakeP; wakeP = null; w(); } };
    const ctl = {
      // consumer side
      next: async () => {
        const w0 = activeNow();
        while (!queue.length && !ended && !(signal && signal.aborted)) await new Promise(r => { wakeC = r; setTimeout(r, 250); });
        T.srcWait += activeNow() - w0;
        return queue.length && !(signal && signal.aborted) ? queue.shift() : null;
      },
      release: e => { if (e && !e.released) { e.released = true; if (e.tex) texPool.push(e.tex); e.tex = null; live--; pokeP(); } },
      // producer side
      room: async () => {
        const w0 = activeNow();
        while (live >= ahead && !cErr && !(signal && signal.aborted)) await new Promise(r => { wakeP = r; });
        T.aheadWait += activeNow() - w0;
        if (cErr) throw cErr;
      },
      push: e => { live++; queue.push(e); pokeC(); },
      check: () => { if (cErr) throw cErr; },
    };
    const consumer = consume(ctl).catch(e => { cErr = e; pokeP(); });
    let pErr = null;
    try {
      await decodeRange(clip, { fromUs, toUs, signal, onFrame: f => { if (cErr) { f.close(); throw cErr; } return produce(f, ctl); } });
    } catch (e) { pErr = e; }
    ended = true; pokeC();
    await consumer;   // draws whatever was already read (a read failure near the end still leaves a usable video)
    for (const e of queue.splice(0)) ctl.release(e);
    if (cErr) throw cErr;
    if (pErr) throw pErr;
  };

  // ── plain path (+ in-between frames, + focus build-up) ──
  const runPlain = () => {
    let lastE = null;
    const produce = async (f, ctl) => {
      const k = tsIndex.get(f.timestamp);
      if (k === undefined || !usedFrames.has(k)) { f.close(); return; }
      await ctl.room();
      const info = frameInfo(clip, f), m = Mi(k);
      const mip = Math.hypot(m[0], m[1]) * info.dispH / outH > 1.25;
      const u0 = activeNow();
      const tex = texPool.pop() || R.makeTex();
      R.uploadTex(tex, f);
      if (mip) R.mipTex(tex);
      T.upload += activeNow() - u0;
      focusGrain(f);
      const needLum = motion && !dry && (betweens(k) || betweens(k - 1));
      const lum = needLum ? lumaOf(tex, info, mip) : null;
      f.close();
      const e = { k, tex, info, mip, lum, flowPrev: null };
      if (lum && lastE && lastE.k === k - 1 && lastE.lum && betweens(k - 1)) {
        const a = lastE.lum;
        e.flowPrev = Promise.all([a, lum]).then(([x, y]) => motion.flow(x, y));
        e.flowPrev.catch(() => {});
      }
      lastE = e;
      ctl.push(e);
      await maybeProgress();
    };
    const holdFrames = async (p, list) => { R.useSrc(p.tex, p.info, p.mip); for (const [ts] of list) await show(Mi(p.k), lookFor(clip.times[p.k]), { pxScale: 1, focus: fo(p.k) }, ts); };
    const consume = async ctl => {
      let prev = null;
      for (let e; (e = await ctl.next());) {
        const pb = prev && betweens(prev.k);
        if (pb && prev.k === e.k - 1) {
          const F = dry || !e.flowPrev ? null : await awaitFlow(e.flowPrev);
          if (!F) { if (!dry) await holdFrames(prev, pb); }
          else {
            R.useSrc(prev.tex, prev.info, prev.mip); R.useSrcB(e.tex);
            R.setFlow(F.gw, F.gh, normFlow(F.gw, F.gh, F.fx, F.fy));
            const iA = prev.k, mA = Mi(iA), mB = Mi(iA + 1), tA = clip.times[iA], tB = clip.times[iA + 1];
            for (const [ts, u] of pb) await show(lerpM(mA, mB, u), lookFor(tA + u * (tB - tA)), { pxScale: 1, interp: u, focus: fo(iA, u, iA + 1) }, ts);
          }
        } else if (pb) await holdFrames(prev, pb);   // its next frame is missing: hold it instead
        ctl.release(prev);
        await focusStep(e.k, e.tex, e.info);
        R.useSrc(e.tex, e.info, e.mip);
        const outs = byFrame.get(e.k);
        if (outs) for (const ts of outs.whole) await show(Mi(e.k), lookFor(clip.times[e.k]), { pxScale: 1, focus: fo(e.k) }, ts);
        prev = e;
        await maybeProgress();
      }
      // last frame still owes in-between frames (end of range): hold it
      if (prev && betweens(prev.k) && !(signal && signal.aborted)) await holdFrames(prev, betweens(prev.k));
      ctl.release(prev);
    };
    return run(produce, consume, motion && !dry ? 5 : 3);
  };

  // ── Extra clean: each frame blended with its ±2 neighbours along measured motion ──
  const D = 2;
  const runDenoised = async () => {
    let H = 0.03, grainChecks = 0;
    // Blend threshold follows the clip's real grain: only differences grain could explain get averaged.
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
        if (fb && !fb.grainSet) fb.setGrain(sigma);
      } catch (e) { /* keep previous */ }
    };
    const needed = usedFrames;
    const near = k => { for (let d = -D; d <= D; d++) if (needed.has(k + d)) return true; return false; };
    let lastP = null;
    const produce = async (f, ctl) => {
      const k = tsIndex.get(f.timestamp);
      if (k === undefined || !near(k)) { f.close(); return; }
      await ctl.room();
      const info = frameInfo(clip, f), m = Mi(k);
      const mip = Math.hypot(m[0], m[1]) * info.dispH / outH > 1.25;
      if (grainChecks === 0 || (k % 30 === 0 && grainChecks < 20)) measureGrain(f);
      const u0 = activeNow();
      const tex = texPool.pop() || R.makeTex();
      R.uploadTex(tex, f);
      T.upload += activeNow() - u0;
      const lum = lumaOf(tex, info, false);
      f.close();
      const e = { k, tex, info, mip, lum, H, sharp: 0, flowPrev: null };
      if (lastP && lastP.k === k - 1) {
        const p = lastP;
        e.flowPrev = Promise.all([p.lum, lum]).then(([x, y]) => motion.flow(x, y)).then(F => {
          p.sharp = F.sharpA; e.sharp = F.sharpB;
          return { fw: { gw: F.gw, gh: F.gh, w: aw, h: ah, x: F.fx, y: F.fy }, bw: { gw: F.gw, gh: F.gh, w: aw, h: ah, x: F.bx, y: F.by } };
        });
        e.flowPrev.catch(() => {});
      }
      lastP = e;
      ctl.push(e);
      await maybeProgress();
    };
    const consume = async ctl => {
      const raw = new Map(), flowPool = [], targets = [], done = new Set();
      let flip = 0, lastK = -2, prevC = null, Hcur = H;
      const fwOf = async c => { const e = raw.get(c + 1); return e && e.flowPrev ? (await awaitFlow(e.flowPrev)).fw : null; };   // c → c+1
      const bwOf = async c => { const e = raw.get(c); return e && e.flowPrev && raw.has(c - 1) ? (await awaitFlow(e.flowPrev)).bw : null; };   // c → c−1
      const flowTex = F => { const t = flowPool.pop() || R.makeTex(); R.setFlowTex(t, F.gw, F.gh, normFlow(F.gw, F.gh, F.x, F.y)); return t; };
      const holdPrev = async () => {
        const pb = prevC && betweens(prevC.c);
        if (pb && !prevC.betweenDone) {
          R.useSrc(prevC.tex, prevC.info, prevC.mip);
          for (const [ts] of pb) await show(Mi(prevC.c), lookFor(clip.times[prevC.c]), { pxScale: 1, focus: fo(prevC.c) }, ts);
          prevC.betweenDone = true;
        }
      };
      const ready = async (c, tex, C) => {
        const pb = prevC && betweens(prevC.c);
        const F = pb && prevC.c === c - 1 ? await fwOf(c - 1) : null;
        if (F) {
          R.useSrc(prevC.tex, prevC.info, prevC.mip); R.useSrcB(tex);
          R.setFlow(F.gw, F.gh, normFlow(F.gw, F.gh, F.x, F.y));
          const mA = Mi(c - 1), mB = Mi(c), tA = clip.times[c - 1], tB = clip.times[c];
          for (const [ts, u] of pb) await show(lerpM(mA, mB, u), lookFor(tA + u * (tB - tA)), { pxScale: 1, interp: u, focus: fo(c - 1, u, c) }, ts);
          prevC.betweenDone = true;
        } else await holdPrev();
        await focusStep(c, tex, C.info);
        R.useSrc(tex, C.info, C.mip);
        const outs = byFrame.get(c);
        if (outs) for (const ts of outs.whole) await show(Mi(c), lookFor(clip.times[c]), { pxScale: 1, focus: fo(c) }, ts);
        prevC = { c, tex, info: C.info, mip: C.mip };
      };
      const finalize = async c => {
        if (done.has(c) || !raw.has(c) || !needed.has(c)) return;
        done.add(c);
        const C = raw.get(c), neigh = [], used = [];
        const fwc = await fwOf(c), bwc = await bwOf(c), fw1 = fwc ? await fwOf(c + 1) : null, bw1 = bwc ? await bwOf(c - 1) : null;
        // a blurrier neighbour (shake blur, focus drift) would soften the result: weight it down
        const add = (k, F) => {
          if (!raw.has(k) || !F) return;
          const ft = flowTex(F); used.push(ft);
          const r = raw.get(k).sharp / Math.max(1e-6, C.sharp);
          neigh.push({ tex: raw.get(k).tex, flowTex: ft, weight: Math.min(1, r * r) });
        };
        add(c + 1, fwc);
        add(c - 1, bwc);
        if (fwc && fw1) add(c + 2, chainField(fwc, fw1));
        if (bwc && bw1) add(c - 2, chainField(bwc, bw1));
        if (!targets.length) for (let k = 0; k < 2; k++) targets.push(R.makeTarget(C.info.texW, C.info.texH));
        const tgt = targets[flip]; flip ^= 1;
        R.denoise(C.tex, neigh, C.info, tgt, Hcur, C.mip);
        flowPool.push(...used);
        await ready(c, tgt.t, C);
      };
      const flushAll = async () => { for (const c of [...raw.keys()].sort((a, b) => a - b)) await finalize(c); };
      const evict = keep => { for (const k of [...raw.keys()]) if (k < keep) { ctl.release(raw.get(k)); raw.delete(k); } };
      for (let e; (e = await ctl.next());) {
        const k = e.k;
        if (k !== lastK + 1) { await flushAll(); await holdPrev(); evict(Infinity); prevC = null; }
        lastK = k; Hcur = e.H;
        raw.set(k, e);
        await finalize(k - D);
        evict(k - 2 * D);
        await maybeProgress();
      }
      if (!(signal && signal.aborted)) { await flushAll(); await holdPrev(); }
      evict(Infinity);
      if (dry) {   // first pass of "sharp from the start": free this pass's frames before the second pass allocates its own
        const gl = R.gl;
        for (const t of [...texPool.splice(0), ...flowPool]) gl.deleteTexture(t);
        for (const t of targets) { gl.deleteTexture(t.t); gl.deleteFramebuffer(t.fb); }
      } else for (const t of flowPool) R.gl.deleteTexture(t);
      console.info(`MacroCut extra clean: blend threshold ${H.toFixed(4)}`);
    };
    // at most ±2 neighbours and the frame itself in use, plus a couple of frames read ahead for the motion threads
    return run(produce, consume, 2 * D + 1 + 2);
  };

  const runPass = () => (plan.denoise ? runDenoised() : runPlain());
  try {
    if (twoPass) {
      // "sharp from the start": pass 1 builds each pose's canvas; pass 2 starts every pose from it
      dry = true; fb.startPass(1);
      await runPass();
      for (const f of dryFences.splice(0)) await R.waitFence(f);
      dry = false; fb.startPass(2);
      if (!(signal && signal.aborted)) await runPass();
    } else await runPass();
    if (!(signal && signal.aborted)) await flushOut();
  } catch (e) {
    // Nearly done and the encoder is fine: keep the video instead of throwing it away; it just ends a moment early.
    if (!encErr && encoder && encoder.state === 'configured' && emitted >= ticks.length * 0.9 && !(signal && signal.aborted)) {
      console.warn('salvaging export', e); salvaged = e;
      try { await flushOut(); } catch (e2) { console.warn('salvage flush', e2); }
    } else {
      closeEncoder(); if (fb) fb.dispose(); if (motion) motion.close(); R.dispose();
      if (Y && emitted <= 30 && e && typeof e === 'object' && (e.fromYuv || e === encErr) && !(signal && signal.aborted)) e.retryCanvas = true;
      throw e;
    }
  }
  const nWorkers = motion ? motion.workers.length : 0;
  if (motion) motion.close();
  const focusInfo = fb ? { mode: fb.mode, frames: readCount, canvas: `${fb.geo.AW}×${fb.geo.AH}` } : null;
  if (fb) { console.info(`MacroCut focus build-up: ${fb.mode}, canvas ${fb.geo.AW}×${fb.geo.AH}, decisions ${fb.geo.DW}×${fb.geo.DH}, ${fb.steps} frame updates`); fb.dispose(); }
  R.dispose();
  if (signal && signal.aborted) { closeEncoder(); return null; }
  if (!encoder) throw new Error('Nothing was rendered: the clip could not be read.');
  const f0 = activeNow();
  await encoder.flush();
  T.encWait += activeNow() - f0;
  if (encErr) throw encErr;
  encoder.close();
  addAudioUpTo(Infinity);
  muxer.finalize();
  onProgress(1);
  const total = activeNow() - tStart;
  const timing = { total, ...T, outMode: Y ? 'yuv' : 'canvas', encoderHw: enc.hw, workers: nWorkers };
  console.info('MacroCut timing ' + JSON.stringify(Object.fromEntries(Object.entries(timing).map(([k, v]) => [k, typeof v === 'number' ? Math.round(v) : v]))));
  return { blob: sink.blob('video/mp4'), codec: enc.cfg.codec, hardware: enc.hw, frames: emitted, hasAudio: withAudio, duration, interpolated: interpCount, cleaned: !!plan.denoise, focus: focusInfo,
    shortBy: salvaged ? Math.max(0, (ticks.length - emitted) / fps) : 0,
    timing };
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
