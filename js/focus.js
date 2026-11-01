// Focus build-up: every part of the spider that has been sharp stays sharp while it hasn't moved.
// Geometry and bookkeeping live here (pure, Node-testable); the per-pixel work is in gpu.js (focus* methods).
//
// The stored ("accumulated") picture lives on a canvas in an ANCHOR frame's normalised coordinates
// (height units, centred, y down; see core.js), widened by a margin. The tracker's camera C_k maps frame k
// onto a world fixed to the tracked layer, so frame k → canvas is G_k = C_a⁻¹ ∘ C_k. Pixels are resampled
// once into the canvas and once out of it, never repeatedly.
import { compose, invert, apply } from './core.js';

export const FOCUS_MARGIN = 0.08;   // canvas = anchor frame + 8% on every side

/**
 * Canvas and decision-grid sizes.
 * dispW, dispH: source display size (px). srcPerOut: source px per output px (export zoom); the canvas is
 * never finer than ~1.6 canvas px per output px, so lower export resolutions use less memory.
 */
export function canvasGeometry(dispW, dispH, srcPerOut = 1, maxTex = 8192) {
  const k = 1 + 2 * FOCUS_MARGIN, A = dispW / dispH;
  let scale = Math.min(1, 1.6 / Math.max(1e-6, srcPerOut));
  let AW = Math.round(dispW * k * scale), AH = Math.round(dispH * k * scale);
  if (Math.max(AW, AH) > maxTex) { const f = maxTex / Math.max(AW, AH); AW = Math.floor(AW * f); AH = Math.floor(AH * f); scale *= f; }
  // decisions are made on a grid of about 540 rows per frame height (fine enough for edges, cheap on a phone)
  const d = Math.min(1, (540 * k) / AH);
  const DW = Math.max(16, Math.round(AW * d)), DH = Math.max(16, Math.round(AH * d));
  return { CW: A * k, CH: k, AW, AH, DW, DH, scale, A };
}

// frame k (normalised) → anchor canvas (normalised)
export function frameToAnchor(Ca, Ck) { return compose(invert(Ca), Ck); }

// Does a frame still fit the canvas? (view moved, turned or zoomed too far → start a fresh canvas)
export function fitsCanvas(G, geo) {
  const s = Math.hypot(G[0], G[1]), th = Math.atan2(G[1], G[0]);
  if (Math.abs(Math.log(s)) > 0.2 || Math.abs(th) > 0.17) return false;
  let out = 0, n = 0;
  for (let j = 0; j <= 8; j++) for (let i = 0; i <= 8; i++) {
    const [px, py] = apply(G, (i / 8 - 0.5) * geo.A, j / 8 - 0.5);
    n++; if (Math.abs(px) > geo.CW / 2 || Math.abs(py) > geo.CH / 2) out++;
  }
  return out / n <= 0.12;
}

// Anchor for every processed frame (in time order). Depends only on camera geometry, so two passes over
// the same frames always agree on where each canvas starts.
export function planAnchors(frames, C, geo) {
  const anchorOf = new Map();
  let a = -1;
  for (const k of frames) {
    if (a < 0 || !fitsCanvas(frameToAnchor(C[a], C[k]), geo)) a = k;
    anchorOf.set(k, a);
  }
  return anchorOf;
}

// Mip level (box-filter pyramid, trilinear) that blurs by about σ pixels.
export function lodForSigma(sigmaPx) { return Math.max(0, Math.log2(Math.max(1e-6, sigmaPx) / 0.29)); }

// Shader parameters that depend on resolution. Blur sizes are set in "720p pixels" (frame height / 720)
// so the same clip behaves the same at any resolution.
export function decisionParams(dispH, geo) {
  const px720 = dispH / 720;                       // source px per 720p px
  const accPx = px720 * geo.scale;                 // canvas px per 720p px
  return {
    lodW: lodForSigma(0.75 * px720),                // the live frame, lightly blurred (grain, blocking)
    lodA: [0.75, 1.5, 3, 6, 12, 24].map(s => lodForSigma(s * accPx)),   // stored view blurred by several amounts
    win: 4 / 720,                                   // spacing of the 3×3 comparison taps (normalised units)
    shift: 2 / 720,                                 // pose tracking forgives shifts this small (lean-in parallax)
    covLod: Math.max(0, Math.log2(geo.AH / geo.DH)),
    down: geo.scale < 0.8 ? Math.log2(1 / geo.scale) : 0,
  };
}

// Pose tracking for "sharp from the start" (first pass). Measured in windows of `refresh` seconds: `since` is the
// share of the spider's detailed area that stopped matching during the window (an area: repeated palp flicks
// count once). Tuned on real clips (see docs/SPEC-multiframe.md §C).
export const POSE = {
  refresh: 0.25, // s per window
  quiet: 0.2,    // a window that moved less than this is a still moment: snapshot "this pose so far"
  brk: 0.4,      // movement since the last still moment above this = a new pose
};
// Measured (360p copies, share of detailed area per 0.25 s window, upper bound): still spider during focus sweeps
// 0.05–0.19, settled 0.01–0.03; head turn 0.5–1.7. Slow or small moves (a head drop) stay below `quiet`: no new
// pose, which only means that stretch behaves like Build up.

/**
 * Pose segmentation (pure; the live code and the tests use the same logic).
 * push(k, t, since) per frame → { snap: take a still snapshot now, brk: null | { end: last still frame },
 * reset: start a new window (clear the GPU "since" mask) }
 */
export function poseTracker(prm = POSE) {
  let tRef = null, candK = null, carried = 0;
  return {
    get candK() { return candK; },
    get moving() { return carried > 0; },
    push(k, t, since) {
      if (tRef == null) tRef = t;
      const out = { k, snap: false, brk: null, reset: false };
      if (carried + since > prm.brk) {
        out.brk = { end: candK }; out.reset = true;
        candK = null; carried = 0; tRef = t;
        return out;
      }
      if (t - tRef >= prm.refresh - 1e-6) {
        if (since < prm.quiet) { out.snap = true; candK = k; carried = 0; } else carried += since;
        tRef = t; out.reset = true;
      }
      return out;
    },
  };
}

/**
 * Drives the GPU accumulator through an export.
 * R: Renderer (focus* methods). C: camera per frame. times: frame times (s).
 * opts: { frames (sorted processed frame indices), mode: 'build'|'full', dispW, dispH, srcPerOut, outH }
 *
 * "full" (sharp from the start) = two passes. Pass 1 builds the canvas and splits the clip into segments: a new
 * canvas (view moved too far) or a new pose (a big part of the spider stopped matching). Each segment keeps a
 * snapshot of its best state (for a pose: the last still moment before the spider moved). Pass 2 starts every
 * segment from its snapshot, so each pose is sharp from its first frame.
 */
export class FocusBuild {
  constructor(R, C, times, opts) {
    this.R = R; this.C = C; this.times = times; this.mode = opts.mode; this.outH = opts.outH;
    this.dispH = opts.dispH; this.frames = opts.frames;
    const maxTex = R.gl ? R.gl.getParameter(R.gl.MAX_TEXTURE_SIZE) : 8192;
    this.geo = canvasGeometry(opts.dispW, opts.dispH, opts.srcPerOut, maxTex);
    this.prm = decisionParams(opts.dispH, this.geo);
    this.anchorOf = planAnchors(opts.frames, C, this.geo);
    this.noise = 0.2 * 0.004 * 0.004; this.eps = 1e-4; this.grainSet = false;
    this.pass = opts.mode === 'full' ? 1 : 2;
    // memory: snapshots kept between the passes (no mipmaps), longest segments first; one more for the live candidate
    const bytes = this.geo.AW * this.geo.AH * 4 + this.geo.DW * this.geo.DH * 8;
    this.maxSeeds = Math.max(1, Math.min(12, Math.floor(160e6 / bytes) - 1));
    this.segments = [];          // pass 1 → 2: { start, end, geo (new canvas), seed }
    this.segAt = new Map();      // pass 2: start frame → segment
    this.cur = null; this.lastK = null; this.steps = 0;
    this.debug = !!opts.debug; this.log = [];
    R.focusBegin(this.geo);
  }
  // grain level of the clip (σ, 0..1): sharpness below what grain alone makes doesn't count
  setGrain(sigma) {
    const s = Math.max(sigma || 0, 0.002);
    this.noise = 0.2 * s * s; this.eps = Math.max(4 * this.noise, 1e-4); this.grainSet = true;
  }
  _nextFrame(k) { const f = this.frames; let lo = 0, hi = f.length; while (lo < hi) { const m = (lo + hi) >> 1; if (f[m] <= k) lo = m + 1; else hi = m; } return lo < f.length ? f[lo] : null; }
  _len(sg) { let n = 0; for (const k of this.frames) if (k >= sg.start && k <= sg.end) n++; return n; }
  _setSeed(sg, snap) {
    sg.seed = snap;
    const kept = this.segments.filter(x => x.seed);
    if (kept.length > this.maxSeeds) {
      kept.sort((a, b) => this._len(a) - this._len(b));
      this.R.focusDrop(kept[0].seed); kept[0].seed = null;
    }
  }
  // pass 1: close the open segment at frame `end`; its snapshot is `snap` (or none)
  _close(end, snap) {
    const sg = this.segments[this.segments.length - 1];
    if (!sg || sg.end != null) { if (snap) this.R.focusDrop(snap); return; }
    sg.end = end;
    if (snap && end >= sg.start) this._setSeed(sg, snap); else if (snap) this.R.focusDrop(snap);
  }
  // pass 1 at a new canvas / end of pass: best snapshot of the open segment = now if still, else the last still one
  _closeHere() {
    const sg = this.segments[this.segments.length - 1];
    if (!sg || sg.end != null || this.lastK == null) return;
    const still = !this.pose.moving || this.cand == null;
    const snap = still ? this.R.focusSnapshot() : this.cand;
    if (still && this.cand) this.R.focusDrop(this.cand);
    this.cand = null;
    this._close(still ? this.lastK : this.pose.candK, snap);
  }
  startPass(p) {
    if (p === 2 && this.pass === 1) {
      this._closeHere();
      for (const sg of this.segments) this.segAt.set(sg.start, sg);
      if (this.debug) this.log.push(['segments', this.segments.map(sg => [sg.start, sg.end, sg.geo ? 'canvas' : 'pose', !!sg.seed])]);
    }
    this.pass = p; this.cur = null; this.lastK = null;
  }
  // Add frame k (its texture is on the GPU, display info as frameInfo()).
  step(k, tex, info) {
    const a = this.anchorOf.get(k);
    if (a === undefined) return false;
    const R = this.R;
    if (this.pass === 1) {
      if (a !== this.cur) {   // new canvas
        this._closeHere();
        this.segments.push({ start: k, end: null, geo: true, seed: null });
        this.pose = poseTracker(); this.cand = null;
        R.focusClear();
      }
    } else {
      const sg = this.segAt.get(k);
      if (sg && sg.seed) { R.focusRestore(sg.seed); R.focusDrop(sg.seed); sg.seed = null; }
      else if (a !== this.cur || (sg && sg.geo)) R.focusClear();
    }
    this.cur = a;
    const dt = this.lastK == null ? 0 : Math.max(0, this.times[k] - this.times[this.lastK]);
    const T = invert(frameToAnchor(this.C[a], this.C[k]));   // canvas → frame k
    R.mipTex(tex);
    R.focusStep(tex, info, T, { ...this.prm, noise: this.noise, eps: this.eps, decay: Math.pow(0.5, dt / 11) });
    this.lastK = k; this.steps++;
    if (this.pass === 1 || this.debug) {
      const st = R.focusStats(this.eps);
      if (this.debug) this.log.push([k, +st.changed.toFixed(4), +st.since.toFixed(4), +st.detail.toFixed(4), this.pass]);
      if (this.pass === 1) this._watch(k, st);
    }
    return true;
  }
  // pass 1: follow the pose; snapshot still moments; split where the spider moved
  _watch(k, st) {
    const d = this.pose.push(k, this.times[k], st.since);
    if (d.reset) this.R.focusResetSince();
    if (d.brk) {
      const end = d.brk.end != null ? d.brk.end : k;
      this._close(end, d.brk.end != null ? this.cand : null);
      this.cand = null;
      const next = this._nextFrame(end);
      if (next != null) this.segments.push({ start: next, end: null, geo: false, seed: null });
      if (this.debug) this.log.push(['pose', k, end, next]);
    } else if (d.snap) this.cand = R_snapshot(this.R, this.cand);
  }
  // Overlay for rendering frame k (or u of the way from k to k2, for in-between frames).
  overlay(k, u = 0, k2 = k) {
    if (this.cur == null || this.pass !== 2) return null;
    let G = frameToAnchor(this.C[this.cur], this.C[k]);
    if (u > 0 && k2 !== k) {
      const G2 = frameToAnchor(this.C[this.cur], this.C[k2]);
      G = G.map((v, i) => v + u * (G2[i] - v));
    }
    return { G };
  }
  dispose() {
    for (const sg of this.segments) if (sg.seed) { this.R.focusDrop(sg.seed); sg.seed = null; }
    if (this.cand) { this.R.focusDrop(this.cand); this.cand = null; }
  }
}
const R_snapshot = (R, into) => R.focusSnapshot(into);
