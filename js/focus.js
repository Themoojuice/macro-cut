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
  // movement is pooled in blocks of ~24 decision cells (doubt rule)
  const MW = Math.max(4, Math.ceil(DW / 24)), MH = Math.max(3, Math.ceil(DH / 24));
  return { CW: A * k, CH: k, AW, AH, DW, DH, MW, MH, scale, A };
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
    shift: 2 / 720,                                 // "moved" = changed even allowing this much extra shift
    covLod: Math.max(0, Math.log2(geo.AH / geo.DH)),
    down: geo.scale < 0.8 ? Math.log2(1 / geo.scale) : 0,
  };
}

/**
 * Drives the GPU accumulator through an export.
 * R: Renderer (focus* methods). C: camera per frame. times: frame times (s).
 * opts: { frames (sorted processed frame indices), mode: 'build'|'full', dispW, dispH, srcPerOut, outH, debug }
 */
export class FocusBuild {
  constructor(R, C, times, opts) {
    this.R = R; this.C = C; this.times = times; this.mode = opts.mode; this.outH = opts.outH;
    this.dispH = opts.dispH;
    const maxTex = R.gl ? R.gl.getParameter(R.gl.MAX_TEXTURE_SIZE) : 8192;
    this.geo = canvasGeometry(opts.dispW, opts.dispH, opts.srcPerOut, maxTex);
    this.prm = decisionParams(opts.dispH, this.geo);
    this.anchorOf = planAnchors(opts.frames, C, this.geo);
    this.noise = 0.2 * 0.004 * 0.004; this.eps = 1e-4; this.grainSet = false;
    this.pass = opts.mode === 'full' ? 1 : 2;
    this.seeds = new Map();          // anchor → { tex, frames } built by pass 1 ("sharp from the start")
    this.segLen = new Map();
    for (const a of this.anchorOf.values()) this.segLen.set(a, (this.segLen.get(a) || 0) + 1);
    // memory: keep at most this many finished canvases between the passes (longest poses first)
    const bytes = this.geo.AW * this.geo.AH * 4 * 1.34 + this.geo.DW * this.geo.DH * 8;
    this.maxSeeds = Math.max(1, Math.min(8, Math.floor(160e6 / bytes)));
    this.cur = null; this.lastK = null; this.steps = 0;
    this.debug = !!opts.debug; this.log = [];   // debug: per frame [k, moved, doubt, detail, pass, frame-wide moved, verified, moved blocks]
    R.focusBegin(this.geo);
  }
  // grain level of the clip (σ, 0..1): sharpness below what grain alone makes doesn't count
  setGrain(sigma) {
    const s = Math.max(sigma || 0, 0.002);
    this.noise = 0.2 * s * s; this.eps = Math.max(4 * this.noise, 1e-4); this.grainSet = true;
  }
  startPass(p) {
    if (p === 2 && this.pass === 1 && this.cur != null) this._keepSeed(this.cur);
    this.pass = p; this.cur = null; this.lastK = null;
  }
  _keepSeed(a) {
    const n = this.segLen.get(a) || 0;
    if (n < 6) return;   // too short to be worth it
    this.seeds.set(a, { tex: this.R.focusDetach(), frames: n });
    while (this.seeds.size > this.maxSeeds) {
      let worst = null; for (const [k, s] of this.seeds) if (!worst || s.frames < worst[1].frames) worst = [k, s];
      this.R.focusDrop(worst[1].tex); this.seeds.delete(worst[0]);
    }
  }
  _start(a) {
    if (this.pass === 1 && this.cur != null) this._keepSeed(this.cur);
    this.cur = a; this.lastK = null;
    const s = this.pass === 2 && this.seeds.get(a);
    if (s) { this.R.focusAttach(s.tex); this.seeds.delete(a); } else this.R.focusClear();
  }
  // Add frame k (its texture is on the GPU, display info as frameInfo()).
  step(k, tex, info) {
    const a = this.anchorOf.get(k);
    if (a === undefined) return false;
    if (a !== this.cur) this._start(a);
    const dt = this.lastK == null ? 0 : Math.max(0, this.times[k] - this.times[this.lastK]);
    const T = invert(frameToAnchor(this.C[a], this.C[k]));   // canvas → frame k
    this.R.mipTex(tex);
    this.R.focusStep(tex, info, T, { ...this.prm, noise: this.noise, eps: this.eps, decay: Math.pow(0.5, dt / 11) });
    this.lastK = k; this.steps++;
    if (this.debug) {
      const st = this.R.focusStats(this.eps);
      this.log.push([k, +st.moved.toFixed(4), +st.doubt.toFixed(4), +st.detail.toFixed(4), this.pass, st.globR, st.globG, st.globA]);
    }
    return true;
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
  dispose() { for (const s of this.seeds.values()) this.R.focusDrop(s.tex); this.seeds.clear(); }
}
