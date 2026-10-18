import { openClip, decodeRange } from './media.js';
import { Renderer, Downscaler, DEFAULT_LOOK } from './gpu.js';
import { computeStabilisation, clampPan, frameAt } from './core.js';
import { analyseClip, exportClip, frameInfo } from './engine.js';

const $ = id => document.getElementById(id);
const video = $('video'), view = $('view');

// ───────────── state ─────────────
const LOOK_PRESETS = {
  natural: { ...DEFAULT_LOOK },
  pop: { ...DEFAULT_LOOK, cas: 0.55, detail: 1.1, radius: 1.4, denoise: 0.25, contrast: 0.12, shadows: 0.25, highlights: 0.3, saturation: 0.15, vibrance: 0.45 },
  off: { cas: 0, detail: 0, radius: 1.5, denoise: 0, exposure: 0, contrast: 0, shadows: 0, highlights: 0, saturation: 0, vibrance: 0, warmth: 0, tint: 0 },
};
const state = {
  clip: null, segments: [], sel: 0,
  analysis: null, analysing: null,
  stab: { mode: 'smooth', sigma: 0.8, maxCrop: 0.2, rotation: true, zoom: true, roiMode: 'all', roi: null },
  frame: { aspect: 'native', rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 },
  look: { ...DEFAULT_LOOK },
  speed: 1,
  exp: { res: 'source', fps: 'auto', audio: true },
  muted: false, playing: false, compare: false,
  mediaTime: 0, plan: null, planRaw: null, previewInfo: null, tab: 'tab-trim',
};
const STORE = 'macrocut.v3';
try {
  const s = JSON.parse(localStorage.getItem(STORE) || 'null');
  if (s) {
    Object.assign(state.look, s.look || {});
    Object.assign(state.stab, s.stab || {}, { roi: null, roiMode: 'all' });
    Object.assign(state.exp, s.exp || {});
  }
} catch (e) { /* storage unavailable */ }
const persist = () => { try { localStorage.setItem(STORE, JSON.stringify({ look: state.look, stab: state.stab, exp: state.exp })); } catch (e) { /* ignore */ } };

// ───────────── toast / progress ─────────────
let toastTimer = null;
function toast(msg, ms = 0) {
  clearTimeout(toastTimer); const t = $('toast'); t.textContent = msg; t.style.display = 'block';
  if (ms) toastTimer = setTimeout(() => { t.style.display = 'none'; }, ms);
}
const hideToast = () => { $('toast').style.display = 'none'; };
const progress = p => { $('progressBar').style.width = p == null ? '0' : `${Math.round(p * 100)}%`; };

// ───────────── renderer ─────────────
let renderer;
try { renderer = new Renderer(view); } catch (e) { toast('WebGL2 is not available in this browser: ' + e.message); }

const fmt = s => { const m = Math.floor(s / 60); return `${String(m).padStart(2, '0')}:${(s % 60).toFixed(2).padStart(5, '0')}`; };
const even = x => Math.max(2, Math.round(x / 2) * 2);
const ASPECTS = { '9:16': 9 / 16, '4:5': 4 / 5, '1:1': 1, '16:9': 16 / 9 };

function srcAspect() { return state.clip.dispW / state.clip.dispH; }
function outAspect() {
  const A = srcAspect();
  if (state.frame.aspect === 'native') return state.frame.quarter & 1 ? 1 / A : A;
  return ASPECTS[state.frame.aspect];
}
function outputDims(maxShort) {
  const c = state.clip, odd = state.frame.quarter & 1;
  const sw = odd ? c.dispH : c.dispW, sh = odd ? c.dispW : c.dispH, Ao = outAspect();
  let w = sw, h = sw / Ao; if (h > sh) { h = sh; w = sh * Ao; }
  const short = Math.min(w, h);
  if (maxShort && short > maxShort) { const k = maxShort / short; w *= k; h *= k; }
  return [even(w), even(h)];
}
const RES = { source: 2160, '1080': 1080, '720': 720 };
function exportDims() { return outputDims(RES[state.exp.res] || 2160); }
function previewDims() {
  const [w, h] = outputDims(2160), k = Math.min(1, 1280 / Math.max(w, h));
  return [even(w * k), even(h * k)];
}

function keptMask() {
  const c = state.clip, m = new Uint8Array(c.times.length);
  c.times.forEach((t, i) => { m[i] = state.segments.some(s => t >= s.start - 1e-4 && t <= s.end + 1e-4) ? 1 : 0; });
  return m;
}

function planOpts(withStab) {
  const st = state.stab, fr = state.frame, c = state.clip;
  return {
    C: withStab && state.analysis && st.mode !== 'off' ? state.analysis.C : null,
    times: c.times, kept: keptMask(), mode: st.mode, sigma: st.sigma, strength: 1,
    rotation: st.rotation, zoom: st.zoom, maxCrop: st.maxCrop,
    A: srcAspect(), Ao: outAspect(), quarter: fr.quarter, rot: fr.rot * Math.PI / 180,
    zoomUser: fr.zoom, panX: fr.panX, panY: fr.panY,
  };
}

let planDirty = false;
function recompute() {
  if (!state.clip) return;
  const [pw, ph] = previewDims();
  renderer.setSize(pw, ph);
  state.plan = computeStabilisation(planOpts(true));
  state.planRaw = computeStabilisation(planOpts(false));
  updateStabStatus();
  planDirty = false;
}
function invalidate() { planDirty = true; requestDraw(); }

let drawQueued = false;
function requestDraw() {
  if (drawQueued) return; drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}
function currentIndex() { return frameAt(state.clip.times, state.mediaTime); }
function currentM() {
  const i = currentIndex(), P = state.compare ? state.planRaw : state.plan;
  return P.M.subarray(4 * i, 4 * i + 4);
}
function draw() {
  if (!state.clip || !renderer || !renderer.srcInfo) return;
  if (planDirty) recompute();
  const [, eh] = exportDims();
  renderer.render(currentM(), state.look, { raw: state.compare, pxScale: renderer.size[1] / eh });
}

// ───────────── loading ─────────────
let videoURL = null;
async function loadFile(file) {
  if (!file) return;
  if (state.analysing) state.analysing.abort();
  try {
    $('emptyState').style.display = 'none';
    state.previewInfo = null;
    const clip = await openClip(file, msg => toast(msg));
    state.clip = clip;
    state.analysis = null; state.stab.roi = null;
    state.segments = [{ start: 0, end: clip.duration }]; state.sel = 0;
    state.frame = { aspect: 'native', rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 };
    if (videoURL) URL.revokeObjectURL(videoURL);
    videoURL = URL.createObjectURL(file);
    video.src = videoURL;
    video.muted = true;
    await new Promise((res, rej) => { video.onloadeddata = res; video.onerror = () => rej(new Error('The browser could not play this file.')); });
    video.currentTime = 0;
    state.previewInfo = await detectPreviewOrientation(clip);
    video.muted = state.muted;
    $('metaBadge').textContent = `${clip.dispW}×${clip.dispH} · ${Math.round(clip.fps)}fps · ${clip.codec.split('.')[0]}`;
    $('durTime').textContent = fmt(clip.duration);
    for (const id of ['btnPlay', 'btnPrev', 'btnNext', 'btnExport', 'btnStill']) $(id).disabled = false;
    clearHeatmap(); renderSegments(); syncControls(); updateSpeedInfo();
    recompute();
    uploadVideo(); draw();
    hideToast(); toast('Loaded. Next: Stabilise → Analyse motion', 2500);
  } catch (e) {
    console.error(e);
    toast(e.message || String(e), 6000);
    if (!state.clip) $('emptyState').style.display = '';
  }
}

const corr = (a, b) => {
  let ma = 0, mb = 0; for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; } ma /= a.length; mb /= b.length;
  let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; }
  return s / Math.sqrt(sa * sb + 1e-9);
};
// Browsers differ on whether <video> textures arrive already rotated. Measure it against a decoded frame.
async function detectPreviewOrientation(clip) {
  const disp = { texW: clip.dispW, texH: clip.dispH, dispW: clip.dispW, dispH: clip.dispH, rot: 0 };
  if (!clip.rotation) return disp;
  const alt = { texW: clip.codedW, texH: clip.codedH, dispW: clip.dispW, dispH: clip.dispH, rot: clip.rotation / 90 };
  try {
    await new Promise(r => { if (video.readyState >= 2 && !video.seeking) r(); else video.addEventListener('seeked', r, { once: true }); setTimeout(r, 800); });
    const ds = new Downscaler(48, 48);
    let ref = null;
    await decodeRange(clip, { fromUs: 0, toUs: 0, onFrame: f => { if (!ref) ref = ds.grab(f, frameInfo(clip, f)).slice(); f.close(); } });
    const a = ds.grab(video, disp).slice(), b = ds.grab(video, alt).slice();
    return corr(a, ref) >= corr(b, ref) ? disp : alt;
  } catch (e) { console.warn('orientation probe failed', e); return disp; }
}

function uploadVideo() {
  if (!state.clip || !state.previewInfo || video.readyState < 2) return;
  const [, ph] = renderer.size.length ? renderer.size : previewDims();
  const i = currentIndex(), m = state.plan ? state.plan.M.subarray(4 * i, 4 * i + 4) : [1, 0, 0, 0];
  const scale = Math.hypot(m[0], m[1]) * state.previewInfo.dispH / ph;
  renderer.upload(video, state.previewInfo, scale > 1.25);
}

// frame callbacks (preview)
function onVideoFrame(now, meta) {
  state.mediaTime = meta.mediaTime;
  uploadVideo(); draw(); updatePlayhead();
  if (state.playing) checkSegmentBounds();
  video.requestVideoFrameCallback(onVideoFrame);
}
if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) video.requestVideoFrameCallback(onVideoFrame);
else {
  const loop = () => { state.mediaTime = video.currentTime; if (state.clip) { uploadVideo(); draw(); updatePlayhead(); if (state.playing) checkSegmentBounds(); } requestAnimationFrame(loop); };
  requestAnimationFrame(loop);
}
video.addEventListener('seeked', () => {
  if (!state.playing) { state.mediaTime = video.currentTime; uploadVideo(); draw(); updatePlayhead(); }
});

function seek(t) {
  if (!state.clip) return;
  t = Math.max(0, Math.min(state.clip.duration - 1e-3, t));
  video.currentTime = t; state.mediaTime = t; updatePlayhead();
}

// ───────────── playback ─────────────
function checkSegmentBounds() {
  const t = video.currentTime, segs = state.segments;
  if (segs.some(s => t >= s.start - 0.02 && t < s.end)) return;
  const next = segs.find(s => s.start > t);
  if (next) video.currentTime = next.start;
  else { pause(); seek(segs[0].start); }
}
function play() {
  if (!state.clip) return;
  const t = video.currentTime;
  if (!state.segments.some(s => t >= s.start && t < s.end - 0.05)) video.currentTime = state.segments[0].start;
  video.playbackRate = state.speed;
  video.play().catch(() => {});
  state.playing = true; $('btnPlay').textContent = '⏸';
  const tickLoop = () => { if (!state.playing) return; checkSegmentBounds(); requestAnimationFrame(tickLoop); };
  requestAnimationFrame(tickLoop);
}
function pause() { video.pause(); state.playing = false; $('btnPlay').textContent = '▶'; }
$('btnPlay').onclick = () => (state.playing ? pause() : play());
video.addEventListener('ended', pause);
function step(d) {
  pause();
  const i = Math.max(0, Math.min(state.clip.times.length - 1, currentIndex() + d));
  seek(state.clip.times[i] + 0.0005);
}
$('btnPrev').onclick = () => step(-1);
$('btnNext').onclick = () => step(1);

$('btnMute').onclick = () => {
  state.muted = !state.muted; video.muted = state.muted;
  $('btnMute').classList.toggle('muted', state.muted);
  $('muteIcon').textContent = state.muted ? '🔇' : '🔊'; $('muteLabel').textContent = state.muted ? 'Muted' : 'Audio';
};
const cmp = $('btnCompare');
const setCompare = v => { state.compare = v; cmp.classList.toggle('held', v); $('compareTag').style.display = v ? 'block' : 'none'; draw(); };
cmp.addEventListener('pointerdown', e => { e.preventDefault(); setCompare(true); });
window.addEventListener('pointerup', () => { if (state.compare) setCompare(false); });
window.addEventListener('pointercancel', () => { if (state.compare) setCompare(false); });

// ───────────── timeline ─────────────
const segEl = $('segments'), track = $('track'), playhead = $('playhead');
function updatePlayhead() {
  if (!state.clip) return;
  playhead.style.left = `${Math.max(0, Math.min(100, (state.mediaTime / state.clip.duration) * 100))}%`;
  $('curTime').textContent = fmt(state.mediaTime);
}
function renderSegments() {
  segEl.innerHTML = '';
  const D = state.clip ? state.clip.duration : 0; if (!D) return;
  state.segments.forEach((seg, idx) => {
    const el = document.createElement('div');
    el.className = 'clip-segment' + (idx === state.sel ? ' selected' : '');
    el.style.left = `${(seg.start / D) * 100}%`;
    el.style.width = `${Math.max(((seg.end - seg.start) / D) * 100, 1.5)}%`;
    const hl = document.createElement('div'), hr = document.createElement('div');
    hl.className = hr.className = 'seg-handle';
    el.append(hl, hr);
    el.addEventListener('pointerdown', ev => {
      if (ev.target !== el) return;
      state.sel = idx; renderSegments();
    });
    handleDrag(hl, idx, true); handleDrag(hr, idx, false);
    segEl.appendChild(el);
  });
}
function handleDrag(h, idx, left) {
  h.addEventListener('pointerdown', ev => {
    ev.stopPropagation(); pause(); state.sel = idx;
    const rect = track.getBoundingClientRect(), D = state.clip.duration;
    h.setPointerCapture(ev.pointerId);
    const move = e => {
      const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)) * D, seg = state.segments[idx];
      if (left) { const lo = idx > 0 ? state.segments[idx - 1].end : 0; seg.start = Math.max(lo, Math.min(t, seg.end - 0.15)); seek(seg.start); }
      else { const hi = idx < state.segments.length - 1 ? state.segments[idx + 1].start : D; seg.end = Math.min(hi, Math.max(t, seg.start + 0.15)); seek(seg.end - 0.01); }
      renderSegments();
    };
    const up = () => { h.removeEventListener('pointermove', move); h.removeEventListener('pointerup', up); h.removeEventListener('pointercancel', up); invalidate(); };
    h.addEventListener('pointermove', move); h.addEventListener('pointerup', up); h.addEventListener('pointercancel', up);
  });
}
track.addEventListener('pointerdown', ev => {
  if (!state.clip || ev.target.classList.contains('seg-handle')) return;
  pause();
  const rect = track.getBoundingClientRect();
  const scrub = x => seek(Math.max(0, Math.min(1, (x - rect.left) / rect.width)) * state.clip.duration);
  scrub(ev.clientX);
  track.setPointerCapture(ev.pointerId);
  const move = e => scrub(e.clientX);
  const up = () => { track.removeEventListener('pointermove', move); track.removeEventListener('pointerup', up); };
  track.addEventListener('pointermove', move); track.addEventListener('pointerup', up);
});
$('btnSplit').onclick = () => {
  if (!state.clip) return;
  const t = state.mediaTime, i = state.segments.findIndex(s => t > s.start + 0.1 && t < s.end - 0.1);
  if (i < 0) return toast('Move the playhead inside a segment first', 1800);
  const s = state.segments[i];
  state.segments.splice(i, 1, { start: s.start, end: t }, { start: t, end: s.end });
  state.sel = i + 1; renderSegments(); invalidate();
};
$('btnDelete').onclick = () => {
  if (state.segments.length <= 1) return toast('Split first, then delete the bad piece', 1800);
  state.segments.splice(state.sel, 1); state.sel = Math.max(0, state.sel - 1);
  seek(state.segments[state.sel].start); renderSegments(); invalidate();
};
$('btnResetCuts').onclick = () => { if (!state.clip) return; state.segments = [{ start: 0, end: state.clip.duration }]; state.sel = 0; renderSegments(); invalidate(); };

function clearHeatmap() { const c = $('heatmap'); c.getContext('2d').clearRect(0, 0, c.width, c.height); }
function drawHeatmap() {
  const a = state.analysis, c = $('heatmap'), x = c.getContext('2d'); if (!a) return clearHeatmap();
  const n = a.sharp.length, sorted = Array.from(a.sharp).sort((p, q) => p - q), ref = sorted[Math.floor(n * 0.9)] || 1;
  x.clearRect(0, 0, c.width, c.height);
  const T = state.clip.times, D = state.clip.duration;
  for (let i = 0; i < n; i++) {
    const v = Math.min(1, a.sharp[i] / ref);
    x.fillStyle = !a.ok[i] ? '#ff4757' : v > 0.7 ? '#2ed573' : v > 0.45 ? '#ffa502' : '#ff4757';
    const x0 = (T[i] / D) * c.width, x1 = ((i + 1 < n ? T[i + 1] : D) / D) * c.width, hh = Math.max(3, v * c.height);
    x.fillRect(x0, c.height - hh, Math.max(1, x1 - x0), hh);
  }
}

// ───────────── tabs ─────────────
document.querySelectorAll('.dock-item').forEach(b => b.onclick = () => {
  document.querySelectorAll('.dock-item').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === b.dataset.tab));
  state.tab = b.dataset.tab; updateReticle();
});

// ───────────── stabilise controls ─────────────
function pills(groupId, attr, value) {
  document.querySelectorAll(`#${groupId} .pill-btn`).forEach(b => b.classList.toggle('active', b.dataset[attr] === String(value)));
}
function bindRange(id, get, set, label) {
  const el = $(id), lab = $('v-' + id);
  el.addEventListener('input', () => { set(parseFloat(el.value)); if (lab) lab.textContent = label(parseFloat(el.value)); invalidate(); persist(); });
  return () => { el.value = get(); if (lab) lab.textContent = label(get()); };
}
const syncers = [
  bindRange('sigma', () => state.stab.sigma, v => { state.stab.sigma = v; }, v => `${v.toFixed(2)} s`),
  bindRange('maxCrop', () => state.stab.maxCrop, v => { state.stab.maxCrop = v; }, v => `${Math.round(v * 100)}%`),
  bindRange('rot', () => state.frame.rot, v => { state.frame.rot = v; clampFramePan(); }, v => `${v.toFixed(1)}°`),
  bindRange('zoom', () => state.frame.zoom, v => { state.frame.zoom = v; clampFramePan(); }, v => `${v.toFixed(2)}×`),
];
$('stabRot').onchange = e => { state.stab.rotation = e.target.checked; invalidate(); persist(); };
$('stabZoom').onchange = e => { state.stab.zoom = e.target.checked; invalidate(); persist(); };
document.querySelectorAll('#stabMode .pill-btn').forEach(b => b.onclick = () => {
  state.stab.mode = b.dataset.mode; pills('stabMode', 'mode', state.stab.mode);
  $('sigma').disabled = state.stab.mode !== 'smooth'; invalidate(); persist();
});
document.querySelectorAll('#roiMode .pill-btn').forEach(b => b.onclick = () => {
  state.stab.roiMode = b.dataset.roi; pills('roiMode', 'roi', state.stab.roiMode);
  if (state.stab.roiMode === 'all') { const had = !!state.stab.roi; state.stab.roi = null; if (had && state.analysis) toast('Re-analyse to track the whole frame', 2000); }
  else toast('Tap the spot to hold still (substrate next to the spood works best)', 2600);
  updateReticle();
});

function syncControls() {
  syncers.forEach(f => f());
  $('stabRot').checked = state.stab.rotation; $('stabZoom').checked = state.stab.zoom;
  pills('stabMode', 'mode', state.stab.mode); pills('roiMode', 'roi', state.stab.roiMode);
  $('sigma').disabled = state.stab.mode !== 'smooth';
  pills('aspectGroup', 'aspect', state.frame.aspect); pills('speedGroup', 'speed', state.speed);
  lookSync.forEach(f => f());
}

function updateStabStatus() {
  const el = $('stabStatus'), a = state.analysis, p = state.plan;
  if (state.analysing) return;
  if (!a) { el.innerHTML = 'Not analysed yet. Analysis tracks hundreds of points at sub-pixel accuracy (~1–3× clip length on a phone).'; return; }
  const weak = a.ok.reduce((s, v) => s + (v ? 0 : 1), 0);
  const parts = [`<span class="good">Tracked ${a.ok.length - weak}/${a.ok.length} frames</span>`];
  if (weak) parts.push(`<span class="warn">${weak} weak (blur/no texture)</span>`);
  if (state.stab.mode !== 'off') {
    parts.push(`crop ${p.cropPct.toFixed(0)}%`);
    if (p.limitedFrames) parts.push(`<span class="warn">${p.limitedFrames} frames hit the crop limit, raise Max crop</span>`);
  } else parts.push('stabilisation off');
  if (state.stab.roi) parts.push('tracking tapped spot');
  el.innerHTML = parts.join(' · ');
}

$('btnAnalyse').onclick = async () => {
  if (!state.clip) return;
  if (state.analysing) { state.analysing.abort(); return; }
  pause();
  const ac = new AbortController(); state.analysing = ac;
  const btn = $('btnAnalyse'); btn.textContent = '✕ Cancel analysis';
  const t0 = performance.now();
  try {
    const res = await analyseClip(state.clip, {
      roi: state.stab.roi, signal: ac.signal,
      onProgress: p => { progress(p); $('stabStatus').textContent = `Tracking motion… ${Math.round(p * 100)}%`; },
    });
    if (res) {
      state.analysis = res;
      if (state.stab.mode === 'off') { state.stab.mode = 'smooth'; pills('stabMode', 'mode', 'smooth'); }
      drawHeatmap();
      toast(`Analysed in ${((performance.now() - t0) / 1000).toFixed(1)} s, hold ❖ to compare`, 2600);
    }
  } catch (e) { console.error(e); toast('Analysis failed: ' + e.message, 5000); }
  state.analysing = null; btn.textContent = '⚡ Re-analyse motion'; progress(null);
  invalidate(); updateStabStatus();
};

// ───────────── frame controls + gestures ─────────────
function clampFramePan() {
  const f = state.frame;
  [f.panX, f.panY] = clampPan(srcAspect(), outAspect(), f.quarter, f.rot * Math.PI / 180, f.zoom, f.panX, f.panY);
}
document.querySelectorAll('#aspectGroup .pill-btn').forEach(b => b.onclick = () => {
  state.frame.aspect = b.dataset.aspect; pills('aspectGroup', 'aspect', b.dataset.aspect); clampFramePan(); invalidate();
});
$('btnRotL').onclick = () => { state.frame.quarter = (state.frame.quarter + 3) % 4; clampFramePan(); invalidate(); };
$('btnRotR').onclick = () => { state.frame.quarter = (state.frame.quarter + 1) % 4; clampFramePan(); invalidate(); };
$('btnFrameReset').onclick = () => { Object.assign(state.frame, { rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 }); syncControls(); invalidate(); };

const pointers = new Map();
let gesture = null;
function canvasToOut(e) {
  const r = view.getBoundingClientRect(), Ao = r.width / r.height;
  return [((e.clientX - r.left) / r.width - 0.5) * Ao, (e.clientY - r.top) / r.height - 0.5, r];
}
view.addEventListener('pointerdown', e => {
  if (!state.clip) return;
  if (state.tab === 'tab-stab' && state.stab.roiMode === 'tap') {
    const [ox, oy] = canvasToOut(e), M = state.plan.M.subarray(4 * currentIndex(), 4 * currentIndex() + 4);
    state.stab.roi = { x: M[0] * ox - M[1] * oy + M[2], y: M[1] * ox + M[0] * oy + M[3], r: 0.2 };
    state.stab.roiScreen = [e.clientX, e.clientY];
    updateReticle(); toast('Spot set. Tap Analyse motion to apply', 2400);
    return;
  }
  if (state.tab !== 'tab-frame') return;
  view.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  startGesture();
});
function startGesture() {
  const pts = [...pointers.values()], f = state.frame;
  gesture = { pts, panX: f.panX, panY: f.panY, zoom: f.zoom, rot: f.rot };
  if (pts.length >= 2) { gesture.d = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]); gesture.a = Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]); }
}
view.addEventListener('pointermove', e => {
  if (!gesture || !pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  const pts = [...pointers.values()], f = state.frame, r = view.getBoundingClientRect();
  if (pts.length >= 2 && gesture.d) {
    const d = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]), a = Math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]);
    f.zoom = Math.max(1, Math.min(4, gesture.zoom * d / gesture.d));
    f.rot = Math.max(-45, Math.min(45, gesture.rot + (a - gesture.a) * 180 / Math.PI));
  } else {
    const dx = (pts[0][0] - gesture.pts[0][0]) / r.height, dy = (pts[0][1] - gesture.pts[0][1]) / r.height;
    const M = state.planRaw.M.subarray(0, 4); // framing-only map (same for every frame)
    f.panX = gesture.panX - (M[0] * dx - M[1] * dy);
    f.panY = gesture.panY - (M[1] * dx + M[0] * dy);
  }
  clampFramePan(); syncers.forEach(s => s()); invalidate();
});
const endPointer = e => {
  if (!pointers.has(e.pointerId)) return;
  pointers.delete(e.pointerId);
  if (pointers.size) startGesture(); else { gesture = null; recompute(); draw(); }
};
view.addEventListener('pointerup', endPointer);
view.addEventListener('pointercancel', endPointer);

function updateReticle() {
  const r = $('reticle'), s = state.stab;
  if (state.tab === 'tab-stab' && s.roiMode === 'tap' && s.roiScreen) {
    const vr = $('viewport').getBoundingClientRect();
    r.style.display = 'block'; r.style.left = `${s.roiScreen[0] - vr.left}px`; r.style.top = `${s.roiScreen[1] - vr.top}px`;
  } else r.style.display = 'none';
}

// ───────────── look ─────────────
const LOOK_UI = [
  ['cas', 'Sharpen (CAS)', 0, 1, 0.01, v => `${Math.round(v * 100)}%`],
  ['detail', 'Micro-detail', 0, 2.5, 0.01, v => `${Math.round(v * 100)}%`],
  ['radius', 'Detail radius', 0.6, 4, 0.1, v => `${v.toFixed(1)} px`],
  ['denoise', 'Bokeh smooth', 0, 1, 0.01, v => `${Math.round(v * 100)}%`],
  ['exposure', 'Exposure', -2, 2, 0.05, v => `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`],
  ['contrast', 'Contrast', -0.5, 0.6, 0.01, v => `${Math.round(v * 100)}`],
  ['shadows', 'Shadows', -0.5, 1, 0.01, v => `${Math.round(v * 100)}`],
  ['highlights', 'Highlights', -0.5, 1, 0.01, v => `${Math.round(v * 100)}`],
  ['saturation', 'Saturation', -1, 1, 0.01, v => `${Math.round(v * 100)}`],
  ['vibrance', 'Vibrance', -0.5, 1.5, 0.01, v => `${Math.round(v * 100)}`],
  ['warmth', 'Warmth', -1, 1, 0.01, v => `${Math.round(v * 100)}`],
  ['tint', 'Tint', -1, 1, 0.01, v => `${Math.round(v * 100)}`],
];
const lookSync = LOOK_UI.map(([key, label, min, max, st, f]) => {
  const row = document.createElement('div'); row.className = 'control-row';
  row.innerHTML = `<div class="control-label">${label} <b></b></div><input type="range" min="${min}" max="${max}" step="${st}" />`;
  const inp = row.querySelector('input'), b = row.querySelector('b');
  inp.addEventListener('input', () => { state.look[key] = parseFloat(inp.value); b.textContent = f(state.look[key]); pills('lookPresets', 'preset', ''); requestDraw(); persist(); });
  inp.addEventListener('dblclick', () => { state.look[key] = DEFAULT_LOOK[key]; inp.value = state.look[key]; b.textContent = f(state.look[key]); requestDraw(); persist(); });
  $('lookSliders').appendChild(row);
  return () => { inp.value = state.look[key]; b.textContent = f(state.look[key]); };
});
document.querySelectorAll('#lookPresets .pill-btn').forEach(b => b.onclick = () => {
  Object.assign(state.look, LOOK_PRESETS[b.dataset.preset]); pills('lookPresets', 'preset', b.dataset.preset);
  lookSync.forEach(f => f()); requestDraw(); persist();
});

// ───────────── speed ─────────────
document.querySelectorAll('#speedGroup .pill-btn').forEach(b => b.onclick = () => {
  state.speed = parseFloat(b.dataset.speed); pills('speedGroup', 'speed', b.dataset.speed);
  video.playbackRate = state.speed; updateSpeedInfo();
});
function updateSpeedInfo() {
  const el = $('speedInfo'); if (!state.clip) { el.textContent = 'Load a clip first.'; return; }
  const f = state.clip.fps, eff = f * state.speed;
  el.innerHTML = state.speed === 1 ? `Source ${f.toFixed(0)} fps.` :
    eff >= 29 ? `<span class="good">Real slow-mo:</span> ${f.toFixed(0)} fps source × ${state.speed} = ${eff.toFixed(0)} unique frames/s.`
      : `<span class="warn">${f.toFixed(0)} fps source × ${state.speed} = only ${eff.toFixed(1)} unique frames/s, so it will step.</span> Shoot at 60 or 120 fps for smooth slow-mo.`;
}

// ───────────── open ─────────────
const pick = () => { $('fileInput').value = ''; $('fileInput').click(); };
$('btnOpen').onclick = pick; $('btnOpen2').onclick = pick;
$('fileInput').onchange = e => loadFile(e.target.files && e.target.files[0]);

// ───────────── still ─────────────
$('btnStill').onclick = async () => {
  if (!state.clip) return;
  pause();
  const [w, h] = outputDims(4320);
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const R = new Renderer(c); R.setSize(w, h);
  const m = currentM();
  R.upload(video, state.previewInfo, Math.hypot(m[0], m[1]) * state.previewInfo.dispH / h > 1.25);
  R.render(m, state.look, { pxScale: 1, raw: state.compare });
  const blob = c.convertToBlob ? await c.convertToBlob({ type: 'image/png' }) : await new Promise(r => c.toBlob(r, 'image/png'));
  download(blob, `MacroCut_${stem()}_${fmt(state.mediaTime).replace(/[:.]/g, '-')}.png`);
  toast(`Saved ${w}×${h} PNG`, 1800);
};
const stem = () => (state.clip.name || 'clip').replace(/\.[^.]+$/, '');
function download(blob, name) {
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
}

// ───────────── export ─────────────
let exportAbort = null, exportResult = null;
function exportFps() {
  if (state.exp.fps !== 'auto') return parseInt(state.exp.fps, 10);
  const f = state.clip.fps;
  return state.speed === 1 ? Math.min(60, Math.max(24, Math.round(f))) : 30;
}
function renderExportOptions() {
  const g = $('resGroup'); g.innerHTML = '';
  const seen = new Set();
  for (const k of ['source', '1080', '720']) {
    const [w, h] = outputDims(RES[k]);
    if (seen.has(`${w}x${h}`)) { if (state.exp.res === k) state.exp.res = 'source'; continue; }
    seen.add(`${w}x${h}`);
    const b = document.createElement('button'); b.className = 'pill-btn' + (state.exp.res === k ? ' active' : '');
    b.textContent = k === 'source' ? `Full ${w}×${h}` : `${w}×${h}`;
    b.onclick = () => { state.exp.res = k; persist(); renderExportOptions(); };
    g.appendChild(b);
  }
  pills('fpsGroup', 'fps', state.exp.fps);
  $('expAudio').checked = state.exp.audio;
  $('expAudio').disabled = state.speed !== 1 || !state.clip.audio;
  const total = state.segments.reduce((s, x) => s + (x.end - x.start), 0) / state.speed;
  $('exportSummary').textContent = `${total.toFixed(1)} s at ${exportFps()} fps · ${state.analysis && state.stab.mode !== 'off' ? `stabilised (${state.stab.mode === 'lock' ? 'tripod lock' : 'smooth'})` : 'not stabilised'} · speed ${state.speed}×`;
}
document.querySelectorAll('#fpsGroup .pill-btn').forEach(b => b.onclick = () => { state.exp.fps = b.dataset.fps; persist(); renderExportOptions(); });
$('expAudio').onchange = e => { state.exp.audio = e.target.checked; persist(); };
function setExportUI(mode) {
  $('exportOptions').style.display = mode === 'options' ? 'flex' : 'none';
  $('exportProgress').style.display = mode === 'options' ? 'none' : 'flex';
  $('btnExportGo').style.display = mode === 'options' ? '' : 'none';
  $('btnSave').style.display = mode === 'done' ? '' : 'none';
  $('btnShare').style.display = mode === 'done' && navigator.canShare ? '' : 'none';
  $('btnExportCancel').textContent = mode === 'running' ? 'Cancel' : 'Close';
}
$('btnExport').onclick = () => {
  if (!state.clip) return;
  pause(); if (!state.analysis) toast('Tip: run Stabilise → Analyse motion first', 2500);
  if (planDirty) recompute();
  setExportUI('options'); renderExportOptions(); $('exportSheet').classList.add('open');
};
$('btnExportCancel').onclick = () => {
  if (exportAbort) { exportAbort.abort(); return; }
  $('exportSheet').classList.remove('open');
};
$('btnExportGo').onclick = async () => {
  const [w, h] = exportDims(), fps = exportFps();
  setExportUI('running'); $('exportBar').style.width = '0';
  $('exportMsg').textContent = `Rendering ${w}×${h} @ ${fps} fps…`;
  exportAbort = new AbortController();
  const t0 = performance.now();
  try {
    if (planDirty) recompute();
    const res = await exportClip(state.clip, {
      segments: state.segments.slice().sort((a, b) => a.start - b.start), speed: state.speed, fps, outW: w, outH: h,
      M: state.plan.M, look: { ...state.look }, audio: state.exp.audio,
    }, {
      signal: exportAbort.signal,
      onProgress: p => { $('exportBar').style.width = `${Math.round(p * 100)}%`; $('exportMsg').textContent = `Rendering ${w}×${h} @ ${fps} fps… ${Math.round(p * 100)}%`; },
    });
    if (!res) { setExportUI('options'); exportAbort = null; return; }
    exportResult = { blob: res.blob, name: `MacroCut_${stem()}.mp4` };
    $('exportMsg').textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(0)} s: ${(res.blob.size / 1e6).toFixed(1)} MB, ${res.frames} frames, ${res.codec.startsWith('avc') ? 'H.264' : 'VP9'}${res.hasAudio ? ' + audio' : ''}.`;
    setExportUI('done');
  } catch (e) {
    console.error(e); $('exportMsg').textContent = 'Export failed: ' + (e.message || e); setExportUI('error');
  }
  exportAbort = null;
};
$('exportSheet').addEventListener('click', e => { if (e.target === $('exportSheet') && !exportAbort) $('exportSheet').classList.remove('open'); });
$('btnSave').onclick = () => exportResult && download(exportResult.blob, exportResult.name);
$('btnShare').onclick = async () => {
  if (!exportResult) return;
  const file = new File([exportResult.blob], exportResult.name, { type: 'video/mp4' });
  try { if (navigator.canShare({ files: [file] })) await navigator.share({ files: [file], title: exportResult.name }); else toast('Sharing not supported here, use Save', 2000); }
  catch (e) { if (e.name !== 'AbortError') toast('Share failed: ' + e.message, 2500); }
};

// ───────────── boot ─────────────
syncControls();
updateSpeedInfo();
if (!('VideoDecoder' in window)) toast('This browser lacks WebCodecs. Use Chrome on Android/desktop.', 8000);
window.addEventListener('resize', () => updateReticle());
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
window.__macrocut = { state, loadFile, recompute }; // debugging / tests
