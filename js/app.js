import { openClip, decodeRange, withTimeout } from './media.js';
import { Renderer } from './gpu.js';
import { KEYS, LIMITS, PRESETS, baseOf, defaultLook, cloneLook, normLook, resolveLook, applyAdj, oneSided, hasTweaks } from './look.js';
import { computeStabilisation, clampPan, frameAt, suggestCuts, lookAt, speedAt, segmentAt } from './core.js';
import { analyseClip, exportClip, frameInfo, stackStill } from './engine.js';
import { clipKey, loadProject, saveProject, deleteProject, savedLooks, setSavedLooks, saveLastExport, loadLastExport, safeCopy } from './store.js';
import { wasHidden, resetHidden } from './media.js';

const $ = id => document.getElementById(id);
const video = $('video'), view = $('view');

// ───────────── state ─────────────
const state = {
  clip: null, segments: [], sel: 0,
  analysis: null, analysing: null,
  stab: { on: false, level: 'medium', maxCrop: 0.22, rotation: true, zoom: true, roiMode: 'all', roi: null },
  frame: { aspect: 'native', rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 },
  look: defaultLook(),
  speed: 1, scope: 'part', smoothSlow: true,
  exp: { res: 'source', fps: 'auto', audio: true, clean: false },
  muted: false, playing: false, compare: false,
  mediaTime: 0, plan: null, planRaw: null, previewInfo: null, tab: 'tab-trim',
};
// Plain-language strength levels → engine settings
const LEVELS = {
  mild: { mode: 'smooth', sigma: 0.35, maxCrop: 0.12, label: 'Mild', desc: 'Takes out the jitter and keeps your camera moves. Least zoom-in.' },
  medium: { mode: 'smooth', sigma: 1.0, maxCrop: 0.22, label: 'Medium', desc: 'Smooth and steady, still follows where you point. Good default.' },
  max: { mode: 'lock', sigma: 1.0, maxCrop: 0.35, label: 'Max', desc: 'Holds the shot as still as possible, like a tripod. Zooms in the most.' },
};
const STORE = 'macrocut.v5';
try {
  const s = JSON.parse(localStorage.getItem(STORE) || 'null');
  if (s) {
    Object.assign(state.stab, s.stab || {}, { roi: null, roiMode: 'all' });
    Object.assign(state.exp, s.exp || {});
    if (typeof s.smoothSlow === 'boolean') state.smoothSlow = s.smoothSlow;
  }
} catch (e) { /* storage unavailable */ }
const persist = () => { scheduleSave(); try { localStorage.setItem(STORE, JSON.stringify({ stab: state.stab, exp: state.exp, smoothSlow: state.smoothSlow })); } catch (e) { /* ignore */ } };

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
    C: withStab && state.analysis && st.on ? state.analysis.C : null,
    times: c.times, kept: keptMask(), mode: LEVELS[st.level].mode, sigma: LEVELS[st.level].sigma, strength: 1,
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
  updateStabStatus(); updateStabChip(); drawMotionGraph();
  planDirty = false;
}
function invalidate() { planDirty = true; requestDraw(); scheduleSave(); }

let drawQueued = false;
function requestDraw() {
  if (drawQueued) return; drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}
function currentIndex() { return frameAt(state.clip.times, state.mediaTime); }
// ── per-part settings: Look and Speed can differ per part of the timeline ──
function partMode() { return state.scope === 'part' && state.segments.length > 1; }
function activeSeg() { const k = segmentAt(state.segments, state.mediaTime); return k >= 0 ? k : Math.min(state.sel, state.segments.length - 1); }
// Looks are stored as preset + tweaks; the renderer wants plain numbers.
function resolvedLookAt(segs, base, t) {
  return lookAt(segs.map(x => x.look ? { start: x.start, end: x.end, look: resolveLook(x.look) } : x), resolveLook(base), t);
}
function lookAtTime(t) { return resolvedLookAt(state.segments, state.look, t); }
function effLook() { if (!partMode()) return state.look; const sg = state.segments[activeSeg()]; return (sg && sg.look) || state.look; }
function editableLook() {
  if (!partMode()) return state.look;
  const sg = state.segments[activeSeg()];
  if (!sg.look) { sg.look = cloneLook(state.look); renderSegments(); }
  return sg.look;
}
function setEditableLook(L) {
  if (!partMode()) { state.look = L; return; }
  state.segments[activeSeg()].look = L; renderSegments();
}
function effSpeed() { if (!partMode()) return state.speed; const sg = state.segments[activeSeg()]; return sg && sg.speed != null ? sg.speed : state.speed; }
function anySlow() { return state.segments.some(sg => Math.abs((sg.speed != null ? sg.speed : state.speed) - 1) > 1e-6); }
function syncPartControls() { syncLookUI(); speedSync(); renderScope(); updateSpeedInfo(); }
function renderScope() {
  const multi = state.segments.length > 1, k = activeSeg(), sg = state.segments[k];
  document.querySelectorAll('[data-scope-bar]').forEach(bar => {
    bar.classList.toggle('show', multi);
    bar.querySelectorAll('.pill-btn').forEach(b => b.classList.toggle('active', b.dataset.scope === state.scope));
    const isLook = !!bar.closest('#tab-look'), own = sg && (isLook ? sg.look : sg.speed != null);
    bar.querySelector('.scope-hint').textContent = state.scope === 'part'
      ? `Editing part ${k + 1} of ${state.segments.length}: the one under the playhead. Move the playhead to edit another part.`
      : `Editing the whole clip.${state.segments.some(x => isLook ? x.look : x.speed != null) ? ' Parts you changed separately keep their own settings.' : ''}`;
    bar.querySelector('.scope-reset').style.display = state.scope === 'part' && own ? '' : 'none';
  });
}
document.querySelectorAll('[data-scope-bar]').forEach(bar => {
  bar.querySelectorAll('.pill-btn').forEach(b => b.onclick = () => { state.scope = b.dataset.scope; syncPartControls(); });
  bar.querySelector('.scope-reset').onclick = () => {
    const sg = state.segments[activeSeg()];
    if (bar.closest('#tab-look')) delete sg.look; else delete sg.speed;
    renderSegments(); syncPartControls(); requestDraw(); applyPreviewRate();
  };
});
function applyPreviewRate() {
  if (!state.clip) return;
  const r = speedAt(state.segments, state.speed, video.currentTime);
  if (Math.abs(video.playbackRate - r) > 1e-3) video.playbackRate = r;
}
function currentM() {
  const i = currentIndex(), P = state.compare ? state.planRaw : state.plan;
  return P.M.subarray(4 * i, 4 * i + 4);
}
function draw() {
  if (!state.clip || !renderer || !renderer.srcInfo || (!state.plan && !planDirty)) return;
  state.draws = (state.draws || 0) + 1;
  if (planDirty) recompute();
  const [, eh] = exportDims();
  renderer.render(currentM(), lookAtTime(state.mediaTime), { raw: state.compare, pxScale: renderer.size[1] / eh });
}

// ───────────── loading ─────────────
// Steps run one at a time, each with a time limit, so a stuck step reports itself instead of hanging.
// The hardware decoder is never asked to serve the analysis decoder and the preview at the same time.
let videoURL = null;
function waitVideo(event, ms) {
  return new Promise((res, rej) => {
    const ok = () => { clean(); res(); }, bad = () => { clean(); rej(new Error('The browser could not play this file' + (video.error ? ` (code ${video.error.code})` : ''))); };
    const t = setTimeout(() => { clean(); rej(new Error('timeout')); }, ms);
    const clean = () => { clearTimeout(t); video.removeEventListener(event, ok); video.removeEventListener('error', bad); };
    video.addEventListener(event, ok, { once: true }); video.addEventListener('error', bad, { once: true });
  });
}
async function loadFile(file) {
  if (!file) return;
  if (state.analysing) state.analysing.abort();
  let stepName = 'reading the file';
  const step = (name, msg) => { stepName = name; toast(msg); };
  try {
    $('emptyState').style.display = 'none';
    pause();
    state.previewInfo = null; state.clip = null; state.plan = state.planRaw = null; state.analysis = null;
    video.removeAttribute('src'); video.load();
    const original = file;
    step('copying', 'Getting the clip ready…');
    file = await safeCopy(original, p => toast(`Getting the clip ready… ${p}%`));
    const clip = await openClip(file, msg => toast(msg));
    state.analysis = null; state.stab.roi = null; state.stab.on = false;
    state.segments = [{ start: 0, end: clip.duration }]; state.sel = 0; clearUndo();
    state.look = defaultLook();   // every new clip starts at Natural with all sliders at 0
    $('shotsGrid').innerHTML = ''; $('btnShotsSaveAll').style.display = 'none';
    state.frame = { aspect: 'native', rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 };
    // 1. reference thumbnail from the decoder (only needed for rotated or HDR clips), before <video> exists
    let probe = null;
    if (clip.rotation || clip.hdr) {
      step('checking colour', clip.hdr ? 'HDR clip: checking colour…' : 'Checking orientation…');
      try { probe = await withTimeout(referenceThumb(clip), 12000, 'colour check'); }
      catch (e) { console.warn('reference thumb failed', e); }
    }
    // 2. preview
    step('loading the preview', 'Loading preview…');
    if (videoURL) URL.revokeObjectURL(videoURL);
    videoURL = URL.createObjectURL(file);
    video.muted = true; video.src = videoURL;
    try { await waitVideo('loadeddata', 20000); }
    catch (e) { throw new Error(e.message === 'timeout' ? `This browser didn't start the preview for this ${clip.codec.split('.')[0]} video within 20 s.` : e.message); }
    video.currentTime = 0;
    await waitVideo('seeked', 3000).catch(() => {});
    // 3. match the preview's colour/orientation handling to the decoder's
    step('matching colour', 'Matching preview colour…');
    state.previewInfo = probe ? matchPreview(clip, probe) : defaultPreview(clip);
    video.muted = state.muted;
    state.clip = clip;
    state.key = clipKey(original);
    let restored = null;
    try { restored = await withTimeout(loadProject(state.key), 4000, 'saved edit'); } catch (e) { console.warn('no saved edit', e); }
    if (restored) { try { applyProject(restored); } catch (e) { console.warn('saved edit unreadable', e); restored = null; } }
    $('metaBadge').textContent = `${clip.dispW}×${clip.dispH} · ${Math.round(clip.fps)}fps${clip.hdr ? ' · HDR' : ''}`;
    $('durTime').textContent = fmt(clip.duration);
    for (const id of ['btnPlay', 'btnPrev', 'btnNext', 'btnExport', 'btnStill']) $(id).disabled = false;
    clearHeatmap(); recompute(); renderSegments(); syncControls(); updateSpeedInfo();
    uploadVideo(); draw(); resetHistory();
    if (state.analysis) drawHeatmap();
    hideToast();
    if (restored) toast(`Welcome back: your edit from ${new Date(restored.savedAt).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })} is restored. (Trim → Start this clip over to reset.)`, 5000);
    else toast(clip.hdr ? 'Loaded. HDR colour is converted automatically.' : 'Loaded', 2500);
    $('saveState').textContent = restored ? '✓ Restored from this phone. Changes keep saving automatically.' : 'Your edits save automatically on this phone. Open the same clip again to carry on.';
  } catch (e) {
    console.error(e);
    const hevc = state.clip === null && /hevc|hvc|hev1/i.test(e.message + (file.type || ''));
    toast(`Couldn't open this clip (while ${stepName}): ${e.message || e}` + (hevc ? ' Tip: in Camera settings turn off "High efficiency video" and "HDR video".' : ''), 9000);
    if (!state.clip) $('emptyState').style.display = '';
  }
}

function defaultPreview(clip) {
  return { texW: clip.dispW, texH: clip.dispH, dispW: clip.dispW, dispH: clip.dispH, rot: 0, hdr: 0 };
}
function thumbRenderer(clip) {
  const tw = 64, th = Math.max(8, Math.round(64 * clip.dispH / clip.dispW));
  const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(tw, th) : Object.assign(document.createElement('canvas'), { width: tw, height: th });
  const R = new Renderer(cv); R.setSize(tw, th); return R;
}
// First frame through the decoder, rendered the way analysis/export will see it.
async function referenceThumb(clip) {
  const R = thumbRenderer(clip);
  try {
    let ref = null;
    await decodeRange(clip, { fromUs: 0, toUs: 0, stallMs: 8000, onFrame: f => { if (!ref) { R.upload(f, frameInfo(clip, f)); ref = R.thumb().slice(); } f.close(); } });
    return ref;
  } finally { R.dispose(); }
}
// Browsers differ on whether <video> textures arrive rotated and whether HDR is colour-managed on upload:
// try each interpretation of the preview frame and keep the one that matches the decoder's.
function matchPreview(clip, ref) {
  const base = defaultPreview(clip), cands = [base];
  if (clip.rotation) cands.push({ texW: clip.codedW, texH: clip.codedH, dispW: clip.dispW, dispH: clip.dispH, rot: clip.rotation / 90, hdr: 0 });
  if (clip.hdr) for (const c of cands.slice()) cands.push({ ...c, hdr: clip.hdr === 'hlg' ? 1 : 2 });
  const R = thumbRenderer(clip);
  try {
    let best = base, bestD = Infinity;
    for (const c of cands) {
      R.upload(video, c); const t = R.thumb();
      let d = 0; for (let i = 0; i < t.length; i += 4) d += Math.abs(t[i] - ref[i]) + Math.abs(t[i + 1] - ref[i + 1]) + Math.abs(t[i + 2] - ref[i + 2]);
      if (d < bestD) { bestD = d; best = c; }
    }
    return best;
  } catch (e) { console.warn('preview match failed', e); return base; }
  finally { R.dispose(); }
}

// Keep the screen on during long jobs: a sleeping phone pauses the page and can kill an export.
let wakeLock = null, wakeWanted = 0;
async function acquireWake() { try { if (navigator.wakeLock && !wakeLock) { wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', () => { wakeLock = null; }); } } catch (e) { /* not allowed */ } }
async function keepAwake() {
  wakeWanted++; await acquireWake();
  return () => { wakeWanted = Math.max(0, wakeWanted - 1); if (!wakeWanted && wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; } };
}
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && wakeWanted) { acquireWake(); toast('Still working. It buzzes when it\'s done', 3000); }
});

// Free the preview's hardware decoder while analysis/export decode the file, then bring it back.
let parked = 0;
async function parkVideo() {
  const t = state.mediaTime;
  parked++; pause(); video.removeAttribute('src'); video.load();
  return async () => {
    video.src = videoURL;
    try { await waitVideo('loadeddata', 15000); video.currentTime = t; await waitVideo('seeked', 3000).catch(() => {}); }
    catch (e) { toast('Preview reload failed: ' + e.message, 4000); }
    finally { parked = Math.max(0, parked - 1); }
  };
}

// Phones reclaim the hardware video decoder (app switch, memory pressure, many tabs). The <video> then sits on
// "waiting" forever. Detect that and reattach the file at the same spot, then carry on as if nothing happened.
let reviving = null;
function reviveVideo(why) {
  if (reviving || parked || !state.clip || !videoURL) return reviving;
  const t = state.mediaTime, wasPlaying = state.playing;
  console.warn('reviving preview:', why);
  toast('Video stalled, restarting the preview…', 2500);
  reviving = (async () => {
    try {
      video.pause(); video.removeAttribute('src'); video.load();
      video.src = videoURL; video.muted = state.muted;
      await waitVideo('loadeddata', 15000);
      video.currentTime = t; await waitVideo('seeked', 4000).catch(() => {});
      showFrame(video.currentTime);
      if (wasPlaying && state.playing) { applyPreviewRate(); await video.play(); }
      hideToast(); return true;
    } catch (e) {
      if (e.name === 'AbortError') return true;
      pause();
      toast('The phone\'s video player stopped responding. Close other tabs that play video, or reload MacroCut; your edit is saved.', 8000);
      return false;
    } finally { reviving = null; playProgress.at = performance.now(); }
  })();
  return reviving;
}
const playProgress = { t: -1, at: 0, revives: 0 };
function watchPlayback() {
  if (!state.playing || reviving || parked) return;
  const now = performance.now(), t = video.currentTime;
  if (Math.abs(t - playProgress.t) > 1e-4) { playProgress.t = t; playProgress.at = now; return; }
  if (now - playProgress.at > 3000 && document.visibilityState === 'visible') {
    if (playProgress.revives >= 2) { pause(); toast('The phone\'s video player stopped responding. Close other tabs that play video, or reload MacroCut; your edit is saved.', 8000); return; }
    playProgress.revives++; reviveVideo('no progress for 3 s');
  }
}
video.addEventListener('error', () => { if (state.clip && !parked && video.getAttribute('src')) reviveVideo('media error ' + (video.error && video.error.code)); });
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && state.clip && !parked && video.getAttribute('src') && (video.readyState < 2 || video.error)) reviveVideo('decoder lost while in background');
  playProgress.at = performance.now();
});

function uploadVideo() {
  if (!state.clip || !state.previewInfo || video.readyState < 2) return;
  const [, ph] = renderer.size.length ? renderer.size : previewDims();
  const i = currentIndex(), m = state.plan ? state.plan.M.subarray(4 * i, 4 * i + 4) : [1, 0, 0, 0];
  const scale = Math.hypot(m[0], m[1]) * state.previewInfo.dispH / ph;
  renderer.upload(video, state.previewInfo, scale > 1.25);
}

// frame callbacks (preview). Re-arm first and never let an exception break the chain.
const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
let lastFrameAt = 0;
function showFrame(t) {
  state.mediaTime = t; lastFrameAt = performance.now();
  try { uploadVideo(); draw(); updatePlayhead(); if (state.playing) checkSegmentBounds(); }
  catch (e) { console.error('preview frame failed', e); }
}
function onVideoFrame(now, meta) {
  video.requestVideoFrameCallback(onVideoFrame);
  showFrame(meta.mediaTime);
}
if (hasRVFC) video.requestVideoFrameCallback(onVideoFrame);
// Watchdog: if frame callbacks stall (decoder hiccup, background tab, older browsers) keep drawing anyway.
(function watchdog() {
  if (state.clip && state.playing && video.readyState >= 2 && (!hasRVFC || performance.now() - lastFrameAt > 250)) showFrame(video.currentTime);
  requestAnimationFrame(watchdog);
})();
video.addEventListener('seeked', () => { if (!state.playing) showFrame(video.currentTime); });
let waitTimer = null;
video.addEventListener('waiting', () => { clearTimeout(waitTimer); waitTimer = setTimeout(() => { if (state.playing && video.readyState < 3) toast('Buffering…', 1200); }, 700); });
video.addEventListener('playing', () => clearTimeout(waitTimer));

// GPU context loss (phones do this under memory pressure): rebuild the preview renderer.
view.addEventListener('webglcontextlost', e => { e.preventDefault(); toast('Graphics reset, recovering…', 1500); });
view.addEventListener('webglcontextrestored', () => {
  try { renderer = new Renderer(view); if (state.clip) { renderer.setSize(...previewDims()); planDirty = true; showFrame(video.currentTime); } }
  catch (e) { toast('Graphics failed to recover, reload the page: ' + e.message, 6000); }
});

function seek(t) {
  if (!state.clip) return;
  t = Math.max(0, Math.min(state.clip.duration - 1e-3, t));
  video.currentTime = t; state.mediaTime = t; updatePlayhead();
  clearTimeout(seekGuard);
  seekGuard = setTimeout(() => { if (!state.playing && !parked && video.seeking && state.clip) reviveVideo('seek never completed'); }, 4000);
}
let seekGuard = null;

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
  applyPreviewRate();
  Object.assign(playProgress, { t: -1, at: performance.now(), revives: 0 });
  if (video.readyState < 2 || video.error) reviveVideo('not ready on play');
  else video.play().catch(e => {
    if (e.name === 'AbortError') return;   // play() cancelled by a pause/seek: harmless, not an error
    state.playing = false; $('btnPlay').textContent = '▶'; toast('Playback blocked: ' + e.message, 3000);
  });
  state.playing = true; $('btnPlay').textContent = '⏸';
  const tickLoop = () => { if (!state.playing) return; if (!reviving) checkSegmentBounds(); applyPreviewRate(); watchPlayback(); requestAnimationFrame(tickLoop); };
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
const setCompare = v => { state.compare = v; cmp.classList.toggle('held', v); updateStabChip(); draw(); };
// tap = toggle before/after, press-and-hold = peek at the original while held
let cmpDown = 0, cmpWas = false;
cmp.addEventListener('pointerdown', e => { e.preventDefault(); cmpDown = performance.now(); cmpWas = state.compare; setCompare(true); });
const cmpUp = () => {
  if (!cmpDown) return;
  const held = performance.now() - cmpDown > 350; cmpDown = 0;
  setCompare(held ? false : !cmpWas);
};
window.addEventListener('pointerup', cmpUp);
window.addEventListener('pointercancel', cmpUp);
function updateStabChip() {
  const tag = $('compareTag');
  if (!state.clip) { tag.style.display = 'none'; return; }
  const stab = state.analysis && state.stab.on;
  tag.style.display = 'block';
  tag.className = 'compare-tag' + (state.compare ? '' : stab ? ' on' : '');
  tag.textContent = state.compare ? 'ORIGINAL' : stab ? `STABILISED · ${LEVELS[state.stab.level].label.toUpperCase()}` : state.analysing ? 'MEASURING SHAKE…' : 'NOT STABILISED';
}

// ───────────── timeline ─────────────
const segEl = $('segments'), track = $('track'), playhead = $('playhead');
let lastActiveKey = '';
function updatePlayhead() {
  if (!state.clip) return;
  const pct = `${Math.max(0, Math.min(100, (state.mediaTime / state.clip.duration) * 100))}%`;
  playhead.style.left = pct; $('graphHead').style.left = pct;
  const k = segmentAt(state.segments, state.mediaTime), key = `${k}/${state.segments.length}`;
  if (k >= 0 && key !== lastActiveKey) { lastActiveKey = key; state.sel = k; renderSegments(); syncPartControls(); }
  $('curTime').textContent = fmt(state.mediaTime);
}
function renderSegments() {
  scheduleSave();
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
    const tags = [];
    if (seg.look) tags.push('🎨');
    if (seg.speed != null) tags.push(`${+seg.speed.toFixed(2)}×`);
    if (tags.length) { const bd = document.createElement('span'); bd.className = 'seg-badge'; bd.textContent = tags.join(' '); el.appendChild(bd); }
    el.addEventListener('pointerdown', ev => {
      if (ev.target !== el) return;
      state.sel = idx; renderSegments();
    });
    handleDrag(hl, idx, true); handleDrag(hr, idx, false);
    segEl.appendChild(el);
  });
  renderScope();
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
  const copy = x => { const o = { ...x }; if (x.look) o.look = cloneLook(x.look); return o; };
  state.segments.splice(i, 1, { ...copy(s), end: t }, { ...copy(s), start: t });
  state.sel = i + 1; renderSegments(); invalidate(); clearUndo(); syncPartControls();
  if (state.segments.length === 2) toast('Split. Look and Speed now change just the part under the playhead', 3200);
};
$('btnDelete').onclick = () => {
  if (state.segments.length <= 1) return toast('Split first, then delete the bad piece', 1800);
  state.segments.splice(state.sel, 1); state.sel = Math.max(0, state.sel - 1);
  seek(state.segments[state.sel].start); renderSegments(); invalidate(); clearUndo(); syncPartControls();
};
$('btnResetCuts').onclick = () => { if (!state.clip) return; state.segments = [{ start: 0, end: state.clip.duration }]; state.sel = 0; renderSegments(); invalidate(); clearUndo(); };
function clearUndo() { undoSegs = null; $('btnAutoTrim').textContent = '✨ Cut out the blurry & searching bits'; }

// Raw vs stabilised camera path (horizontal + vertical), so you can see what the analysis found.
function drawMotionGraph() {
  const wrap = $('graphWrap'), p = state.plan && state.plan.paths;
  if (!p || !state.clip) { wrap.style.display = 'none'; return; }
  wrap.style.display = '';
  const c = $('motionGraph'), x = c.getContext('2d'), W = c.width, H = c.height, T = state.clip.times, D = state.clip.duration;
  x.clearRect(0, 0, W, H);
  const rows = [[p.rx, p.sx], [p.ry, p.sy]];
  rows.forEach(([raw, sm], r) => {
    const top = r * H / 2, hh = H / 2;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < raw.length; i++) { lo = Math.min(lo, raw[i], sm[i]); hi = Math.max(hi, raw[i], sm[i]); }
    const span = Math.max(hi - lo, 0.01), Y = v => top + 4 + (1 - (v - lo) / span) * (hh - 8);
    const line = (arr, col, w) => {
      x.beginPath(); x.strokeStyle = col; x.lineWidth = w;
      for (let i = 0; i < arr.length; i++) { const px = T[i] / D * W, py = Y(arr[i]); i ? x.lineTo(px, py) : x.moveTo(px, py); }
      x.stroke();
    };
    line(raw, '#6b7585', 1.5); line(sm, '#00d2d3', 2.5);
    x.fillStyle = '#8b95a5'; x.font = '18px sans-serif'; x.fillText(r ? '↕' : '↔', 6, top + 22);
  });
  x.strokeStyle = '#232934'; x.beginPath(); x.moveTo(0, H / 2); x.lineTo(W, H / 2); x.stroke();
}

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
  bindRange('maxCrop', () => state.stab.maxCrop, v => { state.stab.maxCrop = v; }, v => `${Math.round(v * 100)}%`),
  bindRange('rot', () => state.frame.rot, v => { state.frame.rot = v; clampFramePan(); }, v => `${v.toFixed(1)}°`),
  bindRange('zoom', () => state.frame.zoom, v => { state.frame.zoom = v; clampFramePan(); }, v => `${v.toFixed(2)}×`),
  bindRange('speed', () => effSpeed(), v => { setSpeed(v); }, v => speedLabel(v)),
];
$('stabRot').onchange = e => { state.stab.rotation = e.target.checked; invalidate(); persist(); };
$('stabZoom').onchange = e => { state.stab.zoom = e.target.checked; invalidate(); persist(); };
$('stabOn').onchange = e => setStabOn(e.target.checked);
document.querySelectorAll('#stabLevel .pill-btn').forEach(b => b.onclick = () => {
  state.stab.level = b.dataset.level; state.stab.maxCrop = LEVELS[b.dataset.level].maxCrop;
  syncControls(); persist();
  if (!state.stab.on) setStabOn(true); else invalidate();
});
document.querySelectorAll('#roiMode .pill-btn').forEach(b => b.onclick = () => {
  state.stab.roiMode = b.dataset.roi; pills('roiMode', 'roi', state.stab.roiMode);
  if (state.stab.roiMode === 'all') { const had = !!state.stab.roi; state.stab.roi = null; if (had && state.analysis) analyse(); }
  else toast('Now tap the spot on the picture that should stay still', 2600);
  updateReticle();
});
$('btnAnalyse').onclick = () => analyse();

function setStabOn(on) {
  state.stab.on = on; $('stabOn').checked = on;
  if (on && !state.analysis && state.clip) analyse();
  if (!on && state.analysing) state.analysing.abort();
  invalidate(); updateStabStatus(); updateStabChip(); persist();
}

function syncControls() {
  syncers.forEach(f => f());
  $('stabRot').checked = state.stab.rotation; $('stabZoom').checked = state.stab.zoom;
  $('stabOn').checked = state.stab.on;
  pills('stabLevel', 'level', state.stab.level); pills('roiMode', 'roi', state.stab.roiMode);
  $('levelDesc').textContent = LEVELS[state.stab.level].desc;
  pills('aspectGroup', 'aspect', state.frame.aspect);
  syncLookUI(); speedSync(); renderScope();
  $('smoothSlow').checked = state.smoothSlow;
  updateStabStatus();
}

function updateStabStatus() {
  const el = $('stabStatus'), a = state.analysis, p = state.plan, st = state.stab;
  if (!p && state.clip) return;
  $('stabSub').textContent = st.on ? `On · ${LEVELS[st.level].label}` : 'Off';
  if (state.analysing) return;
  if (!state.clip) { el.textContent = 'Open a clip, then switch this on.'; return; }
  if (!st.on) { el.textContent = a ? 'Switch on to use the steadied version.' : 'Switch on to steady the shake. The first time, it measures the shake in your clip (takes about as long as the clip).'; return; }
  if (!a) { el.textContent = ''; return; }
  const parts = [`<b>✓ Steadied</b> · zoomed in ${Math.round(p.cropPct)}% to hide the moving edges.`];
  const weak = a.ok.reduce((s, v) => s + (v ? 0 : 1), 0);
  if (weak > a.ok.length * 0.02) parts.push('Some blurry moments couldn\'t be measured; they\'re smoothed over.');
  if (p.limitedFrames > a.ok.length * 0.1) parts.push(`<span class="warn">Your camera moved a lot in places, so those bits are a little less steady.${st.level !== 'max' ? ' Max (or more zoom-in) helps.' : ''}</span>`);
  if (st.roi) parts.push('Keeping the tapped spot still.');
  el.innerHTML = parts.join(' ');
}

async function analyse() {
  if (!state.clip) return;
  if (state.analysing) { state.analysing.abort(); return; }
  const ac = new AbortController(); state.analysing = ac;
  const t0 = performance.now();
  const el = $('stabStatus');
  updateStabChip();
  el.innerHTML = 'Measuring the shake… <a href="#" id="cancelAnalyse">cancel</a>';
  $('cancelAnalyse').onclick = ev => { ev.preventDefault(); ac.abort(); };
  const resume = await parkVideo();
  const awake = await keepAwake();
  try {
    const res = await analyseClip(state.clip, {
      roi: state.stab.roi, signal: ac.signal,
      onProgress: p => { progress(p); const s = el.querySelector('span') || el; el.innerHTML = `Measuring the shake… ${Math.round(p * 100)}% <a href="#" id="cancelAnalyse">cancel</a>`; $('cancelAnalyse').onclick = ev => { ev.preventDefault(); ac.abort(); }; void s; },
    });
    if (res) {
      state.analysis = res; drawHeatmap(); scheduleSave();
      toast(state.stab.on ? `✓ Steadied (${((performance.now() - t0) / 1000).toFixed(0)} s). Tap ❖ to compare before/after` : `✓ Clip measured (${((performance.now() - t0) / 1000).toFixed(0)} s)`, 3500);
    } else if (state.stab.on && !state.analysis) { state.stab.on = false; }
  } catch (e) {
    console.error(e); toast('Couldn\'t measure the shake: ' + e.message, 6000);
    if (!state.analysis) state.stab.on = false;
  }
  state.analysing = null; progress(null); awake();
  syncControls(); invalidate(); updateStabChip(); drawMotionGraph();
  await resume();
}

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
    updateReticle(); state.stab.on = true; analyse();
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
// Each slider is a tweak on top of the chosen preset: 0 = the preset exactly (where every slider starts),
// right = more, left = less, all the way to the setting's limit at each end. Effects that can't go below
// nothing (sharpening with no preset) only go right. Double-tap a slider to put it back to 0.
const LOOK_UI = [
  ['cas', 'Sharpen'], ['detail', 'Hair & texture'], ['radius', 'Detail size'], ['denoise', 'Smooth background'],
  ['exposure', 'Brightness'], ['contrast', 'Contrast'], ['shadows', 'Brighten shadows'], ['highlights', 'Calm bright areas'],
  ['saturation', 'Colour strength'], ['vibrance', 'Boost dull colours'], ['warmth', 'Cool ↔ warm'], ['tint', 'Green ↔ pink'],
];
const PRESET_NOTE = {
  natural: '<b>Natural</b>: light sharpening and gentle colour.',
  pop: '<b>Spood pop</b>: more hair detail, punchier colour and shine.',
  none: '<b>No preset</b>: the clip as filmed, plus only your slider changes.',
};
const sizeWord = v => v < 1.2 ? 'fine' : v < 2.2 ? 'medium' : 'coarse';
const signed = n => n > 0 ? `+${n}` : n < 0 ? `−${-n}` : '0';
function lookChanged() { syncLookUI(); renderScope(); requestDraw(); persist(); }
const lookRows = LOOK_UI.map(([key, label]) => {
  const row = document.createElement('div'); row.className = 'control-row look-row';
  row.innerHTML = `<div class="control-label">${label} <b></b></div><div class="slider-wrap"><input type="range" min="-100" max="100" step="1" value="0" /></div>`;
  const inp = row.querySelector('input'), b = row.querySelector('b');
  inp.addEventListener('input', () => {
    const L = editableLook();
    // Moving a slider while Off means "start from the clip as filmed": no preset, just this change.
    if (L.off) { L.off = false; L.preset = null; L.adj = {}; }
    L.adj[key] = parseInt(inp.value, 10) / 100; if (!L.adj[key]) delete L.adj[key];
    lookChanged();
  });
  // Double-tap to reset. Touch browsers don't fire dblclick on sliders, so time the taps ourselves.
  let lastTap = 0, lastX = 0;
  const reset = () => { const L = editableLook(); delete L.adj[key]; lookChanged(); };
  inp.addEventListener('pointerdown', e => {
    const now = performance.now();
    if (now - lastTap < 350 && Math.abs(e.clientX - lastX) < 40) { lastTap = 0; setTimeout(reset, 0); }
    else { lastTap = now; lastX = e.clientX; }
  });
  inp.addEventListener('dblclick', reset);
  $('lookSliders').appendChild(row);
  return { key, row, inp, b };
});
function syncLookUI() {
  const L = effLook(), base = baseOf(L), R = resolveLook(L);
  for (const { key, row, inp, b } of lookRows) {
    const one = oneSided(key, base), a = Math.round((L.adj[key] || 0) * 100), shown = one ? Math.max(0, a) : a;
    inp.min = one ? 0 : -100; inp.value = shown;
    row.classList.toggle('two-sided', !one);
    const lo = one ? 0 : -100, z = (0 - lo) / (100 - lo) * 100, v = (shown - lo) / (100 - lo) * 100;
    // track fill runs from the 0 mark to the thumb (thumb travel is inset by half its width)
    const px = q => `calc(11px + (100% - 22px) * ${q / 100})`;
    inp.style.setProperty('--f0', px(Math.min(z, v))); inp.style.setProperty('--f1', px(Math.max(z, v)));
    b.textContent = key === 'radius' ? sizeWord(R.radius) + (shown ? ` (${signed(shown)})` : '') : signed(shown);
    // Detail size only matters when hair detail or background smoothing is on.
    const idle = key === 'radius' && !L.off && R.detail < 1e-3 && R.denoise < 1e-3;
    row.classList.toggle('idle', idle);
  }
  $('lookSliders').classList.toggle('look-off', !!L.off);
  document.querySelectorAll('#lookPresets .pill-btn').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.preset === 'off' ? !!L.off : !L.off && L.preset === btn.dataset.preset));
  const tweaked = hasTweaks(L);
  $('lookStatus').innerHTML = L.off
    ? '<b>Off</b>: showing the clip exactly as filmed. Tap Off again to bring your look back.'
    : PRESET_NOTE[L.preset || 'none'] + (tweaked ? ' Plus your slider changes.' : ' Sliders start at 0 = the preset as-is.');
  $('btnLookReset').style.display = tweaked ? '' : 'none';
}
document.querySelectorAll('#lookPresets .pill-btn').forEach(btn => btn.onclick = () => {
  const L = editableLook(), p = btn.dataset.preset;
  if (p === 'off') L.off = !L.off;
  else { L.preset = !L.off && L.preset === p ? null : p; L.off = false; }   // tap the lit one again to switch it off
  lookChanged();
});
$('btnLookReset').onclick = () => { const L = editableLook(); L.adj = {}; L.off = false; lookChanged(); toast('Sliders back to 0', 1200); };

// ───────────── speed ─────────────
function speedLabel(v) { return Math.abs(v - 1) < 1e-6 ? 'normal' : `${+v.toFixed(2)}×`; }
function setSpeed(v) {
  scheduleSave();
  v = Math.round(v * 100) / 100;
  if (partMode()) { state.segments[activeSeg()].speed = v; renderSegments(); }
  else state.speed = v;
  applyPreviewRate(); speedSync(); updateSpeedInfo();
}
function speedSync() {
  const v = effSpeed();
  pills('speedGroup', 'speed', String(v));
  const el = $('speed'); if (el && +el.value !== v) el.value = v;
  $('v-speed').textContent = speedLabel(v);
}
document.querySelectorAll('#speedGroup .pill-btn').forEach(b => b.onclick = () => setSpeed(parseFloat(b.dataset.speed)));
$('smoothSlow').onchange = e => { state.smoothSlow = e.target.checked; updateSpeedInfo(); persist(); };
function updateSpeedInfo() {
  const el = $('speedInfo'); if (!state.clip) { el.textContent = 'Open a clip first.'; return; }
  const f = state.clip.fps, sp = effSpeed(), eff = f * sp, who = partMode() ? 'This part plays' : 'Plays';
  let msg;
  if (sp === 1) msg = `${who} at normal speed (your clip is ${Math.round(f)} frames per second).`;
  else if (sp > 1) msg = `${who} ${sp}× faster.`;
  else if (eff >= Math.min(60, f) - 0.5) msg = `<span class="good">Smooth slow motion.</span> ${who} with ${Math.round(eff)} real frames per second from your ${Math.round(f)} fps clip.`;
  else if (state.smoothSlow) msg = `<span class="good">Smooth slow motion is on.</span> Your clip has ${Math.round(eff)} real frames per second at this speed; the export creates the in-between frames from the motion. The preview still shows the steps. If something moves very fast (a jump), you may see a little smearing in those frames.`;
  else msg = `<span class="warn">Steppy:</span> your ${Math.round(f)} fps clip only has ${Math.round(eff)} real frames per second at this speed. Turn on Smooth slow motion, or film at 120 fps.`;
  if (anySlow()) msg += ' Sound is dropped when any part changes speed.';
  el.innerHTML = msg;
}

// ───────────── open ─────────────
const pick = () => { $('fileInput').value = ''; $('fileInput').click(); };
$('btnOpen').onclick = pick; $('btnOpen2').onclick = pick;
$('fileInput').onchange = e => loadFile(e.target.files && e.target.files[0]);

// ───────────── stills ─────────────
$('btnStill').onclick = () => { if (!state.clip) return; pause(); $('shotsSheet').classList.add('open'); };
$('btnShotsClose').onclick = () => { if (!shotsBusy) $('shotsSheet').classList.remove('open'); };
$('shotsSheet').addEventListener('click', e => { if (e.target === $('shotsSheet') && !shotsBusy) $('shotsSheet').classList.remove('open'); });
$('btnThisFrame').onclick = async () => {
  const [w, h] = outputDims(4320);
  const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const R = new Renderer(c); R.setSize(w, h);
  const m = currentM();
  R.upload(video, state.previewInfo, Math.hypot(m[0], m[1]) * state.previewInfo.dispH / h > 1.25);
  R.render(m, lookAtTime(state.mediaTime), { pxScale: 1, raw: state.compare });
  const blob = c.convertToBlob ? await c.convertToBlob({ type: 'image/png' }) : await new Promise(r => c.toBlob(r, 'image/png'));
  R.dispose();
  download(blob, `MacroCut_${stem()}_${fmt(state.mediaTime).replace(/[:.]/g, '-')}.png`);
  toast(`Saved ${w}×${h} still`, 1800);
};

// Wait for (or start) the shake/focus measurement that several tools rely on.
async function ensureAnalysis(why) {
  if (state.analysis) return true;
  if (!state.analysing) { toast(why, 2500); analyse(); }
  while (state.analysing) await new Promise(r => setTimeout(r, 200));
  return !!state.analysis;
}

// Find the sharpest, well-separated moments and render them as full-resolution stills.
let shotsBusy = false, shots = [];
$('btnFindBest').onclick = async () => {
  if (shotsBusy || !state.clip) return;
  shotsBusy = true; $('btnShotsSaveAll').style.display = 'none';
  const grid = $('shotsGrid'), msg = $('shotsMsg');
  grid.innerHTML = ''; shots = [];
  try {
    msg.textContent = 'Measuring focus across the clip first…';
    if (!await ensureAnalysis('Measuring focus across the clip…')) { msg.textContent = 'Couldn\'t measure the clip.'; return; }
    if (planDirty) recompute();
    const c = state.clip, a = state.analysis, kept = keptMask(), T = c.times;
    const total = state.segments.reduce((x, sg) => x + sg.end - sg.start, 0);
    const gap = Math.max(0.6, total / 12);
    const cand = [...T.keys()].filter(i => kept[i]).sort((i, j) => a.sharp[j] - a.sharp[i]);
    const picks = [];
    for (const i of cand) { if (picks.every(j => Math.abs(T[i] - T[j]) >= gap)) picks.push(i); if (picks.length === 6) break; }
    picks.sort((i, j) => i - j);
    const [w, h] = outputDims(4320);
    const cv = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
    const resume = await parkVideo(), awake = await keepAwake();
    const R = new Renderer(cv); R.setSize(w, h);
    try {
      for (let n = 0; n < picks.length; n++) {
        const i = picks[n], ts = Math.round(T[i] * 1e6);
        msg.textContent = `Rendering sharpest shots… ${n + 1}/${picks.length}`;
        let got = false;
        await decodeRange(c, { fromUs: ts, toUs: ts, onFrame: f => {
          if (!got && f.timestamp === ts) {
            const info = frameInfo(c, f), m = state.plan.M.subarray(4 * i, 4 * i + 4);
            R.upload(f, info, Math.hypot(m[0], m[1]) * info.dispH / h > 1.25); R.render(m, lookAtTime(T[i]), { pxScale: 1 }); got = true;
          }
          f.close();
        } });
        if (!got) continue;
        const blob = cv.convertToBlob ? await cv.convertToBlob({ type: 'image/png' }) : await new Promise(r => cv.toBlob(r, 'image/png'));
        const shot = { i, t: T[i], blob, name: `MacroCut_${stem()}_${fmt(T[i]).replace(/[:.]/g, '-')}.png` };
        shots.push(shot);
        await renderShotCard(shot, false);
      }
    } finally { R.dispose(); awake(); await resume(); }
    msg.textContent = shots.length ? `${shots.length} sharpest moments (${w}×${h}). Tap a picture to jump to it; 📚 Stack combines nearby frames for more in focus.` : 'No frames found.';
    if (shots.length) $('btnShotsSaveAll').style.display = '';
  } catch (e) { console.error(e); msg.textContent = 'Couldn\'t make the stills: ' + e.message; }
  finally { shotsBusy = false; }
};
// ── stills cards (plain or stacked) ──
const thumbOf = async blob => {
  const bmp = await createImageBitmap(blob);
  const tw = 360, th = Math.round(360 * bmp.height / bmp.width), c = document.createElement('canvas');
  c.width = tw; c.height = th; c.getContext('2d').drawImage(bmp, 0, 0, tw, th); bmp.close && bmp.close();
  return c;
};
async function renderShotCard(shot, prepend) {
  const grid = $('shotsGrid');
  const card = shot.card || document.createElement('div');
  card.className = 'shot'; card.innerHTML = '';
  shot.card = card;
  const after = await thumbOf(shot.blob), before = shot.before ? await thumbOf(shot.before) : null;
  let showing = 'after';
  const holder = document.createElement('div'); holder.style.position = 'relative'; holder.appendChild(after);
  const tag = document.createElement('span'); tag.className = 'shot-tag';
  if (before) {
    tag.textContent = `STACKED ×${shot.frames} · tap to compare`; holder.appendChild(tag);
    holder.onclick = () => {
      showing = showing === 'after' ? 'before' : 'after';
      holder.replaceChild(showing === 'after' ? after : before, holder.firstChild);
      tag.textContent = showing === 'after' ? `STACKED ×${shot.frames} · tap to compare` : 'SINGLE FRAME · tap to compare';
      tag.classList.toggle('before', showing === 'before');
    };
  } else {
    holder.title = 'Show this moment';
    holder.onclick = () => { $('shotsSheet').classList.remove('open'); seek(shot.t + 0.0005); };
  }
  const row = document.createElement('div'); row.className = 'shot-row';
  row.innerHTML = `<span>${fmt(shot.t)}</span>`;
  const btns = document.createElement('div'); btns.className = 'shot-btns';
  if (!before) {
    const st = document.createElement('button'); st.className = 'ghost'; st.textContent = '📚 Stack';
    st.onclick = () => stackShot(shot);
    btns.appendChild(st);
  }
  const sv = document.createElement('button'); sv.textContent = 'Save'; sv.onclick = () => download(shot.blob, shot.name);
  btns.appendChild(sv); row.appendChild(btns);
  card.append(holder, row);
  if (!card.parentNode) { if (prepend) grid.prepend(card); else grid.appendChild(card); }
}
async function stackShot(shot) {
  if (shotsBusy || !state.clip) return toast('One thing at a time: wait for the current stills job', 2000);
  shotsBusy = true;
  const msg = $('shotsMsg'), c = state.clip, i = shot.i;
  let prog = null;
  if (shot.card) { prog = document.createElement('div'); prog.className = 'shot-prog'; prog.textContent = 'Stacking…'; shot.card.appendChild(prog); }
  msg.textContent = 'Stacking: lining up the frames around this moment…';
  const resume = await parkVideo(), awake = await keepAwake();
  try {
    if (planDirty) recompute();
    const [w, h] = outputDims(4320);
    const res = await stackStill(c, i, {
      M: Array.from(state.plan.M.subarray(4 * i, 4 * i + 4)), look: lookAtTime(c.times[i]), outW: w, outH: h,
      onProgress: p => { const t = `Stacking… ${Math.round(p * 100)}%`; msg.textContent = t; if (prog) prog.textContent = t; },
    });
    if (!res) return;
    Object.assign(shot, { blob: res.blob, before: res.before, frames: res.frames, name: shot.name.replace(/\.png$/, '') + '_stacked.png' });
    if (!shots.includes(shot)) shots.unshift(shot);
    await renderShotCard(shot, true);
    $('btnShotsSaveAll').style.display = '';
    msg.textContent = `Stacked ${res.frames} frames into one ${w}×${h} still. Tap the picture to compare with the single frame.`;
  } catch (e) { console.error(e); msg.textContent = 'Couldn\'t stack: ' + e.message; if (prog) prog.remove(); }
  finally { awake(); await resume(); shotsBusy = false; }
}
$('btnStackHere').onclick = () => {
  if (!state.clip) return;
  const i = currentIndex();
  stackShot({ i, t: state.clip.times[i], name: `MacroCut_${stem()}_${fmt(state.clip.times[i]).replace(/[:.]/g, '-')}.png` });
};
$('btnShotsSaveAll').onclick = async () => {
  const files = shots.map(x => new File([x.blob], x.name, { type: 'image/png' }));
  try { if (navigator.canShare && navigator.canShare({ files })) { await navigator.share({ files, title: 'MacroCut stills' }); return; } }
  catch (e) { if (e.name === 'AbortError') return; }
  for (const x of shots) { download(x.blob, x.name); await new Promise(r => setTimeout(r, 400)); }
};

// ───────────── auto-trim ─────────────
let undoSegs = null;
$('btnAutoTrim').onclick = async () => {
  if (!state.clip) return;
  const btn = $('btnAutoTrim');
  if (undoSegs) {
    state.segments = undoSegs; undoSegs = null; state.sel = 0;
    btn.textContent = '✨ Cut out the blurry & searching bits'; renderSegments(); invalidate(); return;
  }
  pause();
  if (!await ensureAnalysis('Measuring focus and camera movement first…')) return;
  const a = state.analysis, T = state.clip.times;
  const segs = suggestCuts(T, a.sharp, a.ok, a.C);
  undoSegs = state.segments.map(x => ({ ...x }));
  state.segments = segs; state.sel = 0; renderSegments(); invalidate(); seek(segs[0].start);
  const kept = segs.reduce((x, sg) => x + sg.end - sg.start, 0);
  btn.textContent = '↩ Undo auto-cut';
  toast(`Kept the good ${kept.toFixed(1)} s of ${state.clip.duration.toFixed(1)} s (${segs.length} part${segs.length > 1 ? 's' : ''}). Adjust with the handles if you like.`, 4000);
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
  return Math.min(60, Math.max(24, Math.round(f)));
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
  $('expClean').checked = !!state.exp.clean;
  $('expAudio').disabled = anySlow() || !state.clip.audio;
  const total = state.segments.reduce((s, x) => s + (x.end - x.start) / (x.speed != null ? x.speed : state.speed), 0);
  const speeds = new Set(state.segments.map(x => x.speed != null ? x.speed : state.speed));
  $('exportSummary').textContent = `${total.toFixed(1)} s · ${exportFps()} fps · ${state.analysis && state.stab.on ? `steadied (${LEVELS[state.stab.level].label})` : 'not steadied'} · ${state.exp.clean ? 'extra clean · ' : ''}${speeds.size > 1 ? 'mixed speeds' : speedLabel([...speeds][0]) + ' speed'}${state.segments.some(x => x.look) ? ' · per-part looks' : ''}`;
}
document.querySelectorAll('#fpsGroup .pill-btn').forEach(b => b.onclick = () => { state.exp.fps = b.dataset.fps; persist(); renderExportOptions(); });
$('expAudio').onchange = e => { state.exp.audio = e.target.checked; persist(); };
$('expClean').onchange = e => { state.exp.clean = e.target.checked; persist(); renderExportOptions(); };
// ── never lose a finished video: it saves itself, stays one tap away, and survives closing or reloading ──
let lastExport = null;   // { blob, name, at }
loadLastExport().then(x => { if (x && x.blob && !lastExport) { lastExport = x; renderLastExport(); } }).catch(() => {});
function renderLastExport() {
  const box = $('lastExport');
  if (!lastExport) { box.style.display = 'none'; return; }
  box.style.display = 'flex';
  const when = new Date(lastExport.at).toLocaleString([], { weekday: 'short', hour: 'numeric', minute: '2-digit' });
  $('lastExportInfo').innerHTML = `<b>Your last video</b>: ${lastExport.name} · ${(lastExport.blob.size / 1e6).toFixed(1)} MB · ${when}`;
  $('btnLastShare').style.display = navigator.canShare ? '' : 'none';
}
$('btnLastSave').onclick = () => { if (lastExport) { download(lastExport.blob, lastExport.name); toast('Saving to your phone\'s Downloads', 2000); } };
$('btnLastShare').onclick = () => lastExport && shareVideo(lastExport.blob, lastExport.name);
async function shareVideo(blob, name) {
  const file = new File([blob], name, { type: 'video/mp4' });
  try { if (navigator.canShare && navigator.canShare({ files: [file] })) await navigator.share({ files: [file], title: name }); else toast('Sharing not supported here, use Save', 2000); }
  catch (e) { if (e.name !== 'AbortError') toast('Share failed: ' + e.message, 2500); }
}
// Tell the user it's finished: buzz, chime, and a phone notification if MacroCut isn't on screen.
let chimeCtx = null;
function prepareAlerts() {
  try { chimeCtx = chimeCtx || new (window.AudioContext || window.webkitAudioContext)(); chimeCtx.resume(); } catch (e) { /* no audio */ }
  try { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission().catch(() => {}); } catch (e) { /* not supported */ }
}
function finishedAlert(ok, msg) {
  try { navigator.vibrate && navigator.vibrate(ok ? [180, 90, 180] : [500]); } catch (e) { /* not allowed */ }
  try {
    if (chimeCtx) {
      const t = chimeCtx.currentTime;
      for (const [f, d] of ok ? [[880, 0], [1320, 0.16]] : [[330, 0]]) {
        const o = chimeCtx.createOscillator(), g = chimeCtx.createGain();
        o.frequency.value = f; g.gain.setValueAtTime(0.0001, t + d); g.gain.exponentialRampToValueAtTime(0.25, t + d + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + d + 0.5);
        o.connect(g).connect(chimeCtx.destination); o.start(t + d); o.stop(t + d + 0.55);
      }
    }
  } catch (e) { /* no audio */ }
  try {
    if (document.visibilityState === 'hidden' && 'Notification' in window && Notification.permission === 'granted' && navigator.serviceWorker)
      navigator.serviceWorker.ready.then(r => r.showNotification(ok ? 'MacroCut: video ready' : 'MacroCut: export stopped', { body: msg, tag: 'macrocut-export' })).catch(() => {});
  } catch (e) { /* not supported */ }
}

function setExportUI(mode) {
  if (mode === 'options') renderLastExport(); else $('lastExport').style.display = 'none';
  $('exportOptions').style.display = mode === 'options' ? 'flex' : 'none';
  $('exportProgress').style.display = mode === 'options' ? 'none' : 'flex';
  $('btnExportGo').style.display = mode === 'options' ? '' : 'none';
  $('btnSave').style.display = mode === 'done' ? '' : 'none';
  $('btnShare').style.display = mode === 'done' && navigator.canShare ? '' : 'none';
  $('btnExportCancel').textContent = mode === 'running' ? 'Cancel' : 'Close';
}
$('btnExport').onclick = () => {
  if (!state.clip) return;
  pause(); if (!state.stab.on) toast('Tip: switch on Stabilise first for a steadier video', 2500);
  if (state.analysing) return toast('Wait for the shake measurement to finish', 2000);
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
  prepareAlerts(); resetHidden();
  exportAbort = new AbortController();
  const t0 = performance.now();
  const resume = await parkVideo();
  const awake = await keepAwake();
  try {
    if (planDirty) recompute();
    const expSegs = state.segments.slice().sort((a, b) => a.start - b.start).map(x => ({ ...x, look: x.look ? resolveLook(x.look) : undefined }));
    const baseLook = resolveLook(state.look);
    const res = await exportClip(state.clip, {
      segments: expSegs, speed: state.speed, fps, outW: w, outH: h,
      M: state.plan.M, look: t => lookAt(expSegs, baseLook, t), audio: state.exp.audio, interp: state.smoothSlow, denoise: !!state.exp.clean,
    }, {
      signal: exportAbort.signal,
      onProgress: p => {
        const el = (performance.now() - t0) / 1000, left = p > 0.03 ? el * (1 - p) / p : null;
        $('exportBar').style.width = `${Math.round(p * 100)}%`;
        $('exportMsg').textContent = `Rendering ${w}×${h} @ ${fps} fps… ${Math.round(p * 100)}%${left != null ? ` · about ${left < 90 ? `${Math.max(5, Math.round(left / 5) * 5)} s` : `${Math.round(left / 60)} min`} left` : ''}. It saves itself and buzzes when done.`;
      },
    });
    if (!res) { setExportUI('options'); exportAbort = null; awake(); await resume(); return; }
    const d = new Date(), p2 = n => String(n).padStart(2, '0');
    const name = `MacroCut_${stem()}_${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-${p2(d.getHours())}${p2(d.getMinutes())}.mp4`;
    exportResult = { blob: res.blob, name };
    lastExport = { blob: res.blob, name, at: Date.now() };
    saveLastExport(lastExport).catch(e => console.warn('could not keep last export', e));
    download(res.blob, name);   // save straight away: no extra tap needed
    const secs = (performance.now() - t0) / 1000, T = res.timing;
    let where = '';
    if (T && T.total > 5000) {
      const out = Math.min(100, Math.round(100 * (T.encoder + T.gpu) / T.total));
      where = ` Time split: drawing + encoding frames ${out}%, reading the clip${res.interpolated || res.cleaned ? ' + working out motion' : ''} ${100 - out}%.`;
    }
    $('exportMsg').innerHTML = `<b>✓ Saved to your phone's Downloads</b> as ${name}. Done in ${secs < 90 ? `${secs.toFixed(0)} s` : `${(secs / 60).toFixed(1)} min`}: ${(res.blob.size / 1e6).toFixed(1)} MB, ${res.frames} frames, ${res.codec.startsWith('avc') ? 'H.264' : 'VP9'}${res.hasAudio ? ' + audio' : ''}${res.interpolated ? `, ${res.interpolated} in-between frames created` : ''}${res.cleaned ? ', extra clean' : ''}.${res.shortBy ? ` The phone stopped reading the clip near the end, so it finishes ${res.shortBy < 1 ? 'a moment' : `${res.shortBy.toFixed(1)} s`} early.` : ''} If it isn't in Downloads, tap Save again; it also stays under Export until your next video.${where}`;
    setExportUI('done');
    finishedAlert(true, `${name} is saved to Downloads.`);
  } catch (e) {
    console.error(e);
    const away = wasHidden, lost = e && (e.name === 'NotReadableError' || /could not be read/i.test(e.message || ''));
    $('exportMsg').textContent = lost
      ? 'MacroCut lost access to the video file (the phone took back its permission). Tap Close, tap 📂 and open the same clip again (your edit comes back), then Render again.'
      : away
      ? `Export stopped while MacroCut was off screen: the phone took the video hardware back (${e.message || e}). Tap Close, then Render again and keep MacroCut on screen; the phone can lie face-down, the screen stays on by itself.`
      : 'Export failed: ' + (e.message || e);
    setExportUI('error');
    finishedAlert(false, away ? 'Stopped while MacroCut was in the background. Open it and tap Render again.' : 'Export failed.');
  }
  exportAbort = null; awake();
  await resume();
};
$('exportSheet').addEventListener('click', e => { if (e.target === $('exportSheet') && !exportAbort) $('exportSheet').classList.remove('open'); });
$('btnSave').onclick = () => exportResult && download(exportResult.blob, exportResult.name);
$('btnShare').onclick = () => exportResult && shareVideo(exportResult.blob, exportResult.name);

// ───────────── undo / redo ─────────────
// Snapshots of the edit (cuts, per-part look/speed, framing, look, speed, stabilise settings). Every edit already
// passes through scheduleSave(), so it notes a change; the snapshot is committed once the edit settles (no finger
// down, ~0.45 s quiet), so one slider drag or pinch = one undo step. The shake measurement itself is never undone.
const HIST_MAX = 100;
let histCur = null, undoStack = [], redoStack = [], histTimer = null, restoring = false, fingersDown = 0;
function editState() {
  const st = state.stab;
  return JSON.stringify({
    segments: state.segments.map(x => ({ start: x.start, end: x.end, speed: x.speed, look: x.look })),
    frame: state.frame, look: state.look, speed: state.speed, smoothSlow: state.smoothSlow,
    stab: { on: st.on, level: st.level, maxCrop: st.maxCrop, rotation: st.rotation, zoom: st.zoom },
  });
}
function resetHistory() { clearTimeout(histTimer); undoStack = []; redoStack = []; histCur = state.clip ? editState() : null; updateUndoButtons(); }
function noteEdit() {
  if (restoring || !state.clip) return;
  clearTimeout(histTimer); histTimer = setTimeout(commitEdit, 450);
}
function commitEdit() {
  clearTimeout(histTimer);
  if (!state.clip || histCur == null) return;
  if (fingersDown) { histTimer = setTimeout(commitEdit, 250); return; }
  const now = editState();
  if (now === histCur) return;
  undoStack.push(histCur); if (undoStack.length > HIST_MAX) undoStack.shift();
  redoStack = []; histCur = now; updateUndoButtons();
}
function describeChange(a, b) {
  const A = JSON.parse(a), B = JSON.parse(b), eq = k => JSON.stringify(A[k]) === JSON.stringify(B[k]), what = [];
  if (!eq('segments')) {
    const sa = A.segments, sb = B.segments, cuts = sa.length !== sb.length || sa.some((x, i) => x.start !== sb[i].start || x.end !== sb[i].end);
    what.push(cuts ? 'trim' : sa.some((x, i) => JSON.stringify(x.look) !== JSON.stringify(sb[i].look)) ? 'look' : 'speed');
  }
  if (!eq('frame')) what.push('framing'); if (!eq('look')) what.push('look'); if (!eq('stab')) what.push('stabilise');
  if (!eq('speed') || !eq('smoothSlow')) what.push('speed');
  return [...new Set(what)].join(' + ') || 'edit';
}
function applyEditState(json) {
  const p = JSON.parse(json), wasOn = state.stab.on;
  restoring = true;
  try {
    state.segments = p.segments.map(x => { const o = { start: x.start, end: x.end }; if (x.speed != null) o.speed = x.speed; if (x.look) o.look = normLook(x.look); return o; });
    state.sel = Math.min(state.sel, state.segments.length - 1);
    state.frame = { ...p.frame }; state.look = normLook(p.look); state.speed = p.speed; state.smoothSlow = p.smoothSlow;
    Object.assign(state.stab, p.stab);
    if (undoSegs) clearUndo();   // the auto-cut button's own one-shot undo is superseded
    renderSegments(); syncControls(); syncPartControls(); applyPreviewRate();
    if (state.stab.on !== wasOn) setStabOn(state.stab.on); else invalidate();
    updateStabChip(); syncLookUI();
    const t = state.mediaTime;
    if (!state.segments.some(s => t >= s.start && t <= s.end)) seek(state.segments[0].start);
  } finally { restoring = false; }
  histCur = json;
  try { localStorage.setItem(STORE, JSON.stringify({ stab: state.stab, exp: state.exp, smoothSlow: state.smoothSlow })); } catch (e) { /* ignore */ }
  scheduleSave(); clearTimeout(histTimer);
}
function undo() {
  commitEdit();   // bank any edit still settling, so it's the one undone
  if (!undoStack.length) return toast('Nothing to undo', 1200);
  const prev = undoStack.pop(), label = describeChange(prev, histCur);
  redoStack.push(histCur); applyEditState(prev); updateUndoButtons();
  toast(`↶ Undid ${label}`, 1400);
}
function redo() {
  commitEdit();
  if (!redoStack.length) return toast('Nothing to redo', 1200);
  const next = redoStack.pop(), label = describeChange(histCur, next);
  undoStack.push(histCur); applyEditState(next); updateUndoButtons();
  toast(`↷ Redid ${label}`, 1400);
}
function updateUndoButtons() { $('btnUndo').disabled = !undoStack.length; $('btnRedo').disabled = !redoStack.length; }
$('btnUndo').onclick = undo; $('btnRedo').onclick = redo;
window.addEventListener('pointerdown', () => { fingersDown++; }, true);
const fingerUp = () => { fingersDown = Math.max(0, fingersDown - 1); };
window.addEventListener('pointerup', fingerUp, true); window.addEventListener('pointercancel', fingerUp, true);
window.addEventListener('blur', () => { fingersDown = 0; });
window.addEventListener('keydown', e => {
  if (!(e.ctrlKey || e.metaKey) || e.target.tagName === 'INPUT' && e.target.type === 'text') return;
  const k = e.key.toLowerCase();
  if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
  else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); redo(); }
});

// ───────────── saved edits ─────────────
let saveTimer = null;
function scheduleSave() {
  noteEdit();
  if (!state.clip || !state.key) return;
  clearTimeout(saveTimer); saveTimer = setTimeout(doSave, 900);
}
function snapshot() {
  const a = state.analysis;
  let ana = null;
  if (a) {
    const C = new Float64Array(a.C.length * 4);
    a.C.forEach((m, i) => C.set(m, 4 * i));
    ana = { C, ok: a.ok, sharp: a.sharp, inliers: a.inliers, w: a.w, h: a.h };
  }
  const st = state.stab;
  return {
    v: 1, savedAt: Date.now(), name: state.clip.name,
    segments: state.segments.map(x => ({ start: x.start, end: x.end, speed: x.speed, look: x.look ? cloneLook(x.look) : undefined })),
    frame: { ...state.frame }, look: cloneLook(state.look), speed: state.speed,
    stab: { on: st.on, level: st.level, maxCrop: st.maxCrop, rotation: st.rotation, zoom: st.zoom, roi: st.roi, roiMode: st.roiMode },
    analysis: ana,
  };
}
async function doSave() {
  if (!state.clip || !state.key || state.analysing) { if (state.analysing) scheduleSave(); return; }
  try {
    await saveProject(state.key, snapshot());
    $('saveState').textContent = `✓ Saved on this phone at ${new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}. Open the same clip again to carry on.`;
  } catch (e) { $('saveState').textContent = `Couldn't save on this phone (${e.message}). Your edit still exports fine.`; }
}
function applyProject(p) {
  const D = state.clip.duration;
  const segs = (p.segments || []).filter(x => x.end > x.start && x.start < D).map(x => ({ ...x, end: Math.min(x.end, D) }));
  if (segs.length) state.segments = segs;
  for (const x of state.segments) { if (x.speed == null) delete x.speed; if (!x.look) delete x.look; else x.look = normLook(x.look); }
  if (p.frame) state.frame = { ...state.frame, ...p.frame };
  if (p.look) state.look = normLook(p.look);
  if (typeof p.speed === 'number') state.speed = p.speed;
  if (p.stab) Object.assign(state.stab, p.stab);
  const a = p.analysis;
  if (a && a.C && a.C.length === state.clip.times.length * 4) {
    const C = []; for (let i = 0; i < a.C.length; i += 4) C.push([a.C[i], a.C[i + 1], a.C[i + 2], a.C[i + 3]]);
    state.analysis = { C, ok: a.ok, sharp: a.sharp, inliers: a.inliers, w: a.w, h: a.h };
  } else if (state.stab.on) state.stab.on = false;   // measurement missing: switch on again to re-measure
}
$('btnStartOver').onclick = async () => {
  if (!state.clip) return;
  pause();
  try { await deleteProject(state.key); } catch (e) { /* nothing saved */ }
  state.segments = [{ start: 0, end: state.clip.duration }]; state.sel = 0; clearUndo();
  state.frame = { aspect: 'native', rot: 0, quarter: 0, zoom: 1, panX: 0, panY: 0 };
  state.look = defaultLook(); state.speed = 1; state.stab.on = false; state.stab.roi = null; state.stab.roiMode = 'all';
  renderSegments(); syncControls(); updateStabChip(); invalidate();
  toast('Started over. The shake measurement is kept, so switching Stabilise on is instant.', 3500);
};

// ───────────── my looks ─────────────
function renderMyLooks() {
  const box = $('myLooks'); box.innerHTML = '';
  for (const [n, item] of savedLooks().entries()) {
    const b = document.createElement('button'); b.className = 'pill-btn'; b.textContent = item.name;
    let held = false, timer = null;
    b.onpointerdown = () => { held = false; timer = setTimeout(() => {
      held = true; const list = savedLooks(); list.splice(n, 1); setSavedLooks(list); renderMyLooks(); toast(`Deleted "${item.name}"`, 1800);
    }, 650); };
    b.onpointerup = b.onpointerleave = b.onpointercancel = () => clearTimeout(timer);
    b.onclick = () => {
      if (held) return;
      setEditableLook(normLook(item.look)); lookChanged();
      toast(`Applied "${item.name}"${partMode() ? ' to this part' : ''}`, 1500);
    };
    box.appendChild(b);
  }
  const add = document.createElement('button'); add.className = 'pill-btn'; add.textContent = '＋ Save current look';
  add.onclick = () => { $('saveLookRow').style.display = 'flex'; $('lookName').value = ''; $('lookName').focus(); };
  box.appendChild(add);
}
$('btnSaveLookGo').onclick = () => {
  const name = $('lookName').value.trim() || `Look ${savedLooks().length + 1}`;
  const list = savedLooks().filter(x => x.name !== name);
  list.push({ name, look: { ...cloneLook(effLook()), off: false } }); setSavedLooks(list);
  $('saveLookRow').style.display = 'none'; renderMyLooks();
  toast(`Saved "${name}". Tap it on any clip to use it`, 2200);
};
$('lookName').addEventListener('keydown', e => { if (e.key === 'Enter') $('btnSaveLookGo').click(); });
renderMyLooks();

// ───────────── boot ─────────────
syncControls();
updateSpeedInfo();
if (!('VideoDecoder' in window)) toast('This browser lacks WebCodecs. Use Chrome on Android/desktop.', 8000);
window.addEventListener('resize', () => updateReticle());
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
window.__macrocut = { state, loadFile, recompute, setStabOn, setSpeed, seek, video, lookNow: () => lookAtTime(state.mediaTime) }; // debugging / tests
