# MacroCut

A phone-first video editor for handheld macro footage (jumping spiders, mostly). Runs entirely in the browser; nothing is uploaded.

**Live:** https://themoojuice.github.io/macro-cut/

## Use

1. **Open** a clip (MP4/MOV straight from the phone camera; H.264 or HEVC, HDR is fine).
2. **Trim** → **✨ Cut out the blurry & searching bits** to keep only the good part in one tap (Undo if you disagree), or drag the white handles / Split + Delete yourself.
3. **Stabilise** → switch on, pick **Mild**, **Medium** or **Max**. Tap **❖** to flip between before and after.
4. **Frame** (rotate, crop, zoom, aspect), **Look** (sharpen, colour), **Speed** (0.1×–2×) as you like. Once you split the clip, Look and Speed change just the part under the playhead (or flip to *Whole clip*). In Look, **Natural** and **Spood pop** are starting points (tap the lit one again to switch it off); every slider starts at 0 = the preset as-is and moves it either way. **Off** shows the clip as filmed and tap it again to get your look back. Double-tap a slider to put it back to 0. Save looks you like under **My looks**.
5. **Export** → Render MP4 → Save / Share. Slowed parts get smooth slow motion (Speed tab switch, on by default). Tick **✨ Extra clean** for grainy (low-light) footage.
6. **📸** → *Find the sharpest shots* for full-resolution stills (for iNaturalist), or save the current frame. **Stack this moment** fuses the frames within ±0.3 s into one still, taking each region from whichever frame had it sharpest (tap the result to compare with the single frame).

**↶ / ↷** (top bar, or Ctrl+Z / Ctrl+Shift+Z) undo and redo any edit: cuts, looks, framing, speed, stabilise settings. One slider drag or pinch is one step; the shake measurement is never thrown away.

Your edit saves automatically on the phone; open the same clip again and it's all back (Trim → *Start this clip over* to reset).

## What's under the hood

| Part | How |
|---|---|
| Motion measuring | Frame-exact WebCodecs decode → GPU downscale → ~180 features tracked with pyramidal Lucas–Kanade on a background thread → robust (RANSAC) similarity fit: shift + rotation + zoom per frame. Anchored to keyframes (no drift), sticks to one motion layer (spider vs leaf), re-acquires after focus loss. |
| Steadying | Mild/Medium = low-pass of the camera path (keeps deliberate moves), Max = tripod lock. Auto-zoom never shows black edges; where shake exceeds the zoom budget it eases to lighter smoothing for just those moments, so tremor is still removed. |
| HDR | HLG/PQ clips are detected from the file; if the browser uploads them as plain video, the shader converts properly (HLG OOTF / PQ EOTF, BT.2020→709, highlight roll-off). |
| Look | Bicubic warp → contrast-adaptive sharpening + halo-limited fine detail + background smoothing → grading in OKLab. |
| Auto-cut | Keeps stretches that are in focus and not mid-sweep (sweep = camera motion well above the clip's normal handheld following). Gaps under 0.6 s are kept to avoid jump cuts. |
| Best shots | Ranks frames by focus (Laplacian on a noise-robust half-res grid), picks the 6 sharpest well-separated moments, renders each at full resolution with all settings applied. |
| Smooth slow motion | For parts slowed below the source frame rate, the export measures how every part of the picture moves between two real frames (grid Lucas–Kanade both ways, consistency-checked, gaps filled) and creates the in-between frames by moving both frames along that motion and blending. On a 60 fps ground truth, created frames are within ~1 dB of real ones (vs ~4.5 dB worse when repeating frames). |
| Extra clean (export switch) | Each frame is blended with its ±2 neighbours, lined up along measured motion (chained from one motion measurement per frame). Per pixel, neighbours that don't match the centre (moving legs, bad motion) get zero weight. The blend threshold adapts to the clip's measured grain, and softer neighbours are down-weighted, so clean footage is left alone. Grainy test clip: 32.3 → 36.7 dB vs noise-free truth. Spec: `docs/SPEC-multiframe.md`. |
| Stacked stills | Frames within ±0.3 s are lined up on the chosen frame by measured motion. The chosen frame is the base; another frame only replaces a region where it is clearly sharper (2–6 px detail band, so compression blocking doesn't count) *and* its structure correlates with the base (so a moved leg isn't pasted in). Worst case you get the single frame back. Spec and honest results: `docs/SPEC-multiframe.md`. |
| Saving | Each clip's edit (cuts, per-part looks and speeds, framing, stabilisation and the shake measurement) autosaves to IndexedDB on the phone. |
| Export | Offline, frame-accurate render → H.264 MP4 (VP9 fallback), constant frame rate, original audio passed through at normal speed. Screen kept awake while working. |

## Limits

- Needs WebCodecs + WebGL2 (Chrome/Edge on Android or desktop).
- Smooth slow motion is created on export (the preview shows the real frames). Very fast movement, like a jump, can smear slightly in created frames; filming at 60/120 fps always helps.
- Sound is only kept at normal speed.
- Rolling-shutter wobble isn't corrected.

## Code

```
index.html, css/app.css   UI
js/app.js                 state, controls, preview, gestures, stills, auto-cut
js/core.js                tracking, stabilisation, auto-cut maths (pure, Node-testable)
js/track-worker.js        background-thread tracker
js/gpu.js                 WebGL2: orientation, HDR, warp, sharpen, grade
js/media.js               mp4box demux (header only, streamed reads) + WebCodecs decode
js/engine.js              analysis + export pipelines (incl. in-between frames)
js/store.js               saved edits (IndexedDB) and saved looks
js/look.js                looks: preset + slider tweaks → renderer settings
vendor/                   mp4box.js (BSD-3), mp4-muxer (MIT)
test/                     synthetic clip generators, Node scoring, headless-browser end-to-end tests
```

Tests: `test/make_hard.py` (realistic hard synthetic clip with ground truth) + `test/score.mjs` (tracking error and jitter), `test/real_track.mjs` (real footage), `test/features.py` (every control measured on exported files), `test/e2e.py`, `test/hdr_e2e.py`, `test/hang.py` (simulated dead decoders), `test/upgrades.py` (auto-cut + best shots), `test/parts.py` (per-part look/speed), `test/look.test.mjs` + `test/look_ui.py` (Look presets, Off, sliders, old saved looks), `test/flow.test.mjs` + `test/slowmo.py` (in-between frames vs real 60 fps frames), `test/save.py` (save/restore), `test/undo.py` (undo/redo steps), `test/stall.py` (preview recovers when the phone takes the video decoder away), `test/denoise.proto.py` + `test/clean.py` (Extra clean vs noise-free truth), `test/make_focus.py` + `test/stack.py` (stacked still vs all-sharp truth), `test/stackreal.py` + `test/stackui.py` (stacking on real footage).
