# MacroCut

A phone-first video editor for handheld macro footage (jumping spiders, mostly). Runs entirely in the browser; nothing is uploaded.

**Live:** https://themoojuice.github.io/macro-cut/

## Use

1. **Open** a clip (MP4/MOV straight from the phone camera; H.264 or HEVC, HDR is fine).
2. **Trim** → **✨ Cut out the blurry & searching bits** to keep only the good part in one tap (Undo if you disagree), or drag the white handles / Split + Delete yourself.
3. **Stabilise** → switch on, pick **Mild**, **Medium** or **Max**. Tap **❖** to flip between before and after.
4. **Frame** (rotate, crop, zoom, aspect), **Look** (sharpen, colour), **Speed** (0.1×–2×) as you like. Settings are remembered.
5. **Export** → Render MP4 → Save / Share.
6. **📸** → *Find the sharpest shots* for full-resolution stills (for iNaturalist), or save the current frame.

## What's under the hood

| Part | How |
|---|---|
| Motion measuring | Frame-exact WebCodecs decode → GPU downscale → ~180 features tracked with pyramidal Lucas–Kanade on a background thread → robust (RANSAC) similarity fit: shift + rotation + zoom per frame. Anchored to keyframes (no drift), sticks to one motion layer (spider vs leaf), re-acquires after focus loss. |
| Steadying | Mild/Medium = low-pass of the camera path (keeps deliberate moves), Max = tripod lock. Auto-zoom never shows black edges; where shake exceeds the zoom budget it eases to lighter smoothing for just those moments, so tremor is still removed. |
| HDR | HLG/PQ clips are detected from the file; if the browser uploads them as plain video, the shader converts properly (HLG OOTF / PQ EOTF, BT.2020→709, highlight roll-off). |
| Look | Bicubic warp → contrast-adaptive sharpening + halo-limited fine detail + background smoothing → grading in OKLab. |
| Auto-cut | Keeps stretches that are in focus and not mid-sweep (sweep = camera motion well above the clip's normal handheld following). Gaps under 0.6 s are kept to avoid jump cuts. |
| Best shots | Ranks frames by focus (Laplacian on a noise-robust half-res grid), picks the 6 sharpest well-separated moments, renders each at full resolution with all settings applied. |
| Export | Offline, frame-accurate render → H.264 MP4 (VP9 fallback), constant frame rate, original audio passed through at normal speed. Screen kept awake while working. |

## Limits

- Needs WebCodecs + WebGL2 (Chrome/Edge on Android or desktop).
- Slow motion repeats frames (no frame interpolation): film at 60/120 fps for silky slow-mo.
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
js/engine.js              analysis + export pipelines
vendor/                   mp4box.js (BSD-3), mp4-muxer (MIT)
test/                     synthetic clip generators, Node scoring, headless-browser end-to-end tests
```

Tests: `test/make_hard.py` (realistic hard synthetic clip with ground truth) + `test/score.mjs` (tracking error and jitter), `test/real_track.mjs` (real footage), `test/features.py` (every control measured on exported files), `test/e2e.py`, `test/hdr_e2e.py`, `test/hang.py` (simulated dead decoders), `test/upgrades.py` (auto-cut + best shots).
