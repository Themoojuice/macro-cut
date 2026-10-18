# MacroCut

A phone-first video editor for handheld macro footage (jumping spiders, mostly). It runs entirely in the browser: nothing is uploaded.

**Live:** https://themoojuice.github.io/macro-cut/

## What it does

| Step | How |
|---|---|
| **Stabilise** | Every frame is decoded exactly (WebCodecs, no seeking), downscaled on the GPU, and ~180 corner features are tracked with pyramidal Lucas–Kanade at sub-pixel accuracy. A robust (RANSAC) similarity fit gives shift **+ rotation + scale** (focus breathing) per frame. Frames are matched against a keyframe rather than chained frame-to-frame, so the path doesn't drift. |
| **Smooth / Tripod lock** | *Smooth* low-passes the camera path (Gaussian, adjustable window) to keep intentional moves and remove heartbeat and tremor. *Tripod lock* pins the view. |
| **Auto-crop** | Picks the smallest zoom that never shows black edges, up to your *Max crop*. Frames that would exceed it get proportionally less correction, eased in and out. |
| **Tap a spot** | Track a chosen region (e.g. the leaf) instead of the dominant motion, for when the spider walks. |
| **Frame** | Fine rotation, 90° turns, zoom, drag-to-pan, pinch/twist, aspect presets (9:16, 4:5, 1:1, 16:9). |
| **Look** | Bicubic warp → AMD-style contrast-adaptive sharpening + halo-limited micro-detail (unsharp at adjustable radius, weighted to in-focus areas) + bokeh smoothing in out-of-focus areas → grading in OKLab (exposure, contrast, shadows, highlights, saturation, vibrance, warmth, tint). |
| **Speed** | ½×, ¼×, ⅛×. A 120 fps source at ¼× gives true 30 fps slow-mo. A 30 fps source will step; there is no frame interpolation. |
| **Export** | Frame-accurate offline render → H.264 MP4 (VP9 fallback) at constant frame rate, with the original AAC audio passed through at 1× speed. A 📸 button saves a full-resolution processed PNG still. |

## Use

1. Open a clip (MP4/MOV, H.264 or HEVC; Chrome on Android or desktop).
2. **Trim**: cut out the blurry bits (the focus strip appears after analysis).
3. **Stabilise → Analyse motion**, then pick Smooth or Tripod lock. Hold **❖** to compare with the original.
4. **Frame**, **Look**, **Speed** as wanted. Settings are remembered.
5. **Export → Render MP4 → Save/Share.**

## Limits

- Needs WebCodecs + WebGL2 (Chrome/Edge). Not Firefox or older Safari.
- The whole file is held in memory: very long 4K clips can run a phone out of RAM. Trim on the phone first if that happens.
- Fragmented MP4 isn't supported (phone camera files are fine).
- HDR (10-bit) footage is converted to SDR without tone-mapping control.
- Rolling-shutter wobble isn't corrected (the model is a rigid similarity).

## Code

```
index.html, css/app.css   UI
js/app.js                 state, controls, preview, gestures
js/core.js                tracking + stabilisation maths (pure, Node-testable)
js/gpu.js                 WebGL2 shaders: orientation, warp, sharpen, grade
js/media.js               mp4box demux + WebCodecs decode
js/engine.js              analysis + export pipelines
vendor/                   mp4box.js (BSD-3), mp4-muxer (MIT)
test/                     synthetic shaky-clip generator, Node + headless-browser tests
```

Tests: `python3 test/make_synth.py /tmp/s 640 360 150 30` (add `3` as a 6th argument for 3× shake), then `python3 -c "import numpy as np; np.load('/tmp/s.npy').mean(-1).astype('uint8').tofile('/tmp/s.gray')"` and `node test/core.test.mjs /tmp/s`. `test/e2e.py` drives the real app in headless Chromium, and `test/measure.py` scores any video's residual shake with OpenCV.
