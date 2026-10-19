# Spec: multi-frame quality features

Two features that use detail from neighbouring frames to improve each frame. Both reuse the dense-motion engine built for smooth slow motion (`denseFlow` in `js/core.js`). Neither invents detail: every output pixel comes from real captured pixels, lined up by measured motion.

| | A. Extra clean video | B. Stacked stills |
|---|---|---|
| What the user gets | Cleaner, crisper video: less grain and compression mush, so sharpening can go further without crunch | One still with more of the spider in focus than any single frame |
| Where it lives | Export sheet switch: **Extra clean (slower export)** | Stills sheet: each "sharpest shot" gets a **Stack** button, plus **Stack this moment** |
| Frames used | 5 (the frame ±2) | 8–20 around the chosen moment |
| How pixels combine | Weighted average (noise cancels) | Pick-the-sharpest per region (focus fusion), then light averaging |
| Status | **Built** (see results below) | Next |

---

## A. Extra clean video (multi-frame denoising)

### Behaviour
- Off by default. A switch in the Export sheet: *Extra clean: blends 5 neighbouring frames to remove grain (export takes about 3× longer)*.
- Applies to the whole export, including stabilised, per-part looks and smooth slow motion (in-between frames are made from cleaned frames).
- The preview is unchanged (too heavy for live playback). Stills get the same treatment through feature B.

### Method
1. **Motion.** For each consecutive pair of source frames, measure dense motion once (grid Lucas–Kanade, both directions, consistency-checked, filled). Motion to frames ±2 is chained: `F(i→i+2)(p) = F(i→i+1)(p) + F(i+1→i+2)(p + F(i→i+1)(p))`. That's one motion measurement per source frame, not four.
2. **Align.** On the GPU, each neighbour is resampled (bicubic) onto the centre frame along its motion field.
3. **Robust blend.** Per pixel, each neighbour gets weight `w = exp(-(d/h)²)`, where `d` is the mean colour difference between the aligned neighbour and the centre over a small 5-tap patch. Aligned static detail matches (w≈1) and averages out the noise. Misaligned content (a moving leg, occlusion, a failed motion estimate) differs, so w≈0 and the centre frame shows through, with no ghosting. Output = `(C + Σ wⱼNⱼ) / (1 + Σ wⱼ)`.
4. The cleaned frame replaces the raw frame as input to the existing pipeline (warp → sharpen → grade → encode).

### Pipeline (export)
- Sliding window of 5 raw frame textures on the GPU plus their small greyscale pyramids. A frame is cleaned when frame +2 has arrived; the last 2 frames are cleaned at the end with the neighbours available.
- Cleaned frames feed the existing "whole frame / in-between frame" emit logic unchanged (that logic only needs "frame c is ready").
- Motion is only measured near frames that are actually exported (skips cut-out gaps).

### Budget and risks
- **Time:** about 1 motion measurement plus 1 extra GPU pass per source frame. Desktop roughly 0.2 s/frame extra; phone estimate 0.4–0.6 s/frame (unverified). Export is labelled slower; the screen wake lock covers it.
- **GPU memory:** 5 extra frame textures. 1080p ≈ 40 MB; 4K ≈ 160 MB (watch on older phones; fallback is to clean at most 1440p).
- **Risks:** (1) over-smoothing fine hair if the motion field is coarse, mitigated by the patch test, which rejects the neighbour wherever detail doesn't line up; (2) low-light HDR noise is higher than SDR, so the threshold `h` may need tuning on real clips.

### Results (built)
- **Grain-adaptive threshold.** The first version used a fixed threshold. On a real daylight clip (already clean) it cost ~19% of fine detail for no grain gain. Now the app measures grain at full resolution in the quietest parts of the frame (7 patches, second-quietest wins) and sets `h = 3.5 × patch-difference spread from grain alone` (clamped 0.004–0.06). Neighbours softer than the centre frame are also down-weighted (`w × min(1, (sharpness ratio)²)`).
- **Grainy synthetic clip vs noise-free truth:** normal export 32.3 dB → Extra clean **36.7 dB** whole frame; moving areas 29.7 → **32.1 dB** (no ghosting). ✔
- **Real daylight HDR clip:** threshold settles at the floor (little grain), detail kept within ~4%, i.e. "do no harm". The benefit is for low-light / high-zoom footage.
- Works together with smooth slow motion (in-between frames are made from cleaned frames).
- Export time: ~3–4× normal in the software-GPU test browser; phone timing still to be measured.

### Acceptance tests
- Synthetic clip with known noise-free ground truth plus added noise: Extra clean must raise fidelity in static areas by ≥2 dB and must not lower it in moving areas (no ghosting).
- Real HDR spider clip: visual check of hair and eyes, no double edges.
- Full existing suite still passes (features 16/16, end-to-end, HDR, hang, slow motion).

---

## B. Stacked stills (temporal focus fusion)

### Behaviour
- In the Stills sheet, each sharpest-shot card gets **Stack**, and there's a **Stack this moment** for the playhead position.
- It uses the frames within ±0.3 s (about 8–20 frames) and saves a PNG at full resolution with the look applied.
- Shows a before/after toggle on the result.

### Method
1. Align all frames in the window to the centre frame (dense motion, chained like A, plus the global stabilisation transform).
2. Per frame, per pixel: a **focus measure** (local Laplacian energy over a 7×7 window) on the aligned frame.
3. **Fusion:** a soft choice of the sharpest frames per region (weights ∝ focusᵖ, p≈4, smoothed spatially so the choice doesn't flip pixel to pixel), plus a consistency test against the centre frame so moving legs come from one frame only (no ghost limbs).
4. Light multi-frame averaging within the chosen frames for noise, then the normal look pipeline.

### Risks
- Fusion halos at depth edges (classic focus-stacking artefact): use a guided/edge-aware smoothing of the weight maps.
- Subject movement within ±0.3 s: per-region consistency test falls back to the centre frame.
- This is the same engine as the planned standalone salticid focus stacker; build it once, reuse it for burst photos later.

### Acceptance tests
- Synthetic focus-sweep clip (different depth bands sharp in different frames): the fused still must be sharper than the best single frame in every band.
- Real clip: eyes, carapace and leg tips all sharp in one still, with no double edges.
