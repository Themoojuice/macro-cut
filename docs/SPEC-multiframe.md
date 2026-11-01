# Spec: multi-frame quality features

Two features that use detail from neighbouring frames to improve each frame. Both reuse the dense-motion engine built for smooth slow motion (`denseFlow` in `js/core.js`). Neither invents detail: every output pixel comes from real captured pixels, lined up by measured motion.

| | A. Extra clean video | B. Stacked stills |
|---|---|---|
| What the user gets | Cleaner, crisper video: less grain and compression mush, so sharpening can go further without crunch | One still with more of the spider in focus than any single frame |
| Where it lives | Export sheet switch: **Extra clean (slower export)** | Stills sheet: each "sharpest shot" gets a **Stack** button, plus **Stack this moment** |
| Frames used | 5 (the frame ±2) | 8–20 around the chosen moment |
| How pixels combine | Weighted average (noise cancels) | Pick-the-sharpest per region (focus fusion), then light averaging |
| Status | **Built** (see results below) | **Built** (see results below) |

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

### As built (what changed from the plan, and why)
The first version followed the plan literally and **failed on real footage**: the stacked still was blurrier than a single frame. Three causes, three fixes:
1. **Weights saturated → plain averaging.** Grain measured 0 on clean 8-bit daylight footage, so the "sharpness above grain" floor collapsed and every textured pixel hit the weight cap. Fix: two passes. Pass 1 records the sharpest *agreeing* view of every spot; pass 2 weights each view relative to that (scale-free, can't saturate), with an absolute floor.
2. **"Sharpness" was mostly compression artefacts.** A pixel-scale Laplacian rewards HEVC blocking and grain, giving a stippled, crunchy mosaic. Fix: measure detail in the 2–6 px band (difference of two blur levels), which tracks real focus.
3. **Moving legs were rebuilt from the wrong place.** The motion field is coarse (one vector per ~48 px at 1080p), so an independently moving leg is misaligned, and a brightness check at 8 px doesn't notice. Fix: a structure check, correlation of the 4 px-scale pattern over ~25 px against the full-size centre frame. Blur keeps the correlation; a shifted leg kills it.

Plus a **do-no-harm rule**: the centre frame is the base, and another frame only contributes where it is locally ≥1.2–1.8× sharper *and* passes both checks. Worst case you get the single frame back.

### Results
- **Synthetic focus sweep** (thin-line texture, depth bands sharp in different frames, grain): stacked beats the centre frame by **+0.4 to +3.6 dB** in every band, but is **~1.5–2 dB short of the best single frame per band** (an oracle that picks a different frame per band). Likely limit: sub-pixel alignment of 1 px lines with a grid motion field. The original acceptance bar ("sharper than the best single frame in every band") is **not met**.
- **Real HDR spider clip** (frame already well focused, handheld, legs moving): detail on the subject +15%, background grain unchanged, legs that were soft in the centre frame come out sharper, no mosaic or holes. Mild edge ghosting on a moving leg outline is the remaining artefact.
- Cost: frames are decoded twice (plus a small greyscale pass); about 3–4 min for 16 frames at 1080p in the software-GPU test browser. Phone time unmeasured.

### Acceptance tests (revised)
- Synthetic focus sweep: stacked ≥ centre frame in every band (✔); stretch goal ≥ best single frame per band (✘, see above).
- Real clip: never worse than the single frame (no holes, no mosaic, no added grain) (✔ on the test clip); visibly sharper where the centre frame was soft (✔, modest).

---

## C. Focus build-up (video)

### Behaviour
- Export sheet, **Focus build-up**: *Off* · *Build up* · *Sharp from the start*.
- **Build up:** as the focus moves through the spider, every part that has been sharp stays sharp, until the whole animal is in focus. Anything that moves (palps, a leg, a head turn) shows the live frame instead of an old copy, so nothing ghosts.
- **Sharp from the start:** same, but each still pose is sharp from its first frame: the clip is read twice, and the second pass starts from what the first pass built.
- Needs the shake measurement (it is run first if missing). Works with Extra clean, smooth slow motion, per-part looks and speeds. Preview is unchanged (it needs every frame in order).
- Honest note in the UI: the result is a composite in time (each spot shows its sharpest *still-valid* view), not a single moment.

### Method
1. **A spider-locked canvas.** The tracker's camera `C_k` maps frame k onto a world fixed to the tracked layer (the spider). The accumulator lives in an *anchor* frame's coordinates (plus a margin), so stored pixels are resampled once on the way in and once on the way out, never repeatedly. It re-anchors (starts fresh) only when the current view no longer fits the canvas (big pan, rotation or zoom).
2. **Per frame, on a decision grid (~540p):**
   - *Sharpness now:* detail energy in the 2–6 px band of the incoming frame (same measure as stacked stills, so compression blocking and grain don't win), minus the grain floor.
   - *Still the same thing?* Blur the stored image by several amounts and compare each with what the camera sees now (luma, ~15-px window, brightness offset tolerated). If **no** blur of the stored view explains the current view, the spider moved there: take the live frame.
   - *Store the new frame* where it is clearly sharper than what is stored (soft ramp 1.1–1.5×), where the stored view is no longer valid, or where nothing is stored yet. So the stored view is always the best one seen.
   - *Show the live frame* where it is nearly as sharp as the stored view (≥ 0.7–0.9×) or the stored view is invalid; show the stored view only where it is clearly sharper. Both maps are smoothed slightly so choices don't flicker pixel to pixel.
   - Stored sharpness slowly decays (half-life ~11 s), so a stale view can eventually be replaced by a nearly-as-sharp fresh one.
3. **Colour update** at full canvas resolution: `stored = mix(stored, frame, take)`.
4. **Output:** the normal render path, with one change in the warp shader: where the *show* map says so, the stored pixel; elsewhere the live pixel (so whatever is in focus right now is exactly the live frame, resampled once, and in-between slow-motion frames still move correctly).

### Budget and risks
- **GPU memory:** two RGBA8 canvas textures (+ mipmaps) and small float decision maps. 4K source to 4K export ≈ 2 × 45 MB; lower export resolutions accumulate at a matching lower resolution.
- **Time:** one decision pass (at ~540p) and one full-resolution update per frame; *Sharp from the start* reads and processes the clip twice.
- **Risks:** (1) palp flicks smaller than the defocus blur can pass the "same thing?" test and smear slightly in time; (2) tracking error (~1–2 px at 4K) shows as a faint seam where neighbouring spots come from different frames; (3) exposure drift is tolerated only partly.

### As built (what testing changed, and why)
1. **Ghost edges of a moved object.** The first version compared brightness only and allowed any amount of blur. A striped object, blurred enough, averages to about the brightness of the background it left, so sharp fragments of it stayed behind at its old place. Fixes: compare in **colour**; allow only as much blur as the **drop in sharpness** explains (a view about as sharp as the stored one must match it nearly as it is); grow "changed" spots by ~2 px (720p scale) so the edges go too. Fragments gone in the synthetic test.
2. **"Sharp from the start" stalled the encoder.** The first pass encodes nothing, so nothing waited for the GPU: decoding raced ahead and queued minutes of GPU work, and the second pass's first frames waited behind it ("encoder stopped responding"). Fix: the first pass waits for each frame's GPU work (a fence, polled without blocking the page). The encoder is also now opened at the first encoded frame rather than at the start.
3. With Extra clean, the first pass's frame textures are freed before the second pass starts.
4. **Softer than live where the live frame was in focus.** Showing the stored view wherever the new frame wasn't *clearly* sharper meant an in-focus band showed an older copy (tracking error + one extra resample): 1.5–2 dB below the normal export. Fix: separate *store* (only clearly sharper frames replace the best view) from *show* (live wins whenever it's nearly as sharp).

### Results
- **Synthetic focus-dwell clip** (`test/make_buildup.py` + `test/buildup.py`: 960×540, 4 depth bands, focus dwelling ~1 s per band, hand shake + focus breathing + grain, a striped object that wiggles and jumps sideways), dB vs the all-sharp truth:
  - after a band has been in focus: normal 29.2–30.1 → **build up 32.9–33.1** (+3.0 to +3.7 dB) ✔
  - before the focus first reaches a band: normal 28.8–29.7 → **sharp from the start 32.4–32.7** (+2.8 to +3.6 dB) ✔
  - while a band is in focus: same as normal (the live frame is shown) ✔
  - where the object left: normal 31.0 → 32.6 (no ghost); where it arrived: 20.2 → 20.2 (worst single frame −0.2 dB) ✔
- **Real handheld spider clip** (`test/buildup_real.py`: 20 s, 960×540 30 fps copy of a 4K60 MacroCut export; the spider lowers its head at ~5 s and turns at ~19 s): subject detail vs the normal export ×1.10 (1–4 s), ×1.10 (5–10 s), **×1.25 (10–18 s)**; during the head turn ×1.01 (falls back to the live frame, no ghost). Frame-to-frame flicker is *lower* than normal (held spots are steady). *Sharp from the start* = *Build up* on this clip: the canvas keeps one view per spot, so the second pass starts from the clip's *last* pose, and spots where the spider later moved can't use it.
- **Off = unchanged:** with focus build-up off, exports are pixel-identical to before for normal, Extra clean and smooth slow motion.
- **Next step for *sharp from the start*:** start a fresh canvas whenever a large part of the subject changes (a new pose), so each pose gets its own first-pass result.
- **Remaining gap:** a held view is ~1.5–2 dB below a live in-focus frame (tracking error of ~0.1–0.2 px plus one extra resample). Local sub-pixel refinement of the canvas mapping would close part of it.
- **Cost** in the software-GPU test browser: ~3.3 s/frame at 540p for build up (normal export ~0.25 s/frame); sharp from the start ≈ 1.3× build up. Phone time not yet measured.

### Acceptance tests
- Synthetic focus-dwell clip with an all-sharp truth and a moving object: once each depth band has been in focus, build-up must beat the normal export in that band; where the object moved, it must not be worse than the normal export (no ghost).
- Real handheld spider clip: visibly sharper during still poses, falls back to the live frame during a head turn.
- Existing suite still passes.
