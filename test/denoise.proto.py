"""CPU prototype of multi-frame denoise: motion from the app's JS engine (via node), blend in numpy. Tunes h."""
import json, subprocess, sys, numpy as np, cv2
T = sys.argv[1]; W, H = 640, 360
clean = np.load(T + '/dn_clean.npy').astype(np.float32) / 255; noisy = np.load(T + '/dn_noisy.npy').astype(np.float32) / 255
N = len(noisy)
noisy.mean(-1).__mul__(255).astype(np.uint8).tofile(T + '/dn_noisy.gray')
# motion fields for consecutive pairs from the JS engine
js = r'''
import fs from 'node:fs';
import { denseFlow, buildPyramid, globalShift } from '/home/claude/macro-cut/js/core.js';
const [g, W, H, N] = [process.argv[2], +process.argv[3], +process.argv[4], +process.argv[5]];
const raw = fs.readFileSync(g), pyr = k => buildPyramid(Float32Array.from(raw.subarray(k * W * H, (k + 1) * W * H)), W, H, 4);
const out = []; let p0 = pyr(0), t0 = performance.now();
for (let k = 0; k + 1 < N; k++) { const p1 = pyr(k + 1); const F = denseFlow(null, null, W, H, { pyrA: p0, pyrB: p1, guess: globalShift(p0, p1) });
  out.push({ gw: F.gw, gh: F.gh, fx: Array.from(F.fx), fy: Array.from(F.fy), bx: Array.from(F.bx), by: Array.from(F.by) }); p0 = p1; }
console.error(`motion: ${((performance.now() - t0) / (N - 1)).toFixed(0)} ms per frame pair`);
fs.writeFileSync(process.argv[6], JSON.stringify(out));
'''
open(T + '/dn_flow.mjs', 'w').write(js)
subprocess.run(['node', T + '/dn_flow.mjs', T + '/dn_noisy.gray', str(W), str(H), str(N), T + '/dn_flows.json'], check=True)
flows = json.load(open(T + '/dn_flows.json'))
def field(f, kx, ky):
    g = np.stack([np.array(f[kx], np.float32).reshape(f['gh'], f['gw']), np.array(f[ky], np.float32).reshape(f['gh'], f['gw'])], -1)
    # grid cell centres at (gx+0.5)*W/gw → resize with INTER_LINEAR matches that convention
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
fwd = [field(f, 'fx', 'fy') for f in flows]; bwd = [field(f, 'bx', 'by') for f in flows]
xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
def chain(F1, F2):
    return F1 + cv2.remap(F2, xx + F1[..., 0], yy + F1[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
def warp(img, F): return cv2.remap(img, xx + F[..., 0], yy + F[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def denoise(i, h, K=2):
    C = noisy[i]; acc = C.copy(); ws = np.ones((H, W), np.float32)
    lumC = cv2.blur(C.mean(-1), (3, 3))
    for d in [d for d in range(-K, K + 1) if d and 0 <= i + d < N]:
        if d == 1: F = fwd[i]
        elif d == 2: F = chain(fwd[i], fwd[i + 1])
        elif d == -1: F = bwd[i - 1]
        else: F = chain(bwd[i - 1], bwd[i - 2])
        Nw = warp(noisy[i + d], F)
        dist = np.abs(cv2.blur(Nw.mean(-1), (3, 3)) - lumC)
        w = np.exp(-(dist / h) ** 2)
        acc += Nw * w[..., None]; ws += w
    return acc / ws[..., None]
ps = lambda a, b, m=None: 10 * np.log10(1 / np.mean(((a - b) ** 2) if m is None else ((a - b) ** 2)[m]))
frames = range(3, N - 3, 5)
masks = {}
for i in frames:   # moving subject = where clean frame differs from the camera-aligned clean neighbour
    m = np.zeros((H, W), bool)
    for d in (-2, -1, 1, 2):   # where the true scene differs from a neighbour once the camera is lined up
        j = i + d
        if 0 <= j < N:
            sh, _ = cv2.phaseCorrelate(cv2.GaussianBlur(clean[i].mean(-1), (0, 0), 3), cv2.GaussianBlur(clean[j].mean(-1), (0, 0), 3))
            M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
            m |= np.abs(cv2.warpAffine(clean[j], M, (W, H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR) - clean[i]).max(-1) > 0.08
    masks[i] = cv2.erode(m.astype(np.uint8), np.ones((5, 5))).astype(bool)
frames = [i for i in frames if masks[i].sum() > 300] or list(frames)
print(f'{len(frames)} test frames; baseline noisy: all {np.mean([ps(noisy[i], clean[i]) for i in frames]):.2f} dB | moving {np.mean([ps(noisy[i], clean[i], masks[i]) for i in frames]):.2f} dB')
sp = lambda x: cv2.bilateralFilter(x, 5, 0.08, 3)
print(f'single-frame smoothing (reference): all {np.mean([ps(sp(noisy[i]), clean[i]) for i in frames]):.2f} dB')
for h in [0.02, 0.03, 0.05, 1.0]:
    out = {i: denoise(i, h) for i in frames}
    print(f'h={h:<5} all {np.mean([ps(out[i], clean[i]) for i in frames]):.2f} dB | moving areas {np.mean([ps(out[i], clean[i], masks[i]) for i in frames]):.2f} dB | moving share {np.mean([masks[i].mean() for i in frames])*100:.1f}%')
