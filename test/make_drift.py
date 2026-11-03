"""Slow-drift focus clip with ground truth (for local alignment of stored views).
A large textured body (most of the frame, all 4 depth bands) creeps sideways/down and grows slightly relative to the
background over the clip (posture drift, lean-in), while the focus sweeps band 0 -> 3 -> 0 twice ("sweep"), or sweeps
once and then hovers around the middle bands ("hover": the outer bands are only ever sharp at the start, and stay
moderately defocused, so their stored views drift further and further out of register while still being checkable).
Writes OUT.mp4, OUT_truth.npy (all-sharp, every 3rd frame), OUT.json.
"parallax" (4th argument): instead of the whole body drifting (which the tracker simply follows), near and far parts
drift apart, as when leaning in: band 0 moves +DRIFT, band 3 -DRIFT, the middle stays (the tracker can't absorb it).
usage: make_drift.py OUT [DRIFT_PX] [sweep|hover] [shift|parallax]   (DRIFT_PX = sideways drift over the clip, default 3)"""
import numpy as np, cv2, json, sys, math, subprocess
rng = np.random.default_rng(5)
out = sys.argv[1]; DRIFT = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0; SCHED = sys.argv[3] if len(sys.argv) > 3 else 'sweep'
KIND = sys.argv[4] if len(sys.argv) > 4 else 'shift'
W, H, N, FPS = 640, 360, 180, 30.0
S = 2; PAD = 100; WW, WH = W * S + 2 * PAD, H * S + 2 * PAD
def texture(seed, n, lw):
    r = np.random.default_rng(seed)
    t = cv2.GaussianBlur(r.random((WH, WW)).astype(np.float32), (0, 0), 1.0); L = np.zeros((WH, WW), np.float32)
    for i in range(n):
        x, y = r.integers(0, WW), r.integers(0, WH); a = r.random() * 6.283; l = r.integers(8, 40)
        cv2.line(L, (int(x), int(y)), (int(x + math.cos(a) * l), int(y + math.sin(a) * l)), 1, lw)
    c = cv2.resize(r.random((WH // 80, WW // 80, 3)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
    return np.clip(0.28 + 0.35 * c + 0.22 * t[..., None] - 0.32 * L[..., None], 0, 1).astype(np.float32)
bg, body = texture(11, 1600, 1), texture(12, 2400, 1) * np.array([0.8, 1.0, 1.15], np.float32)
bx0, bx1 = int(0.18 * W * S) + PAD, int(0.78 * W * S) + PAD
bmask = np.zeros((WH, WW), np.float32); bmask[:, bx0:bx1] = 1
t = np.arange(N) / FPS
u = t / t[-1]                                   # 0 -> 1 over the clip
ddx, ddy, dsc = DRIFT * u, 0.6 * DRIFT * u, 1 + 0.004 * DRIFT * u   # px at 640x360, scale
YY, XX = np.mgrid[0:WH, 0:WW].astype(np.float32)
def scene_at(k):
    if KIND == 'parallax':   # displacement varies linearly down the frame: band 0 (top) +1 … band 3 (bottom) -1
        f = np.clip((YY - PAD) / (H * S) * 2 - 1, -1, 1) * -1.0          # band 0 (top) = +1, band 3 (bottom) = -1
        mx, my = (XX - ddx[k] * S * f).astype(np.float32), (YY - 0.6 * ddx[k] * S * f).astype(np.float32)
        b = cv2.remap(body, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        m = cv2.remap(bmask, mx, my, cv2.INTER_LINEAR)[..., None]
        return bg * (1 - m) + b * m
    cx, cy = (bx0 + bx1) / 2, WH / 2
    M = cv2.getRotationMatrix2D((cx, cy), 0, dsc[k]); M[:, 2] += [ddx[k] * S, ddy[k] * S]
    b = cv2.warpAffine(body, M, (WW, WH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    m = cv2.warpAffine(bmask, M, (WW, WH), flags=cv2.INTER_LINEAR)[..., None]
    return bg * (1 - m) + b * m
NB = 4; rows = [(PAD + b * H * S // NB, PAD + (b + 1) * H * S // NB) for b in range(NB)]
band_of_row = np.zeros(WH, np.int32)
for b, (r0, r1) in enumerate(rows): band_of_row[(r0 if b else 0):(r1 if b < NB - 1 else WH)] = b
def focus_at(x):
    if SCHED == 'hover':                         # 0 -> 3 over 1.2 s, then hover 1.5 ± 0.4
        return 3.0 * x / 1.2 if x < 1.2 else 1.5 + 0.4 * math.sin(2 * math.pi * 0.4 * (x - 1.2) + math.pi)
    p = (x % 3.0) / 1.5                          # 0 -> 3 over 1.5 s, back to 0 over the next 1.5 s
    return 3.0 * p if p <= 1 else 3.0 * (2 - p)
focus = np.array([focus_at(x) for x in t])
px = 3 * np.sin(2 * math.pi * 0.9 * t) + rng.normal(0, 0.4, N); py = 2 * np.sin(2 * math.pi * 0.7 * t + 1) + rng.normal(0, 0.4, N)
def warp(img, k):
    M = cv2.getRotationMatrix2D((WW / 2, WH / 2), 0.3 * math.sin(2 * math.pi * 0.6 * t[k]), 1 + 0.008 * math.sin(2 * math.pi * 0.5 * t[k]))
    M[:, 2] += [px[k] * S - PAD, py[k] * S - PAD]
    return cv2.resize(cv2.warpAffine(img, M, (W * S, H * S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_AREA)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out + '.mp4'], stdin=subprocess.PIPE)
truth = []
for k in range(N):
    sc = scene_at(k); img = np.empty_like(sc)
    for b in range(NB):
        sig = 2.0 * abs(b - focus[k]) * S
        src = sc if sig < 0.2 else cv2.GaussianBlur(sc, (0, 0), sig)
        m = band_of_row == b; img[m] = src[m]
    noisy = warp(img, k) + rng.normal(0, 0.012, (H, W, 3)).astype(np.float32)
    enc.stdin.write(np.clip(noisy * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1].tobytes())
    if k % 3 == 0: truth.append(np.clip(warp(sc, k) * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1])
enc.stdin.close(); enc.wait()
np.save(out + '_truth.npy', np.stack(truth))
json.dump({'W': W, 'H': H, 'fps': FPS, 'N': N, 'focus': focus.tolist(), 'bands': [[(r0 - PAD) / S, (r1 - PAD) / S] for r0, r1 in rows],
           'body': [(bx0 - PAD) / S, (bx1 - PAD) / S], 'drift': DRIFT, 'sched': SCHED, 'kind': KIND, 'truth_every': 3}, open(out + '.json', 'w'))
print('ok', N, 'frames, drift', DRIFT, 'px')
