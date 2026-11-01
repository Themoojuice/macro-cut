"""Two-pose focus clip with ground truth (for per-pose "sharp from the start").
A large textured body (most of the frame, all 4 depth bands) shifts sideways at t = 3 s, like a spider turning.
In each pose the focus dwells on band 0, 1, 2, 3 in turn, so each pose's later bands only get sharp late.
Writes OUT.mp4, OUT_truth.npy (all-sharp, every 3rd frame), OUT.json.  usage: make_poses.py OUT"""
import numpy as np, cv2, json, sys, math, subprocess
rng = np.random.default_rng(3)
out = sys.argv[1]; W, H, N, FPS = 640, 360, 180, 30.0
S = 2; PAD = 100; WW, WH = W * S + 2 * PAD, H * S + 2 * PAD
def texture(seed, n, lw):
    r = np.random.default_rng(seed)
    t = cv2.GaussianBlur(r.random((WH, WW)).astype(np.float32), (0, 0), 1.0); L = np.zeros((WH, WW), np.float32)
    for i in range(n):
        x, y = r.integers(0, WW), r.integers(0, WH); a = r.random() * 6.283; l = r.integers(8, 40)
        cv2.line(L, (int(x), int(y)), (int(x + math.cos(a) * l), int(y + math.sin(a) * l)), 1, lw)
    c = cv2.resize(r.random((WH // 80, WW // 80, 3)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
    return np.clip(0.28 + 0.35 * c + 0.22 * t[..., None] - 0.32 * L[..., None], 0, 1).astype(np.float32)
bg, body = texture(1, 1600, 1), texture(2, 2400, 1) * np.array([0.8, 1.0, 1.15], np.float32)
bx0, bx1 = int(0.18 * W * S) + PAD, int(0.78 * W * S) + PAD
SHIFT = 40 * S   # body moves 40 px right at t = 3 s
def scene_at(t):
    u = min(1.0, max(0.0, (t - 3.0) / 0.1)); dx = int(round(SHIFT * u))
    img = bg.copy(); img[:, bx0 + dx:bx1 + dx] = np.roll(body, dx, axis=1)[:, bx0 + dx:bx1 + dx]
    return img, dx
NB = 4; rows = [(PAD + b * H * S // NB, PAD + (b + 1) * H * S // NB) for b in range(NB)]
band_of_row = np.zeros(WH, np.int32)
for b, (r0, r1) in enumerate(rows): band_of_row[(r0 if b else 0):(r1 if b < NB - 1 else WH)] = b
def focus_at(t):
    tp = t % 3.0
    return min(3.0, tp / 0.7) if tp < 2.8 else 3.0   # sweeps band 0 → 3 within each pose
t = np.arange(N) / FPS; focus = np.array([focus_at(x) for x in t])
px = 3 * np.sin(2 * math.pi * 0.9 * t) + rng.normal(0, 0.4, N); py = 2 * np.sin(2 * math.pi * 0.7 * t + 1) + rng.normal(0, 0.4, N)
def warp(img, k):
    M = cv2.getRotationMatrix2D((WW / 2, WH / 2), 0.3 * math.sin(2 * math.pi * 0.6 * t[k]), 1 + 0.008 * math.sin(2 * math.pi * 0.5 * t[k]))
    M[:, 2] += [px[k] * S - PAD, py[k] * S - PAD]
    return cv2.resize(cv2.warpAffine(img, M, (W * S, H * S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_AREA)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out + '.mp4'], stdin=subprocess.PIPE)
truth, dxs = [], []
for k in range(N):
    sc, dx = scene_at(t[k]); dxs.append(dx / S)
    img = np.empty_like(sc)
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
           'body': [(bx0 - PAD) / S, (bx1 - PAD) / S], 'dx': dxs, 'truth_every': 3}, open(out + '.json', 'w'))
print('ok', N, 'frames')
