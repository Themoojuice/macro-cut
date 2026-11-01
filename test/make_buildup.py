"""Focus build-up test clip with ground truth.
A detailed scene in 4 depth bands (rows). Focus dwells on one band at a time and steps through them (like
leaning in and out), with hand shake, focus breathing (slight zoom), grain, and a striped object in band 2
that wiggles (palp flick) and at t = 2 s jumps sideways (anything left behind there would be a ghost).
Writes OUT.mp4 (H.264), OUT_truth.npy (all-sharp, noise-free version of every 3rd frame) and OUT.json.
usage: make_buildup.py OUT [W H N FPS]"""
import numpy as np, cv2, json, sys, math, subprocess
rng = np.random.default_rng(11)
out = sys.argv[1]
W, H, N, FPS = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (960, 540, 180, 30.0)
S = 2; PAD = 120; WW, WH = W * S + 2 * PAD, H * S + 2 * PAD
# scene: hair-like strokes + fine texture + colour patches, detailed everywhere
tex = cv2.GaussianBlur(rng.random((WH, WW)).astype(np.float32), (0, 0), 1.0)
lines = np.zeros((WH, WW), np.float32)
for i in range(2600):
    x, y = rng.integers(0, WW), rng.integers(0, WH); a = rng.random() * 6.283; L = rng.integers(8, 40)
    cv2.line(lines, (int(x), int(y)), (int(x + math.cos(a) * L), int(y + math.sin(a) * L)), 1, 1)
col = cv2.resize(rng.random((WH // 80, WW // 80, 3)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
scene = np.clip(0.28 + 0.35 * col + 0.22 * tex[..., None] - 0.32 * lines[..., None], 0, 1).astype(np.float32)
NB = 4
rows = [(PAD + b * H * S // NB, PAD + (b + 1) * H * S // NB) for b in range(NB)]
band_of_row = np.zeros(WH, np.int32)
for b, (r0, r1) in enumerate(rows): band_of_row[(r0 if b else 0):(r1 if b < NB - 1 else WH)] = b
t = np.arange(N) / FPS
# focus dwells ~1 s per band, quick moves in between (0 → 1 → 2 → 3)
def focus_at(tt):
    steps = [(0.0, 0), (1.0, 1), (2.3, 2), (3.6, 3)]
    f = 0.0
    for t0, b in steps:
        if tt >= t0: f = b - 1 + min(1.0, (tt - t0) / 0.3) if b else 0.0
    return f
focus = np.array([focus_at(x) for x in t])
px = 4 * np.sin(2 * math.pi * 0.9 * t) + rng.normal(0, 0.5, N); py = 3 * np.sin(2 * math.pi * 0.7 * t + 1) + rng.normal(0, 0.5, N)
rot = np.deg2rad(0.4 * np.sin(2 * math.pi * 0.6 * t)); zoom = 1 + 0.01 * np.sin(2 * math.pi * 0.5 * t)
# the moving object (band 2): striped ellipse, wiggles ±1.5 px at 3 Hz, jumps from 30% to 62% of the width at t = 2 s
oy = (rows[2][0] + rows[2][1]) // 2
def obj_x(tt):
    x0, x1 = 0.30 * W * S + PAD, 0.62 * W * S + PAD
    u = min(1.0, max(0.0, (tt - 2.0) / 0.2))
    return x0 + (x1 - x0) * (u * u * (3 - 2 * u)) + 1.5 * S * math.sin(2 * math.pi * 3 * tt)
ox = np.array([obj_x(x) for x in t])
ow, oh = int(0.07 * W * S), int(0.09 * H * S)
yy, xx = np.mgrid[-oh:oh + 1, -ow:ow + 1].astype(np.float32)
mask = np.clip((1 - (xx / ow) ** 2 - (yy / oh) ** 2) * 12, 0, 1)[..., None]
stripes = (0.5 + 0.45 * np.sign(np.sin(xx * 0.35 + yy * 0.12)))[..., None] * np.array([0.25, 0.55, 0.95], np.float32)
def with_object(img, x):
    o = img.copy(); x = int(round(x)); y0, x0 = oy - oh, x - ow
    o[y0:y0 + 2 * oh + 1, x0:x0 + 2 * ow + 1] = o[y0:y0 + 2 * oh + 1, x0:x0 + 2 * ow + 1] * (1 - mask) + stripes * mask
    return o
cache = {}
def blurred(s):
    k = round(s * 4) / 4
    if k not in cache: cache[k] = scene if k < 0.2 else cv2.GaussianBlur(scene, (0, 0), k)
    return cache[k]
def warp(img, k):
    M = cv2.getRotationMatrix2D((WW / 2, WH / 2), math.degrees(rot[k]), zoom[k])
    M[:, 2] += [px[k] * S - PAD, py[k] * S - PAD]
    return cv2.resize(cv2.warpAffine(img, M, (W * S, H * S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_AREA)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '12', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out + '.mp4'], stdin=subprocess.PIPE)
truth = []
for k in range(N):
    img = np.empty_like(scene)
    for b in range(NB):
        sig = 2.0 * abs(b - focus[k]) * S
        src = cv2.GaussianBlur(with_object(scene, ox[k]), (0, 0), sig) if (b == 2 and sig >= 0.2) else (with_object(scene, ox[k]) if b == 2 else blurred(sig))
        m = band_of_row == b
        img[m] = src[m]
    noisy = warp(img, k) + rng.normal(0, 0.012, (H, W, 3)).astype(np.float32)
    enc.stdin.write(np.clip(noisy * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1].tobytes())
    if k % 3 == 0: truth.append(np.clip(warp(with_object(scene, ox[k]), k) * 255 + 0.5, 0, 255).astype(np.uint8)[..., ::-1])
enc.stdin.close(); enc.wait()
np.save(out + '_truth.npy', np.stack(truth))
to_out = lambda v: (v - PAD) / S
json.dump({'W': W, 'H': H, 'fps': FPS, 'N': N, 'focus': focus.tolist(), 'bands': [[to_out(r0), to_out(r1)] for r0, r1 in rows],
           'obj': {'x': [to_out(v) for v in ox], 'y': to_out(oy), 'w': ow / S, 'h': oh / S, 'jump': 2.0}, 'truth_every': 3}, open(out + '.json', 'w'))
print('ok', out + '.mp4', N, 'frames')
