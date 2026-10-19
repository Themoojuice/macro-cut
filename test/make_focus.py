"""Focus-sweep clip: a textured scene at 5 depth bands; focus drifts through depth over time (like breathing in macro),
plus small hand shake and grain. Writes noisy clip frames and an all-sharp 'truth' version of every frame.
usage: make_focus.py out W H N FPS"""
import numpy as np, cv2, json, sys, math
rng = np.random.default_rng(5)
out = sys.argv[1]; W, H, N, FPS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
S = 2; WW, WH = W * S + 200, H * S + 200
# scene: fine hair-like texture + colour patches, everywhere detailed (so sharpness is measurable in every band)
tex = rng.random((WH, WW)).astype(np.float32)
tex = cv2.GaussianBlur(tex, (0, 0), 1.0)
lines = np.zeros((WH, WW), np.float32)
for i in range(900):
    x, y = rng.integers(0, WW), rng.integers(0, WH); a = rng.random() * 6.283; L = rng.integers(6, 30)
    cv2.line(lines, (int(x), int(y)), (int(x + math.cos(a) * L), int(y + math.sin(a) * L)), 1, 1)
col = cv2.resize(rng.random((WH // 60, WW // 60, 3)).astype(np.float32), (WW, WH), interpolation=cv2.INTER_CUBIC)
scene = np.clip(0.25 + 0.35 * col + 0.25 * tex[..., None] - 0.3 * lines[..., None], 0, 1)
# depth bands (rows) 0..4 with smooth boundaries
band = np.clip(np.arange(WH)[:, None] * 5 // WH, 0, 4).repeat(WW, 1).astype(np.float32)
t = np.arange(N) / FPS
focus = 2 + 2.2 * np.sin(2 * math.pi * 0.9 * t)           # sweeps the bands back and forth
px = 3 * np.sin(2 * math.pi * 1.3 * t) + rng.normal(0, 0.6, N); py = 2 * np.sin(2 * math.pi * 1.1 * t + 1) + rng.normal(0, 0.6, N)
rot = np.deg2rad(0.4 * np.sin(2 * math.pi * 0.7 * t))
blurred = {}
def blur_level(s):
    k = round(s * 4) / 4
    if k not in blurred: blurred[k] = scene if k < 0.2 else cv2.GaussianBlur(scene, (0, 0), k)
    return blurred[k]
frames = np.zeros((N, H, W, 3), np.uint8); truth = np.zeros((N, H, W, 3), np.uint8)
for k in range(N):
    img = np.zeros_like(scene)
    for b in range(5):
        sig = 1.6 * abs(b - focus[k]) * S          # blur grows with distance from the focus plane
        m = (band == b)[..., None]
        img = np.where(m, blur_level(sig), img)
    c, s = math.cos(rot[k]), math.sin(rot[k])
    M = cv2.getRotationMatrix2D((WW / 2, WH / 2), math.degrees(rot[k]), 1.0)
    M[:, 2] += [px[k] * S - (WW - W * S) / 2, py[k] * S - (WH - H * S) / 2]
    warp = lambda x: cv2.resize(cv2.warpAffine(x, M, (W * S, H * S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), (W, H), interpolation=cv2.INTER_AREA)
    noisy = warp(img) + rng.normal(0, 0.015, (H, W, 3))
    frames[k] = np.clip(noisy * 255, 0, 255).astype(np.uint8)[..., ::-1]
    truth[k] = np.clip(warp(scene) * 255, 0, 255).astype(np.uint8)[..., ::-1]
np.save(out + '.npy', frames); np.save(out + '_truth.npy', truth)
json.dump({'W': W, 'H': H, 'fps': FPS, 'focus': focus.tolist()}, open(out + '.json', 'w'))
print('ok', frames.shape)
