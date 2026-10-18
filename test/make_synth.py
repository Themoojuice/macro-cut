"""Synthetic shaky macro clip with known camera motion.
World: sharp, hairy 'spider' on textured leaf with heavy bokeh background.
Writes frames (uint8 gray or BGR) + ground-truth per-frame transforms (frame px -> world px)."""
import numpy as np, cv2, json, sys, math
rng = np.random.default_rng(3)
W, H = int(sys.argv[2]) if len(sys.argv) > 2 else 640, int(sys.argv[3]) if len(sys.argv) > 3 else 360
N = int(sys.argv[4]) if len(sys.argv) > 4 else 150
FPS = float(sys.argv[5]) if len(sys.argv) > 5 else 30
out = sys.argv[1]
S = 3  # world oversampling
WW, WH = W * S * 2, H * S * 2
# bokeh background: low-freq colour blobs
bg = rng.random((WH // 40, WW // 40, 3)).astype(np.float32)
bg = cv2.resize(bg, (WW, WH), interpolation=cv2.INTER_CUBIC)
bg = cv2.GaussianBlur(bg, (0, 0), 25) * 0.6 + 0.2
world = bg.copy()
# in-focus leaf band with vein texture
leaf = np.zeros((WH, WW), np.float32)
cv2.ellipse(leaf, (WW // 2, WH // 2 + 60 * S), (WW // 3, WH // 6), 8, 0, 360, 1, -1)
leaf = cv2.GaussianBlur(leaf, (0, 0), 6)
tex = rng.random((WH, WW)).astype(np.float32)
tex = cv2.GaussianBlur(tex, (0, 0), 1.5)
veins = np.zeros((WH, WW), np.float32)
for i in range(60):
    x0 = int(rng.integers(0, WW)); cv2.line(veins, (x0, 0), (x0 + int(rng.integers(-400, 400)), WH), 1, 2 * S)
veins = cv2.GaussianBlur(veins, (0, 0), 1.2)
leafcol = np.dstack([0.15 + 0.5 * tex - 0.2 * veins, 0.45 + 0.4 * tex - 0.25 * veins, 0.12 + 0.3 * tex])
world = world * (1 - leaf[..., None]) + leafcol * leaf[..., None]
# spider: dark body + hairs + eyes
sp = np.zeros((WH, WW), np.float32)
cx, cy = WW // 2, WH // 2
cv2.ellipse(sp, (cx, cy), (55 * S, 38 * S), 0, 0, 360, 1, -1)
cv2.circle(sp, (cx + 60 * S, cy), 30 * S, 1, -1)
for i in range(8):
    a = -1.2 + i * 0.35 + (0 if i < 4 else 2.0)
    cv2.line(sp, (cx, cy), (int(cx + math.cos(a) * 140 * S), int(cy + math.sin(a) * 120 * S)), 1, 4 * S)
hairs = np.zeros((WH, WW), np.float32)
for i in range(1500):
    r = rng.random() * 60 * S; a = rng.random() * 6.283
    x, y = int(cx + math.cos(a) * r * 1.4), int(cy + math.sin(a) * r)
    b = rng.random() * 6.283
    cv2.line(hairs, (x, y), (int(x + math.cos(b) * 5 * S), int(y + math.sin(b) * 5 * S)), 1, 1)
spcol = np.dstack([0.08 + 0.5 * hairs, 0.07 + 0.45 * hairs, 0.1 + 0.55 * hairs])
for ex in (-12, 12):
    cv2.circle(spcol, (cx + 75 * S, cy + ex * S), 7 * S, (0.02, 0.02, 0.02), -1)
    cv2.circle(spcol, (cx + 77 * S, cy + ex * S - 2 * S), 2 * S, (0.95, 0.95, 0.95), -1)
world = world * (1 - sp[..., None]) + spcol * sp[..., None]
world = np.clip(world, 0, 1)

# camera motion: slow drift + heartbeat bursts (1.2 Hz) + tremor (9 Hz) + rotation + focus breathing
t = np.arange(N) / FPS
hb = np.zeros(N)
for k in range(N):
    ph = (t[k] * 1.2) % 1.0
    hb[k] = math.exp(-((ph - 0.1) / 0.06) ** 2) - 0.6 * math.exp(-((ph - 0.25) / 0.07) ** 2)
trem = lambda f, s: s * np.sin(2 * math.pi * f * t + rng.random() * 6) + s * 0.5 * rng.standard_normal(N)
AMP = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
px = AMP * 6 * hb + AMP * trem(9, 1.0) + 10 * t / t[-1]
py = AMP * 8 * hb + AMP * trem(8, 1.0) - 6 * t / t[-1]
rot = np.deg2rad(AMP * 0.8 * hb + trem(7, 0.15) + 1.0 * t / t[-1])
scl = 1 + 0.006 * hb + 0.004 * np.sin(2 * math.pi * 0.3 * t)
gts = []
frames = np.zeros((N, H, W, 3), np.uint8)
for k in range(N):
    # frame pixel p -> world pixel: world = S*(R*s*(p - c) + c + d) + offset
    c, s_ = math.cos(rot[k]), math.sin(rot[k])
    a, b = scl[k] * c, scl[k] * s_
    A = np.array([[a, -b], [b, a]]) * S
    ctr = np.array([W / 2, H / 2])
    offs = np.array([WW / 2, WH / 2]) - S * ctr
    tvec = S * (ctr - (np.array([[a, -b], [b, a]]) @ ctr) + np.array([px[k], py[k]])) + offs
    Mw = np.hstack([A, tvec[:, None]])
    # dst(p) = world(Mw p)
    img = cv2.warpAffine(world, Mw, (W, H), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    img = img + rng.normal(0, 0.008, img.shape)  # sensor noise
    frames[k] = np.clip(img * 255, 0, 255).astype(np.uint8)[..., ::-1]
    gts.append([float(a), float(b), float(px[k]), float(py[k])])
np.save(out + '.npy', frames)
json.dump({'W': W, 'H': H, 'fps': FPS, 'gt': gts, 'S': S}, open(out + '.json', 'w'))
print('ok', frames.shape)
