"""Hard, realistic-ish macro clip: big slow handheld wander + heartbeat + tremor + rotation wander,
focus breathing, a spider that walks/turns independently, blur spans (focus loss), sensor noise.
Ground truth = camera motion relative to the static leaf/background.
usage: make_hard.py out W H N FPS [wander_scale]"""
import numpy as np, cv2, json, sys, math
rng = np.random.default_rng(11)
out = sys.argv[1]; W, H, N, FPS = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
WS = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
S = 2; WW, WH = int(W * S * 1.8), int(H * S * 1.8)
bg = cv2.GaussianBlur(rng.random((WH // 30, WW // 30, 3)).astype(np.float32), (0, 0), 1)
bg = cv2.resize(bg, (WW, WH), interpolation=cv2.INTER_CUBIC)
bg = cv2.GaussianBlur(bg, (0, 0), 30) * 0.5 + 0.25
leaf = np.zeros((WH, WW), np.float32)
cv2.ellipse(leaf, (WW // 2 - 80, WH // 2 + 40), (int(WW * 0.42), int(WH * 0.28)), -12, 0, 360, 1, -1)
leaf = cv2.GaussianBlur(leaf, (0, 0), 4)
# leaf texture: cells + veins, with depth-of-field falloff (sharp centre band, blurrier far edge)
cells = rng.random((WH // 6, WW // 6)).astype(np.float32)
cells = cv2.resize(cells, (WW, WH), interpolation=cv2.INTER_NEAREST)
cells = cv2.GaussianBlur(cells, (0, 0), 2.0)
veins = np.zeros((WH, WW), np.float32)
for i in range(40):
    p = (int(rng.integers(0, WW)), int(rng.integers(0, WH)))
    cv2.line(veins, p, (p[0] + int(rng.integers(-500, 500)), p[1] + int(rng.integers(-300, 300))), 1, 3)
veins = cv2.GaussianBlur(veins, (0, 0), 1.5)
lc = np.dstack([0.25 + 0.25 * cells - 0.1 * veins, 0.55 + 0.3 * cells - 0.15 * veins, 0.2 + 0.15 * cells])
yy = np.abs(np.arange(WH)[:, None] - WH * 0.55) / WH
dof = np.clip((yy - 0.12) * 18, 0, 6)  # blur sigma map
sharp = lc.copy(); blurred = cv2.GaussianBlur(lc, (0, 0), 6)
wgt = (dof / 6)[..., None]
lc = sharp * (1 - wgt) + blurred * wgt
world = bg * (1 - leaf[..., None]) + lc * leaf[..., None]
world = np.clip(world, 0, 1).astype(np.float32)
# spider sprite (RGBA), drawn at world scale
SPS = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
SP = int(260 * S // 2 * SPS)
spr = np.zeros((SP, SP, 3), np.float32); alpha = np.zeros((SP, SP), np.float32)
c = SP // 2
q = SPS
cv2.ellipse(alpha, (c - int(20*q), c), (int(60*q), int(42*q)), 0, 0, 360, 1, -1); cv2.circle(alpha, (c + int(50*q), c), int(34*q), 1, -1)
for i in range(8):
    a = -1.1 + (i % 4) * 0.45 + (0 if i < 4 else math.pi)
    cv2.line(alpha, (c, c), (int(c + math.cos(a) * 120 * q), int(c + math.sin(a) * 105 * q)), 1, int(7 * q))
hair = cv2.GaussianBlur(rng.random((SP, SP)).astype(np.float32), (0, 0), 0.8)
hair = np.clip(hair * 1.6 - 0.3, 0, 1)
spr[:] = np.dstack([0.15 + 0.6 * hair, 0.05 + 0.25 * hair, 0.25 + 0.6 * hair])
for ey in (-14, 14):
    cv2.circle(spr, (c + int(72*q), c + int(ey*q)), int(9*q), (0.02, 0.02, 0.02), -1); cv2.circle(spr, (c + int(74*q), c + int((ey - 3)*q)), int(3*q), (1, 1, 1), -1)
alpha = cv2.GaussianBlur(alpha, (0, 0), 1)
t = np.arange(N) / FPS
# spider path in world px: sits, turns, walks, sits
sx = np.where(t < 4, 0, np.minimum((t - 4) * 40 * S, 160 * S)) + WW / 2 - 60 * S
sy = WH / 2 + np.where(t < 4, 0, np.minimum((t - 4) * 15 * S, 60 * S))
sa = np.where(t < 2, 0, np.minimum((t - 2) * 25, 50)) + 6 * np.sin(t * 2.3)  # degrees
# camera: wander + heartbeat + tremor
def smooth_noise(sig, scale):
    x = rng.standard_normal(N); k = int(sig * FPS)
    x = cv2.GaussianBlur(x.reshape(-1, 1), (1, 0), k).ravel() if k > 0 else x
    return x / (np.std(x) + 1e-9) * scale
hb = np.array([math.exp(-(((tk * 1.15) % 1 - 0.1) / 0.06) ** 2) - 0.6 * math.exp(-(((tk * 1.15) % 1 - 0.25) / 0.07) ** 2) for tk in t])
SMOOTH = len(sys.argv) > 8 and sys.argv[8] == 'smooth'   # smooth camera only (for interpolation ground truth)
J = 0.0 if SMOOTH else 1.0
px = WS * smooth_noise(1.2, 0.07 * W) + J * (9 * hb + 1.5 * rng.standard_normal(N) + 2 * np.sin(2 * math.pi * 9 * t))
py = WS * smooth_noise(1.0, 0.06 * H * 1.6) + J * (12 * hb + 1.5 * rng.standard_normal(N) + 2 * np.sin(2 * math.pi * 8 * t + 1))
rot = np.deg2rad(WS * smooth_noise(1.5, 1.8) + J * (0.6 * hb + 0.15 * rng.standard_normal(N)))
scl = 1 + WS * smooth_noise(2.0, 0.02) + 0.004 * hb
blurspans = [(int(N * 0.30), int(N * 0.30) + int(0.5 * FPS)), (int(N * 0.72), int(N * 0.72) + int(0.3 * FPS))]
frames = np.zeros((N, H, W, 3), np.uint8); gts = []
for k in range(N):
    wk = world.copy()
    M = cv2.getRotationMatrix2D((c, c), -sa[k], 1.0); M[0, 2] += sx[k] - c; M[1, 2] += sy[k] - c
    a_w = cv2.warpAffine(alpha, M, (WW, WH)); s_w = cv2.warpAffine(spr, M, (WW, WH))
    wk = wk * (1 - a_w[..., None]) + s_w * a_w[..., None]
    co, si = math.cos(rot[k]), math.sin(rot[k]); a, b = scl[k] * co, scl[k] * si
    R = np.array([[a, -b], [b, a]]); ctr = np.array([W / 2, H / 2])
    tvec = S * (ctr - R @ ctr + np.array([px[k], py[k]])) + np.array([WW / 2, WH / 2]) - S * ctr
    Mw = np.hstack([R * S, tvec[:, None]])
    img = cv2.warpAffine(wk, Mw, (W, H), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    for (b0, b1) in blurspans:
        if b0 <= k < b1: img = cv2.GaussianBlur(img, (0, 0), 7)
    img = img + rng.normal(0, 0.012, img.shape)
    frames[k] = np.clip(img * 255, 0, 255).astype(np.uint8)[..., ::-1]
    gts.append([float(a), float(b), float(px[k]), float(py[k])])
np.save(out + '.npy', frames)
frames.mean(-1).astype(np.uint8).tofile(out + '.gray')
json.dump({'W': W, 'H': H, 'fps': FPS, 'gt': gts, 'blur': blurspans}, open(out + '.json', 'w'))
print('ok', frames.shape)
