"""Perceived subject shake: track the spider (CSRT) through raw and stabilised renders,
report high-frequency (judder) RMS of its on-screen position, in output pixels."""
import json, sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0])
gray, meta_p, k0, k1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
plans = sys.argv[5:]
meta = json.load(open(meta_p)); W, H = meta['W'], meta['H']; A = W / H
raw = np.fromfile(gray, np.uint8).reshape(-1, H, W)
BOX = (190, 120, 210, 130)
def warpM(m):
    a, b, tx, ty = m; L = np.array([[a, -b], [b, a]]); c = np.array([W / 2, H / 2])
    t = H * (L @ (0.5 / H - np.array([A / 2, 0.5])) + np.array([tx, ty])) + c - 0.5
    return np.hstack([L, t[:, None]])
def track(frames, box):
    """Frame-to-frame NCC template tracking of the subject patch, sub-pixel peak."""
    x, y, w, h = [int(v) for v in box]; cx, cy = x + w / 2, y + h / 2; P = [[cx, cy]]
    for i in range(1, len(frames)):
        a, b = frames[i - 1].astype(np.float32), frames[i].astype(np.float32)
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x0 = min(max(0, x0), W - w); y0 = min(max(0, y0), H - h)
        tpl = a[y0:y0 + h, x0:x0 + w]; R = 40
        sx0, sy0 = max(0, x0 - R), max(0, y0 - R); sx1, sy1 = min(W, x0 + w + R), min(H, y0 + h + R)
        res = cv2.matchTemplate(b[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
        _, _, _, (mx, my) = cv2.minMaxLoc(res)
        def sub(v0, v1, v2): d = v0 - 2 * v1 + v2; return 0 if abs(d) < 1e-9 else 0.5 * (v0 - v2) / d
        dx = sub(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0
        dy = sub(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0
        nx, ny = sx0 + mx + dx, sy0 + my + dy
        cx += nx - x0; cy += ny - y0; P.append([cx, cy])
    return np.array(P)
def judder(P, sig=7.5):
    sm = np.stack([cv2.GaussianBlur(P[:, i].reshape(-1, 1).astype(np.float64), (1, 0), sig).ravel() for i in range(2)], 1)
    return np.sqrt(np.mean(np.sum((P - sm) ** 2, 1)))
fr = raw[k0:k1]
P = track(fr, BOX); print(f'raw            subject judder {judder(P):5.2f} px')
for p in plans:
    M = np.array(json.load(open(p))['M']).reshape(-1, 4)
    Ws = [warpM(m) for m in M[k0:k1]]
    st = np.stack([cv2.warpAffine(f, w, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) for f, w in zip(fr, Ws)])
    # map the start box into stabilised coords (inverse of output->source)
    Wi = cv2.invertAffineTransform(Ws[0]); c = Wi @ np.array([BOX[0] + BOX[2] / 2, BOX[1] + BOX[3] / 2, 1])
    z = 1 / np.sqrt(abs(np.linalg.det(Ws[0][:, :2])))
    bw, bh = BOX[2] * z, BOX[3] * z
    Ps = track(st, (int(c[0] - bw / 2), int(c[1] - bh / 2), int(bw), int(bh)))
    print(f'{p.split("/")[-1]:14s} subject judder {judder(Ps):5.2f} px  (zoom {z:.2f}×)')
