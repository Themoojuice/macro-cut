"""Independent stability metric (OpenCV): mean frame-to-frame motion of frame corners, via LK + robust partial affine."""
import cv2, numpy as np, sys
def motion(path):
    cap = cv2.VideoCapture(path); prev = None; ds = []; n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); n += 1
        if prev is not None:
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
            M, inl = cv2.estimateAffinePartial2D(p0[st == 1], p1[st == 1], method=cv2.RANSAC, ransacReprojThreshold=1.5)
            h, w = g.shape
            c = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            d = c @ M[:, :2].T + M[:, 2] - c
            ds.append(np.linalg.norm(d, axis=1).mean() / h * 540)   # normalise to 540p pixels
        prev = g
    return n, np.mean(ds), np.percentile(ds, 95)
for p in sys.argv[1:]:
    n, m, p95 = motion(p)
    print(f'{p.split("/")[-1]:28s} frames {n:4d}  mean jump {m:6.2f} px  p95 {p95:6.2f} px  (540p-equivalent)')
