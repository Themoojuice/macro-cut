"""Apply a stabilisation plan (per-frame M: output-norm -> source-norm) to frames with OpenCV,
write a video, and measure jitter independently (OpenCV LK + RANSAC partial affine)."""
import json, sys, numpy as np, cv2
gray_path, meta_path, plan_path, out = sys.argv[1:5]
meta = json.load(open(meta_path)); W, H = meta['W'], meta['H']; A = W / H
plan = json.load(open(plan_path)); M = np.array(plan['M']).reshape(-1, 4)
raw = np.fromfile(gray_path, np.uint8).reshape(-1, H, W)
def warp(img, m):
    a, b, tx, ty = m
    # output px (u,v) -> o = ((u+.5)/W-.5)*A, (v+.5)/H-.5 ; s = [[a,-b],[b,a]] o + t ; src px = s*H + (W/2,H/2) - .5
    L = np.array([[a, -b], [b, a]]) * H / H  # norm units are height units on both sides
    # src = H*(L @ o + t) + c - .5, o = (p + .5)/H - (A/2, .5)
    c = np.array([W / 2, H / 2])
    Mx = L.copy(); t = H * (L @ (0.5 / H - np.array([A / 2, 0.5])) + np.array([tx, ty])) + c - 0.5
    return cv2.warpAffine(img, np.hstack([Mx, t[:, None]]), (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
stab = np.stack([warp(f, m) for f, m in zip(raw, M)])
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W * 2, H), False)
for a, b in zip(raw, stab): vw.write(np.hstack([a, b]))
vw.release()
def jitter(frames):
    P = []; acc = np.eye(3)
    corners = np.array([[W*0.15, H*0.15], [W*0.85, H*0.15], [W*0.85, H*0.85], [W*0.15, H*0.85]], np.float32)
    path = [corners.copy()]; fails = 0
    for i in range(1, len(frames)):
        p0 = cv2.goodFeaturesToTrack(frames[i-1], 500, 0.005, 6, mask=None)
        if p0 is None: path.append(path[-1]); fails += 1; continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[i-1], frames[i], p0, None, winSize=(21, 21), maxLevel=4)
        g = st.ravel() == 1
        Mi, _ = cv2.estimateAffinePartial2D(p0[g], p1[g], method=cv2.RANSAC, ransacReprojThreshold=1.0) if g.sum() > 6 else (None, None)
        if Mi is None: path.append(path[-1]); fails += 1; continue
        d = corners @ Mi[:, :2].T + Mi[:, 2] - corners
        path.append(path[-1] + d)
    P = np.array(path)
    d1 = np.linalg.norm(np.diff(P, axis=0), axis=2).mean()
    d2 = np.linalg.norm(np.diff(P, n=2, axis=0), axis=2).mean()
    return d1, d2, fails
for name, fr in [('raw', raw), ('stabilised', stab)]:
    d1, d2, f = jitter(fr)
    print(f'{name:11s} motion/frame {d1:5.2f} px   jitter (frame-to-frame change in motion) {d2:5.2f} px   (independent tracker fails {f})')
