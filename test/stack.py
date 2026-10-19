"""Stacked still vs all-sharp truth on the focus-sweep clip: per depth band, compare the stacked still,
the centre frame, and the best single frame in the window (each lined up on the centre)."""
import asyncio, os, sys, base64, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, TRUTH, RAW, OUT, C = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await b.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
        await pg.goto('http://localhost:8765/stack.html'); await pg.wait_for_function('window.ready')
        if len(sys.argv) > 6: await pg.evaluate(f'window.EXTRA = {sys.argv[6]}')
        r = await pg.evaluate(f'run("{os.path.basename(CLIP)}", {C}, 0.3)')
        print(f"stacked {r['frames']} frames in {r['ms']/1000:.1f} s | grain sigma {r['sigma']:.4f} | errors {errs[:3]}")
        for k in ('after', 'before'): open(f'{OUT}_{k}.png', 'wb').write(base64.b64decode(r[k]))
        await b.close()
    truth = np.load(TRUTH)[C].astype(np.float32); frames = np.load(RAW).astype(np.float32)
    A, Bf = cv2.imread(OUT + '_after.png').astype(np.float32), cv2.imread(OUT + '_before.png').astype(np.float32)
    H = truth.shape[0]
    g = lambda x: cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    def align(x):   # line a raw frame up on the truth/centre with ECC (affine) for a fair "best single frame"
        warp = np.eye(2, 3, dtype=np.float32)
        try: _, warp = cv2.findTransformECC(cv2.GaussianBlur(g(truth), (0, 0), 2), cv2.GaussianBlur(g(x), (0, 0), 2), warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
        except cv2.error: pass
        return cv2.warpAffine(x, warp, (x.shape[1], x.shape[0]), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    win = [k for k in range(len(frames)) if abs(k - C) <= 9]
    aligned = [align(frames[k]) for k in win]
    ps = lambda x, y: 10 * np.log10(255 ** 2 / np.mean((x - y) ** 2))
    print('band | stacked | centre frame | best single frame in window   (dB vs all-sharp truth; higher = sharper & cleaner)')
    wins = 0
    for bnd in range(5):
        y0, y1 = int(bnd * H / 5) + 8, int((bnd + 1) * H / 5) - 8
        sl = lambda x: x[y0:y1, 16:-16]
        best = max(ps(sl(a), sl(truth)) for a in aligned)
        st, ce = ps(sl(A), sl(truth)), ps(sl(Bf), sl(truth))
        wins += st > best
        print(f'  {bnd}  |  {st:5.1f}  |    {ce:5.1f}     |   {best:5.1f}')
    print(f'stacked beats the best single frame in {wins}/5 bands')
    cv2.imwrite(OUT + '_cmp.png', np.vstack([np.hstack([Bf, A]), np.hstack([truth, np.zeros_like(truth)])])[::2, ::2])
asyncio.run(main())
