"""Stacked still vs its own single centre frame on a real clip (no ground truth): detail on the subject and grain in the background."""
import asyncio, os, sys, base64, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
NAME, C, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
RAD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await b.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: print('  console:', m.text) if m.type in ('info', 'error') else None)
        await pg.goto('http://localhost:8765/stack.html'); await pg.wait_for_function('window.ready')
        if len(sys.argv) > 5: await pg.evaluate(f'window.EXTRA = {sys.argv[5]}')
        r = await pg.evaluate(f'run("{NAME}", {C}, {RAD})', )
        print(f"stacked {r['frames']} frames in {r['ms']/1000:.0f} s | sigma {r['sigma']:.4f} | errors {errs[:3]}")
        for k in ('after', 'before'): open(f'{OUT}_{k}.png', 'wb').write(base64.b64decode(r[k]))
        await b.close()
    A, B = cv2.imread(OUT + '_after.png'), cv2.imread(OUT + '_before.png')
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = lambda x: cv2.Laplacian(x, cv2.CV_32F)
    H, W = B.shape[:2]
    # subject = the most detailed 25% of 32px tiles in the centre frame, background = least detailed 25%
    L = np.abs(lap(cv2.GaussianBlur(g(B), (0, 0), 1)))
    t = 32; tiles = [(y, x, L[y:y+t, x:x+t].mean()) for y in range(0, H - t, t) for x in range(0, W - t, t)]
    tiles.sort(key=lambda z: z[2]); q = len(tiles) // 4
    def m(img, sel, f): return np.mean([f(img[y:y+t, x:x+t]) for y, x, _ in sel])
    for nm, img in (('single', B), ('stacked', A)):
        G = g(img)
        det = m(G, tiles[-q:], lambda z: lap(z).var())
        grain = m(G, tiles[:q], lambda z: (z - cv2.GaussianBlur(z, (0, 0), 2)).std())
        print(f'{nm:8s} subject detail {det:8.1f} | background grain {grain:.2f}')
    d = np.abs(g(A) - g(B)); print(f'mean |stacked - single| {d.mean():.2f}')
    s = slice(H // 2 - 200, H // 2 + 200), slice(W // 2 - 300, W // 2 + 300)
    cv2.imwrite(OUT + '_cmp.png', np.hstack([B[s], A[s]]))
asyncio.run(main())
