"""Stacked stills through the real UI: find sharpest shots, stack one, save both versions, compare."""
import asyncio, os, sys, time, cv2, numpy as np
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=120000)
        await pg.evaluate('window.__macrocut.seek(21.4)'); await pg.wait_for_timeout(500)
        await pg.click('#btnStill')
        async with pg.expect_download() as dl: await pg.click('#btnThisFrame')
        await (await dl.value).save_as(OUT + '_single.png')
        t0 = time.time(); await pg.click('#btnStackHere')
        await pg.wait_for_function('document.getElementById("shotsMsg").textContent.startsWith("Stacked") || document.getElementById("shotsMsg").textContent.startsWith("Couldn")', timeout=3000000)
        print(f'{time.time()-t0:.0f}s |', await pg.inner_text('#shotsMsg'), flush=True)
        await pg.screenshot(path=OUT + '_ui.png')
        async with pg.expect_download() as dl: await pg.click('.shot .shot-btns button:last-child')
        await (await dl.value).save_as(OUT + '_stacked.png')
        await pg.click('.shot > div'); await pg.wait_for_timeout(200)
        print('tag after tap:', await pg.inner_text('.shot-tag'))
        print('errors', errs[:4]); await b.close()
    a, s = cv2.imread(OUT + '_single.png'), cv2.imread(OUT + '_stacked.png')
    print('sizes', a.shape, s.shape)
    g = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # spider region = where the single frame has most detail; background = least
    e = cv2.blur(cv2.Laplacian(g(a), cv2.CV_32F) ** 2, (61, 61))
    fg, bgm = e > np.percentile(e, 85), e < np.percentile(e, 30)
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 1.5)
    for n, x in (('single frame', a), ('stacked', s)):
        L = cv2.Laplacian(g(x), cv2.CV_32F)
        print(f'{n:13s} detail on spider (Laplacian var) {L[fg].var():7.1f} | background grain {hp(g(x))[bgm].std():.2f}')
    ys, xs = np.where(fg); cy, cx = int(np.median(ys)), int(np.median(xs))
    crop = lambda x: x[max(0, cy - 200):cy + 200, max(0, cx - 300):cx + 300]
    cv2.imwrite(OUT + '_cmp.png', np.hstack([crop(a), crop(s)]))
asyncio.run(main())
