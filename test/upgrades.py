"""Auto-trim + best-shots through the real app."""
import asyncio, os, sys, time, cv2, numpy as np
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=120000)
        t0 = time.time(); await pg.click('#btnAutoTrim')
        await pg.wait_for_function('document.getElementById("btnAutoTrim").textContent.includes("Undo")', timeout=3600000)
        print(f'auto-trim in {time.time()-t0:.0f}s ->', await pg.evaluate('JSON.stringify(window.__macrocut.state.segments.map(s=>[+s.start.toFixed(2), +s.end.toFixed(2)]))'), '|', await pg.inner_text('#toast'), flush=True)
        await pg.screenshot(path=OUT + '_trim.png')
        await pg.click('#btnStill'); await pg.click('#btnFindBest')
        await pg.wait_for_function('document.getElementById("btnShotsSaveAll").style.display !== "none" || document.getElementById("shotsMsg").textContent.startsWith("Couldn")', timeout=3600000)
        print('shots:', await pg.inner_text('#shotsMsg'), flush=True)
        await pg.screenshot(path=OUT + '_shots.png')
        async with pg.expect_download() as dl: await pg.click('.shot button')
        path = OUT + '_shot.png'; await (await dl.value).save_as(path)
        im = cv2.imread(path); print('saved still', im.shape)
        print('errors', errs); await b.close()
asyncio.run(main())
