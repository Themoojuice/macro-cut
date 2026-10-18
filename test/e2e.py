"""Drive the real app in headless Chromium: load clip, analyse, export, save output + screenshots."""
import asyncio, os, sys, time
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT, MODE = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else 'lock')
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, accept_downloads=True)
        pg = await ctx.new_page()
        errs = []
        pg.on('console', lambda m: (m.type in ('error', 'warning')) and errs.append(m.text))
        pg.on('pageerror', lambda e: errs.append('PAGEERROR ' + str(e)))
        await pg.goto(URL)
        await pg.screenshot(path=OUT + '_0empty.png')
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=30000)
        await pg.wait_for_timeout(800)
        await pg.screenshot(path=OUT + '_1loaded.png')
        await pg.click('.dock-item[data-tab="tab-stab"]')
        LEVEL = {'lock': 'max', 'smooth': 'medium'}.get(MODE, MODE)
        t0 = time.time()
        await pg.click(f'#stabLevel .pill-btn[data-level="{LEVEL}"]')
        await pg.wait_for_function('window.__macrocut.state.analysis && !window.__macrocut.state.analysing', timeout=300000)
        print('analysis s', round(time.time() - t0, 1))
        await pg.wait_for_timeout(500)
        await pg.screenshot(path=OUT + '_2analysed.png')
        print('status:', await pg.inner_text('#stabStatus'))
        await pg.click('.dock-item[data-tab="tab-look"]')
        await pg.click('#lookPresets .pill-btn[data-preset="pop"]')
        await pg.wait_for_timeout(300)
        await pg.screenshot(path=OUT + '_3look.png')
        await pg.click('#btnExport')
        await pg.wait_for_timeout(300)
        await pg.screenshot(path=OUT + '_4sheet.png')
        t0 = time.time()
        await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=600000)
        print('export s', round(time.time() - t0, 1), '|', await pg.inner_text('#exportMsg'))
        await pg.screenshot(path=OUT + '_5done.png')
        async with pg.expect_download() as dl:
            await pg.click('#btnSave')
        d = await dl.value
        await d.save_as(OUT + '.mp4')
        print('saved', OUT + '.mp4')
        print('console errors:', errs[:10])
        await b.close()
asyncio.run(main())
