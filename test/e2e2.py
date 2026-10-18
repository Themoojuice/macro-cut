"""Scenario 2: rotation metadata, 4:5 crop, trimmed middle, half speed, smooth mode."""
import asyncio, os, sys, time
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append('PAGEERROR ' + str(e)))
        pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
        await pg.goto('http://localhost:8765/app/index.html')
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=30000)
        print('clip', await pg.evaluate('(({dispW,dispH,rotation})=>({dispW,dispH,rotation}))(window.__macrocut.state.clip)'), 'preview', await pg.evaluate('window.__macrocut.state.previewInfo'))
        # cut out 2.0–3.0 s via the UI: seek, split, seek, split, select middle, delete
        for t in (2.0, 3.0):
            await pg.evaluate(f'document.getElementById("video").currentTime={t}; window.__macrocut.state.mediaTime={t}')
            await pg.wait_for_timeout(200); await pg.click('#btnSplit')
        await pg.evaluate('window.__macrocut.state.sel = 1')
        await pg.click('#btnDelete')
        print('segments', await pg.evaluate('JSON.stringify(window.__macrocut.state.segments)'))
        await pg.click('.dock-item[data-tab="tab-frame"]'); await pg.click('#aspectGroup .pill-btn[data-aspect="4:5"]')
        await pg.click('.dock-item[data-tab="tab-speed"]'); await pg.click('#speedGroup .pill-btn[data-speed="0.5"]')
        print('speed info:', await pg.inner_text('#speedInfo'))
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.click('#stabLevel .pill-btn[data-level="medium"]')
        await pg.wait_for_function('window.__macrocut.state.analysis && !window.__macrocut.state.analysing', timeout=300000)
        print('status:', await pg.inner_text('#stabStatus'))
        await pg.wait_for_timeout(300); await pg.screenshot(path=OUT + '_ui.png')
        await pg.click('#btnExport'); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=600000)
        print(await pg.inner_text('#exportMsg'))
        async with pg.expect_download() as dl: await pg.click('#btnSave')
        await (await dl.value).save_as(OUT + '.mp4')
        await pg.click('#btnExportCancel')
        # still
        await pg.click('#btnStill')
        async with pg.expect_download() as dl: await pg.click('#btnThisFrame')
        await (await dl.value).save_as(OUT + '_still.png')
        print('errors', errs[:8]); await b.close()
asyncio.run(main())
