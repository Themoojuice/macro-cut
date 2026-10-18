"""Drive the real app to produce a finished edit: trim, stabilise, look, export, save."""
import asyncio, os, sys, json, time
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT, CFG = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, device_scale_factor=2, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=120000)
        print('loaded:', await pg.inner_text('#metaBadge'), await pg.evaluate('JSON.stringify(window.__macrocut.state.previewInfo)'), flush=True)
        await pg.evaluate('(c) => { const s = window.__macrocut.state; s.segments = c.segments; Object.assign(s.stab, c.stab); Object.assign(s.look, c.look); }', CFG)
        await pg.click('.dock-item[data-tab="tab-stab"]')
        t0 = time.time(); await pg.click(f'#stabLevel .pill-btn[data-level="{CFG["stab"]["level"]}"]')
        await pg.wait_for_function('window.__macrocut.state.analysis && !window.__macrocut.state.analysing', timeout=3600000)
        print(f'analysed in {time.time()-t0:.0f}s:', await pg.inner_text('#stabStatus'), flush=True)
        await pg.evaluate('window.__macrocut.recompute()')
        json.dump({'M': list(await pg.evaluate('Array.from(window.__macrocut.state.plan.M)')), 'times': list(await pg.evaluate('Array.from(window.__macrocut.state.clip.times)'))}, open(OUT + '_plan.json', 'w'))
        await pg.screenshot(path=OUT + '_stab.png')
        await pg.click('#btnExport'); await pg.wait_for_timeout(300)
        t0 = time.time(); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=7200000)
        print(f'export {time.time()-t0:.0f}s:', await pg.inner_text('#exportMsg'), flush=True)
        async with pg.expect_download() as dl: await pg.click('#btnSave')
        await (await dl.value).save_as(OUT + '.mp4')
        print('errors', errs); await b.close()
asyncio.run(main())
