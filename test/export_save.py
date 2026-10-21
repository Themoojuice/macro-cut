"""Export never loses the video: it downloads itself when done, stays one tap away under Export after Close
and after a reload, and the done message says where the time went. Usage: export_save.py clip.mp4"""
import asyncio, os, sys
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
S = 'window.__macrocut.state'

async def main():
    fails = []
    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name)
        if not ok: fails.append(name)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []; downloads = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('download', lambda d: downloads.append(d.suggested_filename))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(500)
        await pg.evaluate(f'{S}.segments = [{{start: 0, end: 2}}]')
        await pg.click('#btnExport')
        check('no last-video card before first export', not await pg.is_visible('#lastExport'))
        await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=600000)
        await pg.wait_for_timeout(500)
        msg = await pg.inner_text('#exportMsg'); print('  ', msg)
        check('downloaded by itself, no tap', len(downloads) == 1 and downloads[0].endswith('.mp4'))
        check('message says saved', 'Saved to your phone' in msg)
        await pg.click('#btnExportCancel')   # the fatal "Close"
        await pg.click('#btnExport')
        check('last video card shows after Close', await pg.is_visible('#lastExport'))
        async with pg.expect_download() as dl: await pg.click('#btnLastSave')
        check('card Save downloads it again', (await dl.value).suggested_filename == downloads[0])
        await pg.reload(); await pg.wait_for_timeout(1000)
        await pg.set_input_files('#fileInput', CLIP); await pg.wait_for_function(f'{S}.clip', timeout=60000)
        await pg.click('#btnExport'); await pg.wait_for_timeout(300)
        check('last video survives reload', await pg.is_visible('#lastExport') and downloads[0] in await pg.inner_text('#lastExportInfo'))
        async with pg.expect_download() as dl: await pg.click('#btnLastSave')
        path = await (await dl.value).path()
        check('reloaded copy is a real video', os.path.getsize(path) > 10000)
        check('no page errors', not errs); print('  errors:', errs)
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
