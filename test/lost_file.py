"""The phone can take back read access to the picked video mid-export. MacroCut works from its own copy, so
deleting the original after opening must not break preview, stabilise or export. Usage: lost_file.py clip.mp4"""
import asyncio, os, sys, shutil, tempfile
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
SRC = sys.argv[1]
S = 'window.__macrocut.state'

async def main():
    fails = []
    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name)
        if not ok: fails.append(name)
    tmp = tempfile.mkdtemp(); clip = os.path.join(tmp, 'spood.mp4'); shutil.copy(SRC, clip)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []; downloads = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('download', lambda d: downloads.append(d))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', clip)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(500)
        names = await pg.evaluate('(async () => { const d = await (await navigator.storage.getDirectory()).getDirectoryHandle("clips"); const o = []; for await (const [n] of d.entries()) o.push(n); return o; })()')
        check(f'working copy made ({names})', len(names) == 1 and names[0].endswith('spood.mp4'))
        check('clip name kept', await pg.evaluate(f'{S}.clip.name') == 'spood.mp4' if await pg.evaluate(f'"name" in {S}.clip') else True)
        os.remove(clip)   # the phone takes the file back
        await pg.evaluate(f'{S}.segments = [{{start: 0, end: 2}}]')
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.evaluate('window.__macrocut.setStabOn(true)')
        await pg.wait_for_function(f'{S}.analysis && !{S}.analysing', timeout=300000)
        check('stabilise measured after original vanished', True)
        await pg.click('#btnExport'); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|lost|stopped/i.test(document.getElementById("exportMsg").textContent)', timeout=600000)
        msg = await pg.inner_text('#exportMsg'); print('  ', msg[:160])
        check('export works after original vanished', 'Saved to your phone' in msg and len(downloads) == 1)
        check('no page errors', not errs); print('  errors:', errs)
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
