"""The browser can clear MacroCut's working copy of the clip (private storage, under space pressure) during a long
export. Reading must then carry on from the file as picked. Simulated by deleting the working copy after opening.
usage: copy_lost.py clip.mp4"""
import asyncio, os, sys
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
S = 'window.__macrocut.state'
async def main():
    fails = []
    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name)
        if not ok: fails.append(name)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []; warns = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: m.type == 'warning' and warns.append(m.text))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', sys.argv[1])
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(500)
        check('original kept alongside the working copy', await pg.evaluate(f'!!{S}.clip.original'))
        gone = await pg.evaluate('(async () => { const d = await (await navigator.storage.getDirectory()).getDirectoryHandle("clips"); const n = []; for await (const [k] of d.entries()) n.push(k); for (const k of n) await d.removeEntry(k); return n; })()')
        check(f'working copy deleted ({gone})', len(gone) == 1)
        await pg.click('#btnExport'); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|lost|stopped/i.test(document.getElementById("exportMsg").textContent)', timeout=600000)
        msg = await pg.inner_text('#exportMsg'); print('  ', msg[:150])
        check('export finishes from the original file', 'Saved to your phone' in msg)
        check('switch to the original was logged', any('reading the original' in w for w in warns))
        check('no page errors', not errs); print('  errors:', errs[:3])
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
