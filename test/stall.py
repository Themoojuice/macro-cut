"""Preview survives the phone taking the video decoder away. Usage: stall.py clip.mp4
1. decoder vanishes mid-playback -> playback resumes from the same spot by itself
2. play then immediately pause -> no bogus 'Playback blocked' message
3. decoder vanishes while paused -> pressing play brings it back"""
import asyncio, os, sys
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
S = 'window.__macrocut.state'
KILL = 'const v = document.getElementById("video"); v.removeAttribute("src"); v.load();'

async def main():
    fails = []
    def check(name, ok, extra=''):
        print(('PASS ' if ok else 'FAIL ') + name, extra)
        if not ok: fails.append(name)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915})).new_page(); errs = []; toasts = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.expose_function('logToast', lambda t: toasts.append(t))
        await pg.evaluate('new MutationObserver(() => window.logToast(document.getElementById("toast").textContent)).observe(document.getElementById("toast"), {childList: true, characterData: true, subtree: true})')
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(800)
        vt = lambda: pg.evaluate('document.getElementById("video").currentTime')

        # 1
        await pg.click('#btnPlay'); await pg.wait_for_timeout(1200)
        t_kill = await pg.evaluate(f'{S}.mediaTime'); await pg.evaluate(KILL)
        await pg.wait_for_timeout(5500)
        a = await vt(); await pg.wait_for_timeout(800); b2 = await vt()
        check('1. resumes after decoder loss', await pg.evaluate(f'{S}.playing') and b2 > a > t_kill - 0.1, f'(killed at {t_kill:.2f}s, now {a:.2f}->{b2:.2f}s)')
        check('1. told the user it restarted', any('restarting the preview' in t for t in toasts))
        await pg.click('#btnPlay'); await pg.wait_for_timeout(300)

        # 2
        toasts.clear()
        await pg.evaluate('window.__macrocut.seek(1.0)'); await pg.wait_for_timeout(300)
        await pg.evaluate('document.getElementById("btnPlay").click(); document.getElementById("btnPlay").click();')
        await pg.wait_for_timeout(800)
        check('2. play+pause shows no "Playback blocked"', not any('blocked' in t for t in toasts), str(toasts))

        # 3
        await pg.evaluate(KILL); await pg.wait_for_timeout(500)
        await pg.click('#btnPlay'); await pg.wait_for_timeout(3500)
        a = await vt(); await pg.wait_for_timeout(800); b2 = await vt()
        check('3. play after decoder lost while paused', await pg.evaluate(f'{S}.playing') and b2 > a, f'({a:.2f}->{b2:.2f}s)')
        check('no page errors', not errs, str(errs))
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
