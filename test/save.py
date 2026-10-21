"""Save/restore: edit a clip, reload the app, reopen the same clip, check everything is back; then saved looks + start over."""
import asyncio, os, sys, time
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OTHER = sys.argv[1], sys.argv[2]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915})
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: m.type == 'error' and errs.append(m.text + ' @ ' + str(m.location)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        S = 'window.__macrocut.state'
        async def load(path):
            await pg.set_input_files('#fileInput', path)
            await pg.wait_for_function(f'{S}.clip && {S}.clip.name === "{os.path.basename(path)}"', timeout=60000)
            await pg.wait_for_timeout(500)
        await load(CLIP)
        # edit: split at 3 s, darken + slow part 2, frame rotate, stabilise Max
        await pg.evaluate('window.__macrocut.seek(3.0)'); await pg.wait_for_timeout(200)
        await pg.click('#btnSplit'); await pg.evaluate('window.__macrocut.seek(4.0)'); await pg.wait_for_timeout(200)
        await pg.click('.dock-item[data-tab="tab-look"]')
        await pg.evaluate('''() => { const r = [...document.querySelectorAll('#lookSliders .control-row')].find(r => r.textContent.trim().startsWith('Brightness'));
          const el = r.querySelector('input'); el.value = -40; el.dispatchEvent(new Event('input', {bubbles: true})); }''')
        await pg.evaluate('window.__macrocut.setSpeed(0.5)')
        await pg.evaluate(f'{S}.frame.rot = 4.5')
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.click('#stabLevel .pill-btn[data-level="max"]')
        await pg.wait_for_function(f'{S}.analysis && !{S}.analysing', timeout=300000)
        await pg.wait_for_timeout(1500)
        before = await pg.evaluate(f'JSON.stringify({{segs: {S}.segments, rot: {S}.frame.rot, on: {S}.stab.on, level: {S}.stab.level, n: {S}.analysis.C.length}})')
        print('saved state:', before[:220], '|', await pg.inner_text('#saveState'))
        # reload the whole app and reopen the same clip
        await pg.reload(); await load(CLIP)
        t0 = time.time()
        after = await pg.evaluate(f'JSON.stringify({{segs: {S}.segments, rot: {S}.frame.rot, on: {S}.stab.on, level: {S}.stab.level, n: {S}.analysis && {S}.analysis.C.length}})')
        print('restored:   ', after[:220])
        import json; norm = lambda x: json.dumps(json.loads(x), sort_keys=True)
        print('identical:', norm(before) == norm(after), '| toast:', (await pg.inner_text('#toast'))[:120], '| chip:', await pg.inner_text('#compareTag'))
        # saved look: save part 2's look, apply to another clip
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.evaluate('window.__macrocut.seek(4.0)'); await pg.wait_for_timeout(300)
        await pg.click('#myLooks .pill-btn:last-child'); await pg.fill('#lookName', 'Moody'); await pg.click('#btnSaveLookGo')
        await load(OTHER)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#myLooks .pill-btn:first-child')
        print('look on other clip, exposure =', await pg.evaluate('window.__macrocut.lookNow().exposure'), '| other clip segments:', await pg.evaluate(f'{S}.segments.length'))
        # start over on first clip
        await load(CLIP); await pg.click('.dock-item[data-tab="tab-trim"]'); await pg.click('#btnStartOver'); await pg.wait_for_timeout(1500)
        await pg.reload(); await load(CLIP)
        print('after start over + reload: segments', await pg.evaluate(f'{S}.segments.length'), 'stab on', await pg.evaluate(f'{S}.stab.on'), 'analysis kept', await pg.evaluate(f'!!{S}.analysis'))
        print('errors', errs); await b.close()
asyncio.run(main())
