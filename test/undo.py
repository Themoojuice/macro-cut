"""Undo/redo: each settled edit (split, slider drag, frame, speed, stabilise level) is one step; undo/redo walk them
exactly, a new edit clears redo, history survives a reload but starts fresh per clip. Usage: undo.py clip.mp4"""
import asyncio, json, os, sys
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
S = 'window.__macrocut.state'
SNAP = f'JSON.stringify({{segs: {S}.segments.map(s => [+s.start.toFixed(3), +s.end.toFixed(3), s.speed ?? null, s.look ? (s.look.adj.exposure ?? 0) : null]), frame: {S}.frame, exp: ({S}.look.adj.exposure ?? 0), speed: {S}.speed, level: {S}.stab.level}})'

async def main():
    fails = []
    def check(name, ok):
        print(('PASS ' if ok else 'FAIL ') + name)
        if not ok: fails.append(name)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915})).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(800)
        snap = lambda: pg.evaluate(SNAP)
        settle = lambda: pg.wait_for_timeout(700)
        check('undo disabled on fresh clip', await pg.is_disabled('#btnUndo') and await pg.is_disabled('#btnRedo'))
        hist = [await snap()]
        # 1. split
        await pg.evaluate('window.__macrocut.seek(2.0)'); await pg.wait_for_timeout(200); await pg.click('#btnSplit'); await settle(); hist.append(await snap())
        # 2. a slider "drag": many input events with a finger down = one step
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookSliders')  # no-op click
        await pg.evaluate('window.__macrocut.seek(1.0)'); await pg.wait_for_timeout(200)
        box = await pg.locator('#lookSliders .control-row:nth-child(5) input').bounding_box()
        await pg.mouse.move(box['x'] + box['width'] * 0.5, box['y'] + box['height'] / 2); await pg.mouse.down()
        for k in range(12):
            await pg.mouse.move(box['x'] + box['width'] * (0.5 - 0.03 * k), box['y'] + box['height'] / 2); await pg.wait_for_timeout(120)
        await pg.wait_for_timeout(600)  # pause mid-drag with finger still down: must not commit
        await pg.mouse.up(); await settle(); hist.append(await snap())
        # 3. framing
        await pg.click('.dock-item[data-tab="tab-frame"]'); await pg.click('#btnRotR'); await settle(); hist.append(await snap())
        # 4. speed on this part
        await pg.click('.dock-item[data-tab="tab-speed"]'); await pg.click('#speedGroup .pill-btn[data-speed="0.5"]'); await settle(); hist.append(await snap())
        # 5. stabilise level (changes level + zoom budget; measuring may start, which is fine)
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.click('#stabLevel .pill-btn[data-level="mild"]'); await settle(); hist.append(await snap())
        check('5 distinct steps recorded', len(set(hist)) == 6 and await pg.evaluate('!document.getElementById("btnUndo").disabled'))
        print('  look edit exposure:', json.loads(hist[2])['segs'])
        for i in range(5, 0, -1):
            await pg.click('#btnUndo'); await pg.wait_for_timeout(300)
            now = await snap(); check(f'undo -> step {i - 1} ({(await pg.inner_text("#toast")).strip()})', now == hist[i - 1])
        check('undo disabled at start, redo enabled', await pg.is_disabled('#btnUndo') and not await pg.is_disabled('#btnRedo'))
        for i in range(1, 6):
            await pg.click('#btnRedo'); await pg.wait_for_timeout(300)
            check(f'redo -> step {i}', await snap() == hist[i])
        # keyboard + new edit clears redo
        await pg.keyboard.press('Control+z'); await pg.keyboard.press('Control+z'); await pg.wait_for_timeout(300)
        check('Ctrl+Z twice -> step 3', await snap() == hist[3])
        await pg.click('.dock-item[data-tab="tab-frame"]'); await pg.click('#btnRotL'); await settle()
        check('new edit clears redo', await pg.is_disabled('#btnRedo'))
        # auto-saved edit = state after undo, and a reopened clip starts with empty history
        await pg.wait_for_timeout(1200); want = await snap()
        await pg.reload(); await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(800)
        check('reopened clip restores the undone-to edit', await snap() == want)
        check('history fresh after reopen', await pg.is_disabled('#btnUndo'))
        check('no page errors', not errs); print('  errors:', errs)
        await pg.screenshot(path=os.environ.get('SHOT', '/dev/null') if os.environ.get('SHOT') else 'undo_shot.png')
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
