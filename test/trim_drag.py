"""Trim by dragging: the edge follows the whole drag (it used to die after the first move), grabbing near the
edge works, sliding up gives fine control, edges stop at limits, one drag = one undo step, Start/End here buttons.
Usage: trim_drag.py clip.mp4"""
import asyncio, os, sys, json
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
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915})).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(800)
        D = await pg.evaluate(f'{S}.clip.duration')
        segs = lambda: pg.evaluate(f'{S}.segments.map(s => [s.start, s.end])')
        box = await pg.locator('#track').bounding_box(); y = box['y'] + box['height'] * 0.6
        X = lambda t: box['x'] + box['width'] * t / D
        settle = lambda: pg.wait_for_timeout(700)
        async def drag(x0, x1, y1=None, steps=12):
            await pg.mouse.move(x0, y); await pg.mouse.down()
            for k in range(1, steps + 1):
                await pg.mouse.move(x0 + (x1 - x0) * k / steps, y + ((y1 if y1 is not None else y) - y) * k / steps); await pg.wait_for_timeout(30)
            bubble = await pg.is_visible('#trimBubble'); btxt = await pg.inner_text('#trimBubble') if bubble else ''
            await pg.mouse.up(); await settle(); return bubble, btxt
        # 1. end handle all the way to the middle, many small moves
        bub, txt = await drag(X(D) - 7, X(D / 2))
        s = await segs(); print('  after end drag:', s, '| bubble:', txt.replace('\n', ' '))
        check(f'end follows the whole drag (end {s[0][1]:.2f} ≈ {D/2:.2f})', abs(s[0][1] - D / 2) < D * 0.03)
        check('time bubble shows while dragging', bub and 'End' in txt)
        check('bubble hides after', not await pg.is_visible('#trimBubble'))
        check('preview shows the edge frame', abs(await pg.evaluate(f'{S}.mediaTime') - s[0][1]) < 0.1)
        # 2. grab 20 px inside the start edge (not on the tab) and drag right
        await drag(X(0) + 20, X(D * 0.2) + 20)
        s = await segs(); check(f'grab near the edge works (start {s[0][0]:.2f} ≈ {D*0.2:.2f})', abs(s[0][0] - D * 0.2) < D * 0.03)
        # 3. fine control: same 100 px finger move with the finger 300 px above → ~10× less
        e0 = (await segs())[0][1]
        await pg.mouse.move(X(e0), y); await pg.mouse.down(); await pg.mouse.move(X(e0), y - 300, steps=6)
        for k in range(1, 11): await pg.mouse.move(X(e0) - 10 * k, y - 300); await pg.wait_for_timeout(30)
        await pg.mouse.up(); await settle()
        e1 = (await segs())[0][1]; coarse = 100 * D / box['width']
        check(f'fine control slows the edge ({e0 - e1:.3f} s vs {coarse:.3f} s coarse)', 0 < e0 - e1 < coarse * 0.2)
        # 4. limits: drag start past the end
        s = await segs(); await drag(X(s[0][0]), X(D) + 50)
        s = await segs(); check(f'start stops before the end ({s[0][1] - s[0][0]:.2f} s left)', 0.1 < s[0][1] - s[0][0] < 0.3)
        # 5. undo walks back one drag at a time
        await pg.click('#btnUndo'); await settle()
        s2 = await segs(); check('undo restores the previous drag only', abs(s2[0][1] - e1) < 1e-6 and s2[0][0] < s2[0][1] - 1)
        # 6. Start here / End here
        await pg.evaluate('window.__macrocut.seek(1.0)'); await pg.wait_for_timeout(200); await pg.click('#btnStartHere'); await settle()
        await pg.evaluate('window.__macrocut.seek(2.5)'); await pg.wait_for_timeout(200); await pg.click('#btnEndHere'); await settle()
        s = await segs(); check(f'Start/End here ({s})', abs(s[0][0] - 1.0) < 0.05 and abs(s[0][1] - 2.5) < 0.05)
        # 7. with two parts, dragging one part's edge can't cross the other
        await pg.click('#btnResetCuts'); await settle()
        await pg.evaluate('window.__macrocut.seek(3.0)'); await pg.wait_for_timeout(200); await pg.click('#btnSplit'); await settle()
        await pg.evaluate('window.__macrocut.seek(1.0)'); await pg.wait_for_timeout(200); await pg.click('#btnDelete'); await settle()
        s = await segs(); print('  two-part setup:', s)
        await pg.click('#btnResetCuts'); await settle()
        await pg.evaluate(f'{S}.segments = [{{start: 0, end: 2}}, {{start: 3, end: {D}}}]; window.__macrocut.seek(1)'); await pg.wait_for_timeout(300)
        await drag(X(2), X(4.5))
        s = await segs(); check(f'edge stops at the next part ({s[0][1]:.2f} ≤ 3)', abs(s[0][1] - 3) < 0.02 and len(s) == 2)
        check('no page errors', not errs); print('  errors:', errs)
        if os.environ.get('SHOT'): await pg.screenshot(path=os.environ['SHOT'])
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
