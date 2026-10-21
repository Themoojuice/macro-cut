"""Look tab: presets toggle on/off, Off is a lossless bypass, sliders start at 0 (= preset), double-tap and
Reset put them back, labels match what's rendered, undo covers preset taps, new clips start fresh.
Usage: look_ui.py clip.mp4"""
import asyncio, json, os, sys
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
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, has_touch=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")')
        # an old-style saved look and an old-style remembered "Off" look must not break anything
        await pg.evaluate('''localStorage.setItem("macrocut.looks", JSON.stringify([{name: "Old punchy", look: {cas: 0.55, detail: 1.1, radius: 1.4, denoise: 0.25, exposure: 0.5, contrast: 0.12, shadows: 0.25, highlights: 0.3, saturation: 0.15, vibrance: 0.45, warmth: 0, tint: 0}}]));
          localStorage.setItem("macrocut.v5", JSON.stringify({look: {cas: 0, detail: 0, radius: 1.5, denoise: 0, exposure: 0, contrast: 0, shadows: 0, highlights: 0, saturation: 0, vibrance: 0, warmth: 0, tint: 0}}))''')
        await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(800)
        await pg.click('.dock-item[data-tab="tab-look"]')
        look = lambda: pg.evaluate(f'JSON.stringify({S}.look)')
        active = lambda: pg.evaluate('[...document.querySelectorAll("#lookPresets .pill-btn.active")].map(b => b.dataset.preset).join(",")')
        vals = lambda: pg.evaluate('[...document.querySelectorAll("#lookSliders input")].map(i => +i.value)')
        labels = lambda: pg.evaluate('[...document.querySelectorAll("#lookSliders .control-label b")].map(b => b.textContent)')
        settle = lambda: pg.wait_for_timeout(600)
        check('new clip starts at Natural, not a remembered Off', await active() == 'natural')
        check('all sliders start at 0', all(v == 0 for v in await vals()))
        check('labels read 0 at start', all(l in ('0', 'medium') for l in await labels()))
        # presets toggle
        await pg.click('#lookPresets [data-preset="natural"]'); await settle()
        check('tap lit Natural -> no preset', await active() == '' and json.loads(await look())['preset'] is None)
        check('no preset: sharpen slider is one-sided (0..100)', await pg.evaluate('document.querySelector("#lookSliders input").min') == '0')
        await pg.click('#lookPresets [data-preset="pop"]'); await settle()
        check('tap Spood pop -> pop', await active() == 'pop')
        await pg.click('#lookPresets [data-preset="pop"]'); await settle()
        check('tap pop again -> off', await active() == '')
        await pg.click('#lookPresets [data-preset="natural"]'); await settle()
        # tweak brightness with a drag
        sl = pg.locator('#lookSliders .control-row:nth-child(5) input'); await sl.scroll_into_view_if_needed()
        box = await sl.bounding_box()
        y = box['y'] + box['height'] / 2
        await pg.mouse.move(box['x'] + box['width'] * 0.5, y); await pg.mouse.down()
        await pg.mouse.move(box['x'] + box['width'] * 0.75, y, steps=6); await pg.mouse.up(); await settle()
        L = json.loads(await look()); exp = L['adj'].get('exposure', 0)
        check(f'brightness drag -> positive tweak ({exp:.2f})', 0.3 < exp < 0.7)
        lab = (await labels())[4]
        check(f'label matches tweak ({lab})', lab == f'+{round(exp * 100)}')
        check('Reset sliders button shows', await pg.is_visible('#btnLookReset'))
        rendered = await pg.evaluate('window.__macrocut.lookNow().exposure')
        check(f'rendered brightness = +{exp*2:.2f} stops', abs(rendered - exp * 2) < 1e-6)
        # Off is lossless
        await pg.click('#lookPresets [data-preset="off"]'); await settle()
        check('Off lights alone', await active() == 'off')
        check('Off renders untouched', await pg.evaluate('(l => l.cas === 0 && l.exposure === 0 && l.saturation === 0)(window.__macrocut.lookNow())'))
        await pg.click('#lookPresets [data-preset="off"]'); await settle()
        L2 = json.loads(await look())
        check('Off again restores preset and tweak', await active() == 'natural' and abs(L2['adj'].get('exposure', 0) - exp) < 1e-9)
        # moving a slider while Off starts from the clip as filmed
        await pg.click('#lookPresets [data-preset="off"]'); await settle()
        await sl.scroll_into_view_if_needed(); bb = await sl.bounding_box()
        await pg.mouse.click(bb['x'] + bb['width'] * 0.25, bb['y'] + bb['height'] / 2); await settle()
        L4 = json.loads(await look()); rn = await pg.evaluate('window.__macrocut.lookNow()')
        check(f'slider while Off -> untouched clip + that change only ({L4})', L4['preset'] is None and not L4['off'] and rn['cas'] == 0 and rn['exposure'] < 0)
        await pg.click('#btnUndo'); await settle(); await pg.click('#btnUndo'); await settle()
        check('undo back to Natural + tweak', await active() == 'natural' and abs(json.loads(await look())['adj'].get('exposure', 0) - exp) < 1e-9)
        # switching preset keeps tweaks
        await pg.click('#lookPresets [data-preset="pop"]'); await settle()
        check('switching preset keeps your tweak', abs(json.loads(await look())['adj'].get('exposure', 0) - exp) < 1e-9)
        # double-tap reset (touch)
        await sl.scroll_into_view_if_needed(); box = await sl.bounding_box(); y = box['y'] + box['height'] / 2
        cx = box['x'] + box['width'] * 0.75
        await pg.touchscreen.tap(cx, y); await pg.wait_for_timeout(120); await pg.touchscreen.tap(cx, y); await settle()
        check('double-tap puts slider back to 0', json.loads(await look())['adj'].get('exposure', 0) == 0 and (await vals())[4] == 0)
        # undo walks back the reset and the preset tap
        await pg.click('#btnUndo'); await settle()
        check('undo restores tweak', abs(json.loads(await look())['adj'].get('exposure', 0) - exp) < 1e-9 and (await active()) == 'pop')
        await pg.click('#btnUndo'); await settle()
        check('undo restores previous preset + highlight', await active() == 'natural')
        # reset button
        await pg.click('#btnLookReset'); await settle()
        check('Reset sliders zeroes everything', json.loads(await look())['adj'] == {} and all(v == 0 for v in await vals()))
        check('Reset button hides again', not await pg.is_visible('#btnLookReset'))
        # old saved look migrates exactly
        await pg.click('#myLooks .pill-btn >> text=Old punchy'); await settle()
        L3 = json.loads(await look()); rn = await pg.evaluate('window.__macrocut.lookNow()')
        check(f'old saved look -> pop + brightness tweak ({L3})', L3['preset'] == 'pop' and abs(rn['exposure'] - 0.5) < 1e-9 and abs(rn['cas'] - 0.55) < 1e-9)
        check('its preset lights up', await active() == 'pop')
        # Detail size greys out when nothing uses it
        await pg.click('#lookPresets [data-preset="pop"]'); await pg.click('#btnLookReset'); await settle()
        check('Detail size idle with no detail/smoothing', await pg.evaluate('document.querySelectorAll("#lookSliders .look-row")[2].classList.contains("idle")'))
        await pg.click('#lookPresets [data-preset="natural"]'); await settle()
        sl2 = pg.locator('#lookSliders .control-row:nth-child(9) input'); await sl2.scroll_into_view_if_needed(); bx = await sl2.bounding_box()
        await pg.mouse.click(bx['x'] + bx['width'] * 0.3, bx['y'] + bx['height'] / 2); await settle()
        await pg.locator('#lookSliders .control-row:nth-child(6)').scroll_into_view_if_needed()
        await pg.screenshot(path=os.environ.get('SHOT', 'look_shot.png'))
        check('no page errors', not errs); print('  errors:', errs)
        await b.close()
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')
asyncio.run(main())
