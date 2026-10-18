"""Does preview playback keep rendering after analysis? Counts draws while playing."""
import asyncio, os, sys
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915})).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: m.type == 'error' and errs.append(m.text))
        await pg.goto('http://localhost:8765/app/index.html')
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=30000)
        async def playfor(label, ms=2500):
            d0 = await pg.evaluate('window.__macrocut.state.draws||0'); t0 = await pg.evaluate('document.getElementById("video").currentTime')
            await pg.click('#btnPlay'); await pg.wait_for_timeout(ms); await pg.click('#btnPlay')
            d1 = await pg.evaluate('window.__macrocut.state.draws||0'); t1 = await pg.evaluate('document.getElementById("video").currentTime')
            print(f'{label}: video advanced {t1-t0:.2f}s, draws {d1-d0}')
        await playfor('before analysis')
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.click('#stabOn + .knob')
        await pg.wait_for_function('window.__macrocut.state.analysis && !window.__macrocut.state.analysing', timeout=300000)
        await playfor('after analysis')
        await pg.click('#btnCompare'); print('compare on:', await pg.evaluate('window.__macrocut.state.compare'))
        await playfor('compare on (1)'); await playfor('compare on (2)')
        await pg.click('#btnCompare'); print('compare on:', await pg.evaluate('window.__macrocut.state.compare'))
        await playfor('compare off')
        print('errors', errs[:5]); await b.close()
asyncio.run(main())
