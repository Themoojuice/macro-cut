"""Per-part Look + Speed through the real UI: split, darken + slow only the last part, export, measure."""
import asyncio, os, sys, subprocess, numpy as np
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
def luma_series(path):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', 'scale=96:54,format=gray', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, 54, 96).mean((1, 2))
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        # split at 4.0 s via the UI, playhead lands in part 2
        await pg.evaluate('window.__macrocut.seek(4.0)'); await pg.wait_for_timeout(300)
        await pg.click('.dock-item[data-tab="tab-trim"]'); await pg.click('#btnSplit')
        await pg.evaluate('window.__macrocut.seek(5.0)'); await pg.wait_for_timeout(300)
        await pg.click('.dock-item[data-tab="tab-look"]')
        print('scope hint:', await pg.inner_text('#tab-look .scope-hint'))
        await pg.evaluate('''() => { const row = [...document.querySelectorAll('#lookSliders .control-row')].find(r => r.textContent.trim().startsWith('Brightness'));
          const el = row.querySelector('input'); el.value = -1; el.dispatchEvent(new Event('input', {bubbles: true})); }''')
        await pg.click('.dock-item[data-tab="tab-speed"]'); await pg.click('#speedGroup .pill-btn[data-speed="0.5"]')
        print('speed info:', await pg.inner_text('#speedInfo'))
        # check part 1 is untouched in the UI
        await pg.evaluate('window.__macrocut.seek(1.0)'); await pg.wait_for_timeout(300)
        await pg.click('.dock-item[data-tab="tab-look"]')
        b1 = await pg.evaluate('''() => [...document.querySelectorAll('#lookSliders .control-row')].find(r => r.textContent.trim().startsWith('Brightness')).querySelector('b').textContent''')
        print('part 1 brightness slider shows:', b1, '| badges:', await pg.evaluate('[...document.querySelectorAll(".seg-badge")].map(e=>e.textContent)'))
        await pg.screenshot(path=OUT + '_ui.png')
        await pg.click('#btnExport'); await pg.wait_for_timeout(200)
        print('export summary:', await pg.inner_text('#exportSummary'))
        await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=900000)
        print(await pg.inner_text('#exportMsg'))
        async with pg.expect_download() as dl: await pg.click('#btnSave')
        await (await dl.value).save_as(OUT + '.mp4')
        print('errors', errs); await b.close()
    L = luma_series(OUT + '.mp4'); n = len(L)
    print(f'frames {n} (expect 4 s + 2 s×2 = 8 s → {8*30})')
    print(f'brightness: part 1 mean {L[:110].mean():.1f} | blend zone (3.85-4.15 s) {L[115:125].round(0).tolist()} | part 2 mean {L[130:].mean():.1f}')
asyncio.run(main())
