"""Smooth slow motion vs ground truth: 30 fps clip (every 2nd frame of a 60 fps clip) exported at ½× / 30 fps.
Output frame j should equal real 60 fps frame j. Odd j are the frames that had to be created."""
import asyncio, os, sys, subprocess, numpy as np
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, TRUTH, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 640, 360
truth = np.fromfile(TRUTH, np.uint8).reshape(-1, H, W, 3)[..., ::-1].astype(np.float32)
def frames(path):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
psnr = lambda a, b: 10 * np.log10(255 ** 2 / np.mean((a - b) ** 2))
async def run(pg, smooth, name):
    await pg.evaluate(f'''() => {{ const s = window.__macrocut.state; s.smoothSlow = {str(smooth).lower()}; s.exp.fps = '30'; }}''')
    await pg.click('#btnExport'); await pg.click('#btnExportGo')
    await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=1800000)
    print(name, '|', await pg.inner_text('#exportMsg'), flush=True)
    async with pg.expect_download() as dl: await pg.click('#btnSave')
    p = f'{OUT}_{name}.mp4'; await (await dl.value).save_as(p); await pg.click('#btnExportCancel'); return p
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        await pg.click('.dock-item[data-tab="tab-speed"]'); await pg.click('#speedGroup .pill-btn[data-speed="0.5"]')
        print('speed info:', await pg.inner_text('#speedInfo'))
        on = await run(pg, True, 'smooth'); off = await run(pg, False, 'steps')
        print('errors', errs); await b.close()
    for name, path in (('steps (old)', off), ('smooth (new)', on)):
        F = frames(path); n = min(len(F), len(truth) - 1)
        odd = [psnr(F[j], truth[j]) for j in range(1, n - 2, 2)]; even = [psnr(F[j], truth[j]) for j in range(0, n - 2, 2)]
        print(f'{name:13s} {len(F)} frames | created frames vs real: {np.mean(odd):.1f} dB (worst {np.min(odd):.1f}) | real frames: {np.mean(even):.1f} dB')
asyncio.run(main())
