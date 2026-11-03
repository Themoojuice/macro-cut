"""Focus build-up after a pose change, vs all-sharp truth (two-pose clip from make_poses.py: a large body shifts
40 px at 3 s). At the start of each pose, out-of-focus spots can't be checked, so stored detail from the other pose
must not be shown there (no stale detail): both modes must be no worse than the normal export.
Also exports Build up from the app at MAIN_URL (before this change) for comparison.
usage: poses.py CLIP_BASE OUT"""
import asyncio, os, sys, json, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
BASE, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(BASE + '.json')); W, H, FPS = meta['W'], meta['H'], meta['fps']
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
MAIN_URL = os.environ.get('MAIN_URL', 'http://localhost:8765/base/index.html')
RUNS = [('off', 'off', URL), ('build', 'build', URL), ('full', 'full', URL), ('main', 'build', MAIN_URL)]
def frames(p):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
async def main():
    paths, reuse = {}, set(os.environ.get('REUSE', '').split(','))
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        for name, mode, url in RUNS:
            if name in reuse and os.path.exists(f'{OUT}_{name}.mp4'): paths[name] = f'{OUT}_{name}.mp4'; print(f'{name:6s} | reusing'); continue
            pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            await pg.goto(url); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
            await pg.set_input_files('#fileInput', BASE + '.mp4')
            await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
            await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
            await pg.evaluate(f'() => {{ window.__macrocut.state.exp.focus = "{mode}"; }}')
            await pg.click('#btnExport'); await pg.click('#btnExportGo')
            await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|Couldn/.test(document.getElementById("exportMsg").textContent)', timeout=3000000)
            print(f'{name:6s} |', (await pg.inner_text('#exportMsg'))[:150], flush=True)
            async with pg.expect_download() as dl: await pg.click('#btnSave')
            paths[name] = f'{OUT}_{name}.mp4'; await (await dl.value).save_as(paths[name])
            print('   page errors:', errs[:3]); await pg.context.close()
        await b.close()
    return paths
paths = asyncio.run(main())
T = np.load(BASE + '_truth.npy').astype(np.float32); ev = meta['truth_every']
F = {m: frames(p) for m, p in paths.items()}
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, np.mean((a - b) ** 2)))
def mask(bnd, k):
    m = np.zeros((H, W), bool); y0, y1 = meta['bands'][bnd]; x0, x1 = meta['body']
    m[int(y0) + 12:int(y1) - 12, int(x0 + 40 + 16):int(x1 - 16)] = True   # inside the body at both positions
    return m
fails = []
for name, (t0, t1) in (('start of pose 1', (0.05, 0.6)), ('start of pose 2', (3.2, 3.7))):
    ks = [k for k in range(0, len(T) * ev, ev) if t0 <= k / FPS <= t1]
    print(f'\n{name} ({t0}–{t1} s), body only, dB vs all-sharp truth:  band | normal | build up | sharp from start | build up before (main)')
    for bnd in (1, 2, 3):
        r = {m: np.mean([ps(F[m][k][mask(bnd, k)], T[k // ev][mask(bnd, k)]) for k in ks]) for m in F}
        print(f'      {bnd}   | {r["off"]:5.1f}  |  {r["build"]:5.1f}   |  {r["full"]:5.1f}           |  {r["main"]:5.1f}')
        for m in ('build', 'full'):
            if r[m] < r['off'] - 0.5: fails.append(f'{name} band {bnd}: {m} {r[m] - r["off"]:+.1f} dB vs normal (stale detail)')
k = int(3.3 * FPS) // ev * ev
cv2.imwrite(OUT + '_pose2start.png', np.hstack([F[m][k] for m in ('off', 'main', 'build', 'full')] + [T[k // ev]]).astype(np.uint8)[::2, ::2])
print('\nRESULT:', 'PASS' if not fails else 'FAIL ' + '; '.join(fails))
