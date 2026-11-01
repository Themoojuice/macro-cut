"""Per-pose "sharp from the start" vs all-sharp truth on the two-pose clip (make_poses.py).
At the start of EACH pose (before the focus has reached a band), "sharp from the start" must be sharper than the
normal export and than Build up (which can't know the future). usage: poses.py CLIP_BASE OUT"""
import asyncio, os, sys, json, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
BASE, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(BASE + '.json')); W, H, FPS = meta['W'], meta['H'], meta['fps']
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
def frames(p):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
async def main():
    paths = {}
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        for mode in ('off', 'build', 'full'):
            pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []
            pg.on('pageerror', lambda e: errs.append(str(e)))
            await pg.goto(URL); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
            await pg.set_input_files('#fileInput', BASE + '.mp4')
            await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
            await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
            await pg.evaluate(f'() => {{ window.MC_FOCUS_DEBUG = true; window.__macrocut.state.exp.focus = "{mode}"; }}')
            await pg.click('#btnExport'); await pg.click('#btnExportGo')
            await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|Couldn/.test(document.getElementById("exportMsg").textContent)', timeout=3000000)
            print(f'{mode:6s} |', (await pg.inner_text('#exportMsg'))[:150], flush=True)
            for e in await pg.evaluate('window.MC_FOCUS_LOG || []'):
                if e and e[0] == 'segments': print('   segments:', e[1], flush=True)
            async with pg.expect_download() as dl: await pg.click('#btnSave')
            paths[mode] = f'{OUT}_{mode}.mp4'; await (await dl.value).save_as(paths[mode])
            print('   page errors:', errs[:3]); await pg.context.close()
        await b.close()
    return paths
paths = asyncio.run(main())
T = np.load(BASE + '_truth.npy').astype(np.float32); ev = meta['truth_every']; focus = np.array(meta['focus'])
F = {m: frames(p) for m, p in paths.items()}
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, np.mean((a - b) ** 2)))
def mask(bnd, k):
    m = np.zeros((H, W), bool); y0, y1 = meta['bands'][bnd]; x0, x1 = meta['body']; dx = meta['dx'][k]
    m[int(y0) + 12:int(y1) - 12, int(x0 + 40 + 16):int(x1 - 16)] = True   # inside the body at both positions
    return m
fails = []
for name, (t0, t1) in (('start of pose 1', (0.05, 0.6)), ('start of pose 2', (3.2, 3.7))):
    ks = [k for k in range(0, len(T) * ev, ev) if t0 <= k / FPS <= t1]
    print(f'\n{name} ({t0}–{t1} s), body only, dB vs all-sharp truth:   band | normal | build up | sharp from start')
    for bnd in (1, 2, 3):
        r = {m: np.mean([ps(F[m][k][mask(bnd, k)], T[k // ev][mask(bnd, k)]) for k in ks]) for m in F}
        print(f'      {bnd}   | {r["off"]:5.1f}  |  {r["build"]:5.1f}   |  {r["full"]:5.1f}')
        if bnd >= 2 and r['full'] < max(r['build'], r['off']) + 2: fails.append(f'{name} band {bnd}: {r["full"] - max(r["build"], r["off"]):+.1f} dB')
k = int(3.3 * FPS) // ev * ev
cv2.imwrite(OUT + '_pose2start.png', np.hstack([F[m][k] for m in ('off', 'build', 'full')] + [T[k // ev]]).astype(np.uint8)[::2, ::2])
print('\nRESULT:', 'PASS' if not fails else 'FAIL ' + '; '.join(fails))
