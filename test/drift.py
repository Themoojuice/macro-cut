"""Focus build-up under slow drift (clip from make_drift.py), vs all-sharp truth.
Measured inside the body, in frames where a band is well out of focus (>= 1.5 bands away; >= 1 for the "hover" clip,
whose outer bands are only moderately defocused) but has been in focus before.
Build up must be sharper than normal there late in the clip (whole-body drift, which the tracker follows), or, under
parallax (near and far parts drift apart, so stored views end up out of place), no worse than normal and better than
before; and no single frame more than 1.5 dB worse than normal. Also exports Build up from the app at MAIN_URL (before local alignment).
usage: drift.py CLIP_BASE OUT      env: REUSE=off,main,...  RUNS=off,build,main,full"""
import asyncio, os, sys, json, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
BASE, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(BASE + '.json')); W, H, FPS = meta['W'], meta['H'], meta['fps']
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
MAIN_URL = os.environ.get('MAIN_URL', 'http://localhost:8765/base/index.html')
ALL = {'off': ('off', URL), 'build': ('build', URL), 'full': ('full', URL), 'main': ('build', MAIN_URL)}
RUNS = [(n,) + ALL[n] for n in os.environ.get('RUNS', 'off,build,main').split(',')]
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
T = np.load(BASE + '_truth.npy').astype(np.float32); ev = meta['truth_every']; focus = np.array(meta['focus'])
F = {m: frames(p) for m, p in paths.items()}
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, np.mean((a - b) ** 2)))
D = int(np.ceil(meta['drift'] * 1.4)) + 2
def mask(bnd):
    m = np.zeros((H, W), bool); y0, y1 = meta['bands'][bnd]; x0, x1 = meta['body']
    m[int(y0) + 12:int(y1) - 12, int(x0 + 16 + D):int(x1 - 16 - D)] = True
    return m
ks = list(range(0, len(T) * ev, ev)); names = [m for m in ('off', 'build', 'full', 'main') if m in F]
OUTF = 1.0 if meta.get('sched') == 'hover' else 1.5
seen = {b: next((k for k in range(len(focus)) if abs(focus[k] - b) < 0.3), 10 ** 9) for b in range(4)}
fails = []
print(f'\ndrift {meta["drift"]} px; body only, dB vs all-sharp truth')
print('band | half  | frames | ' + ' | '.join(f'{m:>6s}' for m in names) + '    (out of focus, seen sharp before)')
gain = {m: [] for m in names}
for bnd in range(4):
    for half, (t0, t1) in (('early', (0, 3)), ('late', (3, 6.1))):
        kk = [k for k in ks if t0 <= k / FPS < t1 and k > seen[bnd] + 3 and abs(focus[k] - bnd) >= OUTF]
        if not kk: continue
        r = {m: np.mean([ps(F[m][k][mask(bnd)], T[k // ev][mask(bnd)]) for k in kk]) for m in names}
        print(f'  {bnd}  | {half:5s} | {len(kk):6d} | ' + ' | '.join(f'{r[m]:6.1f}' for m in names))
        for m in names: gain[m].append(r[m] - r['off'])
        if half == 'late':
            if meta.get('kind') == 'parallax':   # near/far drift apart: stored views end up out of place
                if r['build'] < r['off'] - 0.5: fails.append(f'band {bnd} late: build up {r["build"] - r["off"]:+.1f} dB vs normal (detail shown out of place)')
                if 'main' in r and r['build'] < r['main'] + 0.5: fails.append(f'band {bnd} late: build up only {r["build"] - r["main"]:+.1f} dB vs before')
            else:
                if r['build'] < r['off'] + 2: fails.append(f'band {bnd} late: build up only {r["build"] - r["off"]:+.1f} dB vs normal')
                if 'main' in r and r['build'] < r['main'] - 0.5: fails.append(f'band {bnd} late: build up {r["build"] - r["main"]:+.1f} dB vs before')
print('mean gain vs normal: ' + ', '.join(f'{m} {np.mean(gain[m]):+.1f} dB' for m in names if m != 'off'))
# never worse than normal: every truth frame, every band
for m in names:
    if m == 'off': continue
    worst = min((ps(F[m][k][mask(b)], T[k // ev][mask(b)]) - ps(F['off'][k][mask(b)], T[k // ev][mask(b)]), k, b) for k in ks for b in range(4))
    print(f'{m}: worst frame vs normal {worst[0]:+.1f} dB (frame {worst[1]}, band {worst[2]})')
    if m != 'main' and worst[0] < -1.5: fails.append(f'{m} worse than normal by {-worst[0]:.1f} dB at frame {worst[1]} band {worst[2]}')
k = (int(5.5 * FPS) // ev) * ev
cv2.imwrite(OUT + '_late.png', np.hstack([F[m][k] for m in names] + [T[k // ev]]).astype(np.uint8)[::2, ::2])
print('\nRESULT:', 'PASS' if not fails else 'FAIL ' + '; '.join(fails))
