"""Focus build-up through the real app vs the all-sharp truth (clip from make_buildup.py).
Exports Off / Build up / Sharp from the start and checks, per depth band:
  1. after a band has been in focus, build-up keeps it sharper than the normal export;
  2. "sharp from the start" is sharper than normal before the focus ever got there;
  3. where the object moved away (and where it moved to), build-up is no worse than normal: no ghost.
usage: buildup.py CLIP_BASE OUT"""
import asyncio, os, sys, json, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
BASE, OUT = sys.argv[1], sys.argv[2]
meta = json.load(open(BASE + '.json')); W, H, FPS = meta['W'], meta['H'], meta['fps']
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
def frames(p):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
async def run(pg, focus, name):
    await pg.evaluate(f'() => {{ window.__macrocut.state.exp.focus = "{focus}"; }}')
    await pg.click('#btnExport'); await pg.click('#btnExportGo')
    await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|Couldn/.test(document.getElementById("exportMsg").textContent)', timeout=3000000)
    print(f'{name:6s} |', (await pg.inner_text('#exportMsg'))[:260].replace('\n', ' '), flush=True)
    async with pg.expect_download() as dl: await pg.click('#btnSave')
    p = f'{OUT}_{name}.mp4'; await (await dl.value).save_as(p); await pg.click('#btnExportCancel'); return p
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: (m.type == 'error' and errs.append(m.text)) or ('focus build-up' in m.text and print('  ', m.text, flush=True)))
        await pg.goto(URL); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', BASE + '.mp4')
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        paths = {}
        reuse = set(os.environ.get('REUSE', '').split(','))   # e.g. REUSE=off: keep an earlier normal export
        for focus, name in (('build', 'build'), ('full', 'full'), ('off', 'off')):
            if name in reuse and os.path.exists(f'{OUT}_{name}.mp4'): paths[name] = f'{OUT}_{name}.mp4'; print(f'{name:6s} | reusing {paths[name]}'); continue
            paths[name] = await run(pg, focus, name)
        print('page errors:', errs[:5]); await b.close()
    return paths
paths = asyncio.run(main())
T = np.load(BASE + '_truth.npy').astype(np.float32); every = meta['truth_every']
F = {k: frames(v) for k, v in paths.items()}
n = min(len(x) for x in F.values()); print('frames', {k: len(v) for k, v in F.items()})
focus = np.array(meta['focus']); ob = meta['obj']
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, np.mean((a - b) ** 2)))
def box(x, y, w, h, pad=10): return slice(max(0, int(y - h - pad)), int(y + h + pad)), slice(max(0, int(x - w - pad)), int(x + w + pad))
def band_mask(bnd, k):
    m = np.zeros((H, W), bool); y0, y1 = meta['bands'][bnd]; m[int(y0) + 16:int(y1) - 16, 24:-24] = True
    for x in (ob['x'][0], ob['x'][-1], ob['x'][k]):   # leave the object's old and new places out of the band scores
        sy, sx = box(x, ob['y'], ob['w'], ob['h'], 24); m[sy, sx] = False
    return m
rows = []
for j in range(len(T)):
    k = j * every
    if k >= n: break
    r = {'k': k, 't': k / FPS, 'focus': focus[k]}
    for name, V in F.items():
        for bnd in range(4):
            msk = band_mask(bnd, k); r[f'{name}{bnd}'] = ps(V[k][msk], T[j][msk])
        if k / FPS >= 2.4:   # object's old place (it left at 2.0–2.2 s) and new place
            sy, sx = box(ob['x'][0], ob['y'], ob['w'], ob['h']); r[f'{name}_old'] = ps(V[k][sy, sx], T[j][sy, sx])
            sy, sx = box(ob['x'][k], ob['y'], ob['w'], ob['h']); r[f'{name}_new'] = ps(V[k][sy, sx], T[j][sy, sx])
    rows.append(r)
fails = []
print('\n1) after a band has been in focus (and focus moved ≥1 band away): dB vs all-sharp truth')
print('   band | normal | build up | sharp from start | frames')
for bnd in range(3):
    sel = [r for r in rows if r['focus'] >= bnd + 1.0 and r['t'] > [0, 1.3, 2.6][bnd] + 0.3]
    o, bu, fu = (np.mean([r[f'{nm}{bnd}'] for r in sel]) for nm in ('off', 'build', 'full'))
    print(f'    {bnd}   | {o:6.1f} | {bu:7.1f}  | {fu:7.1f}          | {len(sel)}')
    if bu < o + 2: fails.append(f'band {bnd}: build-up only {bu - o:+.1f} dB over normal')
print('\n2) before the focus reached a band (t < 0.9 s): "sharp from the start" vs normal')
early = [r for r in rows if r['t'] < 0.9]
for bnd in (1, 2, 3):
    o, bu, fu = (np.mean([r[f'{nm}{bnd}'] for r in early]) for nm in ('off', 'build', 'full'))
    print(f'    band {bnd}: normal {o:5.1f} | build up {bu:5.1f} | sharp from start {fu:5.1f}')
    if fu < o + 2: fails.append(f'early band {bnd}: sharp-from-start only {fu - o:+.1f} dB over normal')
print('\n3) where the object moved away / to (t ≥ 2.4 s): no ghost = not worse than normal')
late = [r for r in rows if 'off_old' in r]
for where in ('old', 'new'):
    o, bu, fu = (np.mean([r[f'{nm}_{where}'] for r in late]) for nm in ('off', 'build', 'full'))
    worst = min(min(r[f'build_{where}'] - r[f'off_{where}'], r[f'full_{where}'] - r[f'off_{where}']) for r in late)
    print(f'    {where} place: normal {o:5.1f} | build up {bu:5.1f} | sharp from start {fu:5.1f} | worst single frame {worst:+.1f} dB')
    if min(bu, fu) < o - 0.5: fails.append(f'{where} object place worse than normal (ghost?)')
# pictures: a frame late in the clip (focus on band 3), and the object's old place just after the jump
k = (int(5.0 * FPS) // every) * every; j = k // every
strip = lambda V: V[k][::2, ::2]
cv2.imwrite(OUT + '_compare.png', np.vstack([np.hstack([strip(F['off']), strip(F['build'])]), np.hstack([strip(F['full']), T[j][::2, ::2]])]).astype(np.uint8))
k2 = (int(2.5 * FPS) // every) * every; sy, sx = box(ob['x'][0], ob['y'], ob['w'], ob['h'], 40)
cv2.imwrite(OUT + '_ghost.png', np.hstack([F[nm][k2][sy, sx] for nm in ('off', 'build', 'full')] + [T[k2 // every][sy, sx]]).astype(np.uint8))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
for bnd in range(4):
    for nm, c in (('off', '0.5'), ('build', 'C0'), ('full', 'C1')): ax[bnd].plot([r['t'] for r in rows], [r[f'{nm}{bnd}'] for r in rows], c=c, label=nm)
    ax[bnd].set_ylabel(f'band {bnd} dB'); ax2 = ax[bnd].twinx(); ax2.plot([r['t'] for r in rows], [abs(r['focus'] - bnd) for r in rows], 'k:', lw=0.8); ax2.set_ylim(3.5, -0.2)
ax[0].legend(loc='lower right'); ax[-1].set_xlabel('s  (dotted: distance of focus from the band, top = in focus)'); plt.tight_layout(); plt.savefig(OUT + '_bands.png', dpi=80)
print('\nRESULT:', 'PASS' if not fails else 'FAIL ' + '; '.join(fails))
