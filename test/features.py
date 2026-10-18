"""Measure that every control does what it says, on real exported files.
Exports a 1 s slice per scenario through the UI, then checks the output numerically with OpenCV."""
import asyncio, os, sys, json, cv2, numpy as np
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
SRC = cv2.VideoCapture(CLIP)
import subprocess
def ff_frame(path, t, w=None, h=None):
    # decode with an explicit BT.709 matrix so tagged and untagged files compare like-for-like
    if w is None:
        c = cv2.VideoCapture(path); w, h = int(c.get(3)), int(c.get(4))
    out = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', path, '-frames:v', '1', '-vf',
                          'scale=in_color_matrix=bt709:in_range=tv,format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3).copy()
def src_frame(t): return ff_frame(CLIP, t)
def out_info(path):
    c = cv2.VideoCapture(path); n = 0
    while c.read()[0]: n += 1
    return n, ff_frame(path, 0)
def similarity(a, b):
    """rotation (deg) and scale mapping a -> b using features."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000); ka, da = orb.detectAndCompute(ga, None); kb, db = orb.detectAndCompute(gb, None)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2)
    return np.degrees(np.arctan2(M[1, 0], M[0, 0])), np.hypot(M[0, 0], M[1, 0])
def chroma(f): lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32); return np.mean(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128))
def luma(f): return float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
def detail(f): g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32); return float(cv2.Laplacian(g, cv2.CV_32F).var())

async def main():
    results = []
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html')
        await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=30000)
        await pg.evaluate('window.__macrocut.state.segments = [{start: 1.0, end: 2.0}]')
        async def tab(t): await pg.click(f'.dock-item[data-tab="tab-{t}"]')
        async def slider(sel_or_label, v):
            await pg.evaluate('''([s, v]) => {
              let el = document.getElementById(s);
              if (!el) { const row = [...document.querySelectorAll('#lookSliders .control-row')].find(r => r.textContent.trim().startsWith(s)); el = row.querySelector('input'); }
              el.value = v; el.dispatchEvent(new Event('input', {bubbles: true})); }''', [sel_or_label, v])
        async def export(name):
            await pg.click('#btnExport'); await pg.click('#btnExportGo')
            await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=600000)
            msg = await pg.inner_text('#exportMsg')
            async with pg.expect_download() as dl: await pg.click('#btnSave')
            path = f'{OUT}_{name}.mp4'; await (await dl.value).save_as(path)
            await pg.click('#btnExportCancel'); return path, msg
        def check(name, cond, detail_s): results.append((name, bool(cond), detail_s)); print(('PASS ' if cond else 'FAIL ') + name + ': ' + detail_s, flush=True)

        ref = src_frame(1.0)
        # baseline: look off
        await tab('look'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        base, _ = await export('base'); n, f0 = out_info(base)
        check('trim 1 s @30fps', n == 30, f'{n} frames, {f0.shape[1]}x{f0.shape[0]}')
        r, s = similarity(ref, f0); check('look off = untouched geometry', abs(r) < 0.2 and abs(s - 1) < 0.01, f'rot {r:.2f}°, scale {s:.3f}')
        c0, l0, d0 = chroma(f0), luma(f0), detail(f0)
        check('look off ≈ source colour', abs(c0 - chroma(ref)) < 1.5 and abs(l0 - luma(ref)) < 3, f'chroma {c0:.1f} vs {chroma(ref):.1f}, luma {l0:.1f} vs {luma(ref):.1f}')
        # fine rotation
        await tab('frame'); await slider('rot', 10)
        pth, _ = await export('rot10'); _, f = out_info(pth); r, s = similarity(ref, f)
        check('rotate +10°', abs(abs(r) - 10) < 0.3, f'measured {r:.2f}°, auto-zoom {s:.3f}× (no black corners needs ≥1.18)')
        check('rotate fills frame (no black corners)', min(f[:8, :8].mean(), f[-8:, -8:].mean(), f[:8, -8:].mean(), f[-8:, :8].mean()) > 25, 'corner brightness ok')
        await slider('rot', 0)
        # quarter turn + aspect
        await pg.click('#btnRotR'); pth, _ = await export('q90'); _, f = out_info(pth); r, s = similarity(ref, f)
        check('90° turn', f.shape[0] > f.shape[1] and abs(abs(r) - 90) < 0.5, f'{f.shape[1]}x{f.shape[0]}, rot {r:.1f}°')
        await pg.click('#btnRotL')
        await pg.click('#aspectGroup .pill-btn[data-aspect="1:1"]'); pth, _ = await export('sq'); _, f = out_info(pth)
        check('1:1 crop', f.shape[0] == f.shape[1], f'{f.shape[1]}x{f.shape[0]}')
        await pg.click('#aspectGroup .pill-btn[data-aspect="native"]')
        # zoom 2x
        await slider('zoom', 2); pth, _ = await export('zoom2'); _, f = out_info(pth); r, s = similarity(ref, f)
        check('zoom 2×', abs(s - 2) < 0.05, f'scale {s:.3f}')
        await slider('zoom', 1)
        # colour
        await tab('look'); await slider('Colour strength', -1); pth, _ = await export('gray'); _, f = out_info(pth)
        check('saturation −100 → greyscale', chroma(f) < 2.5, f'chroma {chroma(f):.2f} (source {c0:.1f})')
        await slider('Colour strength', 0.8); pth, _ = await export('sat'); _, f = out_info(pth)
        check('saturation +80 → more colour', chroma(f) > c0 * 1.4, f'chroma {chroma(f):.1f} vs {c0:.1f}')
        await slider('Colour strength', 0); await slider('Brightness', 1); pth, _ = await export('ev'); _, f = out_info(pth)
        check('exposure +1 EV → brighter', luma(f) > l0 * 1.25, f'luma {luma(f):.1f} vs {l0:.1f}')
        await slider('Brightness', 0); await slider('Cool', 1); pth, _ = await export('warm'); _, f = out_info(pth)
        bgr = f.reshape(-1, 3).mean(0); bgr0 = f0.reshape(-1, 3).mean(0)
        check('warmth → redder/less blue', (bgr[2] - bgr[0]) > (bgr0[2] - bgr0[0]) + 8, f'R−B {bgr[2]-bgr[0]:.1f} vs {bgr0[2]-bgr0[0]:.1f}')
        await slider('Cool', 0)
        # sharpening
        await slider('Sharpen', 0.8); await slider('Hair', 1.5); pth, _ = await export('sharp'); _, f = out_info(pth)
        check('sharpen → more fine detail', detail(f) > d0 * 1.3, f'Laplacian var {detail(f):.0f} vs {d0:.0f}')
        await slider('Sharpen', 0); await slider('Hair', 0); await slider('Smooth background', 1); pth, _ = await export('smoothbg'); _, f = out_info(pth)
        check('bokeh smooth → less noise', detail(f) < d0, f'Laplacian var {detail(f):.0f} vs {d0:.0f}')
        await slider('Smooth background', 0)
        # speed
        await tab('speed'); await pg.click('#speedGroup .pill-btn[data-speed="0.25"]'); pth, msg = await export('slow'); n, f = out_info(pth)
        check('¼× speed: 1 s → 4 s', n == 120, f'{n} frames ({msg})')
        await pg.evaluate('window.__macrocut.setSpeed(0.8)'); pth, msg = await export('slow08'); n, f = out_info(pth)
        check('0.8× speed: 1 s → 1.25 s', n in (37, 38), f'{n} frames')
        print('page errors:', errs)
        await b.close()
    fails = [r for r in results if not r[1]]
    print(f'\n{len(results) - len(fails)}/{len(results)} passed')
asyncio.run(main())
