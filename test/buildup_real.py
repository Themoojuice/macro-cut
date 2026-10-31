"""Focus build-up on real footage through the app (no ground truth).
Exports Off / Build up / Sharp from the start with the default look and reports, per time window: subject
detail vs the normal export, and frame-to-frame change (flicker). Also writes a labelled 3-up comparison video
(phone-portrait friendly) and comparison stills.
usage: buildup_real.py CLIP OUT [modes=off,build,full]   (env REUSE=off,build to keep earlier exports)"""
import asyncio, os, sys, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
MODES = (sys.argv[3] if len(sys.argv) > 3 else 'off,build,full').split(',')
URL = os.environ.get('APP_URL', 'http://localhost:8765/app/index.html')
LABEL = {'off': 'Normal export', 'build': 'Focus build-up: Build up', 'full': 'Focus build-up: Sharp from the start'}
async def run(pg, focus):
    await pg.evaluate(f'() => {{ window.__macrocut.state.exp.focus = "{focus}"; }}')
    await pg.click('#btnExport'); await pg.click('#btnExportGo')
    await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|Couldn/.test(document.getElementById("exportMsg").textContent)', timeout=7000000)
    print(f'{focus:6s} |', (await pg.inner_text('#exportMsg'))[:300].replace('\n', ' '), flush=True)
    async with pg.expect_download() as dl: await pg.click('#btnSave')
    p = f'{OUT}_{focus}.mp4'; await (await dl.value).save_as(p); await pg.click('#btnExportCancel'); return p
async def main():
    reuse = set(os.environ.get('REUSE', '').split(','))
    paths = {m: f'{OUT}_{m}.mp4' for m in MODES if m in reuse and os.path.exists(f'{OUT}_{m}.mp4')}
    todo = [m for m in MODES if m not in paths]
    if todo:
        async with async_playwright() as p:
            b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
            for m in todo:   # a fresh page per export, like opening the clip anew
                ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
                pg = await ctx.new_page(); errs = []
                pg.on('pageerror', lambda e: errs.append(str(e)))
                pg.on('console', lambda msg: (msg.type == 'error' and errs.append(msg.text)) or ('focus build-up' in msg.text and print('  ', msg.text, flush=True)))
                await pg.goto(URL); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
                await pg.set_input_files('#fileInput', CLIP)
                await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
                paths[m] = await run(pg, m)
                print(f'  {m} page errors:', errs[:5], flush=True); await ctx.close()
            await b.close()
    return paths
paths = asyncio.run(main())
def frames(p, w=640, h=360):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', f'scale={w}:{h}:flags=area,format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, h, w, 3)
def fullres(p, k):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', f'select=eq(n\\,{k}),format=bgr24', '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(o, np.uint8), 1)
F = {m: frames(p) for m, p in paths.items()}
n = min(len(v) for v in F.values()); fps = 30
lap = lambda g: cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F)
gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
# subject = the 25% most detailed 32-px tiles of the normal export's sharpest-ever view (union over the clip)
ref = np.max(np.stack([np.abs(lap(gray(F['off'][k]))) for k in range(0, n, 10)]), 0)
t = 32; H, W = ref.shape
tiles = sorted(((ref[y:y+t, x:x+t].mean(), y, x) for y in range(0, H - t, t) for x in range(0, W - t, t)))[-(H // t) * (W // t) // 4:]
def detail(img): L = lap(gray(img)); return np.mean([L[y:y+t, x:x+t].var() for _, y, x in tiles])
print('\nwindow (s)     | ' + ' | '.join(f'{m:>12s}' for m in F) + ' |  subject detail (× normal) and flicker (mean |frame-to-frame change|)')
for a, b in ((1, 4), (5, 10), (10, 18), (18.6, 19.6)):
    ks = range(int(a * fps), min(n, int(b * fps)))
    d = {m: np.mean([detail(F[m][k]) for k in ks[::3]]) for m in F}
    fl = {m: np.mean(np.abs(np.diff(F[m][ks[0]:ks[-1] + 1].astype(np.float32), axis=0))) for m in F}
    print(f'{a:5.1f}–{b:<5.1f}    | ' + ' | '.join(f'×{d[m] / d["off"]:4.2f} fl {fl[m]:4.2f}' for m in F))
# 3-up comparison video (portrait: stacked), labelled
order = [m for m in ('off', 'build', 'full') if m in F]
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'640x{360 * len(order)}', '-r', '30', '-i', '-',
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT + '_compare.mp4'], stdin=subprocess.PIPE)
for k in range(n):
    rows = []
    for m in order:
        x = F[m][k].copy()
        cv2.putText(x, LABEL[m], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(x, LABEL[m], (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(x)
    cv2.putText(rows[-1], f'{k / fps:4.1f} s', (560, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    enc.stdin.write(np.vstack(rows).tobytes())
enc.stdin.close(); enc.wait()
for sec in (3, 8, 17.5, 19.2):
    k = int(sec * fps); imgs = [fullres(paths[m], k) for m in order]
    h, w = imgs[0].shape[:2]; c = (slice(h // 8, h // 8 + h * 5 // 8), slice(w // 4, w // 4 + w // 2))
    cv2.imwrite(f'{OUT}_still_{sec:04.1f}s.png', np.hstack([x[c] for x in imgs]))
print('wrote', OUT + '_compare.mp4')
