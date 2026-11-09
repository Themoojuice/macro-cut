"""Export pipeline: frames converted to I420 on the GPU (default) vs the browser's own canvas conversion
(localStorage 'macrocut.out' = 'canvas'). Same frame count, same picture and colour (within encoding noise),
a valid fast-start MP4 (ftyp, moov, free, mdat) with sound, and the speed of each.
usage: pipeline.py clip.mp4 OUT [gl=swiftshader|llvmpipe]"""
import asyncio, os, sys, time, struct, subprocess, json
import numpy as np
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
GL = sys.argv[3].split('=')[1] if len(sys.argv) > 3 else 'swiftshader'
ARGS = {'swiftshader': ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
        'llvmpipe': ['--use-gl=angle', '--use-angle=gles-egl', '--ignore-gpu-blocklist', '--enable-gpu']}[GL]
S = 'window.__macrocut.state'

def boxes(path):
    f = open(path, 'rb').read(); off, out = 0, []
    while off + 8 <= len(f):
        sz, t = struct.unpack('>I4s', f[off:off + 8])
        if sz == 1: sz = struct.unpack('>Q', f[off + 8:off + 16])[0]
        if sz == 0: sz = len(f) - off
        out.append(t.decode('latin1')); off += sz
    return out

def frames(path):
    j = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height', '-of', 'json', path], capture_output=True, text=True).stdout)
    v = [s for s in j['streams'] if s['codec_type'] == 'video'][0]
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, v['height'], v['width'], 3).astype(np.float32), any(s['codec_type'] == 'audio' for s in j['streams'])

async def export(pg, mode):
    await pg.evaluate(f'localStorage.setItem("macrocut.out", "{mode}")')
    await pg.click('#btnExport'); await pg.wait_for_timeout(200)
    t0 = time.time(); await pg.click('#btnExportGo')
    await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|lost|stopped/i.test(document.getElementById("exportMsg").textContent)', timeout=1800000, polling=500)
    dt = time.time() - t0; msg = await pg.inner_text('#exportMsg')
    async with pg.expect_download() as dl: await pg.click('#btnSave')
    path = f'{OUT}_{mode}.mp4'; await (await dl.value).save_as(path)
    await pg.click('#btnExportCancel'); await pg.wait_for_timeout(300)
    return path, dt, msg

async def main():
    fails = []
    def check(name, ok, detail=''):
        print(('PASS ' if ok else 'FAIL ') + name + (': ' + detail if detail else ''), flush=True)
        if not ok: fails.append(name)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=ARGS)
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []; logs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: logs.append(m.text) if 'MacroCut timing' in m.text else None)
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip && !document.getElementById("btnPlay").disabled', timeout=60000)
        await pg.evaluate(f'{S}.segments = [{{start: 0.5, end: 2.5}}]')
        await pg.click('.dock-item[data-tab="tab-stab"]'); await pg.click('#stabLevel .pill-btn[data-level="max"]')
        await pg.wait_for_function(f'{S}.analysis && !{S}.analysing', timeout=900000)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="pop"]')
        res = {}
        for mode in ['canvas', 'yuv']:
            path, dt, msg = await export(pg, mode)
            t = json.loads(logs[-1].split('MacroCut timing ', 1)[1]) if logs else {}
            res[mode] = (path, dt, t)
            print(f'  {mode}: {dt:.1f} s, output route {t.get("outMode")}, {msg[:160]}')
        # a phone whose encoder won't take the GPU's I420 frames: the export must redo itself the old way and finish
        ctx2 = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        await ctx2.add_init_script('''(() => { const VF = window.VideoFrame; window.VideoFrame = class extends VF {
          constructor(src, init) { if (init && init.format === 'I420' && init.codedWidth > 16) throw new TypeError('simulated: I420 frames not accepted'); super(src, init); } }; })();''')
        pg2 = await ctx2.new_page(); warns = []
        pg2.on('pageerror', lambda e: errs.append(str(e)))
        pg2.on('console', lambda m: warns.append(m.text) if 'canvas route' in m.text else (logs.append(m.text) if 'MacroCut timing' in m.text else None))
        await pg2.goto('http://localhost:8765/app/index.html'); await pg2.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg2.reload()
        await pg2.set_input_files('#fileInput', CLIP)
        await pg2.wait_for_function(f'{S}.clip && !document.getElementById("btnPlay").disabled', timeout=60000)
        await pg2.evaluate(f'{S}.segments = [{{start: 0.5, end: 1.5}}]')
        await pg2.click('#btnExport'); await pg2.wait_for_timeout(200); await pg2.click('#btnExportGo')
        await pg2.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|lost|stopped/i.test(document.getElementById("exportMsg").textContent)', timeout=1800000, polling=500)
        msg2 = await pg2.inner_text('#exportMsg')
        t2 = json.loads(logs[-1].split('MacroCut timing ', 1)[1]) if logs else {}
        check('falls back to the canvas route if the encoder rejects GPU frames', 'Saved to your phone' in msg2 and t2.get('outMode') == 'canvas' and warns, msg2[:100])
        await b.close()
    check('canvas route used when asked', res['canvas'][2].get('outMode') == 'canvas')
    check('GPU I420 route used by default', res['yuv'][2].get('outMode') == 'yuv')
    fa, aa = frames(res['canvas'][0]); fb, ab = frames(res['yuv'][0])
    check('same number of frames', len(fa) == len(fb), f'{len(fa)} vs {len(fb)}')
    n = min(len(fa), len(fb))
    mse = ((fa[:n] - fb[:n]) ** 2).reshape(n, -1).mean(1); ps = 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-6))
    check('same picture (within encoding noise)', ps.min() > 30, f'PSNR mean {ps.mean():.1f} dB, min {ps.min():.1f} dB')
    da = fa[:n].reshape(-1, 3).mean(0); db = fb[:n].reshape(-1, 3).mean(0)
    print(f'  mean RGB canvas {da.round(2)} vs GPU I420 {db.round(2)} (the canvas route depends on the browser\'s own converter)')
    bx = boxes(res['yuv'][0])
    check('fast-start MP4 (index before the video data)', bx[:2] == ['ftyp', 'moov'] and 'mdat' in bx, ' '.join(bx))
    check('sound kept', ab, 'audio track present' if ab else 'no audio track')
    check('no page errors', not errs, str(errs[:3]))
    print('ALL PASS' if not fails else f'{len(fails)} FAILED')

asyncio.run(main())
