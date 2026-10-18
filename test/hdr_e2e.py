"""HDR clip through the real app: preview calibration + neutral export colour vs ffmpeg colour-managed reference."""
import asyncio, os, sys, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP, OUT = sys.argv[1], sys.argv[2]
def ff(path, t, vf):
    w, h = 960, 540
    o = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(t), '-i', path, '-frames:v', '1', '-vf', vf + f',scale={w}:{h},format=bgr24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(h, w, 3)
def stats(x): lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(float); return lab[..., 0].mean(), np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean()
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear()'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function('window.__macrocut.state.clip && !document.getElementById("btnPlay").disabled', timeout=60000)
        print('badge:', await pg.inner_text('#metaBadge'), '| preview info:', await pg.evaluate('JSON.stringify(window.__macrocut.state.previewInfo)'))
        await pg.wait_for_timeout(600); await pg.screenshot(path=OUT + '_preview.png')
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        await pg.evaluate('window.__macrocut.state.segments = [{start: 1.0, end: 1.5}]')
        await pg.click('#btnExport'); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=600000)
        print(await pg.inner_text('#exportMsg'))
        async with pg.expect_download() as dl: await pg.click('#btnSave')
        await (await dl.value).save_as(OUT + '.mp4')
        print('errors', errs); await b.close()
    out = ff(OUT + '.mp4', 0, 'scale=in_color_matrix=bt709:in_range=tv')
    ref = ff(CLIP, 1.0, 'zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
    naive = ff(CLIP, 1.0, 'scale=in_color_matrix=bt709:in_range=tv')
    for n, x in (('naive (old app)', naive), ('app export', out), ('ffmpeg reference', ref)):
        l, c = stats(x); print(f'{n:18s} luma {l:5.1f} chroma {c:5.1f}')
    print('mean abs diff export vs reference: %.1f' % np.abs(out.astype(int) - ref.astype(int)).mean())
    cv2.imwrite(OUT + '_cmp.png', cv2.resize(np.hstack([naive, out, ref]), None, fx=0.5, fy=0.5))
asyncio.run(main())
