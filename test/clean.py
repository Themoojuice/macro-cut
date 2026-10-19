"""Extra clean vs noise-free ground truth, through the real app. Also checks moving areas for ghosting."""
import asyncio, os, sys, subprocess, numpy as np, cv2
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
NOISY, CLEAN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W, H = 640, 360
def frames(p):
    o = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', 'scale=in_color_matrix=bt709:in_range=tv,format=rgb24', '-f', 'rawvideo', '-'], capture_output=True).stdout
    return np.frombuffer(o, np.uint8).reshape(-1, H, W, 3).astype(np.float32)
async def run(pg, clean, name):
    await pg.evaluate(f'() => {{ window.__macrocut.state.exp.clean = {str(clean).lower()}; }}')
    await pg.click('#btnExport'); await pg.click('#btnExportGo')
    await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || document.getElementById("exportMsg").textContent.includes("failed")', timeout=3000000)
    print(name, '|', await pg.inner_text('#exportMsg'), flush=True)
    async with pg.expect_download() as dl: await pg.click('#btnSave')
    p = f'{OUT}_{name}.mp4'; await (await dl.value).save_as(p); await pg.click('#btnExportCancel'); return p
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        pg = await (await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)).new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: (m.type == 'error' and errs.append(m.text)) or ('blend threshold' in m.text and print('  ', m.text, flush=True)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', NOISY)
        await pg.wait_for_function('window.__macrocut.state.clip', timeout=60000)
        await pg.click('.dock-item[data-tab="tab-look"]'); await pg.click('#lookPresets .pill-btn[data-preset="off"]')
        off = await run(pg, False, 'off'); on = await run(pg, True, 'clean')
        print('errors', errs[:5]); await b.close()
    G = frames(CLEAN)
    masks = []
    for i in range(len(G)):
        m = np.zeros((H, W), bool)
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(G):
                sh, _ = cv2.phaseCorrelate(cv2.GaussianBlur(G[i].mean(-1), (0, 0), 3), cv2.GaussianBlur(G[j].mean(-1), (0, 0), 3))
                M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
                m |= np.abs(cv2.warpAffine(G[j], M, (W, H), flags=cv2.WARP_INVERSE_MAP) - G[i]).max(-1) > 20
        masks.append(cv2.erode(m.astype(np.uint8), np.ones((5, 5))).astype(bool))
    ps = lambda a, b, m=None: 10 * np.log10(255 ** 2 / np.mean(((a - b) ** 2) if m is None else ((a - b) ** 2)[m]))
    for name, path in (('normal export', off), ('extra clean', on)):
        F = frames(path); n = min(len(F), len(G))
        allp = np.mean([ps(F[i], G[i]) for i in range(2, n - 2)])
        mov = np.mean([ps(F[i], G[i], masks[i]) for i in range(2, n - 2) if masks[i].sum() > 300])
        print(f'{name:14s} {len(F)} frames | vs noise-free truth: whole frame {allp:.1f} dB | moving areas {mov:.1f} dB')
    F = frames(on); k = 45
    cv2.imwrite(OUT + '_crop.png', np.hstack([frames(off)[k][100:280, 200:440], F[k][100:280, 200:440], G[k][100:280, 200:440]])[..., ::-1])
asyncio.run(main())
