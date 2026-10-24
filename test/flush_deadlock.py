"""Phone hardware decoders hold back the last few frames until flush and have only a few frame slots. Waiting on
flush without taking frames deadlocked exports at 100% ("Video decoder timed out"). This fakes such a decoder
(holds 6 frames, 3 slots) and checks the export finishes. Usage: flush_deadlock.py clip.mp4"""
import asyncio, os, sys
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/opt/pw-browsers')
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
S = 'window.__macrocut.state'
FAKE = """
(() => {
  const Orig = VideoDecoder, HOLD = 6, SLOTS = 3;
  window.VideoDecoder = class extends Orig {
    constructor(init) {
      const q = []; let open = 0, flushing = false, me = null;
      const pump = () => {
        while (q.length && open < SLOTS && (flushing || q.length > HOLD)) {
          const f = q.shift(); open++; const c = f.close.bind(f); let done = false;
          f.close = () => { if (!done) { done = true; open--; c(); setTimeout(pump, 0); } };
          init.output(f);
        }
        if (flushing && !q.length && me && me._fw) { const w = me._fw; me._fw = null; w(); }
      };
      super({ output: f => { q.push(f); pump(); }, error: init.error });
      me = this; this._q = q; this._pump = pump; this._setFlush = v => { flushing = v; };
    }
    flush() { return super.flush().then(() => new Promise(r => { this._setFlush(true); this._fw = r; this._pump(); })).then(() => this._setFlush(false)); }
  };
})();
"""

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        ctx = await b.new_context(viewport={'width': 412, 'height': 915}, accept_downloads=True)
        await ctx.add_init_script(FAKE)
        pg = await ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        await pg.goto('http://localhost:8765/app/index.html'); await pg.evaluate('localStorage.clear(); indexedDB.deleteDatabase("macrocut")'); await pg.reload()
        await pg.set_input_files('#fileInput', CLIP)
        await pg.wait_for_function(f'{S}.clip', timeout=60000); await pg.wait_for_timeout(500)
        await pg.evaluate(f'{S}.segments = [{{start: 0.5, end: 3}}]')
        await pg.click('#btnExport'); await pg.click('#btnExportGo')
        await pg.wait_for_function('document.getElementById("btnSave").style.display !== "none" || /failed|lost|stopped/i.test(document.getElementById("exportMsg").textContent)', timeout=600000)
        msg = await pg.inner_text('#exportMsg'); print('  ', msg[:200])
        ok = 'Saved to your phone' in msg and 'early' not in msg and not errs
        print('PASS export finishes with a few-slot decoder' if ok else 'FAIL export finishes with a few-slot decoder', errs)
        await b.close()
    print('ALL PASS' if ok else '1 FAILED')
asyncio.run(main())
