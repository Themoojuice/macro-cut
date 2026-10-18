"""Simulate a phone whose hardware decoder never answers: loading must finish or fail with a message, never hang."""
import asyncio, os, sys, time
os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.async_api import async_playwright
CLIP = sys.argv[1]
STUBS = {
 'decoder never outputs': """
   const D = window.VideoDecoder; window.VideoDecoder = class extends D { decode() { /* swallow */ } get decodeQueueSize() { return 9; } flush() { return new Promise(() => {}); } };
   window.VideoDecoder.isConfigSupported = D.isConfigSupported.bind(D);""",
 'decoder check never returns': """
   window.VideoDecoder.isConfigSupported = () => new Promise(() => {});""",
}
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        for name, js in STUBS.items():
            ctx = await b.new_context(viewport={'width': 412, 'height': 915})
            await ctx.add_init_script(js)
            pg = await ctx.new_page()
            await pg.goto('http://localhost:8765/app/index.html')
            t0 = time.time(); await pg.set_input_files('#fileInput', CLIP)
            await pg.wait_for_function('window.__macrocut.state.clip || document.getElementById("emptyState").style.display !== "none"', timeout=60000)
            ok = await pg.evaluate('!!window.__macrocut.state.clip')
            print(f'{name}: settled in {time.time()-t0:.1f}s, loaded={ok}, message: {await pg.inner_text("#toast")}')
            await ctx.close()
        await b.close()
asyncio.run(main())
