// Network-first (new commits show up immediately), falling back to cache so the app also works offline in the field.
const CACHE = 'macrocut-v5';
const SHELL = ['./', './index.html', './css/app.css?v=5', './js/app.js?v=5', './js/core.js', './js/gpu.js', './js/media.js', './js/engine.js',
  './vendor/mp4box.all.min.js', './vendor/mp4-muxer.js', './manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).catch(() => {}).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(
    fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: false }).then(r => r || caches.match(req, { ignoreSearch: true })))
  );
});
