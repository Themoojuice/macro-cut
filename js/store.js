// Saved edits per clip (IndexedDB) and named looks (localStorage). Everything stays on the phone.
const DB = 'macrocut', STORE = 'projects';
let dbp = null;
function db() {
  if (!dbp) dbp = new Promise((res, rej) => {
    if (!('indexedDB' in self)) return rej(new Error('no IndexedDB'));
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(STORE);
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
    setTimeout(() => rej(new Error('storage timed out')), 4000);
  }).catch(e => { dbp = null; throw e; });
  return dbp;
}
async function tx(mode, fn) {
  const d = await db();
  return new Promise((res, rej) => {
    const t = d.transaction(STORE, mode), s = t.objectStore(STORE), r = fn(s);
    t.oncomplete = () => res(r && r.result); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error);
  });
}
export const clipKey = file => `${file.name}|${file.size}|${file.lastModified || 0}`;
export const loadProject = key => tx('readonly', s => s.get(key));
export const saveProject = (key, p) => tx('readwrite', s => s.put(p, key));
export const deleteProject = key => tx('readwrite', s => s.delete(key));

const LOOKS = 'macrocut.looks';
export function savedLooks() { try { return JSON.parse(localStorage.getItem(LOOKS) || '[]'); } catch (e) { return []; } }
export function setSavedLooks(list) { try { localStorage.setItem(LOOKS, JSON.stringify(list)); } catch (e) { /* storage full / blocked */ } }

// The last finished export, kept until the next one, so a video is never lost by closing the sheet or reloading.
const LAST = '__last_export__';
export const saveLastExport = item => tx('readwrite', s => s.put(item, LAST));
export const loadLastExport = () => tx('readonly', s => s.get(LAST));

// Working copy of the opened clip in the browser's private storage. Android only lends the browser read access
// to a picked file, and that access can lapse mid-export ("The requested file could not be read"). Reading
// from our own copy never lapses. Only the current clip is kept; older copies are deleted to free space.
export async function safeCopy(file, onProgress = () => {}) {
  if (!navigator.storage || !navigator.storage.getDirectory) return file;
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle('clips', { create: true });
  const name = `${file.size}_${file.lastModified || 0}_${file.name}`.replace(/[^\w.\-]+/g, '_').slice(-120);
  for await (const [n] of dir.entries()) if (n !== name) await dir.removeEntry(n).catch(() => {});
  try {   // same clip opened again: reuse the copy if it's complete
    const h = await dir.getFileHandle(name), f = await h.getFile();
    if (f.size === file.size) return new File([f], file.name, { type: file.type, lastModified: file.lastModified });
  } catch (e) { /* no copy yet */ }
  try {
    const est = navigator.storage.estimate ? await navigator.storage.estimate() : null;
    if (est && est.quota && est.quota - (est.usage || 0) < file.size * 1.1) return file;   // no room: use the original
    const h = await dir.getFileHandle(name, { create: true }), w = await h.createWritable();
    const reader = file.stream().getReader();
    let done = 0, last = 0;
    try {
      while (true) {
        const { value, done: end } = await reader.read();
        if (end) break;
        await w.write(value); done += value.length;
        const p = Math.floor(100 * done / file.size); if (p !== last) { last = p; onProgress(p); }
      }
      await w.close();
    } catch (e) { await w.abort().catch(() => {}); await dir.removeEntry(name).catch(() => {}); throw e; }
    const f = await h.getFile();
    return new File([f], file.name, { type: file.type, lastModified: file.lastModified });
  } catch (e) { console.warn('working copy failed, reading the original', e); return file; }
}
