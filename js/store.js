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
