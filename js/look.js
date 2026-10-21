// Look = a starting preset (Natural / Spood pop / none) + your own tweaks on top + an Off switch.
// Tweaks are stored as -1…+1 per slider, where 0 means "exactly what the preset gives".
//   +1 pushes the setting to its maximum, -1 to its minimum, so every slider position does something
//   and the slider always starts in the middle (or at the left for effects that can't go below zero).
// resolveLook() turns this into the absolute numbers the renderer uses.

// Absolute limits of each setting (what the renderer accepts).
export const LIMITS = {
  cas: [0, 1], detail: [0, 2.5], radius: [0.6, 4], denoise: [0, 1],
  exposure: [-2, 2], contrast: [-0.5, 0.6], shadows: [-0.5, 1], highlights: [-0.5, 1],
  saturation: [-1, 1], vibrance: [-0.5, 1.5], warmth: [-1, 1], tint: [-1, 1],
};
export const KEYS = Object.keys(LIMITS);

// The picture untouched: no sharpening, no colour, no tone changes.
export const NEUTRAL = Object.freeze({
  cas: 0, detail: 0, radius: 1.5, denoise: 0,
  exposure: 0, contrast: 0, shadows: 0, highlights: 0, saturation: 0, vibrance: 0, warmth: 0, tint: 0,
});
const NATURAL = {
  cas: 0.35, detail: 0.6, radius: 1.6, denoise: 0.15,
  exposure: 0, contrast: 0.05, shadows: 0.15, highlights: 0.2, saturation: 0.1, vibrance: 0.25, warmth: 0, tint: 0,
};
export const PRESETS = {
  natural: { ...NATURAL },
  pop: { ...NATURAL, cas: 0.55, detail: 1.1, radius: 1.4, denoise: 0.25, contrast: 0.12, shadows: 0.25, highlights: 0.3, saturation: 0.15, vibrance: 0.45 },
};

export const baseOf = look => (look && PRESETS[look.preset]) || NEUTRAL;
export const defaultLook = () => ({ preset: 'natural', off: false, adj: {} });
export const cloneLook = L => ({ preset: L.preset ?? null, off: !!L.off, adj: { ...(L.adj || {}) } });

// One setting: preset value moved by tweak a (-1…+1) towards the limit on that side.
export function applyAdj(key, base, a) {
  const [lo, hi] = LIMITS[key];
  if (!a) return base;
  return a > 0 ? base + a * (hi - base) : base + a * (base - lo);
}
// Inverse of applyAdj (used to read old-style saved looks).
export function adjFor(key, base, v) {
  const [lo, hi] = LIMITS[key];
  v = Math.min(hi, Math.max(lo, v));
  if (Math.abs(v - base) < 1e-9) return 0;
  return v > base ? (hi > base ? (v - base) / (hi - base) : 0) : (base > lo ? (v - base) / (base - lo) : 0);
}
// True if this slider can't go below the preset's value (e.g. sharpening when the preset has none).
export const oneSided = (key, base) => base[key] <= LIMITS[key][0] + 1e-9;

export function resolveLook(L) {
  if (!L) return { ...PRESETS.natural };
  if (L.off) return { ...NEUTRAL };
  const base = baseOf(L), o = {};
  for (const k of KEYS) o[k] = applyAdj(k, base[k], L.adj[k] || 0);
  return o;
}
export const hasTweaks = L => KEYS.some(k => Math.abs(L.adj[k] || 0) > 1e-6);

// Accept a look in any format ever saved (new style, or old absolute numbers) and return a new-style look.
export function normLook(x) {
  if (!x || typeof x !== 'object') return defaultLook();
  if ('adj' in x || 'preset' in x) {
    const L = { preset: PRESETS[x.preset] ? x.preset : null, off: !!x.off, adj: {} };
    for (const k of KEYS) { const a = +(x.adj || {})[k]; if (a && isFinite(a)) L.adj[k] = Math.max(-1, Math.min(1, a)); }
    return L;
  }
  // Old style: absolute numbers. Pick the preset it matches best, keep the difference as tweaks.
  const abs = { ...NEUTRAL };
  for (const k of KEYS) if (typeof x[k] === 'number' && isFinite(x[k])) abs[k] = x[k];
  const dist = B => KEYS.reduce((s, k) => s + Math.abs(abs[k] - B[k]) / (LIMITS[k][1] - LIMITS[k][0]), 0);
  const cands = [['natural', PRESETS.natural], ['pop', PRESETS.pop], [null, NEUTRAL]];
  const [preset, base] = cands.reduce((b, c) => dist(c[1]) < dist(b[1]) - 1e-9 ? c : b);
  const L = { preset, off: false, adj: {} };
  for (const k of KEYS) { const a = adjFor(k, base[k], abs[k]); if (Math.abs(a) > 1e-6) L.adj[k] = a; }
  return L;
}
