// Demuxing (mp4box.js) + WebCodecs decode helpers.
/* global MP4Box, DataStream */

function boxDescription(entry) {
  const box = entry.avcC || entry.hvcC || entry.av1C || entry.vpcC;
  if (!box) return undefined;
  const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
  box.write(stream);
  return new Uint8Array(stream.buffer, 8); // strip box header
}

function audioDescription(entry) {
  try {
    const d = entry.esds.esd.descs[0].descs[0];
    if (d && d.data) return new Uint8Array(d.data);
  } catch (e) { /* no esds */ }
  return undefined;
}

// Clockwise display rotation (0/90/180/270) from a tkhd matrix (16.16 fixed point).
export function matrixRotation(m) {
  if (!m || m.length < 5) return 0;
  const deg = Math.atan2(m[1], m[0]) * 180 / Math.PI;
  return ((Math.round(deg / 90) * 90) % 360 + 360) % 360;
}

// Sliding-window reader: sequential decode only ever holds ~16 MB of the file in memory.
export class WindowReader {
  constructor(file, win = 16 << 20) { this.file = file; this.win = win; this.start = 0; this.buf = null; }
  async read(off, len) {
    if (!this.buf || off < this.start || off + len > this.start + this.buf.length) {
      const end = Math.min(this.file.size, off + Math.max(this.win, len));
      this.buf = new Uint8Array(await this.file.slice(off, end).arrayBuffer());
      this.start = off;
    }
    return this.buf.subarray(off - this.start, off - this.start + len);
  }
}

async function bytes(file, off, len) { return new Uint8Array(await file.slice(off, off + len).arrayBuffer()); }

// Walk top-level boxes; return ftyp + moov bytes only (sample data stays on disk).
async function readHeader(file) {
  let off = 0, ftyp = null, moov = null, frag = false;
  while (off + 8 <= file.size) {
    const h = await bytes(file, off, 16), dv = new DataView(h.buffer);
    let size = dv.getUint32(0); const type = String.fromCharCode(h[4], h[5], h[6], h[7]);
    if (size === 1) size = Number(dv.getBigUint64(8)); else if (size === 0) size = file.size - off;
    if (size < 8) break;
    if (type === 'ftyp') ftyp = await bytes(file, off, size);
    else if (type === 'moov') moov = await bytes(file, off, size);
    else if (type === 'moof') frag = true;
    off += size;
  }
  if (!moov) throw new Error(frag ? 'Fragmented MP4 is not supported yet.' : 'Not an MP4/MOV file I can read (no movie header found).');
  const out = new Uint8Array((ftyp ? ftyp.length : 0) + moov.length);
  if (ftyp) out.set(ftyp, 0);
  out.set(moov, ftyp ? ftyp.length : 0);
  return out.buffer;
}

export async function openClip(file, onStatus = () => {}) {
  onStatus('Reading header…');
  const buf = await readHeader(file);
  onStatus('Parsing container…');
  const mp4 = MP4Box.createFile();
  const info = await new Promise((resolve, reject) => {
    mp4.onError = e => reject(new Error('Container error: ' + e));
    mp4.onReady = resolve;
    buf.fileStart = 0;
    mp4.appendBuffer(buf);
    mp4.flush();
    setTimeout(() => reject(new Error('Not an MP4/MOV file I can read (no movie header found).')), 50);
  });
  const vt = info.videoTracks && info.videoTracks[0];
  if (!vt) throw new Error('No video track in this file.');
  const vtrak = mp4.getTrackById(vt.id);
  const entry = vtrak.mdia.minf.stbl.stsd.entries[0];
  const codec = vt.codec;
  const vsamples = vtrak.samples;
  if (!vsamples || !vsamples.length) throw new Error('Video track has no samples (fragmented MP4 not supported yet).');
  const ts = vt.timescale;
  let minCts = Infinity;
  for (const s of vsamples) if (s.cts < minCts) minCts = s.cts;
  const toUs = c => Math.round((c - minCts) * 1e6 / ts);
  const samples = vsamples.map((s, i) => ({ i, offset: s.offset, size: s.size, ts: toUs(s.cts), dur: Math.round(s.duration * 1e6 / ts), key: !!s.is_sync }));
  const pts = samples.map(s => s.ts).sort((a, b) => a - b);
  const times = Float64Array.from(pts, t => t / 1e6);
  const desc = /^(avc|hvc|hev|av01)/.test(codec) ? boxDescription(entry) : undefined;
  const decoderConfig = { codec, codedWidth: entry.width, codedHeight: entry.height };
  if (desc) decoderConfig.description = desc;
  const rotation = matrixRotation(vt.matrix);
  const codedW = entry.width, codedH = entry.height;
  const dispW = rotation % 180 ? codedH : codedW, dispH = rotation % 180 ? codedW : codedH;

  let support;
  try { support = await VideoDecoder.isConfigSupported(decoderConfig); } catch (e) { support = { supported: false }; }
  if (!support.supported) throw new Error(`This browser can't decode ${codec}. Try recording in H.264 (turn off "High efficiency video").`);

  // fps estimate (median frame interval)
  const iv = []; for (let i = 1; i < times.length; i++) iv.push(times[i] - times[i - 1]);
  iv.sort((a, b) => a - b);
  const fps = iv.length ? 1 / iv[iv.length >> 1] : 30;

  let audio = null;
  const at = info.audioTracks && info.audioTracks[0];
  if (at && /^mp4a/.test(at.codec)) {
    const atrak = mp4.getTrackById(at.id), aentry = atrak.mdia.minf.stbl.stsd.entries[0];
    const ats = at.timescale;
    audio = {
      codec: at.codec,
      sampleRate: at.audio.sample_rate,
      channels: at.audio.channel_count,
      description: audioDescription(aentry),
      samples: atrak.samples.map(s => ({ offset: s.offset, size: s.size, ts: Math.round(s.cts * 1e6 / ats), dur: Math.round(s.duration * 1e6 / ats) })),
    };
  }
  const duration = times[times.length - 1] + (iv.length ? iv[iv.length >> 1] : 1 / 30);
  return {
    file, reader: new WindowReader(file), codec, decoderConfig, samples, times, fps, rotation,
    codedW, codedH, dispW, dispH, duration, audio,
    name: file.name,
  };
}

export async function chunkFor(clip, s, reader = clip.reader) {
  return new EncodedVideoChunk({ type: s.key ? 'key' : 'delta', timestamp: s.ts, duration: s.dur, data: await reader.read(s.offset, s.size) });
}

/**
 * Decode frames in presentation order between [fromUs, toUs]. onFrame(frame) may be async and MUST close the frame.
 * Frames outside the range are closed automatically. Returns when done or aborted (signal.aborted).
 */
export async function decodeRange(clip, { fromUs = 0, toUs = Infinity, onFrame, signal, maxQueue = 3 }) {
  const S = clip.samples;
  // decode-order start: last keyframe whose ts <= fromUs (search decode order)
  let start = 0;
  for (let i = 0; i < S.length; i++) if (S[i].key && S[i].ts <= fromUs) start = i;
  // decode-order end: include samples until we're past toUs by a margin (B-frame reordering)
  let end = S.length - 1;
  for (let i = start; i < S.length; i++) { if (S[i].key && S[i].ts > toUs) { end = i - 1; break; } }
  const pending = [];
  let err = null, wake = null;
  const poke = () => { if (wake) { const w = wake; wake = null; w(); } };
  const dec = new VideoDecoder({
    output: f => { pending.push(f); poke(); },
    error: e => { err = e; poke(); },
  });
  dec.addEventListener('dequeue', poke);
  dec.configure(clip.decoderConfig);
  const reader = new WindowReader(clip.file); // own window: decodes may run alongside each other
  let i = start, flushed = false;
  const drain = async () => {
    while (pending.length) {
      const f = pending.shift();
      if (signal && signal.aborted) { f.close(); continue; }
      if (f.timestamp < fromUs || f.timestamp > toUs) { f.close(); continue; }
      await onFrame(f);
    }
  };
  try {
    while (true) {
      if (err) throw err;
      if (signal && signal.aborted) break;
      if (pending.length) { await drain(); continue; }
      if (i <= end) {
        if (dec.decodeQueueSize < 4 && pending.length < maxQueue) { const ch = await chunkFor(clip, S[i++], reader); if (dec.state === 'configured') dec.decode(ch); continue; }
      } else if (!flushed) {
        flushed = true;
        await dec.flush();
        continue;
      } else break;
      await new Promise(r => { wake = r; setTimeout(r, 30); });
    }
    await drain();
  } finally {
    for (const f of pending) f.close();
    if (dec.state !== 'closed') dec.close();
  }
  if (err) throw err;
}
