// WebGL2 pipeline: orientation-aware sampling, bicubic stabilising warp,
// halo-limited detail + CAS sharpening, bokeh smoothing, OKLab grading.

const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

// display-normalised (height units, centred, y down) → texture uv of the uploaded (coded-orientation) image
const ORIENT = `
uniform vec2 u_dispSize; uniform vec2 u_texSize; uniform int u_rot;
vec2 dispToUV(vec2 s) {
  vec2 pd = s * u_dispSize.y, pc;
  if (u_rot == 0) pc = pd;
  else if (u_rot == 1) pc = vec2(pd.y, -pd.x);
  else if (u_rot == 2) pc = -pd;
  else pc = vec2(-pd.y, pd.x);
  return pc / u_texSize + 0.5;
}`;

const FS_ANALYSIS = `#version 300 es
precision highp float;
uniform sampler2D u_src; uniform vec2 u_out;
${ORIENT}
out vec4 o;
void main(){
  // row 0 of readPixels = top of image
  vec2 q = gl_FragCoord.xy;
  float A = u_dispSize.x / u_dispSize.y;
  float acc = 0.0;
  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) {
    vec2 off = (vec2(float(i), float(j)) + 0.5) / 4.0 - 0.5;
    vec2 pn = vec2(((q.x + off.x) / u_out.x - 0.5) * A, (q.y + off.y) / u_out.y - 0.5);
    vec3 c = texture(u_src, dispToUV(pn)).rgb;
    acc += dot(c, vec3(0.299, 0.587, 0.114));
  }
  float l = acc / 16.0;
  o = vec4(l, l, l, 1.0);
}`;

// Undo a BT.709 interpretation of an HDR (BT.2020 HLG/PQ) signal, then convert to SDR BT.709:
// HLG inverse OETF + OOTF (1000 nit) or PQ EOTF, ref-white 203 nit → 1.0, 2020→709 gamut,
// Möbius highlight roll-off on the max channel (hue/saturation preserving), sRGB encode.
const HDR = `
uniform int u_hdr;
vec3 hdrFix(vec3 n) {
  float Y = dot(n, vec3(0.2126, 0.7152, 0.0722));
  float Cb = (n.b - Y) / 1.8556, Cr = (n.r - Y) / 1.5748;
  float R = Y + 1.4746 * Cr, B = Y + 1.8814 * Cb, G = (Y - 0.2627 * R - 0.0593 * B) / 0.6780;
  vec3 E = clamp(vec3(R, G, B), 0.0, 1.0), lin;
  if (u_hdr == 1) {
    const float a = 0.17883277, b = 0.28466892, c = 0.55991073;
    lin = mix(E * E / 3.0, (exp((E - c) / a) + b) / 12.0, step(0.5, E));
    float Ys = dot(lin, vec3(0.2627, 0.6780, 0.0593));
    lin = 1000.0 * pow(max(Ys, 1e-6), 0.2) * lin / 203.0;
  } else {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 p = pow(E, vec3(1.0 / m2));
    lin = 10000.0 * pow(max(p - c1, 0.0) / (c2 - c3 * p), vec3(1.0 / m1)) / 203.0;
  }
  vec3 rgb = max(mat3(1.6605, -0.1246, -0.0182, -0.5876, 1.1329, -0.1006, -0.0728, -0.0083, 1.1187) * lin, 0.0);
  const float P = 1000.0 / 203.0, j = 0.3;
  float sig = max(max(rgb.r, max(rgb.g, rgb.b)), 1e-6);
  float aa = -j * j * (P - 1.0) / (j * j - 2.0 * j + P), bb = (j * j - 2.0 * j * P + P) / (P - 1.0);
  float m = sig <= j ? sig : (bb * bb + 2.0 * bb * j + j * j) / (bb - aa) * (sig + aa) / (sig + bb);
  rgb = clamp(rgb * (m / sig), 0.0, 1.0);
  return mix(rgb * 12.92, 1.055 * pow(rgb, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, rgb));
}`;

const FS_WARP = `#version 300 es
precision highp float;
uniform sampler2D u_src; uniform vec2 u_out; uniform mat2 u_L; uniform vec2 u_t; uniform int u_cubic;
uniform sampler2D u_srcB; uniform sampler2D u_flow; uniform int u_interp; uniform float u_it;
// focus build-up overlay: stored canvas (rgb + coverage), take-map (g), frame → canvas similarity
uniform int u_fo, u_fcubic; uniform sampler2D u_facc, u_fdec; uniform mat2 u_G; uniform vec2 u_Gt, u_fC;
${ORIENT}
${HDR}
out vec4 o;
vec4 cubicSz(sampler2D src, vec2 uv, vec2 sz) {
  vec2 sp = uv * sz, t1 = floor(sp - 0.5) + 0.5, f = sp - t1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / sz, t0 = (t1 - 1.0) / sz, t3 = (t1 + 2.0) / sz;
  vec4 r = vec4(0.0);
  r += textureLod(src, vec2(t0.x, t0.y), 0.0) * w0.x * w0.y;   r += textureLod(src, vec2(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureLod(src, vec2(t3.x, t0.y), 0.0) * w3.x * w0.y;   r += textureLod(src, vec2(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureLod(src, vec2(t12.x, t12.y), 0.0) * w12.x * w12.y; r += textureLod(src, vec2(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureLod(src, vec2(t0.x, t3.y), 0.0) * w0.x * w3.y;   r += textureLod(src, vec2(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r += textureLod(src, vec2(t3.x, t3.y), 0.0) * w3.x * w3.y;
  return r;
}
// where the stored view is valid and clearly sharper than the live frame, show it instead of the live pixel
vec3 focusOverlay(vec3 live, vec2 s) {
  vec2 p = u_G * s + u_Gt, auv = vec2(p.x / u_fC.x + 0.5, p.y / u_fC.y + 0.5);
  if (any(lessThan(auv, vec2(0.001))) || any(greaterThan(auv, vec2(0.999)))) return live;
  vec4 a = u_fcubic == 1 ? cubicSz(u_facc, auv, vec2(textureSize(u_facc, 0))) : texture(u_facc, auv);
  float use = clamp(a.a, 0.0, 1.0) * (1.0 - texture(u_fdec, auv).b);
  return mix(live, clamp(a.rgb, 0.0, 1.0), use);
}
vec4 cubic(sampler2D u_src, vec2 uv) {
  vec2 sp = uv * u_texSize, t1 = floor(sp - 0.5) + 0.5, f = sp - t1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / u_texSize, t0 = (t1 - 1.0) / u_texSize, t3 = (t1 + 2.0) / u_texSize;
  vec4 r = vec4(0.0);
  r += textureLod(u_src, vec2(t0.x, t0.y), 0.0) * w0.x * w0.y;
  r += textureLod(u_src, vec2(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureLod(u_src, vec2(t3.x, t0.y), 0.0) * w3.x * w0.y;
  r += textureLod(u_src, vec2(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureLod(u_src, vec2(t12.x, t12.y), 0.0) * w12.x * w12.y;
  r += textureLod(u_src, vec2(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureLod(u_src, vec2(t0.x, t3.y), 0.0) * w0.x * w3.y;
  r += textureLod(u_src, vec2(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r += textureLod(u_src, vec2(t3.x, t3.y), 0.0) * w3.x * w3.y;
  return r;
}
void main(){
  vec2 q = vec2(gl_FragCoord.x, u_out.y - gl_FragCoord.y);
  vec2 on = vec2((q.x / u_out.x - 0.5) * (u_out.x / u_out.y), q.y / u_out.y - 0.5);
  vec2 s = u_L * on + u_t;
  vec2 uv = dispToUV(s);
  if (any(lessThan(uv, vec2(-0.002))) || any(greaterThan(uv, vec2(1.002)))) { o = vec4(0.06, 0.07, 0.08, 1.0); return; }
  vec3 c;
  if (u_interp == 1) {
    // in-between frame: move both neighbouring frames along the measured motion and blend
    float A = u_dispSize.x / u_dispSize.y;
    vec2 F = texture(u_flow, vec2(s.x / A + 0.5, s.y + 0.5)).rg;
    vec2 ua = dispToUV(s - u_it * F), ub = dispToUV(s + (1.0 - u_it) * F);
    vec3 a = u_cubic == 1 ? cubic(u_src, ua).rgb : texture(u_src, ua).rgb;
    vec3 b = u_cubic == 1 ? cubic(u_srcB, ub).rgb : texture(u_srcB, ub).rgb;
    c = clamp(mix(a, b, u_it), 0.0, 1.0);
  } else {
    c = clamp(u_cubic == 1 ? cubic(u_src, uv).rgb : texture(u_src, uv).rgb, 0.0, 1.0);
  }
  if (u_fo == 1) c = focusOverlay(c, s);
  if (u_hdr > 0) c = hdrFix(c);
  o = vec4(c, 1.0);
}`;

// ── Focus build-up (canvas = anchor frame's normalised coords + margin; see js/focus.js) ──
const FOCUS_COMMON = `
uniform mat2 u_T; uniform vec2 u_Tt; uniform vec2 u_C;
vec2 accToFrame(vec2 p) { return u_T * p + u_Tt; }
vec2 accP(vec2 uv) { return vec2((uv.x - 0.5) * u_C.x, (uv.y - 0.5) * u_C.y); }
vec2 accUV(vec2 p) { return vec2(p.x / u_C.x + 0.5, p.y / u_C.y + 0.5); }
bool outside(vec2 uv) { return any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))); }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }`;

// Pass 1 (decision grid): how sharp is the incoming frame here, is the stored view still the same thing,
// so should we take the new frame? Out: r = take (sharper / nothing stored), g = sharpness now,
// b = stored sharpness (decayed), a = 0 outside the frame, else 1 + "this spot changed" (0..1).
const FS_FDECIDE = `#version 300 es
precision highp float;
uniform sampler2D u_src, u_acc; uniform highp sampler2D u_dec;
uniform float u_noise, u_eps, u_decay, u_lodW, u_win, u_covLod;
uniform float u_lodA[6];
${ORIENT}
${FOCUS_COMMON}
out vec4 o;
// detail energy in the 2–6 px band of the live frame (as for stacked stills), 5×5 taps 3 source px apart
float focusEnergy(vec2 s) {
  float px = 1.0 / u_dispSize.y, e = 0.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    vec2 uv = dispToUV(s + vec2(float(i), float(j)) * 3.0 * px);
    float l = lum(textureLod(u_src, uv, 0.7).rgb) - lum(textureLod(u_src, uv, 2.2).rgb);
    e += l * l;
  }
  return e / 25.0;
}
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_dec, 0)), p = accP(uv), s = accToFrame(p);
  float Sd = texelFetch(u_dec, ivec2(gl_FragCoord.xy), 0).r * u_decay;
  if (outside(dispToUV(s))) { o = vec4(0.0, 0.0, Sd, 0.0); return; }
  float e = max(focusEnergy(s) - u_noise, 0.0) * 1000.0, k = u_eps * 1000.0;   // ×1000: half-float range
  float sharper = smoothstep(1.1, 1.5, (e + k) / (Sd + k));
  float cov = textureLod(u_acc, uv, u_covLod).a;
  // Same thing as stored? Some blur of the stored view must explain what the camera sees now (a sharp and a
  // defocused view of the same thing agree after blurring; a palp or leg that moved doesn't). Compared in colour
  // (blurred stripes can match a plain background's brightness, rarely its colour), and only as much blur as the
  // drop in sharpness can account for: a view about as sharp as the stored one must match it nearly as it is.
  float lmax = 1.0 + 2.0 * log2(max((Sd + k) / (e + k), 1.0));
  vec3 sd[6]; float sd2[6];
  for (int l = 0; l < 6; l++) { sd[l] = vec3(0.0); sd2[l] = 0.0; }
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
    vec2 q = p + vec2(float(i), float(j)) * u_win;
    vec3 w = textureLod(u_src, dispToUV(accToFrame(q)), u_lodW).rgb;
    vec2 au = accUV(q);
    for (int l = 0; l < 6; l++) { vec3 d = textureLod(u_acc, au, u_lodA[l]).rgb - w; sd[l] += d; sd2[l] += dot(d, d); }
  }
  float emin = 1e9;
  for (int l = 0; l < 6; l++) {
    if (float(l) > lmax) break;
    vec3 m = sd[l] / 9.0;
    emin = min(emin, (sd2[l] / 9.0 - 0.3 * dot(m, m)) / 3.0);   // small brightness offsets (exposure drift) forgiven
  }
  float changed = smoothstep(0.025 * 0.025, 0.05 * 0.05, emin) * smoothstep(0.5, 0.9, cov);
  float take = max(sharper, 1.0 - smoothstep(0.5, 0.9, cov));
  o = vec4(take, e, Sd, 1.0 + changed);
}`;

// Pass 2 (decision grid): soften the decisions so they don't flicker pixel to pixel; "changed" spots are
// grown by ~2 cells first (decision cells are ~1.3 px at 720p scale) so a moved palp is replaced whole, edges
// included. Two decisions: STORE the new frame (clearly sharper, changed, or nothing stored) and SHOW the live
// frame (it is nearly as sharp as the best stored view, so the live pixel — current, resampled once — wins).
// Out: r = stored sharpness, g = store, b = show live.
const FS_FSMOOTH = `#version 300 es
precision highp float;
uniform highp sampler2D u_raw; uniform float u_eps;
out vec4 o;
void main() {
  ivec2 P = ivec2(gl_FragCoord.xy), mx = textureSize(u_raw, 0) - 1;
  vec4 c = texelFetch(u_raw, P, 0);
  if (c.a < 0.5) { o = vec4(c.b, 0.0, 0.0, 0.0); return; }
  float k = u_eps * 1000.0, ts = 0.0, tl = 0.0, ws = 0.0, tc = 0.0;
  for (int j = -3; j <= 3; j++) for (int i = -3; i <= 3; i++) {
    float d2 = float(i * i + j * j);
    if (d2 > 9.5) continue;
    vec4 r = texelFetch(u_raw, clamp(P + ivec2(i, j), ivec2(0), mx), 0);
    if (r.a < 0.5) continue;
    if (d2 <= 2.0) { float w = exp(-0.5 * d2); ts += w * r.r; tl += w * smoothstep(0.7, 0.9, (r.g + k) / (r.b + k)); ws += w; }
    tc = max(tc, (r.a - 1.0) * clamp(1.6 - 0.3 * sqrt(d2), 0.0, 1.0));
  }
  float take = clamp(max(ws > 0.0 ? ts / ws : 0.0, tc), 0.0, 1.0);
  float live = max(take, ws > 0.0 ? tl / ws : 0.0);
  o = vec4(mix(c.b, c.g, take), take, live, 0.0);
}`;

// Pass 3 (canvas, full resolution): stored = mix(stored, frame, take). Alpha = "something stored here".
const FS_FUPDATE = `#version 300 es
precision highp float;
uniform sampler2D u_src, u_acc, u_dec; uniform float u_down;
${ORIENT}
${FOCUS_COMMON}
out vec4 o;
vec4 cubic(sampler2D src, vec2 uv) {
  vec2 sp = uv * u_texSize, t1 = floor(sp - 0.5) + 0.5, f = sp - t1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / u_texSize, t0 = (t1 - 1.0) / u_texSize, t3 = (t1 + 2.0) / u_texSize;
  vec4 r = vec4(0.0);
  r += textureLod(src, vec2(t0.x, t0.y), 0.0) * w0.x * w0.y;   r += textureLod(src, vec2(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureLod(src, vec2(t3.x, t0.y), 0.0) * w3.x * w0.y;   r += textureLod(src, vec2(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureLod(src, vec2(t12.x, t12.y), 0.0) * w12.x * w12.y; r += textureLod(src, vec2(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureLod(src, vec2(t0.x, t3.y), 0.0) * w0.x * w3.y;   r += textureLod(src, vec2(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r += textureLod(src, vec2(t3.x, t3.y), 0.0) * w3.x * w3.y;
  return r;
}
void main() {
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_acc, 0));
  vec4 old = texelFetch(u_acc, ivec2(gl_FragCoord.xy), 0);
  vec2 su = dispToUV(accToFrame(accP(uv)));
  float take = max(texture(u_dec, uv).g, 1.0 - old.a);   // nothing stored yet: always take the frame
  if (outside(su) || take < 0.002) { o = old; return; }
  vec3 w = u_down > 0.0 ? textureLod(u_src, su, u_down).rgb : clamp(cubic(u_src, su).rgb, 0.0, 1.0);
  o = vec4(mix(old.rgb, w, take), 1.0);
}`;

// Multi-frame denoise (texture space of the source frame): neighbours are lined up along measured motion
// and blended with weights that drop to zero wherever they don't match the centre (moving legs, bad motion).
const FS_DENOISE = `#version 300 es
precision highp float;
uniform sampler2D u_c, u_n0, u_n1, u_n2, u_n3, u_f0, u_f1, u_f2, u_f3;
uniform int u_cnt; uniform float u_h; uniform vec4 u_sw;
${ORIENT}
out vec4 o;
vec2 uvToDisp(vec2 uv) {
  vec2 pc = (uv - 0.5) * u_texSize, pd;
  if (u_rot == 0) pd = pc;
  else if (u_rot == 1) pd = vec2(-pc.y, pc.x);
  else if (u_rot == 2) pd = -pc;
  else pd = vec2(pc.y, -pc.x);
  return pd / u_dispSize.y;
}
vec4 cubic(sampler2D src, vec2 uv) {
  vec2 sp = uv * u_texSize, t1 = floor(sp - 0.5) + 0.5, f = sp - t1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / u_texSize, t0 = (t1 - 1.0) / u_texSize, t3 = (t1 + 2.0) / u_texSize;
  vec4 r = vec4(0.0);
  r += textureLod(src, vec2(t0.x, t0.y), 0.0) * w0.x * w0.y;   r += textureLod(src, vec2(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureLod(src, vec2(t3.x, t0.y), 0.0) * w3.x * w0.y;   r += textureLod(src, vec2(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureLod(src, vec2(t12.x, t12.y), 0.0) * w12.x * w12.y; r += textureLod(src, vec2(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureLod(src, vec2(t0.x, t3.y), 0.0) * w0.x * w3.y;   r += textureLod(src, vec2(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r += textureLod(src, vec2(t3.x, t3.y), 0.0) * w3.x * w3.y;
  return r;
}
float patchLuma(sampler2D s, vec2 uv) {
  vec2 p = 1.25 / u_texSize; const vec3 k = vec3(1.0 / 3.0);
  return (dot(textureLod(s, uv, 0.0).rgb, k) + dot(textureLod(s, uv + vec2(p.x, 0.0), 0.0).rgb, k) + dot(textureLod(s, uv - vec2(p.x, 0.0), 0.0).rgb, k)
        + dot(textureLod(s, uv + vec2(0.0, p.y), 0.0).rgb, k) + dot(textureLod(s, uv - vec2(0.0, p.y), 0.0).rgb, k)) * 0.2;
}
vec4 neighbour(sampler2D n, sampler2D f, vec2 s, float lc, float sw) {
  float A = u_dispSize.x / u_dispSize.y;
  vec2 F = texture(f, vec2(s.x / A + 0.5, s.y + 0.5)).rg;
  vec2 uv = dispToUV(s + F);
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec4(0.0);
  float d = abs(patchLuma(n, uv) - lc) / u_h;
  float w = exp(-d * d) * sw;
  return vec4(clamp(cubic(n, uv).rgb, 0.0, 1.0) * w, w);
}
void main() {
  vec2 uv = gl_FragCoord.xy / u_texSize, s = uvToDisp(uv);
  vec3 c = textureLod(u_c, uv, 0.0).rgb;
  float lc = patchLuma(u_c, uv);
  vec4 acc = vec4(c, 1.0);
  if (u_cnt > 0) acc += neighbour(u_n0, u_f0, s, lc, u_sw.x);
  if (u_cnt > 1) acc += neighbour(u_n1, u_f1, s, lc, u_sw.y);
  if (u_cnt > 2) acc += neighbour(u_n2, u_f2, s, lc, u_sw.z);
  if (u_cnt > 3) acc += neighbour(u_n3, u_f3, s, lc, u_sw.w);
  o = vec4(acc.rgb / acc.a, 1.0);
}`;

// Focus stacking (texture space of the centre frame), one pass per frame, ping-pong accumulation:
// each frame is lined up on the centre frame along measured motion and added with weight
// (local sharpness)² × (agrees with the centre at coarse scale), so the sharpest frame wins each region
// while anything that moved (legs) only comes from frames where it matches the centre.
const FS_STACK = `#version 300 es
precision highp float;
uniform sampler2D u_frame, u_flow, u_cgray, u_centre; uniform highp sampler2D u_prev, u_emax;
uniform float u_eps, u_tol, u_lod, u_noise, u_pow, u_bias; uniform int u_first, u_mode;
${ORIENT}
out vec4 o;
vec2 uvToDisp(vec2 uv) {
  vec2 pc = (uv - 0.5) * u_texSize, pd;
  if (u_rot == 0) pd = pc; else if (u_rot == 1) pd = vec2(-pc.y, pc.x); else if (u_rot == 2) pd = -pc; else pd = vec2(pc.y, -pc.x);
  return pd / u_dispSize.y;
}
vec4 cubic(sampler2D src, vec2 uv) {
  vec2 sp = uv * u_texSize, t1 = floor(sp - 0.5) + 0.5, f = sp - t1;
  vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f)), w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f)), w3 = f * f * (-0.5 + 0.5 * f);
  vec2 w12 = w1 + w2, t12 = (t1 + w2 / w12) / u_texSize, t0 = (t1 - 1.0) / u_texSize, t3 = (t1 + 2.0) / u_texSize;
  vec4 r = vec4(0.0);
  r += textureLod(src, vec2(t0.x, t0.y), 0.0) * w0.x * w0.y;   r += textureLod(src, vec2(t12.x, t0.y), 0.0) * w12.x * w0.y;
  r += textureLod(src, vec2(t3.x, t0.y), 0.0) * w3.x * w0.y;   r += textureLod(src, vec2(t0.x, t12.y), 0.0) * w0.x * w12.y;
  r += textureLod(src, vec2(t12.x, t12.y), 0.0) * w12.x * w12.y; r += textureLod(src, vec2(t3.x, t12.y), 0.0) * w3.x * w12.y;
  r += textureLod(src, vec2(t0.x, t3.y), 0.0) * w0.x * w3.y;   r += textureLod(src, vec2(t12.x, t3.y), 0.0) * w12.x * w3.y;
  r += textureLod(src, vec2(t3.x, t3.y), 0.0) * w3.x * w3.y;
  return r;
}
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
// detail energy in the 2–6 px band (difference of two blur levels), averaged over a ~13×13 neighbourhood.
// Pixel-scale "detail" is left out on purpose: in phone video it's mostly compression artefacts and grain,
// which would otherwise win the "sharpest view" contest.
float focusEnergy(sampler2D t, vec2 uv) {
  vec2 p = 1.0 / u_texSize; float e = 0.0;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    vec2 q = uv + vec2(float(i), float(j)) * 3.0 * p;
    float l = lum(textureLod(t, q, 0.7).rgb) - lum(textureLod(t, q, 2.2).rgb);
    e += l * l;
  }
  return e / 25.0;
}
void main() {
  ivec2 P = ivec2(gl_FragCoord.xy);
  vec4 prev = u_first == 1 ? vec4(0.0) : texelFetch(u_prev, P, 0);
  vec2 uv = gl_FragCoord.xy / u_texSize, s = uvToDisp(uv);
  float A = u_dispSize.x / u_dispSize.y;
  vec2 g = vec2(s.x / A + 0.5, s.y + 0.5);
  vec2 un = dispToUV(s + texture(u_flow, g).rg);
  if (any(lessThan(un, vec2(0.0))) || any(greaterThan(un, vec2(1.0)))) { o = prev; return; }
  // compare with the centre frame at an 8 px scale: a blurry and a sharp view of the same thing agree there,
  // a leg that moved a few pixels doesn't
  float d = (lum(textureLod(u_frame, un, 3.0).rgb) - lum(textureLod(u_centre, uv, 3.0).rgb)) / u_tol;
  // …and the same structure in the same place: correlation of the 4 px-scale pattern over ~25 px. Blur lowers
  // contrast but keeps the correlation; a leg that's shifted or a different leg kills it.
  float sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0; vec2 p = 1.0 / u_texSize;
  for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++) {
    vec2 o2 = vec2(float(i), float(j)) * 5.0 * p;
    float a = lum(textureLod(u_frame, un + o2, 2.0).rgb), b = lum(textureLod(u_centre, uv + o2, 2.0).rgb);
    sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
  }
  float va = saa / 25.0 - sa * sa / 625.0, vb = sbb / 25.0 - sb * sb / 625.0, cv = sab / 25.0 - sa * sb / 625.0;
  float ncc = cv / sqrt(max(va * vb, 1e-12));
  float ok = exp(-d * d) * (max(va, vb) < 4e-5 ? 1.0 : smoothstep(0.7, 0.9, ncc));
  // sharpness above what grain alone would produce
  float e = max(focusEnergy(u_frame, un) - u_noise, 0.0) * 1000.0;   // ×1000 keeps it in half-float's comfortable range
  if (u_mode == 0) { o = vec4(max(prev.r, ok > 0.5 ? e : 0.0), 0.0, 0.0, 1.0); return; }   // pass 1: sharpest agreeing view
  // pass 2: weight relative to the sharpest view of this spot (scale-free, can't saturate);
  // the floor stops grain-level differences in flat areas from picking a "winner"
  float emax = texelFetch(u_emax, P, 0).r, k = u_eps * 1000.0;
  float r = min((e + k) / (emax + k), 1.0);
  float w;
  if (u_bias > 1.5) w = 1.0;   // the centre frame itself
  else {
    // do no harm: another view only counts where it is clearly sharper than the centre frame here
    float ec = max(focusEnergy(u_centre, uv) - u_noise, 0.0) * 1000.0;
    float gain = (e + k) / (ec + k);
    w = pow(r, u_pow) * ok * smoothstep(1.2, 1.8, gain) * 8.0;
  }
  w += 1e-4;
  vec3 col = clamp(cubic(u_frame, un).rgb, 0.0, 1.0);
  o = prev + vec4(col * w, w);
}`;
const FS_STACKNORM = `#version 300 es
precision highp float;
uniform highp sampler2D u_acc;
out vec4 o;
void main() { vec4 a = texelFetch(u_acc, ivec2(gl_FragCoord.xy), 0); o = vec4(a.rgb / max(a.a, 1e-6), 1.0); }`;

// Finished frame → the encoder's own format (I420, BT.709 limited range), packed 4 bytes per RGBA texel so one
// readback returns the three planes laid out exactly as VideoFrame wants them. Rows [0, H): Y. Then U, then V, each
// ceil(H/4) rows holding two chroma rows (stride S/2) per row. Chroma = 2×2 average (as the browser's own converter).
// About 1/3 of the bytes of reading back RGBA, no colour conversion on the CPU, and the readback doesn't stall.
const FS_YUV = `#version 300 es
precision highp float;
uniform sampler2D u_rgb; uniform int u_W, u_H, u_S, u_Q;
out vec4 o;
const vec3 KY = vec3(0.2126, 0.7152, 0.0722);
vec3 px(int x, int y) { return texelFetch(u_rgb, ivec2(min(x, u_W - 1), u_H - 1 - min(y, u_H - 1)), 0).rgb; }
float Yb(vec3 c) { return (16.0 + 219.0 * dot(c, KY)) / 255.0; }
vec2 UVb(int cx, int cy) {
  vec3 c = 0.25 * (px(2 * cx, 2 * cy) + px(2 * cx + 1, 2 * cy) + px(2 * cx, 2 * cy + 1) + px(2 * cx + 1, 2 * cy + 1));
  float y = dot(c, KY);
  return (128.0 + 224.0 * vec2((c.b - y) / 1.8556, (c.r - y) / 1.5748)) / 255.0;
}
void main() {
  int tx = int(gl_FragCoord.x), ty = int(gl_FragCoord.y);
  if (ty < u_H) { int x = tx * 4; o = vec4(Yb(px(x, ty)), Yb(px(x + 1, ty)), Yb(px(x + 2, ty)), Yb(px(x + 3, ty))); return; }
  int r = ty - u_H, plane = r < u_Q ? 0 : 1; r -= plane * u_Q;
  int hw = u_S / 8, crow = 2 * r + (tx >= hw ? 1 : 0), cx0 = (tx % hw) * 4;
  if (crow >= (u_H + 1) / 2) { o = vec4(0.5); return; }
  vec4 v;
  for (int i = 0; i < 4; i++) { vec2 uv = UVb(cx0 + i, crow); v[i] = plane == 0 ? uv.x : uv.y; }
  o = v;
}`;

const FS_BLUR = `#version 300 es
precision highp float;
uniform sampler2D u_tex; uniform vec2 u_dir; uniform float u_sigma; uniform int u_rad;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy), sz = textureSize(u_tex, 0) - 1;
  vec3 acc = texelFetch(u_tex, p, 0).rgb; float ws = 1.0;
  for (int i = 1; i <= 24; i++) {
    if (i > u_rad) break;
    float w = exp(-float(i * i) / (2.0 * u_sigma * u_sigma));
    ivec2 d = ivec2(u_dir) * i;
    acc += w * (texelFetch(u_tex, clamp(p + d, ivec2(0), sz), 0).rgb + texelFetch(u_tex, clamp(p - d, ivec2(0), sz), 0).rgb);
    ws += 2.0 * w;
  }
  o = vec4(acc / ws, 1.0);
}`;

const FS_FINAL = `#version 300 es
precision highp float;
uniform sampler2D u_a; uniform sampler2D u_b; uniform int u_useBlur; uniform int u_raw;
uniform float u_cas, u_detail, u_denoise;
uniform float u_exposure, u_contrast, u_shadows, u_highlights, u_sat, u_vib, u_warm, u_tint;
out vec4 o;
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 tx(ivec2 p){ return texelFetch(u_a, clamp(p, ivec2(0), textureSize(u_a, 0) - 1), 0).rgb; }
vec3 toLin(vec3 c){ return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 toSrgb(vec3 c){ c = max(c, 0.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
float cbrt(float x){ return sign(x) * pow(abs(x), 1.0 / 3.0); }
vec3 lin2lab(vec3 c){
  float l = cbrt(0.4122214708*c.r + 0.5363325363*c.g + 0.0514459929*c.b);
  float m = cbrt(0.2119034982*c.r + 0.6806995451*c.g + 0.1073969566*c.b);
  float s = cbrt(0.0883024619*c.r + 0.2817188376*c.g + 0.6299787005*c.b);
  return vec3(0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
              1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
              0.0259040371*l + 0.7827717662*m - 0.8086757660*s);
}
vec3 lab2lin(vec3 c){
  float l = c.x + 0.3963377774*c.y + 0.2158037573*c.z;
  float m = c.x - 0.1055613458*c.y - 0.0638541728*c.z;
  float s = c.x - 0.0894841775*c.y - 1.2914855480*c.z;
  l = l*l*l; m = m*m*m; s = s*s*s;
  return vec3( 4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
              -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
              -0.0041960863*l - 0.7034186147*m + 1.7076147010*s);
}
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 e = tx(p);
  if (u_raw == 1) { o = vec4(e, 1.0); return; }
  vec3 a = tx(p + ivec2(-1,-1)), b = tx(p + ivec2(0,-1)), c = tx(p + ivec2(1,-1)), d = tx(p + ivec2(-1,0));
  vec3 f = tx(p + ivec2(1,0)), g = tx(p + ivec2(-1,1)), h = tx(p + ivec2(0,1)), i = tx(p + ivec2(1,1));
  vec3 mnC = min(min(min(d, e), min(f, b)), h), mxC = max(max(max(d, e), max(f, b)), h);
  vec3 mn9 = min(mnC, min(min(a, c), min(g, i))), mx9 = max(mxC, max(max(a, c), max(g, i)));
  float focus = smoothstep(0.03, 0.14, luma(mx9) - luma(mn9));
  vec3 blur = u_useBlur == 1 ? texelFetch(u_b, p, 0).rgb : e;
  vec3 col = mix(e, blur, u_denoise * (1.0 - focus));
  vec3 sh = col + (e - blur) * u_detail * (0.35 + 0.65 * focus);
  if (u_cas > 0.0) {
    vec3 mn = mnC + mn9, mx = mxC + mx9;
    vec3 amp = sqrt(clamp(min(mn, 2.0 - mx) / max(mx, vec3(1e-4)), 0.0, 1.0));
    vec3 w = amp * (-1.0 / mix(8.0, 4.5, u_cas));
    vec3 cas = (b*w + d*w + f*w + h*w + e) / (1.0 + 4.0 * w);
    sh += (cas - e) * min(1.0, u_cas * 2.5);
  }
  col = clamp(sh, mn9 - 0.03, mx9 + 0.03);   // halo limiter

  vec3 lin = toLin(clamp(col, 0.0, 1.0)) * exp2(u_exposure);
  lin *= vec3(1.0 + 0.15 * u_warm, 1.0 - 0.10 * u_tint, 1.0 - 0.15 * u_warm);
  vec3 lab = lin2lab(max(lin, 0.0));
  float L = lab.x;
  L += u_shadows * 0.16 * (1.0 - smoothstep(0.05, 0.6, L));
  L -= u_highlights * 0.16 * smoothstep(0.55, 1.0, L);
  L = 0.55 + (L - 0.55) * (1.0 + u_contrast);
  float C = length(lab.yz);
  float k = u_sat * (1.0 + u_vib * (1.0 - smoothstep(0.0, 0.18, C)));
  lab = vec3(max(L, 0.0), lab.yz * k);
  o = vec4(clamp(toSrgb(lab2lin(lab)), 0.0, 1.0), 1.0);
}`;

function compile(gl, vs, fs) {
  const mk = (t, s) => {
    const sh = gl.createShader(t); gl.shaderSource(sh, s); gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error('Shader: ' + gl.getShaderInfoLog(sh));
    return sh;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vs)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('Link: ' + gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const nm = gl.getActiveUniform(p, i).name; u[nm] = gl.getUniformLocation(p, nm); }
  return { p, u };
}

function makeTex(gl, w, h) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.NEAREST], [gl.TEXTURE_MAG_FILTER, gl.NEAREST], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  const fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
  return { t, fb, w, h };
}

// Let the page breathe without a timer (timers are throttled to 1/s when the page isn't in front).
let _yc = null; const _yq = [];
export function yieldNow() {
  if (!_yc && typeof MessageChannel !== 'undefined' && typeof document !== 'undefined') { _yc = new MessageChannel(); _yc.port1.onmessage = () => { const r = _yq.shift(); if (r) r(); }; }
  return _yc ? new Promise(r => { _yq.push(r); _yc.port2.postMessage(0); }) : new Promise(r => setTimeout(r, 0));
}

class Base {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 unavailable');
    this.gl = gl;
    this.vao = gl.createVertexArray();
    const mk = () => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      return t;
    };
    this.src = mk(); this.srcB = mk(); this.flowTex = mk();
    gl.bindTexture(gl.TEXTURE_2D, this.flowTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, 1, 1, 0, gl.RG, gl.FLOAT, new Float32Array(2));
    this.srcInfo = null;
  }
  // Free the GPU context now instead of waiting for GC (phones cap live WebGL contexts and
  // kill the oldest one, which would be the preview).
  dispose() {
    try { const ext = this.gl.getExtension('WEBGL_lose_context'); if (ext) ext.loseContext(); } catch (e) { /* ignore */ }
  }
  // info: { texW, texH, dispW, dispH, rot (0..3 quarter turns CW) }
  // Second frame (for in-between frames). Afterwards swapFrames() makes it the current one.
  uploadB(source, info, mip = false) {
    const keep = this.src; this.src = this.srcB;
    this.upload(source, info, mip);
    this.srcB = this.src; this.src = keep;
  }
  swapFrames() { const t = this.src; this.src = this.srcB; this.srcB = t; }
  // ── multi-frame denoise support ──
  makeTex() {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_MIN_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  }
  uploadTex(tex, source) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  makeTarget(w, h) {
    const gl = this.gl, t = this.makeTex();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return { t, fb, w, h };
  }
  setFlowTex(tex, gw, gh, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, gw, gh, 0, gl.RG, gl.FLOAT, data);
  }
  // Blend up to 4 motion-aligned neighbours into the centre frame; result goes to target (texture space).
  denoise(centre, neigh, info, target, h = 0.03, mip = false) {
    const gl = this.gl;
    if (!this.dn) this.dn = compile(gl, VS, FS_DENOISE);
    const u = this.dn.u;
    gl.useProgram(this.dn.p);
    const bind = (unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(u[name], unit); };
    bind(0, centre, 'u_c');
    for (let j = 0; j < 4; j++) {
      const nb = neigh[j] || neigh[0];
      bind(1 + j, nb ? nb.tex : centre, `u_n${j}`);
      bind(5 + j, nb ? nb.flowTex : this.flowTex, `u_f${j}`);
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(u.u_cnt, neigh.length); gl.uniform1f(u.u_h, h);
    const sw = [0, 1, 2, 3].map(j => (neigh[j] && neigh[j].weight != null ? neigh[j].weight : 1));
    gl.uniform4f(u.u_sw, sw[0], sw[1], sw[2], sw[3]);
    this.setOrient(this.dn, info);
    this.draw(target.fb, target.w, target.h);
    gl.bindTexture(gl.TEXTURE_2D, target.t);
    if (mip) { gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
    else gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  // ── focus stacking ──
  stackBegin(w, h) {
    const gl = this.gl;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('This phone\'s graphics chip can\'t do stacking (no float render targets).');
    if (!this.stk) { this.stk = compile(gl, VS, FS_STACK); this.stkN = compile(gl, VS, FS_STACKNORM); }
    const mk = () => {
      const t = this.makeTex(); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, w, h);
      const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      return { t, fb, w, h };
    };
    this.acc = [mk(), mk()]; this.accI = 0; this.accFirst = true; this.accMk = mk;
  }
  // after pass 1 (mode 0): keep the max-sharpness map aside and start a fresh accumulation for pass 2
  stackCentre(tex) { this.centreTex = tex; }
  stackSecondPass() {
    this.emax = this.acc[this.accI]; const other = this.acc[1 - this.accI];
    const gl = this.gl; gl.deleteTexture(other.t); gl.deleteFramebuffer(other.fb);
    this.acc = [this.accMk(), this.accMk()]; this.accI = 0; this.accFirst = true;
  }
  setGrayTex(tex, w, h, bytes) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex); gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
    gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  }
  stackAdd(frameTex, flowTex, grayTex, info, { eps, tol, lod, noise = 0, pow = 2, mode = 1, bias = 1 }) {
    const gl = this.gl, u = this.stk.u, src = this.acc[this.accI], dst = this.acc[1 - this.accI];
    gl.useProgram(this.stk.p);
    const bind = (unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(u[name], unit); };
    bind(0, frameTex, 'u_frame'); bind(1, flowTex, 'u_flow'); bind(2, grayTex, 'u_cgray'); bind(3, src.t, 'u_prev');
    bind(4, this.emax ? this.emax.t : src.t, 'u_emax'); bind(5, this.centreTex, 'u_centre');
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(u.u_mode, mode); gl.uniform1f(u.u_bias, bias);
    gl.uniform1f(u.u_eps, eps); gl.uniform1f(u.u_tol, tol); gl.uniform1f(u.u_noise, noise); gl.uniform1f(u.u_pow, pow); gl.uniform1f(u.u_lod, lod); gl.uniform1i(u.u_first, this.accFirst ? 1 : 0);
    this.setOrient(this.stk, info);
    this.draw(dst.fb, dst.w, dst.h);
    this.accI = 1 - this.accI; this.accFirst = false;
  }
  stackFinish(target, mip) {
    const gl = this.gl;
    gl.useProgram(this.stkN.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.acc[this.accI].t); gl.uniform1i(this.stkN.u.u_acc, 0);
    this.draw(target.fb, target.w, target.h);
    gl.bindTexture(gl.TEXTURE_2D, target.t);
    if (mip) { gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
    for (const a of [...this.acc, ...(this.emax ? [this.emax] : [])]) { gl.deleteTexture(a.t); gl.deleteFramebuffer(a.fb); }
    this.acc = null; this.emax = null;
  }
  mipTex(tex) { const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, tex); gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
  // ── asynchronous readback: GPU → pixel-pack buffer, fenced; collected once the GPU is done (no stall) ──
  makePackBuffer(bytes) {
    const gl = this.gl, b = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, b); gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ); gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    return { b, bytes };
  }
  // read the w×h RGBA8 framebuffer fb into pack buffer pb, return a fence for it
  readAsync(fb, w, h, pb) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pb.b);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    const f = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    return f;
  }
  fenceDone(f) {
    const gl = this.gl;
    if (!f || gl.isContextLost()) return true;
    const st = gl.clientWaitSync(f, 0, 0);
    return st === gl.ALREADY_SIGNALED || st === gl.CONDITION_SATISFIED || st === gl.WAIT_FAILED;
  }
  // Resolves once fence f has passed. Polls without blocking the page: a few immediate re-checks (the browser
  // updates fence status between tasks), then short sleeps.
  async waitFence(f, maxMs = 30000) {
    if (!f) return;
    const t0 = performance.now();
    for (let n = 0; !this.fenceDone(f); n++) {
      if (performance.now() - t0 > maxMs) break;
      await (n < 4 ? yieldNow() : new Promise(r => setTimeout(r, n < 40 ? 1 : 4)));
    }
    this.gl.deleteSync(f);
  }
  // copy a finished pack buffer into dst (Uint8Array)
  collect(pb, dst) {
    const gl = this.gl;
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pb.b);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, dst, 0, pb.bytes);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    return dst;
  }
  // Small greyscale copy of texture tex for motion measuring, exactly as Downscaler.grabBytes makes it (same shader,
  // same mipmapped sampling), but from the frame already on this GPU context and read back without waiting.
  // Returns a fence; the w×h RGBA rows land in pack buffer pb (red = luma).
  analysisAsync(tex, info, w, h, pb, hasMips = false) {
    const gl = this.gl;
    if (!this.anP) this.anP = compile(gl, VS, FS_ANALYSIS);
    if (!this.anT || this.anT.w !== w || this.anT.h !== h) { if (this.anT) { gl.deleteTexture(this.anT.t); gl.deleteFramebuffer(this.anT.fb); } this.anT = makeTex(gl, w, h); }
    const mip = info.dispH / h > 2.5;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
    if (mip) { if (!hasMips) gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
    else gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.useProgram(this.anP.p);
    gl.uniform1i(this.anP.u.u_src, 0);
    gl.uniform2f(this.anP.u.u_out, w, h);
    this.setOrient(this.anP, info);
    this.draw(this.anT.fb, w, h);
    // the render path keeps its own sampling (hasMips = the frame is drawn from its mipmaps)
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, hasMips ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    return this.readAsync(this.anT.fb, w, h, pb);
  }
  // Resolves when the GPU has finished everything queued so far (fence, polled without blocking the page).
  async gpuDone(maxMs = 30000) {
    const gl = this.gl, f = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!f) return;
    gl.flush();
    const t0 = performance.now();
    while (gl.getSyncParameter(f, gl.SYNC_STATUS) !== gl.SIGNALED && performance.now() - t0 < maxMs && !gl.isContextLost()) await new Promise(r => setTimeout(r, 4));
    gl.deleteSync(f);
  }
  // ── focus build-up (see js/focus.js) ──
  // Canvas: RGBA8 + mipmaps (rgb = stored picture, a = something stored). Decision maps: RGBA16F.
  focusBegin(geo) {
    const gl = this.gl;
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('This phone\'s graphics chip can\'t do focus build-up (no float render targets).');
    if (!this.fdP) { this.fdP = compile(gl, VS, FS_FDECIDE); this.fsP = compile(gl, VS, FS_FSMOOTH); this.fuP = compile(gl, VS, FS_FUPDATE); }
    if (this.facc) this.focusEnd();
    const levels = 1 + Math.floor(Math.log2(Math.max(geo.AW, geo.AH)));
    const fbo = (t, w, h) => { const fb = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0); return { t, fb, w, h }; };
    this.fMkAcc = () => {
      const t = this.makeTex(); gl.texStorage2D(gl.TEXTURE_2D, levels, gl.RGBA8, geo.AW, geo.AH);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      return fbo(t, geo.AW, geo.AH);
    };
    this.fMkDec = () => { const t = this.makeTex(); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA16F, geo.DW, geo.DH); return fbo(t, geo.DW, geo.DH); };
    this.fgeo = geo;
    this.facc = [this.fMkAcc(), this.fMkAcc()]; this.fdec = [this.fMkDec(), this.fMkDec()]; this.fraw = this.fMkDec(); this.fi = 0;
    this.focusClear();
  }
  focusClear() {
    const gl = this.gl;
    gl.clearColor(0, 0, 0, 0);
    for (const x of [...this.facc, ...this.fdec]) { gl.bindFramebuffer(gl.FRAMEBUFFER, x.fb); gl.viewport(0, 0, x.w, x.h); gl.clear(gl.COLOR_BUFFER_BIT); }
    for (const x of this.facc) { gl.bindTexture(gl.TEXTURE_2D, x.t); gl.generateMipmap(gl.TEXTURE_2D); }
  }
  _fFree(list) { const gl = this.gl; for (const x of list) if (x) { gl.deleteTexture(x.t); gl.deleteFramebuffer(x.fb); } }
  // hand the current canvas + decision map over (kept for a second pass); fresh ones take their place
  focusDetach() { const s = { acc: this.facc[this.fi], dec: this.fdec[this.fi] }; this.facc[this.fi] = this.fMkAcc(); this.fdec[this.fi] = this.fMkDec(); return s; }
  focusAttach(s) { this._fFree([this.facc[this.fi], this.fdec[this.fi]]); this.facc[this.fi] = s.acc; this.fdec[this.fi] = s.dec; }
  focusDrop(s) { this._fFree([s.acc, s.dec]); }
  focusEnd() { if (this.facc) this._fFree([...this.facc, ...this.fdec, this.fraw]); this.facc = this.fdec = this.fraw = null; }
  // One frame: decide (decision grid) → soften → update the canvas → mipmaps. srcTex must have mipmaps.
  // T: canvas-normalised → frame display-normalised similarity [a,b,tx,ty].
  focusStep(srcTex, info, T, prm) {
    const gl = this.gl, geo = this.fgeo, i0 = this.fi, i1 = 1 - this.fi;
    const A0 = this.facc[i0], A1 = this.facc[i1], D0 = this.fdec[i0], D1 = this.fdec[i1];
    const bindTo = (prog, unit, tex, name) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(prog.u[name], unit); };
    const geom = prog => {
      gl.uniformMatrix2fv(prog.u.u_T, false, [T[0], T[1], -T[1], T[0]]); gl.uniform2f(prog.u.u_Tt, T[2], T[3]);
      gl.uniform2f(prog.u.u_C, geo.CW, geo.CH); this.setOrient(prog, info);
    };
    let P = this.fdP; gl.useProgram(P.p);
    bindTo(P, 0, srcTex, 'u_src'); bindTo(P, 1, A0.t, 'u_acc'); bindTo(P, 2, D0.t, 'u_dec');
    geom(P);
    gl.uniform1f(P.u.u_noise, prm.noise); gl.uniform1f(P.u.u_eps, prm.eps); gl.uniform1f(P.u.u_decay, prm.decay);
    gl.uniform1f(P.u.u_lodW, prm.lodW); gl.uniform1f(P.u.u_win, prm.win); gl.uniform1f(P.u.u_covLod, prm.covLod);
    gl.uniform1fv(P.u['u_lodA[0]'], prm.lodA);
    this.draw(this.fraw.fb, geo.DW, geo.DH);
    P = this.fsP; gl.useProgram(P.p);
    bindTo(P, 0, this.fraw.t, 'u_raw'); gl.uniform1f(P.u.u_eps, prm.eps);
    this.draw(D1.fb, geo.DW, geo.DH);
    P = this.fuP; gl.useProgram(P.p);
    bindTo(P, 0, srcTex, 'u_src'); bindTo(P, 1, A0.t, 'u_acc'); bindTo(P, 2, D1.t, 'u_dec');
    geom(P); gl.uniform1f(P.u.u_down, prm.down);
    this.draw(A1.fb, geo.AW, geo.AH);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, A1.t); gl.generateMipmap(gl.TEXTURE_2D);
    this.fi = i1;
  }
  useSrc(tex, info, mip) { this.src = tex; this.srcInfo = info; this.mip = mip; }
  useSrcB(tex) { this.srcB = tex; }
  // Motion field from frame A to frame B: grid of [dx, dy] in display-normalised units (height = 1).
  setFlow(gw, gh, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.flowTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, gw, gh, 0, gl.RG, gl.FLOAT, data);
  }
  upload(source, info, mip = false) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    if (mip) { gl.generateMipmap(gl.TEXTURE_2D); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); }
    else gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    this.srcInfo = info; this.mip = mip;
  }
  setOrient(prog, info) {
    const gl = this.gl, u = prog.u;
    gl.uniform2f(u.u_dispSize, info.dispW, info.dispH);
    gl.uniform2f(u.u_texSize, info.texW, info.texH);
    gl.uniform1i(u.u_rot, info.rot);
  }
  draw(fb, w, h) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}

// Downscales a frame to a small luma image (Float32, row 0 = top), display-oriented.
export class Downscaler extends Base {
  constructor(w, h) {
    const c = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
    super(c);
    this.w = w; this.h = h;
    this.prog = compile(this.gl, VS, FS_ANALYSIS);
    this.target = makeTex(this.gl, w, h);
    this.px = new Uint8Array(w * h * 4);
  }
  grab(source, info, out) {
    const gl = this.gl;
    this.upload(source, info, info.dispH / this.h > 2.5);
    gl.useProgram(this.prog.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.uniform1i(this.prog.u.u_src, 0);
    gl.uniform2f(this.prog.u.u_out, this.w, this.h);
    this.setOrient(this.prog, info);
    this.draw(this.target.fb, this.w, this.h);
    gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.UNSIGNED_BYTE, this.px);
    const g = out || new Float32Array(this.w * this.h);
    for (let i = 0, j = 0; j < g.length; i += 4, j++) g[j] = this.px[i];
    return g;
  }
  // Same as grab() but the result is read back without waiting for the GPU: draws now (the frame can be closed
  // straight away), returns a fence; once it has passed, collect(pb, rgba) gives the RGBA rows (red = luma).
  grabAsync(source, info, pb) {
    const gl = this.gl;
    this.upload(source, info, info.dispH / this.h > 2.5);
    gl.useProgram(this.prog.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.uniform1i(this.prog.u.u_src, 0);
    gl.uniform2f(this.prog.u.u_out, this.w, this.h);
    this.setOrient(this.prog, info);
    this.draw(this.target.fb, this.w, this.h);
    return this.readAsync(this.target.fb, this.w, this.h, pb);
  }
  // Same as grab() but into a Uint8Array (compact, transferable to a worker).
  grabBytes(source, info, out) {
    const gl = this.gl;
    this.upload(source, info, info.dispH / this.h > 2.5);
    gl.useProgram(this.prog.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.uniform1i(this.prog.u.u_src, 0);
    gl.uniform2f(this.prog.u.u_out, this.w, this.h);
    this.setOrient(this.prog, info);
    this.draw(this.target.fb, this.w, this.h);
    gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.UNSIGNED_BYTE, this.px);
    for (let i = 0, j = 0; j < out.length; i += 4, j++) out[j] = this.px[i];
    return out;
  }
}

export const DEFAULT_LOOK = Object.freeze({
  cas: 0.35, detail: 0.6, radius: 1.6, denoise: 0.15,
  exposure: 0, contrast: 0.05, shadows: 0.15, highlights: 0.2, saturation: 0.1, vibrance: 0.25, warmth: 0, tint: 0,
});

export class Renderer extends Base {
  constructor(canvas) {
    super(canvas);
    const gl = this.gl;
    this.warp = compile(gl, VS, FS_WARP);
    this.blur = compile(gl, VS, FS_BLUR);
    this.dn = null;   // denoise program, compiled on first use
    this.fin = compile(gl, VS, FS_FINAL);
    this.size = [0, 0];
    this.outFB = null;   // null = draw finished frames onto the canvas
  }
  setSize(w, h) {
    if (this.size[0] === w && this.size[1] === h) return;
    const gl = this.gl;
    this.canvas.width = w; this.canvas.height = h;
    for (const t of [this.A, this.B, this.Cc]) if (t) { gl.deleteTexture(t.t); gl.deleteFramebuffer(t.fb); }
    this.A = makeTex(gl, w, h); this.B = makeTex(gl, w, h); this.Cc = makeTex(gl, w, h);
    this.size = [w, h];
    if (this.yuv) this.useYuvOutput(true);
  }
  // Export: draw finished frames into a texture (not the canvas) and hand them over as I420 (see FS_YUV).
  // Returns the packed layout { S (luma stride), Q (rows per chroma plane), TW×TH (pack texture), bytes }.
  useYuvOutput(on) {
    const gl = this.gl;
    for (const t of [this.O, this.P]) if (t) { gl.deleteTexture(t.t); gl.deleteFramebuffer(t.fb); }
    this.O = this.P = null; this.outFB = null; this.yuv = null;
    if (!on) return null;
    const [w, h] = this.size;
    if (!this.yuvP) this.yuvP = compile(gl, VS, FS_YUV);
    const S = Math.ceil(w / 8) * 8, Q = Math.ceil(h / 4), TW = S / 4, TH = h + 2 * Q;
    this.O = makeTex(gl, w, h); this.P = makeTex(gl, TW, TH); this.outFB = this.O.fb;
    this.yuv = { S, Q, TW, TH, bytes: S * TH, W: w, H: h };
    return this.yuv;
  }
  // pack the last rendered frame and start reading it into pack buffer pb; returns a fence
  readYuvAsync(pb) {
    const gl = this.gl, y = this.yuv, P = this.yuvP;
    gl.useProgram(P.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.O.t); gl.uniform1i(P.u.u_rgb, 0);
    gl.uniform1i(P.u.u_W, y.W); gl.uniform1i(P.u.u_H, y.H); gl.uniform1i(P.u.u_S, y.S); gl.uniform1i(P.u.u_Q, y.Q);
    this.draw(this.P.fb, y.TW, y.TH);
    return this.readAsync(this.P.fb, y.TW, y.TH, pb);
  }
  // Small RGB thumbnail of the current source (whole frame, no look), for calibration probes.
  thumb() {
    const gl = this.gl, [w, h] = this.size;
    this.render([1, 0, 0, 0], null, { raw: true });
    const px = new Uint8Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  }
  // M: [a,b,tx,ty] output-norm → source display-norm. look: params. opts: { raw, pxScale }
  render(M, look, opts = {}) {
    const gl = this.gl, [w, h] = this.size, info = this.srcInfo;
    if (!info || !w) return;
    const L = look || DEFAULT_LOOK;
    // warp
    const scale = Math.hypot(M[0], M[1]) * info.dispH / h; // source px per output px
    gl.useProgram(this.warp.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.uniform1i(this.warp.u.u_src, 0);
    gl.uniform2f(this.warp.u.u_out, w, h);
    gl.uniformMatrix2fv(this.warp.u.u_L, false, [M[0], M[1], -M[1], M[0]]);
    gl.uniform2f(this.warp.u.u_t, M[2], M[3]);
    gl.uniform1i(this.warp.u.u_cubic, !this.mip && scale < 1.25 ? 1 : 0);
    gl.uniform1i(this.warp.u.u_hdr, info.hdr || 0);
    const it = opts.interp;
    gl.uniform1i(this.warp.u.u_interp, it != null ? 1 : 0);
    if (it != null) {
      gl.uniform1f(this.warp.u.u_it, it);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.srcB); gl.uniform1i(this.warp.u.u_srcB, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.flowTex); gl.uniform1i(this.warp.u.u_flow, 2);
      gl.activeTexture(gl.TEXTURE0);
    } else {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.src); gl.uniform1i(this.warp.u.u_srcB, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.flowTex); gl.uniform1i(this.warp.u.u_flow, 2);
      gl.activeTexture(gl.TEXTURE0);
    }
    // focus build-up: opts.focus = { G } (frame → canvas similarity); canvas state is this.facc / this.fdec (.b = show live)
    const fo = opts.focus && this.facc ? opts.focus : null;
    gl.uniform1i(this.warp.u.u_fo, fo ? 1 : 0);
    if (fo) {
      const G = fo.G, geo = this.fgeo;
      gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.facc[this.fi].t); gl.uniform1i(this.warp.u.u_facc, 3);
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.fdec[this.fi].t); gl.uniform1i(this.warp.u.u_fdec, 4);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniformMatrix2fv(this.warp.u.u_G, false, [G[0], G[1], -G[1], G[0]]); gl.uniform2f(this.warp.u.u_Gt, G[2], G[3]);
      gl.uniform2f(this.warp.u.u_fC, geo.CW, geo.CH);
      const canvasPerOut = Math.hypot(M[0], M[1]) * Math.hypot(G[0], G[1]) * (geo.AH / geo.CH) / h;
      gl.uniform1i(this.warp.u.u_fcubic, canvasPerOut < 1.25 ? 1 : 0);
    } else {
      gl.uniform1i(this.warp.u.u_facc, 0); gl.uniform1i(this.warp.u.u_fdec, 0);
    }
    this.setOrient(this.warp, info);
    this.draw(this.A.fb, w, h);
    const raw = !!opts.raw;
    const sigma = Math.max(0.5, L.radius * (opts.pxScale || 1));
    const useBlur = !raw && (L.detail > 0.001 || L.denoise > 0.001);
    if (useBlur) {
      gl.useProgram(this.blur.p);
      gl.uniform1f(this.blur.u.u_sigma, sigma);
      gl.uniform1i(this.blur.u.u_rad, Math.min(24, Math.ceil(sigma * 3)));
      gl.uniform1i(this.blur.u.u_tex, 0);
      gl.bindTexture(gl.TEXTURE_2D, this.A.t); gl.uniform2f(this.blur.u.u_dir, 1, 0); this.draw(this.B.fb, w, h);
      gl.bindTexture(gl.TEXTURE_2D, this.B.t); gl.uniform2f(this.blur.u.u_dir, 0, 1); this.draw(this.Cc.fb, w, h);
    }
    const u = this.fin.u;
    gl.useProgram(this.fin.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.A.t); gl.uniform1i(u.u_a, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.Cc.t); gl.uniform1i(u.u_b, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(u.u_useBlur, useBlur ? 1 : 0);
    gl.uniform1i(u.u_raw, raw ? 1 : 0);
    gl.uniform1f(u.u_cas, L.cas); gl.uniform1f(u.u_detail, L.detail); gl.uniform1f(u.u_denoise, L.denoise);
    gl.uniform1f(u.u_exposure, L.exposure); gl.uniform1f(u.u_contrast, L.contrast);
    gl.uniform1f(u.u_shadows, L.shadows); gl.uniform1f(u.u_highlights, L.highlights);
    gl.uniform1f(u.u_sat, 1 + L.saturation); gl.uniform1f(u.u_vib, L.vibrance);
    gl.uniform1f(u.u_warm, L.warmth); gl.uniform1f(u.u_tint, L.tint);
    this.draw(this.outFB, w, h);
  }
}
