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
${ORIENT}
${HDR}
out vec4 o;
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
  if (u_hdr > 0) c = hdrFix(c);
  o = vec4(c, 1.0);
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
  }
  setSize(w, h) {
    if (this.size[0] === w && this.size[1] === h) return;
    const gl = this.gl;
    this.canvas.width = w; this.canvas.height = h;
    for (const t of [this.A, this.B, this.Cc]) if (t) { gl.deleteTexture(t.t); gl.deleteFramebuffer(t.fb); }
    this.A = makeTex(gl, w, h); this.B = makeTex(gl, w, h); this.Cc = makeTex(gl, w, h);
    this.size = [w, h];
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
    this.draw(null, w, h);
  }
}
