// WebXR port of openxr-skybox: an optometrist-office cubemap skybox with
// two lighting states and a per-eye prism-prescription simulation.
// Plain WebGL2 + WebXR (no frameworks) so the code mirrors the native app:
// same shaders, matrix math, cubemap conventions, and label-atlas overlay.
//
// Runs in Meta Quest Browser and Apple Vision Pro Safari (visionOS 2+).
// Without a headset the canvas shows a drag-to-look preview.

'use strict';

// ---- prism prescription (see native src/main.cpp for the derivation) ----
// A prism bends light toward its base; the image shifts toward the apex.
//   OD (right): 5.50 PD base-down + 1.00 PD base-out -> image up + nasal
//   OS (left):  5.50 PD base-up   + 1.00 PD base-out -> image down + nasal
// 1 prism diopter deviates by atan(1/100).
const PRISM_VERTICAL_PD = 5.5;
const PRISM_HORIZONTAL_PD = 1.0;
const PRISM_STEPS = [1.0, 1.25, 0.0, 0.25, 0.5, 0.75];

// prism_labels.png atlas: one row per 5% step, 0..200%
const LABEL_ROWS = 41;
const LABEL_ROW_PX = 80;
const LABEL_TEX_H = LABEL_ROWS * LABEL_ROW_PX;

// ---------------------------------------------------------------- matrices
// Column-major 4x4, matching WebXR's projectionMatrix layout.
function mul(a, b) {
  const r = new Float32Array(16);
  for (let c = 0; c < 4; ++c)
    for (let row = 0; row < 4; ++row) {
      let s = 0;
      for (let k = 0; k < 4; ++k) s += a[k * 4 + row] * b[c * 4 + k];
      r[c * 4 + row] = s;
    }
  return r;
}

// Inverse (transpose) of the rotation described by a unit quaternion —
// the view matrix for a skybox, where position is ignored.
function viewRotationFromQuat(q) {
  const { x, y, z, w } = q;
  const r = new Float32Array(16);
  r[0] = 1 - 2 * (y * y + z * z);
  r[4] = 2 * (x * y + w * z);
  r[8] = 2 * (x * z - w * y);
  r[1] = 2 * (x * y - w * z);
  r[5] = 1 - 2 * (x * x + z * z);
  r[9] = 2 * (y * z + w * x);
  r[2] = 2 * (x * z + w * y);
  r[6] = 2 * (y * z - w * x);
  r[10] = 1 - 2 * (x * x + y * y);
  r[15] = 1;
  return r;
}

function rotationX(a) {
  const c = Math.cos(a), s = Math.sin(a);
  const m = new Float32Array(16);
  m[0] = 1; m[5] = c; m[6] = s; m[9] = -s; m[10] = c; m[15] = 1;
  return m;
}

function rotationY(a) {
  const c = Math.cos(a), s = Math.sin(a);
  const m = new Float32Array(16);
  m[0] = c; m[2] = -s; m[5] = 1; m[8] = s; m[10] = c; m[15] = 1;
  return m;
}

function prismRotation(rightEye, scale) {
  const sign = rightEye ? 1 : -1; // OS mirrors OD
  const pitch = sign * scale * Math.atan(PRISM_VERTICAL_PD / 100);
  const yaw = sign * scale * Math.atan(PRISM_HORIZONTAL_PD / 100);
  return mul(rotationX(pitch), rotationY(yaw));
}

function perspective(fovYDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovYDeg * Math.PI) / 360);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = -(far + near) / (far - near);
  m[11] = -1;
  m[14] = -(2 * far * near) / (far - near);
  return m;
}

function translationMat(x, y, z) {
  const t = new Float32Array(16);
  t[0] = t[5] = t[10] = t[15] = 1;
  t[12] = x; t[13] = y; t[14] = z;
  return t;
}

// Model matrix for a pose: rotation (not transposed) plus translation.
// Used to place the gaze beams; the skybox itself stays rotation-only.
function poseMatrix(pos, q) {
  const { x, y, z, w } = q;
  const r = new Float32Array(16);
  r[0] = 1 - 2 * (y * y + z * z);
  r[1] = 2 * (x * y + w * z);
  r[2] = 2 * (x * z - w * y);
  r[4] = 2 * (x * y - w * z);
  r[5] = 1 - 2 * (x * x + z * z);
  r[6] = 2 * (y * z + w * x);
  r[8] = 2 * (x * z + w * y);
  r[9] = 2 * (y * z - w * x);
  r[10] = 1 - 2 * (x * x + y * y);
  r[12] = pos.x; r[13] = pos.y; r[14] = pos.z;
  r[15] = 1;
  return r;
}

// rotate a direction by the upper-left 3x3 of m (translation ignored)
function mulDir(m, v) {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2],
  ];
}

// forward (-Z) axis of a unit quaternion
function quatForward(q) {
  return [
    -2 * (q.x * q.z + q.w * q.y),
    -2 * (q.y * q.z - q.w * q.x),
    -(1 - 2 * (q.x * q.x + q.y * q.y)),
  ];
}

function quatMul(a, b) {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

// preview-camera orientation as a quaternion (yaw about Y, pitch about X),
// so the beams/targets have a gaze pose without a headset
function quatFromYawPitch(yaw, pitch) {
  const qy = { x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) };
  const qx = { x: Math.sin(pitch / 2), y: 0, z: 0, w: Math.cos(pitch / 2) };
  return quatMul(qy, qx);
}

// ---------------------------------------------------------------- shaders
const SKY_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
out vec3 vDir;
void main() {
  vDir = aPos;
  vec4 p = uViewProj * vec4(aPos, 0.0);  // rotation only: skybox at infinity
  gl_Position = p.xyww;                  // force depth to the far plane
}`;

// uFilter simulates anaglyph test filters over the eye (red OD, green OS
// per the Worth 4-dot convention); (1,1,1) when the filters are off
const SKY_FS = `#version 300 es
precision mediump float;
in vec3 vDir;
uniform samplerCube uSky;
uniform vec3 uFilter;
out vec4 outColor;
void main() { outColor = vec4(texture(uSky, vDir).rgb * uFilter, 1.0); }`;

const LABEL_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
uniform mat4 uViewProj;
uniform vec2 uUVRange;  // (v offset, v scale) selecting the atlas row
out vec2 vUV;
void main() {
  gl_Position = (uViewProj * vec4(aPos, 0.0)).xyww;
  vUV = vec2(aUV.x, uUVRange.x + aUV.y * uUVRange.y);
}`;

const LABEL_FS = `#version 300 es
precision mediump float;
in vec2 vUV;
uniform sampler2D uLabel;
uniform vec3 uFilter;
out vec4 outColor;
void main() {
  vec4 c = texture(uLabel, vUV);
  outColor = vec4(c.rgb * uFilter, c.a);
}`;

// solid-color gaze beams / chart markers — full MVP (translation matters)
const BEAM_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
uniform mat4 uMvp;
void main() { gl_Position = uMvp * vec4(aPos, 1.0); }`;

const BEAM_FS = `#version 300 es
precision mediump float;
uniform vec4 uColor;
uniform vec3 uFilter;
out vec4 outColor;
void main() { outColor = vec4(uColor.rgb * uFilter, uColor.a); }`;

// ---------------------------------------------------------------- app state
let gl = null;
let canvas = null;
let xrSession = null;
let xrRefSpace = null;

let skyProgram, locSkyVP, locSkySampler, locSkyFilter;
let labelProgram, locLabelVP, locLabelUV, locLabelTex, locLabelFilter;
let beamProgram, locBeamMvp, locBeamColor, locBeamFilter;
let cubeVao, quadVao, beamVao, targetVao, crossVao, panelVao;
let texBright, texDim, texLabels, texDisclaimer;

let lightsOn = true;
let prismStep = 2; // start with the prism off (PRISM_STEPS[2] === 0)
let prismScale = PRISM_STEPS[2];
let beamsVisible = false;      // gaze beams + chart targets (red OD, green OS)
let filtersOn = false;         // red/green Worth 4-dot test filters

// non-XR preview camera
let previewYaw = 0, previewPitch = 0;

// ---- intro narration (welcome + introduction with disclaimers) ----------
// Either controller trigger (select) or Space skips the current clip; a
// disclaimer panel covers the acuity display while it plays; the lights
// dim a few seconds after it ends.
const INTRO_DIM_DELAY_MS = 4000;
let audioCtx = null;
let introBuffers = [null, null, null];  // welcome, disclaimer, intro
let introSource = null;
let introClip = -1;                // -1 idle, 0 welcome, 1 disclaimer, 2 intro
const INTRO_CLIP_COUNT = 3;
let audioOk = false;
let introFinished = false;
let introFinishedTime = 0;
let introDimDone = false;

function introPlaying() {
  return introClip >= 0 && introClip < INTRO_CLIP_COUNT;
}

// ---------------------------------------------------------------- helpers
function compile(type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
    throw new Error('shader: ' + gl.getShaderInfoLog(sh));
  return sh;
}

function buildProgram(vsSrc, fsSrc) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
    throw new Error('link: ' + gl.getProgramInfoLog(prog));
  return prog;
}

function buildCubeVao() {
  const k = 1;
  const v = new Float32Array([
    -k, k, -k, -k, -k, -k, k, -k, -k, k, -k, -k, k, k, -k, -k, k, -k,
    -k, -k, k, -k, -k, -k, -k, k, -k, -k, k, -k, -k, k, k, -k, -k, k,
    k, -k, -k, k, -k, k, k, k, k, k, k, k, k, k, -k, k, -k, -k,
    -k, -k, k, -k, k, k, k, k, k, k, k, k, k, -k, k, -k, -k, k,
    -k, k, -k, k, k, -k, k, k, k, k, k, k, -k, k, k, -k, k, -k,
    -k, -k, -k, -k, -k, k, k, -k, -k, k, -k, -k, -k, -k, k, k, -k, k,
  ]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
  gl.bindVertexArray(null);
  return vao;
}

// The chart's caption band: authored face pixels (724..1324, 1076..1156)
// on the -Z wall, in cube space X = 2u/2048 - 1, Y = 1 - 2v/2048.
function buildLabelQuad() {
  const x0 = -0.29296875, x1 = 0.29296875;
  const yT = -0.05078125, yB = -0.12890625;
  const v = new Float32Array([
    x0, yT, -1, 0, 0, x0, yB, -1, 0, 1,
    x1, yT, -1, 1, 0, x1, yB, -1, 1, 1,
  ]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
  gl.bindVertexArray(null);
  return vao;
}

// ---- gaze beams, chart targets, disclaimer panel (mirror native) --------
// Colored by the Worth 4-dot lens convention: red OD (right), green OS.
const BEAM_COLOR_OS = [0.10, 0.85, 0.25, 1.0]; // green (left)
const BEAM_COLOR_OD = [0.95, 0.15, 0.10, 1.0]; // red (right)
// anaglyph test-filter transmissions (Worth 4-dot: red OD, green OS)
const FILTER_OFF = [1, 1, 1];
const FILTER_OD = [1, 0, 0];
const FILTER_OS = [0, 1, 0];
// digital acuity display bounds on the -Z wall (CHART_BOX in the generator)
const CHART_MIN_X = -0.458984375, CHART_MAX_X = 0.458984375;
const CHART_MIN_Y = -0.171875, CHART_MAX_Y = 0.47265625;
const TARGET_SEGMENTS = 32;
const TARGET_VERTS = 2 * (TARGET_SEGMENTS + 1);
const CROSS_VERTS = 12;

// Two crossed quads along -Z so the beam is visible from any angle.
function buildBeamVao() {
  const w = 0.004, z0 = -0.10, z1 = -8.0;
  const v = new Float32Array([
    -w, 0, z0, -w, 0, z1, w, 0, z0, w, 0, z0, -w, 0, z1, w, 0, z1,
    0, -w, z0, 0, -w, z1, 0, w, z0, 0, w, z0, 0, -w, z1, 0, w, z1,
  ]);
  return simpleVao(v, 12, 0);
}

// gaze-hit ring (triangle strip) for the with-prism fixation spot
function buildTargetVao() {
  const outer = 0.024, inner = 0.015;
  const v = [];
  for (let s = 0; s <= TARGET_SEGMENTS; ++s) {
    const a = (2 * Math.PI * s) / TARGET_SEGMENTS;
    const c = Math.cos(a), sn = Math.sin(a);
    v.push(c * outer, sn * outer, 0, c * inner, sn * inner, 0);
  }
  return simpleVao(new Float32Array(v), 12, 0);
}

// crosshair marker for the no-prism gaze point
function buildCrossVao() {
  const l = 0.030, t = 0.004;
  const v = new Float32Array([
    -l, -t, 0, -l, t, 0, l, -t, 0, l, -t, 0, -l, t, 0, l, t, 0,
    -t, -l, 0, t, -l, 0, -t, l, 0, -t, l, 0, t, -l, 0, t, l, 0,
  ]);
  return simpleVao(v, 12, 0);
}

// position-only VAO helper (stride bytes, attrib 0 = vec3)
function simpleVao(data, stride, offset) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, offset);
  gl.bindVertexArray(null);
  return vao;
}

// disclaimer panel covering the acuity display (CHART_BOX footprint),
// drawn through the label shader with the full texture selected
function buildPanelQuad() {
  const v = new Float32Array([
    CHART_MIN_X, CHART_MAX_Y, -1, 0, 0,
    CHART_MIN_X, CHART_MIN_Y, -1, 0, 1,
    CHART_MAX_X, CHART_MAX_Y, -1, 1, 0,
    CHART_MAX_X, CHART_MIN_Y, -1, 1, 1,
  ]);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
  gl.bindVertexArray(null);
  return vao;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('failed to load ' + url));
    img.src = url;
  });
}

async function loadCubemap(dir) {
  const faces = ['px', 'nx', 'py', 'ny', 'pz', 'nz'];
  const images = await Promise.all(
    faces.map((f) => loadImage(`${dir}/${f}.png`)));
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
  for (let i = 0; i < 6; ++i)
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.RGBA8,
                  gl.RGBA, gl.UNSIGNED_BYTE, images[i]);
  gl.generateMipmap(gl.TEXTURE_CUBE_MAP);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER,
                   gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

async function loadTexture2D(url) {
  const img = await loadImage(url);
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, img);
  // no mipmaps: atlas rows would bleed into each other
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

// ---------------------------------------------------------------- controls
function statusText() {
  const pct = Math.round(prismScale * 100);
  const prism = pct === 0
      ? 'off'
      : `${pct}% (${(PRISM_VERTICAL_PD * prismScale).toFixed(2)}PD V, ` +
        `${(PRISM_HORIZONTAL_PD * prismScale).toFixed(2)}PD H)`;
  return `Lights: ${lightsOn ? 'on' : 'down for the eye test'} · ` +
         `Prism: ${prism} · Beams: ${beamsVisible ? 'on' : 'off'} · ` +
         `Filters: ${filtersOn ? 'on' : 'off'}`;
}

function updateStatus() {
  const el = document.getElementById('status');
  if (el) el.textContent = statusText();
}

function toggleLights() {
  lightsOn = !lightsOn;
  updateStatus();
}

function cyclePrism() {
  prismStep = (prismStep + 1) % PRISM_STEPS.length;
  prismScale = PRISM_STEPS[prismStep];
  updateStatus();
}

function nudgePrism(delta) {
  prismScale = Math.min(2, Math.max(0, prismScale + delta));
  updateStatus();
}

function toggleBeams() {
  beamsVisible = !beamsVisible;
  updateStatus();
}

function toggleFilters() {
  filtersOn = !filtersOn;
  updateStatus();
}

// ---- intro narration playback (Web Audio) ----
async function initIntroAudio() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
    const urls = ['assets/audio/welcome.wav', 'assets/audio/disclaimer.wav',
                  'assets/audio/intro.wav'];
    introBuffers = await Promise.all(urls.map(async (u) => {
      const res = await fetch(u);
      if (!res.ok) throw new Error(u);
      return audioCtx.decodeAudioData(await res.arrayBuffer());
    }));
    audioOk = true;
  } catch (e) {
    audioOk = false;
    introDimDone = true; // no narration -> no auto-dim (native parity)
  }
}

function playIntroClip(i) {
  introClip = i;
  if (!audioOk || !introBuffers[i]) return finishIntro();
  const src = audioCtx.createBufferSource();
  src.buffer = introBuffers[i];
  src.connect(audioCtx.destination);
  src.onended = () => { if (introSource === src) advanceIntro(i); };
  introSource = src;
  src.start();
}

function advanceIntro(i) {
  introSource = null;
  if (i + 1 < INTRO_CLIP_COUNT) playIntroClip(i + 1);
  else finishIntro();
}

function finishIntro() {
  introClip = -1;
  if (!introFinished) {
    introFinished = true;
    introFinishedTime = performance.now();
  }
  updateStatus();
}

// starts on the first user gesture (VR entry or preview click) so the
// AudioContext is allowed to sound
function startIntro() {
  if (!audioOk || introClip !== -1 || introFinished) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  playIntroClip(0);
}

function skipIntroClip() {
  if (!introPlaying()) return;
  const i = introClip;
  if (introSource) {
    const s = introSource;
    introSource = null;
    s.onended = null;
    try { s.stop(); } catch (e) { /* already stopped */ }
  }
  advanceIntro(i);
}

// a few seconds after the introduction ends, dim the lights for the test
function updateIntroDim() {
  if (introDimDone || !introFinished) return;
  if (performance.now() - introFinishedTime >= INTRO_DIM_DELAY_MS) {
    introDimDone = true;
    if (lightsOn) toggleLights();
  }
}

// ---------------------------------------------------------------- drawing
// curPos = this eye's position; eyePoses[b] = {pos, quat} per eye (0 OS/left,
// 1 OD/right), located = how many are valid. WebXR exposes no eye-gaze API
// (see the sibling RESEARCH.md), so the beams/targets follow each eye's
// view direction — the native app's head-forward fallback rung.
function drawScene(projMatrix, viewRotMatrix, rightEye, curPos, eyePoses,
                  located) {
  const vp = mul(projMatrix,
                 mul(prismRotation(rightEye, prismScale), viewRotMatrix));
  const filt = filtersOn ? (rightEye ? FILTER_OD : FILTER_OS) : FILTER_OFF;

  gl.useProgram(skyProgram);
  gl.uniform3f(locSkyFilter, filt[0], filt[1], filt[2]);
  gl.uniformMatrix4fv(locSkyVP, false, vp);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_CUBE_MAP, lightsOn ? texBright : texDim);
  gl.uniform1i(locSkySampler, 0);
  gl.bindVertexArray(cubeVao);
  gl.drawArrays(gl.TRIANGLES, 0, 36);

  // live prism readout on the chart's bottom band
  let row = Math.round(prismScale * 20);
  row = Math.min(LABEL_ROWS - 1, Math.max(0, row));
  gl.useProgram(labelProgram);
  gl.uniform3f(locLabelFilter, filt[0], filt[1], filt[2]);
  gl.uniformMatrix4fv(locLabelVP, false, vp);
  gl.uniform2f(locLabelUV, (row * LABEL_ROW_PX + 1) / LABEL_TEX_H,
               (LABEL_ROW_PX - 2) / LABEL_TEX_H);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texLabels);
  gl.uniform1i(locLabelTex, 0);
  gl.bindVertexArray(quadVao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

  // disclaimer panel over the acuity display during the narration
  if (introPlaying() && texDisclaimer) {
    gl.uniform2f(locLabelUV, 0, 1);
    gl.bindTexture(gl.TEXTURE_2D, texDisclaimer);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  gl.bindVertexArray(null);

  // gaze beams + chart targets (ring = with-prism fixation spot, crosshair
  // = no-prism gaze point; they coincide at prism 0%)
  if (beamsVisible && located > 0) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix,
                        mul(prismRotation(rightEye, prismScale), viewFull));
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, filt[0], filt[1], filt[2]);
    for (let b = 0; b < 2; ++b) {
      const ep = eyePoses[b];
      if (!ep) continue; // closed / untracked eye
      const c = b === 1 ? BEAM_COLOR_OD : BEAM_COLOR_OS;
      gl.uniform4f(locBeamColor, c[0], c[1], c[2], c[3]);

      const mvp = mul(vpWorld, poseMatrix(ep.pos, ep.quat));
      gl.uniformMatrix4fv(locBeamMvp, false, mvp);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);

      // chart-space gaze directions with and without the prism
      const g = quatForward(ep.quat);
      const gLocal = mulDir(viewRotationFromQuat(ep.quat), g);
      const tmp = mulDir(prismRotation(b === 1, -prismScale), gLocal);
      const gPrism = mulDir(poseMatrix({ x: 0, y: 0, z: 0 }, ep.quat), tmp);
      const marks = [
        { dir: gPrism, vao: targetVao, mode: gl.TRIANGLE_STRIP,
          count: TARGET_VERTS },
        { dir: g, vao: crossVao, mode: gl.TRIANGLES, count: CROSS_VERTS },
      ];
      for (const mk of marks) {
        if (mk.dir[2] >= -1e-3) continue; // away from the chart wall
        const u = mk.dir[0] / -mk.dir[2];
        const v2 = mk.dir[1] / -mk.dir[2];
        if (u < CHART_MIN_X || u > CHART_MAX_X ||
            v2 < CHART_MIN_Y || v2 > CHART_MAX_Y) continue; // off display
        const mmvp = mul(vp, translationMat(u, v2, -1));
        gl.uniformMatrix4fv(locBeamMvp, false, mmvp);
        gl.bindVertexArray(mk.vao);
        gl.drawArrays(mk.mode, 0, mk.count);
      }
    }
    gl.bindVertexArray(null);
  }
}

// ---------------------------------------------------------------- XR loop
// edge-detected controller buttons: A/X (index 4) = beams, thumbstick
// click (index 3) = filters, B/Y (index 5) = prism. Trigger stays on the
// 'select' event.
const btnPrev = new WeakMap();
function pollControllers(session) {
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp) continue;
    const prev = btnPrev.get(src) || [];
    const edge = (i) => {
      const down = !!(gp.buttons[i] && gp.buttons[i].pressed);
      const was = !!prev[i];
      prev[i] = down;
      return down && !was;
    };
    if (edge(4)) toggleBeams();          // A / X
    if (edge(3)) toggleFilters();        // thumbstick click
    if (edge(5)) cyclePrism();           // B / Y
    edge(0); edge(1);                    // keep trigger/grip edges current
    btnPrev.set(src, prev);
  }
}

function onXRFrame(_t, frame) {
  const session = frame.session;
  session.requestAnimationFrame(onXRFrame);
  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose) return;

  pollControllers(session);
  updateIntroDim();

  // per-eye poses (0 = left/OS, 1 = right/OD) for the beams/targets
  const eyePoses = [];
  let located = 0;
  for (const view of pose.views) {
    const idx = view.eye === 'right' ? 1 : 0;
    eyePoses[idx] = {
      pos: view.transform.position,
      quat: view.transform.orientation,
    };
    located++;
  }

  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0.05, 0.05, 0.06, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  for (const view of pose.views) {
    const vp = layer.getViewport(view);
    if (!vp || vp.width === 0) continue;
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    drawScene(view.projectionMatrix,
              viewRotationFromQuat(view.transform.orientation),
              view.eye === 'right', view.transform.position, eyePoses,
              located);
  }
}

async function enterVR() {
  try {
    await gl.makeXRCompatible();
    xrSession = await navigator.xr.requestSession('immersive-vr');
    xrSession.updateRenderState({
      baseLayer: new XRWebGLLayer(xrSession, gl),
    });
    // LOCAL space: -Z is wherever the user faces at session start, so the
    // eye-chart wall begins directly in front of them.
    xrRefSpace = await xrSession.requestReferenceSpace('local');

    // right-hand pinch/trigger (or unhanded input) toggles the lights;
    // left-hand pinch/trigger or squeeze (grip) cycles prism strength.
    // While the narration plays, a trigger press skips the current clip.
    xrSession.addEventListener('select', (ev) => {
      if (introPlaying()) { skipIntroClip(); return; }
      if (ev.inputSource.handedness === 'left') cyclePrism();
      else toggleLights();
    });
    xrSession.addEventListener('squeeze', () => cyclePrism());
    xrSession.addEventListener('end', () => {
      xrSession = null;
      requestAnimationFrame(onPreviewFrame);
    });
    startIntro(); // enterVR is a user gesture, so audio is allowed to sound
    xrSession.requestAnimationFrame(onXRFrame);
  } catch (err) {
    setMessage('Could not start VR session: ' + err.message);
  }
}

// ---------------------------------------------------------------- preview
function onPreviewFrame() {
  if (xrSession) return; // XR loop has taken over
  requestAnimationFrame(onPreviewFrame);

  const w = canvas.clientWidth * devicePixelRatio;
  const h = canvas.clientHeight * devicePixelRatio;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.05, 0.05, 0.06, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  updateIntroDim();
  const proj = perspective(80, w / h, 0.05, 100);
  const viewRot = mul(rotationX(-previewPitch), rotationY(-previewYaw));
  const quat = quatFromYawPitch(previewYaw, previewPitch);
  const eyePoses = [{ pos: { x: 0, y: 0, z: 0 }, quat }];
  drawScene(proj, viewRot, false, { x: 0, y: 0, z: 0 }, eyePoses, 1);
}

function setMessage(text) {
  const el = document.getElementById('message');
  if (el) el.textContent = text;
}

// ---------------------------------------------------------------- init
async function main() {
  canvas = document.getElementById('canvas');
  gl = canvas.getContext('webgl2', { xrCompatible: true, antialias: true });
  if (!gl) {
    setMessage('WebGL2 is not available in this browser.');
    return;
  }
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.CULL_FACE);

  skyProgram = buildProgram(SKY_VS, SKY_FS);
  locSkyVP = gl.getUniformLocation(skyProgram, 'uViewProj');
  locSkySampler = gl.getUniformLocation(skyProgram, 'uSky');
  locSkyFilter = gl.getUniformLocation(skyProgram, 'uFilter');
  labelProgram = buildProgram(LABEL_VS, LABEL_FS);
  locLabelVP = gl.getUniformLocation(labelProgram, 'uViewProj');
  locLabelUV = gl.getUniformLocation(labelProgram, 'uUVRange');
  locLabelTex = gl.getUniformLocation(labelProgram, 'uLabel');
  locLabelFilter = gl.getUniformLocation(labelProgram, 'uFilter');
  beamProgram = buildProgram(BEAM_VS, BEAM_FS);
  locBeamMvp = gl.getUniformLocation(beamProgram, 'uMvp');
  locBeamColor = gl.getUniformLocation(beamProgram, 'uColor');
  locBeamFilter = gl.getUniformLocation(beamProgram, 'uFilter');
  cubeVao = buildCubeVao();
  quadVao = buildLabelQuad();
  beamVao = buildBeamVao();
  targetVao = buildTargetVao();
  crossVao = buildCrossVao();
  panelVao = buildPanelQuad();

  setMessage('Loading skybox…');
  [texBright, texDim, texLabels, texDisclaimer] = await Promise.all([
    loadCubemap('assets/skybox'),
    loadCubemap('assets/skybox_dim'),
    loadTexture2D('assets/prism_labels.png'),
    loadTexture2D('assets/disclaimer.png').catch(() => null),
  ]);
  initIntroAudio(); // fire-and-forget; degrades to silent if it fails
  setMessage('');
  updateStatus();

  // Enter VR button
  const button = document.getElementById('enter-vr');
  if (navigator.xr) {
    const supported = await navigator.xr
        .isSessionSupported('immersive-vr').catch(() => false);
    if (supported) {
      button.disabled = false;
      button.addEventListener('click', enterVR);
    } else {
      button.textContent = 'VR not supported here';
    }
  } else {
    button.textContent = 'WebXR not available';
  }

  // desktop preview controls: drag to look, L lights, P prism, [ ] fine-tune
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', (e) => {
    startIntro(); // first gesture in preview lets the narration sound
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    previewYaw -= (e.clientX - lastX) * 0.005;
    previewPitch -= (e.clientY - lastY) * 0.005;
    previewPitch = Math.max(-1.5, Math.min(1.5, previewPitch));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('pointerup', () => (dragging = false));
  window.addEventListener('keydown', (e) => {
    if (e.key === ' ' && introPlaying()) { skipIntroClip(); return; }
    if (e.key === 'l' || e.key === 'L') toggleLights();
    else if (e.key === 'p' || e.key === 'P') cyclePrism();
    else if (e.key === '[') nudgePrism(-0.05);
    else if (e.key === ']') nudgePrism(0.05);
    else if (e.key === 'g' || e.key === 'G') toggleBeams();
    else if (e.key === 'f' || e.key === 'F') toggleFilters();
  });

  requestAnimationFrame(onPreviewFrame);
}

main();
