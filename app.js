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

function scaleMat(sx, sy, sz) {
  const m = new Float32Array(16);
  m[0] = sx; m[5] = sy; m[10] = sz; m[15] = 1;
  return m;
}

// rotation about +Z (radians) — for the Double Maddox tilted line
function rotZMat(a) {
  const m = new Float32Array(16);
  const c = Math.cos(a), s = Math.sin(a);
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c; m[10] = 1; m[15] = 1;
  return m;
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

// y-component of a quaternion's right (+X) axis — the head-roll signal
// (>0 tilted toward the left shoulder); used by the Initial inspection.
function quatRightY(q) {
  return 2 * (q.x * q.y + q.w * q.z);
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

// world-anchored textured quad (test checklist panel) — full MVP, normal
// depth; paired with LABEL_FS
const PANEL3D_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
uniform mat4 uMvp;
out vec2 vUV;
void main() {
  gl_Position = uMvp * vec4(aPos, 1.0);
  vUV = aUV;
}`;

// ---------------------------------------------------------------- app state
let gl = null;
let canvas = null;
let xrSession = null;
let xrRefSpace = null;

let skyProgram, locSkyVP, locSkySampler, locSkyFilter;
let labelProgram, locLabelVP, locLabelUV, locLabelTex, locLabelFilter;
let beamProgram, locBeamMvp, locBeamColor, locBeamFilter;
let panelProgram, locPanelMvp, locPanelTex, locPanelFilter;
let cubeVao, quadVao, beamVao, targetVao, crossVao, panelVao;
let checklistPanelVao, checklistMarkVao, checklistCaretVao;
let workflowPanelVao, therapyDotVao;
let texBright, texDim, texLabels, texDisclaimer, texChecklist, texWorkflow;
let texTitleCards, texThpChecklist, texThpTitle;

let lightsOn = true;
let prismStep = 2; // start with the prism off (PRISM_STEPS[2] === 0)
let prismScale = PRISM_STEPS[2];
let beamsVisible = false;      // gaze beams + chart targets (red OD, green OS)
let filtersOn = false;         // red/green Worth 4-dot test filters
let aimPoses = [];             // controller/hand target-ray poses this frame

// test-selection workflow: pick tests (checkbox), hear a description (Talk),
// then START runs the selected tests one at a time. All selected by default;
// un-checks persist to localStorage.
let testSelected = [true, true, true, true, true, true, true];
let testMode = 'select';       // 'select' | 'run'
let runList = [], runIdx = 0;
let clHit = null;              // hovered panel element this frame {kind,row}
let thpHit = null;             // hovered therapy-panel element {kind,row}
// Initial inspection assessment (head tilt only in WebXR — no browser gaze)
let inspActive = false, inspStage = 0, inspT = 0;
let inspRollSum = 0, inspRollN = 0, inspRollDeg = 0, inspTilted = false;
// Cover test: occlude each eye in turn (visual only in WebXR — no gaze to
// measure the drift; shown for demonstration + spoken explanation)
let coverActive = false, coverStage = 0, coverT = 0, coverLast = 0;
// Eye movement test: follow the moving light (visual only in WebXR — no gaze
// to measure tracking or calibrate; self-report + demonstration)
let eyeActive = false, eyeStage = 0, eyeT = 0, eyeLast = 0, eyeFlagged = false;
let eyeDoubleAck = false;         // played the "seeing two is expected" clip
// Worth 4-dot: dichoptic dots (red->OD, green->OS, white->both); press once
// per dot seen. Count -> fuse (4) / suppress right (3) or left (2) / double.
// voice-enumerated forced choice: ask "one?..five?", press on your count,
// press again to confirm; no press through all five -> re-ask once then none
let worthActive = false, worthPhase = 0, worthAsk = 1, worthSelected = 0;
let worthPasses = 0, worthAskSaid = false, worthT = 0, worthLast = 0;
// Running prism estimate (dioptres) with per-axis evidence weight.
let estV = 0, estH = 0, estWv = 0, estWh = 0;
// Maddox rod: dichoptic line (right eye) + dot (left); a nulling sweep, the
// subject presses when the line crosses the dot -> subjective Rx.
let maddoxActive = false, maddoxStage = 0, maddoxT = 0, maddoxLast = 0;
let maddoxVDone = false, maddoxHDone = false;
// Von Graefe 2AFC prism staircase: show one H with a candidate prism (option
// A then B); press on the more-single one; the chosen candidate becomes the
// centre, a reversal halves the step. Vertical axis then horizontal.
let vgActive = false, vgStage = 0, vgPrism = 0, vgStep = 3, vgTrials = 0;
let vgLastDir = 0, vgOpt = 1, vgNoPress = 0, vgT = 0, vgLast = 0, vgSaid = false;
let vgResultV = 0, vgResultH = 0, vgHaveV = false, vgHaveH = false;
// Double Maddox rod (cyclotorsion): red line (OD) + white line (OS); the red
// line's tilt sweeps, press when parallel -> the tilt is the ocular torsion.
let dmActive = false, dmStage = 0, dmT = 0, dmLast = 0, dmDone = false;
let dmTorsion = 0;
function vgCandidate() { return vgPrism + (vgOpt === 1 ? -vgStep : vgStep); }
function dmTilt(t) {
  const s = t % 16;
  const tri = s < 8 ? s / 8 : (16 - s) / 8;  // 0->1->0
  return -10 + tri * 20;  // degrees
}

// non-XR preview camera
let previewYaw = 0, previewPitch = 0;

// ---- narration + workflow phase machine (mirrors native main.cpp) --------
// welcome -> disclaimer -> choose -> (Vision Testing | Vision Therapy).
// Narration is an on-demand single-clip player: playClip(i) sounds one
// buffer; on its end (or a skip) a per-frame check advances the phase.
const INTRO_DIM_DELAY_MS = 4000;
const CLIP_WELCOME = 0, CLIP_DISCLAIMER = 1, CLIP_CHOOSE = 2,
      CLIP_INTRO_TEST = 3, CLIP_INTRO_THER = 4;
// per-test description clips are contiguous from CLIP_DESC0 (by test row; 7)
const CLIP_DESC0 = 5;
const CLIP_INSP_LOOK = 12, CLIP_INSP_LEVEL = 13, CLIP_INSP_LEFT = 14,
      CLIP_INSP_RIGHT = 15, CLIP_INSP_ALIGNED = 16, CLIP_INSP_MISALIGNED = 17,
      CLIP_INSP_NOGAZE = 18, CLIP_INSP_KEEPLEVEL = 19, CLIP_INSP_DONE = 20;
const CLIP_COVER_LOOK = 21, CLIP_COVER_ALIGNED = 22, CLIP_COVER_DEVIATION = 23,
      CLIP_COVER_NOGAZE = 24, CLIP_COVER_DONE = 25;
const CLIP_EYE_LOOK = 26, CLIP_EYE_SMOOTH = 27, CLIP_EYE_LIMITED = 28,
      CLIP_EYE_REPEAT = 29, CLIP_EYE_NOGAZE = 30, CLIP_EYE_DONE = 31,
      CLIP_EYE_DOUBLE = 32;
const CLIP_WORTH_LOOK = 33, CLIP_WORTH_ASK1 = 34, CLIP_WORTH_ASK2 = 35,
      CLIP_WORTH_ASK3 = 36, CLIP_WORTH_ASK4 = 37, CLIP_WORTH_ASK5 = 38,
      CLIP_WORTH_CONFIRM = 39, CLIP_WORTH_FUSED = 40,
      CLIP_WORTH_SUPPRESS_RIGHT = 41, CLIP_WORTH_SUPPRESS_LEFT = 42,
      CLIP_WORTH_DOUBLE = 43, CLIP_WORTH_NONE = 44, CLIP_WORTH_DONE = 45;
const CLIP_MADDOX_LOOK = 46, CLIP_MADDOX_HORIZ = 47, CLIP_MADDOX_NONE = 48,
      CLIP_MADDOX_DONE = 49;
const CLIP_VG_LOOK = 50, CLIP_VG_ONE = 51, CLIP_VG_TWO = 52, CLIP_VG_HORIZ = 53,
      CLIP_VG_NONE = 54, CLIP_VG_DONE = 55;
const CLIP_DM_LOOK = 56, CLIP_DM_ALIGNED = 57, CLIP_DM_TORSION = 58,
      CLIP_DM_NONE = 59, CLIP_DM_DONE = 60;
// therapy: 9 desc_thp (Talk) + 9 thp_look (instructions) + thp_done, appended
const CLIP_THP_DESC0 = 61, CLIP_THP_LOOK0 = 70, CLIP_THP_DONE = 79;
const thpDescClip = r => CLIP_THP_DESC0 + r;
const thpLookClip = r => CLIP_THP_LOOK0 + r;
let audioCtx = null;
let introBuffers = [];            // indexed by the CLIP_* constants above
let introSource = null;
let audioOk = false;
let playingClip = -1;             // buffer index currently sounding, -1 = silence
let phaseStarted = false;         // first clip kicked off on the opening gesture
// phase: 'welcome','disclaimer','choose','select','intro_test','testing',
//        'intro_ther','therapy'
let phase = 'welcome';
let testDimAt = 0;                 // performance.now() target for the auto-dim
let testDimDone = false;
let workflowHovered = -1;

// Vision Therapy: a selectable list of 9 Vision-Home activities, run one at a
// time (mirrors the testing Talk/START/run). Un-checks persist to localStorage.
const THP_ROWS = 9;
const THP_ROW0_V = 0.175, THP_ROW_DV = 0.075;
const THP_CNP = 0, THP_DIVRANGE = 1, THP_DIVJUMPS = 2, THP_PRISM = 3,
      THP_VERT = 4, THP_SUSTAIN = 5, THP_BOTH = 6, THP_STEREO = 7,
      THP_CONTRAST = 8;
let therapyT = 0;                 // seconds since the current activity/stage
let therapyLast = 0;              // performance.now() of the previous frame
let thpSelected = [true, true, true, true, true, true, true, true, true];
let thpMode = 'select';           // 'select' | 'run'
let thpRunList = [], thpRunIdx = 0;
let thpAct = -1, thpStage = 0, thpSaid = false, thpCycle = 0;
let thpBeadZ = 0.30, thpBeadDir = -1;
let thpPrismH = 0, thpPrismV = 0;
let thpAccA = 0, thpAccB = 0, thpNa = 0, thpNb = 0, thpSeen = 0;
let thpArcsec = 400, thpContrast = 0, thpOnset = 0;

function narrating() {
  return playingClip >= 0 &&
         (phase === 'welcome' || phase === 'disclaimer' ||
          phase === 'intro_test' || phase === 'intro_ther');
}
function menuPhase() { return phase === 'choose' || phase === 'select'; }
function testingPhase() {
  return phase === 'intro_test' || phase === 'testing';
}
function therapyPhase() {
  return phase === 'intro_ther' || phase === 'therapy';
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

// ---- test selection panel (mirrors native + tools/generate_skybox.py) ---
const CHECKLIST_ROWS = 7;
const CHECKLIST_ROW0_V = 0.20;
const CHECKLIST_ROW_DV = 0.093;
const CHECKLIST_BOX_U = 0.075;
const CHECKLIST_BOX_HALF_U = 0.028;
const CHECKLIST_TALK_U = 0.205;
const CHECKLIST_TALK_MIN_U = 0.135, CHECKLIST_TALK_MAX_U = 0.275;
const CHECKLIST_START_V = 0.905, CHECKLIST_START_HALF_V = 0.05;
const CHECKLIST_DIST = 2.0;        // metres along -Z
const CHECKLIST_W = 1.40;          // panel width (m)
const CHECKLIST_H = CHECKLIST_W * 1040 / 1200;
const ROW_INSPECTION = 0, ROW_EYEMOVE = 1, ROW_COVER = 2, ROW_WORTH = 3,
      ROW_MADDOX = 4, ROW_VONGRAEFE = 5, ROW_DOUBLEMADDOX = 6;
const TITLE_CARD_ROWS = 7;
// Eye movement test: point-light waypoints (chart space), a sweep through the
// 9 cardinal gaze positions and back to centre.
const EYE_TARGETS = [[0, 0.15], [0.45, 0.15], [0.45, 0.5], [0, 0.55],
                     [-0.45, 0.5], [-0.45, 0.15], [-0.45, -0.2], [0, -0.25],
                     [0.45, -0.2], [0, 0.15]];
const EYE_WAYPOINTS = 10;

function checklistLocal(u, v) {
  return [(u - 0.5) * CHECKLIST_W, (0.5 - v) * CHECKLIST_H];
}

// ray/panel intersection: aim pose (pos + quaternion) vs a world-anchored
// panel at (0,0,-dist) spanning w x h -> normalized {u,v} or null
function menuHitUV(pos, q, dist, w, h) {
  const f = quatForward(q);
  if (f[2] >= -1e-4) return null;
  const t = (-dist - pos.z) / f[2];
  if (t <= 0) return null;
  const hx = pos.x + t * f[0];
  const hy = pos.y + t * f[1];
  if (Math.abs(hx) > w / 2 || Math.abs(hy) > h / 2) return null;
  return { u: hx / w + 0.5, v: 0.5 - hy / h, t };  // t = ray distance to panel
}

// simple row hit (for the workflow menu)
function menuHitRow(pos, q, dist, w, h, row0V, rowDV, rows) {
  const uv = menuHitUV(pos, q, dist, w, h);
  if (!uv) return -1;
  for (let i = 0; i < rows; ++i)
    if (Math.abs(uv.v - (row0V + i * rowDV)) <= rowDV / 2) return i;
  return -1;
}

// test-select panel hit: START band, else a row's Talk button or checkbox
function checklistHit(pos, q) {
  const uv = menuHitUV(pos, q, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
  if (!uv) return null;
  if (Math.abs(uv.v - CHECKLIST_START_V) <= CHECKLIST_START_HALF_V &&
      uv.u > 0.12 && uv.u < 0.88)
    return { kind: 'start', row: -1 };
  for (let i = 0; i < CHECKLIST_ROWS; ++i)
    if (Math.abs(uv.v - (CHECKLIST_ROW0_V + i * CHECKLIST_ROW_DV)) <=
        CHECKLIST_ROW_DV / 2) {
      const kind = (uv.u >= CHECKLIST_TALK_MIN_U && uv.u <= CHECKLIST_TALK_MAX_U)
          ? 'talk' : 'check';
      return { kind, row: i };
    }
  return null;
}

// therapy activity panel hit (same zones as checklistHit, therapy rows)
function therapyPanelHit(pos, q) {
  const uv = menuHitUV(pos, q, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
  if (!uv) return null;
  if (Math.abs(uv.v - CHECKLIST_START_V) <= CHECKLIST_START_HALF_V &&
      uv.u > 0.12 && uv.u < 0.88)
    return { kind: 'start', row: -1 };
  for (let i = 0; i < THP_ROWS; ++i)
    if (Math.abs(uv.v - (THP_ROW0_V + i * THP_ROW_DV)) <= THP_ROW_DV / 2) {
      const kind = (uv.u >= CHECKLIST_TALK_MIN_U && uv.u <= CHECKLIST_TALK_MAX_U)
          ? 'talk' : 'check';
      return { kind, row: i };
    }
  return null;
}

// workflow-choice menu (mirrors build_workflow_menu in generate_skybox.py)
const WORKFLOW_ROWS = 2;
const WORKFLOW_ROW0_V = 0.52;
const WORKFLOW_ROW_DV = 0.26;
const WORKFLOW_DIST = 2.0;
const WORKFLOW_W = 1.30;
const WORKFLOW_H = WORKFLOW_W * 560 / 1024;
function workflowHitRow(pos, q) {
  return menuHitRow(pos, q, WORKFLOW_DIST, WORKFLOW_W, WORKFLOW_H,
                    WORKFLOW_ROW0_V, WORKFLOW_ROW_DV, WORKFLOW_ROWS);
}

// world-anchored panel quad (w x h metres, uv over the whole texture)
function buildMenuPanelVao(w, h) {
  const hw = w / 2, hh = h / 2;
  const v = new Float32Array([
    -hw, hh, 0, 0, 0, -hw, -hh, 0, 0, 1,
    hw, hh, 0, 1, 0, hw, -hh, 0, 1, 1,
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

// therapy target colours (mirror native)
const T_RED = [0.95, 0.25, 0.20, 1], T_GREEN = [0.20, 0.85, 0.35, 1];
const T_YELLOW = [0.95, 0.85, 0.25, 1], T_WHITE = [0.95, 0.95, 0.95, 1];
const T_CYAN = [0.45, 0.85, 0.95, 1];

function buildChecklistPanelVao() {
  const hw = CHECKLIST_W / 2, hh = CHECKLIST_H / 2;
  const v = new Float32Array([
    -hw, hh, 0, 0, 0, -hw, -hh, 0, 0, 1,
    hw, hh, 0, 1, 0, hw, -hh, 0, 1, 1,
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

function buildFilledQuadVao(h) {
  return simpleVao(new Float32Array([
    -h, -h, 0, -h, h, 0, h, -h, 0, h, -h, 0, -h, h, 0, h, h, 0,
  ]), 12, 0);
}

function buildCaretVao() {
  const s = 0.018;
  return simpleVao(new Float32Array([-s, s, 0, -s, -s, 0, s, 0, 0]), 12, 0);
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
const TEST_NAMES = ['Initial inspection', 'Cover test', 'Worth 4-Dot',
                    'Prism simulation', 'Gaze tracking', 'Maddox rod'];
function statusText() {
  const pct = Math.round(prismScale * 100);
  const prism = pct === 0
      ? 'off'
      : `${pct}% (${(PRISM_VERTICAL_PD * prismScale).toFixed(2)}PD V, ` +
        `${(PRISM_HORIZONTAL_PD * prismScale).toFixed(2)}PD H)`;
  let head = '';
  if (typeof testingPhase === 'function' && testingPhase()) {
    head = testMode === 'run' && runList.length
        ? `Test ${runIdx + 1}/${runList.length}: ${TEST_NAMES[runList[runIdx]]} · `
        : 'Select tests, then Start · ';
  }
  return head +
         `Lights: ${lightsOn ? 'on' : 'down for the eye test'} · ` +
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

// no prism or colored lenses during the inspection (a bare head-posture +
// fixation check); gaze beams are still allowed
function cyclePrism() {
  if (inspActive || coverActive || eyeActive || worthActive || maddoxActive ||
      vgActive || dmActive) return;
  prismStep = (prismStep + 1) % PRISM_STEPS.length;
  prismScale = PRISM_STEPS[prismStep];
  updateStatus();
}

function nudgePrism(delta) {
  if (inspActive || coverActive || eyeActive || worthActive || maddoxActive ||
      vgActive || dmActive) return;
  prismScale = Math.min(2, Math.max(0, prismScale + delta));
  updateStatus();
}

function toggleBeams() {
  beamsVisible = !beamsVisible;
  updateStatus();
}

function toggleFilters() {
  if (inspActive || coverActive || eyeActive || worthActive || maddoxActive ||
      vgActive || dmActive) return;
  filtersOn = !filtersOn;
  updateStatus();
}

// ---- test selection + run flow (mirrors native main.cpp) ----
function loadSelection() {
  try {
    const s = localStorage.getItem('vision.testSelection');
    if (s && s.length >= CHECKLIST_ROWS)
      for (let i = 0; i < CHECKLIST_ROWS; ++i)
        if (s[i] === '0' || s[i] === '1') testSelected[i] = s[i] === '1';
  } catch (e) { /* private mode / disabled storage */ }
}
function saveSelection() {
  try {
    localStorage.setItem('vision.testSelection',
                         testSelected.map((b) => (b ? '1' : '0')).join(''));
  } catch (e) { /* ignore */ }
}
function loadThpSelection() {
  try {
    const s = localStorage.getItem('vision.therapySelection');
    if (s && s.length >= THP_ROWS)
      for (let i = 0; i < THP_ROWS; ++i)
        if (s[i] === '0' || s[i] === '1') thpSelected[i] = s[i] === '1';
  } catch (e) { /* ignore */ }
}
function saveThpSelection() {
  try {
    localStorage.setItem('vision.therapySelection',
                         thpSelected.map((b) => (b ? '1' : '0')).join(''));
  } catch (e) { /* ignore */ }
}
// ---- Vision Therapy activity run helpers (mirror native) ----
function activateThp(idx) {
  thpAct = thpRunList[idx];
  thpStage = 0; thpSaid = false; thpCycle = 0;
  thpPrismH = thpPrismV = 0; thpAccA = thpAccB = 0; thpNa = thpNb = 0;
  thpSeen = 0; thpArcsec = 400; thpContrast = 0; therapyT = 0;
  thpBeadZ = thpAct === THP_CNP ? 0.20 : thpAct === THP_DIVRANGE ? 0.30
             : thpAct === THP_DIVJUMPS ? 0.25
             : thpAct === THP_SUSTAIN ? 0.14 : 0.40;
  thpBeadDir = thpAct === THP_CNP ? -1 : 1;
  thpOnset = 1.5;
  playClip(thpLookClip(thpAct));
}
function thpStartRun() {
  thpRunList = [];
  for (let i = 0; i < THP_ROWS; ++i) if (thpSelected[i]) thpRunList.push(i);
  if (thpRunList.length === 0) return;
  thpRunIdx = 0; thpMode = 'run'; lightsOn = false;  // dim for exercises
  activateThp(0);
}
function thpAdvance() {  // called when the current activity finishes
  if (++thpRunIdx >= thpRunList.length) {
    thpMode = 'select'; thpAct = -1; lightsOn = true;  // lights back up
  } else {
    activateThp(thpRunIdx);
  }
}
function hasDemo(t) { return false; }  // every row is now a real test
function resetDemos() {
  filtersOn = false;
  beamsVisible = false;
  prismStep = 2;
  prismScale = 0;
  inspActive = false;
  coverActive = false;
  eyeActive = false;
  worthActive = false;
  maddoxActive = false;
  vgActive = false;
  dmActive = false;
}
// which eye view is occluded during the cover test, by elapsed time:
// 0-3s settle, 3-7s left (0), 7-11s right (1), else -1 (nothing covered)
function coveredView(t) {
  if (t >= 3 && t < 7) return 0;
  if (t >= 7 && t < 11) return 1;
  return -1;
}
// eye-movement point-light position (chart space) at time t: glide to the
// next waypoint over the first half of a segment, then dwell.
const EYE_PASS_DUR = (EYE_WAYPOINTS - 1) * 1.3;
function eyeLight(t) {
  const s = t / 1.3;
  let i = Math.floor(s);
  if (i > EYE_WAYPOINTS - 2) i = EYE_WAYPOINTS - 2;
  const frac = s - i;
  const mv = frac < 0.5 ? frac / 0.5 : 1.0;
  const sm = mv * mv * (3 - 2 * mv);
  return [EYE_TARGETS[i][0] + (EYE_TARGETS[i + 1][0] - EYE_TARGETS[i][0]) * sm,
          EYE_TARGETS[i][1] + (EYE_TARGETS[i + 1][1] - EYE_TARGETS[i][1]) * sm];
}
function activateRunTest(idx) {
  resetDemos();
  const t = runList[idx];
  if (t === ROW_WORTH) {
    worthActive = true; worthPhase = 0; worthAsk = 1; worthSelected = 0;
    worthPasses = 0; worthAskSaid = false; worthT = 0; worthLast = 0;
    playClip(CLIP_WORTH_LOOK);
  } else if (t === ROW_INSPECTION) {
    inspActive = true; inspStage = 0; inspT = 0;
    inspRollSum = 0; inspRollN = 0; inspRollDeg = 0;
    playClip(CLIP_INSP_LOOK);
  } else if (t === ROW_COVER) {
    coverActive = true; coverStage = 0; coverT = 0; coverLast = 0;
    playClip(CLIP_COVER_LOOK);
  } else if (t === ROW_EYEMOVE) {
    eyeActive = true; eyeStage = 0; eyeT = 0; eyeLast = 0; eyeFlagged = false;
    eyeDoubleAck = false;
    playClip(CLIP_EYE_LOOK);
  } else if (t === ROW_MADDOX) {
    maddoxActive = true; maddoxStage = 0; maddoxT = 0; maddoxLast = 0;
    maddoxVDone = false; maddoxHDone = false;
    playClip(CLIP_MADDOX_LOOK);
  } else if (t === ROW_VONGRAEFE) {
    vgActive = true; vgStage = 0; vgPrism = estWv > 0 ? estV : 0; vgStep = 3;
    vgTrials = 0; vgLastDir = 0; vgOpt = 1; vgNoPress = 0; vgT = 0; vgLast = 0;
    vgSaid = false; vgHaveV = false; vgHaveH = false; vgResultV = 0; vgResultH = 0;
    playClip(CLIP_VG_LOOK);
  } else if (t === ROW_DOUBLEMADDOX) {
    dmActive = true; dmStage = 0; dmT = 0; dmLast = 0; dmDone = false; dmTorsion = 0;
    playClip(CLIP_DM_LOOK);
  } else playClip(CLIP_DESC0 + t);  // title card + description
  updateStatus();
}
function startRun() {
  runList = [];
  for (let i = 0; i < CHECKLIST_ROWS; ++i)
    if (testSelected[i]) runList.push(i);
  if (runList.length === 0) return;
  runIdx = 0;
  testMode = 'run';
  estV = estH = estWv = estWh = 0;   // fresh prism estimate per run
  lightsOn = false;   // dim the room for the whole test run
  activateRunTest(0);
}
function advanceRun() {
  if (++runIdx >= runList.length) {
    testMode = 'select';
    resetDemos();
    skipClip();
    lightsOn = true;  // testing finished — bring the lights back up
  } else {
    activateRunTest(runIdx);
  }
  updateStatus();
}

// pick a workflow from the menu (plays its intro clip, then runs it)
function chooseWorkflow(r) {
  if (r === 0) { phase = 'intro_test'; playClip(CLIP_INTRO_TEST); }
  else { phase = 'intro_ther'; playClip(CLIP_INTRO_THER); }
  updateStatus();
}
// the run trigger is the activity's "press" (fusion break/recovery, re-fusion,
// detection, or "I see it") — record + advance (mirror native)
function thpRunPress() {
  if (!(thpStage >= 1 && thpStage < 90)) return;
  let finish = false;
  if (thpAct === THP_CNP || thpAct === THP_DIVRANGE) {
    if (thpStage === 1) {  // break
      thpAccA += thpBeadZ * 100; thpNa++; thpStage = 2; thpBeadDir = -thpBeadDir;
    } else {  // recovery
      thpAccB += thpBeadZ * 100; thpNb++;
      if (++thpCycle >= 3) finish = true;
      else {
        thpStage = 1;
        thpBeadZ = thpAct === THP_CNP ? 0.20 : 0.30;
        thpBeadDir = thpAct === THP_CNP ? -1 : 1;
      }
    }
  } else if (thpAct === THP_PRISM) {
    if (thpStage === 1) {  // break at current prism
      thpAccA = Math.abs(thpPrismH); thpNa++; thpStage = 2; thpBeadDir = -thpBeadDir;
    } else {
      thpAccB = Math.abs(thpPrismH); thpNb++;
      if (++thpCycle >= 2) finish = true;  // base-out then base-in
      else { thpStage = 1; thpPrismH = 0; thpBeadDir = -1; }
    }
  } else if (thpAct === THP_STEREO) {
    thpSeen = Math.round(thpArcsec);
    if (thpArcsec <= 100) finish = true; else thpArcsec *= 0.5;
  } else if (thpAct === THP_CONTRAST) {
    if (thpNa === 0 || thpContrast < thpAccA) thpAccA = thpContrast;
    if (++thpNa >= 3) finish = true;
    else { thpContrast = 0; thpOnset = therapyT + 1.5; }
  } else {  // jumps / vertical / sustained / both-eyes: count presses
    thpSeen++;
    if ((thpAct === THP_VERT || thpAct === THP_BOTH) && ++thpCycle >= 3)
      finish = true;
  }
  if (finish) { playClip(CLIP_THP_DONE); thpStage = 90; }
}
// therapy trigger (shared by the XR trigger and the desktop Space key)
function therapyTrigger() {
  if (thpMode === 'select') {
    if (thpHit) {
      if (thpHit.kind === 'check') {
        thpSelected[thpHit.row] = !thpSelected[thpHit.row];
        saveThpSelection();
      } else if (thpHit.kind === 'talk') {
        playClip(thpDescClip(thpHit.row));
      } else if (thpHit.kind === 'start') {
        thpStartRun();
      }
    } else if (playingClip >= 0) skipClip();
    else toggleLights();
  } else {  // run
    if (playingClip >= 0) skipClip();
    else thpRunPress();
  }
}

// ---- narration playback (Web Audio) — on-demand single-clip player ----
async function initIntroAudio() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
    const urls = ['assets/audio/welcome.wav', 'assets/audio/disclaimer.wav',
                  'assets/audio/choose.wav', 'assets/audio/intro_testing.wav',
                  'assets/audio/intro_therapy.wav',
                  'assets/audio/desc_inspection.wav', 'assets/audio/desc_eyemove.wav',
                  'assets/audio/desc_cover.wav',
                  'assets/audio/desc_worth.wav', 'assets/audio/desc_maddox.wav',
                  'assets/audio/desc_vongraefe.wav', 'assets/audio/desc_doublemaddox.wav',
                  'assets/audio/insp_look.wav', 'assets/audio/insp_level.wav',
                  'assets/audio/insp_left.wav', 'assets/audio/insp_right.wav',
                  'assets/audio/insp_aligned.wav', 'assets/audio/insp_misaligned.wav',
                  'assets/audio/insp_nogaze.wav', 'assets/audio/insp_keeplevel.wav',
                  'assets/audio/insp_done.wav',
                  'assets/audio/cover_look.wav', 'assets/audio/cover_aligned.wav',
                  'assets/audio/cover_deviation.wav', 'assets/audio/cover_nogaze.wav',
                  'assets/audio/cover_done.wav',
                  'assets/audio/eyemove_look.wav', 'assets/audio/eyemove_smooth.wav',
                  'assets/audio/eyemove_limited.wav', 'assets/audio/eyemove_repeat.wav',
                  'assets/audio/eyemove_nogaze.wav', 'assets/audio/eyemove_done.wav',
                  'assets/audio/eyemove_double.wav',
                  'assets/audio/worth_look.wav', 'assets/audio/worth_ask_one.wav',
                  'assets/audio/worth_ask_two.wav', 'assets/audio/worth_ask_three.wav',
                  'assets/audio/worth_ask_four.wav', 'assets/audio/worth_ask_five.wav',
                  'assets/audio/worth_confirm.wav', 'assets/audio/worth_fused.wav',
                  'assets/audio/worth_suppress_right.wav',
                  'assets/audio/worth_suppress_left.wav',
                  'assets/audio/worth_double.wav', 'assets/audio/worth_none.wav',
                  'assets/audio/worth_done.wav',
                  'assets/audio/maddox_look.wav', 'assets/audio/maddox_horiz.wav',
                  'assets/audio/maddox_none.wav', 'assets/audio/maddox_done.wav',
                  'assets/audio/vg_look.wav', 'assets/audio/vg_one.wav',
                  'assets/audio/vg_two.wav', 'assets/audio/vg_horiz.wav',
                  'assets/audio/vg_none.wav', 'assets/audio/vg_done.wav',
                  'assets/audio/dm_look.wav', 'assets/audio/dm_aligned.wav',
                  'assets/audio/dm_torsion.wav', 'assets/audio/dm_none.wav',
                  'assets/audio/dm_done.wav',
                  'assets/audio/desc_thp0.wav', 'assets/audio/desc_thp1.wav',
                  'assets/audio/desc_thp2.wav', 'assets/audio/desc_thp3.wav',
                  'assets/audio/desc_thp4.wav', 'assets/audio/desc_thp5.wav',
                  'assets/audio/desc_thp6.wav', 'assets/audio/desc_thp7.wav',
                  'assets/audio/desc_thp8.wav', 'assets/audio/thp_look0.wav',
                  'assets/audio/thp_look1.wav', 'assets/audio/thp_look2.wav',
                  'assets/audio/thp_look3.wav', 'assets/audio/thp_look4.wav',
                  'assets/audio/thp_look5.wav', 'assets/audio/thp_look6.wav',
                  'assets/audio/thp_look7.wav', 'assets/audio/thp_look8.wav',
                  'assets/audio/thp_done.wav'];
    introBuffers = await Promise.all(urls.map(async (u) => {
      const res = await fetch(u);
      if (!res.ok) throw new Error(u);
      return audioCtx.decodeAudioData(await res.arrayBuffer());
    }));
    audioOk = true;
  } catch (e) {
    audioOk = false;
  }
}

// play one clip; on natural end or skip, playingClip returns to -1 and the
// per-frame advancePhase() moves the phase machine on
function playClip(i) {
  playingClip = i;
  if (!audioOk || !introBuffers[i]) { playingClip = -1; return; }
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const src = audioCtx.createBufferSource();
  src.buffer = introBuffers[i];
  src.connect(audioCtx.destination);
  src.onended = () => { if (introSource === src) playingClip = -1; };
  introSource = src;
  src.start();
}

function skipClip() {
  if (introSource) {
    const s = introSource;
    introSource = null;
    s.onended = null;
    try { s.stop(); } catch (e) { /* already stopped */ }
  }
  playingClip = -1;
}

// first user gesture (VR entry / preview click): kick off the welcome clip
function startPhases() {
  if (phaseStarted) return;
  phaseStarted = true;
  if (audioOk) { phase = 'welcome'; playClip(CLIP_WELCOME); }
  else phase = 'select';  // no audio -> straight to the menu
  updateStatus();
}

// advance the phase machine when the current clip finishes or is skipped;
// playingClip is -1 whenever nothing is sounding (incl. no-audio)
function advancePhase() {
  if (!phaseStarted || playingClip >= 0) return;
  if (phase === 'welcome') { phase = 'disclaimer'; playClip(CLIP_DISCLAIMER); }
  else if (phase === 'disclaimer') { phase = 'choose'; playClip(CLIP_CHOOSE); }
  else if (phase === 'choose') { phase = 'select'; }
  else if (phase === 'intro_test') {
    phase = 'testing';  // lights stay up for test selection; dim on START
  } else if (phase === 'intro_ther') { phase = 'therapy'; thpMode = 'select'; }
  updateStatus();
}

// a few seconds into the testing workflow, dim the lights for the exam
function updateTestDim() {
  if (phase !== 'testing' || testDimDone || testDimAt === 0) return;
  if (performance.now() >= testDimAt) {
    testDimDone = true;
    if (lightsOn) toggleLights();
  }
}

// Initial inspection (WebXR: head-tilt only, no gaze): sample head roll from
// the viewer orientation, then report through a short spoken sequence.
let inspLast = 0;
function updateInspection(headQuat) {
  if (!inspActive) { inspLast = 0; return; }
  const now = performance.now();
  const dt = inspLast ? Math.min(0.1, (now - inspLast) / 1000) : 0;
  inspLast = now;
  inspT += dt;
  let ry = quatRightY(headQuat);
  ry = Math.max(-1, Math.min(1, ry));
  const rollDeg = Math.asin(ry) * 57.29578;
  if (inspT > 1) { inspRollSum += rollDeg; inspRollN++; }
  inspRollDeg = inspRollN ? inspRollSum / inspRollN : rollDeg;
  // after ~10s of fixation, report the findings, then advance automatically
  if (playingClip < 0) {
    if (inspStage === 0 && inspT > 10) {
      const avg = inspRollN ? inspRollSum / inspRollN : 0;
      inspTilted = Math.abs(avg) >= 3;
      playClip(!inspTilted ? CLIP_INSP_LEVEL
               : avg > 0 ? CLIP_INSP_LEFT : CLIP_INSP_RIGHT);
      inspStage = 1;
    } else if (inspStage === 1 && inspTilted) {
      // one-time explanation: keep the head level for the tests to come
      playClip(CLIP_INSP_KEEPLEVEL);
      inspTilted = false;
    } else if (inspStage === 1) {
      playClip(CLIP_INSP_NOGAZE);   // no per-eye gaze in the browser
      inspStage = 2;
    } else if (inspStage === 2) {
      playClip(CLIP_INSP_DONE);
      inspStage = 3;
    } else if (inspStage === 3) {
      advanceRun();                 // auto-advance to the next test
    }
  }
}

// Cover test clock: the browser has no per-eye gaze, so this occludes each
// eye in turn as a demonstration, explains it can't be measured here, then
// auto-advances. (OpenXR measures the covered eye's drift.)
function updateCover() {
  if (!coverActive) { coverLast = 0; return; }
  const now = performance.now();
  coverT += coverLast ? Math.min(0.1, (now - coverLast) / 1000) : 0;
  coverLast = now;
  if (coverT > 11.5 && playingClip < 0) {
    if (coverStage === 0) { playClip(CLIP_COVER_NOGAZE); coverStage = 1; }
    else if (coverStage === 1) { playClip(CLIP_COVER_DONE); coverStage = 2; }
    else if (coverStage === 2) advanceRun();
  }
}

// Eye movement clock: the browser has no gaze, so the light is a follow-along
// demonstration (self-report via trigger/pinch sets eyeFlagged); explains it
// can't be measured here, then auto-advances. (OpenXR measures + calibrates.)
function updateEye() {
  if (!eyeActive) { eyeLast = 0; return; }
  const now = performance.now();
  eyeT += eyeLast ? Math.min(0.1, (now - eyeLast) / 1000) : 0;
  eyeLast = now;
  if (eyeT >= EYE_PASS_DUR && playingClip < 0) {
    if (eyeStage === 0) { playClip(CLIP_EYE_NOGAZE); eyeStage = 1; }
    else if (eyeStage === 1) { playClip(CLIP_EYE_DONE); eyeStage = 2; }
    else if (eyeStage === 2) advanceRun();
  }
}

// Worth 4-dot: voice enumerates 1..5, the subject presses on the option they
// see (trigger routing), then presses to confirm; no press through all five
// re-asks once then reports no clear answer. worthT = time since the current
// clip ended (0 while it plays).
function updateWorth() {
  if (!worthActive) { worthLast = 0; return; }
  const now = performance.now();
  const dt = worthLast ? Math.min(0.1, (now - worthLast) / 1000) : 0;
  worthLast = now;
  worthT = playingClip >= 0 ? 0 : worthT + dt;
  const wPlaying = playingClip >= 0;
  if (worthPhase === 0) {  // instruction
    if (!wPlaying && worthT > 0.3) { worthPhase = 1; worthAsk = 1; worthAskSaid = false; }
  } else if (worthPhase === 1) {  // enumerate: name option worthAsk
    if (!worthAskSaid) {
      playClip(CLIP_WORTH_ASK1 + (worthAsk - 1)); worthAskSaid = true;
    } else if (!wPlaying && worthT > 1.4) {  // no press -> next option
      worthAsk++; worthAskSaid = false;
      if (worthAsk > 5) {
        worthPasses++;
        if (worthPasses >= 2) { worthPhase = 4; worthAskSaid = false; }
        else worthAsk = 1;  // re-ask the sequence once
      }
    }
  } else if (worthPhase === 2) {  // confirm
    if (!worthAskSaid) { playClip(CLIP_WORTH_CONFIRM); worthAskSaid = true; }
    else if (!wPlaying && worthT > 4.0) {  // no confirm -> re-choose
      worthPasses++; worthAskSaid = false;
      worthPhase = worthPasses >= 3 ? 4 : 1; worthAsk = 1;
    }
  } else if (worthPhase === 3) {  // report the finding
    if (!worthAskSaid) {
      playClip(worthSelected >= 5 ? CLIP_WORTH_DOUBLE
               : worthSelected === 4 ? CLIP_WORTH_FUSED
               : worthSelected === 3 ? CLIP_WORTH_SUPPRESS_RIGHT
               : CLIP_WORTH_SUPPRESS_LEFT);
      worthAskSaid = true;
    } else if (!wPlaying && worthT > 0.3) {
      playClip(CLIP_WORTH_DONE); worthPhase = 5; worthAskSaid = false;
    }
  } else if (worthPhase === 4) {  // no clear answer
    if (!worthAskSaid) { playClip(CLIP_WORTH_NONE); worthAskSaid = true; }
    else if (!wPlaying && worthT > 0.3) {
      playClip(CLIP_WORTH_DONE); worthPhase = 5; worthAskSaid = false;
    }
  } else if (worthPhase === 5) {  // done
    if (!wPlaying && worthT > 0.3) advanceRun();
  }
}

// fold a per-axis prism measurement (dioptres, weight) into the estimate
function addEvidenceV(v, w) { estV = (estV * estWv + v * w) / (estWv + w); estWv += w; }
function addEvidenceH(h, w) { estH = (estH * estWh + h * w) / (estWh + w); estWh += w; }
// Maddox nulling sweep: line offset (cube units) triangle-waves ±0.11 over
// 20 s. 100*offset at the press is the prism dioptres at this ~1 m depth.
function maddoxOffset(t) {
  const s = t % 20;
  const tri = s < 10 ? s / 10 : (20 - s) / 10;   // 0->1->0
  return -0.11 + tri * 0.22;
}
// ---- seven-segment readout (beamProgram active, therapyDotVao bound, colour
// set by caller). Draws below the acuity display so the estimate is visible.
function segRect(vpm, cx, cy, hw, hh) {
  gl.uniformMatrix4fv(locBeamMvp, false,
      mul(vpm, mul(translationMat(cx, cy, -1), scaleMat(hw, hh, 1))));
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}
const SEVEN_SEG = [0x3F, 0x06, 0x5B, 0x4F, 0x66, 0x6D, 0x7D, 0x07, 0x7F, 0x6F];
function drawDigit(vpm, d, x, y, w, h) {
  if (d < 0 || d > 9) return;
  const s = SEVEN_SEG[d];
  const t = w * 0.14, mxx = x + w * 0.5, sw = w * 0.5 - t, sh = h * 0.25 - t;
  const uy = y + h * 0.75, ly = y + h * 0.25;
  if (s & 0x01) segRect(vpm, mxx, y + h - t, sw, t);      // a
  if (s & 0x02) segRect(vpm, x + w - t, uy, t, sh);       // b
  if (s & 0x04) segRect(vpm, x + w - t, ly, t, sh);       // c
  if (s & 0x08) segRect(vpm, mxx, y + t, sw, t);          // d
  if (s & 0x10) segRect(vpm, x + t, ly, t, sh);           // e
  if (s & 0x20) segRect(vpm, x + t, uy, t, sh);           // f
  if (s & 0x40) segRect(vpm, mxx, y + h * 0.5, sw, t);    // g
}
function drawNumber(vpm, value, x, y, w, h) {
  if (value < 0) value = 0;
  const iv = Math.round(value * 10), frac = iv % 10, intp = Math.floor(iv / 10);
  const gap = w * 0.4;
  let cx = x;
  if (intp >= 10) { drawDigit(vpm, Math.floor(intp / 10) % 10, cx, y, w, h); cx += w + gap; }
  drawDigit(vpm, intp % 10, cx, y, w, h); cx += w + gap;
  segRect(vpm, cx, y + h * 0.12, w * 0.13, w * 0.13); cx += w * 0.5;  // point
  drawDigit(vpm, frac, cx, y, w, h);
}

// Maddox clock: vertical phase then horizontal then report; the sweep timer
// runs only between clips (starts after each spoken instruction).
function updateMaddox() {
  if (!maddoxActive) { maddoxLast = 0; return; }
  const now = performance.now();
  if (playingClip < 0)
    maddoxT += maddoxLast ? Math.min(0.1, (now - maddoxLast) / 1000) : 0;
  maddoxLast = now;
  if (playingClip >= 0) return;
  if (maddoxStage === 0 && (maddoxVDone || maddoxT > 15)) {
    playClip(CLIP_MADDOX_HORIZ); maddoxStage = 1; maddoxT = 0;
  } else if (maddoxStage === 1 && (maddoxHDone || maddoxT > 15)) {
    playClip((maddoxVDone || maddoxHDone) ? CLIP_MADDOX_DONE : CLIP_MADDOX_NONE);
    maddoxStage = 2;
  } else if (maddoxStage === 2) {
    advanceRun();
  }
}

// Von Graefe 2AFC staircase: cycle option A / B prompts; the press (trigger
// routing) records the choice + advances the staircase and stage.
function updateVg() {
  if (!vgActive) { vgLast = 0; return; }
  const now = performance.now();
  vgT = playingClip >= 0
      ? 0
      : vgT + (vgLast ? Math.min(0.1, (now - vgLast) / 1000) : 0);
  vgLast = now;
  const playing = playingClip >= 0;
  if (vgStage === 0) {
    if (!playing && vgT > 0.3) { vgStage = 1; vgOpt = 1; vgSaid = false; }
  } else if (vgStage === 1 || vgStage === 3) {
    if (!vgSaid) {
      playClip(vgOpt === 1 ? CLIP_VG_ONE : CLIP_VG_TWO); vgSaid = true;
    } else if (!playing && vgT > 1.6) {
      if (vgOpt === 1) { vgOpt = 2; vgSaid = false; }
      else {
        vgOpt = 1; vgSaid = false;
        if (++vgNoPress >= 3) { vgStage = vgStage === 1 ? 2 : 4; vgSaid = false; }
      }
    }
  } else if (vgStage === 2) {
    if (!vgSaid) { playClip(CLIP_VG_HORIZ); vgSaid = true; }
    else if (!playing && vgT > 0.3) {
      vgStage = 3; vgOpt = 1; vgSaid = false;
      vgPrism = estWh > 0 ? estH : 0; vgStep = 3; vgTrials = 0;
      vgLastDir = 0; vgNoPress = 0;
    }
  } else if (vgStage === 4) {
    if (!vgSaid) {
      if (vgHaveV) addEvidenceV(vgResultV, 1.5);
      if (vgHaveH) addEvidenceH(vgResultH, 1.5);
      playClip((vgHaveV || vgHaveH) ? CLIP_VG_DONE : CLIP_VG_NONE);
      vgSaid = true;
    } else if (!playing && vgT > 0.3) vgStage = 5;
  } else if (vgStage === 5) {
    if (!playing && vgT > 0.3) advanceRun();
  }
}

// Double Maddox: the red line's tilt sweeps; the press records the torsion.
function updateDm() {
  if (!dmActive) { dmLast = 0; return; }
  const now = performance.now();
  if (playingClip < 0)
    dmT += dmLast ? Math.min(0.1, (now - dmLast) / 1000) : 0;
  dmLast = now;
  if (playingClip >= 0) return;
  if (dmStage === 0) {
    if (dmT > 0.3) { dmStage = 1; dmT = 0; }
  } else if (dmStage === 1) {
    if (dmDone || dmT > 20) {
      playClip(!dmDone ? CLIP_DM_NONE
               : Math.abs(dmTorsion) > 2 ? CLIP_DM_TORSION : CLIP_DM_ALIGNED);
      dmStage = 2;
    }
  } else if (dmStage === 2) {
    playClip(CLIP_DM_DONE); dmStage = 3;
  } else if (dmStage === 3) {
    advanceRun();
  }
}

// press = "this setting is the more single letter" -> advance the staircase
function vgPress() {
  if (vgStage === 1 || vgStage === 3) {
    const dir = vgOpt === 1 ? -1 : 1;
    if (vgLastDir !== 0 && dir !== vgLastDir) vgStep *= 0.5;
    vgLastDir = dir; vgPrism = vgCandidate(); vgNoPress = 0;
    if (++vgTrials >= 8 || vgStep < 0.5) {
      if (vgStage === 1) { vgResultV = vgPrism; vgHaveV = true; vgStage = 2; }
      else { vgResultH = vgPrism; vgHaveH = true; vgStage = 4; }
      vgSaid = false;
    } else { vgOpt = 1; vgSaid = false; vgT = 0; }
  } else if (playingClip >= 0) skipClip();
}
// press = "the two lines look parallel now" -> the tilt is the torsion
function dmPress() {
  if (playingClip >= 0) skipClip();
  else if (dmStage === 1 && !dmDone) { dmTorsion = dmTilt(dmT); dmDone = true; }
}

// advance the therapy activity clock + drive the frame stage machine
function updateTherapyClock() {
  const now = performance.now();
  const dt = therapyLast ? Math.min(0.1, (now - therapyLast) / 1000) : 0;
  therapyLast = now;
  const running = phase === 'therapy' && thpMode === 'run';
  // clock advances between clips (so the sweep starts after the instruction)
  if ((running && playingClip < 0) || phase === 'intro_ther') therapyT += dt;
  if (running && playingClip < 0) {
    if (thpAct === THP_CNP || thpAct === THP_DIVRANGE)
      thpBeadZ += thpBeadDir * 0.015 * dt;
    if (thpAct === THP_PRISM)
      thpPrismH += (thpBeadDir >= 0 ? 1 : -1) * 0.5 * dt;  // 0.5Δ/s
    if (thpAct === THP_CONTRAST && therapyT > thpOnset)
      thpContrast += 0.02 * dt;  // 2%/s Weber
  }
  if (!running) return;
  const tp = playingClip >= 0;
  if (thpStage === 0) {            // instruction clip playing
    if (!tp) { thpStage = 1; therapyT = 0; }
  } else if (thpStage === 90) {    // done clip
    if (!tp) thpAdvance();
  } else {                         // active
    let finish = false;
    if (thpAct === THP_CNP && thpBeadZ <= 0.04) { thpBeadZ = 0.04; thpBeadDir = 1; }
    if (thpAct === THP_DIVRANGE && thpBeadZ >= 1.40) { thpBeadZ = 1.40; thpBeadDir = -1; }
    if (thpBeadZ < 0.03) thpBeadZ = 0.03;
    thpPrismH = Math.max(-15, Math.min(15, thpPrismH));
    if (thpContrast > 0.6) thpContrast = 0.6;
    if (thpAct === THP_DIVJUMPS) {
      thpBeadZ = (Math.floor(therapyT / 1.5) % 2 === 0) ? 0.25 : 1.2;
      if (therapyT > 60) finish = true;
    }
    if (thpAct === THP_SUSTAIN && therapyT > 60) finish = true;
    if (thpAct === THP_VERT) thpPrismV = 2.5 * (thpCycle % 2 === 0 ? 1 : -1);
    if (thpAct === THP_BOTH && therapyT > 15) {
      if (++thpCycle >= 3) finish = true; else therapyT = 0;
    }
    if (thpAct === THP_STEREO && therapyT > 15) finish = true;
    if (thpAct === THP_CONTRAST && thpContrast >= 0.6) {
      if (thpNa === 0) thpAccA = 0.6;
      if (++thpNa >= 3) finish = true;
      else { thpContrast = 0; thpOnset = 1.5; therapyT = 0; }
    }
    if (finish) { playClip(CLIP_THP_DONE); thpStage = 90; }
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

  // disclaimer panel over the acuity display while the welcome/disclaimer
  // clip plays
  if ((phase === 'welcome' || phase === 'disclaimer') && texDisclaimer) {
    gl.uniform2f(locLabelUV, 0, 1);
    gl.bindTexture(gl.TEXTURE_2D, texDisclaimer);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  gl.bindVertexArray(null);

  // test-select panel (select mode) — world-anchored, no prism, so the ray
  // and the visual align
  if (testingPhase() && testMode === 'select' && texChecklist) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);

    gl.useProgram(panelProgram);
    gl.uniform3f(locPanelFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locPanelMvp, false,
                        mul(vpWorld, translationMat(0, 0, -CHECKLIST_DIST)));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texChecklist);
    gl.uniform1i(locPanelTex, 0);
    gl.bindVertexArray(checklistPanelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    for (let r = 0; r < CHECKLIST_ROWS; ++r) {
      if (!testSelected[r]) continue;
      const [bx, by] = checklistLocal(CHECKLIST_BOX_U,
                                      CHECKLIST_ROW0_V + r * CHECKLIST_ROW_DV);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(bx, by, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.20, 0.85, 0.35, 1);
      gl.bindVertexArray(checklistMarkVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    if (clHit) {  // caret marks the hovered element
      let cu, cv;
      if (clHit.kind === 'start') { cu = 0.13; cv = CHECKLIST_START_V; }
      else {
        cv = CHECKLIST_ROW0_V + clHit.row * CHECKLIST_ROW_DV;
        cu = clHit.kind === 'talk' ? (CHECKLIST_TALK_MIN_U - 0.03) : 0.03;
      }
      const [cx, cy] = checklistLocal(cu, cv);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(cx, cy, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.40, 0.85, 0.95, 1);
      gl.bindVertexArray(checklistCaretVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    for (const a of aimPoses) {
      // stop the ray at the panel so it visibly lands on the button
      const uv = menuHitUV(a.pos, a.quat, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
      const bz = uv ? uv.t / 8.0 : 1.0;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(poseMatrix(a.pos, a.quat), scaleMat(1, 1, bz))));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
      if (uv) {  // cursor dot where the ray meets the panel
        gl.uniformMatrix4fv(locBeamMvp, false,
            mul(vpWorld, mul(translationMat((uv.u - 0.5) * CHECKLIST_W,
                                            (0.5 - uv.v) * CHECKLIST_H,
                                            -CHECKLIST_DIST + 0.01),
                             scaleMat(0.012, 0.012, 1))));
        gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
        gl.bindVertexArray(therapyDotVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    }
    gl.bindVertexArray(null);
  }

  // running a descriptive test (Cover/Maddox): a title card over the chart
  if (testingPhase() && testMode === 'run' && texTitleCards &&
      !hasDemo(runList[runIdx]) && runList[runIdx] !== ROW_INSPECTION &&
      runList[runIdx] !== ROW_COVER && runList[runIdx] !== ROW_EYEMOVE &&
      runList[runIdx] !== ROW_WORTH && runList[runIdx] !== ROW_MADDOX &&
      runList[runIdx] !== ROW_VONGRAEFE && runList[runIdx] !== ROW_DOUBLEMADDOX) {
    const t = runList[runIdx];
    gl.useProgram(labelProgram);
    gl.uniform3f(locLabelFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locLabelVP, false, vp);
    gl.uniform2f(locLabelUV, t / TITLE_CARD_ROWS, 1 / TITLE_CARD_ROWS);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texTitleCards);
    gl.uniform1i(locLabelTex, 0);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }

  // Inspection + Cover test share a clean fixation scene (Worth chart
  // covered, a large letter "H"). Inspection adds a tilt meter; the Cover
  // test blacks out one eye at a time.
  if (inspActive || coverActive) {
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    // cover the Worth 4-dot chart — irrelevant to these tests
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0.03, 0.03, 0.045, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    gl.uniform4f(locBeamColor, 0.95, 0.95, 0.98, 1);
    const hz = -1.0, hHalfW = 0.05, hHalfH = 0.07, hStroke = 0.009;
    const bars = [[-hHalfW, 0.15, hStroke, hHalfH],
                  [hHalfW, 0.15, hStroke, hHalfH],
                  [0, 0.15, hHalfW, hStroke]];
    for (const b of bars) {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(b[0], b[1], hz),
                      scaleMat(b[2], b[3], 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    if (inspActive) {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(0, -0.18, -1), scaleMat(0.30, 0.006, 1))));
      gl.uniform4f(locBeamColor, 0.35, 0.40, 0.48, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      let mx = inspRollDeg / 20;
      mx = Math.max(-1, Math.min(1, mx));
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(mx * 0.30, -0.18, -1),
                      scaleMat(0.012, 0.03, 1))));
      gl.uniform4f(locBeamColor, 0.45, 0.85, 0.95, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    // Cover test: black out the covered eye (fullscreen NDC quad)
    if (coverActive && coveredView(coverT) === (rightEye ? 1 : 0)) {
      gl.uniformMatrix4fv(locBeamMvp, false, scaleMat(1, 1, 1));
      gl.uniform4f(locBeamColor, 0, 0, 0, 1);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindVertexArray(null);
  }

  // Eye movement test: a point of light (glow + bright core) glides through
  // the gaze positions; the subject follows it with the eyes
  if (eyeActive) {
    const [lx, ly] = eyeLight(eyeT);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    // blank the acuity display: a black panel over the whole chart
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    gl.uniform4f(locBeamColor, 0.55, 0.42, 0.20, 1);    // outer glow
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vp, mul(translationMat(lx, ly, -1), scaleMat(0.028, 0.028, 1))));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.uniform4f(locBeamColor, 1.0, 0.93, 0.72, 1);     // warm halo
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vp, mul(translationMat(lx, ly, -1), scaleMat(0.016, 0.016, 1))));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.uniform4f(locBeamColor, 1, 1, 1, 1);             // bright core
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vp, mul(translationMat(lx, ly, -1), scaleMat(0.008, 0.008, 1))));
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
  }

  // Worth 4-dot: dichoptic dots — red to the right eye only, green to the
  // left only, white to both. The per-eye rendering is the dissociation.
  if (worthActive) {
    const wcx = 0, wcy = 0.15, wsp = 0.20, wr = 0.028;
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);             // blank display
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    const worthDot = (x, y) => {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(x, y, -1), scaleMat(wr, wr, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);       // white, both eyes
    worthDot(wcx, wcy - wsp);
    if (rightEye) {                                     // red top -> OD only
      gl.uniform4f(locBeamColor, 1, 0.15, 0.15, 1);
      worthDot(wcx, wcy + wsp);
    } else {                                            // two green -> OS only
      gl.uniform4f(locBeamColor, 0.15, 0.95, 0.25, 1);
      worthDot(wcx - wsp, wcy);
      worthDot(wcx + wsp, wcy);
    }
    gl.bindVertexArray(null);
  }

  // Maddox rod: white dot -> left eye, red line -> right eye. Stage 0 =
  // horizontal line swept vertically; stage 1 = vertical line swept sideways.
  if (maddoxActive) {
    const off = maddoxOffset(maddoxT);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);             // blank display
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    const quad = (x, y, hw, hh) => {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(x, y, -1), scaleMat(hw, hh, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    if (!rightEye) {                                    // white dot -> OS
      gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
      quad(0, 0.15, 0.012, 0.012);
    } else {                                            // red line -> OD
      gl.uniform4f(locBeamColor, 1, 0.2, 0.2, 1);
      if (maddoxStage === 0) quad(0, 0.15 + off, 0.16, 0.006);
      else quad(0 + off, 0.15, 0.006, 0.16);
    }
    gl.bindVertexArray(null);
  }

  // Von Graefe: one white letter H, shifted per eye by the candidate prism
  // (vertical phase then horizontal). Reads single when it matches the
  // deviation.
  if (vgActive) {
    const vAxis = vgStage < 3;
    const cand = (vgStage === 1 || vgStage === 3) ? vgCandidate() : 0;
    const off = (rightEye ? 1 : -1) * cand / 200;
    const hcx = vAxis ? 0 : off;
    const hcy = 0.15 + (vAxis ? off : 0);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
    const hHalfW = 0.05, hHalfH = 0.07, hStroke = 0.009;
    const bars = [[hcx - hHalfW, hcy, hStroke, hHalfH],
                  [hcx + hHalfW, hcy, hStroke, hHalfH],
                  [hcx, hcy, hHalfW, hStroke]];
    for (const b of bars) {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(b[0], b[1], -1), scaleMat(b[2], b[3], 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindVertexArray(null);
  }

  // Double Maddox: white horizontal line (OS) + red line (OD) whose tilt
  // sweeps; the subject presses when the two look parallel.
  if (dmActive) {
    const tilt = dmStage === 1 ? dmTilt(dmT) * 0.0174533 : 0;
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    if (!rightEye) {  // white line -> OS
      gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(0, 0.19, -1), scaleMat(0.16, 0.006, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    } else {          // red line -> OD, tilted
      gl.uniform4f(locBeamColor, 1, 0.2, 0.2, 1);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(0, 0.11, -1),
                      mul(rotZMat(tilt), scaleMat(0.16, 0.006, 1)))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindVertexArray(null);
  }

  // Prism estimate readout below the acuity display (seven-segment): a
  // vertical row and a horizontal row (Δ) + a confidence bar. No calibration
  // line — WebXR has no gaze.
  if (testingPhase()) {
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.bindVertexArray(therapyDotVao);
    const conf = 0.5 * (Math.min(estWv, 1) + Math.min(estWh, 1));
    const found = estWv >= 1 && estWh >= 1;
    const cr = found ? 0.30 : 0.95, cg = found ? 0.90 : 0.82,
          cb = found ? 0.42 : 0.38;
    const dw = 0.040, dh = 0.060;
    gl.uniform4f(locBeamColor, cr, cg, cb, 1);
    segRect(vp, -0.34, -0.25, 0.008, 0.035);           // vertical-bar icon
    drawNumber(vp, Math.abs(estV), -0.30, -0.28, dw, dh);
    segRect(vp, -0.34, -0.37, 0.035, 0.008);           // horizontal-bar icon
    drawNumber(vp, Math.abs(estH), -0.30, -0.40, dw, dh);
    gl.uniform4f(locBeamColor, 0.25, 0.25, 0.30, 1);   // confidence track
    segRect(vp, 0, -0.47, 0.30, 0.008);
    gl.uniform4f(locBeamColor, cr, cg, cb, 1);          // confidence fill
    segRect(vp, -0.30 + 0.30 * conf, -0.47, 0.30 * conf, 0.008);
    gl.bindVertexArray(null);
  }

  // workflow-choice menu ("what are you looking for?")
  if (menuPhase() && texWorkflow) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.useProgram(panelProgram);
    gl.uniform3f(locPanelFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locPanelMvp, false,
                        mul(vpWorld, translationMat(0, 0, -WORKFLOW_DIST)));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texWorkflow);
    gl.uniform1i(locPanelTex, 0);
    gl.bindVertexArray(workflowPanelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    if (workflowHovered >= 0) {
      const cx = (0.14 - 0.5) * WORKFLOW_W;
      const cy = (0.5 - (WORKFLOW_ROW0_V + workflowHovered * WORKFLOW_ROW_DV)) *
                 WORKFLOW_H;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(translationMat(cx, cy, -WORKFLOW_DIST + 0.004),
                           scaleMat(1.6, 1.6, 1.6))));
      gl.uniform4f(locBeamColor, 0.40, 0.85, 0.95, 1);
      gl.bindVertexArray(checklistCaretVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    for (const a of aimPoses) {
      const uv = menuHitUV(a.pos, a.quat, WORKFLOW_DIST, WORKFLOW_W, WORKFLOW_H);
      const bz = uv ? uv.t / 8.0 : 1.0;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(poseMatrix(a.pos, a.quat), scaleMat(1, 1, bz))));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
      if (uv) {
        gl.uniformMatrix4fv(locBeamMvp, false,
            mul(vpWorld, mul(translationMat((uv.u - 0.5) * WORKFLOW_W,
                                            (0.5 - uv.v) * WORKFLOW_H,
                                            -WORKFLOW_DIST + 0.01),
                             scaleMat(0.012, 0.012, 1))));
        gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
        gl.bindVertexArray(therapyDotVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    }
    gl.bindVertexArray(null);
  }

  // Vision Therapy activity-select panel (checklist_therapy.png), world-anchored
  if (therapyPhase() && thpMode === 'select' && texThpChecklist) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.useProgram(panelProgram);
    gl.uniform3f(locPanelFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locPanelMvp, false,
                        mul(vpWorld, translationMat(0, 0, -CHECKLIST_DIST)));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texThpChecklist);
    gl.uniform1i(locPanelTex, 0);
    gl.bindVertexArray(checklistPanelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    for (let r = 0; r < THP_ROWS; ++r) {
      if (!thpSelected[r]) continue;
      const [bx, by] = checklistLocal(CHECKLIST_BOX_U, THP_ROW0_V + r * THP_ROW_DV);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(bx, by, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.20, 0.85, 0.35, 1);
      gl.bindVertexArray(checklistMarkVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    if (thpHit) {  // caret marks the hovered element
      let cu, cv;
      if (thpHit.kind === 'start') { cu = 0.13; cv = CHECKLIST_START_V; }
      else {
        cv = THP_ROW0_V + thpHit.row * THP_ROW_DV;
        cu = thpHit.kind === 'talk' ? (CHECKLIST_TALK_MIN_U - 0.03) : 0.03;
      }
      const [cx, cy] = checklistLocal(cu, cv);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(cx, cy, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.40, 0.85, 0.95, 1);
      gl.bindVertexArray(checklistCaretVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    for (const a of aimPoses) {  // stop the ray at the panel + cursor dot
      const uv = menuHitUV(a.pos, a.quat, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
      const bz = uv ? uv.t / 8.0 : 1.0;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(poseMatrix(a.pos, a.quat), scaleMat(1, 1, bz))));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
      if (uv) {
        gl.uniformMatrix4fv(locBeamMvp, false,
            mul(vpWorld, mul(translationMat((uv.u - 0.5) * CHECKLIST_W,
                                            (0.5 - uv.v) * CHECKLIST_H,
                                            -CHECKLIST_DIST + 0.01),
                             scaleMat(0.012, 0.012, 1))));
        gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);
        gl.bindVertexArray(therapyDotVao);
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    }
    gl.bindVertexArray(null);
  }

  // Vision Therapy activity targets (world-anchored). Bead-on-cord activities
  // are binocular (true stereo poses the vergence demand); prism/vertical/both/
  // stereo are dichoptic — offset per eye. NOTE: the bead is a simple quad +
  // a beam "cord"; realistic drilled-bead / twisted-cord geometry is a
  // follow-up (needs a depth buffer this pass doesn't allocate).
  if (therapyPhase() && thpMode === 'run' && thpStage !== 0) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    const RED = [0.95, 0.25, 0.20, 1], GREEN = [0.20, 0.85, 0.35, 1];
    const YELLOW = [0.95, 0.85, 0.25, 1], WHITE = [0.95, 0.95, 0.95, 1];
    const eyeSign = rightEye ? 1 : -1;  // OD +, OS -
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    const solid = (model, c, vao, mode, count) => {
      gl.uniformMatrix4fv(locBeamMvp, false, mul(vpWorld, model));
      gl.uniform4f(locBeamColor, c[0], c[1], c[2], c[3]);
      gl.bindVertexArray(vao);
      gl.drawArrays(mode, 0, count);
    };
    if (thpAct === THP_CNP || thpAct === THP_DIVRANGE ||
        thpAct === THP_DIVJUMPS || thpAct === THP_SUSTAIN) {
      // bead on a cord along -Z at the current distance
      solid(scaleMat(1, 1, thpBeadZ / 8.0), [0.80, 0.80, 0.88, 1], beamVao,
            gl.TRIANGLES, 12);
      const s = 0.018;
      solid(mul(translationMat(0, 0, -thpBeadZ), scaleMat(s, s, s)),
            YELLOW, therapyDotVao, gl.TRIANGLES, 6);
    } else if (thpAct === THP_PRISM || thpAct === THP_VERT) {
      // one bead, pulled apart per eye by the activity prism
      const d = 1.2;
      const dx = eyeSign * (thpPrismH / 100) * d;
      const dy = eyeSign * (thpPrismV / 100) * d;
      solid(mul(translationMat(dx, dy, -d), scaleMat(0.04, 0.04, 0.04)),
            WHITE, therapyDotVao, gl.TRIANGLES, 6);
    } else if (thpAct === THP_BOTH) {
      // dichoptic: red mark to OD, green ring to OS — both must be seen
      if (rightEye)
        solid(mul(translationMat(0.10, 0, -1.4), scaleMat(0.05, 0.05, 0.05)),
              RED, therapyDotVao, gl.TRIANGLES, 6);
      else
        solid(mul(translationMat(-0.10, 0, -1.4), scaleMat(9, 9, 9)),
              GREEN, targetVao, gl.TRIANGLE_STRIP, TARGET_VERTS);
    } else if (thpAct === THP_STEREO) {
      // simplified stereo: a ring floated forward by a per-eye disparity that
      // shrinks with the arcsec rung (a true random-dot stereogram is a follow-up)
      const disp = eyeSign * (thpArcsec / 400) * 0.03;
      solid(mul(translationMat(disp, 0, -1.5), scaleMat(6, 6, 6)),
            WHITE, targetVao, gl.TRIANGLE_STRIP, TARGET_VERTS);
    } else if (thpAct === THP_CONTRAST) {
      // faint patch brightening against the dim room (binocular)
      const g = 0.18 + thpContrast;
      solid(mul(translationMat(0, 0, -1.5), scaleMat(0.12, 0.12, 0.12)),
            [g, g, g, 1], therapyDotVao, gl.TRIANGLES, 6);
    }
    for (const a of aimPoses) {
      gl.uniformMatrix4fv(locBeamMvp, false,
                          mul(vpWorld, poseMatrix(a.pos, a.quat)));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
    }
    gl.bindVertexArray(null);
  }

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
    const aX = edge(4), bY = edge(5), thumb = edge(3);
    if (testingPhase()) {
      if (aX) toggleBeams();
      if (bY) cyclePrism();
      if (thumb) toggleFilters();
    }
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
  advancePhase();
  updateTestDim();
  updateTherapyClock();
  updateInspection(pose.transform.orientation);
  updateCover();
  updateEye();
  updateWorth();
  updateMaddox();
  updateVg();
  updateDm();

  // controller aim rays -> hovered element on whichever panel is active
  aimPoses = [];
  clHit = null;
  thpHit = null;
  workflowHovered = -1;
  if (menuPhase() || (testingPhase() && testMode === 'select') ||
      (therapyPhase() && thpMode === 'select')) {
    for (const src of session.inputSources) {
      if (!src.targetRaySpace) continue;
      const rp = frame.getPose(src.targetRaySpace, xrRefSpace);
      if (!rp) continue;
      const a = {
        pos: rp.transform.position,
        quat: rp.transform.orientation,
        left: src.handedness === 'left',
      };
      aimPoses.push(a);
      if (menuPhase()) {
        const r = workflowHitRow(a.pos, a.quat);
        if (r >= 0 && (workflowHovered < 0 || !a.left)) workflowHovered = r;
      } else if (testingPhase()) {
        const h = checklistHit(a.pos, a.quat);
        if (h && (!clHit || !a.left)) clHit = h;
      } else if (therapyPhase()) {
        const h = therapyPanelHit(a.pos, a.quat);
        if (h && (!thpHit || !a.left)) thpHit = h;
      }
    }
  } else if (therapyPhase() && thpMode === 'run') {
    for (const src of session.inputSources) {  // keep the ray during a run
      if (!src.targetRaySpace) continue;
      const rp = frame.getPose(src.targetRaySpace, xrRefSpace);
      if (rp) aimPoses.push({ pos: rp.transform.position,
                              quat: rp.transform.orientation,
                              left: src.handedness === 'left' });
    }
  }

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

    // trigger is phase-routed: skip narration / pick a workflow / toggle a
    // checklist row (or lights/prism) / advance a therapy exercise
    xrSession.addEventListener('select', (ev) => {
      // a test-panel hit acts immediately, even over the intro narration, so
      // START starts the first test without waiting for commentary
      const panelHit = testingPhase() && testMode === 'select' && clHit;
      if (narrating() && !panelHit) { skipClip(); return; }
      if (menuPhase()) {
        if (workflowHovered >= 0) chooseWorkflow(workflowHovered);
        else if (playingClip >= 0) skipClip();
        return;
      }
      if (testingPhase()) {
        if (testMode === 'select') {
          if (clHit) {
            if (clHit.kind === 'check') {
              testSelected[clHit.row] = !testSelected[clHit.row];
              saveSelection();
            } else if (clHit.kind === 'talk') {
              playClip(CLIP_DESC0 + clHit.row);
            } else if (clHit.kind === 'start') {
              startRun();
            }
            return;
          }
          if (playingClip >= 0) { skipClip(); return; }
          if (ev.inputSource.handedness === 'left') cyclePrism();
          else toggleLights();
        } else if (worthActive) {
          // enumerate: press picks the named option; confirm: press locks it
          if (worthPhase === 1) {
            worthSelected = worthAsk; worthPhase = 2; worthAskSaid = false; worthT = 0;
          } else if (worthPhase === 2) {
            worthPhase = 3; worthAskSaid = false; worthT = 0;
          } else if (playingClip >= 0) skipClip();
        } else if (maddoxActive) {
          if (playingClip >= 0) skipClip();
          else if (maddoxStage === 0 && !maddoxVDone) {
            addEvidenceV(maddoxOffset(maddoxT) * 100, 1.0); maddoxVDone = true;
          } else if (maddoxStage === 1 && !maddoxHDone) {
            addEvidenceH(maddoxOffset(maddoxT) * 100, 1.0); maddoxHDone = true;
          }
        } else if (vgActive) {
          vgPress();
        } else if (dmActive) {
          dmPress();
        } else if (eyeActive) {
          eyeFlagged = true;  // self-report: "it doubled / I lost it"
          if (!eyeDoubleAck) { eyeDoubleAck = true; playClip(CLIP_EYE_DOUBLE); }
        } else {
          advanceRun();   // run mode: trigger advances to the next test
        }
        return;
      }
      if (therapyPhase()) therapyTrigger();
    });
    xrSession.addEventListener('squeeze', () => {
      if (testingPhase()) cyclePrism();
    });
    xrSession.addEventListener('end', () => {
      xrSession = null;
      requestAnimationFrame(onPreviewFrame);
    });
    startPhases(); // enterVR is a user gesture, so audio is allowed to sound
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

  advancePhase();
  updateTestDim();
  updateTherapyClock();
  const proj = perspective(80, w / h, 0.05, 100);
  const viewRot = mul(rotationX(-previewPitch), rotationY(-previewYaw));
  const quat = quatFromYawPitch(previewYaw, previewPitch);
  updateInspection(quat);
  updateCover();
  updateEye();
  updateWorth();
  updateMaddox();
  updateVg();
  updateDm();
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
  panelProgram = buildProgram(PANEL3D_VS, LABEL_FS);
  locPanelMvp = gl.getUniformLocation(panelProgram, 'uMvp');
  locPanelTex = gl.getUniformLocation(panelProgram, 'uLabel');
  locPanelFilter = gl.getUniformLocation(panelProgram, 'uFilter');
  cubeVao = buildCubeVao();
  quadVao = buildLabelQuad();
  beamVao = buildBeamVao();
  targetVao = buildTargetVao();
  crossVao = buildCrossVao();
  panelVao = buildPanelQuad();
  checklistPanelVao = buildChecklistPanelVao();
  checklistMarkVao = buildFilledQuadVao(CHECKLIST_BOX_HALF_U * CHECKLIST_W * 0.55);
  checklistCaretVao = buildCaretVao();
  workflowPanelVao = buildMenuPanelVao(WORKFLOW_W, WORKFLOW_H);
  therapyDotVao = buildFilledQuadVao(1.0);  // unit quad, scaled per target

  loadSelection();  // restore any remembered test un-checks
  loadThpSelection();

  setMessage('Loading skybox…');
  [texBright, texDim, texLabels, texDisclaimer, texChecklist, texWorkflow,
   texTitleCards, texThpChecklist, texThpTitle] = await Promise.all([
    loadCubemap('assets/skybox'),
    loadCubemap('assets/skybox_dim'),
    loadTexture2D('assets/prism_labels.png'),
    loadTexture2D('assets/disclaimer.png').catch(() => null),
    loadTexture2D('assets/checklist.png').catch(() => null),
    loadTexture2D('assets/workflows.png').catch(() => null),
    loadTexture2D('assets/titlecards.png').catch(() => null),
    loadTexture2D('assets/checklist_therapy.png').catch(() => null),
    loadTexture2D('assets/titlecards_therapy.png').catch(() => null),
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
    startPhases(); // first gesture in preview lets the narration sound
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
    if (e.key === ' ') {
      if (narrating()) { skipClip(); return; }
      if (therapyPhase()) { therapyTrigger(); return; }
      if (testingPhase() && testMode === 'run') {
        if (worthActive) {
          if (worthPhase === 1) {
            worthSelected = worthAsk; worthPhase = 2; worthAskSaid = false; worthT = 0;
          } else if (worthPhase === 2) {
            worthPhase = 3; worthAskSaid = false; worthT = 0;
          } else if (playingClip >= 0) skipClip();
        } else if (maddoxActive) {
          if (playingClip >= 0) skipClip();
          else if (maddoxStage === 0 && !maddoxVDone) {
            addEvidenceV(maddoxOffset(maddoxT) * 100, 1.0); maddoxVDone = true;
          } else if (maddoxStage === 1 && !maddoxHDone) {
            addEvidenceH(maddoxOffset(maddoxT) * 100, 1.0); maddoxHDone = true;
          }
        } else if (vgActive) {
          vgPress();
        } else if (dmActive) {
          dmPress();
        } else if (eyeActive) {
          eyeFlagged = true;
          if (!eyeDoubleAck) { eyeDoubleAck = true; playClip(CLIP_EYE_DOUBLE); }
        } else advanceRun();
        return;
      }
    }
    if (menuPhase()) {
      if (e.key === '1') { chooseWorkflow(0); return; }
      if (e.key === '2') { chooseWorkflow(1); return; }
    }
    if (e.key === 'Enter' && therapyPhase() && thpMode === 'select') {
      thpStartRun();
      return;
    }
    if (!testingPhase()) return;  // remaining keys are testing controls
    if (testMode === 'select' && e.key === 'Enter') { startRun(); return; }
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
