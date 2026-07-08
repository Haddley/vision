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

// Per-eye prism from an explicit prescription (vertical + horizontal Δ),
// same convention as prismRotation (OS mirrors OD). Per-person therapy prism.
function prismRotationPD(rightEye, pdV, pdH) {
  const sign = rightEye ? 1 : -1;
  return mul(rotationX(sign * Math.atan(pdV / 100)),
             rotationY(sign * Math.atan(pdH / 100)));
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
uniform float uMono;  // 1 in anaglyph mode: luminance so no stimulus vanishes
out vec4 outColor;
void main() {
  vec3 c = texture(uSky, vDir).rgb * uFilter;
  c = mix(c, vec3(max(c.r, max(c.g, c.b))), uMono);
  outColor = vec4(c, 1.0);
}`;

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
uniform float uMono;
out vec4 outColor;
void main() {
  vec3 c = texture(uLabel, vUV).rgb * uFilter;
  c = mix(c, vec3(max(c.r, max(c.g, c.b))), uMono);
  outColor = vec4(c, texture(uLabel, vUV).a);
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
uniform float uMono;
out vec4 outColor;
void main() {
  vec3 c = uColor.rgb * uFilter;
  c = mix(c, vec3(max(c.r, max(c.g, c.b))), uMono);
  outColor = vec4(c, uColor.a);
}`;

// Lit (Lambert) program for the 3-D Brock string. Lighting is in the string's
// local frame with a fixed light direction, so the shading is stable as the
// head moves and the cord's twist + the bead's bore read.
const LIT_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uMvp;
out vec3 vN;
out vec3 vPos;
void main() {
  gl_Position = uMvp * vec4(aPos, 1.0);
  vN = aNormal;
  vPos = aPos;
}`;

// Glossy: diffuse + Blinn-Phong specular + hemisphere ambient, in the string's
// local frame (uEyePos is the eye transformed into that frame). Matches native.
const LIT_FS = `#version 300 es
precision mediump float;
in vec3 vN;
in vec3 vPos;
uniform vec3 uColor;
uniform vec3 uLightDir;
uniform vec3 uEyePos;
uniform vec3 uSky;
uniform vec3 uGround;
uniform float uShininess;
uniform float uSpec;
uniform float uMono;
out vec4 outColor;
void main() {
  vec3 N = normalize(vN);
  vec3 L = normalize(uLightDir);
  float diff = max(dot(N, L), 0.0);
  vec3 V = normalize(uEyePos - vPos);
  vec3 H = normalize(L + V);
  float spec = uSpec * pow(max(dot(N, H), 0.0), uShininess);
  vec3 hemi = mix(uGround, uSky, 0.5 * (N.y + 1.0));
  vec3 col = uColor * (hemi + vec3(diff)) + vec3(spec);
  vec3 cM = col;
  cM = mix(cM, vec3(max(cM.r, max(cM.g, cM.b))), uMono);
  outColor = vec4(cM, 1.0);
}`;

// monospace bitmap-font text: a unit quad placed per glyph, its UV picking a
// cell of font_atlas.png (white-on-black, luminance = coverage = alpha).
const TEXT_VS = `#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;
uniform mat4 uMvp;
uniform vec4 uUvRect;
out vec2 vUV;
void main() {
  gl_Position = uMvp * vec4(aPos, 1.0);
  vUV = uUvRect.xy + aUV * uUvRect.zw;
}`;
const TEXT_FS = `#version 300 es
precision mediump float;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3 uColor;
uniform float uMono;
out vec4 outColor;
void main() {
  vec3 c = mix(uColor, vec3(max(uColor.r, max(uColor.g, uColor.b))), uMono);
  outColor = vec4(c, texture(uTex, vUV).r);
}`;

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
let litProgram, locLitMvp, locLitColor, locLitLight, locLitEyePos, locLitSky,
    locLitGround, locLitShininess, locLitSpec;
let beadMesh, cordMesh;  // { vao, count } for the Brock string
let panelProgram, locPanelMvp, locPanelTex, locPanelFilter;
let cubeVao, quadVao, beamVao, targetVao, crossVao, panelVao;
let checklistPanelVao, checklistMarkVao, checklistCaretVao;
let workflowPanelVao, therapyDotVao;
let texBright, texDim, texLabels, texDisclaimer, texChecklist, texWorkflow;
let texTitleCards, texThpChecklist, texThpTitle, texGmChecklist;
let textProgram, locTextMvp, locTextUvRect, locTextColor, locTextTex;
let textQuadVao, texFont;
// session results record + on-screen summary panel
let sessionLines = [];
let summaryActive = false;
// player profiles (local only): the active name scopes prefs + session records
let profiles = [];
let activeProfile = 'Guest';
// Prism prescriptions quantized to 0.25 Δ; default = the forced Rx (OD 5.5
// base-down + 1.0 base-out, OS 5.5 base-up + 1.0 base-out).
const q25 = x => Math.round(x * 4) / 4;
let profPrismV = PRISM_VERTICAL_PD, profPrismH = PRISM_HORIZONTAL_PD;  // per-person prism prescription (Δ)
let prismHit = null;                 // hovered prism-panel element
let prismBtnHit = false;             // "Adjust prism" chooser button hover
let personBtnHit = false;            // "Switch person" chooser button hover
let profPrismSaved = false;          // active person had a saved prescription
let profPrismKnown = false;          // prescription set (else shown "not set")
let pendingDelete = -1;    // a delete tap arms this row; a second tap acts
let kbActive = false;      // the new-name virtual keyboard is up
let newName = '';          // the name being typed

let lightsOn = true;
let prismStep = 2; // start with the prism off (PRISM_STEPS[2] === 0)
let prismScale = PRISM_STEPS[2];
let beamsVisible = false;      // gaze beams + chart targets (red OD, green OS)
let filtersOn = false;         // red/green Worth 4-dot test filters
// The 'select' handler, exposed so the gamepad A button can invoke it directly
// (A = primary interact; X = toggle beams — mirrors the OpenXR rebind). Quest
// Touch face button (A/X) is gamepad button index 4. select/pinch stays a
// fallback for non-Quest WebXR.
let doSelect = null;
let refRebased = false;  // 'local' origin rebased to the first viewer pose

// ---- anaglyph laptop mode (colored-lens glasses on a flat screen) ----
// Per-eye colour-channel masks, mirroring vlogic::anaglyphMasks (tested
// there): 6 presets = {red-cyan, red-blue, red-green} x {red-left, red-right}.
// There is no universal convention (consumer 3D: red LEFT; clinical Worth:
// red RIGHT), so the setup screen lets the wearer verify + swap.
let anaglyph = false;
let sbs = false;  // 3D TV mode (spatial per-eye, like a headset)
let tbFormat = false;  // 3D TV split: false = side-by-side, true = top-bottom
let anaPreset = 0;  // ANA_MASKS index; persisted to localStorage
const ANA_MASKS = [
  [[1, 0, 0], [0, 1, 1]],  // red-cyan,  red LEFT (consumer default)
  [[0, 1, 1], [1, 0, 0]],  // red-cyan,  red RIGHT
  [[1, 0, 0], [0, 0, 1]],  // red-blue,  red LEFT
  [[0, 0, 1], [1, 0, 0]],  // red-blue,  red RIGHT
  [[1, 0, 0], [0, 1, 0]],  // red-green, red LEFT
  [[0, 1, 0], [1, 0, 0]],  // red-green, red RIGHT (clinical Worth colours)
];
// human colour names per preset ([left lens, right lens]) for dynamic hints
const ANA_NAMES = [
  ['red', 'cyan'], ['cyan', 'red'], ['red', 'blue'],
  ['blue', 'red'], ['red', 'green'], ['green', 'red'],
];
let anaWorthMsg = false;  // Worth colour hint currently shown in the overlay
let mouseX = -1, mouseY = -1;  // pointer position for the preview aim ray
const gpPrev = { a: false, x: false };
let aimPoses = [];             // controller/hand target-ray poses this frame

// test-selection workflow: pick tests (checkbox), hear a description (Talk),
// then START runs the selected tests one at a time. All selected by default;
// un-checks persist to localStorage.
let testSelected = [true, true, true, true, true, true, true];
let testMode = 'select';       // 'select' | 'run'
let runList = [], runIdx = 0;
let clHit = null;              // hovered panel element this frame {kind,row}
let thpHit = null;             // hovered therapy-panel element {kind,row}
let profHit = null;            // hovered profile-panel element {kind,row}
let kbHit = null;              // hovered virtual-keyboard cell {col,row}
// Initial inspection assessment (head tilt only in WebXR — no browser gaze)
let inspActive = false, inspStage = 0, inspT = 0;
let inspRollSum = 0, inspRollN = 0, inspTilted = false;
// moving-H subjective field of single vision: which of the 9 positions the
// subject reported the H doubled at (gaze-free — the web can't measure the
// per-eye deviation the native build does, but the diplopia field is subjective
// so it works identically).
let inspDip = new Array(9).fill(false);
let inspResultPanel = false;    // show the results card after the report, wait
let inspFlashFrames = 0;        // brief H colour flash confirming a press
let inspLiveRoll = 0;           // live head roll (deg) for the spirit level
let lastInspRec = null;         // {tilt, dip:[9]} of the most recent run
const INSP_DIRNM = ['CENTRE', 'UP', 'UP-RIGHT', 'RIGHT', 'DOWN-RIGHT',
                    'DOWN', 'DOWN-LEFT', 'LEFT', 'UP-LEFT'];
// step-and-hold sweep: 1.3s dwell + 0.5s move per position, 9 positions.
function inspStepJS(t) {
  const per = 1.3 + 0.5;
  const idx = Math.floor(t / per);
  if (idx >= 9) return { index: 9, dwelling: false };
  return { index: idx, dwelling: (t - idx * per) < 1.3 };
}
function inspSweepXY(t) {  // H centre (display coords), squeezed into bounds
  const per = 1.3 + 0.5;
  const idx = Math.floor(t / per);
  // done: the H is back at CENTRE for the spoken report (waypoint 9 is the
  // return-home leg — mirrors vlogic::inspHoldIndex on native)
  if (idx >= 9) return [0, 0.15];
  const nxt = Math.min(idx + 1, 9);  // last slot glides home via waypoint 9
  let a = (t - idx * per) < 1.3 ? 0 : ((t - idx * per) - 1.3) / 0.5;
  a = a * a * (3 - 2 * a);  // smoothstep during the move
  const x = EYE_TARGETS[idx][0] + (EYE_TARGETS[nxt][0] - EYE_TARGETS[idx][0]) * a;
  const y = EYE_TARGETS[idx][1] + (EYE_TARGETS[nxt][1] - EYE_TARGETS[idx][1]) * a;
  return [x * 0.90, 0.15 + (y - 0.15) * 0.62];
}
// Cover test: occlude each eye in turn (visual only in WebXR — no gaze to
// measure the drift; shown for demonstration + spoken explanation)
let coverActive = false, coverStage = 0, coverT = 0, coverLast = 0;
// Eye movement test: follow the moving light (visual only in WebXR — no gaze
// to measure tracking or calibrate; self-report + demonstration)
let eyeActive = false, eyeStage = 0, eyeT = 0, eyeLast = 0, eyeFlagged = false;
let eyeDoubleAck = false;         // (legacy; unused by the subjective test)
// subjective motility: per-position diplopia field (press where it doubles /
// you lose it during the smooth pursuit), + a brief press-confirm flash.
let eyeDip = new Array(9).fill(false);
let eyeFlash = 0;
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
// "Show me the prism" verification: after a run finds a confident estimate,
// re-show the H and animate the found prism onto it (per eye) so the two
// images merge. pvStage: 0 intro, 1 ramp, 2 hold, 3 done.
let pvActive = false, pvStage = 0, pvT = 0, pvLast = 0;
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
// "Show me the prism" verification (end of a testing run), appended
const CLIP_PV_INTRO = 80, CLIP_PV_RESULT = 81, CLIP_PV_NONE = 82;
const CLIP_INSP_SELFREPORT = 83;  // static-H "if you see two, pull the trigger"
// Inspection results summary: overall trend verdict + worst direction. CLIP_DIR
// order follows vlogic::kInspDir (centre, up, up-right, right, down-right, down,
// down-left, left, up-left).
const CLIP_INSP_SUM_STABLE = 84, CLIP_INSP_SUM_BETTER = 85,
      CLIP_INSP_SUM_WORSE = 86, CLIP_INSP_SUM_WORST = 87;
const CLIP_DIR0 = 88;  // CLIP_DIR0..CLIP_DIR0+8
const CLIP_INTRO_GAMES = 97, CLIP_DESC_FLAPPY = 98;  // Vision Games
const INSP_DIR = [0, 3, 2, 1, 8, 7, 6, 5, 4];  // EYE_TARGETS idx -> dir clip
const thpDescClip = r => CLIP_THP_DESC0 + r;
const thpLookClip = r => CLIP_THP_LOOK0 + r;
let audioCtx = null;
let introBuffers = [];            // indexed by the CLIP_* constants above
let introSource = null;
let audioOk = false;
let playingClip = -1;             // buffer index currently sounding, -1 = silence
let phaseStarted = false;         // first clip kicked off on the opening gesture
// phase: 'welcome','disclaimer','choose','select','intro_test','testing',
//        'intro_ther','therapy','intro_games','games'
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
let therapyPrismOn = true;        // apply the person's prism during therapy
let thpPrismToggleHit = false;    // therapy prism On/Off toggle hover
let thpRunList = [], thpRunIdx = 0;
let thpAct = -1, thpStage = 0, thpSaid = false, thpCycle = 0;
let thpBeadZ = 0.30, thpBeadDir = -1;
let thpPrismH = 0, thpPrismV = 0;
let thpAccA = 0, thpAccB = 0, thpNa = 0, thpNb = 0, thpSeen = 0;
let thpArcsec = 400, thpContrast = 0, thpOnset = 0;

// ---- Vision Games (dichoptic amblyopia games) ----
// Per-profile amblyopic eye (mirror of the prism-Rx storage): 0 none,1 OD,2 OS.
const AMBLY_NONE = 0, AMBLY_OD = 1, AMBLY_OS = 2;
let amblyEye = AMBLY_NONE, amblyKnown = false;
const GAME_DICHOPTIC = 0, GAME_MONOCULAR = 1;
let gameMode = GAME_DICHOPTIC;      // per-session play mode
let gmMode = 'select';              // 'select' | 'run'
let gameSelected = [true];          // one game (Flappy) for now
let gmHit = null;                   // hovered games-panel element this frame
let gmTalking = false;
// games-panel layout (mirrors kGame*/kChecklist* in vision_logic.h)
const GAME_ROW0_V = 0.22, GAME_EYE_V = 0.42, GAME_MODE_V = 0.55;
// Flappy Bird — hand-port of vlogic::flappy* (deterministic physics)
const FLAPPY_GRAVITY = -1.6, FLAPPY_FLAP = 0.62, FLAPPY_MAXFALL = -1.2;
const FLAPPY_BIRDX = 0.30;
const FLAPPY_GAPTBL = [0.50, 0.66, 0.40, 0.72, 0.34, 0.58];
let flappy = { birdY: 0.5, birdV: 0, score: 0, dead: false };
let flappyScroll = 0;               // pipe scroll position (pipe-widths)
let flappyPending = false;          // a flap press awaiting the next update
let flappyRecorded = false;         // score written for the current life
let gameSessions = 0;               // this-session play count (contrast ramp)
function flappyGapCenter(i) { return FLAPPY_GAPTBL[((i % 6) + 6) % 6]; }
function flappyGapHalf(logMAR, level) {
  let b = 0.15 + 0.05 * logMAR + 0.03 * level;
  return Math.max(0.09, Math.min(0.26, b));
}
function flappyFellowContrast(n) {
  return Math.min(0.9, 0.15 + 0.05 * Math.max(0, n));
}
function amblyEyeText() {
  if (!amblyKnown || amblyEye === AMBLY_NONE) return 'not set';
  return amblyEye === AMBLY_OD ? 'Right (OD)' : 'Left (OS)';
}
function gameModeText() {
  return gameMode === GAME_MONOCULAR ? 'Monocular' : 'Dichoptic';
}
function flappyInit() {
  flappy = { birdY: 0.5, birdV: 0, score: 0, dead: false };
  flappyScroll = 0; flappyPending = false; flappyRecorded = false;
}

function narrating() {
  return playingClip >= 0 &&
         (phase === 'welcome' || phase === 'disclaimer' ||
          phase === 'intro_test' || phase === 'intro_ther' ||
          phase === 'intro_games');
}
function menuPhase() { return phase === 'choose' || phase === 'select'; }
function testingPhase() {
  return phase === 'intro_test' || phase === 'testing';
}
function therapyPhase() {
  return phase === 'intro_ther' || phase === 'therapy';
}
function gamesPhase() {
  return phase === 'intro_games' || phase === 'games';
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

// ---- Realistic Brock string geometry (lit, pos+normal interleaved) --------
// The hole axis of every bead and the string run along -Z; positions are in
// metres in the string's local frame.

// Upload an interleaved (pos.xyz, normal.xyz) triangle list; attribs 0,1.
function litVao(arr) {
  const data = new Float32Array(arr);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  gl.bindVertexArray(null);
  return { vao, count: arr.length / 6 };
}

// Revolve a profile (list of {r,z,nr,nz}) around the Z axis into a triangle
// list with normals, appended to out.
function revolveInto(prof, seg, out) {
  const TAU = 6.28318530718;
  const emit = (p, c, s) => {
    out.push(p.r * c, p.r * s, p.z, p.nr * c, p.nr * s, p.nz);
  };
  for (let k = 0; k + 1 < prof.length; ++k) {
    const a = prof[k], b = prof[k + 1];
    for (let j = 0; j < seg; ++j) {
      const c0 = Math.cos((TAU * j) / seg), s0 = Math.sin((TAU * j) / seg);
      const c1 = Math.cos((TAU * (j + 1)) / seg),
            s1 = Math.sin((TAU * (j + 1)) / seg);
      emit(a, c0, s0); emit(b, c0, s0); emit(b, c1, s1);
      emit(a, c0, s0); emit(b, c1, s1); emit(a, c1, s1);
    }
  }
}

// A drilled wooden bead: a sphere of `radius` with a cylindrical bore of
// `bore` radius through it along Z (outer sphere surface + inner bore wall).
function buildBeadVao(radius, bore) {
  const PI = Math.PI;
  const rimA = Math.asin(bore / radius);
  const zRim = Math.sqrt(radius * radius - bore * bore);
  const sphere = [], boreWall = [];
  const Ns = 28;
  for (let k = 0; k <= Ns; ++k) {
    const phi = rimA + ((PI - 2 * rimA) * k) / Ns;
    sphere.push({ r: radius * Math.sin(phi), z: radius * Math.cos(phi),
                  nr: Math.sin(phi), nz: Math.cos(phi) });
  }
  const Nb = 4;
  for (let k = 0; k <= Nb; ++k) {
    const z = -zRim + (2 * zRim * k) / Nb;
    boreWall.push({ r: bore, z, nr: -1, nz: 0 });
  }
  const v = [];
  revolveInto(sphere, 24, v);
  revolveInto(boreWall, 24, v);
  return litVao(v);
}

// The twisted cord: a straight core plus two helical strands, so it reads as
// two-ply twine up close. Runs from z=0 to z=-length.
function buildCordVao(length) {
  const TAU = 6.28318530718;
  const coreR = 0.0006, twistR = 0.0007, strandR = 0.0008, pitch = 0.004;
  const v = [];
  revolveInto([{ r: coreR, z: 0, nr: 1, nz: 0 },
               { r: coreR, z: -length, nr: 1, nz: 0 }], 8, v);
  const turns = length / pitch;
  const rings = Math.max(2, Math.floor(turns * 6));
  const sides = 5;
  for (let strand = 0; strand < 2; ++strand) {
    const phase = strand * Math.PI;
    const ringPos = [], ringNrm = [];
    for (let ri = 0; ri <= rings; ++ri) {
      const t = ri / rings;
      const ang = TAU * turns * t + phase;
      const cx = twistR * Math.cos(ang), cy = twistR * Math.sin(ang),
            cz = -length * t;
      let tx = -twistR * TAU * turns * Math.sin(ang);
      let ty = twistR * TAU * turns * Math.cos(ang);
      let tz = -length;
      const tl = Math.hypot(tx, ty, tz); tx /= tl; ty /= tl; tz /= tl;
      let ux = 0, uy = 0, uz = 1;
      if (Math.abs(tz) > 0.9) { ux = 1; uz = 0; }
      let nx = uy * tz - uz * ty, ny = uz * tx - ux * tz, nz = ux * ty - uy * tx;
      const nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
      const bx = ty * nz - tz * ny, by = tz * nx - tx * nz, bz = tx * ny - ty * nx;
      const pr = [], nr2 = [];
      for (let si = 0; si <= sides; ++si) {
        const a = (TAU * si) / sides;
        const ca = Math.cos(a), sa = Math.sin(a);
        const dx = ca * nx + sa * bx, dy = ca * ny + sa * by, dz = ca * nz + sa * bz;
        pr.push(cx + strandR * dx, cy + strandR * dy, cz + strandR * dz);
        nr2.push(dx, dy, dz);
      }
      ringPos.push(pr); ringNrm.push(nr2);
    }
    const pv = (ri, si) => {
      v.push(ringPos[ri][si * 3], ringPos[ri][si * 3 + 1], ringPos[ri][si * 3 + 2],
             ringNrm[ri][si * 3], ringNrm[ri][si * 3 + 1], ringNrm[ri][si * 3 + 2]);
    };
    for (let ri = 0; ri < rings; ++ri)
      for (let si = 0; si < sides; ++si) {
        pv(ri, si); pv(ri + 1, si); pv(ri + 1, si + 1);
        pv(ri, si); pv(ri + 1, si + 1); pv(ri, si + 1);
      }
  }
  return litVao(v);
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
// the "Menu" back button, top-left of both select panels (normalized u,v)
function inBackZone(u, v) {
  return u >= 0.02 && u <= 0.22 && v >= 0.035 && v <= 0.11;
}

// ---- player-profile panel + virtual keyboard (mirror native) ----
const PROFILE_DIST = CHECKLIST_DIST;
const PROFILE_W = 1.28, PROFILE_H = 1.08;
const PROFILE_ROW0V = 0.26, PROFILE_ROWDV = 0.105, PROFILE_MAXROWS = 6;
function profilePux(u) { return (u - 0.5) * PROFILE_W; }
function profilePuy(v) { return (0.5 - v) * PROFILE_H; }
// hit a name row (right ~20% = delete), or the trailing "+ New Player" row
function profilePanelHit(pos, q, nRows) {
  const uv = menuHitUV(pos, q, PROFILE_DIST, PROFILE_W, PROFILE_H);
  if (!uv) return null;
  for (let i = 0; i < nRows + 1; ++i)
    if (Math.abs(uv.v - (PROFILE_ROW0V + i * PROFILE_ROWDV)) <= PROFILE_ROWDV / 2) {
      if (i === nRows) return { kind: 'new', row: -1 };
      return { kind: uv.u >= 0.80 ? 'delete' : 'select', row: i };
    }
  return null;
}
// virtual keyboard: 3 rows x 10 cols. Codes: letter chars, ' ', 8 backspace,
// '\n' OK (10), 27 cancel. 0 = empty cell.
function kbKeyAt(col, row) {
  if (col < 0 || col > 9 || row < 0 || row > 2) return 0;
  if (row === 0) return 65 + col;          // A..J
  if (row === 1) return 75 + col;          // K..T
  if (col <= 5) return 85 + col;           // U..Z
  if (col === 6) return 32;                // space
  if (col === 7) return 8;                 // backspace
  if (col === 8) return 10;                // OK
  return 27;                               // cancel
}
const KB_U0 = 0.06, KB_U1 = 0.94, KB_V0 = 0.40, KB_V1 = 0.92;
function keyboardHit(pos, q) {
  const uv = menuHitUV(pos, q, PROFILE_DIST, PROFILE_W, PROFILE_H);
  if (!uv || uv.u < KB_U0 || uv.u > KB_U1 || uv.v < KB_V0 || uv.v > KB_V1)
    return null;
  let col = Math.floor((uv.u - KB_U0) / ((KB_U1 - KB_U0) / 10));
  let row = Math.floor((uv.v - KB_V0) / ((KB_V1 - KB_V0) / 3));
  col = Math.max(0, Math.min(9, col));
  row = Math.max(0, Math.min(2, row));
  return { col, row };
}

// Prism-prescription panel (shares the profile card): Vertical row (v 0.34),
// Horizontal row (v 0.48), Done (v 0.72). minus u∈[0.46,0.64], plus u∈[0.78,0.96].
function prismPanelHit(pos, q) {
  const uv = menuHitUV(pos, q, PROFILE_DIST, PROFILE_W, PROFILE_H);
  if (!uv) return null;
  const minus = uv.u >= 0.46 && uv.u <= 0.64;
  const plus = uv.u >= 0.78 && uv.u <= 0.96;
  if (Math.abs(uv.v - 0.34) <= 0.06) {
    if (minus) return 'vdn';
    if (plus) return 'vup';
  }
  if (Math.abs(uv.v - 0.48) <= 0.06) {
    if (minus) return 'hdn';
    if (plus) return 'hup';
  }
  if (Math.abs(uv.v - 0.72) <= 0.06 && uv.u >= 0.28 && uv.u <= 0.72) return 'done';
  return null;
}

function checklistHit(pos, q) {
  const uv = menuHitUV(pos, q, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
  if (!uv) return null;
  if (inBackZone(uv.u, uv.v)) return { kind: 'back', row: -1 };
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
  if (inBackZone(uv.u, uv.v)) return { kind: 'back', row: -1 };
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
// Rows: 0 Vision Testing, 1 Vision Therapy, 2 Quit (ends the XR session).
const WORKFLOW_ROWS = 4;          // Testing / Therapy / Games / Quit
const WORKFLOW_QUIT_ROW = 3;      // Quit stays last
const WORKFLOW_ROW0_V = 0.320;    // match vlogic + build_workflow_menu
const WORKFLOW_ROW_DV = 0.160;
const WORKFLOW_DIST = 2.0;
const WORKFLOW_W = 1.30;
const WORKFLOW_H = WORKFLOW_W * 860 / 1024;
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

// unit quad [0,1]x[0,1] with matching UVs, placed per glyph by the text MVP
const FONT_COLS = 16, FONT_ROWS = 6;  // font_atlas.png grid (chars 32..126)
function buildTextQuad() {
  const v = new Float32Array([
    0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1,
  ]);
  const vao = gl.createVertexArray();
  const vbo = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
  gl.bindVertexArray(null);
  return vao;
}

// Monospace text at a world-anchored panel: pen (x,y) top-left, glyphs cw wide
// advancing right, ch tall descending; '\n' wraps. Colors r,g,b (alpha=atlas).
function drawText(vpWorld, x, y, cw, ch, r, g, b, s) {
  gl.useProgram(textProgram);
  gl.uniform3f(locTextColor, r, g, b);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texFont);
  gl.uniform1i(locTextTex, 0);
  gl.bindVertexArray(textQuadVao);
  let penX = x, penY = y;
  for (let k = 0; k < s.length; ++k) {
    const c = s.charCodeAt(k);
    if (c === 10) { penX = x; penY -= ch * 1.25; continue; }
    const idx = c - 32;
    if (idx >= 0 && idx < 95) {
      const col = idx % FONT_COLS, rowc = Math.floor(idx / FONT_COLS);
      gl.uniform4f(locTextUvRect, col / FONT_COLS, rowc / FONT_ROWS,
                   1 / FONT_COLS, 1 / FONT_ROWS);
      gl.uniformMatrix4fv(locTextMvp, false,
          mul(vpWorld, mul(translationMat(penX, penY, -CHECKLIST_DIST + 0.01),
                           scaleMat(cw, -ch, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    penX += cw;
  }
  gl.bindVertexArray(null);
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
const TEST_NAMES = ['Ocular inspection', 'Cover test', 'Worth 4-Dot',
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
      vgActive || dmActive || pvActive) return;
  prismStep = (prismStep + 1) % PRISM_STEPS.length;
  prismScale = PRISM_STEPS[prismStep];
  updateStatus();
}

function nudgePrism(delta) {
  if (inspActive || coverActive || eyeActive || worthActive || maddoxActive ||
      vgActive || dmActive || pvActive) return;
  prismScale = Math.min(2, Math.max(0, prismScale + delta));
  updateStatus();
}

function toggleBeams() {
  beamsVisible = !beamsVisible;
  updateStatus();
}

function toggleFilters() {
  if (inspActive || coverActive || eyeActive || worthActive || maddoxActive ||
      vgActive || dmActive || pvActive) return;
  filtersOn = !filtersOn;
  updateStatus();
}

// ---- test selection + run flow (mirrors native main.cpp) ----
// per-profile localStorage keys (mirror native's test_prefs_<slug>.txt files)
function profileSlug(name) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').slice(0, 24) || 'guest';
}
function testKey() { return 'vision.testSelection.' + profileSlug(activeProfile); }
function thpKey() { return 'vision.therapySelection.' + profileSlug(activeProfile); }
function loadSelection() {
  try {
    const s = localStorage.getItem(testKey());
    if (s && s.length >= CHECKLIST_ROWS)
      for (let i = 0; i < CHECKLIST_ROWS; ++i)
        if (s[i] === '0' || s[i] === '1') testSelected[i] = s[i] === '1';
  } catch (e) { /* private mode / disabled storage */ }
}
function saveSelection() {
  try {
    localStorage.setItem(testKey(),
                         testSelected.map((b) => (b ? '1' : '0')).join(''));
  } catch (e) { /* ignore */ }
}
function loadThpSelection() {
  try {
    const s = localStorage.getItem(thpKey());
    if (s && s.length >= THP_ROWS)
      for (let i = 0; i < THP_ROWS; ++i)
        if (s[i] === '0' || s[i] === '1') thpSelected[i] = s[i] === '1';
  } catch (e) { /* ignore */ }
}
function saveThpSelection() {
  try {
    localStorage.setItem(thpKey(),
                         thpSelected.map((b) => (b ? '1' : '0')).join(''));
  } catch (e) { /* ignore */ }
}
// ---- player profiles + session results record (local only) ----
function loadProfiles() {
  profiles = [];
  try {
    const s = localStorage.getItem('vision.profiles');
    if (s) profiles = JSON.parse(s).filter((n) => typeof n === 'string' && n);
  } catch (e) { profiles = []; }
}
function saveProfiles() {
  try { localStorage.setItem('vision.profiles', JSON.stringify(profiles)); }
  catch (e) { /* ignore */ }
}
// Per-person prism prescription (signed Δ): vertical (base-up OD / down OS
// split) + horizontal (base-out). Persisted per profile in localStorage.
function prismKey(name) { return 'vision.prism.' + profileSlug(name); }
function loadPrism(name) {
  // a person's prism starts UNKNOWN (never defaulted): 0/0 applied
  // internally, shown as "not set" until set by the person or a test result
  profPrismV = 0; profPrismH = 0;
  profPrismKnown = false;
  profPrismSaved = false;
  try {
    const s = localStorage.getItem(prismKey(name));
    profPrismSaved = !!s;  // returning person: skip the prism stop
    if (s) {
      const p = s.split(/\s+/).map(Number);
      if (p.length === 2 && isFinite(p[0]) && isFinite(p[1])) {
        profPrismV = p[0]; profPrismH = p[1];
        profPrismKnown = true;  // anything else (e.g. "unset") stays unknown
      }
    }
  } catch (e) { /* ignore */ }
  profPrismV = q25(profPrismV); profPrismH = q25(profPrismH);
}
function savePrism() {
  try {
    localStorage.setItem(prismKey(activeProfile),
        profPrismKnown ? (profPrismV.toFixed(2) + ' ' + profPrismH.toFixed(2))
                       : 'unset');  // seen before, prism still unknown
  } catch (e) { /* ignore */ }
}
// Per-person amblyopic eye (which eye the games train): starts UNKNOWN,
// stored like the prism prescription. 'OD'/'OS' or 'unset'.
function amblyKey(name) { return 'vision.amblyeye.' + profileSlug(name); }
function loadAmblyEye(name) {
  amblyEye = AMBLY_NONE; amblyKnown = false;
  try {
    const s = localStorage.getItem(amblyKey(name));
    if (s === 'OD') { amblyEye = AMBLY_OD; amblyKnown = true; }
    else if (s === 'OS') { amblyEye = AMBLY_OS; amblyKnown = true; }
  } catch (e) { /* ignore */ }
}
function saveAmblyEye() {
  try {
    localStorage.setItem(amblyKey(activeProfile),
        amblyKnown && amblyEye === AMBLY_OD ? 'OD'
        : amblyKnown && amblyEye === AMBLY_OS ? 'OS' : 'unset');
  } catch (e) { /* ignore */ }
}
function scopeProfile(name) {
  activeProfile = name;
  for (let i = 0; i < CHECKLIST_ROWS; ++i) testSelected[i] = true;
  for (let i = 0; i < THP_ROWS; ++i) thpSelected[i] = true;
  loadSelection();
  loadThpSelection();
  loadPrism(name);
  loadAmblyEye(name);
}
function recordResult(label, value) { sessionLines.push(label + ': ' + value); }
function beginSession(workflow) {
  sessionLines = [];
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  sessionLines.push(workflow + '  ' + d.getFullYear() + '-' +
                    p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' ' +
                    p(d.getHours()) + ':' + p(d.getMinutes()));
  sessionLines.push('Person: ' + activeProfile);
}
function writeSession() {
  if (sessionLines.length <= 2) return;  // header + player -> nothing measured
  const text = sessionLines.join('\n') + '\n';
  try {
    const key = 'vision.session.' + profileSlug(activeProfile) + '.' + Date.now();
    localStorage.setItem(key, text);
    // keep a rolling log of the most recent sessions
    const log = (localStorage.getItem('vision.sessions.log') || '') + text + '----\n';
    localStorage.setItem('vision.sessions.log', log);
  } catch (e) { /* ignore */ }
  // offer a download of this session record
  try {
    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'session_' + profileSlug(activeProfile) + '_' + Date.now() + '.txt';
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  } catch (e) { /* headless / no DOM */ }
}
const THP_REC_NAMES = [
  'Convergence near pt', 'Divergence range', 'Divergence jumps',
  'Prism stress', 'Vertical fusion', 'Sustained vergence',
  'Both-eyes check', 'Stereo acuity', 'Contrast sensitivity'];
function recordThp() {
  if (thpAct < 0 || thpAct >= 9) return;
  let tv;
  if (thpAct === THP_CNP || thpAct === THP_DIVRANGE)
    tv = 'break ' + (thpNa ? (thpAccA / thpNa).toFixed(0) : '0') + 'cm / recover ' +
         (thpNb ? (thpAccB / thpNb).toFixed(0) : '0') + 'cm';
  else if (thpAct === THP_PRISM)
    tv = 'break ' + thpAccA.toFixed(1) + 'D / recover ' + thpAccB.toFixed(1) + 'D';
  else if (thpAct === THP_STEREO) tv = 'finest ' + thpArcsec.toFixed(0) + ' arcsec';
  else if (thpAct === THP_CONTRAST) tv = (thpAccA * 100).toFixed(0) + '% Weber';
  else if (thpAct === THP_BOTH) tv = 'seen ' + thpSeen + '/3';
  else tv = thpSeen + ' responses';
  recordResult(THP_REC_NAMES[thpAct], tv);
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
  beginSession('Vision Therapy');
  activateThp(0);
}
function thpAdvance() {  // called when the current activity finishes
  if (++thpRunIdx >= thpRunList.length) {
    writeSession();
    summaryActive = sessionLines.length > 2;
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
  inspResultPanel = false;
  coverActive = false;
  eyeActive = false;
  worthActive = false;
  maddoxActive = false;
  vgActive = false;
  dmActive = false;
  pvActive = false;
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
    inspRollSum = 0; inspRollN = 0;
    inspDip = new Array(9).fill(false);
    inspResultPanel = false;
    playClip(CLIP_INSP_SELFREPORT);  // invites the one-vs-two press
  } else if (t === ROW_COVER) {
    coverActive = true; coverStage = 0; coverT = 0; coverLast = 0;
    playClip(CLIP_COVER_LOOK);
  } else if (t === ROW_EYEMOVE) {
    eyeActive = true; eyeStage = 0; eyeT = 0; eyeLast = 0; eyeFlagged = false;
    eyeDoubleAck = false;
    eyeDip = new Array(9).fill(false); eyeFlash = 0;
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
  beginSession('Vision Testing');
  activateRunTest(0);
}
function finishRun() {  // record + summarize, then return to the panel
  if (estWv >= 1 && estWh >= 1)
    recordResult('Prism estimate',
                 'V ' + estV.toFixed(1) + 'D H ' + estH.toFixed(1) + 'D (found)');
  else if (estWv > 0 || estWh > 0)
    recordResult('Prism estimate', 'V ' + estV.toFixed(1) + 'D H ' +
                 estH.toFixed(1) + 'D (low confidence)');
  writeSession();
  summaryActive = sessionLines.length > 2;
  testMode = 'select';
  resetDemos();
  lightsOn = true;
  updateStatus();
}
function advanceRun() {
  if (++runIdx >= runList.length) {
    // Stop the just-finished test's stage machine, or its auto-advance (e.g.
    // Double Maddox's `if (!playing) advanceRun()`) keeps re-calling this and
    // the prism-verification intro loops forever.
    inspActive = coverActive = eyeActive = worthActive = maddoxActive =
      vgActive = dmActive = false;
    // run finished: if a confident prism estimate was found, show the
    // "here's how that prism could help" demo before returning to select
    const found = estWv >= 1 && estWh >= 1;
    skipClip();
    if (found) {
      pvActive = true; pvStage = 0; pvT = 0;
      playClip(CLIP_PV_INTRO);  // stay in run mode, room dim
      updateStatus();
    } else {
      playClip(CLIP_PV_NONE);
      finishRun();
    }
  } else {
    activateRunTest(runIdx);
    updateStatus();
  }
}

// pick a workflow from the menu (plays its intro clip, then runs it)
function chooseWorkflow(r) {
  if (r === WORKFLOW_QUIT_ROW) {  // Quit: end the immersive session
    if (xrSession) xrSession.end();
    return;
  }
  if (r === 0) { phase = 'intro_test'; playClip(CLIP_INTRO_TEST); }
  else if (r === 1) { phase = 'intro_ther'; playClip(CLIP_INTRO_THER); }
  else { phase = 'intro_games'; playClip(CLIP_INTRO_GAMES); }
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
  if (finish) { recordThp(); playClip(CLIP_THP_DONE); thpStage = 90; }
}
// therapy trigger (shared by the XR trigger and the desktop Space key)
function therapyTrigger() {
  if (thpMode === 'select') {
    if (thpPrismToggleHit) {
      therapyPrismOn = !therapyPrismOn;
    } else if (thpHit) {
      if (thpHit.kind === 'check') {
        thpSelected[thpHit.row] = !thpSelected[thpHit.row];
        saveThpSelection();
      } else if (thpHit.kind === 'talk') {
        playClip(thpDescClip(thpHit.row));
      } else if (thpHit.kind === 'start') {
        thpStartRun();
      } else if (thpHit.kind === 'back') {  // back to the workflow chooser
        phase = 'choose'; thpMode = 'select';
        playClip(CLIP_CHOOSE);
      }
    } else if (playingClip >= 0) skipClip();
    else toggleLights();
  } else {  // run
    if (playingClip >= 0) skipClip();
    else thpRunPress();
  }
}

// ---- Vision Games machinery (mirrors the therapy select/run pattern) ----
function gamesPanelHit(pos, q) {
  const uv = menuHitUV(pos, q, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
  if (!uv) return null;
  if (inBackZone(uv.u, uv.v)) return { kind: 'menu' };
  if (Math.abs(uv.v - CHECKLIST_START_V) <= CHECKLIST_START_HALF_V &&
      uv.u > 0.12 && uv.u < 0.88) return { kind: 'start' };
  if (Math.abs(uv.v - GAME_ROW0_V) <= 0.045) {
    if (uv.u >= CHECKLIST_TALK_MIN_U && uv.u <= CHECKLIST_TALK_MAX_U)
      return { kind: 'talk', row: 0 };
    return { kind: 'check', row: 0 };
  }
  if (Math.abs(uv.v - GAME_EYE_V) <= 0.05 && uv.u > 0.10 && uv.u < 0.90)
    return { kind: 'eye' };
  if (Math.abs(uv.v - GAME_MODE_V) <= 0.05 && uv.u > 0.10 && uv.u < 0.90)
    return { kind: 'mode' };
  return null;
}
function gamesStartRun() {
  if (!gameSelected[0]) return;
  // START always starts. If no eye was chosen, default to the LEFT eye
  // (clearly labelled + changeable) rather than silently refusing.
  if (!amblyKnown || amblyEye === AMBLY_NONE) {
    amblyEye = AMBLY_OS; amblyKnown = true; saveAmblyEye();
  }
  const eyeNm = amblyEye === AMBLY_OD ? 'Right' : 'Left';
  setMessage('Flappy — flap: tap/click/Space · exit: M/grip · training ' +
             eyeNm + ' eye (change on the games panel)');
  gmMode = 'run'; lightsOn = false;
  gameSessions++;
  beginSession('Vision Games');
  flappyInit();
  playClip(CLIP_DESC_FLAPPY);   // brief how-to; the first flap skips it
}
function gamesEndRun() {         // record (once) + return to the select panel
  if (!flappyRecorded) {
    recordResult('Flappy', 'score ' + flappy.score);
    writeSession();
    flappyRecorded = true;
  }
  gmMode = 'select'; lightsOn = true;
  setMessage(''); setAnaHint('');
}
function gamesTrigger() {
  if (gmMode === 'select') {
    if (gmHit) {
      if (gmHit.kind === 'check') {
        gameSelected[0] = !gameSelected[0];
      } else if (gmHit.kind === 'talk') {
        playClip(CLIP_DESC_FLAPPY);
      } else if (gmHit.kind === 'eye') {
        amblyEye = amblyEye === AMBLY_NONE ? AMBLY_OD
                 : amblyEye === AMBLY_OD ? AMBLY_OS : AMBLY_NONE;
        amblyKnown = amblyEye !== AMBLY_NONE;
        saveAmblyEye();
      } else if (gmHit.kind === 'mode') {
        gameMode = gameMode === GAME_DICHOPTIC ? GAME_MONOCULAR : GAME_DICHOPTIC;
      } else if (gmHit.kind === 'start') {
        gamesStartRun();
      } else if (gmHit.kind === 'menu') {
        phase = 'choose'; gmMode = 'select';
        playClip(CLIP_CHOOSE);
      }
    } else if (playingClip >= 0) skipClip();
    else toggleLights();
  } else {  // run: the trigger is a flap (a clip playing skips first)
    if (playingClip >= 0) { skipClip(); return; }
    if (flappy.dead) { flappyInit(); return; }   // tap to replay
    flappyPending = true;
  }
}
// advance the deterministic Flappy physics; called each frame while running
let gamesLast = 0;
function gamesUpdate() {
  const now = performance.now();
  let dt = gamesLast ? (now - gamesLast) / 1000 : 0;
  gamesLast = now;
  if (dt > 0.05) dt = 0.05;             // clamp tab-switch gaps
  if (!(gamesPhase() && gmMode === 'run')) return;
  if (playingClip >= 0) return;         // hold during the how-to clip
  if (flappy.dead) return;
  const flap = flappyPending; flappyPending = false;
  if (flap) flappy.birdV = FLAPPY_FLAP;
  flappy.birdV += FLAPPY_GRAVITY * dt;
  if (flappy.birdV < FLAPPY_MAXFALL) flappy.birdV = FLAPPY_MAXFALL;
  flappy.birdY += flappy.birdV * dt;
  if (flappy.birdY <= 0) { flappy.birdY = 0; flappy.dead = true; }
  if (flappy.birdY >= 1) { flappy.birdY = 1; flappy.birdV = 0; }
  // scroll pipes right->left; one pipe per unit, spaced by PIPE_SPACING
  const PIPE_SPACING = 0.75, PIPE_HALFW = 0.06, SPEED = 0.24;
  flappyScroll += SPEED * dt;
  // the nearest pipe ahead of / at the bird
  const logMAR = 0.5;   // TODO: from per-profile baseline once calibrated
  const gapHalf = flappyGapHalf(logMAR, gameMode === GAME_MONOCULAR ? 1 : 0);
  const idx = Math.floor(flappyScroll / PIPE_SPACING);
  for (let k = idx; k <= idx + 1; ++k) {
    const px = (k * PIPE_SPACING - flappyScroll) + FLAPPY_BIRDX; // world x of pipe
    if (Math.abs(px - FLAPPY_BIRDX) <= PIPE_HALFW) {
      const gc = flappyGapCenter(k);
      if (flappy.birdY > gc + gapHalf || flappy.birdY < gc - gapHalf)
        flappy.dead = true;
    }
  }
  // score: passed pipe centers behind the bird
  const passed = Math.floor((flappyScroll - 0.0) / PIPE_SPACING);
  if (passed > flappy.score && passed >= 1) flappy.score = passed;
  if (flappy.dead && !flappyRecorded) {   // game-over beat: freeze, show score
    playClick();
    recordResult('Flappy', 'score ' + flappy.score);
    writeSession();
    flappyRecorded = true;
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
                  'assets/audio/thp_done.wav', 'assets/audio/pv_intro.wav',
                  'assets/audio/pv_result.wav', 'assets/audio/pv_none.wav',
                  'assets/audio/insp_selfreport.wav',
                  'assets/audio/insp_sum_stable.wav', 'assets/audio/insp_sum_better.wav',
                  'assets/audio/insp_sum_worse.wav', 'assets/audio/insp_sum_worst.wav',
                  'assets/audio/dir_centre.wav', 'assets/audio/dir_up.wav',
                  'assets/audio/dir_up_right.wav', 'assets/audio/dir_right.wav',
                  'assets/audio/dir_down_right.wav', 'assets/audio/dir_down.wav',
                  'assets/audio/dir_down_left.wav', 'assets/audio/dir_left.wav',
                  'assets/audio/dir_up_left.wav',
                  'assets/audio/intro_games.wav', 'assets/audio/desc_flappy.wav'];
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
// per-frame advancePhase() moves the phase machine on. A queued sequence
// (playClipSeq) drains clip-by-clip before playingClip returns to -1.
let clipQueue = [];
function playClip(i) {
  // single-clip player: stop whatever is playing so clips never talk over
  // each other (e.g. quickly clicking a menu button while narration plays).
  // We null onended first so the old clip's end doesn't advance the machine.
  if (introSource) {
    const prev = introSource;
    introSource = null;
    prev.onended = null;
    try { prev.stop(); } catch (e) { /* already stopped */ }
  }
  playingClip = i;
  if (!audioOk || !introBuffers[i]) { onClipEnd(); return; }
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const src = audioCtx.createBufferSource();
  src.buffer = introBuffers[i];
  src.connect(audioCtx.destination);
  src.onended = () => { if (introSource === src) onClipEnd(); };
  introSource = src;
  src.start();
}
// play a sequence of clips back to back (e.g. summary verdict + direction)
function playClipSeq(seq) {
  clipQueue = seq.slice();
  if (clipQueue.length) playClip(clipQueue.shift());
  else playingClip = -1;
}
function onClipEnd() {  // a clip finished (or was skipped): next, or idle
  introSource = null;
  if (clipQueue.length) playClip(clipQueue.shift());
  else playingClip = -1;
}

function skipClip() {
  clipQueue = [];  // a skip drops any queued remainder too
  if (introSource) {
    const s = introSource;
    introSource = null;
    s.onended = null;
    try { s.stop(); } catch (e) { /* already stopped */ }
  }
  playingClip = -1;
}

// short synthesized UI blip, mixed over any narration (a separate node), so a
// button/trigger press is audibly confirmed. Mirrors the native introPlayClick.
function playClick() {
  if (!audioOk || !audioCtx) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();
  const t = audioCtx.currentTime;
  const osc = audioCtx.createOscillator();
  const g = audioCtx.createGain();
  osc.type = 'triangle';
  osc.frequency.setValueAtTime(880, t);
  osc.frequency.exponentialRampToValueAtTime(520, t + 0.06);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.3, t + 0.004);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.09);
  osc.connect(g); g.connect(audioCtx.destination);
  osc.start(t); osc.stop(t + 0.10);
}

// first user gesture (VR entry / preview click): kick off the welcome clip
function startPhases() {
  if (phaseStarted) return;
  phaseStarted = true;
  if (audioOk) { phase = 'welcome'; playClip(CLIP_WELCOME); }
  else phase = 'profile';  // no audio -> pick a player, then the menu
  updateStatus();
}

// advance the phase machine when the current clip finishes or is skipped;
// playingClip is -1 whenever nothing is sounding (incl. no-audio)
function advancePhase() {
  if (!phaseStarted || playingClip >= 0) return;
  if (phase === 'welcome') { phase = 'disclaimer'; playClip(CLIP_DISCLAIMER); }
  else if (phase === 'disclaimer') { phase = 'profile'; }  // pick a player first
  else if (phase === 'choose') { phase = 'select'; }
  else if (phase === 'intro_test') {
    phase = 'testing';  // lights stay up for test selection; dim on START
  } else if (phase === 'intro_ther') { phase = 'therapy'; thpMode = 'select'; }
  else if (phase === 'intro_games') { phase = 'games'; gmMode = 'select'; }
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
  if (inspFlashFrames > 0) inspFlashFrames--;  // press-confirm flash
  let ry = quatRightY(headQuat);
  ry = Math.max(-1, Math.min(1, ry));
  const rollDeg = Math.asin(ry) * 57.29578;
  inspLiveRoll = rollDeg;  // live head roll for the spirit level
  if (inspT > 1) { inspRollSum += rollDeg; inspRollN++; }
  // once the H has stepped through all 9 positions, report + advance
  if (playingClip < 0) {
    if (inspStage === 0 && inspStepJS(inspT).index >= 9) {
      const avg = inspRollN ? inspRollSum / inspRollN : 0;
      inspTilted = Math.abs(avg) >= 3;
      const dipN = inspDip.reduce((a, d) => a + (d ? 1 : 0), 0);
      playClip(!inspTilted ? CLIP_INSP_LEVEL
               : avg > 0 ? CLIP_INSP_LEFT : CLIP_INSP_RIGHT);
      recordResult('Ocular inspection', 'head roll ' + avg.toFixed(1) +
                   ' deg (' + (inspTilted ? 'tilted' : 'level') +
                   '), diplopia ' + dipN + '/9');
      saveInspRecord(avg, inspDip.slice());   // persist to localStorage
      inspStage = 1;
    } else if (inspStage === 1 && inspTilted) {
      // one-time explanation: keep the head level for the tests to come
      playClip(CLIP_INSP_KEEPLEVEL);
      inspTilted = false;
    } else if (inspStage === 1) {
      // subjective diplopia verdict: saw two anywhere (misaligned) vs fused
      const dipN = inspDip.reduce((a, d) => a + (d ? 1 : 0), 0);
      playClip(dipN > 0 ? CLIP_INSP_MISALIGNED : CLIP_INSP_ALIGNED);
      inspStage = 2;
    } else if (inspStage === 2) {
      // spoken results summary: field-of-single-vision trend + worst direction
      speakInspSummary();
      inspStage = 3;
    } else if (inspStage === 3) {
      playClip(CLIP_INSP_DONE);
      inspStage = 4;
    } else if (inspStage === 4) {
      // show the results card ONCE, then stage 5 is terminal — a repeating
      // finalize would clobber any state the card carries (native hit this:
      // its gallery page reset every frame until the stage became terminal)
      inspResultPanel = true;
      inspStage = 5;
    }
  }
}

// per-profile inspection history in localStorage: [{t, dip:[9], tilt}]. Used
// for the subjective field-of-single-vision trend + worst-direction summary.
function inspHistKey() { return 'insp_hist_' + (activeProfile || 'default'); }
function loadInspHist() {
  try { return JSON.parse(localStorage.getItem(inspHistKey())) || []; }
  catch (e) { return []; }
}
function saveInspRecord(tilt, dip) {
  const h = loadInspHist();
  const rec = { t: Date.now(), tilt: tilt, dip: dip.map(d => d ? 1 : 0) };
  lastInspRec = rec;
  h.push(rec);
  while (h.length > 1000) h.shift();
  try { localStorage.setItem(inspHistKey(), JSON.stringify(h)); } catch (e) {}
}
function speakInspSummary() {
  const h = loadInspHist();
  if (!h.length) { playClip(CLIP_INSP_SUM_STABLE); return; }
  const cur = h[h.length - 1];
  const curN = cur.dip.reduce((a, d) => a + d, 0);
  // trend: today's diplopia count vs the mean of prior sittings
  const prior = h.slice(0, -1);
  let verdict = CLIP_INSP_SUM_STABLE;
  if (prior.length) {
    const mean = prior.reduce((a, r) => a + r.dip.reduce((b, d) => b + d, 0), 0) /
                 prior.length;
    if (curN < mean - 0.5) verdict = CLIP_INSP_SUM_BETTER;
    else if (curN > mean + 0.5) verdict = CLIP_INSP_SUM_WORSE;
  }
  // worst position: most-frequently-doubled across all history
  const freq = new Array(9).fill(0);
  for (const r of h) for (let i = 0; i < 9; i++) freq[i] += r.dip[i] || 0;
  let worst = -1, best = 0;
  for (let i = 0; i < 9; i++) if (freq[i] > best) { best = freq[i]; worst = i; }
  const seq = [verdict];
  if (worst >= 0) { seq.push(CLIP_INSP_SUM_WORST); seq.push(CLIP_DIR0 + INSP_DIR[worst]); }
  playClipSeq(seq);
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
    if (coverStage === 0) { recordResult('Cover test', 'no eye tracking'); playClip(CLIP_COVER_NOGAZE); coverStage = 1; }
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
  const dt = eyeLast ? Math.min(0.1, (now - eyeLast) / 1000) : 0;
  eyeLast = now;
  if (eyeFlash > 0) eyeFlash--;
  if (playingClip >= 0) return;      // hold the pursuit while a clip plays
  eyeT += dt;                        // the light glides only between clips
  if (eyeStage === 0 && eyeT >= EYE_PASS_DUR) {
    const dipN = eyeDip.reduce((a, d) => a + (d ? 1 : 0), 0);
    recordResult('Ocular motility', dipN === 0
        ? 'smooth pursuit, no diplopia'
        : 'diplopia in gaze ' + dipN + '/9 (worst ' + eyeWorstName() + ')');
    speakMotilitySummary(dipN);
    eyeStage = 1;
  } else if (eyeStage === 1) {
    advanceRun();
  }
}
// most-flagged gaze position (first flagged wins ties); -1 if none
function eyeWorstIdx() {
  for (let i = 0; i < 9; i++) if (eyeDip[i]) return i;
  return -1;
}
function eyeWorstName() {
  const w = eyeWorstIdx();
  return w >= 0 ? INSP_DIRNM[INSP_DIR[w]] : 'none';
}
// spoken summary: smooth vs limited, worst gaze direction, done
function speakMotilitySummary(dipN) {
  const seq = [dipN === 0 ? CLIP_EYE_SMOOTH : CLIP_EYE_LIMITED];
  const w = eyeWorstIdx();
  if (dipN > 0 && w >= 0) {
    seq.push(CLIP_INSP_SUM_WORST); seq.push(CLIP_DIR0 + INSP_DIR[w]);
  }
  seq.push(CLIP_EYE_DONE);
  playClipSeq(seq);
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
      recordResult('Worth 4-dot', (worthSelected >= 5 ? 'diplopia'
                   : worthSelected === 4 ? 'fused'
                   : worthSelected === 3 ? 'suppress right' : 'suppress left') +
                   ' (' + worthSelected + ' dots)');
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
    recordResult('Maddox rod', 'V ' + (maddoxVDone ? estV.toFixed(1) + 'D' : 'n/a') +
                 ' H ' + (maddoxHDone ? estH.toFixed(1) + 'D' : 'n/a'));
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
      recordResult('Von Graefe', 'V ' + (vgHaveV ? vgResultV.toFixed(1) + 'D' : 'n/a') +
                   ' H ' + (vgHaveH ? vgResultH.toFixed(1) + 'D' : 'n/a'));
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
      recordResult('Double Maddox', !dmDone ? 'no clear match'
                   : dmTorsion.toFixed(1) + ' deg torsion (' +
                     (Math.abs(dmTorsion) > 2 ? 'refer' : 'aligned') + ')');
      dmStage = 2;
    }
  } else if (dmStage === 2) {
    playClip(CLIP_DM_DONE); dmStage = 3;
  } else if (dmStage === 3) {
    advanceRun();
  }
}

// "Show me the prism" verification stage machine
function updatePv() {
  if (!pvActive) { pvLast = 0; return; }
  const now = performance.now();
  pvT += pvLast ? Math.min(0.1, (now - pvLast) / 1000) : 0;
  pvLast = now;
  const playing = playingClip >= 0;
  if (pvStage === 0) {          // intro clip -> begin the ramp when it ends
    if (!playing) { pvStage = 1; pvT = 0; }
  } else if (pvStage === 1) {   // ramp the found prism onto the H
    if (pvT >= 3.5) { pvStage = 2; pvT = 0; playClip(CLIP_PV_RESULT); }
  } else if (pvStage === 2) {   // hold at full correction over the result
    if (!playing && pvT > 0.5) { pvStage = 3; pvT = 0; }
  } else if (pvStage === 3) {   // linger on the fused H, then finish
    if (pvT > 2.5) finishRun();
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
    if (finish) { recordThp(); playClip(CLIP_THP_DONE); thpStage = 90; }
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
      else if (clHit.kind === 'back') { cu = 0.03; cv = 0.071; }
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
    // blank the acuity display to black (a real display showing nothing) —
    // the H below is the only lit content
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(therapyDotVao);
    // flashes green for a few frames after a diplopia press (visual confirm)
    if (inspActive && inspFlashFrames > 0)
      gl.uniform4f(locBeamColor, 0.45, 1.0, 0.55, 1);
    else
      gl.uniform4f(locBeamColor, 0.95, 0.95, 0.98, 1);
    const hz = -1.0, hHalfW = 0.05, hHalfH = 0.07, hStroke = 0.009;
    // the inspection H steps through the 9 positions; the cover test keeps it centred
    const hc = inspActive ? inspSweepXY(inspT) : [0, 0.15];
    const bars = [[hc[0] - hHalfW, hc[1], hStroke, hHalfH],
                  [hc[0] + hHalfW, hc[1], hStroke, hHalfH],
                  [hc[0], hc[1], hHalfW, hStroke]];
    // the H + tilt meter hide while the results card is up (drawn later)
    if (!inspResultPanel)
      for (const b of bars) {
        gl.uniformMatrix4fv(locBeamMvp, false,
            mul(vp, mul(translationMat(b[0], b[1], hz),
                        scaleMat(b[2], b[3], 1))));
        gl.drawArrays(gl.TRIANGLES, 0, 6);
      }
    if (inspActive && !inspResultPanel) {
      // spirit level: LIVE roll, fades toward invisible when level, bold past
      // ~8 deg; marker green (<3) / yellow (3-6) / red (>6) by tilt.
      const tilt = Math.abs(inspLiveRoll);
      const att = tilt < 1.5 ? 0 : tilt > 8 ? 1 : (tilt - 1.5) / 6.5;
      const m = 0.08 + 0.92 * att;
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(0, -0.18, -1), scaleMat(0.30, 0.006, 1))));
      gl.uniform4f(locBeamColor, 0.35, 0.40, 0.48, m);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      let mx = inspLiveRoll / 15;
      mx = Math.max(-1, Math.min(1, mx));
      if (tilt < 3) gl.uniform4f(locBeamColor, 0.35, 0.90, 0.50, m);
      else if (tilt < 6) gl.uniform4f(locBeamColor, 0.95, 0.85, 0.30, m);
      else gl.uniform4f(locBeamColor, 0.95, 0.30, 0.28, m);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vp, mul(translationMat(mx * 0.30, -0.18, -1),
                      scaleMat(0.013, 0.03, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.disable(gl.BLEND);
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
    if (eyeFlash > 0) gl.uniform4f(locBeamColor, 0.30, 1.0, 0.40, 1);  // press confirm
    else gl.uniform4f(locBeamColor, 1, 1, 1, 1);        // bright core
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
    gl.uniform4f(locBeamColor, 0.95, 0.97, 1, 1);       // white bottom, both eyes
    worthDot(wcx, wcy - wsp);
    if (xrSession || sbs) {
      // headset / SBS 3D TV: true dichoptic by DISPLAY — top only to OD,
      // sides only to OS, in real colours (the separation is spatial, not by
      // colour, so no lens-matching is needed)
      if (rightEye) {
        gl.uniform4f(locBeamColor, 1, 0.2, 0.2, 1);
        worthDot(wcx, wcy + wsp);
      } else {
        gl.uniform4f(locBeamColor, 0.2, 0.95, 0.3, 1);
        worthDot(wcx - wsp, wcy);
        worthDot(wcx + wsp, wcy);
      }
    } else if (anaglyph) {
      // laptop glasses: real lens-matched colours, drawn once; the physical
      // filters dissociate. top -> right-eye lens, sides -> left-eye lens
      // (disjoint masks => each eye sees only its own dots). Mirrors
      // vlogic::worthColors.
      const tc = ANA_MASKS[anaPreset][1], sc = ANA_MASKS[anaPreset][0];
      gl.uniform4f(locBeamColor, tc[0], tc[1], tc[2], 1);
      worthDot(wcx, wcy + wsp);
      gl.uniform4f(locBeamColor, sc[0], sc[1], sc[2], 1);
      worthDot(wcx - wsp, wcy);
      worthDot(wcx + wsp, wcy);
    } else {
      // no glasses (flat preview): the canonical 4-dot pattern so the layout
      // is visible — a demo, not a real dissociated test
      gl.uniform4f(locBeamColor, 1, 0.2, 0.2, 1);
      worthDot(wcx, wcy + wsp);
      gl.uniform4f(locBeamColor, 0.2, 0.95, 0.3, 1);
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

  // "Show me the prism": the H with the found estimate ramped on per eye. At
  // full correction the two eyes' images coincide, so a wearer whose deviation
  // matches the estimate sees the two H's fuse to one.
  // NOTE: per-eye sign matches the Von Graefe convention; if the two H's
  // DIVERGE instead of merging in-headset, negate ov/oh.
  if (pvActive) {
    const fr = pvStage === 0 ? 0 : pvStage >= 2 ? 1 : Math.min(1, pvT / 3.5);
    const s = (rightEye ? 1 : -1) * fr;
    const ov = s * estV / 200;   // vertical split
    const oh = -s * estH / 200;   // horizontal split (base-out +)
    const hcx = oh, hcy = 0.15 + ov;
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
    if (pvStage >= 2) {  // offer to keep the found prism as this person's default
      const vpW = mul(projMatrix,
                      mul(viewRotMatrix,
                          translationMat(-curPos.x, -curPos.y, -curPos.z)));
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      drawText(vpW, -0.42, -0.32, 0.022, 0.032, 0.6, 0.9, 0.72,
               'Press to save this as your prism');
      gl.disable(gl.BLEND);
    }
  }

  // Prism estimate readout below the acuity display (seven-segment): a
  // vertical row and a horizontal row (Δ) + a confidence bar. No calibration
  // line — WebXR has no gaze.
  if (testingPhase() && testMode === 'run' && !inspActive &&
      (estWv > 0 || estWh > 0)) {
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
  // (the Ocular-inspection findings are reviewed on the end-of-run results
  // card, not repeated as an unreadable strip under the display every test.)

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
    // Chooser buttons below the menu: Adjust Prism (left) + Switch Person
    // (right) — zones mirror vlogic::workflowButtonAt — plus the active
    // person's name ("signed in as").
    {
      const by = (0.5 - 0.92) * PROFILE_H;
      const bh = 0.04 * PROFILE_H, bwHalf = 0.21 * PROFILE_W;
      const lx = (0.27 - 0.5) * PROFILE_W, rx = (0.73 - 0.5) * PROFILE_W;
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(therapyDotVao);
      gl.uniform4f(locBeamColor, 0.16, 0.42, 0.46, prismBtnHit ? 0.85 : 0.5);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(translationMat(lx, by, -CHECKLIST_DIST + 0.007),
                           scaleMat(bwHalf, bh, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.uniform4f(locBeamColor, 0.16, 0.42, 0.46, personBtnHit ? 0.85 : 0.5);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(translationMat(rx, by, -CHECKLIST_DIST + 0.007),
                           scaleMat(bwHalf, bh, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindVertexArray(null);
      drawText(vpWorld, lx - 0.185, by + 0.016, 0.020, 0.032, 0.9, 0.97, 0.9,
               profPrismKnown
                   ? 'Prism V ' + profPrismV.toFixed(2) + ' H ' + profPrismH.toFixed(2)
                   : 'Prism not set');
      drawText(vpWorld, rx - 0.155, by + 0.016, 0.020, 0.032, 0.9, 0.97, 0.9,
               'SWITCH PERSON');
      const who = 'for ' + activeProfile;
      drawText(vpWorld, -0.5 * 0.028 * who.length, (0.5 - 0.84) * PROFILE_H,
               0.028, 0.042, 0.55, 0.90, 0.98, who);
      gl.disable(gl.BLEND);
    }
  }

  // Session summary panel: a dark card + the recorded result lines, shown at
  // the end of a run (dismissed by a trigger press).
  if (summaryActive) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniform4f(locBeamColor, 0.04, 0.05, 0.07, 0.96);
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vpWorld, mul(translationMat(0, 0, -CHECKLIST_DIST + 0.006),
                         scaleMat(0.64, 0.54, 1))));
    gl.bindVertexArray(therapyDotVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
    let ty = 0.44;
    for (let li = 0; li < sessionLines.length && li < 15; ++li) {
      const head = li === 0;
      drawText(vpWorld, -0.57, ty, 0.026, 0.040, head ? 0.55 : 0.86,
               head ? 0.90 : 0.88, head ? 0.98 : 0.92, sessionLines[li]);
      ty -= head ? 0.075 : 0.058;
    }
    drawText(vpWorld, -0.57, ty - 0.01, 0.021, 0.032, 0.45, 0.65, 0.72,
             'press to dismiss');
    gl.disable(gl.BLEND);
  }

  // Post-inspection results: drawn on the (black-blanked) acuity display in
  // cube space, where the wearer is already looking. A subjective field-of-
  // single-vision map (which of the 9 positions doubled) + tilt/diplopia.
  // Trigger dismisses -> next test.
  if (inspResultPanel && lastInspRec) {
    const r = lastInspRec;
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.bindVertexArray(therapyDotVao);
    // title bar (a teal underline strip) so the panel reads as titled
    gl.uniform4f(locBeamColor, 0.16, 0.42, 0.46, 1);
    segRect(vp, 0.0, 0.22, 0.42, 0.006);
    const sx = 0.18, sy = 0.095;
    for (let k = 0; k < 9; k++) {
      const nx = EYE_TARGETS[k][0] / 0.45, ny = (EYE_TARGETS[k][1] - 0.15) / 0.40;
      const px = nx * sx, py = 0.03 + ny * sy;
      if (r.dip[k]) gl.uniform4f(locBeamColor, 0.95, 0.30, 0.28, 1);  // doubled
      else gl.uniform4f(locBeamColor, 0.35, 0.90, 0.50, 1);          // fused
      segRect(vp, px, py, 0.018, 0.018);
    }
    gl.bindVertexArray(null);
    const tl = r.tilt, tdir = Math.abs(tl) < 3 ? 'LEVEL' : (tl > 0 ? 'L' : 'R');
    const dipN = r.dip.reduce((a, d) => a + d, 0);
    let wi = -1; for (let k = 0; k < 9; k++) if (r.dip[k]) { wi = k; break; }
    const l2 = dipN > 0 && wi >= 0
        ? 'DIPLOPIA ' + dipN + '/9  (worst ' + INSP_DIRNM[INSP_DIR[wi]] + ')'
        : 'DIPLOPIA 0/9  FUSED';
    drawText(vp, -0.40, 0.26, 0.026, 0.040, 0.55, 0.9, 0.98,
             'OCULAR INSPECTION - RESULTS');
    drawText(vp, -0.40, -0.16, 0.022, 0.034, 0.9, 0.92, 0.94,
             'TILT ' + Math.abs(tl).toFixed(0) + ' ' + tdir);
    drawText(vp, -0.40, -0.22, 0.022, 0.034,
             dipN > 0 ? 0.95 : 0.6, dipN > 0 ? 0.7 : 0.9, dipN > 0 ? 0.4 : 0.7, l2);
    drawText(vp, -0.40, -0.30, 0.018, 0.028, 0.45, 0.72, 0.80,
             'pull the trigger to continue');
  }

  // Player-profile: the Select-Player list, or the new-name keyboard.
  if (phase === 'profile') {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniform4f(locBeamColor, 0.04, 0.05, 0.07, 0.96);
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vpWorld, mul(translationMat(0, 0, -CHECKLIST_DIST + 0.006),
                         scaleMat(0.64, 0.54, 1))));
    gl.bindVertexArray(therapyDotVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    const hi = (u, v, hu, hv, r, g, b, a) => {
      gl.uniform4f(locBeamColor, r, g, b, a);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(translationMat(profilePux(u), profilePuy(v),
                                          -CHECKLIST_DIST + 0.007),
                           scaleMat(hu * PROFILE_W, hv * PROFILE_H, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    const profRows = Math.min(profiles.length, PROFILE_MAXROWS);
    if (kbActive) {
      const cw = (KB_U1 - KB_U0) / 10, ch = (KB_V1 - KB_V0) / 3;
      if (kbHit)
        hi(KB_U0 + (kbHit.col + 0.5) * cw, KB_V0 + (kbHit.row + 0.5) * ch,
           cw * 0.46, ch * 0.44, 0.16, 0.42, 0.46, 0.9);
      gl.bindVertexArray(null);
      drawText(vpWorld, profilePux(0.06), profilePuy(0.10), 0.03, 0.05,
               0.55, 0.9, 0.98, 'NEW PERSON');
      drawText(vpWorld, profilePux(0.06), profilePuy(0.24), 0.034, 0.052,
               0.9, 0.95, 0.7, '> ' + newName + '_');
      for (let kr = 0; kr < 3; ++kr)
        for (let kc = 0; kc < 10; ++kc) {
          const key = kbKeyAt(kc, kr);
          if (!key) continue;
          let lbl;
          if (key >= 65 && key <= 90) lbl = String.fromCharCode(key);
          else if (key === 32) lbl = '_';
          else if (key === 8) lbl = 'DEL';
          else if (key === 10) lbl = 'OK';
          else lbl = 'X';
          const gw = 0.028;
          drawText(vpWorld,
                   profilePux(KB_U0 + (kc + 0.5) * cw) - lbl.length * gw * 0.5,
                   profilePuy(KB_V0 + (kr + 0.5) * ch) + 0.02, gw, 0.038,
                   0.92, 0.95, 0.96, lbl);
        }
    } else {
      for (let r = 0; r < profRows; ++r) {
        const v = PROFILE_ROW0V + r * PROFILE_ROWDV;
        if (r === pendingDelete)
          hi(0.5, v, 0.47, PROFILE_ROWDV * 0.45, 0.5, 0.12, 0.12, 0.5);
        else if (profHit && profHit.row === r)
          hi(0.5, v, 0.47, PROFILE_ROWDV * 0.45, 0.16, 0.42, 0.46, 0.55);
      }
      if (profHit && profHit.kind === 'new')
        hi(0.5, PROFILE_ROW0V + profRows * PROFILE_ROWDV, 0.47,
           PROFILE_ROWDV * 0.45, 0.16, 0.42, 0.46, 0.55);
      gl.bindVertexArray(null);
      drawText(vpWorld, profilePux(0.06), profilePuy(0.10), 0.03, 0.05,
               0.55, 0.9, 0.98, 'SELECT PERSON');
      for (let r = 0; r < profRows; ++r) {
        const v = PROFILE_ROW0V + r * PROFILE_ROWDV;
        const act = profiles[r] === activeProfile;
        drawText(vpWorld, profilePux(0.10), profilePuy(v) + 0.018, 0.03, 0.046,
                 act ? 0.55 : 0.9, act ? 0.95 : 0.92, act ? 0.7 : 0.94,
                 profiles[r]);
        drawText(vpWorld, profilePux(0.83), profilePuy(v) + 0.018, 0.03, 0.046,
                 0.85, 0.4, 0.4, r === pendingDelete ? '?' : 'x');
      }
      drawText(vpWorld, profilePux(0.10),
               profilePuy(PROFILE_ROW0V + profRows * PROFILE_ROWDV) + 0.018,
               0.03, 0.046, 0.6, 0.85, 0.6, '+ Add Person');
    }
    // controller ray, shortened to stop at the panel (like the others)
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    for (const a of aimPoses) {
      const uv = menuHitUV(a.pos, a.quat, PROFILE_DIST, PROFILE_W, PROFILE_H);
      const bz = uv ? uv.t / 8.0 : 1.0;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(poseMatrix(a.pos, a.quat), scaleMat(1, 1, bz))));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
    }
    gl.disable(gl.BLEND);
  }

  // Prism-prescription panel: Vertical / Horizontal rows with -/+ and Done.
  // Values in Δ (shown as "D"; the font atlas has no Δ glyph).
  if (phase === 'prism') {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniform4f(locBeamColor, 0.04, 0.05, 0.07, 0.96);
    gl.uniformMatrix4fv(locBeamMvp, false,
        mul(vpWorld, mul(translationMat(0, 0, -CHECKLIST_DIST + 0.006),
                         scaleMat(0.64, 0.54, 1))));
    gl.bindVertexArray(therapyDotVao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    const hi = (u, v, hu, hv, r, g, b, a) => {
      gl.uniform4f(locBeamColor, r, g, b, a);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(translationMat(profilePux(u), profilePuy(v),
                                          -CHECKLIST_DIST + 0.007),
                           scaleMat(hu * PROFILE_W, hv * PROFILE_H, 1))));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    const btn = (u, v, hot) => hi(u, v, 0.07, 0.045, 0.16, 0.42, 0.46, hot ? 0.85 : 0.4);
    btn(0.55, 0.34, prismHit === 'vdn');
    btn(0.87, 0.34, prismHit === 'vup');
    btn(0.55, 0.48, prismHit === 'hdn');
    btn(0.87, 0.48, prismHit === 'hup');
    hi(0.5, 0.72, 0.22, 0.05, 0.16, 0.42, 0.46, prismHit === 'done' ? 0.85 : 0.5);
    gl.bindVertexArray(null);
    drawText(vpWorld, profilePux(0.06), profilePuy(0.10), 0.03, 0.05, 0.55, 0.9,
             0.98, 'PRISM FOR ' + activeProfile);
    const vs = profPrismKnown ? profPrismV.toFixed(2) + 'D' : 'not set';
    const hs = profPrismKnown ? profPrismH.toFixed(2) + 'D' : 'not set';
    drawText(vpWorld, profilePux(0.08), profilePuy(0.34) + 0.018, 0.03, 0.046, 0.9, 0.92, 0.94, 'Vertical');
    drawText(vpWorld, profilePux(0.54), profilePuy(0.34) + 0.02, 0.045, 0.06, 0.9, 0.95, 0.7, '-');
    drawText(vpWorld, profilePux(0.66), profilePuy(0.34) + 0.018, 0.03, 0.046, 0.7, 0.95, 0.8, vs);
    drawText(vpWorld, profilePux(0.86), profilePuy(0.34) + 0.02, 0.045, 0.06, 0.9, 0.95, 0.7, '+');
    drawText(vpWorld, profilePux(0.08), profilePuy(0.48) + 0.018, 0.03, 0.046, 0.9, 0.92, 0.94, 'Horizontal');
    drawText(vpWorld, profilePux(0.54), profilePuy(0.48) + 0.02, 0.045, 0.06, 0.9, 0.95, 0.7, '-');
    drawText(vpWorld, profilePux(0.66), profilePuy(0.48) + 0.018, 0.03, 0.046, 0.7, 0.95, 0.8, hs);
    drawText(vpWorld, profilePux(0.86), profilePuy(0.48) + 0.02, 0.045, 0.06, 0.9, 0.95, 0.7, '+');
    drawText(vpWorld, profilePux(0.43), profilePuy(0.72) + 0.018, 0.032, 0.05, 0.9, 0.98, 0.9, 'DONE');
    // controller ray, shortened to stop at the panel (like the others)
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    for (const a of aimPoses) {
      const uv = menuHitUV(a.pos, a.quat, PROFILE_DIST, PROFILE_W, PROFILE_H);
      const bz = uv ? uv.t / 8.0 : 1.0;
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, mul(poseMatrix(a.pos, a.quat), scaleMat(1, 1, bz))));
      gl.uniform4f(locBeamColor, 0.75, 0.80, 0.90, 1);
      gl.bindVertexArray(beamVao);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
    }
    gl.disable(gl.BLEND);
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
      else if (thpHit.kind === 'back') { cu = 0.03; cv = 0.071; }
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
    // Prism On/Off toggle (top-right of the panel), shows the value
    {
      const [tx, ty] = checklistLocal(0.585, 0.075);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      const r = thpPrismToggleHit ? 0.6 : 0.4;
      const g = therapyPrismOn ? 0.92 : 0.6;
      drawText(vpWorld, tx, ty, 0.024, 0.036, r, g, 0.62,
               'Prism ' + (therapyPrismOn ? 'ON' : 'OFF') + '  ' +
               profPrismKnown
                   ? profPrismV.toFixed(2) + '/' + profPrismH.toFixed(2)
                   : 'not set');
      gl.disable(gl.BLEND);
    }
  }

  // ---- Vision Games: the select panel (texGmChecklist + live values) ----
  if (gamesPhase() && gmMode === 'select' && texGmChecklist) {
    const viewFull = mul(viewRotMatrix,
                         translationMat(-curPos.x, -curPos.y, -curPos.z));
    const vpWorld = mul(projMatrix, viewFull);
    gl.useProgram(panelProgram);
    gl.uniform3f(locPanelFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locPanelMvp, false,
                        mul(vpWorld, translationMat(0, 0, -CHECKLIST_DIST)));
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texGmChecklist);
    gl.uniform1i(locPanelTex, 0);
    gl.bindVertexArray(checklistPanelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    if (gameSelected[0]) {          // green checkbox fill
      const [bx, by] = checklistLocal(CHECKLIST_BOX_U, GAME_ROW0_V);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(bx, by, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.20, 0.85, 0.35, 1);
      gl.bindVertexArray(checklistMarkVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    if (gmHit) {                    // caret on the hovered control
      let cu = 0.03, cv = GAME_ROW0_V;
      if (gmHit.kind === 'start') { cu = 0.13; cv = CHECKLIST_START_V; }
      else if (gmHit.kind === 'menu') { cu = 0.03; cv = 0.071; }
      else if (gmHit.kind === 'eye') { cu = 0.55; cv = GAME_EYE_V; }
      else if (gmHit.kind === 'mode') { cu = 0.55; cv = GAME_MODE_V; }
      else if (gmHit.kind === 'talk') cu = CHECKLIST_TALK_MIN_U - 0.03;
      const [cx, cy] = checklistLocal(cu, cv);
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpWorld, translationMat(cx, cy, -CHECKLIST_DIST + 0.004)));
      gl.uniform4f(locBeamColor, 0.40, 0.85, 0.95, 1);
      gl.bindVertexArray(checklistCaretVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
    for (const a of aimPoses) {     // ray + cursor dot (clone of therapy)
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
    // overlay the live values into the two control boxes
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const [ex, ey] = checklistLocal(0.655, GAME_EYE_V + 0.012);
    drawText(vpWorld, ex, ey, 0.022, 0.033, 0.85, 0.90, 0.98, amblyEyeText());
    const [mx, my] = checklistLocal(0.655, GAME_MODE_V + 0.012);
    drawText(vpWorld, mx, my, 0.022, 0.033, 0.85, 0.90, 0.98, gameModeText());
    // START needs the amblyopic eye set — say so on the panel (the overlay
    // message is hidden in glasses/TV modes, so START looked dead otherwise)
    const [px, py] = checklistLocal(0.10, 0.70);
    if (!amblyKnown || amblyEye === AMBLY_NONE)
      drawText(vpWorld, px, py, 0.020, 0.030, 0.80, 0.85, 0.95,
               'Tap "Amblyopic eye" to choose  ·  START plays (defaults to Left)');
    else
      drawText(vpWorld, px, py, 0.020, 0.030, 0.55, 0.90, 0.55,
               'Training ' + (amblyEye === AMBLY_OD ? 'Right' : 'Left') +
               ' eye  ·  press START to play');
    gl.disable(gl.BLEND);
  }

  // ---- Vision Games: Flappy Bird (blank display, prism applied, per eye) ----
  if (gamesPhase() && gmMode === 'run') {
    const gPrism = profPrismKnown && (profPrismV !== 0 || profPrismH !== 0);
    const vrot = gPrism
      ? mul(prismRotationPD(rightEye, profPrismV, profPrismH), viewRotMatrix)
      : viewRotMatrix;
    const vpG = mul(projMatrix, vrot);   // rotation-only: the game rides -Z
    // blank the acuity display
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vpG);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    // play field maps [0,1]x[0,1] onto a rectangle on the display
    const FX0 = -0.45, FW = 0.90, FY0 = -0.15, FH = 0.60;
    const fx = t => FX0 + t * FW, fy = t => FY0 + t * FH;
    const quad = (cx, cy, hw, hh, c) => {
      gl.uniformMatrix4fv(locBeamMvp, false,
          mul(vpG, mul(translationMat(cx, cy, -1), scaleMat(hw, hh, 1))));
      gl.uniform4f(locBeamColor, c[0], c[1], c[2], 1);
      gl.bindVertexArray(therapyDotVao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    };
    const amblyThisEye = amblyEye === AMBLY_OD ? rightEye : !rightEye;
    const PIPE_SPACING = 0.75, PIPE_HALFW = 0.06;
    const gapHalf = flappyGapHalf(0.5, gameMode === GAME_MONOCULAR ? 1 : 0);
    // pipes: monocular -> amblyopic eye (full); dichoptic -> fellow eye (dim)
    const showPipes = gameMode === GAME_MONOCULAR ? amblyThisEye : !amblyThisEye;
    if (showPipes) {
      const k = gameMode === GAME_MONOCULAR ? 1.0
              : flappyFellowContrast(gameSessions);
      const pc = [0.25 * k + 0.02, 0.75 * k + 0.05, 0.30 * k + 0.02];
      const idx = Math.floor(flappyScroll / PIPE_SPACING);
      for (let p = idx - 1; p <= idx + 2; ++p) {
        const pf = (p * PIPE_SPACING - flappyScroll) + FLAPPY_BIRDX;
        if (pf < 0 || pf > 1) continue;
        const gc = flappyGapCenter(p);
        const cx = fx(pf), hw = PIPE_HALFW * FW;
        // top bar (gc+gapHalf .. 1) and bottom bar (0 .. gc-gapHalf)
        const topLo = gc + gapHalf, botHi = gc - gapHalf;
        quad(cx, fy((topLo + 1) / 2), hw, (1 - topLo) / 2 * FH, pc);
        quad(cx, fy(botHi / 2), hw, botHi / 2 * FH, pc);
      }
    }
    // bird: a little multi-part sprite (always the amblyopic eye)
    if (amblyThisEye) {
      const bx = fx(FLAPPY_BIRDX), by = fy(flappy.birdY), d = flappy.dead;
      const body = d ? [0.85, 0.45, 0.20] : [1.0, 0.82, 0.20];
      quad(bx, by, 0.026, 0.019, body);                 // body
      quad(bx, by - 0.009, 0.019, 0.012, body);         // belly
      quad(bx - 0.006, by - 0.002, 0.013, 0.009, [0.90, 0.55, 0.15]);  // wing
      quad(bx + 0.026, by + 0.001, 0.009, 0.005, [1.0, 0.55, 0.10]);   // beak
      quad(bx + 0.011, by + 0.009, 0.006, 0.006, [0.98, 0.98, 0.98]);  // eye
      quad(bx + 0.013, by + 0.009, 0.0025, 0.0035, [0.05, 0.05, 0.05]);// pupil
    }
    // score (both eyes, top-left of the field)
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    drawText(vpG, FX0 + 0.02, FY0 + FH - 0.02, 0.024, 0.036, 0.95, 0.97, 1.0,
             flappy.dead ? ('GAME OVER  ' + flappy.score + '   tap to fly again')
                         : ('' + flappy.score));
    drawText(vpG, FX0 + 0.02, FY0 + 0.06, 0.017, 0.025, 0.70, 0.85, 0.95,
             'exit: grip / B / Y button  (or M key)');
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  // Vision Therapy activity targets (world-anchored). Bead-on-cord activities
  // are binocular (true stereo poses the vergence demand); prism/vertical/both/
  // stereo are dichoptic — offset per eye. Bead activities draw a realistic
  // lit Brock string (drilled beads + twisted cord); the rest use flat quads.
  // Blank the acuity display for the WHOLE therapy run, including the stage-0
  // instruction, so the Worth pattern never shows behind an activity.
  const thpPrism = therapyPrismOn && (profPrismV !== 0 || profPrismH !== 0);
  if (therapyPhase() && thpMode === 'run') {
    gl.useProgram(beamProgram);
    gl.uniform3f(locBeamFilter, 1, 1, 1);
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (thpPrism) {  // show the applied prism value (the display is blanked)
      const vpPlain = mul(projMatrix,
                          mul(viewRotMatrix,
                              translationMat(-curPos.x, -curPos.y, -curPos.z)));
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      drawText(vpPlain, -0.52, -0.34, 0.02, 0.03, 0.55, 0.85, 0.95,
               profPrismKnown
                   ? 'PRISM  V ' + profPrismV.toFixed(2) + '  H ' + profPrismH.toFixed(2)
                   : 'PRISM  not set');
      gl.disable(gl.BLEND);
    }
  }
  if (therapyPhase() && thpMode === 'run' && thpStage !== 0) {
    const vrot = thpPrism
      ? mul(prismRotationPD(rightEye, profPrismV, profPrismH), viewRotMatrix)
      : viewRotMatrix;
    const viewFull = mul(vrot,
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
    // Blank the digital acuity display for every therapy activity so the Worth
    // 4-dot pattern doesn't glow behind the target (rotation-only chart vp);
    // for the depth-tested Brock string it lands behind the beads.
    gl.uniformMatrix4fv(locBeamMvp, false, vp);
    gl.uniform4f(locBeamColor, 0, 0, 0, 1);
    gl.bindVertexArray(panelVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (thpAct === THP_CNP || thpAct === THP_DIVRANGE ||
        thpAct === THP_DIVJUMPS || thpAct === THP_SUSTAIN) {
      // Realistic Brock string: a lit twisted cord threaded through drilled
      // wooden beads, running along -Z and inclined slightly down. This is the
      // ONLY depth-tested content — enable depth (the XR framebuffer already
      // carries a depth buffer, cleared each frame), then restore the flat
      // (depth-off) pipeline for everything that follows.
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(litProgram);
      gl.uniform3f(locLitLight, 0.4, 0.7, 0.55);
      gl.uniform3f(locLitSky, 0.62, 0.64, 0.66);      // hemisphere fill (tunable)
      gl.uniform3f(locLitGround, 0.20, 0.24, 0.30);
      // Head pose = inter-eye midpoint + head orientation (view 0), so the
      // string is one shared object seen by both eyes; the -7.5 deg tilt droops
      // the cord down the head's forward (-Z). HEAD-LOCKED like a real Brock
      // string pinched to the nose (and like ../vision-home) -> a bead's
      // distance from the eyes (the vergence demand) is invariant to head motion
      // (a world-fixed far end made it squirm and unfusible).
      const p0 = eyePoses[0].pos, p1 = eyePoses[1].pos;
      const head = { x: 0.5 * (p0.x + p1.x), y: 0.5 * (p0.y + p1.y),
                     z: 0.5 * (p0.z + p1.z) };
      const rig = mul(poseMatrix(head, eyePoses[0].quat), rotationX(-0.131));
      // current eye position in the rig's LOCAL frame (for the specular V) =
      // R^T * (eyeWorld - t): invert the rigid rig transform onto the point.
      const ep = eyePoses[rightEye ? 1 : 0].pos;
      const vx = ep.x - rig[12], vy = ep.y - rig[13], vz = ep.z - rig[14];
      gl.uniform3f(locLitEyePos,
        rig[0] * vx + rig[1] * vy + rig[2] * vz,
        rig[4] * vx + rig[5] * vy + rig[6] * vz,
        rig[8] * vx + rig[9] * vy + rig[10] * vz);
      const lit = (model, r, g, b, shininess, spec, mesh) => {
        gl.uniformMatrix4fv(locLitMvp, false, mul(vpWorld, mul(rig, model)));
        gl.uniform3f(locLitColor, r, g, b);
        gl.uniform1f(locLitShininess, shininess);
        gl.uniform1f(locLitSpec, spec);
        gl.bindVertexArray(mesh.vao);
        gl.drawArrays(gl.TRIANGLES, 0, mesh.count);
      };
      // the cord runs along local -Z, length kStringLen from the eyes (matte)
      lit(translationMat(0, 0, 0), 0.85, 0.82, 0.72, 20.0, 0.15, cordMesh);
      // a single moving fixation bead at its distance-from-eyes (glossy)
      lit(translationMat(0, 0, -thpBeadZ), 0.88, 0.10, 0.08, 90.0, 0.32, beadMesh);  // saturated red, small glint
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(beamProgram);
      gl.uniform3f(locBeamFilter, 1, 1, 1);
    } else if (thpAct === THP_PRISM || thpAct === THP_VERT) {
      // a single glossy bead head-locked ~0.55 m ahead, pulled apart per eye by
      // the activity prism (the vergence demand); depth-tested + lit like the
      // Brock string so it reads as a bead
      gl.clear(gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(litProgram);
      gl.uniform3f(locLitLight, 0.4, 0.7, 0.55);
      gl.uniform3f(locLitSky, 0.62, 0.64, 0.66);
      gl.uniform3f(locLitGround, 0.20, 0.24, 0.30);
      gl.uniform1f(locLitShininess, 90.0);
      gl.uniform1f(locLitSpec, 0.32);
      const bp0 = eyePoses[0].pos, bp1 = eyePoses[1].pos;
      const bhead = { x: 0.5 * (bp0.x + bp1.x), y: 0.5 * (bp0.y + bp1.y),
                      z: 0.5 * (bp0.z + bp1.z) };
      const bd = 0.55;
      const bdx = eyeSign * (thpPrismH / 100) * bd;
      const bdy = eyeSign * (thpPrismV / 100) * bd;
      const bfr = mul(poseMatrix(bhead, eyePoses[0].quat),
                      translationMat(bdx, bdy, -bd));
      // eye-in-local for the specular V = R^T * (eyeWorld - t) (bfr is rigid)
      const bep = eyePoses[rightEye ? 1 : 0].pos;
      const bvx = bep.x - bfr[12], bvy = bep.y - bfr[13], bvz = bep.z - bfr[14];
      gl.uniform3f(locLitEyePos,
        bfr[0] * bvx + bfr[1] * bvy + bfr[2] * bvz,
        bfr[4] * bvx + bfr[5] * bvy + bfr[6] * bvz,
        bfr[8] * bvx + bfr[9] * bvy + bfr[10] * bvz);
      gl.uniformMatrix4fv(locLitMvp, false,
                          mul(vpWorld, mul(bfr, scaleMat(2.6, 2.6, 2.6))));
      gl.uniform3f(locLitColor, 0.95, 0.82, 0.22);  // yellow bead
      gl.bindVertexArray(beadMesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, beadMesh.count);
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(beamProgram);
      gl.uniform3f(locBeamFilter, 1, 1, 1);
    } else if (thpAct === THP_BOTH) {
      // dichoptic suppression check: a red dot to OD, a green dot to OS at the
      // SAME spot -> seeing both (overlapping) = both eyes on; one colour
      // missing = that eye is suppressing
      solid(mul(translationMat(0, 0, -1.2), scaleMat(0.07, 0.07, 0.07)),
            rightEye ? RED : GREEN, therapyDotVao, gl.TRIANGLES, 6);
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

// The trigger dispatcher is shared by XR select events, the gamepad path,
// and laptop-mode clicks — installed at startup (it reads globals only).
function installDoSelect() {
    doSelect = (ev) => {
      // Vision Pro (and any transient-pointer runtime): input sources exist
      // only during the pinch, so per-frame hover may be empty — classify
      // the EVENT's own target ray before acting on the hover state.
      if (ev && ev.frame && ev.inputSource && ev.inputSource.targetRaySpace) {
        const rp = ev.frame.getPose(ev.inputSource.targetRaySpace, xrRefSpace);
        if (rp) classifyAim({ pos: rp.transform.position,
                              quat: rp.transform.orientation,
                              left: ev.inputSource.handedness === 'left' });
      }
      // a test-panel hit acts immediately, even over the intro narration, so
      // START starts the first test without waiting for commentary
      const panelHit = testingPhase() && testMode === 'select' && clHit;
      if (summaryActive) { summaryActive = false; return; }  // dismiss results
      if (phase === 'profile') {
        if (kbActive) {  // virtual keyboard: build the new name
          const key = kbHit ? kbKeyAt(kbHit.col, kbHit.row) : 0;
          if (key >= 65 && key <= 90) {
            if (newName.length < 24) newName += String.fromCharCode(key);
          } else if (key === 32) {
            if (newName && newName[newName.length - 1] !== ' ' &&
                newName.length < 24) newName += ' ';
          } else if (key === 8) {
            newName = newName.slice(0, -1);
          } else if (key === 27) {
            kbActive = false; newName = '';
          } else if (key === 10) {  // OK -> create the profile
            newName = newName.replace(/ +$/, '');
            if (newName) {
              if (!profiles.includes(newName)) profiles.push(newName);
              saveProfiles();
              scopeProfile(newName);
              kbActive = false; newName = '';
              phase = 'prism';  // set this person's prism before choosing
            }
          }
        } else if (profHit && profHit.kind === 'select') {
          scopeProfile(profiles[profHit.row]); pendingDelete = -1;
          if (profPrismSaved) {  // returning person: no prism toll booth
            phase = 'choose'; playClip(CLIP_CHOOSE);
          } else {
            phase = 'prism';  // first time: set this person's prism once
          }
        } else if (profHit && profHit.kind === 'new') {
          kbActive = true; newName = ''; pendingDelete = -1;
        } else if (profHit && profHit.kind === 'delete') {
          if (pendingDelete === profHit.row) {  // second tap confirms
            const gone = profiles[profHit.row];
            try {
              localStorage.removeItem('vision.testSelection.' + profileSlug(gone));
              localStorage.removeItem('vision.therapySelection.' + profileSlug(gone));
            } catch (e) { /* ignore */ }
            profiles.splice(profHit.row, 1);
            if (profiles.length === 0) profiles.push('Guest');
            saveProfiles();
            if (!profiles.includes(activeProfile)) scopeProfile(profiles[0]);
            pendingDelete = -1;
          } else {
            pendingDelete = profHit.row;  // arm; a second tap deletes
          }
        } else {
          pendingDelete = -1;  // any other tap cancels a pending delete
        }
        return;
      }
      if (phase === 'prism') {
        if (prismHit === 'vup') { profPrismKnown = true; profPrismV = Math.min(10, profPrismV + 0.25); savePrism(); }
        else if (prismHit === 'vdn') { profPrismKnown = true; profPrismV = Math.max(-10, profPrismV - 0.25); savePrism(); }
        else if (prismHit === 'hup') { profPrismKnown = true; profPrismH = Math.min(10, profPrismH + 0.25); savePrism(); }
        else if (prismHit === 'hdn') { profPrismKnown = true; profPrismH = Math.max(-10, profPrismH - 0.25); savePrism(); }
        else if (prismHit === 'done') { savePrism(); phase = 'choose'; playClip(CLIP_CHOOSE); }
        return;
      }
      if (narrating() && !panelHit) { skipClip(); return; }
      if (menuPhase()) {
        if (prismBtnHit) { phase = 'prism'; return; }  // reopen the prism panel
        if (personBtnHit) {  // "logout": back to the Select-Player list
          phase = 'profile'; pendingDelete = -1; return;
        }
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
            } else if (clHit.kind === 'back') {  // back to the workflow chooser
              phase = 'choose'; testMode = 'select';
              playClip(CLIP_CHOOSE);
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
        } else if (inspResultPanel) {
          inspResultPanel = false; advanceRun();  // dismiss + next test
        } else if (inspActive) {
          // subjective self-report: the H split into two at this position
          playClick();  // audible confirm: press was registered
          inspFlashFrames = 10;  // + visual: flash the H green briefly
          if (playingClip >= 0) skipClip();
          else {
            const s = inspStepJS(inspT);
            if (s.index >= 0 && s.index < 9) inspDip[s.index] = true;
          }
        } else if (eyeActive) {
          // subjective self-report: doubled / lost it. Tag the nearest gaze
          // position (a moving-target diplopia field), or skip a clip.
          if (playingClip >= 0) { skipClip(); }
          else {
            playClick(); eyeFlash = 10; eyeFlagged = true;
            eyeDip[Math.max(0, Math.min(8, Math.round(eyeT / 1.3)))] = true;
          }
        } else if (pvActive) {  // prism verification: skip / snap / finish
          if (playingClip >= 0) skipClip();
          else if (pvStage === 1) { pvStage = 2; pvT = 0; playClip(CLIP_PV_RESULT); }
          else {  // the result press saves the found prism as this person's default
            // base-out reads +ve (Rx convention: H = +1.0 base-out); the
            // horizontal deviation is measured -ve, so negate.
            profPrismKnown = true;
            profPrismV = Math.max(-10, Math.min(10, estV));
            profPrismH = Math.max(-10, Math.min(10, -estH));
            savePrism();  // savePrism() quantizes to 0.25
            finishRun();
          }
        } else {
          advanceRun();   // run mode: trigger advances to the next test
        }
        return;
      }
      if (gamesPhase()) { gamesTrigger(); return; }
      if (therapyPhase()) therapyTrigger();
    };
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
    const aX = edge(4), bY = edge(5), thumb = edge(3), grip = edge(1);
    edge(0);                             // keep the trigger edge current
    if (testingPhase()) {
      if (aX) toggleBeams();
      if (bY) cyclePrism();
      if (thumb) toggleFilters();
    }
    // Vision Games in immersive WebXR: grip / B / Y exits a running game back
    // to the games panel (the trigger is the flap; mirrors the native menu
    // button). The M-key exit only exists on desktop/laptop.
    if (gamesPhase() && gmMode === 'run' && (grip || bY)) gamesEndRun();
    btnPrev.set(src, prev);
  }
}

// Classify one aim ray against the panels of the current phase, folding the
// result into the hover state. Called per frame for each persistent input
// source AND from the select event with the event-frame ray — on Vision Pro
// (transient-pointer: gaze + pinch) sources exist only during the pinch, so
// the event-frame ray is the ONLY reliable one.
function classifyAim(a) {
  const profRows = Math.min(profiles.length, PROFILE_MAXROWS);
  if (phase === 'profile') {
    if (kbActive) {
      const k = keyboardHit(a.pos, a.quat);
      if (k && (!kbHit || !a.left)) kbHit = k;
    } else {
      const h = profilePanelHit(a.pos, a.quat, profRows);
      if (h && (!profHit || !a.left)) profHit = h;
    }
  } else if (phase === 'prism') {
    const k = prismPanelHit(a.pos, a.quat);
    if (k && (!prismHit || !a.left)) prismHit = k;
  } else if (menuPhase()) {
    const r = workflowHitRow(a.pos, a.quat);
    if (r >= 0 && (workflowHovered < 0 || !a.left)) workflowHovered = r;
    const uv = menuHitUV(a.pos, a.quat, CHECKLIST_DIST, PROFILE_W, PROFILE_H);
    if (uv && uv.v > 0.88 && uv.v < 0.96) {  // vlogic::workflowButtonAt
      if (uv.u > 0.06 && uv.u < 0.48) prismBtnHit = true;
      else if (uv.u > 0.52 && uv.u < 0.94) personBtnHit = true;
    }
  } else if (testingPhase()) {
    const h = checklistHit(a.pos, a.quat);
    if (h && (!clHit || !a.left)) clHit = h;
  } else if (therapyPhase()) {
    const h = therapyPanelHit(a.pos, a.quat);
    if (h && (!thpHit || !a.left)) thpHit = h;
    const uv = menuHitUV(a.pos, a.quat, CHECKLIST_DIST, CHECKLIST_W, CHECKLIST_H);
    if (uv && uv.v > 0.035 && uv.v < 0.11 && uv.u > 0.58 && uv.u < 0.98)
      thpPrismToggleHit = true;
  } else if (gamesPhase() && gmMode === 'select') {
    const h = gamesPanelHit(a.pos, a.quat);
    if (h && (!gmHit || !a.left)) gmHit = h;
  }
}

function onXRFrame(_t, frame) {
  const session = frame.session;
  session.requestAnimationFrame(onXRFrame);
  const pose = frame.getViewerPose(xrRefSpace);
  if (!pose) return;
  if (!refRebased) {
    // Emulate OpenXR LOCAL semantics: origin at the viewer's first pose.
    // visionOS Safari's 'local' can anchor away from head height, which sank
    // the world panels well below the (head-centred) acuity display.
    refRebased = true;
    const p = pose.transform.position;
    if (Math.hypot(p.x, p.y, p.z) > 0.05) {
      xrRefSpace = xrRefSpace.getOffsetReferenceSpace(
          new XRRigidTransform({ x: p.x, y: p.y, z: p.z }));
      return;  // re-fetch poses from the rebased space next frame
    }
  }

  pollControllers(session);
  // Gamepad face button A (right) = primary interact; X (left) = toggle beams.
  // A/X is button index 4 on the Quest Touch profile. Rising-edge; guarded for
  // devices without a gamepad. select/pinch stays a fallback.
  let aDown = false, xDown = false;
  for (const src of session.inputSources) {
    const gp = src.gamepad;
    if (!gp || !gp.buttons || gp.buttons.length <= 4) continue;
    const pressed = !!gp.buttons[4].pressed;
    if (src.handedness === 'right') aDown = aDown || pressed;
    else if (src.handedness === 'left') xDown = xDown || pressed;
  }
  if (aDown && !gpPrev.a && doSelect) doSelect({ inputSource: { handedness: 'right' } });
  if (xDown && !gpPrev.x) toggleBeams();
  gpPrev.a = aDown; gpPrev.x = xDown;
  advancePhase();
  updateTestDim();
  updateTherapyClock();
  gamesUpdate();
  updateInspection(pose.transform.orientation);
  updateCover();
  updateEye();
  updateWorth();
  updateMaddox();
  updateVg();
  updateDm();
  updatePv();

  // controller aim rays -> hovered element on whichever panel is active
  resetHover();
  let profRows = Math.min(profiles.length, PROFILE_MAXROWS);
  if (menuPhase() || (testingPhase() && testMode === 'select') ||
      (therapyPhase() && thpMode === 'select') ||
      (gamesPhase() && gmMode === 'select') || phase === 'profile' ||
      phase === 'prism') {
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
      classifyAim(a);
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
    anaglyph = false;  // a real headset takes over from web display modes
    sbs = false;
    setMono(0);
    xrSession = await navigator.xr.requestSession('immersive-vr');
    xrSession.updateRenderState({
      baseLayer: new XRWebGLLayer(xrSession, gl),
    });
    // LOCAL space: -Z is wherever the user faces at session start, so the
    // eye-chart wall begins directly in front of them.
    xrRefSpace = await xrSession.requestReferenceSpace('local');
    refRebased = false;  // rebase to the first viewer pose of this session

    // trigger is phase-routed: skip narration / pick a workflow / toggle a
    // checklist row (or lights/prism) / advance a therapy exercise
    xrSession.addEventListener('select', doSelect);
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

// clear the per-frame hover state (XR frame loop + laptop preview loop)
function resetHover() {
  aimPoses = [];
  clHit = null;
  thpHit = null;
  profHit = null;
  kbHit = null;
  prismHit = null;
  prismBtnHit = false; personBtnHit = false;
  thpPrismToggleHit = false;
  workflowHovered = -1;
  gmHit = null;
}

// set the grayscale-for-anaglyph uniform on every program that draws colour
function setMono(v) {
  for (const p of [skyProgram, labelProgram, beamProgram, panelProgram,
                   litProgram, textProgram]) {
    if (!p) continue;
    gl.useProgram(p);
    const loc = gl.getUniformLocation(p, 'uMono');
    if (loc) gl.uniform1f(loc, v);
  }
}

// preview-mode aim ray from the mouse position: unproject through the
// preview camera, then express as a pose for the same hit classifiers the
// XR path uses (yaw/pitch whose forward is the ray direction).
function previewAim(aspect) {
  if (mouseX < 0) return null;
  const tanH = Math.tan((80 * Math.PI / 180) / 2);
  // in SBS each eye occupies half the width; map the pointer into its half so
  // aiming works by pointing at either eye's copy of the UI
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  // in a 3D-TV mode each eye occupies half the frame; map the pointer into its
  // own half so aiming works on either eye's copy (SBS = X halves, TB = Y).
  let px = mouseX, denomX = cw, py = mouseY, denomY = ch;
  if (sbs && tbFormat) { py = mouseY >= ch / 2 ? mouseY - ch / 2 : mouseY; denomY = ch / 2; }
  else if (sbs) { px = mouseX >= cw / 2 ? mouseX - cw / 2 : mouseX; denomX = cw / 2; }
  const ndcX = (px / denomX) * 2 - 1;
  const ndcY = 1 - (py / denomY) * 2;
  let dx = ndcX * tanH * aspect, dy = ndcY * tanH, dz = -1;
  // rotate view-space ray into world: Ry(yaw) * Rx(pitch)
  const cy = Math.cos(previewYaw), sy = Math.sin(previewYaw);
  const cp = Math.cos(previewPitch), sp = Math.sin(previewPitch);
  const y1 = dy * cp - dz * sp, z1 = dy * sp + dz * cp, x1 = dx;
  const wx = x1 * cy + z1 * sy, wz = -x1 * sy + z1 * cy, wy = y1;
  const len = Math.hypot(wx, wy, wz);
  const yaw = Math.atan2(-wx, -wz), pitch = Math.asin(wy / len);
  return { pos: { x: 0, y: 0, z: 0 }, quat: quatFromYawPitch(yaw, pitch),
           left: false };
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
  gamesUpdate();
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
  updatePv();
  // A real interocular offset (not zero IPD): flat dichoptic tests (Worth,
  // Maddox, cover) are drawn in the rotation-only vp so they ignore eye
  // POSITION and stay screen-fixed; but the Brock-string vergence therapy
  // reads eyePoses[..].pos for depth — a zero-IPD camera would flatten it.
  // Half-IPD along the camera's right axis (yaw only; pitch leaves X alone).
  const HALF_IPD = 0.0315;
  const rgt = { x: Math.cos(previewYaw), y: 0, z: -Math.sin(previewYaw) };
  const lPos = { x: -HALF_IPD * rgt.x, y: 0, z: -HALF_IPD * rgt.z };
  const rPos = { x: HALF_IPD * rgt.x, y: 0, z: HALF_IPD * rgt.z };
  const eyePoses = [{ pos: lPos, quat }, { pos: rPos, quat }];  // 0=OS,1=OD

  // Worth 4-dot hint: the audio is positional, so name the per-eye colours.
  // Anaglyph colours follow the lenses; SBS shows real red/green (spatial).
  if ((anaglyph || sbs) && worthActive && !anaWorthMsg) {
    const nm = sbs ? ['green', 'red'] : (ANA_NAMES[anaPreset] || ANA_NAMES[0]);
    setAnaHint('Worth dots — top: ' + nm[1] + ' (right eye) · sides: ' +
               nm[0] + ' (left eye) · bottom: white (both eyes).');
    anaWorthMsg = true;
  } else if (anaWorthMsg && !worthActive) {
    setAnaHint('');
    anaWorthMsg = false;
  }

  // mouse = the aim ray: same hover classifiers as the XR path, so panels
  // highlight and clicks land wherever the pointer points
  resetHover();
  const aim = previewAim(w / h);
  if (aim) { aimPoses.push(aim); classifyAim(aim); }

  if (sbs) {
    // 3D TV: true per-eye content into two half viewports at full aspect (the
    // TV stretches each half back out and interlaces it for the glasses).
    // Full colour, spatial — like a headset. Side-by-Side squeezes each eye
    // into a half-WIDTH viewport; Top-Bottom into a half-HEIGHT one (GL y=0 is
    // the bottom, so the left eye takes the upper half).
    if (tbFormat) {
      const half = h >> 1;
      gl.viewport(0, half, w, half);     // left eye -> top half
      drawScene(proj, viewRot, false, lPos, eyePoses, 1);
      gl.viewport(0, 0, w, half);        // right eye -> bottom half
      drawScene(proj, viewRot, true, rPos, eyePoses, 1);
    } else {
      const half = w >> 1;
      gl.viewport(0, 0, half, h);        // left eye -> left half
      drawScene(proj, viewRot, false, lPos, eyePoses, 1);
      gl.viewport(half, 0, half, h);     // right eye -> right half
      drawScene(proj, viewRot, true, rPos, eyePoses, 1);
    }
    gl.viewport(0, 0, w, h);
  } else if (anaglyph && worthActive) {
    // Worth is the special case: the dots are drawn ONCE in real lens-matched
    // colours and the physical glasses do the dissociation (like the original
    // coloured-filter test) — no grayscale, no channel mask. See drawScene's
    // Worth block for the top/side/bottom colours per preset.
    setMono(0);
    gl.colorMask(true, true, true, true);
    drawScene(proj, viewRot, false, lPos, eyePoses, 1);
  } else if (anaglyph) {
    // everything else: two passes into the lens channels; grayscale so no
    // stimulus can vanish from the eye whose channels exclude its colour
    const m = ANA_MASKS[anaPreset] || ANA_MASKS[0];
    setMono(1);
    gl.colorMask(!!m[0][0], !!m[0][1], !!m[0][2], true);
    drawScene(proj, viewRot, false, lPos, eyePoses, 1);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.colorMask(!!m[1][0], !!m[1][1], !!m[1][2], true);
    drawScene(proj, viewRot, true, rPos, eyePoses, 1);
    gl.colorMask(true, true, true, true);
    setMono(0);
  } else {
    drawScene(proj, viewRot, false, { x: 0, y: 0, z: 0 }, eyePoses, 1);
  }
}

function setMessage(text) {
  const el = document.getElementById('message');
  if (el) el.textContent = text;
}

// a slim hint bar shown in glasses mode (the chrome footer is hidden there)
function setAnaHint(text) {
  const el = document.getElementById('ana-hint');
  if (el) { el.textContent = text; el.style.display = text ? 'block' : 'none'; }
}

// enter/leave an immersive web display mode ('anaglyph' | 'sbs' | 'none').
// Entering hides the browser chrome and goes fullscreen so it reads as a mode
// (answering "why do I still see the footer"); leaving restores everything.
// Esc / exiting fullscreen also leaves.
function setWebImmersive(mode) {
  anaglyph = mode === 'anaglyph';
  sbs = mode === 'sbs';
  const on = mode !== 'none';
  const ov = document.getElementById('overlay');
  if (ov) ov.style.display = on ? 'none' : '';
  if (on) {
    if (canvas.requestFullscreen) canvas.requestFullscreen().catch(() => {});
  } else {
    setAnaHint('');
    anaWorthMsg = false;
    setMono(0);
    if (document.fullscreenElement && document.exitFullscreen)
      document.exitFullscreen().catch(() => {});
  }
}

// ---------------------------------------------------------------- init
async function main() {
  installDoSelect();
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
  // Realistic Brock string: Lambert program + drilled bead / twisted cord.
  litProgram = buildProgram(LIT_VS, LIT_FS);
  locLitMvp = gl.getUniformLocation(litProgram, 'uMvp');
  locLitColor = gl.getUniformLocation(litProgram, 'uColor');
  locLitLight = gl.getUniformLocation(litProgram, 'uLightDir');
  locLitEyePos = gl.getUniformLocation(litProgram, 'uEyePos');
  locLitSky = gl.getUniformLocation(litProgram, 'uSky');
  locLitGround = gl.getUniformLocation(litProgram, 'uGround');
  locLitShininess = gl.getUniformLocation(litProgram, 'uShininess');
  locLitSpec = gl.getUniformLocation(litProgram, 'uSpec');
  beadMesh = buildBeadVao(0.011, 0.002);  // 22mm bead, 4mm bore
  cordMesh = buildCordVao(1.6);           // covers the far divergence beads
  checklistPanelVao = buildChecklistPanelVao();
  checklistMarkVao = buildFilledQuadVao(CHECKLIST_BOX_HALF_U * CHECKLIST_W * 0.55);
  checklistCaretVao = buildCaretVao();
  workflowPanelVao = buildMenuPanelVao(WORKFLOW_W, WORKFLOW_H);
  therapyDotVao = buildFilledQuadVao(1.0);  // unit quad, scaled per target
  // runtime text: monospace atlas + glyph program (session summary, profiles)
  textProgram = buildProgram(TEXT_VS, TEXT_FS);
  locTextMvp = gl.getUniformLocation(textProgram, 'uMvp');
  locTextUvRect = gl.getUniformLocation(textProgram, 'uUvRect');
  locTextColor = gl.getUniformLocation(textProgram, 'uColor');
  locTextTex = gl.getUniformLocation(textProgram, 'uTex');
  textQuadVao = buildTextQuad();

  loadProfiles();
  if (profiles.length === 0) profiles.push('Guest');
  scopeProfile(profiles[0]);  // scopes + loads per-profile test/therapy prefs

  setMessage('Loading skybox…');
  [texBright, texDim, texLabels, texDisclaimer, texChecklist, texWorkflow,
   texTitleCards, texThpChecklist, texThpTitle, texFont,
   texGmChecklist] = await Promise.all([
    loadCubemap('assets/skybox'),
    loadCubemap('assets/skybox_dim'),
    loadTexture2D('assets/prism_labels.png'),
    loadTexture2D('assets/disclaimer.png').catch(() => null),
    loadTexture2D('assets/checklist.png').catch(() => null),
    loadTexture2D('assets/workflows.png').catch(() => null),
    loadTexture2D('assets/titlecards.png').catch(() => null),
    loadTexture2D('assets/checklist_therapy.png').catch(() => null),
    loadTexture2D('assets/titlecards_therapy.png').catch(() => null),
    loadTexture2D('assets/font_atlas.png').catch(() => null),
    loadTexture2D('assets/checklist_games.png').catch(() => null),
  ]);
  initIntroAudio(); // fire-and-forget; degrades to silent if it fails
  setMessage('▲ Choose a viewing mode above to begin '
             + '(Enter VR / Laptop + 3D glasses / 3D TV). '
             + 'Flat preview is look-around only.');
  updateStatus();

  // Enter VR button
  // ---- anaglyph laptop mode: setup screen wiring ----
  const ANA_CSS = [
    ['#d22', '#0cc'], ['#0cc', '#d22'], ['#d22', '#22d'],
    ['#22d', '#d22'], ['#d22', '#2b2'], ['#2b2', '#d22'],
  ];
  const anaSetup = document.getElementById('ana-setup');
  const anaSel = document.getElementById('ana-preset');
  const refreshSwatches = () => {
    const c = ANA_CSS[anaPreset] || ANA_CSS[0];
    const l = document.getElementById('ana-left-swatch');
    const r = document.getElementById('ana-right-swatch');
    if (l) l.style.background = c[0];
    if (r) r.style.background = c[1];
    if (anaSel) anaSel.value = String(anaPreset);
    const w = document.getElementById('ana-worth-note');
    const nm = ANA_NAMES[anaPreset] || ANA_NAMES[0];
    if (w) w.textContent = 'In the Worth 4-dot test, the top dot will look ' +
        nm[1] + ', the two side dots ' + nm[0] +
        ', and the bottom dot white-ish.';
  };
  try {
    const sv = parseInt(localStorage.getItem('vision.anaPreset'), 10);
    if (sv >= 0 && sv < ANA_MASKS.length) anaPreset = sv;
  } catch (e) { /* ignore */ }
  const anaBtn = document.getElementById('enter-ana');
  if (anaBtn && anaSetup) {
    anaBtn.addEventListener('click', () => {
      refreshSwatches();
      anaSetup.classList.add('open');
    });
    anaSel.addEventListener('change', () => {
      anaPreset = parseInt(anaSel.value, 10) || 0;
      refreshSwatches();
    });
    document.getElementById('ana-swap').addEventListener('click', () => {
      anaPreset = anaPreset ^ 1;  // presets are laid out in swap pairs
      refreshSwatches();
    });
    document.getElementById('ana-cancel').addEventListener('click', () => {
      anaSetup.classList.remove('open');
    });
    document.getElementById('ana-start').addEventListener('click', () => {
      try { localStorage.setItem('vision.anaPreset', String(anaPreset)); }
      catch (e) { /* ignore */ }
      anaSetup.classList.remove('open');
      setWebImmersive('anaglyph');  // hide chrome + fullscreen (a real mode)
      startPhases();  // the click is the user gesture: narration may sound
    });
  }

  // ---- side-by-side 3D TV mode wiring ----
  const sbsSetup = document.getElementById('sbs-setup');
  const sbsBtn = document.getElementById('enter-sbs');
  if (sbsBtn && sbsSetup) {
    sbsBtn.addEventListener('click', () => sbsSetup.classList.add('open'));
    document.getElementById('sbs-cancel').addEventListener('click', () => {
      sbsSetup.classList.remove('open');
    });
    // format toggle: Side-by-Side (default) vs Top-Bottom (over/under)
    const fmtSbs = document.getElementById('sbs-fmt-sbs');
    const fmtTb = document.getElementById('sbs-fmt-tb');
    const setFmt = (tb) => {
      tbFormat = tb;
      if (fmtSbs) fmtSbs.classList.toggle('fmt-on', !tb);
      if (fmtTb) fmtTb.classList.toggle('fmt-on', tb);
    };
    if (fmtSbs) fmtSbs.addEventListener('click', () => setFmt(false));
    if (fmtTb) fmtTb.addEventListener('click', () => setFmt(true));
    document.getElementById('sbs-start').addEventListener('click', () => {
      sbsSetup.classList.remove('open');
      setWebImmersive('sbs');  // fullscreen frame-packed 3D for the TV
      startPhases();
    });
  }

  // Esc leaves glasses mode; so does the user exiting fullscreen by any means
  // Esc = full reset back to the start (the "choose a viewing mode" landing).
  // A page reload is the robust way to clear every phase / test / game / mode
  // state at once — no risk of stale in-progress state leaking across.
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { location.reload(); return; }
    if ((e.key === 'm' || e.key === 'M') && gamesPhase() && gmMode === 'run')
      gamesEndRun();
  });
  // Leaving fullscreen while in a glasses/TV mode (e.g. the browser eats the
  // Esc to exit fullscreen before our keydown sees it) is also a full reset.
  document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement && (anaglyph || sbs)) location.reload();
  });

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
  let dragging = false, lastX = 0, lastY = 0, downX = 0, downY = 0,
      dragDist = 0;
  canvas.addEventListener('pointerdown', (e) => {
    // NOTE: no startPhases() here — the flat preview is look-around only.
    // The guided flow starts ONLY from a chosen viewing mode (Enter VR /
    // Laptop + 3D glasses / 3D TV), because the tests are dichoptic and are
    // meaningless without one. See setWebImmersive / enterVR.
    dragging = true;
    lastX = downX = e.clientX;
    lastY = downY = e.clientY;
    dragDist = 0;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    mouseX = e.clientX;  // pointer = the preview aim ray, drag or not
    mouseY = e.clientY;
    if (!dragging) return;
    previewYaw -= (e.clientX - lastX) * 0.005;
    previewPitch -= (e.clientY - lastY) * 0.005;
    previewPitch = Math.max(-1.5, Math.min(1.5, previewPitch));
    dragDist += Math.abs(e.clientX - lastX) + Math.abs(e.clientY - lastY);
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('pointerup', (e) => {
    dragging = false;
    // a click (no meaningful drag) is the trigger: same dispatch as an XR
    // select — panels, diplopia presses, advances, everything
    if (dragDist < 6 && doSelect && !xrSession) doSelect({});
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === ' ') {
      if (narrating()) { skipClip(); return; }
      if (gamesPhase()) { gamesTrigger(); return; }
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
          if (playingClip >= 0) { skipClip(); }
          else {
            playClick(); eyeFlash = 10; eyeFlagged = true;
            eyeDip[Math.max(0, Math.min(8, Math.round(eyeT / 1.3)))] = true;
          }
        } else if (pvActive) {  // prism verification: skip / snap / finish
          if (playingClip >= 0) skipClip();
          else if (pvStage === 1) { pvStage = 2; pvT = 0; playClip(CLIP_PV_RESULT); }
          else finishRun();
        } else advanceRun();
        return;
      }
    }
    if (menuPhase()) {
      if (e.key === '1') { chooseWorkflow(0); return; }
      if (e.key === '2') { chooseWorkflow(1); return; }
      if (e.key === '3') { chooseWorkflow(2); return; }
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
