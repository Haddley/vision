# WebXR Optometrist Office Skybox

WebXR port of [openxr-skybox](https://github.com/Haddley/openxr-skybox):
stand inside a stylized optometrist's office rendered as a cubemap skybox,
dim the lights for an eye test, and simulate a prism prescription — from a
browser, no install.

**Live**: https://haddley.github.io/webxr-skybox/

Plain WebGL2 + WebXR, no frameworks — the code mirrors the native app's
shaders, matrix math, and cubemap conventions.

## Using it

- **Meta Quest Browser**: open the URL, tap **Enter VR**.
- **Apple Vision Pro** (Safari, visionOS 2+): open the URL, tap
  **Enter VR**; pinch gestures act as select.
- **Desktop browser**: drag to look around (no headset needed).

Controls in VR:

- **Right-hand pinch / trigger** — toggle the exam-room lights.
- **Left-hand pinch / trigger, or grip squeeze** — cycle the prism
  simulation (100% → 125% → off → 25% → 50% → 75%); the current strength is
  displayed at the bottom of the digital eye chart.

Desktop keys: <kbd>L</kbd> lights, <kbd>P</kbd> cycle prism,
<kbd>[</kbd>/<kbd>]</kbd> fine-tune ±5%.

The prism simulation applies the wearer's prescription (OD 5.50Δ base-down
+ 1.00Δ base-out, OS 5.50Δ base-up + 1.00Δ base-out) as per-eye view
rotations — the image shifts toward the prism apex, 1Δ = atan(1/100).

## Local development

WebXR needs a secure context; localhost qualifies:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

To test on a Quest over USB without deploying:

```sh
adb reverse tcp:8000 tcp:8000   # then open http://localhost:8000 in Quest Browser
```

## Regenerating the skybox

Same pipeline as the native project (needs Pillow; numpy for previews):

```sh
python3 tools/generate_skybox.py   # writes assets/skybox*, prism_labels.png
python3 tools/preview.py .         # desktop sanity renders
```
