# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WebXR port of the native [openxr-skybox](https://github.com/Haddley/openxr-skybox)
project — an optometrist-office cubemap skybox with two lighting states and
a per-eye prism-prescription simulation. Plain WebGL2 + WebXR in two files
(`index.html`, `app.js`), no frameworks, no build step. Deployed to GitHub
Pages (main branch, root): https://haddley.github.io/webxr-skybox/

**Keep parity with the native repo**: shaders, matrix math, cubemap
conventions, prism constants, and the label-atlas overlay are line-for-line
ports of `openxr-skybox/src/*.cpp`. A logic change there should be mirrored
here (and vice versa). `assets/` and `tools/` are copies of the native
repo's — regenerate with `python3 tools/generate_skybox.py` after edits and
copy changes between repos.

## Commands

```sh
python3 -m http.server 8000        # local dev (WebXR works on localhost)
adb reverse tcp:8000 tcp:8000      # test on Quest Browser without deploying
node --check app.js                # syntax check
python3 tools/generate_skybox.py   # regenerate assets (needs Pillow)
git push                           # deploys — Pages serves main/root directly
```

## Architecture notes

- Cubemaps upload with default UNPACK_FLIP_Y (false), matching the
  pre-oriented faces; textures are RGBA8 (not sRGB) because XRWebGLLayer
  does no linear→sRGB encode — decode/encode cancel and colors pass through
  as authored.
- LOCAL reference space so the eye-chart wall (-Z) faces the user at
  session start.
- Per-view prism rotation: `view.eye === 'right'` selects the OD sign.
- Input mapping (Vision Pro has no squeeze, so handedness splits duties):
  right-hand select = lights, left-hand select or squeeze = prism cycle.
- The prism readout quad and `assets/prism_labels.png` atlas follow the
  native repo's layout: band at chart pixels (724..1324, 1076..1156) on the
  -Z wall, 41 rows, one per 5% step.
- The non-XR preview path (`onPreviewFrame`) reuses the same drawScene; it
  is the only part testable without a headset.
