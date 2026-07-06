# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

WebXR port of the native [openxr-skybox](https://github.com/Haddley/openxr-skybox)
project — an optometrist-office cubemap skybox with two lighting states and
a per-eye prism-prescription simulation. Plain WebGL2 + WebXR in two files
(`index.html`, `app.js`), no frameworks, no build step. Deployed to GitHub
Pages (main branch, root): https://haddley.github.io/visiontest/

**Keep parity with the native repo**: shaders, matrix math, cubemap
conventions, prism constants, and the label-atlas overlay are line-for-line
ports of `openxr-skybox/src/*.cpp`. A logic change there should be mirrored
here (and vice versa), with the ported commit referencing the source one
(`parity: <desc> (mirrors openxr-skybox@<sha>)`).

The native repo is the **source of truth for `tools/` and `assets/`** —
never edit those here. Edit in `openxr-skybox`, regenerate with
`python3 tools/generate_skybox.py` there, then run `./sync-from-native.sh`
to copy them over. `./sync-from-native.sh --check` verifies the copies are
identical AND that the hand-ported constants (prism prescription, step
table, label-atlas layout, readout quad corners) match across `main.cpp`,
`android_main.cpp`, `app.js`, and `generate_skybox.py` — run it before
pushing either repo.

## Commands

```sh
python3 -m http.server 8000        # local dev (WebXR works on localhost)
adb reverse tcp:8000 tcp:8000      # test on Quest Browser without deploying
node --check app.js                # syntax check
./sync-from-native.sh              # pull tools/+assets/ from ../openxr-skybox, verify parity
./sync-from-native.sh --check      # verify only — run before pushing either repo
git push                           # deploys — Pages serves main/root directly
```

If the automatic "pages build and deployment" run fails with "Deployment
failed, try again later" (transient GitHub error), retry with:

```sh
gh api -X POST repos/Haddley/visiontest/pages/builds
gh api repos/Haddley/visiontest/pages/builds/latest --jq .status   # wait for "built"
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
