#!/bin/sh
# Generate the welcome/introduction narration WAVs (macOS: say + afconvert).
# Output: assets/audio/{welcome,intro}.wav — 24 kHz mono 16-bit PCM, decoded
# and resampled by miniaudio at runtime (see wiki/intro-audio-and-disclaimer.md).
# Set VOICE to override the system voice, e.g. VOICE=Samantha ./tools/generate_audio.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p assets/audio
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

speak() { # $1 = basename, $2 = text
  say ${VOICE:+-v "$VOICE"} -o "$TMP/$1.aiff" "$2"
  afconvert -f WAVE -d LEI16@24000 -c 1 "$TMP/$1.aiff" "assets/audio/$1.wav"
  echo "wrote assets/audio/$1.wav"
}

speak welcome "Welcome to Haddley Optometry. Please make yourself comfortable — your virtual exam room is ready. Before we begin, please listen to this short disclaimer. You can skip any part of this introduction by clicking either controller's trigger, or by pinching with either hand."

# The disclaimer wording lives in tools/disclaimer.txt so the spoken clip
# and the on-screen panel (build_disclaimer in generate_skybox.py) stay
# identical. Only the spoken title differs from the panel's heading.
speak disclaimer "Important disclaimer. $(cat "$(dirname "$0")/disclaimer.txt")"

speak intro "This room recreates a double vision assessment. In a real consultation an eye doctor builds a complete picture of how your eyes work together, one test at a time — so let's walk through what each test is for. \
First, a simple inspection. The doctor looks at your eyelids, your pupils, and how you naturally hold your head. A drooping lid, uneven pupils, or a habitual head tilt are small clues that often point toward a cause. \
Next, eye movement testing. You follow a target through every direction of gaze, so the doctor can spot an individual eye muscle that is weak or restricted, and see whether your double vision changes as you look around. \
Then the cover tests. By covering one eye and watching how the other moves, the doctor separates a misalignment your brain is quietly holding in check from one it cannot control — and measures its size in prism dioptres using a prism bar. \
For vertical double vision, a red glass test and the Parks three step test help identify which eye, and which specific muscle, is responsible. \
The screen in front of you shows the Worth Four Dot test: a red dot, two green dots, and a white dot. Viewed through a red filter over the right eye and a green filter over the left, it reveals whether your brain fuses both images, ignores one eye — which we call suppression — or sees double. Press a thumbstick, or the F key, to put those filters on and watch what changes. \
Finally, refined measurements such as the Maddox rod and fixation disparity tests estimate how much prism restores single, comfortable vision. Because vertical misalignment is the least comfortable to live with, it is usually measured and corrected first. \
So what does prism do? A prism lens bends light toward its base, shifting each eye's image so your eyes no longer fight to fuse. Prescribed well, it can relieve double vision, eye strain, and headaches, especially for vertical misalignments like the one this headset simulates. But prism is not a cure-all. The eyes can slowly adapt around it, the evidence is strongest for vertical deviations and weaker for horizontal ones, and stronger prisms mean thicker, heavier lenses with colour fringing. That is why a careful eye doctor prescribes the smallest amount that does the job, and often trials it before committing. \
This headset simulates one real prescription: five and a half prism dioptres base down in the right eye, five and a half base up in the left, and one dioptre base out in each — an eleven dioptre vertical correction split between the eyes so the lenses stay wearable. The menu button changes the simulated strength, the trigger toggles the room lights, the A or X button shows where each eye is looking, and the thumbstick adds the red and green test filters. \
In a few seconds the lights will dim, and your virtual exam will begin."
