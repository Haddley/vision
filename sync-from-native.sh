#!/bin/bash
# Sync shared files from the canonical openxr-skybox checkout and check
# cross-repo parity. openxr-skybox is the source of truth for tools/ and
# assets/; the ported logic constants (prism prescription, step table,
# label atlas layout, readout quad) must stay equal across main.cpp,
# android_main.cpp, and app.js.
#
# Usage:
#   ./sync-from-native.sh [--check] [path-to-openxr-skybox]
#
#   default   rsync tools/ and assets/ from the native repo, then verify
#   --check   verify only (no copying); exit 1 on any drift
#
# The native repo defaults to ../openxr-skybox next to this checkout.
set -u

WEB_ROOT="$(cd "$(dirname "$0")" && pwd)"
CHECK_ONLY=0
NATIVE=""

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) NATIVE="$arg" ;;
  esac
done
NATIVE="${NATIVE:-$WEB_ROOT/../openxr-skybox}"

if [ ! -f "$NATIVE/src/main.cpp" ]; then
  echo "error: openxr-skybox checkout not found at: $NATIVE" >&2
  echo "       pass its path as an argument" >&2
  exit 2
fi
NATIVE="$(cd "$NATIVE" && pwd)"

FAIL=0

if [ "$CHECK_ONLY" -eq 0 ]; then
  echo "syncing tools/ and assets/ from $NATIVE"
  rsync -a --delete "$NATIVE/tools/" "$WEB_ROOT/tools/"
  rsync -a --delete "$NATIVE/assets/" "$WEB_ROOT/assets/"
fi

echo "checking copied files (tools/, assets/)..."
if ! diff -rq "$NATIVE/tools" "$WEB_ROOT/tools" || \
   ! diff -rq "$NATIVE/assets" "$WEB_ROOT/assets"; then
  echo "DRIFT: copied files differ from native repo (run without --check to sync)"
  FAIL=1
fi

echo "checking ported constants..."
python3 - "$NATIVE" "$WEB_ROOT" <<'PYEOF' || FAIL=1
import re, sys

native, web = sys.argv[1], sys.argv[2]
files = {
    "main.cpp": open(f"{native}/src/main.cpp").read(),
    "android_main.cpp": open(f"{native}/src/android_main.cpp").read(),
    "app.js": open(f"{web}/app.js").read(),
    "generate_skybox.py": open(f"{native}/tools/generate_skybox.py").read(),
}

def grab(name, pattern, conv=float):
    m = re.search(pattern, files[name])
    if not m:
        return f"<pattern not found in {name}>"
    g = m.groups() if len(m.groups()) > 1 else m.group(1)
    return tuple(conv(x) for x in g) if isinstance(g, tuple) else conv(g)

def floats(s):
    return tuple(float(x) for x in re.findall(r"-?[\d.]+", s))

# each check: description -> {file: extracted value}; all values must be equal
checks = {
    "prism vertical PD": {
        "main.cpp": grab("main.cpp", r"kPrismVerticalPD = ([\d.]+)f"),
        "android_main.cpp": grab("android_main.cpp", r"kPrismVerticalPD = ([\d.]+)f"),
        "app.js": grab("app.js", r"PRISM_VERTICAL_PD = ([\d.]+)"),
        "generate_skybox.py": grab("generate_skybox.py",
                                   r"([\d.]+) \* pct / 100\.0, [\d.]+ \* pct / 100\.0"),
    },
    "prism horizontal PD": {
        "main.cpp": grab("main.cpp", r"kPrismHorizontalPD = ([\d.]+)f"),
        "android_main.cpp": grab("android_main.cpp", r"kPrismHorizontalPD = ([\d.]+)f"),
        "app.js": grab("app.js", r"PRISM_HORIZONTAL_PD = ([\d.]+)"),
        "generate_skybox.py": grab("generate_skybox.py",
                                   r"[\d.]+ \* pct / 100\.0, ([\d.]+) \* pct / 100\.0"),
    },
    "prism step table": {
        "main.cpp": grab("main.cpp", r"kPrismSteps\[\] = \{([^}]*)\}", floats),
        "android_main.cpp": grab("android_main.cpp", r"kPrismSteps\[\] = \{([^}]*)\}", floats),
        "app.js": grab("app.js", r"PRISM_STEPS = \[([^\]]*)\]", floats),
    },
    "label atlas rows": {
        "main.cpp": grab("main.cpp", r"kLabelRows = (\d+)", int),
        "android_main.cpp": grab("android_main.cpp", r"kLabelRows = (\d+)", int),
        "app.js": grab("app.js", r"LABEL_ROWS = (\d+)", int),
        "generate_skybox.py": grab("generate_skybox.py", r"PRISM_LABEL_ROWS = (\d+)", int),
    },
    "label atlas row height (px)": {
        "main.cpp": grab("main.cpp", r"kLabelRowPx = ([\d.]+)f"),
        "android_main.cpp": grab("android_main.cpp", r"kLabelRowPx = ([\d.]+)f"),
        "app.js": grab("app.js", r"LABEL_ROW_PX = ([\d.]+)"),
        "generate_skybox.py": grab("generate_skybox.py",
                                   r"PRISM_LABEL_W, PRISM_LABEL_ROW_H = \d+, ([\d.]+)"),
    },
    "readout quad corners (cube space)": {
        "main.cpp": grab("main.cpp",
            r"x0 = (-?[\d.]+)f, x1 = (-?[\d.]+)f;\s*\n\s*const float yT = (-?[\d.]+)f, yB = (-?[\d.]+)f"),
        "android_main.cpp": grab("android_main.cpp",
            r"x0 = (-?[\d.]+)f, x1 = (-?[\d.]+)f;\s*\n\s*const float yT = (-?[\d.]+)f, yB = (-?[\d.]+)f"),
        "app.js": grab("app.js",
            r"x0 = (-?[\d.]+), x1 = (-?[\d.]+);\s*\n\s*const yT = (-?[\d.]+), yB = (-?[\d.]+)"),
    },
}

drift = False
for desc, values in checks.items():
    unique = set(map(repr, values.values()))
    if len(unique) == 1 and not any("not found" in u for u in unique):
        print(f"  ok    {desc}: {next(iter(values.values()))}")
    else:
        drift = True
        print(f"  DRIFT {desc}:")
        for fname, val in values.items():
            print(f"          {fname}: {val}")

sys.exit(1 if drift else 0)
PYEOF

if [ "$FAIL" -ne 0 ]; then
  echo "FAILED: repos have drifted — see above"
  exit 1
fi
echo "OK: visiontest is in sync with openxr-skybox"
