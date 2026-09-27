#!/usr/bin/env bash
# Fetch the TuragaLab flybody fly model (Apache-2.0) for the visualization (flyfollow.viz).
#
# Copies only what fruitfly.xml needs (the XML, its .obj meshes and the LICENSE, about 130 MB)
# into data/flybody/ (gitignored). Never commit these assets.
#
#   scripts/setup_viz.sh                 # sparse, blob-filtered fetch of a pinned commit from GitHub
#   FLYBODY_SRC=/path/to/flybody scripts/setup_viz.sh   # copy from an existing clone instead
#   FORCE=1 scripts/setup_viz.sh         # refresh even if data/flybody/fruitfly.xml exists
#   FLYFOLLOW_FLYBODY=/other/dir scripts/setup_viz.sh   # install somewhere else (the viz reads the same variable)
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="https://github.com/TuragaLab/flybody.git"
COMMIT="d015e9bfe441bd90ae431bac24c55cb74bdbce26"
DEST="${FLYFOLLOW_FLYBODY:-data/flybody}"
PY="${PYTHON:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

if [ -f "$DEST/fruitfly.xml" ] && [ -z "${FORCE:-}" ]; then
  echo "flybody already installed in $DEST (FORCE=1 to refresh)"
  exit 0
fi

SRC="${FLYBODY_SRC:-}"
TMP=""
cleanup() { if [ -n "$TMP" ]; then rm -rf "$TMP"; fi; }
trap cleanup EXIT

if [ -z "$SRC" ]; then
  TMP="$(mktemp -d)"
  SRC="$TMP/flybody"
  echo "fetching flybody $COMMIT (sparse: fruitfly assets + LICENSE only)"
  git init -q "$SRC"
  git -C "$SRC" remote add origin "$REPO"
  git -C "$SRC" config core.sparseCheckout true
  git -C "$SRC" sparse-checkout set --no-cone \
    "/LICENSE" "/flybody/fruitfly/assets/fruitfly.xml" "/flybody/fruitfly/assets/*.obj"
  if ! git -C "$SRC" fetch -q --depth 1 --filter=blob:none origin "$COMMIT"; then
    echo "pinned commit fetch failed; falling back to the default branch" >&2
    git -C "$SRC" fetch -q --depth 1 --filter=blob:none origin
  fi
  git -C "$SRC" checkout -q FETCH_HEAD
fi

ASSETS="$SRC/flybody/fruitfly/assets"
[ -f "$ASSETS/fruitfly.xml" ] || { echo "no fruitfly.xml under $ASSETS" >&2; exit 1; }
mkdir -p "$DEST"

"$PY" - "$ASSETS" "$DEST" "$SRC" "$COMMIT" <<'EOF'
import re, shutil, subprocess, sys
from pathlib import Path

assets, dest, src, commit = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
xml = (assets / "fruitfly.xml").read_text()
files = sorted(set(re.findall(r'file="([^"]+)"', xml)))
missing = [f for f in files if not (assets / f).exists()]
if missing:
    sys.exit(f"missing mesh files: {missing[:5]} ...")
shutil.copy2(assets / "fruitfly.xml", dest / "fruitfly.xml")
for f in files:
    shutil.copy2(assets / f, dest / f)
shutil.copy2(src / "LICENSE", dest / "LICENSE")
try:
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    head = commit
(dest / "SOURCE.txt").write_text(
    "TuragaLab flybody fruit fly model (female Drosophila melanogaster), Apache-2.0 (see LICENSE).\n"
    f"https://github.com/TuragaLab/flybody commit {head}\n"
    "Files: flybody/fruitfly/assets/fruitfly.xml and the meshes it references. Not committed to this repo.\n"
)
size = sum((dest / f).stat().st_size for f in files) / 1e6
print(f"installed fruitfly.xml + {len(files)} meshes ({size:.0f} MB) + LICENSE into {dest}")
EOF
