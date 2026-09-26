#!/usr/bin/env bash
# Reproduce data/ from scratch: download MaleCNS v1.0 (resumable, idempotent), then build the
# pursuit brains, the audit and the FLY-SHUF shuffles. Safe to re-run: finished downloads and
# built files are skipped.
#
#   scripts/setup_data.sh            # download + build cores (+ shuffles)
#   FORCE=1 scripts/setup_data.sh    # rebuild the full brain and cores even if present
#   AUDIT=1 scripts/setup_data.sh    # also run the G0 audit (about 1 minute)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
# the venv's editable-install .pth files can be skipped on macOS when flagged hidden; be explicit
export PYTHONPATH="$ROOT:$ROOT/third_party/FlyDrones/src${PYTHONPATH:+:$PYTHONPATH}"
DATA="${FLYFOLLOW_DATA:-$ROOT/data}"
RAW="$DATA/malecns_v1"
mkdir -p "$RAW" "$DATA/brains"

# URLs come from flydrones.brain.connectome (MALECNS_BASE / MALECNS_FILES)
BASE="$("$PY" -c 'from flydrones.brain.connectome import MALECNS_BASE; print(MALECNS_BASE)')"
FILES="$("$PY" -c 'from flydrones.brain.connectome import MALECNS_FILES; print(" ".join(MALECNS_FILES.values()))')"

for f in $FILES; do
  url="$BASE/$f"
  out="$RAW/$f"
  remote_size="$(curl -sI "$url" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r' | tail -1)"
  if [[ -f "$out" && -n "$remote_size" && "$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out")" == "$remote_size" ]]; then
    echo "have $f ($remote_size bytes)"
    continue
  fi
  echo "downloading $f"
  curl -fL --retry 5 -C - -o "$out" "$url"
done

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then FORCE_FLAG=(--force); fi
if [[ "${FORCE:-0}" == "1" || ! -f "$DATA/brains/pursuit_core1.npz" || ! -f "$DATA/brains/pursuit_core2.npz" ]]; then
  "$PY" -m flyfollow.brain.build --data-dir "$RAW" ${FORCE_FLAG[@]+"${FORCE_FLAG[@]}"}
else
  echo "have pursuit cores (FORCE=1 to rebuild)"
fi

# FLY-SHUF shuffles for the recommended core only. core2 shuffles never pass the path check
# (plain about 54 %, layered about 90 % of the real walk count; see docs/audit_result.md).
if [[ "${FORCE:-0}" == "1" || ! -f "$DATA/brains/pursuit_core1_shuf3.npz" ]]; then
  "$PY" -m flyfollow.brain.shuffle --hops 1
else
  echo "have core1 shuffles"
fi

"$PY" -m flyfollow.brain.build bench

if [[ "${AUDIT:-0}" == "1" ]]; then
  "$PY" -m flyfollow.audit.audit --brain "$DATA/brains/pursuit_core1.npz" "$DATA/brains/pursuit_core2.npz" --lc10-only
fi
echo "done: $(ls "$DATA/brains")"
