#!/usr/bin/env bash
# Create the Python 3.12 venv with every dependency (training, Modal, visualization).
# Idempotent. Run from anywhere: scripts/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,viz]"

# macOS quirk: files under .venv can carry the "hidden" flag, and Python 3.12 skips hidden
# .pth files, which breaks editable installs. sitecustomize.py re-applies them.
SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp scripts/venv_sitecustomize.py "$SITE/sitecustomize.py"
chflags -R nohidden .venv 2>/dev/null || true

.venv/bin/python -c "import flydrones, flyfollow, cma, modal; print('env OK: flyfollow, flydrones, cma, modal', modal.__version__)"
echo "Next: .venv/bin/modal setup   (once per machine), then scripts/setup_data.sh"
