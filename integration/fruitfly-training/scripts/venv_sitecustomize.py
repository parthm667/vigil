"""Installed as site-packages/sitecustomize.py by scripts/setup_env.sh.

On this macOS setup every file under .venv ends up with the BSD "hidden" flag, and
Python 3.12's site.py skips hidden .pth files. That silently disables editable
installs (flyfollow, flydrones) and packages that rely on a .pth (rerun-sdk).
This re-applies any .pth file that site.py skipped because it was hidden.
"""

import os
import stat
import sys


def _load_hidden_pth() -> None:
    for sitedir in [p for p in sys.path if p.endswith("site-packages")]:
        try:
            names = sorted(os.listdir(sitedir))
        except OSError:
            continue
        for name in names:
            if not name.endswith(".pth"):
                continue
            full = os.path.join(sitedir, name)
            try:
                if not (os.lstat(full).st_flags & stat.UF_HIDDEN):
                    continue  # site.py already processed it
                lines = open(full, encoding="utf-8").read().splitlines()
            except (OSError, AttributeError):
                continue
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("import ", "import\t")):
                    try:
                        exec(line)
                    except Exception as e:  # match site.py: report and continue
                        print(f"sitecustomize: error in {name}: {e}", file=sys.stderr)
                    continue
                path = os.path.abspath(os.path.join(sitedir, line))
                if os.path.exists(path) and path not in sys.path:
                    sys.path.append(path)


_load_hidden_pth()
