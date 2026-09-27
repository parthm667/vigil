"""One-command setup for flying person following from this repo alone (no ReachGlass): docs/STANDALONE.md.

Run from this repo with normal internet, BEFORE joining the Tello Wi-Fi (macOS, Windows, Linux):

    python scripts/setup_standalone.py            # add --viz for the live fly body + brain window

What it does, skipping anything already done:
1. .venv (Python 3.12) with this repo, FlyDrones, the drone runtime and the YOLO person detector (ultralytics, torch).
2. Downloads the person detector weights into models/ and times them on a test image (--selftest).
3. Runs scripts/preflight.py --standalone (ports and Wi-Fi fail until you are at the drone: that is expected).
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

FT = Path(__file__).resolve().parents[1]
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=FT, check=check)


def find_python312() -> list[str]:
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    for cand in (["py", "-3.12"], ["python3.12"], ["python3"], ["python"]):
        try:
            out = subprocess.run(cand + ["-c", "import sys; print(sys.version_info[:2] == (3, 12))"], capture_output=True, text=True, check=False)
        except FileNotFoundError:
            continue
        if out.stdout.strip() == "True":
            return cand
    sys.exit("Python 3.12 not found. Install it (python.org or `uv python install 3.12`) and rerun with it.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--viz", action="store_true", help="also install the fly body + brain viewer (mujoco, rerun)")
    a = ap.parse_args()
    vpy = FT / (".venv/Scripts/python.exe" if IS_WIN else ".venv/bin/python")
    print("\n[1] .venv with the runtime and the person detector", flush=True)
    if not vpy.exists():
        run(find_python312() + ["-m", "venv", FT / ".venv"])
    uv = shutil.which("uv")
    pip = [uv, "pip", "install", "--python", vpy] if uv else [vpy, "-m", "pip", "install"]
    if not uv:
        run([vpy, "-m", "pip", "install", "--upgrade", "pip"])
    extras = "drone,perception" + (",viz" if a.viz else "")
    run(pip + ["-e", FT / "third_party/FlyDrones", "-e", f"{FT.as_posix()}[{extras}]"])
    if IS_MAC:  # files under .venv can get the "hidden" flag and Python 3.12 then skips the editable-install .pth files
        site = subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                              capture_output=True, text=True, check=True).stdout.strip()
        shutil.copy(FT / "scripts/venv_sitecustomize.py", Path(site) / "sitecustomize.py")
        subprocess.run(["chflags", "-R", "nohidden", str(FT / ".venv")], check=False)
    run([vpy, "-c", "import flyfollow, flydrones, ultralytics, djitellopy, zmq; print('  imports ok')"])
    print("\n[2] person detector weights (needed offline on the Tello Wi-Fi)", flush=True)
    r = run([vpy, "-m", "flyfollow.runtime.detector", "--selftest"], check=False)
    if r.returncode != 0:
        print("  detector self-test failed: see the line above")
    if a.viz:
        print("\n[2b] fly body + brain viewer assets", flush=True)
        bash = shutil.which("bash")
        if bash:
            subprocess.run([bash, str(FT / "scripts/setup_viz.sh")], cwd=FT, check=False, env=dict(os.environ, PYTHON=str(vpy)))
        else:
            print("  bash not found: the viewer runs without the fly body panel")
    print("\n[3] preflight", flush=True)
    p = subprocess.run([str(vpy), str(FT / "scripts/preflight.py"), "--standalone"], cwd=FT, check=False)
    print("\nSetup done." if p.returncode == 0 and r.returncode == 0 else
          "\nSetup done; ports and Wi-Fi FAIL until you are at the drone, anything else needs fixing.")
    py = ".venv\\Scripts\\python" if IS_WIN else ".venv/bin/python"
    print(f"Next, on the Tello Wi-Fi: {py} -m flyfollow.runtime.launch --dry   (bench test, sends nothing)")
    print(f"then:                     {py} -m flyfollow.runtime.launch --send  (real flight). Steps: docs/STANDALONE.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
