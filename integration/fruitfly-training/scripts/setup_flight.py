"""One-command setup of the flying laptop: ReachGlass + the fruit fly steering (docs/FLIGHT_TEST_CHECKLIST.md 1).

Run from this repo, with normal internet, BEFORE joining the Tello Wi-Fi (works on macOS, Windows and Linux):

    python scripts/setup_flight.py                               # uses ~/Documents/GitHub/jerkgt13 (clones it if missing)
    python scripts/setup_flight.py --rg C:/code/jerkgt13         # another ReachGlass checkout
    python scripts/setup_flight.py --skip-models                 # models already downloaded

What it does, skipping anything already done:
1. ReachGlass checkout: refuses to touch uncommitted changes; switches to a branch `fly-demo` and applies
   docs/integration/reachglass_flysteer.patch then reachglass_follow_avoid.patch (made on ReachGlass 2302928).
2. ReachGlass venv (.venv, Python 3.12): their requirements + this repo (flyfollow, flydrones) installed into it,
   so ONE venv runs the stack, the fly and the preflight.
3. Their models (tools/download_models.py), needed offline on the Tello Wi-Fi.
4. site.yaml: appends the fly steering block (trained smooth fly, deadband 4, hysteresis 3, 1.6 m follow, avoidance
   off) and writes site_pid.yaml, the same with their PID steering, for the baseline runs.
5. Runs scripts/preflight.py.
6. With --viz: the live fly body + brain window's assets (MaleCNS soma positions, 14 MB; the flybody model,
   about 130 MB, via scripts/setup_viz.sh when bash is available, e.g. Git Bash on Windows).
The trained fly, the brain and its calibration are in git (data/brains/), so no big download is needed.
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
RG_URL = "https://github.com/nathanwuzhao/jerkgt13"
PATCH_BASE = "2302928"
PATCHES = [FT / "docs/integration/reachglass_flysteer.patch", FT / "docs/integration/reachglass_follow_avoid.patch"]
PARAMS = FT / "data/brains/trained/FLY-YAW_smooth_best.json"
MARKER = "# --- fruit fly steering"
IS_WIN = platform.system() == "Windows"


def run(cmd: list, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=check)


def git(rg: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(rg), *args], capture_output=True, text=True, check=check)


def step(n: int, text: str) -> None:
    print(f"\n[{n}] {text}", flush=True)


def find_python312() -> list[str]:
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    for cand in (["py", "-3.12"], ["python3.12"], ["python3"], ["python"]):
        try:
            out = subprocess.run(cand + ["-c", "import sys; print(sys.version_info[:2] == (3, 12))"], capture_output=True, text=True)
        except FileNotFoundError:
            continue
        if out.stdout.strip() == "True":
            return cand
    sys.exit("Python 3.12 not found. Install it (python.org or `uv python install 3.12`) and rerun with it.")


def checkout(rg: Path) -> None:
    step(1, f"ReachGlass checkout at {rg}")
    if not (rg / ".git").exists():
        rg.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", RG_URL, rg])
    if git(rg, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        applied = all(git(rg, "apply", "--reverse", "--check", str(p), check=False).returncode == 0 for p in PATCHES)
        if applied:
            print("  both patches are already applied, keeping the working tree")
            return
        sys.exit(f"{rg} has uncommitted changes that are not our patches. Commit or stash them, then rerun.")
    branch = git(rg, "branch", "--show-current").stdout.strip()
    if branch != "fly-demo":
        base = "HEAD"
        if git(rg, "apply", "--check", str(PATCHES[0]), check=False).returncode != 0:
            base = PATCH_BASE  # their main moved past the commit the patches were made on
            print(f"  patches do not apply on HEAD; branching from {PATCH_BASE} instead")
        exists = git(rg, "rev-parse", "--verify", "fly-demo", check=False).returncode == 0
        run(["git", "-C", rg, "checkout", "fly-demo"] if exists else ["git", "-C", rg, "checkout", "-b", "fly-demo", base])
    for p in PATCHES:
        if git(rg, "apply", "--reverse", "--check", str(p), check=False).returncode == 0:
            print(f"  already applied: {p.name}")
            continue
        r = git(rg, "apply", "--check", str(p), check=False)
        if r.returncode != 0:
            sys.exit(f"{p.name} does not apply:\n{r.stderr}\nTell the teammate which ReachGlass commit you are on.")
        run(["git", "-C", rg, "apply", p])
    print("  note: this branch is local; do not push it to the ReachGlass repo")


def venv(rg: Path, py312: list[str]) -> Path:
    step(2, "ReachGlass venv with the fly installed")
    vpy = rg / (".venv/Scripts/python.exe" if IS_WIN else ".venv/bin/python")
    if not vpy.exists():
        run(py312 + ["-m", "venv", rg / ".venv"])
    uv = shutil.which("uv")
    pip = [uv, "pip", "install", "--python", vpy] if uv else [vpy, "-m", "pip", "install"]
    if not uv:
        run([vpy, "-m", "pip", "install", "--upgrade", "pip"])
    run(pip + ["-r", rg / "requirements.txt"])
    run(pip + ["-e", FT / "third_party/FlyDrones", "-e", f"{FT.as_posix()}[viz]"])
    run([vpy, "-c", "import reachglass, flyfollow, flydrones; from flyfollow.steer import FlySteer; print('  imports ok')"], cwd=rg)  # reachglass runs from its checkout
    return vpy


def models(rg: Path, vpy: Path, skip: bool) -> None:
    step(3, "ReachGlass models (needed offline)")
    if skip:
        print("  skipped (--skip-models)")
        return
    run([vpy, "tools/download_models.py"], cwd=rg)


def site_yaml(rg: Path, vpy: Path) -> None:
    step(4, "site.yaml fly block and site_pid.yaml")
    site = rg / "site.yaml"
    text = site.read_text(encoding="utf-8") if site.exists() else ""
    if MARKER in text:
        print("  fly block already present")
    else:
        block = f"""
{MARKER} (docs/FLIGHT_TEST_CHECKLIST.md in fruitfly-training)
follow:
  steering: fly            # pid = their yaw law. This one key switches the fly off.
  distance_m: 1.6          # their default 1.0 sees only the head; 1.6 keeps shoulders in view (fly trained at 1.5 to 2.5 m)
  altitude_m: 2.0          # must be >= wearer height + 0.2
  avoid: {{enabled: false}}  # turned on only in checklist Section 4
approach: {{steering: fly}}
fly:
  params_path: {PARAMS.as_posix()}
  smoothing_ms: 0          # no low-pass: every low-pass in the sim sweep added lag and cost accuracy
  deadband: 4              # stick units
  hysteresis: 3            # only move the sent stick when the new value is more than 3 away
  viz: false               # true = publish the fly body + brain view (run flyfollow.viz.live first)
safety: {{max_altitude_m: 2.3}}   # keep below your ceiling
"""
        if text and not text.endswith("\n"):
            text += "\n"
        site.write_text(text + block, encoding="utf-8")
        print(f"  appended to {site}")
    (rg / "site_pid.yaml").write_text(site.read_text(encoding="utf-8").replace("steering: fly", "steering: pid"), encoding="utf-8")
    code = ("from reachglass.config import load_config\n"
            "for f in ('site.yaml', 'site_pid.yaml'):\n"
            "    c = load_config(f); print('  ', f, '->', c.follow.steering, '| params ok' if c.fly.params_path else '')")
    run([vpy, "-c", code], cwd=rg)


def viz_assets(vpy: Path) -> None:
    step(6, "fly visualization assets")
    import urllib.request

    ann = FT / "data/malecns_v1/body-annotations-male-cns-v1.0-minconf-0.5.feather"
    if not ann.exists():
        ann.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/" + ann.name
        print(f"  downloading {ann.name} (14 MB)")
        urllib.request.urlretrieve(url, ann)
    else:
        print("  MaleCNS soma positions already present")
    bash = shutil.which("bash")
    if bash:
        env = dict(os.environ, PYTHON=str(vpy))
        subprocess.run([bash, str(FT / "scripts/setup_viz.sh")], cwd=FT, env=env, check=False)
    else:
        print("  bash not found: skipping the flybody model (the viewer runs without the body panel)")
    print(f"  viewer: {vpy} -m flyfollow.viz.live --brain {(FT / 'data/brains/pursuit_core1.npz').as_posix()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up the flying laptop (ReachGlass + fruit fly steering)")
    ap.add_argument("--rg", default=str(Path.home() / "Documents/GitHub/jerkgt13"), help="ReachGlass checkout (cloned if missing)")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--viz", action="store_true", help="also fetch the fly body + brain viewer assets")
    a = ap.parse_args()
    rg = Path(a.rg).expanduser().resolve()
    if not PARAMS.exists():
        sys.exit(f"{PARAMS} missing: git pull this repo first")
    checkout(rg)
    vpy = venv(rg, find_python312())
    models(rg, vpy, a.skip_models)
    site_yaml(rg, vpy)
    if a.viz:
        viz_assets(vpy)
    step(5, "preflight")
    env = dict(os.environ)
    r = subprocess.run([str(vpy), str(FT / "scripts/preflight.py"), "--rg", str(rg)], env=env)
    print("\nSetup done." if r.returncode == 0 else "\nSetup done; fix the FAIL lines above before flying (ports and Wi-Fi fail until you are at the drone).")
    print("Next: docs/FLIGHT_TEST_CHECKLIST.md Section 2 (dry-run sign check), then Section 3 (flights).")


if __name__ == "__main__":
    main()
