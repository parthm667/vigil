"""One-command preflight for flying the fly-steered Tello (docs/FLIGHT_TEST_CHECKLIST.md).

    .venv/bin/python scripts/preflight.py [--rg ~/Documents/GitHub/jerkgt13] [--config site.yaml]
    .venv/bin/python scripts/preflight.py --standalone     # this repo alone (docs/STANDALONE.md): detector, no ReachGlass

Prints PASS / FAIL / WARN per check and exits non-zero if anything FAILs. It never talks to the drone.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"

REPO = Path(__file__).resolve().parents[1]
PARAMS = REPO / "data/brains/trained/FLY-YAW_smooth_best.json"
RESULTS: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    RESULTS.append((status, name, detail))
    print(f"[{status:4s}] {name}{': ' + detail if detail else ''}", flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def check_ports() -> None:
    ports = (8889, 8890, 11111)
    holders: set[str] = set()
    if IS_WIN:
        out = sh("netstat -ano -p udp")
        for line in out.splitlines():
            m = re.search(r":(\d+)\s+\S+\s+(\d+)\s*$", line.strip())
            if m and int(m.group(1)) in ports:
                holders.add(f"PID {m.group(2)} (port {m.group(1)})")
    else:
        out = sh("lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111 2>/dev/null | tail -n +2")
        holders = {f"{line.split()[0]} (PID {line.split()[1]})" for line in out.splitlines() if line.strip()}
    if not holders:
        report("PASS", "Tello UDP ports 8889/8890/11111 free")
    else:
        kill = "taskkill /PID <PID> /F" if IS_WIN else "kill <PID>"
        report("FAIL", "Tello UDP ports in use", ", ".join(sorted(holders)) + f". Stop them ({kill}) before flying")


def check_load() -> None:
    if not hasattr(os, "getloadavg"):
        report("WARN", "CPU load", "not measurable on this OS; the brain-speed check below covers it")
        return
    one = os.getloadavg()[0]
    cores = os.cpu_count() or 1
    if one < cores:
        report("PASS", "CPU load", f"{one:.1f} (cores {cores})")
    else:
        report("FAIL", "CPU load", f"{one:.1f} with {cores} cores: quit sims, video calls, browsers; the brain needs a 50 ms tick")


def check_brain() -> None:
    try:
        from flyfollow.steer import FlySteer
    except Exception as e:  # noqa: BLE001
        report("FAIL", "import flyfollow.steer", repr(e))
        return
    if not PARAMS.exists():
        report("FAIL", "trained parameters", f"{PARAMS} missing (git pull)")
        return
    s = FlySteer(params_path=str(PARAMS))
    signs = {}
    for deg in (15, 0, -15):
        s.reset(z_ref_m=1.6, kind="follow")
        ys = [s.yaw(0.05 * k, math.radians(deg), t_frame=0.05 * k) for k in range(60)]
        signs[deg] = sum(ys[-20:]) / 20
    ok = signs[15] > 15 and abs(signs[0]) < 5 and signs[-15] < -15
    report("PASS" if ok else "FAIL", "steering sign (person right -> turn right)",
           f"+15 deg -> {signs[15]:+.0f}, 0 -> {signs[0]:+.0f}, -15 -> {signs[-15]:+.0f}")
    # real-time budget: one 50 ms brain tick per control step
    s.reset(z_ref_m=1.6, kind="follow")
    t0 = time.perf_counter()
    n = 100
    for k in range(n):
        s.yaw(0.05 * k, math.radians(10 * math.sin(k / 10)), t_frame=0.05 * k)
    ms = (time.perf_counter() - t0) / n * 1000
    st = s.stats
    mean = st.get("tick_ms_mean") or ms
    p95 = st.get("tick_ms_p95") or ms
    status = "PASS" if mean < 15 and p95 < 30 else "FAIL"
    report(status, "brain speed", f"tick mean {mean:.1f} ms, p95 {p95:.1f} ms (limit 15 / 30 ms per 50 ms tick)")


def check_config(rg: Path, config: str) -> None:
    py = rg / (".venv/Scripts/python.exe" if IS_WIN else ".venv/bin/python")
    if not py.exists():
        report("FAIL", "ReachGlass venv", f"{py} missing (checklist 1.2)")
        return
    code = (
        "from reachglass.config import load_config; import os;"
        f"c = load_config('{config}');"
        "print(c.follow.steering, c.approach.steering, c.fly.params_path, c.fly.deadband, c.fly.hysteresis,"
        " c.follow.avoid.enabled, c.follow.distance_m, c.follow.altitude_m, getattr(c.perception, 'person_height_m', None))"
    )
    r = subprocess.run([str(py), "-c", code], cwd=rg, capture_output=True, text=True)
    if r.returncode != 0:
        report("FAIL", f"ReachGlass config {config}", r.stderr.strip().splitlines()[-1] if r.stderr else "load failed")
        return
    steer, appr, params, db, hy, avoid, dist, alt, height = r.stdout.split()
    report("PASS" if steer == "fly" else "WARN", "follow steering", f"{steer} (approach {appr})")
    report("PASS" if Path(params).exists() else "FAIL", "fly params_path", params)
    report("PASS", "fly output shaping", f"deadband {db}, hysteresis {hy}")
    report("PASS", "follow geometry", f"distance {dist} m, altitude {alt} m, avoidance {'on' if avoid == 'True' else 'off'}")
    try:
        h = float(height)
        report("PASS" if float(alt) >= h + 0.2 else "FAIL", "wearer height", f"{h:.2f} m (altitude must be >= height + 0.2)")
    except ValueError:
        report("WARN", "wearer height", "perception.person_height_m not set: range estimates use the default (checklist 2.3)")
    diff = subprocess.run(["git", "-C", str(rg), "diff", "--stat", "HEAD"], capture_output=True, text=True).stdout.strip()
    patched = diff.splitlines()[-1] if diff else ""
    report("PASS" if patched else "FAIL", "patches applied in ReachGlass", patched or "no local changes: apply the two patches (checklist 1.2)")


def check_detector() -> None:
    try:
        from flyfollow.runtime.detector import DEFAULT_WEIGHTS, YoloDetector, resolve_weights
    except Exception as e:  # noqa: BLE001
        report("FAIL", "import flyfollow.runtime.detector", repr(e))
        return
    w = resolve_weights(DEFAULT_WEIGHTS)
    if not w.exists():
        report("FAIL", "person detector weights", f"{w} missing: python -m flyfollow.runtime.detector --selftest (needs internet)")
        return
    try:
        import numpy as np

        det = YoloDetector()
        img = np.zeros((720, 960, 3), np.uint8)
        det.detect(img)  # first call at the Tello frame size compiles kernels
        ms_all = []
        for _ in range(10):
            t0 = time.perf_counter()
            det.detect(img)
            ms_all.append((time.perf_counter() - t0) * 1000)
        ms = sorted(ms_all)[len(ms_all) // 2]
    except Exception as e:  # noqa: BLE001
        report("FAIL", "person detector", repr(e))
        return
    report("PASS" if ms < 80 else "WARN", "person detector", f"{w.name} on {det.device}, {ms:.0f} ms per frame"
           + ("" if ms < 80 else ": slow, the follow will lag (close other apps, or plug in the laptop)"))


def check_network() -> None:
    if IS_WIN:
        m = re.search(r"^\s*SSID\s*:\s*(.+)$", sh("netsh wlan show interfaces"), re.M)
        ssid = m.group(1).strip() if m else ""
    elif IS_MAC:
        ssid = sh("networksetup -getairportnetwork en0 2>/dev/null") or sh("ipconfig getsummary en0 2>/dev/null | awk -F' : ' '/ SSID/ {print $2}'")
    else:
        ssid = sh("iwgetid -r 2>/dev/null")
    on_tello = "TELLO" in ssid.upper()
    report("PASS" if on_tello else "WARN", "Wi-Fi", (ssid or "unknown") + ("" if on_tello else ": join the TELLO-xxxxxx network to fly"))
    ping = ["ping", "-n", "1", "-w", "2000", "1.1.1.1"] if IS_WIN else ["ping", "-c", "1", "-W" if not IS_MAC else "-t", "2", "1.1.1.1"]
    net = subprocess.run(ping, capture_output=True).returncode == 0
    report("PASS" if net else "WARN", "internet", "reachable" if net else "none: fine for flying (everything is local)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rg", default=str(Path.home() / "Documents/GitHub/jerkgt13"), help="ReachGlass checkout with the patches applied")
    ap.add_argument("--config", default="site.yaml")
    ap.add_argument("--standalone", action="store_true", help="this repo's own runtime and detector, no ReachGlass")
    a = ap.parse_args()
    check_ports()
    check_load()
    check_brain()
    if a.standalone:
        check_detector()
    else:
        check_config(Path(a.rg).expanduser(), a.config)
    check_network()
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\n{'READY TO FLY' if not fails else 'NOT READY'}: {len(fails)} fail, {sum(r[0] == 'WARN' for r in RESULTS)} warn")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
