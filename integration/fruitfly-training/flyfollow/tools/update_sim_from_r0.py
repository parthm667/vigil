"""Fold the R0 measurements back into the simulator (plan Section 0 and 4.8 R0).

    python -m flyfollow.tools.update_sim_from_r0              # print the configs/env.yaml diff and the fine-tune verdict
    python -m flyfollow.tools.update_sim_from_r0 --write      # apply it

Reads the newest data/r0/stick_response_*.json (gains, dead times, taus), data/r0/latency_*.json (video_latency_s)
and configs/camera.json (fx) unless paths are given. Sets profiles.demo to the measured values and recenters the
profiles.train ranges on them (plus or minus 30 %, plan Section 0; fx plus or minus 10 % since a checkerboard fx is
good to about 1 % and fx_calib_err already adds 3 %; dead times and taus keep at least +-0.05 / +-0.04 s because the
state stream is 10 Hz; gain ranges also cover every per-stick gain when the stick map is nonlinear).
Edits are line-based so the file's comments survive. The verdict compares the measurements with the CURRENT train
ranges (what the running Modal job trains on): any value outside means a fine-tune from the best checkpoint.
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import re
from pathlib import Path

import yaml

from flyfollow.interfaces import configs_dir, data_root

# key: (relative half-width, minimum half-width, floor, decimals)
RULES = {
    "fwd_gain_mps": (0.30, 0.0, 0.05, 2),
    "yaw_gain_dps": (0.30, 0.0, 5.0, 1),
    "vz_gain_mps": (0.30, 0.0, 0.05, 2),
    "tau_fwd_s": (0.30, 0.05, 0.02, 2),
    "tau_yaw_s": (0.30, 0.04, 0.02, 2),
    "tau_z_s": (0.30, 0.04, 0.02, 2),
    "fwd_dead_s": (0.30, 0.05, 0.0, 2),
    "yaw_dead_s": (0.30, 0.05, 0.0, 2),
    "vz_dead_s": (0.30, 0.05, 0.0, 2),
    "video_latency_s": (0.30, 0.05, 0.05, 2),
    "fx_px": (0.10, 0.0, 300.0, 0),
}
GAIN_AXIS = {"fwd_gain_mps": "fb", "yaw_gain_dps": "yaw", "vz_gain_mps": "ud"}


def newest(pattern: str) -> Path | None:
    files = glob.glob(str(data_root() / "r0" / pattern))
    return Path(max(files, key=os.path.getmtime)) if files else None


def measurements(stick: Path | None, latency: Path | None, camera: Path | None) -> tuple[dict, dict, list[str]]:
    """(values, per-key extra spans from nonlinear gains, source notes)."""
    vals, spans, notes = {}, {}, []
    if stick and stick.exists():
        r = json.loads(stick.read_text())
        vals.update(r.get("sim_update", {}))
        for k, ax in GAIN_AXIS.items():
            per = r.get("summary", {}).get(ax, {}).get("per_stick", {})
            g = [p["gain_per_100"] for p in per.values()]
            if g:
                spans[k] = (min(g), max(g))
        notes.append(f"stick response: {stick}")
    if latency and latency.exists():
        r = json.loads(latency.read_text())
        if "video_latency_s" in r:
            vals["video_latency_s"] = r["video_latency_s"]
            notes.append(f"latency: {latency} (median {r['video_latency_s']}, p90 {r.get('p90_s')})")
    if camera and camera.exists():
        r = json.loads(camera.read_text())
        vals["fx_px"] = r["fx"]
        notes.append(f"camera: {camera} (fx {r['fx']}, HFOV {r.get('hfov_deg')})")
    return vals, spans, notes


def recenter(key: str, m: float, span: tuple[float, float] | None = None) -> list[float]:
    frac, min_half, floor, dec = RULES[key]
    half = max(frac * abs(m), min_half)
    lo, hi = max(floor, m - half), m + half
    if span:
        lo, hi = min(lo, max(floor, span[0] * 0.9)), max(hi, span[1] * 1.1)
    r = (lambda x: round(x, dec)) if dec else (lambda x: float(round(x)))
    return [r(lo), r(hi)]


def fmt(v) -> str:
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return repr(float(v)) if not float(v).is_integer() else f"{float(v):.1f}"


def set_in_section(lines: list[str], section: tuple[str, ...], key: str, value: str) -> bool:
    """Replace `key: value` inside the nested YAML block `section` (keeps the trailing comment); append if missing."""
    depth, start, end, indent = 0, None, None, 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        ind = len(ln) - len(ln.lstrip())
        if start is None:
            if depth < len(section) and ind == 2 * depth and s.split(":")[0] == section[depth] and s.endswith(":") \
                    or (depth < len(section) and ind == 2 * depth and re.match(rf"^{re.escape(section[depth])}:\s*(#.*)?$", s)):
                depth += 1
                if depth == len(section):
                    start, indent = i + 1, 2 * depth
            continue
        if ind < indent:
            end = i
            break
    if start is None:
        return False
    end = end if end is not None else len(lines)
    pat = re.compile(rf"^(\s{{{indent}}}){re.escape(key)}:(\s*)([^#\n]*?)(\s*)(#.*)?$")
    for i in range(start, end):
        mm = pat.match(lines[i].rstrip("\n"))
        if mm:
            pre, sp, _, gap, com = mm.groups()
            lines[i] = f"{pre}{key}:{sp or ' '}{value}{(gap or '  ') + com if com else ''}\n"
            return True
    last = end
    while last > start and not lines[last - 1].strip():
        last -= 1
    lines.insert(last, f"{' ' * indent}{key}: {value}  # R0 measurement\n")
    return True


def verdict(vals: dict, train: dict) -> tuple[bool, list[str]]:
    need, out = False, []
    for k, m in vals.items():
        rg = train.get(k)
        if not isinstance(rg, (list, tuple)):
            continue
        lo, hi = float(rg[0]), float(rg[1])
        edge = 0.1 * (hi - lo)
        if m < lo or m > hi:
            need = True
            out.append(f"OUTSIDE  {k} = {m} not in trained [{lo}, {hi}]")
        elif m < lo + edge or m > hi - edge:
            out.append(f"marginal {k} = {m} near the edge of [{lo}, {hi}]")
        else:
            out.append(f"inside   {k} = {m} in [{lo}, {hi}]")
    return need, out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="update configs/env.yaml profiles.demo and train ranges from R0 results")
    ap.add_argument("--stick", type=Path, default=None, help="default: newest data/r0/stick_response_*.json")
    ap.add_argument("--latency", type=Path, default=None, help="default: newest data/r0/latency_*.json")
    ap.add_argument("--camera", type=Path, default=None, help="default: configs/camera.json")
    ap.add_argument("--env", type=Path, default=None, help="default: configs/env.yaml")
    ap.add_argument("--no-train", action="store_true", help="only update profiles.demo")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    env = a.env or configs_dir() / "env.yaml"
    vals, spans, notes = measurements(a.stick or newest("stick_response_*.json"), a.latency or newest("latency_*.json"),
                                      a.camera or configs_dir() / "camera.json")
    print("\n".join(notes) or "no R0 results found in data/r0 or configs/camera.json")
    if not vals:
        return
    text = env.read_text()
    cfg = yaml.safe_load(text)
    train = cfg["profiles"]["train"]
    need, lines_v = verdict(vals, train)
    lines = text.splitlines(keepends=True)
    print(f"\n{'key':16s} {'measured':>9s}  demo old -> new            train old -> new")
    for k, m in vals.items():
        if k not in RULES:
            continue
        dec = RULES[k][3]
        v = round(float(m), dec + 1) if dec else float(round(m))
        new_rng = recenter(k, float(m), spans.get(k))
        old_demo = cfg["profiles"].get("demo", {}).get(k)
        print(f"{k:16s} {v:>9} {old_demo!s:>9} -> {v:<9}   {train.get(k)!s:>14} -> {new_rng}")
        set_in_section(lines, ("profiles", "demo"), k, fmt(v))
        if not a.no_train:
            set_in_section(lines, ("profiles", "train"), k, fmt(new_rng))
    new = "".join(lines)
    yaml.safe_load(new)  # still valid YAML
    diff = "".join(difflib.unified_diff(text.splitlines(keepends=True), lines, str(env), str(env) + " (new)"))
    print("\n" + (diff or "(no change)"))
    print("\n".join(["", "trained ranges vs measurements:"] + lines_v))
    print("\nMODAL FINE-TUNE NEEDED: " + ("YES. Start a 1 to 2 h fine-tune from the best checkpoint with the recentered "
                                          "ranges; fly PID meanwhile (plan 4.8 R0)." if need else
                                          "no, every measured value is inside the ranges the running job trains on."))
    if "video_latency_s" in vals or "fx_px" in vals:
        print("runtime settings to send (operator console or configs): " + ", ".join(
            f"{s}={vals[k]}" for k, s in (("video_latency_s", "video_latency_s"), ("fx_px", "fx")) if k in vals))
    if a.write and diff:
        env.write_text(new)
        print(f"wrote {env}")
    elif diff:
        print("(dry: rerun with --write to apply)")


if __name__ == "__main__":
    main()
