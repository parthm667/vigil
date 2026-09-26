"""Fit the Tello response model from a lag-test log.

The lag test sends rc steps (stick = `mag`) for `dur` seconds on each axis and
logs the Tello state stream (about 10 Hz) and a per-frame video motion score.
For each step we fit

    rate(t) = 0                                  for t < d
    rate(t) = g * (1 - exp(-(t - d) / tau))      for d <= t < t_release (+ d)

to the telemetry, where `d` is dead time (command to first motion in the state
stream), `tau` the first-order time constant and `g` the steady rate at that
stick value. Yaw uses the integrated yaw angle, forward/lateral use vgx/vgy
(integers in dm/s, so heavily quantized).

Usage:
    python -m flyfollow.tools.analyze_lag data/lag_test/drone_fulllogs_20260926_052637.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

STATE_COLS = ["t", "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "tof", "h", "bat"]
AXIS_CHANNEL = {"yaw": "yaw", "pitch": "vgx", "roll": "vgy", "throttle": "vgz"}


def _model(t: np.ndarray, d: float, tau: float, g: float, t_off: float) -> np.ndarray:
    """First-order response to a stick step from 0 to t_off (both shifted by dead time d)."""
    on = np.clip(t - d, 0, None)
    y = g * (1 - np.exp(-on / tau))
    off = t - d - t_off
    after = off > 0
    y_at_off = g * (1 - np.exp(-t_off / tau))
    y[after] = y_at_off * np.exp(-off[after] / tau)
    return y


def _fit(t: np.ndarray, y: np.ndarray, t_off: float) -> tuple[float, float, float, float]:
    """Grid search over d and tau, closed-form least squares for g. Returns d, tau, g, rmse."""
    best = (np.inf, 0.0, 0.0, 0.0)
    for d in np.arange(0.05, 1.2, 0.01):
        for tau in np.arange(0.02, 1.5, 0.01):
            basis = _model(t, d, tau, 1.0, t_off)
            den = float(basis @ basis)
            if den <= 0:
                continue
            g = float(basis @ y) / den
            err = float(np.mean((y - g * basis) ** 2))
            if err < best[0]:
                best = (err, d, tau, g)
    err, d, tau, g = best
    return d, tau, g, float(np.sqrt(err))


def analyze(path: str | Path) -> dict:
    log = json.loads(Path(path).read_text())
    S = np.asarray(log["state"], dtype=float)
    col = {c: i for i, c in enumerate(STATE_COLS)}
    t = S[:, col["t"]]
    V = np.asarray(log["video"], dtype=float) if log.get("video") else None
    mag = log["meta"]["args"]["mag"]
    out = {"stick": mag, "state_hz": float(1 / np.median(np.diff(t))), "steps": []}
    for ev in log["events"]:
        if ev.get("type") != "rc":
            continue
        ax, sign = ev["axis"], ev["sign"]
        t0, t1 = ev["t_cmd"], ev["t_release"]
        m = (t >= t0 - 0.3) & (t <= t1 + 2.5)
        tt = t[m] - t0
        if ax == "yaw":
            ang = np.unwrap(np.deg2rad(S[m, col["yaw"]]))
            ang = np.rad2deg(ang - ang[tt < 0].mean()) * sign
            # fit the angle by integrating the rate model numerically
            best = (np.inf, 0, 0, 0)
            grid_t = np.linspace(tt[0], tt[-1], 600)
            for d in np.arange(0.05, 0.8, 0.01):
                for tau in np.arange(0.02, 0.8, 0.01):
                    rate = _model(grid_t, d, tau, 1.0, t1 - t0)
                    integ = np.interp(tt, grid_t, np.concatenate([[0], np.cumsum(rate[1:] * np.diff(grid_t))]))
                    den = float(integ @ integ)
                    g = float(integ @ ang) / den if den > 0 else 0.0
                    err = float(np.mean((ang - g * integ) ** 2))
                    if err < best[0]:
                        best = (err, d, tau, g)
            err, d, tau, g = best
            unit = "deg/s"
        else:
            y = S[m, col[AXIS_CHANNEL[ax]]] * sign
            if ax in ("pitch", "roll"):
                y = -y  # this Tello reports forward / right motion as negative vgx / vgy
            if ax == "throttle":
                y = -y  # vgz is negative when climbing
            d, tau, g, err = _fit(tt, y / 10.0, t1 - t0)  # dm/s -> m/s
            unit = "m/s"
        video_onset = None
        if V is not None:
            vm = (V[:, 0] >= t0) & (V[:, 0] <= t1)
            thr = log.get("video_threshold", 3.5)
            above = np.flatnonzero(V[vm, 1] > thr)
            if above.size:
                video_onset = float(V[vm, 0][above[0]] - t0)
        out["steps"].append({"axis": ax, "sign": sign, "dead_time_s": round(d, 3), "tau_s": round(tau, 3),
                             "gain": round(g, 3), "unit": unit, "rmse": round(float(np.sqrt(err)) if ax == "yaw" else err, 3),
                             "video_onset_s": None if video_onset is None else round(video_onset, 3)})
    summary = {}
    for ax in ("yaw", "pitch", "roll", "throttle"):
        rows = [s for s in out["steps"] if s["axis"] == ax]
        if not rows:
            continue
        arr = lambda k: np.array([r[k] for r in rows if r[k] is not None], dtype=float)  # noqa: E731
        vid = arr("video_onset_s")
        summary[ax] = {
            "dead_time_s": [round(float(np.median(arr("dead_time_s"))), 3), round(float(arr("dead_time_s").min()), 3), round(float(arr("dead_time_s").max()), 3)],
            "tau_s": [round(float(np.median(arr("tau_s"))), 3), round(float(arr("tau_s").min()), 3), round(float(arr("tau_s").max()), 3)],
            "gain_at_stick": [round(float(np.median(arr("gain"))), 3), round(float(arr("gain").min()), 3), round(float(arr("gain").max()), 3)],
            "gain_per_100_stick": round(float(np.median(arr("gain"))) * 100.0 / mag, 3),
            "unit": rows[0]["unit"],
            "video_onset_s_median": None if vid.size == 0 else round(float(np.median(vid)), 3),
        }
    out["summary"] = summary
    return out


if __name__ == "__main__":
    res = analyze(sys.argv[1])
    print(json.dumps(res["summary"], indent=1))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(res, indent=1))
