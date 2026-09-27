"""R0 stick response (plan 4.8 R0): rc steps on each axis at sticks 30, 60 and 100, then a fit of gain, dead time
and time constant per axis and stick, a nonlinearity report and the values for configs/env.yaml profiles.demo.

    python -m flyfollow.tools.stick_response --send                         # fly (typed confirmation), log, fit
    python -m flyfollow.tools.stick_response --analyze data/r0/stick_log_<stamp>.json
    python -m flyfollow.tools.stick_response --sim                          # rehearse on the simulated Tello, no drone

The lag test (origin/rl-pipeline data/lag_test) measured stick 30 only; the controllers use up to 60 (forward, yaw)
and the Tello's stick map may be nonlinear, hence 60 and 100. Each step is + then - so the drone comes back.
After the steps, "forward 50" / "back 50" and the same after "cw 90" check the vgx unit (integrated -vgx / 10 should
be 0.5 m if vgx is dm/s) and whether vgx/vgy are body frame (after cw 90 a body-frame vgx still responds).

Fit (adapted from Parth Mhaske's flyfollow/tools/analyze_lag.py on origin/rl-pipeline, 2026-09-26): per step,
rate(t) = g (1 - exp(-(t - d) / tau)) from dead time d until release (+ d), then exponential decay; grid search over d
and tau with g in closed form. Forward/lateral/vertical fit vgx/vgy/vgz / 10 in m/s with the lag test's signs
(forward, right and up read negative). Yaw fits the unwrapped yaw angle with the closed-form integral of that rate.
Writes data/r0/stick_log_<stamp>.json (raw) and data/r0/stick_response_<stamp>.json (fit, feeds update_sim_from_r0).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flyfollow.interfaces import DT, data_root

STATE_COLS = ["t", "pitch", "roll", "yaw", "vgx", "vgy", "vgz", "tof", "h", "bat"]  # same as analyze_lag
AXES = ("yaw", "fb", "lr", "ud")
LEGACY_AXES = {"pitch": "fb", "roll": "lr", "throttle": "ud", "yaw": "yaw"}  # the lag test's names
AXIS_VEL = {"fb": ("vgx", -1.0), "lr": ("vgy", -1.0), "ud": ("vgz", -1.0)}  # channel, sign so + stick reads +
RC_INDEX = {"lr": 0, "fb": 1, "ud": 2, "yaw": 3}
D_GRID = np.arange(0.05, 1.2, 0.01)
TAU_GRID = np.arange(0.02, 1.5, 0.01)
DEFAULT_DURS = {30: 1.5, 60: 1.2, 100: 1.0}


def r0_dir() -> Path:
    d = data_root() / "r0"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------------------------ fit
def _basis_rate(t: np.ndarray, d: float, taus: np.ndarray, t_off: float) -> np.ndarray:
    """(n_tau, n_t) unit-gain rate response to a stick step held for t_off, delayed by d."""
    on = np.clip(t - d, 0.0, None)[None, :]
    tau = taus[:, None]
    y = 1.0 - np.exp(-np.minimum(on, t_off) / tau)
    off = (t - d - t_off)[None, :]
    return np.where(off > 0, y * np.exp(-np.clip(off, 0.0, None) / tau), y)


def _basis_angle(t: np.ndarray, d: float, taus: np.ndarray, t_off: float) -> np.ndarray:
    """Closed-form integral of _basis_rate (unit gain), for fitting the yaw angle."""
    tau = taus[:, None]
    s = np.clip(t - d, 0.0, None)[None, :]
    s_on = np.minimum(s, t_off)
    i_on = s_on - tau * (1.0 - np.exp(-s_on / tau))
    after = np.clip(s - t_off, 0.0, None)
    r_off = 1.0 - np.exp(-t_off / tau)
    return i_on + r_off * tau * (1.0 - np.exp(-after / tau))


def fit_step(t: np.ndarray, y: np.ndarray, t_off: float, angle: bool = False) -> tuple[float, float, float, float]:
    """(dead time, tau, gain, rmse) by grid search over d and tau, least-squares gain."""
    best = (np.inf, 0.0, 0.0, 0.0)
    for d in D_GRID:
        B = (_basis_angle if angle else _basis_rate)(t, d, TAU_GRID, t_off)
        den = np.einsum("ij,ij->i", B, B)
        g = np.where(den > 0, B @ y / np.maximum(den, 1e-12), 0.0)
        err = np.mean((y[None, :] - g[:, None] * B) ** 2, axis=1)
        i = int(np.argmin(err))
        if err[i] < best[0]:
            best = (float(err[i]), float(d), float(TAU_GRID[i]), float(g[i]))
    err, d, tau, g = best
    return d, tau, g, math.sqrt(err)


def _event_axis(ev: dict) -> str:
    return LEGACY_AXES.get(ev.get("axis"), ev.get("axis"))


def analyze(log: dict) -> dict:
    """Fit every rc step of a stick log (ours or the lag test's) and summarize per axis and stick."""
    S = np.asarray(log["state"], dtype=float)
    cols = log.get("state_cols", STATE_COLS)
    col = {c: i for i, c in enumerate(cols)}
    t = S[:, col["t"]]
    mag0 = (log.get("meta", {}).get("args") or {}).get("mag")
    steps = []
    for ev in log["events"]:
        if ev.get("type") != "rc":
            continue
        ax, sign = _event_axis(ev), int(ev["sign"])
        stick = int(ev.get("stick", mag0 or 30))
        t0, t1 = ev["t_cmd"], ev["t_release"]
        m = (t >= t0 - 0.3) & (t <= t1 + 2.5)
        if m.sum() < 8:
            continue
        tt = t[m] - t0
        if ax == "yaw":
            ang = np.rad2deg(np.unwrap(np.deg2rad(S[m, col["yaw"]])))
            pre = ang[tt < 0]
            ang = (ang - (pre.mean() if pre.size else ang[0])) * sign
            d, tau, g, rmse = fit_step(tt, ang, t1 - t0, angle=True)
            unit = "deg/s"
        else:
            ch, sg = AXIS_VEL[ax]
            y = S[m, col[ch]] * sg * sign / 10.0  # dm/s -> m/s (unit checked below)
            d, tau, g, rmse = fit_step(tt, y, t1 - t0)
            unit = "m/s"
        steps.append({"axis": ax, "stick": stick, "sign": sign, "dead_time_s": round(d, 3), "tau_s": round(tau, 3),
                      "gain": round(g, 4), "gain_per_100": round(g * 100.0 / stick, 4), "unit": unit, "rmse": round(rmse, 4)})
    summary: dict = {}
    for ax in AXES:
        rows = [s for s in steps if s["axis"] == ax]
        if not rows:
            continue
        per = {}
        for st in sorted({r["stick"] for r in rows}):
            rr = [r for r in rows if r["stick"] == st]
            med = lambda k: round(float(np.median([r[k] for r in rr])), 4)
            per[str(st)] = {"n": len(rr), "gain": med("gain"), "gain_per_100": med("gain_per_100"),
                            "dead_time_s": med("dead_time_s"), "tau_s": med("tau_s"), "unit": rr[0]["unit"],
                            "gain_spread": round(float(np.ptp([r["gain_per_100"] for r in rr])), 4)}
        # linear gain through the origin over the sticks the controllers use (<= 60), for the linear sim model
        use = [r for r in rows if r["stick"] <= 60] or rows
        x = np.array([r["stick"] for r in use], dtype=float)
        yv = np.array([r["gain"] for r in use], dtype=float)
        lin = float(x @ yv / (x @ x)) * 100.0
        ref = per[min(per, key=int)]["gain_per_100"]
        ratios = {k: round(v["gain_per_100"] / ref, 3) if ref else None for k, v in per.items()}
        nonlin = any(r is not None and not 0.8 <= r <= 1.25 for r in ratios.values())
        summary[ax] = {"per_stick": per, "gain_per_100_linear": round(lin, 4),
                       "dead_time_s": round(float(np.median([r["dead_time_s"] for r in use])), 3),
                       "tau_s": round(float(np.median([r["tau_s"] for r in use])), 3),
                       "ratio_to_lowest_stick": ratios, "nonlinear": nonlin, "unit": rows[0]["unit"] + " per 100 stick"}
    out = {"steps": steps, "summary": summary, "unit_check": unit_check(log, t, S, col)}
    su = {}
    if "fb" in summary:
        su.update(fwd_gain_mps=summary["fb"]["gain_per_100_linear"], tau_fwd_s=summary["fb"]["tau_s"],
                  fwd_dead_s=summary["fb"]["dead_time_s"])
    if "yaw" in summary:
        su.update(yaw_gain_dps=summary["yaw"]["gain_per_100_linear"], tau_yaw_s=summary["yaw"]["tau_s"],
                  yaw_dead_s=summary["yaw"]["dead_time_s"])
    if "ud" in summary:
        su.update(vz_gain_mps=summary["ud"]["gain_per_100_linear"], tau_z_s=summary["ud"]["tau_s"],
                  vz_dead_s=summary["ud"]["dead_time_s"])
    out["sim_update"] = su
    return out


def unit_check(log: dict, t: np.ndarray, S: np.ndarray, col: dict) -> list[dict]:
    """Integrate -vgx/10 and -vgy/10 over each SDK move: 0.5 m for a 50 cm move means vg* are dm/s."""
    res = []
    for ev in log["events"]:
        if ev.get("type") != "move" or not str(ev.get("cmd", "")).split()[0] in ("forward", "back", "left", "right"):
            continue
        m = (t >= ev["t_cmd"]) & (t <= ev["t_release"] + 0.5)
        if m.sum() < 3:
            continue
        tt = t[m]
        fx = float(np.trapezoid(-S[m, col["vgx"]] / 10.0, tt))
        fy = float(np.trapezoid(-S[m, col["vgy"]] / 10.0, tt))
        cm = int(str(ev["cmd"]).split()[1])
        res.append({"cmd": ev["cmd"], "yaw_before_deg": ev.get("yaw_before"), "expected_m": cm / 100.0,
                    "int_fwd_m": round(fx, 3), "int_right_m": round(fy, 3),
                    "ratio": round(math.hypot(fx, fy) / (cm / 100.0), 2)})
    return res


def report(res: dict) -> str:
    lines = []
    for ax, s in res["summary"].items():
        lines.append(f"{ax:4s} linear gain {s['gain_per_100_linear']:.3f} {s['unit']}, dead {s['dead_time_s']:.2f} s, "
                     f"tau {s['tau_s']:.2f} s{'   NONLINEAR' if s['nonlinear'] else ''}")
        for st, p in s["per_stick"].items():
            lines.append(f"       stick {st:>3}: gain/100 {p['gain_per_100']:.3f} (n {p['n']}, spread {p['gain_spread']:.3f}), "
                         f"dead {p['dead_time_s']:.2f} s, tau {p['tau_s']:.2f} s, ratio {s['ratio_to_lowest_stick'][st]}")
    for u in res["unit_check"]:
        lines.append(f"unit check {u['cmd']} (yaw before {u['yaw_before_deg']}): integrated fwd {u['int_fwd_m']} m, "
                     f"right {u['int_right_m']} m, expected {u['expected_m']} m -> ratio {u['ratio']} "
                     f"({'dm/s confirmed' if 0.6 <= u['ratio'] <= 1.5 else 'NOT dm/s: check units'})")
    lines.append("sim_update (profiles.demo): " + json.dumps(res["sim_update"]))
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ synthetic log
def synth_log(dp, sticks=(30, 60, 100), axes=AXES, durs=None, settle=2.5, seed=0, state_hz=10.0, unit_moves=True) -> dict:
    """Fly the procedure on flyfollow.sim.drone_model and log it like the real Tello (ints, signs, 10 Hz jitter).

    Same sign conventions as sim_world.SimWorld.raw_state: forward and right read negative vgx and vgy, up reads
    negative vgz, yaw grows clockwise.
    """
    from flyfollow.sim.drone_model import DroneModel

    rng = np.random.default_rng(seed)
    durs = durs or DEFAULT_DURS
    dm = DroneModel(dp)
    dm.reset(0.0, 0.0, 1.0, 0.0)
    t, state, events = 0.0, [], []
    next_s = 0.0
    sched: list[tuple[float, float, tuple]] = []  # (t_start, t_end, sticks)
    tc = 2.0
    for ax in axes:
        for st in sticks:
            for sign in (1, -1):
                u = [0, 0, 0, 0]
                u[RC_INDEX[ax]] = sign * st
                d = durs.get(st, 1.0)
                sched.append((tc, tc + d, tuple(u)))
                events.append({"type": "rc", "axis": ax, "stick": st, "sign": sign, "t_cmd": tc, "t_release": tc + d})
                tc += d + settle
    moves = []
    if unit_moves:
        for cmd, yaw_before in (("forward 50", 0), ("back 50", 0)):
            moves.append((tc, tc + 1.0, cmd, yaw_before))
            events.append({"type": "move", "cmd": cmd, "t_cmd": tc, "t_release": tc + 1.6, "yaw_before": yaw_before})
            tc += 3.0
    t_end = tc + 1.0
    n = int(t_end / DT)
    for i in range(n):
        t = i * DT
        u = next((s[2] for s in sched if s[0] <= t < s[1]), (0, 0, 0, 0))
        dm.step(*u)
        for (a, b, cmd, _) in moves:  # an SDK move as a fixed 0.5 m/s for 1 s, stopping sharply
            if a <= t < b + DT:
                v = (0.5 if cmd.startswith("forward") else -0.5) if t < b else 0.0
                dm.vx, dm.vy = v * math.cos(dm.psi), v * math.sin(dm.psi)
        t += DT  # the state after this step
        if t >= next_s:
            next_s = t + 1.0 / state_hz + rng.uniform(-0.01, 0.01)
            c, s = math.cos(dm.psi), math.sin(dm.psi)
            v_fwd = dm.vx * c + dm.vy * s
            v_right = dm.vx * s - dm.vy * c
            yaw = (-math.degrees(dm.psi) + 180.0) % 360.0 - 180.0
            state.append([t + rng.uniform(0.0, 0.02), 0, 0, int(round(yaw)), int(round(-v_fwd * 10)),
                          int(round(-v_right * 10)), int(round(-dm.vz * 10)), int(dm.z * 100), int(dm.z * 100) - 15, 80])
    return {"meta": {"source": "synth", "params": {k: getattr(dp, k) for k in dp.__dataclass_fields__}},
            "state_cols": STATE_COLS, "state": state, "events": events}


# ------------------------------------------------------------------------------------------------ real flight
class Flight:
    """Real-Tello procedure: rc keepalive at 20 Hz, 10 Hz state logger, key watcher (l = land, e = emergency)."""

    def __init__(self, link, rc_hz: float = 20.0):
        self.link = link
        self.sticks = (0, 0, 0, 0)
        self.state: list[list] = []
        self.events: list[dict] = []
        self.stop = threading.Event()
        self.abort: str | None = None
        self.rc_on = False
        self.rc_hz = rc_hz

    def start(self) -> None:
        threading.Thread(target=self._rc_loop, daemon=True).start()
        threading.Thread(target=self._state_loop, daemon=True).start()
        threading.Thread(target=self._keys, daemon=True).start()

    def _rc_loop(self) -> None:
        while not self.stop.is_set():
            if self.rc_on:
                self.link.rc(*self.sticks)
            time.sleep(1.0 / self.rc_hz)

    def _state_loop(self) -> None:
        last = None
        while not self.stop.is_set():
            st = self.link.state()
            if st and st is not last:
                last = st
                self.state.append([time.time()] + [st.get(k, 0) for k in STATE_COLS[1:]])
            time.sleep(0.005)

    def _keys(self) -> None:
        for line in sys.stdin:
            k = line.strip().lower()
            if k in ("l", "e"):
                self.abort = "land" if k == "l" else "emergency"
                if k == "e":
                    self.link.raw("emergency")
                return

    def h_cm(self) -> float:
        return float(self.state[-1][STATE_COLS.index("h")]) if self.state else 0.0

    def bat(self) -> float:
        return float(self.state[-1][STATE_COLS.index("bat")]) if self.state else 100.0

    def hold(self, s: float) -> None:
        t_end = time.time() + s
        while time.time() < t_end:
            if self.abort:
                raise KeyboardInterrupt(self.abort)
            time.sleep(0.01)

    def step(self, axis: str, stick: int, sign: int, dur: float, settle: float) -> None:
        u = [0, 0, 0, 0]
        u[RC_INDEX[axis]] = sign * stick
        t0 = time.time()
        self.sticks = tuple(u)
        self.hold(dur)
        self.sticks = (0, 0, 0, 0)
        self.events.append({"type": "rc", "axis": axis, "stick": stick, "sign": sign, "t_cmd": t0, "t_release": time.time()})
        print(f"  {axis} {'+' if sign > 0 else '-'}{stick} for {dur:.1f} s   (h {self.h_cm():.0f} cm, bat {self.bat():.0f} %)")
        self.hold(settle)

    def move(self, cmd: str) -> str:
        self.rc_on = False
        time.sleep(0.1)
        yaw = self.state[-1][STATE_COLS.index("yaw")] if self.state else None
        t0 = time.time()
        resp = self.link.cmd(cmd, 10)
        self.events.append({"type": "move", "cmd": cmd, "t_cmd": t0, "t_release": time.time(), "reply": resp, "yaw_before": yaw})
        self.link.rc(0, 0, 0, 0)
        self.rc_on = True
        print(f"  {cmd}: {resp}")
        self.hold(1.5)
        return resp


def fly(a) -> dict:
    import logging

    from djitellopy import Tello

    from flyfollow.runtime.tello_io import FIREWALL_HINT, STATE_PORT, TelloLink, check_udp_port, port_hint

    logging.getLogger("djitellopy").setLevel(logging.WARNING)
    for port in (8889, STATE_PORT):
        if check_udp_port(port):
            sys.exit(port_hint(port))
    sticks = [int(s) for s in a.sticks.split(",")]
    axes = [x for x in a.axes.split(",") if x]
    durs = {**DEFAULT_DURS, **{int(k): float(v) for k, v in (p.split(":") for p in a.durs.split(",") if p)}}
    n_steps = len(axes) * len(sticks) * 2 * a.trials
    t_est = n_steps * (a.settle + 1.3) + 40
    print(f"""
R0 STICK RESPONSE: {n_steps} rc steps ({', '.join(axes)} at sticks {sticks}, + then -, {a.trials} trial(s)), about {t_est:.0f} s.
 1. Clear area: at least 4 x 4 m, 2 m ceiling, nothing fragile. Stick 100 forward for 1 s moves 1 m or more.
 2. Tello on the floor in the middle, camera toward the longest free direction. Battery >= {a.min_battery:.0f} %.
 3. Spotter ready. During the flight type l + Enter to LAND, e + Enter for EMERGENCY (motors off). Ctrl+C lands.
 4. The drone takes off, hovers {a.hover:.0f} s, steps each axis (+ then - so it comes back), then runs the vgx unit
    check (forward 50, back 50, cw 90, forward 50, back 50, ccw 90) unless --no-unit-check, and lands.
""")
    tello = Tello(host=a.ip)
    link = TelloLink(tello)
    try:
        link.connect()
    except Exception as e:
        sys.exit(f"connect failed: {e}\n{FIREWALL_HINT}")
    fl = Flight(link)
    fl.start()
    time.sleep(0.5)
    print(f"battery {fl.bat():.0f} %")
    if fl.bat() < a.min_battery:
        sys.exit(f"battery below {a.min_battery} %: charge first")
    if input('Type FLY and Enter to take off (anything else aborts): ').strip() != "FLY":
        sys.exit("aborted")
    meta = {"stamp": time.strftime("%Y%m%d_%H%M%S"), "args": vars(a), "battery": fl.bat(), "tool": "stick_response"}
    try:
        t0 = time.time()
        resp = link.cmd("takeoff", 20)
        meta["takeoff"] = [resp, time.time() - t0]
        print(f"takeoff: {resp}")
        fl.rc_on = True
        fl.hold(a.hover)
        for _ in range(a.trials):
            for ax in axes:
                for st in sticks:
                    for sign in (1, -1):
                        if fl.bat() < 20:
                            raise KeyboardInterrupt("battery < 20 %")
                        if ax == "ud" and sign < 0 and fl.h_cm() < 60:
                            print("  skip ud down: below 60 cm")
                            continue
                        fl.step(ax, st, sign, durs.get(st, 1.0), a.settle)
        if not a.no_unit_check:
            for cmd in ("forward 50", "back 50", "cw 90", "forward 50", "back 50", "ccw 90"):
                fl.move(cmd)
    except KeyboardInterrupt as e:
        print(f"abort: {e or 'Ctrl+C'}")
    finally:
        fl.sticks = (0, 0, 0, 0)
        fl.rc_on = False
        if fl.abort != "emergency":
            print("land:", link.cmd("land", 7))
        time.sleep(1.0)
        fl.stop.set()
    return {"meta": meta, "state_cols": STATE_COLS, "state": fl.state, "events": fl.events}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="R0 stick response: rc steps, fit gain / dead time / tau per axis and stick")
    ap.add_argument("--send", action="store_true", help="fly (asks for typed confirmation)")
    ap.add_argument("--analyze", metavar="LOG", help="re-fit a saved stick log (ours or data/lag_test/*fulllogs*.json)")
    ap.add_argument("--sim", action="store_true", help="rehearse on flyfollow.sim.drone_model with the demo profile")
    ap.add_argument("--ip", default="192.168.10.1")
    ap.add_argument("--sticks", default="30,60,100")
    ap.add_argument("--axes", default="yaw,fb,lr,ud")
    ap.add_argument("--durs", default="", help="per-stick step length, e.g. 30:1.5,60:1.2,100:1.0")
    ap.add_argument("--settle", type=float, default=2.5)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--hover", type=float, default=3.0)
    ap.add_argument("--min-battery", type=float, default=50.0)
    ap.add_argument("--no-unit-check", action="store_true")
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if a.analyze:
        log = json.loads(Path(a.analyze).read_text())
        src = a.analyze
    elif a.sim:
        from flyfollow.runtime.sim_world import demo_drone_params

        dp = demo_drone_params()
        dp.drift_sigma_mps = 0.0
        log = synth_log(dp, sticks=[int(s) for s in a.sticks.split(",")], axes=a.axes.split(","))
        src = "sim"
        print("true (demo profile):", json.dumps(log["meta"]["params"]))
    elif a.send:
        log = fly(a)
        p = r0_dir() / f"stick_log_{stamp}.json"
        p.write_text(json.dumps(log))
        print(f"log: {p}")
        src = str(p)
    else:
        ap.error("pick one of --send, --analyze LOG, --sim")
    res = analyze(log)
    res["source"] = src
    res["stamp"] = stamp
    print(report(res))
    if not a.sim:
        out = r0_dir() / f"stick_response_{stamp}.json"
        out.write_text(json.dumps(res, indent=1))
        print(f"results: {out}\nnext: python -m flyfollow.tools.update_sim_from_r0 (prints the env.yaml diff; --write applies it)")


if __name__ == "__main__":
    main()
