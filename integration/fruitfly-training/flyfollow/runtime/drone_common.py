"""Shared by tello_io (real Tello) and sim_world (simulated Tello): rc arbitration, safety rules, move parsing and
the tello_state message, so both drone back ends obey exactly the same rules.

Arbitration (messages.RC_OWNER, RC_TIMEOUT_S):
- The current mode comes from "mode" messages (field "to"). Before the first one the mode is IDLE.
- An "rc" message is accepted if src == "operator" (any mode), or src == RC_OWNER[mode] and its "mode" field equals
  the current mode. Messages older than RC_TIMEOUT_S on arrival are dropped as stale.
- Each tick the sent sticks are: the newest operator rc if it arrived within RC_TIMEOUT_S, else the owner's newest rc
  if it arrived within RC_TIMEOUT_S, else 0 0 0 0 (hover). Sticks are rounded and clamped to -100..100.
- A mode change drops every stored non-operator rc, so the new owner starts from hover.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from flyfollow.runtime.messages import MODES, RC_OWNER, RC_SOURCES, RC_TIMEOUT_S

# Safety limits (plan 3.2, 9; governor.min_battery_pct in configs/env.yaml).
MIN_BATTERY_PCT = 25.0
# temph at which we land. The Tello powers itself off when it overheats; DJI publishes no threshold (the manual only
# gives a 0 to 40 C ambient range) and forum logs show temph in the 80s to 90s C before shutdown (unverified).
MAX_TEMP_C = 90.0
VIDEO_LOST_S = 5.0
BUS_LOST_S = 2.0  # no loopback of our own tello_state through the broker for this long = bus lost

MOVE_DIST = ("forward", "back", "left", "right", "up", "down")
MOVE_ROT = ("cw", "ccw")
HOVER = (0, 0, 0, 0)


def clamp_stick(v) -> int:
    """Round and clamp to a final Tello stick integer; NaN, None or garbage become 0."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f):
        return 0
    return int(max(-100, min(100, round(f))))


class RcArbiter:
    """Picks the sticks to send each tick. Times are laptop time.time() seconds."""

    def __init__(self, timeout_s: float = RC_TIMEOUT_S):
        self.timeout_s = timeout_s
        self.mode = "IDLE"
        self.latest: dict[str, tuple[float, tuple[int, int, int, int]]] = {}  # src -> (t_rx, (lr, fb, ud, yaw))
        self.rejected: Counter = Counter()
        self.accepted = 0

    def set_mode(self, mode: str) -> bool:
        if mode not in MODES:
            self.rejected["bad_mode"] += 1
            return False
        if mode != self.mode:
            self.mode = mode
            self.latest = {k: v for k, v in self.latest.items() if k == "operator"}
        return True

    def offer(self, m: dict, now: float) -> bool:
        """Store an rc message if its source may command in the current mode. Returns True if accepted."""
        src = m.get("src")
        if src not in RC_SOURCES:
            self.rejected["bad_src"] += 1
            return False
        if src != "operator":
            if src != RC_OWNER.get(self.mode):
                self.rejected["not_owner"] += 1
                return False
            if m.get("mode") != self.mode:
                self.rejected["mode_mismatch"] += 1
                return False
        t = m.get("t")
        if isinstance(t, (int, float)) and now - float(t) > self.timeout_s:
            self.rejected["stale"] += 1
            return False
        self.latest[src] = (now, tuple(clamp_stick(m.get(k)) for k in ("lr", "fb", "ud", "yaw")))
        self.accepted += 1
        return True

    def select(self, now: float) -> tuple[tuple[int, int, int, int], str]:
        """((lr, fb, ud, yaw), src) for this tick; src "hover" when nothing fresh is accepted."""
        op = self.latest.get("operator")
        if op is not None and now - op[0] <= self.timeout_s:
            return op[1], "operator"
        owner = RC_OWNER.get(self.mode)
        if owner is not None:
            r = self.latest.get(owner)
            if r is not None and now - r[0] <= self.timeout_s:
                return r[1], owner
        return HOVER, "hover"

    def clear(self) -> None:
        self.latest.clear()


def parse_move(args: dict | None) -> tuple[str | None, str]:
    """tello_cmd move args -> (SDK command text, error). Distances 20..500 cm, rotations 1..360 deg (SDK limits)."""
    args = args or {}
    d = str(args.get("direction", "")).lower()
    try:
        v = int(round(float(args.get("value"))))
    except (TypeError, ValueError):
        return None, f"bad value {args.get('value')!r}"
    if d in MOVE_DIST:
        if not 20 <= v <= 500:
            return None, f"{d} {v} cm outside 20..500"
    elif d in MOVE_ROT:
        if not 1 <= v <= 360:
            return None, f"{d} {v} deg outside 1..360"
    else:
        return None, f"bad direction {d!r}"
    return f"{d} {v}", ""


def move_timeout_s(sdk: str, args: dict | None = None) -> float:
    """How long to wait for the Tello's "ok" after a move: about 0.4 m/s or 40 deg/s plus 5 s of spin-up and settle."""
    if args and args.get("timeout_s"):
        return float(args["timeout_s"])
    v = float(sdk.split()[1])
    return 5.0 + v / 40.0


@dataclass
class SafetyLimits:
    min_battery_pct: float = MIN_BATTERY_PCT
    max_temp_c: float = MAX_TEMP_C
    video_lost_s: float = VIDEO_LOST_S
    bus_lost_s: float = BUS_LOST_S
    idle_land_s: float = 0.0  # land if flying with no accepted rc or command for this long (0 = off)


def safety_reason(lim: SafetyLimits, *, flying: bool, bat_pct, temph_c, video_enabled: bool, video_age_s: float,
                  bus_age_s: float | None, idle_s: float) -> str | None:
    """Why we must land now (only while flying), or None."""
    if not flying:
        return None
    if bat_pct is not None and bat_pct < lim.min_battery_pct:
        return f"battery {bat_pct}% < {lim.min_battery_pct:g}%"
    if temph_c is not None and temph_c >= lim.max_temp_c:
        return f"temperature {temph_c} C >= {lim.max_temp_c:g} C"
    if video_enabled and video_age_s > lim.video_lost_s:
        return f"video lost for {video_age_s:.1f} s > {lim.video_lost_s:g} s"
    if bus_age_s is not None and bus_age_s > lim.bus_lost_s:
        return f"bus lost for {bus_age_s:.1f} s > {lim.bus_lost_s:g} s"
    if lim.idle_land_s > 0 and idle_s > lim.idle_land_s:
        return f"no accepted rc or command for {idle_s:.0f} s"
    return None


def takeoff_block_reason(lim: SafetyLimits, *, bat_pct, temph_c, video_enabled: bool, video_ok: bool) -> str | None:
    if bat_pct is not None and bat_pct < lim.min_battery_pct:
        return f"battery {bat_pct}% < {lim.min_battery_pct:g}%"
    if temph_c is not None and temph_c >= lim.max_temp_c:
        return f"temperature {temph_c} C >= {lim.max_temp_c:g} C"
    if video_enabled and not video_ok:
        return "no video (start without --no-video only when the stream works)"
    return None


def _num(raw: dict, k: str, default=0):
    v = raw.get(k, default)
    return v if isinstance(v, (int, float)) else default


def state_fields(raw: dict) -> dict:
    """Tello state packet (djitellopy keys) -> tello_state fields, without the runtime flags.

    vg* stay in the units received (dm/s per SDK 3.0, confirm at R0 with stick_response). vg*_mps = raw / 10.
    v_fwd_mps, v_right_mps, v_up_mps flip the signs: the 2026-09-26 lag test (origin/rl-pipeline analyze_lag.py) saw
    negative vgx for forward, negative vgy for right and negative vgz for climbing. Whether vgx/vgy are in the body
    frame or the takeoff-heading frame is not verified.
    """
    vgx, vgy, vgz = _num(raw, "vgx"), _num(raw, "vgy"), _num(raw, "vgz")
    return {
        "yaw_deg": _num(raw, "yaw"), "pitch_deg": _num(raw, "pitch"), "roll_deg": _num(raw, "roll"),
        "vgx_dms": vgx, "vgy_dms": vgy, "vgz_dms": vgz,
        "vgx_mps": vgx / 10.0, "vgy_mps": vgy / 10.0, "vgz_mps": vgz / 10.0,
        "v_fwd_mps": -vgx / 10.0, "v_right_mps": -vgy / 10.0, "v_up_mps": -vgz / 10.0,
        "h_cm": _num(raw, "h"), "tof_cm": _num(raw, "tof"), "bat_pct": _num(raw, "bat", None),
        "temph_c": _num(raw, "temph", None), "templ_c": _num(raw, "templ", None),
        "agx": _num(raw, "agx", 0.0), "agy": _num(raw, "agy", 0.0), "agz": _num(raw, "agz", 0.0),
        "baro_m": _num(raw, "baro", 0.0), "flight_time_s": _num(raw, "time", 0),
    }
