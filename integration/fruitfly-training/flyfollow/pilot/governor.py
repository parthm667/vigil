"""PersonGovernor (plan 4.1): the shared rules between every pursuit controller and the rc stream.

The controller proposes raw (yaw, fb). The governor owns ud (image-row P loop with altitude
floors), lr (always 0), lost-target handling, minimum distance, stick clamps, slew limits,
battery and the brain watchdog, and returns the final integer sticks plus flags. SAFETY
interventions (min distance, lost-target takeover, watchdog) are flagged apart from clamps and
slew limits, because training penalizes only the former.

The minimum-distance rule uses the range ESTIMATE from box size (fy * target_size / h), exactly
as at runtime, never simulator truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

from flyfollow.interfaces import IMG_H, BoxState, Settings


@dataclass
class GovernorConfig:
    ud_gain: float = 60.0  # ud = clip(ud_gain * (cy_ref - cy) / H, -ud_max, ud_max)
    ud_max: float = 30.0
    lost_yaw_stick: float = 20.0
    lost_hover_s: float = 5.0  # lost this long (after the filter's hold): hover, say "I lost you"
    lost_land_s: float = 10.0  # lost this long: land, only for kinds in land_kinds
    land_kinds: tuple[str, ...] = ("follow",)
    brain_timeout_s: float = 0.5
    backoff_stick: float = 20.0  # reverse stick while inside z_min (capped by max_back_stick)
    slew_yaw_per_s: float = 400.0
    slew_fb_per_s: float = 200.0
    slew_ud_per_s: float = 200.0
    alt_floor_follow_m: float = 0.5
    alt_floor_approach_m: float = 0.8
    alt_ceiling_m: float = 2.4
    min_battery_pct: float = 25.0

    @classmethod
    def from_dict(cls, d: dict | None) -> GovernorConfig:
        d = d or {}
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        if "land_kinds" in kw:
            kw["land_kinds"] = tuple(kw["land_kinds"])
        return cls(**kw)


@dataclass
class GovOutput:
    lr: int
    fb: int
    ud: int
    yaw: int
    safety: bool = False  # min distance, lost-target takeover or watchdog (penalized in training)
    clamped: bool = False  # a stick clamp or slew limit changed the command (logged, not penalized)
    hover: bool = False
    land: bool = False
    reasons: tuple[str, ...] = ()


def _slew(v: float, prev: float, step: float) -> float:
    """Rate-limit moves away from zero; moves toward zero are instant (as in FlyDrones' SafetyGovernor).
    A sign change goes to zero instantly and is then limited."""
    if v == 0.0 or (v > 0.0) == (prev > 0.0) and abs(v) <= abs(prev):
        return v
    if (v > 0.0) != (prev > 0.0) and prev != 0.0:
        prev = 0.0
    return max(prev - step, min(prev + step, v))


class PersonGovernor:
    def __init__(self, cfg: GovernorConfig | dict | None = None):
        self.cfg = cfg if isinstance(cfg, GovernorConfig) else GovernorConfig.from_dict(cfg)
        self.reset()

    def reset(self) -> None:
        self._prev = [0.0, 0.0, 0.0]  # yaw, fb, ud as sent
        self._last_bearing: float | None = None  # raw bearing (rad) of the last valid box
        self.landed = False

    def filter(self, yaw: float, fb: float, box: BoxState, st: Settings, dt: float, alt_m: float | None = None,
               brain_age_s: float = 0.0, battery_pct: float | None = None) -> GovOutput:
        c = self.cfg
        reasons: list[str] = []
        safety = clamped = hover = land = False
        yaw = 0.0 if yaw != yaw else float(yaw)  # NaN guard
        fb = 0.0 if fb != fb else float(fb)
        y = max(-st.max_yaw_stick, min(st.max_yaw_stick, yaw))
        f = max(-st.max_back_stick, min(st.max_fwd_stick, fb))
        if y != yaw or f != fb:
            clamped = True
            reasons.append("clamp")
        ud = 0.0
        if box.valid:
            self._last_bearing = math.atan((box.cx - st.cx0) / st.fx)
        if brain_age_s > c.brain_timeout_s:
            y = f = 0.0
            safety = hover = True
            reasons.append("watchdog")
        elif not box.valid:
            safety = True
            f = 0.0
            if box.lost_s >= c.lost_land_s and st.kind in c.land_kinds:
                land = True
                y = 0.0
                reasons.append("lost_land")
            elif box.lost_s >= c.lost_hover_s:
                hover = True
                y = 0.0
                reasons.append("lost_hover")
            else:
                lb = self._last_bearing
                y = 0.0 if lb is None else (c.lost_yaw_stick if lb >= 0.0 else -c.lost_yaw_stick)
                reasons.append("lost_target")
        else:
            ud = c.ud_gain * (st.cy_ref_frac * IMG_H - box.cy) / IMG_H
            ud = max(-c.ud_max, min(c.ud_max, ud))
            if box.h > 0.0 and st.fy * st.target_size_m / box.h < st.z_min_m:
                back = -min(st.max_back_stick, c.backoff_stick)
                if f > back:
                    f = back
                safety = True
                reasons.append("min_distance")
        if alt_m is not None:
            floor = c.alt_floor_approach_m if st.kind == "approach" else c.alt_floor_follow_m
            if alt_m <= floor and ud < 0.0:
                ud = 0.0
                reasons.append("alt_floor")
            elif alt_m >= c.alt_ceiling_m and ud > 0.0:
                ud = 0.0
                reasons.append("alt_ceiling")
        if battery_pct is not None and battery_pct < c.min_battery_pct:
            land = True
            reasons.append("battery")
        if land:
            y = f = ud = 0.0
            self.landed = True
        p = self._prev
        y2 = _slew(y, p[0], c.slew_yaw_per_s * dt)
        f2 = _slew(f, p[1], c.slew_fb_per_s * dt)
        u2 = _slew(ud, p[2], c.slew_ud_per_s * dt)
        if y2 != y or f2 != f:
            clamped = True
            reasons.append("slew")
        yi, fi, ui = int(round(y2)), int(round(f2)), int(round(u2))
        self._prev = [float(yi), float(fi), float(ui)]
        return GovOutput(0, fi, ui, yi, safety, clamped, hover, land, tuple(reasons))
