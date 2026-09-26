"""Follow-behind: hover behind and above the person's head, facing their back (continuous rc).

Every fresh frame with the locked person:
  yaw      turns toward them (bearing, with a deadband)
  forward  holds the follow distance; moving TOWARD them only when even the most conservative range cue
           (range_lo_m) says it is safe and their box does not fill the frame; too close -> back off
  up/down  holds follow.altitude_m (above their head)
  lateral  orbits around them when they turn (facing_deg != 0), so the drone ends up behind them
When they are not seen: hold briefly, then yaw toward where they were last seen, then hover.
Never finishes on its own: the mission switches behaviour when a query arrives.
"""

from __future__ import annotations

import math
import statistics
from collections import deque

from ..config import FollowCfg
from .base import RUNNING, Behavior, Ctx, clamp, deadband


class FollowBehind(Behavior):
    name = "follow"

    def __init__(self, cfg: FollowCfg):
        super().__init__()
        self.c = cfg
        self.max_box_width_frac = cfg.max_box_width_frac
        self.ranges: deque = deque(maxlen=5)  # (t, range): only recent ones count, never stale values
        self.facings: deque = deque(maxlen=5)
        self.range_hist: deque = deque(maxlen=30)  # (t, filtered range) for the approach-rate feed-forward
        self.fwd_hist: deque = deque(maxlen=60)  # (t, commanded forward rc)
        self.last_close = False
        self.cmd = (0, 0, 0, 0)
        self.last_seen_t: float | None = None
        self.last_bearing = 0.0

    def start(self, ctx: Ctx) -> None:
        ctx.perception.set_mode("follow")
        self.ranges.clear()
        self.facings.clear()
        self.range_hist.clear()
        self.fwd_hist.clear()
        self.last_close = False
        self.cmd = (0, 0, 0, 0)
        self.last_seen_t = None

    def _range(self, now: float) -> float | None:
        """Median of the range estimates from the last 0.7 s (None when there are none)."""
        recent = [r for t, r in self.ranges if now - t <= 0.7]
        return statistics.median(recent) if recent else None

    def _range_rate(self, now: float) -> float | None:
        """Slope of the filtered range over the last ~1 s (m/s, negative = they come closer). None unless the
        history is recent and gap-free: a rate across a dropout is meaningless."""
        h = [x for x in self.range_hist if now - x[0] <= 1.0]
        if len(h) < 5 or h[-1][0] - h[0][0] < 0.5 or now - h[-1][0] > 0.3:
            return None
        if any(b[0] - a[0] > 0.4 for a, b in zip(h, h[1:])):
            return None
        ts = [t for t, _ in h]
        rs = [r for _, r in h]
        tm, rm = sum(ts) / len(ts), sum(rs) / len(rs)
        den = sum((t - tm) ** 2 for t in ts)
        return None if den <= 0 else sum((t - tm) * (r - rm) for t, r in zip(ts, rs)) / den

    def _climb_rc(self, ctx: Ctx) -> int:
        """Climb to get clear of a person, at most 0.25 m above the follow altitude (the governor also caps)."""
        alt = ctx.altitude
        return self.c.max_rc_up if (alt is None or alt < self.c.altitude_m + 0.25) else self._altitude_rc(ctx)

    def _altitude_rc(self, ctx: Ctx) -> int:
        alt = ctx.altitude
        if alt is None:
            return 0
        return int(clamp(deadband(self.c.altitude_m - alt, self.c.alt_deadband_m) * self.c.alt_gain, self.c.max_rc_up))

    def step(self, ctx: Ctx) -> str:
        c = self.c
        ud = self._altitude_rc(ctx)
        p = ctx.res.person if (ctx.res is not None and ctx.new_frame) else None
        if p is not None:
            self.last_seen_t, self.last_bearing = ctx.now, p.bearing_deg
            if p.range_m is not None:
                self.ranges.append((ctx.now, p.range_m))
            r = self._range(ctx.now)
            if r is not None:
                self.range_hist.append((ctx.now, r))
            r_lo = min(v for v in (p.range_lo_m, r) if v is not None) if (p.range_lo_m or r) else None
            yaw = clamp(deadband(p.bearing_deg, c.yaw_deadband_deg) * c.yaw_gain, c.max_rc_yaw)
            fwd, approaching = 0.0, False
            width, height = ctx.res.image_size
            too_big = p.det.w > self.max_box_width_frac * width
            if r is not None:
                e = deadband(r - c.distance_m, c.dist_deadband_m)
                if e > 0:  # too far: only approach if even the most conservative cue says we are not too close
                    if r_lo is not None and r_lo > c.min_range_m and not too_big:
                        fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                else:
                    fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                # feed-forward: they walk toward us -> back away at their speed now (rc 100 ~ 1 m/s), don't
                # wait for the distance error to build up (the head leaves the frame at ~0.6 m)
                # our own forward motion also shrinks the range (and video lags ~0.35 s): subtract the speed we
                # commanded (rc 100 ~ 1 m/s) so only THEIR motion toward us counts
                rate = self._range_rate(ctx.now)
                own = [f for t, f in self.fwd_hist if 0.3 <= ctx.now - t <= 1.3]
                if rate is not None and own:
                    closing = rate + (sum(own) / len(own)) / 100.0
                    approaching = closing < -c.approach_rate_mps
                    if approaching:
                        fwd = min(fwd, clamp(100.0 * closing - 10, c.max_rc_backoff))
            if (r_lo is not None and r_lo < c.min_range_m) or too_big:
                fwd = -c.max_rc_backoff  # too close: back off fast and gain a little height
                ud = max(ud, self._climb_rc(ctx) // 2)
            # coming closer, nearly too close, or the top of their head is at the bottom of the frame: if they
            # vanish now, they are walking under us (a plain dropout at the follow distance is not this)
            self.last_close = (approaching or (r is not None and r < c.min_range_m + 0.25)
                               or p.det.bbox[1] >= 0.88 * height)
            lat = 0.0
            if p.facing_deg is not None and p.facing_conf >= 0.5:
                self.facings.append(math.radians(p.facing_deg))
            if len(self.facings) >= 3:
                # circular mean of recent confident estimates: one mislabelled frame cannot swing the orbit
                f = math.degrees(math.atan2(sum(math.sin(a) for a in self.facings), sum(math.cos(a) for a in self.facings)))
                # facing +90 = they face image-right, so their back is to the image-left: move left
                lat = clamp(-c.orbit_gain * deadband(f, c.orbit_deadband_deg), c.max_rc_lateral)
            self.cmd = (int(lat), int(fwd), ud, int(yaw))
            self.fwd_hist.append((ctx.now, int(fwd)))
            self.status = (f"person {p.range_m or float('nan'):.1f} m (lo {r_lo or float('nan'):.1f}) "
                           f"bearing {p.bearing_deg:+.0f} facing {p.facing_deg if p.facing_deg is None else round(p.facing_deg)}")
            ctx.drone.rc(*self.cmd)
            return RUNNING
        unseen = math.inf if self.last_seen_t is None else ctx.now - self.last_seen_t
        if self.last_close and 0.2 < unseen < 2.5:
            # lost while close: they are walking under the camera's view. Back off, step out of their path
            # (away from the side they were on) and climb; never hover in their way
            side = -1 if self.last_bearing >= 0 else 1
            self.cmd = (side * c.max_rc_backoff, -c.max_rc_backoff, self._climb_rc(ctx), 0)
            ctx.drone.rc(*self.cmd)
            self.status = "person close and out of view: backing off to the side"
        elif unseen < 0.35:  # between frames: keep the last command
            ctx.drone.rc(*self.cmd[:2], ud, self.cmd[3])
        elif unseen < 1.0:
            ctx.drone.rc(0, 0, ud, 0)
            self.status = "person briefly lost: holding"
        elif unseen < 20.0 or self.last_seen_t is None:
            direction = 1 if self.last_bearing >= 0 else -1
            ctx.drone.rc(0, 0, ud, direction * c.search_yaw_rc)
            self.status = f"person lost {unseen:.0f} s: turning {'right' if direction > 0 else 'left'}"
        else:
            ctx.drone.rc(0, 0, ud, 0)
            self.status = "person lost: hovering"
        return RUNNING
