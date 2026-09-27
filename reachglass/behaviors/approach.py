"""Approach: close in on the confirmed target with discrete moves, re-measuring after every move.

Each cycle (after the drone has settled and a fresh frame shows the target):
  1. target sinking out of the bottom of the frame -> descend 20-30 cm (keeps a low object in view)
  2. |bearing| > align_deg                          -> rotate onto it (approach.steering: fly -> the fruit fly
                                                        controller turns onto it with continuous rc yaw; if it
                                                        stops short, one discrete rotate trims the rest)
  3. range > standoff + tolerance                   -> move forward min(range - standoff, max_step),
                                                        never beyond the clearance proven along the way
  4. otherwise                                      -> ARRIVED (SUCCESS); `final` holds the last observation
Lost for > 1.5 s: turn toward where memory says it is, then small alternating sweeps; give up after a few.
"""

from __future__ import annotations

import math

from ..config import ApproachCfg
from ..types import TargetObs
from .base import FAILURE, RUNNING, SUCCESS, Behavior, Ctx, Discrete, clamp
from .explore import FOOTPRINT, observe
from .fly_steer import FlyYaw

FLY_ALIGN_TIMEOUT_S = 5.0  # fly alignment taking longer than this: the discrete rotate finishes the turn


def observe_people(ctx: Ctx) -> None:
    pose = ctx.odom.pose
    for p in ctx.res.persons:
        if p.range_m is not None:
            x, y = pose.point_at(p.range_m, p.bearing_deg)
            ctx.memory.add("person", x, y, p.det.conf, p.range_m, ctx.now, ctx.vantage)


def recent_people(ctx: Ctx, max_age_s: float = 15.0) -> list[tuple[float, float]]:
    """Mission-frame positions of people: seen in this frame, remembered from the last seconds, and where the
    wearer stood when they asked (a blind user usually waits there)."""
    pts = []
    pose = ctx.odom.pose
    if ctx.res is not None:
        for p in ctx.res.persons:
            r = p.range_m if p.range_m is not None else p.range_lo_m  # too close to measure: the lower bound
            if r is not None:
                pts.append(pose.point_at(r, p.bearing_deg))
    for o in ctx.memory.of_class("person"):
        if ctx.now - o.last_t <= max_age_s:
            pts.append(o.xy)
    if ctx.person_origin is not None:
        pts.append(ctx.person_origin)
    return pts


def person_near_path(ctx: Ctx, step_m: float, clearance_m: float) -> float | None:
    """If the straight flight segment ahead (0..step_m) passes within clearance_m of a person (including the
    end point: a step that stops right in front of someone counts), which side they are on: +1 right,
    -1 left. None if the path is clear."""
    pose = ctx.odom.pose
    h = math.radians(pose.heading_deg)
    fx, fy = math.cos(h), math.sin(h)
    for px, py in recent_people(ctx):
        vx, vy = px - pose.x, py - pose.y
        along = vx * fx + vy * fy
        if along < -0.3:
            continue  # behind us
        a = max(0.0, min(along, step_m))
        d = math.hypot(vx - a * fx, vy - a * fy)  # closest distance between the person and the segment
        if d < clearance_m:
            return 1.0 if fx * vy - fy * vx > 0 else -1.0
    return None


def person_free_distance(ctx: Ctx, clearance_m: float) -> float:
    """How far straight ahead before coming within clearance_m of a person."""
    pose = ctx.odom.pose
    h = math.radians(pose.heading_deg)
    fx, fy = math.cos(h), math.sin(h)
    best = math.inf
    for px, py in recent_people(ctx):
        vx, vy = px - pose.x, py - pose.y
        along = vx * fx + vy * fy
        lateral = abs(fx * vy - fy * vx)
        if along > 0 and lateral < clearance_m:
            best = min(best, along - math.sqrt(max(clearance_m ** 2 - lateral ** 2, 0.0)))
    return max(0.0, best)


class Approach(Behavior):
    name = "approach"

    def __init__(self, cfg: ApproachCfg, settle_s: float = 0.7, min_altitude_m: float = 0.5,
                 person_clearance_m: float = 1.0):
        super().__init__()
        self.c = cfg
        self.settle_s = settle_s
        self.min_altitude_m = min_altitude_m
        self.person_clearance_m = person_clearance_m
        self.final: TargetObs | None = None

    def start(self, ctx: Ctx) -> None:
        ctx.perception.set_mode("approach")
        ctx.perception.set_target(ctx.target_cls)
        self.cmd: Discrete | None = None
        self.settle_until = ctx.now + self.settle_s
        self.lost_since: float | None = None
        self.search_tries = 0
        self.steps = 0
        self.final = None
        self.last_seen: TargetObs | None = None
        self.sidesteps = 0
        self.fly = FlyYaw.for_behavior(ctx, "approach", self.c.standoff_m)  # None: discrete rotates
        self.fly_align_t0: float | None = None
        self.fly_ok = 0
        self.fly_just_aligned = False  # the next alignment (if still needed) is a discrete rotate

    def _issue(self, ctx: Ctx, cmd: Discrete, why: str) -> str:
        self.cmd = cmd
        self.steps += 1
        self.status = why
        ctx.note(f"approach: {why}")
        return RUNNING

    def step(self, ctx: Ctx) -> str:
        c = self.c
        if self.fly is not None and self.fly_align_t0 is None:
            self.fly.yaw(ctx.now)  # keep the brain ticking (no target input) while discrete moves own the drone
        if self.cmd is not None:
            r = self.cmd.step(ctx)
            if r == RUNNING:
                return RUNNING
            failed = r == FAILURE
            self.cmd = None
            self.settle_until = ctx.now + self.settle_s
            if failed:
                self.status = "move refused/failed: re-measuring"
            return RUNNING
        if self.fly_align_t0 is not None:
            return self._fly_align(ctx)
        if self.steps > c.max_steps:
            self.status = "too many approach steps"
            return FAILURE
        ctx.drone.rc(0, 0, 0, 0)
        if ctx.new_frame and ctx.res is not None and ctx.res.persons and ctx.now >= self.settle_until:
            observe_people(ctx)  # keep track of people on every settled frame (the path check needs them)
        if ctx.now < self.settle_until or not ctx.new_frame or ctx.res is None or not ctx.frame_after_cmd():
            return RUNNING
        if not ctx.res.target_ran:
            return RUNNING  # the target detector skips frames: "no target" on this one means nothing
        t = ctx.res.target
        if t is None or not t.confirmed:
            return self._lost(ctx)
        self.lost_since = None
        self.search_tries = 0
        self.last_seen = t
        observe(ctx, mark_view=False)
        H = ctx.res.image_size[1]
        alt = ctx.altitude
        known = [v for v in (alt, ctx.clearance) if v is not None]
        below = min(known) if known else None  # clearance to whatever is under us
        if (t.det.bbox[3] > 0.92 * H and alt is not None and below is not None and below - 0.25 >= self.min_altitude_m
                and self._safe_to_descend(ctx, alt - 0.25)):
            return self._issue(ctx, Discrete("move", 25, "down"), "target low in the frame: descend 25 cm")
        if abs(t.bearing_deg) > c.align_deg:
            if self.fly is not None and not self.fly.failed and not self.fly_just_aligned:
                self.steps += 1
                self.fly_align_t0, self.fly_ok = ctx.now, 0
                ctx.note(f"approach: fly steering onto the target ({t.bearing_deg:+.0f} deg)")
                return self._fly_align(ctx)
            self.fly_just_aligned = False
            return self._issue(ctx, Discrete("rotate", t.bearing_deg), f"align {t.bearing_deg:+.0f} deg")
        self.fly_just_aligned = False
        r = t.range_m
        if r is None:
            # no size-based range (box cut by the frame edge): we are close; accept if it is big
            if t.det.h > 0.35 * H:
                return self._arrive(ctx, t)
            pose = ctx.odom.pose
            step = min(0.3, ctx.grid.clear_distance(pose, pose.heading_deg, 0.3, unknown_ok_m=0.3, altitude_m=alt),
                       person_free_distance(ctx, self.person_clearance_m))
            if person_near_path(ctx, 0.3, self.person_clearance_m) is not None or step < c.min_step_m:
                return self._arrive(ctx, t)  # close already and cannot safely creep closer
            return self._issue(ctx, Discrete("move", 30, "forward"), "range unknown: small step")
        if r > c.standoff_m + c.tolerance_m:
            step = min(r - c.standoff_m, c.max_step_m)
            pose = ctx.odom.pose
            clear = ctx.grid.clear_distance(pose, pose.heading_deg, step, unknown_ok_m=step, altitude_m=alt)
            step = min(step, clear)
            # keep a LOW target in view: the camera cannot tilt, so after the step the target must still be
            # above the bottom of the frame; descend first if needed, else shorten the step
            view_limited = False
            if alt is not None:
                need = self._altitude_for(ctx, t, r, r - step, alt)
                if need is not None and need < alt - 0.2:
                    down = min(alt - need, below - self.min_altitude_m if below is not None else 0.0)
                    if down >= 0.2 and self._safe_to_descend(ctx, alt - down):
                        cm = int(min(down, 1.0) * 100)
                        return self._issue(ctx, Discrete("move", cm, "down"), f"target low ahead: descend {cm} cm first")
                    in_view = self._max_step_in_view(ctx, t, r, alt)
                    if in_view < step:
                        step, view_limited = in_view, True
            if step < c.min_step_m:
                if view_limited or r < c.standoff_m + 0.6:
                    return self._arrive(ctx, t)  # as close as we can get while still seeing it
                # blocked (e.g. chairs in front of the table) and still far: first try to go around
                if self.sidesteps < 3:
                    left = ctx.grid.clear_distance(pose, pose.heading_deg - 90, 0.8, unknown_ok_m=0.5, altitude_m=alt)
                    right = ctx.grid.clear_distance(pose, pose.heading_deg + 90, 0.8, unknown_ok_m=0.5, altitude_m=alt)
                    side, room = ("left", left) if left >= right else ("right", right)
                    if room >= 0.5 and person_near_path(ctx, 0.0, self.person_clearance_m) is None:
                        self.sidesteps += 1
                        cm = int(min(room, 0.7) * 100)
                        return self._issue(ctx, Discrete("move", cm, side), f"path blocked: sidestep {side} {cm} cm")
                if r < c.standoff_m + 1.5:
                    return self._arrive(ctx, t)  # as close as the furniture allows
                return self._fail(ctx, "path blocked")
            # never fly past a person: the drone starts BEHIND the wearer, so they are often near the line
            side = person_near_path(ctx, step, self.person_clearance_m)
            if side is not None:
                if self.sidesteps >= 3:
                    return self._arrive(ctx, t) if r < c.standoff_m + 1.5 else self._fail(ctx, "a person is in the way")
                away = "left" if side > 0 else "right"
                heading = pose.heading_deg + (-90 if away == "left" else 90)
                if ctx.grid.clear_distance(pose, heading, 0.8, unknown_ok_m=0.8, altitude_m=alt) >= 0.6:
                    self.sidesteps += 1
                    return self._issue(ctx, Discrete("move", 70, away), f"person near the path: sidestep {away} 70 cm")
                step = min(step, person_free_distance(ctx, self.person_clearance_m))
                if step < c.min_step_m:
                    return self._fail(ctx, "a person is in the way")
            return self._issue(ctx, Discrete("move", step * 100, "forward"), f"target {r:.2f} m: forward {step:.2f} m")
        return self._arrive(ctx, t)

    def _fly_align(self, ctx: Ctx) -> str:
        """The fly turns the drone onto the target with rc yaw, until it is centred (within align_deg on 3
        frames), lost for 0.5 s, or FLY_ALIGN_TIMEOUT_S passes. Then hover, settle, re-measure.
        No range goes to the fly (as in follow): it sees the target at its standoff size (s = 1), where its
        steering is calibrated; a bottle several metres away would look smaller than anything it trained on."""
        t = ctx.res.target if (ctx.new_frame and ctx.res is not None) else None
        if t is not None and t.confirmed:
            self.last_seen = t
            yaw = self.fly.yaw(ctx.now, t.bearing_deg, None, ctx.frame.t, fallback=clamp(t.bearing_deg, 20))
            self.fly_ok = self.fly_ok + 1 if abs(t.bearing_deg) <= self.c.align_deg else 0
            self.status = f"fly steering: target {t.bearing_deg:+.0f} deg, yaw {yaw:+.0f}"
        else:
            yaw = self.fly.yaw(ctx.now, fallback=0.0)
        timeout = ctx.now - self.fly_align_t0 > FLY_ALIGN_TIMEOUT_S
        if self.fly_ok >= 3 or timeout or self.fly.failed or not self.fly.target_valid:
            ctx.drone.rc(0, 0, 0, 0)
            self.fly_align_t0 = None
            self.fly_just_aligned = True
            ctx.cmd_done_t = ctx.now  # like a finished rotate: only frames after the turn (+ video lag) count
            self.settle_until = ctx.now + self.settle_s
            why = "centred" if self.fly_ok >= 3 else "timeout" if timeout else "target lost" if not self.fly.failed else "failed"
            ctx.note(f"approach: fly steering done ({why})")
            return RUNNING
        ctx.drone.rc(0, 0, 0, int(yaw))
        return RUNNING

    # ------------------------------------------------------------------ keeping a low target in view
    def _target_base_height(self, ctx: Ctx, t: TargetObs, r: float, alt: float) -> float:
        """Height of the target's bottom above the floor, from its range and the elevation of its box bottom."""
        cam = ctx.perception.camera.for_frame(*ctx.res.image_size)
        el = cam.elevation_deg(t.det.bbox[3], 0.0, t.det.cx)
        return alt + r * math.tan(math.radians(el))

    def _allowed_depression(self, ctx: Ctx) -> float:
        cam = ctx.perception.camera.for_frame(*ctx.res.image_size)
        return cam.vfov_deg / 2.0 - 4.0 - cam.pitch_deg  # keep the bottom 4 deg away from the frame edge

    def _altitude_for(self, ctx: Ctx, t: TargetObs, r_now: float, r_next: float, alt: float) -> float | None:
        """Highest altitude at which the target's bottom stays in view from r_next, or None if no limit."""
        z_b = self._target_base_height(ctx, t, r_now, alt)
        dep = math.radians(self._allowed_depression(ctx))
        need = z_b + max(r_next, 0.3) * math.tan(dep)
        return need if need < alt else None

    def _max_step_in_view(self, ctx: Ctx, t: TargetObs, r: float, alt: float) -> float:
        z_b = self._target_base_height(ctx, t, r, alt)
        dep = math.radians(self._allowed_depression(ctx))
        closest = (alt - z_b) / math.tan(dep) if alt > z_b else 0.0
        return max(0.0, r - closest)

    def _safe_to_descend(self, ctx: Ctx, new_alt: float, margin_m: float = 0.3, near_m: float = 0.45) -> bool:
        """No known furniture under/next to the drone whose top would come within margin_m. The downward
        ToF only sees what is straight below, not a chair back 20 cm to the side."""
        pose = ctx.odom.pose
        heights = ctx.cfg.perception.object_heights_m
        for o in ctx.memory.objects:
            h = heights.get(o.cls)
            if h is None or o.cls in ("person", ctx.target_cls):
                continue
            if pose.distance_to(*o.xy) < FOOTPRINT.get(o.cls, 0.3) + near_m and new_alt < h + margin_m:
                self.status = f"not descending: {o.cls} below/next to us"
                return False
        g = ctx.grid
        c = g.idx(pose.x, pose.y)
        if c is not None:
            r = int(near_m / g.cell) + 1
            i0, j0 = c
            for i in range(max(0, i0 - r), min(g.n, i0 + r + 1)):
                for j in range(max(0, j0 - r), min(g.n, j0 + r + 1)):
                    if g.blocks((i, j), new_alt, clearance_m=margin_m):
                        self.status = "not descending: obstacle nearby would be too close below"
                        return False
        return True

    def _arrive(self, ctx: Ctx, t: TargetObs) -> str:
        self.final = t
        self.status = f"arrived: target {t.range_m if t.range_m is None else round(t.range_m, 2)} m ahead"
        return SUCCESS

    def _fail(self, ctx: Ctx, why: str) -> str:
        self.status = why
        return FAILURE

    def _lost(self, ctx: Ctx) -> str:
        if self.lost_since is None:
            self.lost_since = ctx.now
        if ctx.now - self.lost_since < 1.5:
            self.status = "target not in view: waiting"
            return RUNNING
        if self.search_tries >= 4:
            return self._fail(ctx, "target lost")
        self.search_tries += 1
        self.lost_since = None
        # last seen low in the frame: it most likely dropped out of the bottom -> go down before sweeping
        last = self.last_seen
        alt = ctx.altitude
        if (self.search_tries == 1 and last is not None and ctx.res is not None
                and last.det.bbox[3] > 0.75 * ctx.res.image_size[1] and alt is not None):
            below = min(v for v in (alt, ctx.clearance) if v is not None)  # alt is not None here
            down = min(0.5, below - self.min_altitude_m)
            if down >= 0.2 and self._safe_to_descend(ctx, alt - down):
                cm = int(down * 100)
                return self._issue(ctx, Discrete("move", cm, "down"), f"target lost low: descend {cm} cm")
        obj = ctx.memory.best(ctx.target_cls, confirmed_only=False)
        pose = ctx.odom.pose
        if obj is not None and self.search_tries == 1:
            b = pose.bearing_to(*obj.xy)
            if abs(b) >= 5:
                return self._issue(ctx, Discrete("rotate", b), f"target lost: turn {b:+.0f} deg toward memory")
        sweep = self.c.reacquire_scan_deg * (1 if self.search_tries % 2 else -2)
        return self._issue(ctx, Discrete("rotate", sweep), f"target lost: sweep {sweep:+d} deg")
