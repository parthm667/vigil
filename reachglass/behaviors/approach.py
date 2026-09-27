"""Approach: fly straight to where the MAP puts the target, and a little past it.

The search already located it (semantic memory, from the scans). No re-measuring, no re-searching:
  1. turn to the mapped position
  2. go to travel_altitude_m (1.5 m) above the floor; re-levelled before every later forward move and on
     arrival (moves are relative and the Tello drifts, e.g. rising over a table), so it travels and ends there
  3. fly forward distance + overshoot_m (in moves of at most 4 m): the drone ends just past the target
The target IS a person ("find arthur"): no climb, no fly-over; stop person_standoff_m short of them, and
their own position does not count as a person in the way.
A person ahead near that straight path (or where the wearer stood) -> do not go: hover here (SUCCESS).
A lost reply (timeout: the Tello most likely did it) or a refused height change (safety floor/ceiling) -> carry
on with the plan; a refused turn or forward move -> stop there (SUCCESS). No target in memory -> FAILURE.
"""

from __future__ import annotations

import math

from ..config import ApproachCfg
from ..types import TargetObs
from .base import FAILURE, RUNNING, SUCCESS, Behavior, Ctx, Discrete

MAX_MOVE_M = 4.0  # the Tello's move command takes at most 5 m
LEVEL = "level"  # plan step: move to travel_altitude_m, computed when it is reached


def observe_people(ctx: Ctx) -> None:
    pose = ctx.odom.pose
    for p in ctx.res.persons:
        if p.range_m is not None:
            x, y = pose.point_at(p.range_m, p.bearing_deg)
            ctx.memory.add("person", x, y, p.det.conf, p.range_m, ctx.now, ctx.vantage)


def recent_people(ctx: Ctx, max_age_s: float = 15.0,
                  exclude_xy: tuple[float, float] | None = None, exclude_r: float = 0.9) -> list[tuple[float, float]]:
    """Mission-frame positions of people: seen in this frame, remembered from the last seconds, and where the
    wearer stood when they asked (a blind user usually waits there). exclude_xy drops sightings near that
    point: when the target IS a person, they must not block the approach to themself."""
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
    if exclude_xy is not None:
        pts = [p for p in pts if math.hypot(p[0] - exclude_xy[0], p[1] - exclude_xy[1]) >= exclude_r]
    return pts


def person_near_path(ctx: Ctx, step_m: float, clearance_m: float,
                     exclude_xy: tuple[float, float] | None = None) -> float | None:
    """If the straight flight segment ahead (0..step_m) passes within clearance_m of a person (including the
    end point: a step that stops right in front of someone counts), which side they are on: +1 right,
    -1 left. None if the path is clear."""
    pose = ctx.odom.pose
    h = math.radians(pose.heading_deg)
    fx, fy = math.cos(h), math.sin(h)
    for px, py in recent_people(ctx, exclude_xy=exclude_xy):
        vx, vy = px - pose.x, py - pose.y
        along = vx * fx + vy * fy
        if along < -0.3:
            continue  # behind us
        a = max(0.0, min(along, step_m))
        d = math.hypot(vx - a * fx, vy - a * fy)  # closest distance between the person and the segment
        if d < clearance_m:
            return 1.0 if fx * vy - fy * vx > 0 else -1.0
    return None


def person_free_distance(ctx: Ctx, clearance_m: float,
                         exclude_xy: tuple[float, float] | None = None) -> float:
    """How far straight ahead before coming within clearance_m of a person."""
    pose = ctx.odom.pose
    h = math.radians(pose.heading_deg)
    fx, fy = math.cos(h), math.sin(h)
    best = math.inf
    for px, py in recent_people(ctx, exclude_xy=exclude_xy):
        vx, vy = px - pose.x, py - pose.y
        along = vx * fx + vy * fy
        lateral = abs(fx * vy - fy * vx)
        if along > 0 and lateral < clearance_m:
            best = min(best, along - math.sqrt(max(clearance_m ** 2 - lateral ** 2, 0.0)))
    return max(0.0, best)


class Approach(Behavior):
    name = "approach"

    def __init__(self, cfg: ApproachCfg, settle_s: float = 0.7, min_altitude_m: float = 0.5,
                 person_clearance_m: float = 1.0, person_target: bool = False):
        super().__init__()
        self.c = cfg
        self.person_clearance_m = person_clearance_m
        self.person_target = person_target  # also set in start() when perception says the target is a name
        self.final: TargetObs | None = None  # the confirming sighting (guidance reads it)
        self.last_top: float | None = None  # height of the target's top above the floor

    def start(self, ctx: Ctx) -> None:
        ctx.perception.set_mode("approach")
        ctx.perception.set_target(ctx.target_cls)
        # same test as the search: an enrolled person's name -> never fly over them
        self.person_target = self.person_target or bool(
            getattr(ctx.perception, "name_target", None) and ctx.perception.name_target())
        self.plan: list[tuple[Discrete, str]] | None = None
        self.cmd: Discrete | None = None
        t = ctx.res.target if ctx.res is not None else None
        self.final = t if (t is not None and t.confirmed) else None
        alt = ctx.altitude
        self.last_top = (self._top_height(ctx, t, t.range_m, alt)
                         if self.final is not None and t.range_m is not None and alt is not None else None)

    def step(self, ctx: Ctx) -> str:
        cls = ctx.target_cls
        if self.plan is None:
            obj = ctx.memory.best(cls, confirmed_only=False)
            if obj is None:
                self.status = f"no {cls} on the map"
                return FAILURE
            self.plan = self._plan(ctx, obj.xy)
            if not self.plan:
                return SUCCESS  # status says why it stays here
        if self.cmd is not None:
            r = self.cmd.step(ctx)
            if r == RUNNING:
                return RUNNING
            failed, self.cmd = self.cmd, None
            if r == FAILURE and failed.result != "error: timeout" and failed.direction not in ("up", "down"):
                self.status = f"stopped short of the {cls} ({failed.result})"
                return SUCCESS
        while self.plan:
            item = self.plan.pop(0)
            if item == LEVEL:
                item = self._level(ctx)  # computed now, from the altitude now
                if item is None:
                    continue  # already within 10 cm of travel_altitude_m
            self.cmd, self.status = item
            ctx.note(f"approach: {self.status}")
            return RUNNING
        self.status = f"{'next to' if self.person_target else 'over'} the {cls}"
        return SUCCESS

    def _level(self, ctx: Ctx) -> tuple[Discrete, str] | None:
        """A move to travel_altitude_m above the floor, or None when already within 10 cm of it. The Tello moves
        at least 20 cm: 10-20 cm off moves 20 cm, which still ends within 10 cm."""
        alt, want = ctx.altitude, self.c.travel_altitude_m
        if alt is None or abs(want - alt) < 0.1:
            return None
        up = want > alt
        cm = max(int(round(self.c.min_step_m * 100)), int(round(abs(want - alt) * 100)))
        return (Discrete("move", cm, "up" if up else "down"),
                f"{'climb' if up else 'descend'} {cm} cm to {want:.1f} m for the trip")

    def _plan(self, ctx: Ctx, xy: tuple[float, float]) -> list:
        """Level at travel_altitude_m, turn to the mapped target, fly past it (a person: stop short); re-level
        before each later forward move and at the end."""
        c, cls, pose = self.c, ctx.target_cls, ctx.odom.pose
        b = pose.bearing_to(*xy)
        if self.person_target:
            dist = pose.distance_to(*xy) - c.person_standoff_m
        else:
            dist = pose.distance_to(*xy) + c.overshoot_m
        h = math.radians(pose.heading_deg + b)
        end = (pose.x + max(dist, 0.0) * math.cos(h), pose.y + max(dist, 0.0) * math.sin(h))
        if self._person_near(ctx, (pose.x, pose.y), end, xy if self.person_target else None):
            self.status = f"a person is between us and the {cls}: staying here"
            return []
        plan: list = [LEVEL]
        if abs(b) >= 3:
            plan.append((Discrete("rotate", b), f"turn {b:+.0f} deg to the {cls} on the map"))
        if dist >= c.min_step_m:
            n = max(1, math.ceil(dist / MAX_MOVE_M))
            why = (f"fly to {c.person_standoff_m:.1f} m from {cls}" if self.person_target
                   else f"fly to the {cls} and {c.overshoot_m:.1f} m past it")
            for k in range(n):
                if k:
                    plan.append(LEVEL)
                plan.append((Discrete("move", int(round(dist / n * 100)), "forward"), f"{why}: forward {dist / n:.2f} m"))
        plan.append(LEVEL)  # and stay at travel_altitude_m once there
        return plan

    def _person_near(self, ctx: Ctx, a: tuple[float, float], b: tuple[float, float],
                     exclude_xy: tuple[float, float] | None) -> bool:
        """A person ahead within person_clearance_m of the straight path a -> b (exclude_xy: the target person)."""
        ax, ay = a
        dx, dy = b[0] - ax, b[1] - ay
        L2 = dx * dx + dy * dy or 1e-9
        for px, py in recent_people(ctx, exclude_xy=exclude_xy):
            u = ((px - ax) * dx + (py - ay) * dy) / L2
            if u <= 0:
                continue  # beside or behind the start (we begin ~1 m behind the wearer): not in the way
            u = min(1.0, u)
            if math.hypot(px - ax - u * dx, py - ay - u * dy) < self.person_clearance_m:
                return True
        return False

    def _top_height(self, ctx: Ctx, t: TargetObs, r: float, alt: float) -> float:
        """Height of the target's top above the floor, from its range and the elevation of its box top."""
        cam = ctx.perception.camera.for_frame(*ctx.res.image_size)
        return alt + r * math.tan(math.radians(cam.elevation_deg(t.det.bbox[1], 0.0, t.det.cx)))
