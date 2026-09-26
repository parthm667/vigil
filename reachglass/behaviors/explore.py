"""Exploration: SCAN at a vantage point, SCORE directions, HOP, repeat until the target is confirmed.

observe(ctx)      one fresh perception result -> semantic memory + grid (coverage, free, blocked)
Scan              rotate in scan_step_deg increments through 360 deg; after each rotation wait dwell_s
                  (video lag + settling), then observe frames_per_dwell frames. Stops early on a confirmed target.
choose_hop(ctx)   score every 15 deg:  score = open x (0.3 + novelty) x semantic x people x not_revisited
                    open      free distance proven along the direction (grid), / hop_max
                    novelty   fraction of never-seen cells in that direction (within view_range)
                    semantic  1 + 2 x sum of prior weights of objects that way (bottles live on tables...)
                              + 4 if an unconfirmed target candidate lies that way
                    people    0 toward a person closer than person_clearance
                    revisit   0.2 if the hop would end within 0.8 m of a previous vantage point
Hop               rotate to the chosen heading, move forward
Explore           scan -> (target confirmed? done) -> choose -> hop -> scan ... within the budgets
"""

from __future__ import annotations

import math

from ..config import ExploreCfg
from ..types import Pose2D
from .base import FAILURE, RUNNING, SUCCESS, Behavior, Ctx, Discrete, angdiff, rotation_to

# footprint radius (m) marked as blocked around detected objects
FOOTPRINT = {"dining table": 0.6, "couch": 0.8, "bed": 0.9, "chair": 0.3, "refrigerator": 0.4, "tv": 0.3,
             "potted plant": 0.3, "bench": 0.5, "toilet": 0.3, "sink": 0.3, "oven": 0.3, "suitcase": 0.3,
             "person": 0.5}


def observe(ctx: Ctx, mark_view: bool = True) -> None:
    """Put one fresh perception result into memory and the grid."""
    res = ctx.res
    if res is None:
        return
    pose = ctx.odom.pose
    g = ctx.grid
    for t in res.targets:
        if t.range_m is None:
            continue
        x, y = pose.point_at(t.range_m, t.bearing_deg)
        ctx.memory.add(t.cls, x, y, t.det.conf, t.range_m, ctx.now, ctx.vantage, confirmed=t.confirmed)
        g.mark_free_ray(pose.x, pose.y, pose.heading_deg + t.bearing_deg, t.range_m - 0.3)
    cam = ctx.perception.camera.for_frame(*res.image_size)
    alt = ctx.altitude
    for o in res.context:
        if o.range_m is None:
            continue
        x, y = pose.point_at(o.range_m, o.bearing_deg)
        ctx.memory.add(o.cls, x, y, o.det.conf, o.range_m, ctx.now, ctx.vantage)
        # how high does it reach? from the elevation of its box's top edge (a size prior would miss a TV on a
        # stand); a box cut by the top of the image, or no altitude: treat as reaching flight level
        top = math.inf
        if alt is not None and o.det.bbox[1] > 3:
            el = cam.elevation_deg(o.det.bbox[1], 0.0, o.det.cx)
            top = max(0.0, alt + o.range_m * math.tan(math.radians(el)))
        g.mark_blocked(x, y, FOOTPRINT.get(o.cls, 0.3), top_m=top)
        g.mark_free_ray(pose.x, pose.y, pose.heading_deg + o.bearing_deg, o.range_m - FOOTPRINT.get(o.cls, 0.3) - 0.2)
    for p in res.persons:
        if p.range_m is None:
            continue
        x, y = pose.point_at(p.range_m, p.bearing_deg)
        ctx.memory.add("person", x, y, p.det.conf, p.range_m, ctx.now, ctx.vantage)
        g.mark_free_ray(pose.x, pose.y, pose.heading_deg + p.bearing_deg, p.range_m - 0.7)
    if mark_view and ctx.frame is not None:
        cam = ctx.perception.camera.for_frame(ctx.frame.width, ctx.frame.height)
        free = None
        if ctx.freespace is not None:
            fs = ctx.freespace.estimate(ctx.frame.image, cam, ctx.altitude)
            if fs.metric:
                free = fs.as_free_list()
        g.mark_view(pose, cam.hfov_deg, ctx.cfg.explore.view_range_m, free)


class Scan(Behavior):
    name = "scan"

    def __init__(self, cfg: ExploreCfg, stop_on_target: bool = True):
        super().__init__()
        self.c = cfg
        self.stop_on_target = stop_on_target
        self.found = False

    def start(self, ctx: Ctx) -> None:
        ctx.perception.set_mode("search")
        self.n_steps = max(1, int(round(360 / self.c.scan_step_deg)))
        self.steps_done = 0
        self.phase = "dwell"
        self.dwell_start = ctx.now
        self.frames = 0
        self.cmd: Discrete | None = None
        self.found = False
        ctx.vantages.append((ctx.odom.pose.x, ctx.odom.pose.y))

    def step(self, ctx: Ctx) -> str:
        if self.phase == "rotate":
            r = self.cmd.step(ctx)
            if r == RUNNING:
                return RUNNING
            if r == FAILURE:
                self.status = f"rotation failed: {self.cmd.result}"
                return FAILURE
            self.phase, self.dwell_start, self.frames = "dwell", ctx.now, 0
            return RUNNING
        ctx.drone.rc(0, 0, 0, 0)  # keep-alive while looking
        if ctx.now - self.dwell_start < self.c.dwell_s:
            return RUNNING
        if ctx.new_frame and ctx.frame_after_cmd():  # a frame that really shows the new heading
            observe(ctx)
            self.frames += 1
            t = ctx.res.target if ctx.res else None
            if self.stop_on_target and t is not None and t.confirmed:
                self.found = True
                self.status = f"target confirmed at {t.range_m or float('nan'):.1f} m, bearing {t.bearing_deg:+.0f}"
                return SUCCESS
        if self.frames >= self.c.frames_per_dwell:
            self.steps_done += 1
            if self.steps_done >= self.n_steps:
                self.status = "scan complete, target not confirmed"
                return SUCCESS
            self.phase = "rotate"
            self.cmd = Discrete("rotate", self.c.scan_step_deg)
            self.status = f"scan {self.steps_done}/{self.n_steps}"
        return RUNNING


def choose_hop(ctx: Ctx) -> tuple[float, float, dict] | None:
    """Best (world heading, distance) to hop to next, or None if nothing is worth it."""
    c = ctx.cfg.explore
    pose = ctx.odom.pose
    weights = c.semantic_weights.get(ctx.target_cls or "", {})
    objs = ctx.memory.objects
    best = None
    for k in range(24):
        h = k * 15.0
        dist = ctx.grid.clear_distance(pose, h, c.hop_max_m, unknown_ok_m=c.hop_min_m, altitude_m=ctx.altitude)
        if dist < c.hop_min_m:
            continue
        open_ = dist / c.hop_max_m
        hr = math.radians(h)
        end = (pose.x + dist * math.cos(hr), pose.y + dist * math.sin(hr))
        # what would we see from there that we have not seen yet? (the scan just covered everything within
        # view range of HERE, so novelty is measured beyond the hop's end point)
        novelty = ctx.grid.novelty(Pose2D(end[0], end[1], h), h, c.view_range_m)
        sem, cand, people = 1.0, 0.0, 1.0
        for o in objs:
            ox, oy = o.xy
            d = pose.distance_to(ox, oy)
            if d < 0.3:
                continue
            ang = angdiff(math.degrees(math.atan2(oy - pose.y, ox - pose.x)), h)
            if o.cls == "person" and ctx.now - o.last_t < 10.0 and d < c.person_clearance_m + dist and ang < 30:
                people = 0.0  # (people move: only recent sightings count)
            if ang < 20 and d < 6.0:
                sem += 2.0 * weights.get(o.cls, 0.0)
            lock_confirmed = any(s.confirmed for s in o.sightings)
            if o.cls == ctx.target_cls and not lock_confirmed and ang < 15:
                cand = 4.0  # a weak sighting of the target that way: go and have a closer look
        previous = ctx.vantages[:-1]  # not the vantage we are at (the novelty term covers redundancy)
        revisit = 0.2 if any(math.hypot(end[0] - vx, end[1] - vy) < 0.8 for vx, vy in previous) else 1.0
        score = open_ * (0.3 + novelty) * (sem + cand) * people * revisit
        info = {"heading": h, "dist": round(dist, 2), "open": round(open_, 2), "novelty": round(novelty, 2),
                "semantic": round(sem, 2), "candidate": cand, "people": people, "revisit": revisit, "score": round(score, 3)}
        if best is None or score > best[2]["score"]:
            best = (h, dist, info)
    if best is None or best[2]["score"] < 0.05:
        return None
    return best


class Hop(Behavior):
    name = "hop"

    def __init__(self, heading_deg: float, dist_m: float):
        super().__init__()
        self.heading, self.dist = heading_deg, dist_m

    def start(self, ctx: Ctx) -> None:
        rot = rotation_to(ctx, self.heading)
        self.cmds = []
        if abs(rot) >= 5:
            self.cmds.append(Discrete("rotate", rot))
        cm = int(round(self.dist * 100))
        if cm >= 20:
            self.cmds.append(Discrete("move", cm, "forward"))
        self.status = f"hop {self.dist:.1f} m toward {self.heading:.0f} deg"

    def step(self, ctx: Ctx) -> str:
        while self.cmds:
            r = self.cmds[0].step(ctx)
            if r == RUNNING:
                return RUNNING
            if r == FAILURE:
                self.status = f"hop failed: {self.cmds[0].result}"
                return FAILURE
            self.cmds.pop(0)
        return SUCCESS


class Explore(Behavior):
    """Scan -> hop -> scan ... until the target is confirmed (SUCCESS) or a budget runs out (FAILURE)."""

    name = "explore"

    def __init__(self, cfg: ExploreCfg):
        super().__init__()
        self.c = cfg
        self.child: Behavior | None = None
        self.hops = 0
        self.decisions: list[dict] = []

    def start(self, ctx: Ctx) -> None:
        self.t0 = ctx.now
        self.hops = 0
        self._start_child(ctx, Scan(self.c))

    def _start_child(self, ctx: Ctx, b: Behavior) -> None:
        self.child = b
        b.start(ctx)

    def step(self, ctx: Ctx) -> str:
        if ctx.now - self.t0 > self.c.max_search_s:
            self.status = f"search time over ({self.c.max_search_s:.0f} s)"
            return FAILURE
        r = self.child.step(ctx)
        self.status = f"{self.child.name}: {self.child.status}"
        if r == RUNNING:
            return RUNNING
        if isinstance(self.child, Scan):
            if r == SUCCESS and self.child.found:
                return SUCCESS
            if self.hops >= self.c.max_vantage_points - 1:
                self.status = f"searched {self.hops + 1} vantage points"
                return FAILURE
            choice = choose_hop(ctx)
            if choice is None:
                self.status = "no useful direction left"
                return FAILURE
            h, d, info = choice
            self.decisions.append(info)
            ctx.note(f"hop decision {info}")
            self._start_child(ctx, Hop(h, d))
            return RUNNING
        # a hop finished (or failed: try scanning from wherever we are)
        self.hops += 1
        ctx.vantage += 1
        self._start_child(ctx, Scan(self.c))
        return RUNNING
