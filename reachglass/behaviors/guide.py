"""GUIDE, the last stage: the drone has reached the target; bring the wearer to it (the mission then lands).

1. LOWER  (guide.lower, off by default: the drone stays at flight height, above the wearer's head)
          If the drone flew over the target (it is now behind the drone), descend to guide.above_object_m over
          the object's top, never below guide.min_altitude_m and never climbing. Lower, the camera keeps the
          wearer in view until they arrive, and its blind zone near the drone shrinks.
2. LOOK   Turn toward where the wearer stood when they asked, then look_deg to the left and to the right.
          Each view maps furniture from this end of the room (its blind zone is the opposite of the search's
          first view, so together they cover the corridor) and measures the wearer again from here: wearer ->
          target becomes one direct measurement without the search's odometry drift, and seen from the front,
          which way they face. Nobody near where they stood: keep turning in look_deg steps.
3. PLAN   Walking path from the wearer to within reach of the target (mapping/walkpath.py). A target whose
          top is above raised_top_m stands on furniture the drone cannot see from above it: the walk stops
          raised_stop_m short of it, on the wearer's side.
4. GUIDE  Every frame: the wearer = the detected person nearest their track; position smoothed; heading =
          their walking direction once they moved, else the facing from pose keypoints. Aim lookahead_m along
          the path -> cue -1 (turn left, step that way) / 0 (forward) / +1 (right), emitted on every change and
          repeated at cue_hz. The drone turns to keep them centred. Re-plans when they stray off the path or a
          new obstacle lands on it. Cue 2 once they are within reach of the target -> SUCCESS.
          GLASSES (glasses.enabled, object targets). ONE camera guides at a time, and nothing gives up:
          the glasses camera is not touched until the drone measures the wearer within handoff_m of the
          target AND facing within handoff_facing_deg of it. Then its stream opens (and stays open) while the
          drone keeps guiding; once frames arrive, the glasses take over and the drone's camera is no longer
          used (perception idle). Cues come from the target's bearing in the glasses camera (the wearer's
          point of view), which corrects the map's error at the end; arrived = its distance within the same
          reach the drone would have used. Not seen by the glasses for glasses_lost_s (or no frames): back to
          the drone's camera, and the glasses are tried again after glasses_retry_s. No time limit.
Wearer not seen: the last cue is repeated, marked "not seen", and the drone turns toward where they were.
Close to the drone (under ~1 m) they fill the frame and cannot be measured: last measured within reach +
lost_close_m and then unmeasurable for lost_close_s also counts as arrived (the camera's limit).
"""

from __future__ import annotations

import math
import statistics
from typing import Callable

from ..config import GuideCfg
from ..glasses import GlassesTargetView
from ..mapping.walkpath import (WalkPath, cue_from_error, plan_walk, point_along, project_on_path, segment_max_cost,
                                steer, walk_costs)
from ..types import wrap_deg
from .base import FAILURE, RUNNING, SUCCESS, Behavior, Ctx, Discrete, rotation_to
from .explore import observe

ARRIVED_CUE = 2


def _heading_to(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _circular_mean(angles_deg: list[float]) -> float:
    return math.degrees(math.atan2(sum(math.sin(math.radians(a)) for a in angles_deg),
                                   sum(math.cos(math.radians(a)) for a in angles_deg)))


class Guide(Behavior):
    name = "guide"
    _live_view: GlassesTargetView | None = None  # at most one glasses stream (a guide left mid-way leaves it open)

    def __init__(self, cfg: GuideCfg, target_xy: tuple[float, float], target_top_m: float | None,
                 emit: Callable[[int, str], None]):
        super().__init__()
        self.c = cfg
        self.target = (float(target_xy[0]), float(target_xy[1]))
        self.top = target_top_m  # object's top above the floor (from the approach), None = never measured
        self.emit = emit  # emit(cue, detail): -1 / 0 / 1 / 2

    # ------------------------------------------------------------------ setup
    def start(self, ctx: Ctx) -> None:
        c = self.c
        ctx.perception.set_mode("guide")
        self.t0 = ctx.now
        self.cmd: Discrete | None = None
        self.settle_until = ctx.now
        # where the wearer should be: where they stood when they asked (else just ahead of the drone's start: it
        # followed them from behind), facing away from the drone that followed them unless it saw their facing
        self.expected = ctx.person_origin if ctx.person_origin is not None else (ctx.cfg.follow.distance_m, 0.0)
        self.heading = (ctx.person_heading if ctx.person_heading is not None
                        else math.degrees(math.atan2(self.expected[1], self.expected[0])))
        self.heading_src = "assumed"
        self.person_xy: tuple[float, float] | None = None
        self.person_t = -math.inf
        self.miss_since: float | None = None  # first fresh frame that missed the wearer (None = seen)
        self.lost_turned = False
        self.last_d_target: float | None = None
        self.hist: list[tuple[float, float, float]] = []  # (t, x, y) smoothed positions
        self.sightings: list[tuple[float, float, float | None, float]] = []  # look-back: (x, y, facing, t)
        self.views: list[float] = []
        self.extra_views: list[float] = []
        self.view_t0: float | None = None
        self.runs = {"person": 0, "context": 0}
        self.path: WalkPath | None = None
        self.costs = None
        self.seg = 0
        self.reach = c.arrive_m
        self.replan_t = -math.inf
        self.cue: int | None = None
        self.detail = ""
        self.emit_t = -math.inf
        self.arrive_hits = 0
        if Guide._live_view is not None:  # a guide left mid-way (land/hold) may have left its stream open
            Guide._live_view.close()
            Guide._live_view = None
        self.gv: GlassesTargetView | None = None  # opened only at the handoff, never before
        person = bool(getattr(ctx.perception, "name_target", None) and ctx.perception.name_target())
        self.can_glasses = (c.glasses_handoff and ctx.cfg.glasses.enabled and not person
                            and ctx.perception.detectors.get("target") is not None)
        self.glasses_mode = False
        self.g_arrive = 0
        self.g_seen_t = self.g_retry_at = -math.inf
        # like the search's scan: hover still before turning (the approach's last move just ended: momentum)
        self.hold_until = ctx.now + c.pre_turn_hover_s
        self.partial_turn = False  # a look-back turn split into max_turn_deg steps is under way
        self.phase = "lower"
        cmd = self._lower_cmd(ctx) if c.lower else None
        if cmd is not None:
            self._issue(ctx, cmd, f"lower to {(ctx.altitude or 0) - cmd.value / 100:.2f} m, just above the {ctx.target_cls}")
        else:
            self._begin_look(ctx)

    def _issue(self, ctx: Ctx, cmd: Discrete, why: str) -> None:
        self.cmd = cmd
        self.status = why
        ctx.note(f"guide: {why}")

    def _lower_cmd(self, ctx: Ctx) -> Discrete | None:
        c = self.c
        alt = ctx.altitude
        if alt is None or abs(ctx.odom.pose.bearing_to(*self.target)) <= 90.0:
            return None  # the target is ahead (no fly-over): stay above head height, the wearer walks toward us
        top = self.top if self.top is not None else c.unknown_top_m
        down = alt - max(top + c.above_object_m, c.min_altitude_m)
        clear = ctx.clearance  # to whatever is directly below (a table, the object itself)
        if clear is not None:
            down = min(down, clear - max(c.above_object_m, ctx.cfg.safety.min_altitude_m + 0.05))
        cm = int(round(min(down, 2.0) * 100))
        return Discrete("move", cm, "down") if cm >= 20 else None

    # ------------------------------------------------------------------ main
    def step(self, ctx: Ctx) -> str:
        r = self._step(ctx)
        if r != RUNNING and self.gv is not None:
            self.gv.close()  # the CAM serves one stream client: let it go
            self.gv = Guide._live_view = None
        return r

    def _step(self, ctx: Ctx) -> str:
        c = self.c
        if ctx.now - self.t0 > c.max_s:
            self.status = f"guiding took over {c.max_s:.0f} s"
            return FAILURE
        if ctx.now < self.hold_until:  # hovering still before the next turn step (a queued command waits)
            ctx.drone.rc(0, 0, 0, 0)
            self._repeat_cue(ctx)
            return RUNNING
        if self.cmd is not None:
            r = self.cmd.step(ctx)
            if r == RUNNING:
                if self.glasses_mode:  # a turn issued just before the handoff: the glasses go on meanwhile
                    return self._step_glasses(ctx)
                self._repeat_cue(ctx)
                return RUNNING
            if r == FAILURE:
                ctx.note(f"guide: {self.cmd.kind} {self.cmd.value:.0f} failed ({self.cmd.result})")
            self.cmd = None
            self.settle_until = ctx.now + c.settle_s
            if self.phase == "lower":
                self._begin_look(ctx)
            elif self.phase == "look":
                if self.partial_turn and r != FAILURE:
                    self.partial_turn = False
                    self.hold_until = ctx.now + c.turn_pause_s  # still hover, then the next step of the turn
                    self._next_view(ctx)
                else:
                    if self.partial_turn:  # a step failed: look from here (as a failed turn always did)
                        self.partial_turn = False
                        self.views.pop(0)
                    self._start_view(ctx)
            elif self.phase == "face":
                self._start_guiding(ctx)
            return RUNNING
        ctx.drone.rc(0, 0, 0, 0)  # keep-alive while hovering
        if self.phase == "look":
            return self._step_look(ctx)
        if self.phase == "guide":
            return self._step_guide(ctx)
        return RUNNING

    def _fresh(self, ctx: Ctx) -> bool:
        """A new frame that shows the world after the last command (settled, video lag passed)."""
        return ctx.new_frame and ctx.res is not None and ctx.now >= self.settle_until and ctx.frame_after_cmd()

    # ------------------------------------------------------------------ look-back
    def _begin_look(self, ctx: Ctx) -> None:
        self.phase = "look"
        pose = ctx.odom.pose
        h0 = _heading_to((pose.x, pose.y), self.expected)
        d = self.c.look_deg
        self.views = [h0, h0 - d, h0 + d]
        self.extra_views = [h0 + 2 * d, h0 + 3 * d, h0 + 4 * d, h0 - 3 * d, h0 - 2 * d]  # rest of the circle
        self._next_view(ctx)

    def _next_view(self, ctx: Ctx) -> None:
        h = self.views.pop(0)
        self.view_t0 = None
        rot = rotation_to(ctx, h)
        if abs(rot) > self.c.max_turn_deg:  # a big turn (the 180 back to the wearer) in steps, like the scan
            self.views.insert(0, h)  # the same heading again after this step
            self.partial_turn = True
            rot = math.copysign(self.c.max_turn_deg, rot)
        if abs(rot) >= 3:
            self._issue(ctx, Discrete("rotate", rot), f"look-back: turn {rot:+.0f} deg")
        else:
            self._start_view(ctx)

    def _start_view(self, ctx: Ctx) -> None:
        self.view_t0 = ctx.now
        self.runs = {"person": 0, "context": 0}
        self.status = f"look-back: mapping and looking for you ({len(self.views)} view(s) left)"

    def _step_look(self, ctx: Ctx) -> str:
        c = self.c
        if self._fresh(ctx):
            observe(ctx, exclude=[(self.expected[0], self.expected[1], c.wearer_exclude_m)])
            pose = ctx.odom.pose
            if ctx.res.ran.get("person"):
                self.runs["person"] += 1
                for p in ctx.res.persons:
                    r = self._range(ctx, p)
                    if r is None:
                        continue
                    x, y = pose.point_at(r, p.bearing_deg)
                    f = (wrap_deg(pose.heading_deg + p.bearing_deg + p.facing_deg)
                         if p.facing_deg is not None and p.facing_conf >= c.facing_conf else None)
                    self.sightings.append((x, y, f, ctx.now))
            if ctx.res.ran.get("context"):
                self.runs["context"] += 1
        enough = (self.runs["person"] >= c.view_person_runs
                  and (self.runs["context"] >= c.view_context_runs or "context" not in ctx.perception.active_roles()))
        timeout = ctx.now - self.view_t0 > c.settle_s + ctx.cfg.drone.video_lag_s + c.view_timeout_s
        if not (enough or timeout):
            return RUNNING
        if self.views:
            self._next_view(ctx)
            return RUNNING
        return self._acquire(ctx)

    def _acquire(self, ctx: Ctx) -> str:
        """The wearer = the sightings nearest to where they stood."""
        c = self.c
        near = [s for s in self.sightings if math.dist(s[:2], self.expected) <= c.acquire_gate_m]
        if not near and self.extra_views:
            self.views = [self.extra_views.pop(0)]
            self._next_view(ctx)
            return RUNNING
        near = near or self.sightings  # all the way round and nobody near where they stood: whoever was seen
        if not near:
            self.status = "could not find the wearer"
            return FAILURE
        best = min(near, key=lambda s: math.dist(s[:2], self.expected))
        cluster = [s for s in near if math.dist(s[:2], best[:2]) <= 0.8]
        self.person_xy = (statistics.median(s[0] for s in cluster), statistics.median(s[1] for s in cluster))
        self.person_t = max(s[3] for s in cluster)
        self.hist = [(self.person_t, *self.person_xy)]
        facings = [s[2] for s in cluster if s[2] is not None]
        if facings:
            self.heading, self.heading_src = _circular_mean(facings), "pose"
        ctx.note(f"guide: wearer at ({self.person_xy[0]:.2f}, {self.person_xy[1]:.2f}), "
                 f"{math.dist(self.person_xy, self.target):.1f} m from the {ctx.target_cls}, "
                 f"heading {self.heading:.0f} deg ({self.heading_src}), {len(cluster)} sightings")
        self.phase = "face"
        b = ctx.odom.pose.bearing_to(*self.person_xy)
        if abs(b) >= 5:
            self._issue(ctx, Discrete("rotate", b), f"face the wearer: turn {b:+.0f} deg")
        else:
            self._start_guiding(ctx)
        return RUNNING

    # ------------------------------------------------------------------ guiding
    def _start_guiding(self, ctx: Ctx) -> None:
        self.phase = "guide"
        self._plan(ctx, "start")

    def _walk_costs(self, ctx: Ctx):
        c = self.c
        top = self.top if self.top is not None else c.unknown_top_m
        # a raised target stands on furniture the drone cannot see while over it: stop raised_stop_m short
        keep = [(*self.target, c.raised_stop_m - c.inflate_m)] if top >= c.raised_top_m else []
        return walk_costs(ctx.grid, c.inflate_m, c.unknown_cost, [ctx.odom.path], keep)

    def _plan(self, ctx: Ctx, why: str) -> None:
        c = self.c
        self.costs = self._walk_costs(ctx)
        p = plan_walk(ctx.grid, self.costs, self.person_xy, self.target, c.unknown_cost, c.goal_slack_m)
        if p is None:  # off the map: straight at it
            p = WalkPath([self.person_xy, self.target], self.target, self.target, c.unknown_cost)
        self.path, self.seg, self.replan_t = p, 0, ctx.now
        # arrived: within arrive_m of the target, or (it cannot be reached, e.g. on a table) at the path's end
        self.reach = max(c.arrive_m, math.dist(p.goal, self.target) + c.reach_slack_m)
        legs = " -> ".join(f"({x:.1f}, {y:.1f})" for x, y in p.points)
        ctx.note(f"guide: path ({why}): {p.length:.1f} m in {len(p.points) - 1} leg(s): {legs}")

    def _step_guide(self, ctx: Ctx) -> str:
        c = self.c
        if self.glasses_mode:
            return self._step_glasses(ctx)
        if self._fresh(ctx):
            observe(ctx, mark_view=False, exclude=[(self.person_xy[0], self.person_xy[1], c.wearer_exclude_m)])
            if ctx.res.ran.get("person"):
                m = self._measure(ctx)
                if m is not None:
                    self._update_track(ctx, *m)
                    if self._arrived(ctx):
                        return SUCCESS
                    self._replan_if_needed(ctx, bool(ctx.res.ran.get("context")))
                    err, self.seg, _ = steer(self.path.points, self.person_xy, self.heading, self.seg, c.lookahead_m,
                                             final_xy=self.target)
                    if (self.can_glasses and self.last_d_target <= c.handoff_m
                            and abs(wrap_deg(self.heading - _heading_to(self.person_xy, self.target)))
                            <= c.handoff_facing_deg):
                        if self.gv is None:  # first time in range and facing it: open the stream, keep it open
                            self.gv = Guide._live_view = GlassesTargetView(
                                ctx.cfg.glasses, ctx.perception.detectors["target"], ctx.target_cls,
                                ctx.cfg.perception.object_heights_m.get(ctx.target_cls), c.glasses_stride)
                            ctx.note("guide: in range and facing it: opening the glasses camera (the drone guides meanwhile)")
                            print(f"[guide] {self.last_d_target:.1f} m out and facing the target: opening the "
                                  f"ESP32 glasses camera (drone keeps guiding until frames arrive)", flush=True)
                        elif ctx.now >= self.g_retry_at and self.gv.streaming():
                            self._to_glasses(ctx, f"{self.last_d_target:.1f} m away and facing it")
                            return RUNNING
                    cue = cue_from_error(err, self.cue, c.forward_deg, c.forward_exit_deg)
                    self._send(ctx, cue, f"{self.last_d_target:.1f} m to go, turn {err:+.0f} deg "
                                         f"(heading from {self.heading_src})")
                    b = ctx.odom.pose.bearing_to(*self.person_xy)
                    if abs(b) > c.center_deg:
                        self._issue(ctx, Discrete("rotate", b), f"keep the wearer in view: turn {b:+.0f} deg")
                    return RUNNING
        unseen = 0.0 if self.miss_since is None else ctx.now - self.miss_since
        if (unseen > c.lost_close_s and self.last_d_target is not None
                and self.last_d_target <= self.reach + c.lost_close_m):
            self._send(ctx, ARRIVED_CUE, f"arrived: out of view {self.last_d_target:.2f} m from the {ctx.target_cls}",
                       force=True)
            return SUCCESS  # too close to the drone to measure (they fill the frame), still walking in
        if unseen > c.lost_timeout_s:
            self.status = f"lost the wearer for {unseen:.0f} s"
            return FAILURE
        if unseen > c.lost_turn_s and not self.lost_turned:
            self.lost_turned = True
            b = ctx.odom.pose.bearing_to(*self.person_xy)
            if abs(b) > 10:
                self._issue(ctx, Discrete("rotate", b), f"wearer not seen: turn {b:+.0f} deg to where they were")
        self._repeat_cue(ctx)
        return RUNNING

    def _to_glasses(self, ctx: Ctx, why: str) -> None:
        """Hand the guiding to the glasses camera (its stream is already delivering frames)."""
        ctx.perception.set_mode("idle")  # no detector runs on the drone's frames while the glasses guide
        self.glasses_mode, self.g_seen_t, self.g_arrive = True, ctx.now, 0  # g_seen_t: time to find it
        ctx.note(f"guide: switching to the glasses camera ({why}); the drone's camera is no longer used")
        print(f"[guide] SWITCHED to the ESP32 glasses camera ({why}); drone camera idle", flush=True)

    def _to_drone(self, ctx: Ctx, why: str) -> None:
        """Back to the drone's camera; the glasses are tried again after glasses_retry_s (never given up)."""
        self.glasses_mode, self.g_retry_at = False, ctx.now + self.c.glasses_retry_s
        self.miss_since = None  # the drone was not looking meanwhile: not "unseen" for all that time
        ctx.perception.set_mode("guide")
        ctx.note(f"guide: {why}: back to the drone's camera (the glasses will be tried again)")
        print(f"[guide] back to the DRONE camera ({why}); glasses retried in {self.c.glasses_retry_s:.0f} s", flush=True)

    def _step_glasses(self, ctx: Ctx) -> str:
        """After the handoff, the only camera: cue from the target's bearing in the glasses camera; arrived once
        its distance is within the reach the drone would have used (arrive_frames in a row)."""
        c, cls = self.c, ctx.target_cls
        o = self.gv.poll()
        if o is not None:
            seen, bearing, rng = o
            if seen:
                self.g_seen_t = ctx.now
                if rng is not None and rng <= self.reach:
                    self.g_arrive += 1
                    if self.g_arrive >= c.arrive_frames:
                        self._send(ctx, ARRIVED_CUE, f"arrived (glasses camera): {rng:.2f} m from the {cls}", force=True)
                        return SUCCESS
                else:
                    self.g_arrive = 0
                cue = cue_from_error(bearing, self.cue, c.glasses_forward_deg, c.glasses_forward_exit_deg)
                self._send(ctx, cue, f"glasses camera: {cls} {bearing:+.0f} deg" + (f", {rng:.1f} m" if rng else ""))
                return RUNNING
        if ctx.now - self.g_seen_t > c.glasses_lost_s:
            self._to_drone(ctx, f"the glasses camera has not seen the {cls} for {c.glasses_lost_s:.0f} s")
            return RUNNING
        self._repeat_cue(ctx)  # between detector runs / while it is out of view: the last cue keeps going
        return RUNNING

    def _measure(self, ctx: Ctx) -> tuple[tuple[float, float], object] | None:
        """This frame's position of the wearer: the person nearest their track, within the gate."""
        c = self.c
        pose = ctx.odom.pose
        gate = c.track_gate_m + c.max_speed_mps * min(ctx.now - self.person_t, 3.0)
        best = None
        for p in ctx.res.persons:
            r = self._range(ctx, p)
            if r is None:
                continue
            xy = pose.point_at(r, p.bearing_deg)
            d = math.dist(xy, self.person_xy)
            if d <= gate and (best is None or d < best[0]):
                best = (d, xy, p)
        if best is None:
            if self.miss_since is None:
                self.miss_since = ctx.now
            return None
        self.miss_since = None
        self.lost_turned = False
        return best[1], best[2]

    @staticmethod
    def _range(ctx: Ctx, p) -> float | None:
        """The estimator's range. Close up (head above and feet below the frame) it has none: then the shoulder
        width from the keypoints, else the box width (if not cut at the sides), assuming a front view (they walk
        toward the drone; turned away they read farther, i.e. arrive late rather than early). Its conservative
        lower bound only as the last resort."""
        if p.range_m is not None:
            return p.range_m
        cam = ctx.perception.camera.for_frame(*ctx.res.image_size)
        pc, det = ctx.cfg.perception, p.det
        kp = det.keypoints
        if kp is not None and len(kp) > 6 and kp[5][2] >= 0.4 and kp[6][2] >= 0.4 and abs(kp[6][0] - kp[5][0]) >= 10:
            r = cam.range_from_width(abs(float(kp[6][0] - kp[5][0])), pc.shoulder_width_m, 0.5 * float(kp[5][0] + kp[6][0]))
            if r is not None:
                return r
        if det.bbox[0] > 3 and det.bbox[2] < cam.width - 3:
            r = cam.range_from_width(det.w, pc.body_width_m, det.cx)
            if r is not None:
                return r
        return p.range_lo_m

    def _update_track(self, ctx: Ctx, xy: tuple[float, float], p) -> None:
        c = self.c
        a = c.smooth
        x = self.person_xy[0] + a * (xy[0] - self.person_xy[0])
        y = self.person_xy[1] + a * (xy[1] - self.person_xy[1])
        self.person_xy, self.person_t = (x, y), ctx.now
        # distance for the arrival decision: this frame's measurement (smoothing lags ~0.3 m behind someone
        # walking in; two consecutive frames must agree, see _arrived)
        self.last_d_target = math.dist(xy, self.target)
        self.hist = [h for h in self.hist if ctx.now - h[0] <= 4.0] + [(ctx.now, x, y)]
        # heading: where they walked over the last motion_window_s, else where their body faces
        old = next((h for h in reversed(self.hist) if ctx.now - h[0] >= c.motion_window_s), None)
        if old is not None and math.hypot(x - old[1], y - old[2]) >= c.min_move_m:
            self.heading, self.heading_src = math.degrees(math.atan2(y - old[2], x - old[1])), "walking"
        elif p.facing_deg is not None and p.facing_conf >= c.facing_conf:
            pose = ctx.odom.pose
            self.heading, self.heading_src = wrap_deg(pose.heading_deg + p.bearing_deg + p.facing_deg), "pose"

    def _arrived(self, ctx: Ctx) -> bool:
        c = self.c
        d_t = self.last_d_target
        if d_t <= self.reach:
            self.arrive_hits += 1
        else:
            self.arrive_hits = 0
        if self.arrive_hits < c.arrive_frames:
            return False
        self._send(ctx, ARRIVED_CUE, f"arrived: {d_t:.2f} m from the {ctx.target_cls}", force=True)
        return True

    def _replan_if_needed(self, ctx: Ctx, grid_changed: bool) -> None:
        c = self.c
        if ctx.now - self.replan_t < 1.5:
            return
        pts = self.path.points
        k, s, off = project_on_path(pts, self.person_xy, self.seg)
        why = None
        if off > c.replan_m:
            why = f"{off:.1f} m off the path"
        elif grid_changed:
            self.costs = self._walk_costs(ctx)
            rest = [point_along(pts, s)] + pts[k + 1:]
            worst = max((segment_max_cost(ctx.grid, self.costs, a, b) for a, b in zip(rest, rest[1:])), default=0.0)
            if worst > max(self.path.max_cost, c.unknown_cost) + 1e-6:
                why = "new obstacle on the path"
        if why:
            self._plan(ctx, why)

    def _send(self, ctx: Ctx, cue: int, detail: str, force: bool = False) -> None:
        if force or cue != self.cue or ctx.now - self.emit_t >= 1.0 / self.c.cue_hz:
            self.emit(cue, detail)
            self.emit_t = ctx.now
        self.cue, self.detail = cue, detail
        self.status = f"cue {cue}: {detail}"

    def _repeat_cue(self, ctx: Ctx) -> None:
        """Between measurements (turning, or the wearer not seen) the last cue keeps going out at cue_hz."""
        if self.cue is None or self.cue == ARRIVED_CUE or ctx.now - self.emit_t < 1.0 / self.c.cue_hz:
            return
        self.emit(self.cue, self.detail + (" (not seen)" if self.miss_since is not None else ""))
        self.emit_t = ctx.now
