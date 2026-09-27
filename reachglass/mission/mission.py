"""Mission state machine.

    TAKEOFF -> CLIMB -> FOLLOW --(query "find X")--> EXPLORE -> APPROACH -> ARRIVED (hovering just past the object)
    (the search starts at the follow altitude: no descent)
                          ^                                      |            |          |
                          +------------- REACQUIRE <--------------+-(failed)---+---(query "follow me")
    any state: "land" -> LAND -> LANDED;  "stop"/"cancel" -> REACQUIRE (back to following)

When the query arrives the mission frame is reset at the drone (x = its heading), and the person's
position and facing are recorded: that is what the guidance (turn / distance for the person) is
computed against once the target is reached. announce() is the audio hook (printed for now).
"""

from __future__ import annotations

import logging
import math
from typing import Callable

from ..behaviors import FAILURE, RUNNING, SUCCESS, Approach, Behavior, Ctx, Discrete, Explore, FollowBehind, ReacquirePerson
from ..mapping import Grid, SemanticMemory
from ..query import ParsedQuery, QueryParser
from ..types import wrap_deg
from .guidance import Guidance, compute_guidance

log = logging.getLogger("reachglass.mission")

STATES = ("IDLE", "TAKEOFF", "CLIMB", "FOLLOW", "EXPLORE", "APPROACH", "ARRIVED", "REACQUIRE", "HOLD",
          "LAND", "LANDED")


class Mission:
    def __init__(self, ctx: Ctx, parser: QueryParser, announce: Callable[[str], None] | None = None):
        self.ctx = ctx
        self.parser = parser
        self.announce_fn = announce or (lambda s: print(f"[say] {s}", flush=True))
        self.state = "IDLE"
        self.state_t = 0.0
        self.child: Behavior | None = None
        self.cmd: Discrete | None = None
        self.guidance: Guidance | None = None
        self.last_query: ParsedQuery | None = None
        self.history: list[tuple[float, str, str]] = []  # (t, state, why)
        self.said: list[tuple[float, str]] = []
        self._pending: list[str] = []
        self._deferred: str | None = None  # a 'find' heard while taking off, handled once following
        self._approach_retries = 0
        self._resume_state: str | None = None
        self._land_sent = -1e9

    # ------------------------------------------------------------------ helpers
    def announce(self, text: str) -> None:
        self.said.append((self.ctx.now, text))
        if self.ctx.cfg.mission.announce:
            self.announce_fn(text)

    def _go(self, state: str, why: str = "", child: Behavior | None = None) -> None:
        self.state, self.state_t = state, self.ctx.now
        self.history.append((self.ctx.now, state, why))
        self.ctx.note(f"-> {state} {why}")
        self.child = child
        self.cmd = None
        if child is not None:
            child.start(self.ctx)

    @property
    def status(self) -> str:
        s = self.child.status if self.child is not None else ""
        return f"{self.state}: {s}" if s else self.state

    def start(self, wait_for_operator: bool = False) -> None:
        """wait_for_operator: stay on the ground (perception running, so the dashboard can be checked)
        until the operator says 'takeoff' / presses t."""
        if wait_for_operator:
            self.ctx.perception.set_mode("follow")
            self._go("IDLE", "waiting for 'takeoff'")
            self.announce("Ready. Say or type 'takeoff' to start.")
            return
        if self.ctx.cfg.mission.takeoff:
            self.ctx.drone.takeoff()
            self._go("TAKEOFF", "start")
        else:
            self._go("FOLLOW", "no takeoff (ground / dry run)", FollowBehind(self.ctx.cfg.follow))

    def query(self, text: str) -> None:
        self._pending.append(text)

    def force_land(self, why: str) -> None:
        if self.state not in ("LAND", "LANDED"):
            self._land(why)

    def hold(self) -> None:
        """Operator pause: stop whatever runs and hover. resume() continues with a fresh search/follow.
        Ignored while taking off / climbing / landing (a 'stop' there would confuse the drone's state)."""
        if self.state in ("IDLE", "TAKEOFF", "CLIMB", "LAND", "LANDED", "HOLD"):
            return
        self._resume_state = self.state
        self.ctx.drone.stop()
        self._go("HOLD", "operator pause")

    def resume(self) -> None:
        if self.state == "HOLD":
            prev = self._resume_state
            if prev in ("EXPLORE", "APPROACH", "ARRIVED") and self.ctx.target_cls:
                self._go("EXPLORE", "resume search", Explore(self.ctx.cfg.explore))
            else:
                self._go("REACQUIRE", "resume", ReacquirePerson())

    # ------------------------------------------------------------------ queries
    def _handle(self, text: str) -> None:
        vocab = self.ctx.perception.vocabulary()
        if self.state == "IDLE":
            if text.strip().lower().rstrip("!.") in ("takeoff", "take off", "start", "go", "launch", "fly"):
                self.start()
            elif "land" in text.lower():
                self._go("LANDED", "stayed on the ground")
            else:
                self.announce("I'm still on the ground. Say 'takeoff' first.")
            return
        q = self.parser.parse(text, vocab)
        self.last_query = q
        self.ctx.note(f"query {text!r} -> {q.intent} {q.target} ({q.reason})")
        if q.intent == "land":
            self._land("asked to land")
        elif q.intent == "cancel":
            if self.state in ("EXPLORE", "APPROACH", "ARRIVED", "HOLD"):
                self.ctx.drone.stop()
                self.announce("Okay, stopping. Coming back to you.")
                self._reacquire("cancelled")
        elif q.intent == "follow":
            lost = self.ctx.res is None or self.ctx.res.person_unseen_s > 2.0
            if self.state in ("EXPLORE", "APPROACH", "ARRIVED", "HOLD") or (self.state == "FOLLOW" and lost):
                self.ctx.drone.stop()
                self.announce("Coming back to you." if self.state != "FOLLOW" else "Looking for you.")
                self._reacquire("asked to follow")
        elif q.intent == "describe":
            self.announce(self.describe())
        elif q.intent == "find":
            if q.target is None:
                what = q.unknown_target or "that"
                self.announce(f"Sorry, I can't look for {what} yet. I can look for: {', '.join(vocab[:8])}.")
                return
            if self.state in ("EXPLORE", "APPROACH") and q.target == self.ctx.target_cls:
                self.announce(f"Still looking for your {q.target}.")  # a repeat must not restart the search
                return
            if self.state in ("FOLLOW", "ARRIVED", "HOLD", "REACQUIRE", "EXPLORE", "APPROACH"):
                self.ctx.drone.stop()
                self._begin_search(q.target)
            elif self.state in ("TAKEOFF", "CLIMB"):
                if self._deferred is None:
                    self.announce("Give me a second, I'm still taking off.")
                self._deferred = text  # handled once we are following
            else:
                self.announce("I've landed. Say 'takeoff' to start again." if self.state == "LANDED" else "I'm landing.")
        else:
            self.announce("Sorry, I didn't understand. Try: find my water bottle.")

    def _begin_search(self, target: str) -> None:
        ctx = self.ctx
        # the old person reference, expressed relative to the drone NOW (only meaningful if the drone moved by
        # discrete, odometry-tracked commands since then, i.e. not after following with rc)
        old = None
        if self.state not in ("FOLLOW",) and ctx.person_origin is not None:
            pose = ctx.odom.pose
            h = math.radians(pose.heading_deg)
            dx, dy = ctx.person_origin[0] - pose.x, ctx.person_origin[1] - pose.y
            rel = (dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h))
            rel_h = None if ctx.person_heading is None else wrap_deg(ctx.person_heading - pose.heading_deg)
            old = (rel, rel_h)
        ctx.target_cls = target
        ctx.perception.set_target(target)
        # new mission frame at the drone
        ctx.odom.reset(ctx.tel)
        ctx.grid = Grid(ctx.cfg.explore.grid_size_m, ctx.cfg.explore.grid_cell_m)
        ctx.memory = SemanticMemory()
        ctx.vantages.clear()
        ctx.vantage = 0
        ctx.person_origin, ctx.person_heading = old if old is not None else (None, None)
        # where the person is and which way they face: seen in the last 2 s (a single frame may be a detector
        # dropout) beats the old reference
        p = ctx.last_person if ctx.now - ctx.last_person_t <= 2.0 else None
        if p is not None and p.range_m is not None:
            ctx.person_origin = ctx.odom.pose.point_at(p.range_m, p.bearing_deg)
            facing = self._smoothed_facing(p)
            if facing is not None:
                ctx.person_heading = wrap_deg(ctx.odom.pose.heading_deg + p.bearing_deg + facing)
        self.guidance = None
        self.announce(f"Looking for your {target}.")
        self._go("EXPLORE", f"search for {target}", Explore(self.ctx.cfg.explore))

    def _smoothed_facing(self, p) -> float | None:
        """The follow behaviour's circular mean of recent confident facings, else this frame's if confident."""
        f = getattr(self.child, "facings", None)
        if f and len(f) >= 3:
            return math.degrees(math.atan2(sum(math.sin(a) for a in f), sum(math.cos(a) for a in f)))
        if p.facing_deg is not None and p.facing_conf >= 0.5:
            return p.facing_deg
        return None

    def describe(self) -> str:
        objs = [o for o in self.ctx.memory.objects if o.cls != "person"]
        if not objs:
            return "I haven't seen anything around here yet."
        names = sorted({o.cls for o in objs})
        return "I've seen: " + ", ".join(names) + "."

    def _reacquire(self, why: str) -> None:
        self._go("REACQUIRE", why)

    def _land(self, why: str) -> None:
        self.ctx.drone.land()  # cancels a running move itself; idempotent if already landing
        self._land_sent = self.ctx.now
        self.announce("Landing.")
        self._go("LAND", why)

    # ------------------------------------------------------------------ main step
    def step(self) -> None:
        for text in self._pending[:]:
            self._pending.remove(text)
            self._handle(text)
            if self.state in ("LAND", "LANDED"):
                break
        ctx, cfg, d = self.ctx, self.ctx.cfg, self.ctx.drone
        st = self.state
        if st == "TAKEOFF":
            if not d.busy():
                if d.last_result() == "ok" and d.flying:
                    self._go("CLIMB", "airborne")
                else:
                    self.announce("Takeoff failed.")
                    self._go("LANDED", f"takeoff: {d.last_result()}")
        elif st == "CLIMB":
            if self.cmd is None:
                alt = ctx.altitude
                cm = int(round((cfg.follow.altitude_m - alt) * 100)) if alt is not None else 0
                if cm < 20:
                    self._go("FOLLOW", "at follow altitude", FollowBehind(cfg.follow))
                    return
                self.cmd = Discrete("move", min(cm, 200), "up")
            r = self.cmd.step(ctx)
            if r != RUNNING:
                self.cmd = None
                self._go("FOLLOW", "climbed" if r == SUCCESS else f"climb: {d.last_result()}", FollowBehind(cfg.follow))
        elif st == "FOLLOW":
            if self._deferred is not None:
                text, self._deferred = self._deferred, None
                self._handle(text)
                return
            self.child.step(ctx)
            if ctx.res is not None and ctx.res.person_unseen_s > 20.0 and ctx.now - self.state_t > 20.0:
                self._reacquire("person lost for 20 s")
        elif st == "EXPLORE":
            r = self.child.step(ctx)
            if r == SUCCESS:
                self.announce(f"I see the {ctx.target_cls}. Going there.")
                self._go("APPROACH", self.child.status, Approach(cfg.approach, min_altitude_m=cfg.safety.min_altitude_m))
            elif r == FAILURE:
                self.announce(f"I couldn't find the {ctx.target_cls}. Coming back to you.")
                self._reacquire(self.child.status)
        elif st == "APPROACH":
            r = self.child.step(ctx)
            if r == SUCCESS:
                self._approach_retries = 0
                self._arrived(self.child.status)
            elif r == FAILURE:
                obj = ctx.memory.best(ctx.target_cls, confirmed_only=False)
                if obj is not None and self._approach_retries < 1:
                    self._approach_retries += 1  # e.g. someone walked through the path: try once more
                    self._go("APPROACH", f"retry ({self.child.status})",
                             Approach(cfg.approach, min_altitude_m=cfg.safety.min_altitude_m))
                elif obj is not None:
                    self._approach_retries = 0
                    far = ctx.odom.pose.distance_to(*obj.xy) > cfg.approach.standoff_m + 1.5
                    if far:
                        self.announce(f"I saw the {ctx.target_cls} but couldn't get close to it.")
                    self._arrived(f"approach failed ({self.child.status}); using memory")
                else:
                    self._approach_retries = 0
                    self.announce(f"I lost the {ctx.target_cls}. Coming back to you.")
                    self._reacquire(self.child.status)
        elif st == "ARRIVED":
            d.rc(0, 0, 0, 0)  # hover next to the target as a beacon (keep-alive)
        elif st == "REACQUIRE":
            self._step_reacquire()
        elif st == "HOLD":
            if not d.busy():
                d.rc(0, 0, 0, 0)
        elif st == "LAND":
            if not d.busy():
                if not d.flying:
                    self._go("LANDED", "landed")
                elif ctx.now - self._land_sent > 2.0:
                    # the land failed or its reply was lost while still in the air: ask again
                    ctx.note(f"land not confirmed ({d.last_result()}): sending land again")
                    self._land_sent = ctx.now
                    d.land()

    def _arrived(self, why: str) -> None:
        ctx = self.ctx
        obj = ctx.memory.best(ctx.target_cls, confirmed_only=False)
        target_xy = obj.xy if obj is not None else ctx.odom.pose.point_at(ctx.cfg.approach.standoff_m, 0.0)
        self.guidance = compute_guidance(ctx.target_cls, target_xy, ctx.person_origin, ctx.person_heading)
        self.announce(self.guidance.text)
        self._go("ARRIVED", why)

    def _step_reacquire(self) -> None:
        """Go to LOOK height (just below head height: from follow height a person closer than ~1.5 m is out of
        the bottom of the frame), turn until the person is seen, then follow (FOLLOW climbs back up)."""
        ctx, cfg = self.ctx, self.ctx.cfg
        if self.child is None:
            if self.cmd is None:
                alt = ctx.altitude
                look = max(cfg.safety.min_altitude_m + 0.2, cfg.perception.person_height_m - 0.15)
                cm = int(round((look - alt) * 100)) if alt is not None else 0
                if abs(cm) >= 20:
                    self.cmd = Discrete("move", min(abs(cm), 200), "up" if cm > 0 else "down")
                else:
                    self.child = ReacquirePerson()
                    self.child.start(ctx)
                    return
            if self.cmd is not None and self.cmd.step(ctx) != RUNNING:
                self.cmd = None
                self.child = ReacquirePerson()
                self.child.start(ctx)
            return
        r = self.child.step(ctx)
        if r == SUCCESS:
            self._go("FOLLOW", "person found", FollowBehind(cfg.follow))
        elif r == FAILURE:
            self.announce("I can't see you. Say 'follow me' when you're in front of me, or 'land'.")
            self._go("HOLD", "person not found")
            self._resume_state = "FOLLOW"
