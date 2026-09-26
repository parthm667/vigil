"""Behaviour interface and the shared mission context.

A behaviour is a small state machine: start(ctx) once, then step(ctx) every loop until it returns
SUCCESS or FAILURE. Behaviours only talk to the world through the context (drone, perception result,
odometry, grid, memory), so each can be replaced or re-tuned on its own.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

from ..config import Config
from ..drone.base import Drone
from ..mapping import FreeSpaceEstimator, Grid, Odometry, SemanticMemory
from ..perception import Perception
from ..types import Frame, PerceptionResult, Telemetry, wrap_deg

RUNNING, SUCCESS, FAILURE = "running", "success", "failure"
log = logging.getLogger("reachglass.behaviors")


@dataclass
class Ctx:
    cfg: Config
    drone: Drone
    perception: Perception
    odom: Odometry = field(default_factory=Odometry)
    grid: Grid | None = None
    memory: SemanticMemory = field(default_factory=SemanticMemory)
    freespace: FreeSpaceEstimator | None = None
    now: float = 0.0
    frame: Frame | None = None
    new_frame: bool = False  # `frame` arrived this loop (perception ran on it)
    res: PerceptionResult | None = None  # latest perception result
    tel: Telemetry | None = None
    say: Callable[[str], None] = print
    vantage: int = 0
    vantages: list[tuple[float, float]] = field(default_factory=list)
    person_origin: tuple[float, float] | None = None  # where the person stood when the query came
    person_heading: float | None = None  # which way they faced then (mission frame, clockwise)
    target_cls: str | None = None
    notes: list[str] = field(default_factory=list)
    cmd_done_t: float = float("-inf")  # when the last discrete command finished
    last_person: object | None = None  # latest PersonObs of the locked person (survives detector dropouts)
    last_person_t: float = float("-inf")

    def frame_after_cmd(self) -> bool:
        """The current frame shows the world AFTER the last discrete command (video lag accounted for)."""
        return self.frame is not None and self.frame.t >= self.cmd_done_t + self.cfg.drone.video_lag_s

    def __post_init__(self):
        if self.grid is None:
            self.grid = Grid(self.cfg.explore.grid_size_m, self.cfg.explore.grid_cell_m)

    def note(self, msg: str) -> None:
        self.notes.append(f"{self.now:7.2f} {msg}")
        log.info(msg)

    @property
    def altitude(self) -> float | None:
        """Height above the FLOOR (holding altitude, flying over furniture decisions)."""
        return None if self.tel is None else self.tel.floor_altitude_m()

    @property
    def clearance(self) -> float | None:
        """Distance to whatever is directly below (downward ToF): for descending."""
        return None if self.tel is None else self.tel.altitude_m()


class Behavior:
    name = "behavior"

    def __init__(self):
        self.status = ""

    def start(self, ctx: Ctx) -> None:
        pass

    def step(self, ctx: Ctx) -> str:
        raise NotImplementedError


class Discrete:
    """One discrete drone command: issue once, wait for the reply, update odometry on success."""

    def __init__(self, kind: str, value: float, direction: str = ""):
        self.kind, self.value, self.direction = kind, value, direction  # kind: rotate | move
        self.issued = False
        self.result: str | None = None
        self._yaw_before: float | None = None

    def step(self, ctx: Ctx) -> str:
        d = ctx.drone
        if not self.issued:
            if d.busy():
                return RUNNING  # something else still running (should not happen)
            self._yaw_before = ctx.tel.yaw_deg if ctx.tel else None
            if self.kind == "rotate":
                d.rotate(int(round(self.value)))
            else:
                d.move(self.direction, int(round(self.value)))
            self.issued = True
            if not d.busy():  # refused immediately (e.g. safety)
                self.result = d.last_result()
                return FAILURE if self.result != "ok" else SUCCESS
            return RUNNING
        if d.busy():
            return RUNNING
        self.result = d.last_result()
        ctx.cmd_done_t = ctx.now
        ctx.perception.reset_tracks()  # boxes jump after a discrete command: start tracking afresh
        if self.result != "ok":
            if self.kind == "move" and self.result == "error: timeout":
                # the reply was lost but the Tello most likely flew the move: credit it (better than a jump
                # of the whole map later), and report failure so the caller re-measures
                ctx.odom.on_move_done(self.direction, self.value, d.telemetry())
            return FAILURE
        tel = d.telemetry()
        if self.kind == "rotate":
            ctx.odom.on_rotation_done(self.value, self._yaw_before, tel.yaw_deg)
            ctx.odom.update(tel)
        else:
            ctx.odom.on_move_done(self.direction, self.value, tel)
        return SUCCESS


def rotation_to(ctx: Ctx, world_heading_deg: float) -> float:
    """Signed clockwise rotation (deg) from the current heading to a world heading."""
    return wrap_deg(world_heading_deg - ctx.odom.pose.heading_deg)


def angdiff(a: float, b: float) -> float:
    return abs(wrap_deg(a - b))


def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def deadband(v: float, db: float) -> float:
    return 0.0 if abs(v) < db else v - math.copysign(db, v)
