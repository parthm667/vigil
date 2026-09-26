"""Find the person again (after a search): rotate in steps until the person detector sees them."""

from __future__ import annotations

from .base import FAILURE, RUNNING, SUCCESS, Behavior, Ctx, Discrete


class ReacquirePerson(Behavior):
    name = "reacquire"

    def __init__(self, step_deg: int = 30, dwell_s: float = 0.8, max_steps: int = 13):
        super().__init__()
        self.step_deg, self.dwell_s, self.max_steps = step_deg, dwell_s, max_steps

    def start(self, ctx: Ctx) -> None:
        ctx.perception.set_mode("follow")
        self.cmd: Discrete | None = None
        self.steps = 0
        self.dwell_start = ctx.now
        self.first_turn = None
        # turn first toward where the person was when the query came, if we know it
        if ctx.person_origin is not None:
            b = ctx.odom.pose.bearing_to(*ctx.person_origin)
            if abs(b) >= 10:
                self.first_turn = b

    def step(self, ctx: Ctx) -> str:
        if self.cmd is not None:
            r = self.cmd.step(ctx)
            if r == RUNNING:
                return RUNNING
            self.cmd = None
            self.dwell_start = ctx.now
            return RUNNING
        ctx.drone.rc(0, 0, 0, 0)
        if ctx.now - self.dwell_start < self.dwell_s:
            return RUNNING
        if ctx.new_frame and ctx.res is not None and ctx.res.person is not None:
            self.status = "person found"
            return SUCCESS
        if not ctx.new_frame:
            return RUNNING
        if self.steps >= self.max_steps:
            self.status = "person not found"
            return FAILURE
        self.steps += 1
        if self.first_turn is not None:
            turn, self.first_turn = self.first_turn, None
        else:
            turn = self.step_deg
        self.cmd = Discrete("rotate", turn)
        self.status = f"looking for the person ({self.steps})"
        return RUNNING
