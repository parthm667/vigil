"""One sense step shared by the app and the simulator: frame -> perception -> telemetry -> odometry."""

from __future__ import annotations

from ..sources.base import FrameSource
from .base import Ctx


def sense(ctx: Ctx, source: FrameSource, now: float) -> None:
    """Update ctx with the newest frame (perception runs only on NEW frames), telemetry and odometry."""
    ctx.now = now
    ctx.tel = ctx.drone.telemetry()
    f = source.read()
    ctx.new_frame = f is not None and (ctx.frame is None or f.seq != ctx.frame.seq)
    if ctx.new_frame:
        ctx.frame = f
        ctx.res = ctx.perception.update(f, ctx.tel)
        if ctx.res.person is not None:
            ctx.last_person, ctx.last_person_t = ctx.res.person, now
    ctx.odom.update(ctx.tel)
