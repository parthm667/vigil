"""Guidance info: where the target is relative to the PERSON (for telling them how to get there).

At the moment the query arrives the drone knows where the person stands and which way they face
(person estimator: range, bearing, facing). After the search, memory knows where the target is. Both are
in the mission frame, so the target's direction relative to the person's own heading follows directly:

    g = compute_guidance(ctx)
    g.distance_m, g.turn_deg (+ = to their right), g.clock ("2 o'clock"), g.text

This is also the hook for the next stage (guiding the person with the glasses): `relative_to(x, y, h)`
recomputes distance/turn from the person's CURRENT position and heading as they walk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import wrap_deg


def clock_face(turn_deg: float) -> str:
    """0 = 12 o'clock (straight ahead), +90 = 3 o'clock (right), 180 = 6 o'clock (behind)."""
    h = int(round((turn_deg % 360) / 30.0)) % 12
    return f"{12 if h == 0 else h} o'clock"


def direction_words(turn_deg: float) -> str:
    a = wrap_deg(turn_deg)
    side = "right" if a > 0 else "left"
    m = abs(a)
    if m < 12:
        return "straight ahead"
    if m < 35:
        return f"ahead, slightly to your {side}"
    if m < 70:
        return f"ahead to your {side}"
    if m < 110:
        return f"to your {side}"
    if m < 160:
        return f"behind you on your {side}"
    return "behind you"


@dataclass
class Guidance:
    target_cls: str
    target_xy: tuple[float, float]
    person_xy: tuple[float, float] | None
    person_heading_deg: float | None
    distance_m: float | None
    turn_deg: float | None  # + = the person turns right
    text: str

    @property
    def clock(self) -> str | None:
        return None if self.turn_deg is None else clock_face(self.turn_deg)

    def relative_to(self, x: float, y: float, heading_deg: float | None) -> tuple[float, float | None]:
        """(distance, turn) from a person at (x, y) facing heading_deg (mission frame)."""
        dx, dy = self.target_xy[0] - x, self.target_xy[1] - y
        dist = math.hypot(dx, dy)
        if heading_deg is None:
            return dist, None
        return dist, wrap_deg(math.degrees(math.atan2(dy, dx)) - heading_deg)


def compute_guidance(target_cls: str, target_xy: tuple[float, float], person_xy: tuple[float, float] | None,
                     person_heading_deg: float | None) -> Guidance:
    name = "bottle" if target_cls == "bottle" else target_cls
    if person_xy is None:
        return Guidance(target_cls, target_xy, None, None, None, None,
                        f"I found the {name}. I'm hovering right next to it.")
    g = Guidance(target_cls, target_xy, person_xy, person_heading_deg, None, None, "")
    dist, turn = g.relative_to(*person_xy, person_heading_deg)
    g.distance_m, g.turn_deg = dist, turn
    meters = f"about {dist:.0f} meters" if dist >= 1.5 else "about a meter"
    if turn is None:
        g.text = f"I found the {name}, {meters} from where you were standing. I'm hovering next to it."
    else:
        g.text = f"I found the {name}: {meters} away, {direction_words(turn)} ({clock_face(turn)})."
    return g
