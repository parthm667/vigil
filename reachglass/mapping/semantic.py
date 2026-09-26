"""Semantic memory: what was seen where, in the mission frame.

Each sighting (class, world x/y, confidence, range) joins an existing object of the same class within
`merge_radius` (small things 0.6 m, furniture 1.0 m) or starts a new one. An object's position is the
weighted mean of its sightings, weight = conf / range^2: range errors grow with distance, so a close view
counts much more than a far one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SMALL = {"bottle", "cup", "cell phone", "remote", "book", "laptop", "backpack", "handbag", "keyboard", "mouse"}


@dataclass
class Sighting:
    x: float
    y: float
    conf: float
    range_m: float
    t: float
    vantage: int = -1
    confirmed: bool = False  # the perception lock had confirmed it


@dataclass
class MemObject:
    id: int
    cls: str
    sightings: list[Sighting] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return sum(s.conf / max(s.range_m, 0.3) ** 2 for s in self.sightings)

    @property
    def xy(self) -> tuple[float, float]:
        w = [s.conf / max(s.range_m, 0.3) ** 2 for s in self.sightings]
        tot = sum(w)
        return (sum(wi * s.x for wi, s in zip(w, self.sightings)) / tot, sum(wi * s.y for wi, s in zip(w, self.sightings)) / tot)

    @property
    def best_conf(self) -> float:
        return max(s.conf for s in self.sightings)

    @property
    def n(self) -> int:
        return len(self.sightings)

    @property
    def confirmed(self) -> bool:
        return any(s.confirmed for s in self.sightings) or self.n >= 3

    @property
    def closest_range(self) -> float:
        return min(s.range_m for s in self.sightings)

    @property
    def last_t(self) -> float:
        return max(s.t for s in self.sightings)


class SemanticMemory:
    def __init__(self, small_radius_m: float = 0.6, large_radius_m: float = 1.0):
        self.small_radius_m = small_radius_m
        self.large_radius_m = large_radius_m
        self.objects: list[MemObject] = []
        self._next = 1

    def clear(self) -> None:
        self.objects.clear()

    def add(self, cls: str, x: float, y: float, conf: float, range_m: float, t: float, vantage: int = -1,
            confirmed: bool = False) -> MemObject:
        radius = self.small_radius_m if cls in SMALL else self.large_radius_m
        best, best_d = None, math.inf
        for o in self.objects:
            if o.cls != cls:
                continue
            ox, oy = o.xy
            d = math.hypot(ox - x, oy - y)
            # far sightings are uncertain: allow a bigger merge radius for them
            r = radius + 0.15 * range_m
            if d < r and d < best_d:
                best, best_d = o, d
        s = Sighting(x, y, conf, range_m, t, vantage, confirmed)
        if best is None:
            best = MemObject(self._next, cls)
            self._next += 1
            self.objects.append(best)
        best.sightings.append(s)
        return best

    def of_class(self, cls: str) -> list[MemObject]:
        return [o for o in self.objects if o.cls == cls]

    def best(self, cls: str, confirmed_only: bool = True) -> MemObject | None:
        cands = [o for o in self.of_class(cls) if o.confirmed or not confirmed_only]
        return max(cands, key=lambda o: (o.confirmed, o.weight), default=None)

    def summary(self) -> list[dict]:
        out = []
        for o in self.objects:
            x, y = o.xy
            out.append({"id": o.id, "cls": o.cls, "x": round(x, 2), "y": round(y, 2), "n": o.n,
                        "conf": round(o.best_conf, 2), "confirmed": o.confirmed})
        return out
