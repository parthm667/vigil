"""Static obstacles for follow episodes: footprints with heights, placed around the user's path.

Each obstacle is an oriented rectangle on the ground plane (center, half extents, yaw) with a
height. Placement uses the walker's pre-sampled base path and its own RNG stream:
- on-path obstacles sit on the user's path and the user's path gets a smooth detour around
  them, so the drone trailing behind finds the obstacle between itself and the user;
- near-path obstacles sit beside the path with a gap, so the drone can clip them when it cuts
  a corner, and tall ones can hide the user's head.
The user's path always keeps person_clear_m from every footprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Obstacle:
    x: float
    y: float
    hx: float  # half extent along the obstacle's own x axis (m)
    hy: float
    yaw: float  # rad, counter-clockwise from world +x
    height: float  # top above the floor (m)
    kind: str = "box"
    c: float = field(init=False, repr=False)
    s: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.c, self.s = math.cos(self.yaw), math.sin(self.yaw)

    def dist(self, px: float, py: float) -> float:
        """Horizontal distance from a point to the footprint (0 inside)."""
        dx, dy = px - self.x, py - self.y
        lx = abs(self.c * dx + self.s * dy) - self.hx
        ly = abs(-self.s * dx + self.c * dy) - self.hy
        if lx <= 0.0 and ly <= 0.0:
            return 0.0
        if lx <= 0.0:
            return ly
        if ly <= 0.0:
            return lx
        return math.hypot(lx, ly)

    def segment_interval(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float] | None:
        """Parameter interval [s_in, s_out] within [0, 1] where the segment p0 -> p1 is over the footprint."""
        c, s = self.c, self.s
        ax, ay = x0 - self.x, y0 - self.y
        bx, by = x1 - x0, y1 - y0
        p0 = (c * ax + s * ay, -s * ax + c * ay)
        d = (c * bx + s * by, -s * bx + c * by)
        lo, hi = 0.0, 1.0
        for k, h in ((0, self.hx), (1, self.hy)):
            if abs(d[k]) < 1e-12:
                if abs(p0[k]) > h:
                    return None
                continue
            t0, t1 = (-h - p0[k]) / d[k], (h - p0[k]) / d[k]
            if t0 > t1:
                t0, t1 = t1, t0
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return None
        return lo, hi

    def extent_along(self, nx: float, ny: float) -> float:
        """Half extent of the footprint along the unit vector (nx, ny)."""
        return self.hx * abs(self.c * nx + self.s * ny) + self.hy * abs(-self.s * nx + self.c * ny)

    def overlaps_rect(self, cx: float, cy: float, ux: float, uy: float, a: float, b: float) -> bool:
        """Separating-axis test against a rectangle centered at (cx, cy), unit axis (ux, uy), half extents a, b."""
        tx, ty = cx - self.x, cy - self.y
        vx, vy = -uy, ux
        for lx, ly in ((ux, uy), (vx, vy), (self.c, self.s), (-self.s, self.c)):
            ra = a * abs(ux * lx + uy * ly) + b * abs(vx * lx + vy * ly)
            rb = self.hx * abs(self.c * lx + self.s * ly) + self.hy * abs(-self.s * lx + self.c * ly)
            if abs(tx * lx + ty * ly) > ra + rb:
                return False
        return True

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "hx": self.hx, "hy": self.hy, "yaw": self.yaw, "height": self.height, "kind": self.kind}


def _u(rng: np.random.Generator, r) -> float:
    return float(rng.uniform(r[0], r[1])) if isinstance(r, (list, tuple)) else float(r)


def _sample_type(rng: np.random.Generator, types: dict, tall_only: bool) -> tuple[str, dict]:
    names = [k for k, v in types.items() if v.get("tall", False) or not tall_only]
    p = np.array([float(types[k]["p"]) for k in names])
    k = names[int(rng.choice(len(names), p=p / p.sum()))]
    return k, types[k]


def _clear(xs: np.ndarray, ys: np.ndarray, obs: list[Obstacle], d_min: float) -> bool:
    for o in obs:
        dx, dy = xs - o.x, ys - o.y
        lx = np.abs(o.c * dx + o.s * dy) - o.hx
        ly = np.abs(-o.s * dx + o.c * dy) - o.hy
        d = np.hypot(np.maximum(lx, 0.0), np.maximum(ly, 0.0))
        if float(d.min()) < d_min:
            return False
    return True


def place_obstacles(rng: np.random.Generator, xs: list[float], ys: list[float], drone_xy: tuple[float, float],
                    cfg: dict, dt: float) -> tuple[list[Obstacle], list[float], list[float]]:
    """Returns (obstacles, detoured xs, detoured ys). Candidates that break a clearance rule are redrawn (20 tries)."""
    X, Y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(X)
    types = cfg["types"]
    clear = float(cfg.get("person_clear_m", 0.3))
    start_clear = float(cfg.get("start_clear_m", 1.0))
    w0, w1 = cfg.get("window_s", (4.0, 54.0))
    i_lo, i_hi = max(6, int(w0 / dt)), min(n - 7, int(w1 / dt))
    sp = np.zeros(n)
    sp[1:] = np.hypot(np.diff(X), np.diff(Y)) / dt
    moving = [i for i in range(i_lo, i_hi) if sp[i] > 0.3]
    anchors = [(float(X[0]), float(Y[0])), drone_xy]
    n_on = int(rng.integers(cfg["n_on_path"][0], cfg["n_on_path"][1] + 1))
    n_near = int(rng.integers(cfg["n_near_path"][0], cfg["n_near_path"][1] + 1))
    obs: list[Obstacle] = []

    def direction(i: int) -> tuple[float, float]:
        for k in (5, 15, 40):
            a, b = max(0, i - k), min(n - 1, i + k)
            dx, dy = X[b] - X[a], Y[b] - Y[a]
            d = math.hypot(dx, dy)
            if d > 0.1:
                return dx / d, dy / d
        return 1.0, 0.0

    def ok_start(o: Obstacle) -> bool:
        return all(o.dist(ax, ay) >= start_clear for ax, ay in anchors)

    for _ in range(n_on):
        if not moving:
            break
        for _try in range(20):
            i0 = moving[int(rng.integers(len(moving)))]
            kind, t = _sample_type(rng, types, rng.random() < cfg.get("on_path_tall_prob", 0.7))
            tx, ty = direction(i0)
            o = Obstacle(float(X[i0]), float(Y[i0]), 0.5 * _u(rng, t["sx"]), 0.5 * _u(rng, t["sy"]),
                         math.atan2(ty, tx) + math.radians(_u(rng, (-30.0, 30.0))), _u(rng, t["h"]), kind)
            side = 1.0 if rng.random() < 0.5 else -1.0
            nx, ny = -ty * side, tx * side
            amp = o.extent_along(nx, ny) + clear + 0.1
            half = o.extent_along(tx, ty) + float(cfg.get("detour_run_m", 1.2))
            # detour window by arc length around i0, raised-cosine bump of height amp along the normal
            arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(Y)))])
            a0 = arc[i0]
            idx = np.nonzero(np.abs(arc - a0) <= half)[0]
            if len(idx) < 3 or idx[0] < 20 or not ok_start(o):
                continue
            w = 0.5 * (1.0 + np.cos(np.pi * (arc[idx] - a0) / half))
            X2, Y2 = X.copy(), Y.copy()
            X2[idx] += nx * amp * w
            Y2[idx] += ny * amp * w
            if _clear(X2, Y2, obs + [o], clear):
                X, Y = X2, Y2
                obs.append(o)
                break

    for _ in range(n_near):
        for _try in range(20):
            i0 = int(rng.integers(i_lo, i_hi))
            kind, t = _sample_type(rng, types, False)
            tx, ty = direction(i0)
            side = 1.0 if rng.random() < 0.5 else -1.0
            nx, ny = -ty * side, tx * side
            o = Obstacle(0.0, 0.0, 0.5 * _u(rng, t["sx"]), 0.5 * _u(rng, t["sy"]), float(rng.uniform(-math.pi, math.pi)),
                         _u(rng, t["h"]), kind)
            off = o.extent_along(nx, ny) + _u(rng, cfg.get("near_gap_m", (0.3, 1.2)))
            o.x, o.y = float(X[i0] + nx * off), float(Y[i0] + ny * off)
            if ok_start(o) and _clear(X, Y, [o], clear):
                obs.append(o)
                break
    return obs, X.tolist(), Y.tolist()
