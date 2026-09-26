"""Coverage / obstacle grid in the mission frame (0.25 m cells).

  seen[c]      how many dwell frames have looked at cell c (within view range and free distance)
  free[c]      evidence that c is empty at flight level: a line of sight to a detected object passes
               through it (we saw something farther away), or a metric free-space ray covered it
  blocked[c]   evidence of an obstacle at c (a detected object's footprint, end of a free-space ray)
Used by the explorer: novelty = unseen fraction along a direction; hops only through free-evidenced cells
(plus a short cautious step into the unknown).
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Pose2D


class Grid:
    def __init__(self, size_m: float = 16.0, cell_m: float = 0.25):
        self.cell = cell_m
        self.n = int(round(size_m / cell_m))
        self.half = size_m / 2.0
        self.seen = np.zeros((self.n, self.n), np.int32)
        self.free = np.zeros((self.n, self.n), np.int32)
        self.blocked = np.zeros((self.n, self.n), np.float32)
        self.top = np.zeros((self.n, self.n), np.float32)  # highest known obstacle top (m); inf = unknown height

    def idx(self, x: float, y: float) -> tuple[int, int] | None:
        i, j = int((x + self.half) / self.cell), int((y + self.half) / self.cell)
        return (i, j) if 0 <= i < self.n and 0 <= j < self.n else None

    def center(self, i: int, j: int) -> tuple[float, float]:
        return (i + 0.5) * self.cell - self.half, (j + 0.5) * self.cell - self.half

    def _ray_cells(self, x0: float, y0: float, heading_deg: float, dist: float, start: float = 0.0):
        h = math.radians(heading_deg)
        step = self.cell * 0.5
        out, last = [], None
        r = start
        while r <= dist + 1e-9:
            c = self.idx(x0 + r * math.cos(h), y0 + r * math.sin(h))
            if c is None:
                break
            if c != last:
                out.append(c)
                last = c
            r += step
        return out

    def mark_view(self, pose: Pose2D, hfov_deg: float, max_range_m: float,
                  free: list[tuple[float, float | None]] | None = None, rays: int = 15) -> None:
        """Mark the cells a dwell frame looked at. free: [(bearing_deg, free_dist_m or None)] sectors; a
        known free distance shortens the rays there and marks the obstacle at its end."""
        half = hfov_deg / 2.0
        for k in range(rays):
            b = -half + hfov_deg * k / (rays - 1)
            limit, hit = max_range_m, False
            if free:
                nearest = min(free, key=lambda s: abs(s[0] - b))
                if nearest[1] is not None and nearest[1] < max_range_m:
                    limit, hit = nearest[1], True
            cells = self._ray_cells(pose.x, pose.y, pose.heading_deg + b, limit)
            for c in cells:
                self.seen[c] += 1
            if hit and cells:
                self.blocked[cells[-1]] += 1.0
                self.top[cells[-1]] = np.inf  # a free-space ray ends at flight level: treat as tall
                for c in cells[:-1]:
                    self.free[c] += 1

    def mark_free_ray(self, x: float, y: float, heading_deg: float, dist_m: float) -> None:
        """Everything along this ray up to dist_m is empty (e.g. we saw an object beyond it)."""
        if dist_m <= 0:
            return
        for c in self._ray_cells(x, y, heading_deg, dist_m):
            self.free[c] += 1

    def mark_blocked(self, x: float, y: float, radius_m: float = 0.0, weight: float = 1.0, top_m: float = np.inf) -> None:
        """Obstacle footprint around (x, y) reaching up to top_m (inf = unknown height)."""
        r = max(0, int(math.ceil(radius_m / self.cell)))
        c = self.idx(x, y)
        if c is None:
            return
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i, j = c[0] + di, c[1] + dj
                if 0 <= i < self.n and 0 <= j < self.n and math.hypot(di, dj) * self.cell <= radius_m + 1e-9:
                    self.blocked[i, j] += weight
                    self.top[i, j] = max(self.top[i, j], top_m)

    def blocks(self, c: tuple[int, int], altitude_m: float | None, thresh: float = 1.0, clearance_m: float = 0.2) -> bool:
        """Is cell c an obstacle for a drone flying at altitude_m (None = unknown altitude: any obstacle)?"""
        if self.blocked[c] < thresh:
            return False
        return altitude_m is None or self.top[c] + clearance_m > altitude_m

    def novelty(self, pose: Pose2D, heading_deg: float, dist_m: float) -> float:
        """Fraction of never-seen cells along a direction (0 = all seen, 1 = all new)."""
        cells = self._ray_cells(pose.x, pose.y, heading_deg, dist_m, start=self.cell)
        if not cells:
            return 0.0
        return float(np.mean([self.seen[c] == 0 for c in cells]))

    def clear_distance(self, pose: Pose2D, heading_deg: float, max_m: float, unknown_ok_m: float = 0.5,
                       thresh: float = 1.0, margin_m: float = 0.35, altitude_m: float | None = None) -> float:
        """How far the drone may go along a direction: through free-evidenced cells, stopping `margin_m`
        before an obstacle that reaches its flight altitude; beyond the evidence only `unknown_ok_m`."""
        h = math.radians(heading_deg)
        r, last_ok = self.cell * 0.5, 0.0
        while r <= max_m + 1e-9:
            c = self.idx(pose.x + r * math.cos(h), pose.y + r * math.sin(h))
            if c is None or self.blocks(c, altitude_m, thresh):
                return max(0.0, r - margin_m)
            if self.free[c] <= 0 and r > unknown_ok_m:
                return last_ok if last_ok > 0 else min(r, unknown_ok_m)
            last_ok = r
            r += self.cell * 0.5
        return max_m

    def coverage(self) -> float:
        return float((self.seen > 0).mean())
