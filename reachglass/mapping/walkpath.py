"""Walking path for the wearer: from where they stand to within reach of the target, around mapped obstacles.

Cost map on the exploration grid (same cells, mission frame):
  core      a detected obstacle's footprint (grid.blocked >= 1) at ANY height: a chair the drone flies over
            still blocks a person                                                   -> BLOCKED_COST
  margin    within inflate_m of a core cell (half a body width plus a margin)        -> MARGIN_COST
  proven    proven clear: a line of sight to a detected object crossed it, or the drone flew through it -> 1
  unknown   never proven clear (open space: walkable, just trusted less)             -> unknown_cost
Blocked cells keep a finite (high) cost, so a path always exists: a person standing inside a margin (next to
a chair), or walled in by a noisy detection, still gets the least-bad way out.

plan_walk: Dijkstra from the person's cell over 8 neighbours. Goal = the walkable cell nearest the target
(next to a bottle on the open floor; outside keep_out for one on a table), ties within goal_slack_m -> the
cheapest to reach (the person's side of the table). Then string-pulling: the fewest straight legs that never
cross a cell costlier than the walkable ones (or than what the raw path itself had to cross).

Following it: project the person onto the path, aim `lookahead_m` further along it (past its end: at the
target itself), compare with their heading -> cue -1 (turn left), 0 (forward), +1 (turn right).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..types import wrap_deg
from .grid import Grid

BLOCKED_COST = 50.0
MARGIN_COST = 10.0
SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


@dataclass
class WalkPath:
    points: list[tuple[float, float]]  # person -> ... -> goal (mission frame); always >= 2 points
    goal: tuple[float, float]  # where the walk ends (within reach of the target)
    target: tuple[float, float]
    max_cost: float  # costliest cell any leg crosses (<= unknown_cost when every leg is walkable)

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.points, self.points[1:]))


def _dilate(mask: np.ndarray, radius_m: float, cell_m: float) -> np.ndarray:
    """Cells within radius_m (centre to centre) of a True cell. No wrap-around at the edges."""
    r = int(math.floor(radius_m / cell_m + 1e-9))
    out = mask.copy()
    n0, n1 = mask.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if (di or dj) and math.hypot(di, dj) * cell_m <= radius_m + 1e-9:
                # out[i + di, j + dj] |= mask[i, j]
                out[max(0, di):n0 - max(0, -di), max(0, dj):n1 - max(0, -dj)] |= \
                    mask[max(0, -di):n0 - max(0, di), max(0, -dj):n1 - max(0, dj)]
    return out


def walk_costs(grid: Grid, inflate_m: float = 0.4, unknown_cost: float = 1.5,
               proven_paths: list[list[tuple[float, float]]] = (),
               keep_out: list[tuple[float, float, float]] = ()) -> np.ndarray:
    """Per-cell cost of walking through (see the module docstring). keep_out: extra (x, y, radius) obstacles,
    e.g. the furniture under a raised target, which the drone cannot see while hovering over it."""
    cost = np.where(grid.free > 0, 1.0, unknown_cost)
    for path in proven_paths:  # where the drone flew: nothing tall there
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / (grid.cell * 0.5))))
            for k in range(n + 1):
                c = grid.idx(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
                if c is not None:
                    cost[c] = 1.0
    core = grid.blocked >= 1.0
    if keep_out:
        xs = (np.arange(grid.n) + 0.5) * grid.cell - grid.half
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        for kx, ky, kr in keep_out:
            core |= np.hypot(X - kx, Y - ky) <= kr
    cost[_dilate(core, inflate_m, grid.cell) & ~core] = MARGIN_COST
    cost[core] = BLOCKED_COST
    return cost


def segment_max_cost(grid: Grid, costs: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> float:
    """Highest cell cost along the straight segment a -> b (inf if it leaves the grid)."""
    n = max(1, int(math.ceil(math.dist(a, b) / (grid.cell * 0.25))))
    worst = 0.0
    for k in range(n + 1):
        c = grid.idx(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        if c is None:
            return math.inf
        worst = max(worst, float(costs[c]))
    return worst


def plan_walk(grid: Grid, costs: np.ndarray, start_xy: tuple[float, float], target_xy: tuple[float, float],
              walkable_cost: float = 1.5, goal_slack_m: float = 0.3) -> WalkPath | None:
    """Cheapest walk from start_xy to the walkable cell nearest target_xy. None if either is off the grid."""
    s, t = grid.idx(*start_xy), grid.idx(*target_xy)
    if s is None or t is None:
        return None
    n, cell = grid.n, grid.cell
    cl = costs.tolist()
    dist = [[math.inf] * n for _ in range(n)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dist[s[0]][s[1]] = 0.0
    heap = [(0.0, s)]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > dist[i][j]:
            continue
        for di, dj, step in NEIGHBOURS:
            a, b = i + di, j + dj
            if 0 <= a < n and 0 <= b < n:
                nd = d + step * cell * cl[a][b]
                if nd < dist[a][b]:
                    dist[a][b] = nd
                    parent[(a, b)] = (i, j)
                    heapq.heappush(heap, (nd, (a, b)))
    # goal: among walkable cells about as close to the target as the closest one, the cheapest to reach
    ii, jj = np.nonzero(costs <= walkable_cost + 1e-9)
    if ii.size:
        xs, ys = (ii + 0.5) * cell - grid.half, (jj + 0.5) * cell - grid.half
        d_t = np.hypot(xs - target_xy[0], ys - target_xy[1])
        near = np.nonzero(d_t <= d_t.min() + goal_slack_m)[0]
        k = min(near, key=lambda q: dist[ii[q]][jj[q]])
        goal = (int(ii[k]), int(jj[k]))
    else:
        goal = t
    cells = [goal]
    while cells[-1] != s:
        cells.append(parent[cells[-1]])
    cells.reverse()
    pts = [start_xy] + [grid.center(*c) for c in cells[1:]]
    if len(pts) == 1:
        pts.append(grid.center(*goal))  # already standing in the goal cell
    raw = [float(costs[c]) for c in cells]
    raw += [raw[-1]] * (len(pts) - len(raw))
    # string-pulling: from each anchor, the farthest raw point reachable in one straight leg
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        best, raw_max = i + 1, raw[i]
        for j in range(i + 1, len(pts)):
            raw_max = max(raw_max, raw[j])
            if segment_max_cost(grid, costs, pts[i], pts[j]) <= max(walkable_cost, raw_max) + 1e-9:
                best = j
        out.append(pts[best])
        i = best
    max_cost = max(segment_max_cost(grid, costs, a, b) for a, b in zip(out, out[1:]))
    return WalkPath(out, out[-1], target_xy, max_cost)


def project_on_path(points: list[tuple[float, float]], p: tuple[float, float], from_seg: int = 0) -> tuple[int, float, float]:
    """(segment index, arc length along the path, distance off the path) of the path point closest to p,
    looking only at segments from `from_seg` on (progress along the path never goes backwards)."""
    best = None
    s0 = sum(math.dist(a, b) for a, b in zip(points[:from_seg + 1], points[1:from_seg + 1]))
    for k in range(from_seg, len(points) - 1):
        (ax, ay), (bx, by) = points[k], points[k + 1]
        vx, vy = bx - ax, by - ay
        seg2 = vx * vx + vy * vy
        u = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * vx + (p[1] - ay) * vy) / seg2))
        d = math.hypot(ax + u * vx - p[0], ay + u * vy - p[1])
        if best is None or d < best[2] - 1e-9:
            best = (k, s0 + u * math.sqrt(seg2), d)
        s0 += math.sqrt(seg2)
    return best if best is not None else (0, 0.0, math.dist(points[0], p))


def point_along(points: list[tuple[float, float]], s: float) -> tuple[float, float]:
    """The point at arc length s along the path (clamped to its ends)."""
    for a, b in zip(points, points[1:]):
        L = math.dist(a, b)
        if s <= L:
            u = 0.0 if L <= 1e-9 else max(0.0, s / L)
            return a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1])
        s -= L
    return points[-1]


def steer(points: list[tuple[float, float]], p: tuple[float, float], heading_deg: float, from_seg: int = 0,
          lookahead_m: float = 0.8, final_xy: tuple[float, float] | None = None) -> tuple[float, int, float]:
    """(heading error in deg, + = the path goes to their RIGHT; segment index; distance off the path).
    Aims lookahead_m along the path; once that passes the path's end, at final_xy (the target) if given."""
    k, s, off = project_on_path(points, p, from_seg)
    total = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    ax, ay = final_xy if (final_xy is not None and s + lookahead_m >= total) else point_along(points, s + lookahead_m)
    if math.hypot(ax - p[0], ay - p[1]) < 1e-6:
        ax, ay = points[-1]
    desired = math.degrees(math.atan2(ay - p[1], ax - p[0]))
    return wrap_deg(desired - heading_deg), k, off


def cue_from_error(err_deg: float, last_cue: int | None, forward_deg: float = 15.0, forward_exit_deg: float = 30.0) -> int:
    """-1 = turn left, 0 = forward, +1 = turn right. Forward starts below forward_deg and is kept until the
    error exceeds forward_exit_deg; a turn already under way is kept when the path is nearly behind them
    (so +-180 deg noise cannot flip it)."""
    a = abs(err_deg)
    if a <= forward_deg or (last_cue == 0 and a <= forward_exit_deg):
        return 0
    if last_cue in (-1, 1) and a > 150.0:
        return last_cue
    return 1 if err_deg > 0 else -1
