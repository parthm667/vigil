"""Simulated room: walls, floor, coloured furniture boxes, the dummy target, and a walking person.

Frames as in types.py: x forward, y right, z up; headings clockwise positive (world = sim ground truth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .person_model import SimPerson


@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    color: tuple[int, int, int] = (90, 110, 140)  # BGR
    cls: str = ""  # detector class name for the oracle ("" = not detectable, e.g. a wall pillar)
    solid: bool = True  # the drone collides with it

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple(0.5 * (a + b) for a, b in zip(self.lo, self.hi))

    def contains(self, p, margin: float = 0.0) -> bool:
        return all(self.lo[i] - margin <= p[i] <= self.hi[i] + margin for i in range(3))


@dataclass
class PersonScript:
    """Piecewise-linear walk: list of (t, x, y, heading_deg). Holds the last entry afterwards."""

    keys: list[tuple[float, float, float, float]]

    def at(self, t: float) -> tuple[float, float, float]:
        k = self.keys
        if t <= k[0][0]:
            return k[0][1:]
        for (t0, x0, y0, h0), (t1, x1, y1, h1) in zip(k, k[1:]):
            if t0 <= t <= t1:
                a = (t - t0) / max(t1 - t0, 1e-9)
                dh = (h1 - h0 + 180) % 360 - 180
                return x0 + a * (x1 - x0), y0 + a * (y1 - y0), h0 + a * dh
        return k[-1][1:]


@dataclass
class World:
    size_x: float = 7.0  # room spans x in [0, size_x], y in [0, size_y]
    size_y: float = 6.0
    height: float = 2.6
    boxes: list[Box] = field(default_factory=list)
    person: SimPerson | None = None
    person_script: PersonScript | None = None

    def update(self, t: float) -> None:
        if self.person is not None and self.person_script is not None:
            self.person.x, self.person.y, self.person.heading_deg = self.person_script.at(t)

    def person_boxes(self) -> list[Box]:
        """The person as three boxes (legs, torso, head) for rendering and collisions. Blue jeans (the same
        hue as the dummy: the person guard must drop them), a grey-green top."""
        p = self.person
        if p is None:
            return []
        s = p.height_m / 1.75
        out = []
        for (hw, z0, z1, col) in ((0.13, 0.0, 0.9, (90, 50, 40)), (0.22, 0.9, 1.5, (80, 110, 90)), (0.10, 1.5, 1.75, (120, 150, 190))):
            hw_s = hw * s
            out.append(Box((p.x - hw_s, p.y - hw_s, z0 * s), (p.x + hw_s, p.y + hw_s, z1 * s), col, "person"))
        return out

    def all_boxes(self) -> list[Box]:
        return self.boxes + self.person_boxes()

    def floor_below(self, x: float, y: float, z: float) -> float:
        """Height of the surface under (x, y) below z (floor or a box top)."""
        top = 0.0
        for b in self.all_boxes():
            if b.lo[0] <= x <= b.hi[0] and b.lo[1] <= y <= b.hi[1] and b.hi[2] <= z:
                top = max(top, b.hi[2])
        return top

    def collides(self, p, radius: float = 0.12) -> bool:
        x, y, z = p
        if not (radius <= x <= self.size_x - radius and radius <= y <= self.size_y - radius and 0.0 <= z <= self.height - 0.05):
            return True
        return any(b.solid and b.contains(p, radius) for b in self.all_boxes())

    def ray_distance(self, x: float, y: float, z: float, heading_deg: float, max_m: float = 10.0) -> float:
        """Horizontal distance from (x, y, z) along a heading to the first wall or solid box."""
        h = math.radians(heading_deg)
        o = np.array([[x, y, z]])
        d = np.array([[math.cos(h), math.sin(h), 0.0]])
        best = max_m
        for axis, val in ((0, 0.0), (0, self.size_x), (1, 0.0), (1, self.size_y)):
            if abs(d[0, axis]) > 1e-12:
                t = (val - o[0, axis]) / d[0, axis]
                if 1e-6 < t < best:
                    best = t
        for b in self.all_boxes():
            if not b.solid:
                continue
            t, _ = ray_box(o, d, b.lo, b.hi)
            if t[0] < best:
                best = float(t[0])
        return best

    def line_of_sight(self, a, b, ignore: list[Box] | None = None, include_person: bool = True) -> bool:
        """True if the segment a->b does not pass through any box (other than `ignore`; the person's
        own boxes are skipped with include_person=False, since they are rebuilt on every call)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        for bx in (self.all_boxes() if include_person else self.boxes):
            if ignore and any(bx is i for i in ignore):
                continue
            t0, t1 = 0.0, 1.0
            ok = True
            for i in range(3):
                if abs(d[i]) < 1e-12:
                    if not (bx.lo[i] <= a[i] <= bx.hi[i]):
                        ok = False
                        break
                else:
                    ta, tb = (bx.lo[i] - a[i]) / d[i], (bx.hi[i] - a[i]) / d[i]
                    t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
                    if t0 > t1:
                        ok = False
                        break
            if ok and t1 > 1e-6 and t0 < 1 - 1e-6:
                return False
        return True


def dummy_bottle(x: float, y: float, base_z: float, height: float = 0.24, width: float = 0.09,
                 color=(167, 92, 52)) -> list[Box]:
    """The team's blue water bottle, all blue (BGR measured from a photo) and as tall as the real one with its
    cap, so the simulator's colour detector sees the same box a YOLO model sees on the drone: a '+'-shaped prism whose apparent width is ~constant from every direction (like a
    cylinder; a square box would look up to 41 % wider at 45 deg)."""
    a, b = width / 2, width * 0.41 / 2
    return [Box((x - a, y - b, base_z), (x + a, y + b, base_z + height), color, "bottle", solid=False),
            Box((x - b, y - a, base_z), (x + b, y + a, base_z + height), color, "bottle", solid=False)]


def demo_world(target_xy: tuple[float, float] = (6.2, 5.0), on_table: bool = True, seed: int = 0) -> World:
    """7 x 6 m living room: a table with the blue dummy 'bottle', chairs, a couch, a shelf.

    The person starts near (1.8, 1.5) facing +x; the drone is meant to start behind them.
    """
    tx, ty = target_xy
    boxes = [
        Box((5.3, 4.2, 0.0), (6.9, 5.9, 0.75), (60, 90, 120), "dining table"),  # table in the far corner
        Box((4.8, 4.5, 0.0), (5.2, 4.9, 0.9), (70, 120, 160), "chair"),
        Box((5.6, 3.7, 0.0), (6.0, 4.1, 0.9), (70, 120, 160), "chair"),
        Box((0.1, 4.6, 0.0), (2.2, 5.9, 0.85), (100, 140, 90), "couch"),  # couch along the left... (+y wall)
        Box((6.6, 0.2, 0.0), (6.95, 1.6, 1.8), (150, 150, 150), ""),  # shelf (not a detector class)
        Box((3.2, 2.6, 0.0), (3.7, 3.1, 0.45), (40, 170, 200), ""),  # a low pouf
    ]
    base = 0.75 if on_table else 0.0
    boxes += dummy_bottle(tx, ty, base)
    person = SimPerson(1.8, 1.5, 0.0)
    return World(boxes=boxes, person=person)


def ray_box(o: np.ndarray, d: np.ndarray, lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Slab test for many rays: returns (t_hit (inf if none), face_axis)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (np.asarray(lo) - o) * inv
        t2 = (np.asarray(hi) - o) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tmin = np.where(np.isnan(tmin), -np.inf, tmin)
    tmax = np.where(np.isnan(tmax), np.inf, tmax)
    t_near = tmin.max(axis=1)
    t_far = tmax.min(axis=1)
    hit = (t_far >= t_near) & (t_far > 1e-6)
    t = np.where(hit, np.where(t_near > 1e-6, t_near, np.inf), np.inf)
    axis = tmin.argmax(axis=1)
    return t, axis


class Renderer:
    """Ray-cast BGR images of the world from a camera pose (numpy, ~20-40 ms at 480 x 360)."""

    def __init__(self, cam, seed: int = 0):
        self.cam = cam
        xs = (np.arange(cam.width) + 0.5 - cam.cx) / cam.fx
        ys = -(np.arange(cam.height) + 0.5 - cam.cy) / cam.fy
        self._r, self._u = np.meshgrid(xs, ys)  # camera-frame right / up per pixel (forward = 1)
        rng = np.random.default_rng(seed)
        self._noise = rng.normal(0, 3.0, (cam.height, cam.width, 1))

    def render(self, world: World, x: float, y: float, z: float, heading_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
        h, p = math.radians(heading_deg), math.radians(pitch_deg)
        fwd = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])
        right = np.array([-math.sin(h), math.cos(h), 0.0])
        up = np.array([-math.sin(p) * math.cos(h), -math.sin(p) * math.sin(h), math.cos(p)])
        d = fwd[None, None, :] + self._r[..., None] * right[None, None, :] + self._u[..., None] * up[None, None, :]
        d = d.reshape(-1, 3)
        o = np.array([x, y, z])
        n = d.shape[0]
        t_best = np.full(n, np.inf)
        color = np.zeros((n, 3))
        # floor / ceiling / walls as planes
        with np.errstate(divide="ignore", invalid="ignore"):
            planes = [
                (2, 0.0, "floor"), (2, world.height, "ceiling"),
                (0, 0.0, "wall"), (0, world.size_x, "wall"), (1, 0.0, "wall"), (1, world.size_y, "wall"),
            ]
            for axis, val, kind in planes:
                t = (val - o[axis]) / d[:, axis]
                t = np.where((t > 1e-6) & np.isfinite(t), t, np.inf)
                m = t < t_best
                if not m.any():
                    continue
                pts = o + d[m] * t[m, None]
                if kind == "floor":
                    chk = ((np.floor(pts[:, 0] / 0.5) + np.floor(pts[:, 1] / 0.5)) % 2)[:, None]
                    c = np.where(chk > 0, [[150, 145, 140]], [[115, 110, 105]])
                elif kind == "ceiling":
                    c = np.tile([[225, 225, 225]], (m.sum(), 1))
                else:
                    along = pts[:, 1] if axis == 0 else pts[:, 0]
                    band = ((np.floor(along / 0.4) % 2) * 18)[:, None]
                    skirting = (pts[:, 2] < 0.1)[:, None] * -40
                    c = np.array([[185, 195, 200]]) - band + skirting
                color[m] = c
                t_best[m] = t[m]
        for b in world.all_boxes():
            t, axis = ray_box(o, d, b.lo, b.hi)
            m = t < t_best
            if not m.any():
                continue
            shade = np.array([0.8, 0.9, 1.0])[axis[m]][:, None]  # face orientation shading
            color[m] = np.array(b.color)[None, :] * shade
            t_best[m] = t[m]
        img = color.reshape(self.cam.height, self.cam.width, 3) + self._noise
        return np.clip(img, 0, 255).astype(np.uint8)
