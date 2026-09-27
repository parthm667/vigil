"""Simulated Tello + room + user: a drop-in replacement for tello_io, so the whole runtime runs with no hardware.

    python -m flyfollow.runtime.sim_world --scenario follow|find|full [--seed N] [--det synthetic|none] [--speed 1.0]
                                          [--frames] [--auto-takeoff] [--no-script]

Same inputs as tello_io (mode, rc with the same RC_OWNER arbitration and timeout, tello_cmd, kill) and the same
outputs (tello_state, rc_sent, tello_ack, tello_event, and with --frames: frame + FrameRing), plus:
- det (--det synthetic, default): boxes from ground truth through the flyfollow.sim.camera parameters and rules
  (CameraParams noise, iid dropout and bursts, (h / h_full)^2 detection probability, capture-to-box latency; the head
  box uses camera.project like PursuitEnv). Classes: person + person_head (user track_id 1), bottle, cup, backpack,
  and furniture: dining table, desk, counter, couch, chair, tv. --det none: an external detector reads --frames.
- sim_truth (10 Hz): drone, user, object and furniture poses (world frame x, y on the floor, z up; room 7 x 6 m).

Dynamics: flyfollow.sim.drone_model.DroneModel with configs/env.yaml profiles.demo (the measured Tello). Takeoff,
land and moves are scripted with timings from the 2026-09-26 lag test (takeoff "ok" after about 8 s at about 0.85 m,
"forward 50" about 2.2 s, land about 2.8 s). State signs match the real Tello: forward flight gives negative vgx and
pitch, right gives negative vgy, climbing gives negative vgz; yaw grows clockwise; h_cm in 10 cm steps.

Scenarios: follow (drone on the floor 1.8 m behind the user; the user walks an 8 m path with two turns and stops once
the drone has flown 3 s), find (user stands still, bottle on the dining table behind the drone; other seeds randomize
the bottle surface and the drone heading), full (the follow walk, then the user stops and a scripted operator sends
mode_cmd FIND bottle, disable with --no-script).
"""

from __future__ import annotations

import argparse
import collections
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from flyfollow.interfaces import DT, IMG_H, IMG_W
from flyfollow.rl.env import load_env_config, profile_ranges
from flyfollow.runtime.drone_common import (
    HOVER,
    RcArbiter,
    SafetyLimits,
    parse_move,
    safety_reason,
    state_fields,
    takeoff_block_reason,
)
from flyfollow.runtime.messages import RC_HZ, SETTINGS_DEFAULTS, msg
from flyfollow.runtime.node import Rate, log, stop_event
from flyfollow.sim.camera import CameraParams, project
from flyfollow.sim.drone_model import DroneModel, DroneParams

NAME = "sim_world"
TOPICS_IN = ["mode", "rc", "tello_cmd", "kill", "tello_state"]
ROOM_W, ROOM_D, ROOM_H = 7.0, 6.0, 2.6
DRONE_R = 0.15
FX, FY = SETTINGS_DEFAULTS["fx"], SETTINGS_DEFAULTS["fy"]
CX0, CY0 = IMG_W / 2, IMG_H / 2
NEAR = 0.05
USER_ID = 1


# ------------------------------------------------------------------------------------------------ world
@dataclass
class Box:
    """Axis-aligned box resting at z0 (world meters). Furniture and objects are unions of parts for rendering."""

    cx: float
    cy: float
    z0: float
    sx: float
    sy: float
    sz: float
    color: tuple = (160, 160, 160)

    def lo(self) -> np.ndarray:
        return np.array([self.cx - self.sx / 2, self.cy - self.sy / 2, self.z0])

    def hi(self) -> np.ndarray:
        return np.array([self.cx + self.sx / 2, self.cy + self.sy / 2, self.z0 + self.sz])

    def corners(self) -> np.ndarray:
        lo, hi = self.lo(), self.hi()
        return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


@dataclass
class Item:
    name: str
    cls: str
    track_id: int
    extent: Box  # for detection boxes, occlusion and collisions
    parts: list[Box] = field(default_factory=list)  # for rendering (defaults to the extent)
    kind: str = "furniture"

    def render_parts(self) -> list[Box]:
        return self.parts or [self.extent]

    def truth(self) -> dict:
        e = self.extent
        return {"name": self.name, "cls": self.cls, "id": self.track_id, "x": round(e.cx, 3), "y": round(e.cy, 3),
                "z": round(e.z0, 3), "size": [e.sx, e.sy, e.sz]}


def _table(cx, cy, sx, sy, h, color, leg=0.05, top=0.05) -> list[Box]:
    out = [Box(cx, cy, h - top, sx, sy, top, color)]
    for dx in (-1, 1):
        for dy in (-1, 1):
            out.append(Box(cx + dx * (sx / 2 - leg), cy + dy * (sy / 2 - leg), 0.0, leg, leg, h - top, color))
    return out


def build_furniture() -> list[Item]:
    brown, oak, grey, blue, black = (120, 80, 50), (170, 130, 90), (190, 190, 185), (70, 90, 140), (25, 25, 30)
    return [
        Item("table", "dining table", 101, Box(5.3, 4.3, 0.0, 1.2, 0.8, 0.75), _table(5.3, 4.3, 1.2, 0.8, 0.75, brown)),
        Item("desk", "desk", 102, Box(0.9, 5.4, 0.0, 1.2, 0.6, 0.75),
             [Box(0.9, 5.4, 0.72, 1.2, 0.6, 0.03, oak), Box(0.33, 5.4, 0.0, 0.04, 0.6, 0.72, oak),
              Box(1.47, 5.4, 0.0, 0.04, 0.6, 0.72, oak)]),
        Item("counter", "counter", 103, Box(6.65, 1.6, 0.0, 0.6, 2.4, 0.9),
             [Box(6.65, 1.6, 0.0, 0.6, 2.4, 0.86, (230, 230, 225)), Box(6.65, 1.6, 0.86, 0.62, 2.42, 0.04, grey)]),
        Item("couch", "couch", 104, Box(1.4, 0.5, 0.0, 2.0, 0.9, 0.8),
             [Box(1.4, 0.55, 0.0, 1.7, 0.8, 0.42, blue), Box(1.4, 0.15, 0.0, 2.0, 0.2, 0.8, blue),
              Box(0.5, 0.55, 0.0, 0.2, 0.8, 0.6, blue), Box(2.3, 0.55, 0.0, 0.2, 0.8, 0.6, blue)]),
        Item("chair", "chair", 105, Box(4.45, 4.3, 0.0, 0.45, 0.45, 0.9),
             [Box(4.45, 4.3, 0.42, 0.45, 0.45, 0.05, oak), Box(4.25, 4.3, 0.0, 0.05, 0.45, 0.9, oak)]
             + [Box(4.45 + dx * 0.2, 4.3 + dy * 0.2, 0.0, 0.04, 0.04, 0.42, oak) for dx in (-1, 1) for dy in (-1, 1)]),
        Item("tv", "tv", 106, Box(3.5, 5.95, 1.0, 1.1, 0.06, 0.62), [Box(3.5, 5.95, 1.0, 1.1, 0.06, 0.62, black)]),
    ]


OBJ_SPECS = {  # cls -> (sx, sy, sz, color)
    "bottle": (0.07, 0.07, 0.22, (80, 150, 230)),
    "cup": (0.08, 0.08, 0.10, (230, 60, 50)),
    "backpack": (0.32, 0.20, 0.45, (40, 110, 60)),
}


def make_object(cls: str, x: float, y: float, z0: float, tid: int) -> Item:
    sx, sy, sz, col = OBJ_SPECS[cls]
    parts = [Box(x, y, z0, sx, sy, sz, col)]
    if cls == "bottle":
        parts = [Box(x, y, z0, sx, sy, sz * 0.8, col), Box(x, y, z0 + sz * 0.8, sx * 0.45, sy * 0.45, sz * 0.2, (240, 240, 240))]
    return Item(cls, cls, tid, Box(x, y, z0, sx, sy, sz), parts, kind="object")


def ray_blocked(p0: np.ndarray, p1: np.ndarray, boxes: list[Box]) -> bool:
    """Segment p0 -> p1 passes through any box (slab test), ignoring the last 2 % near p1."""
    d = p1 - p0
    for b in boxes:
        lo, hi = b.lo(), b.hi()
        t0, t1 = 0.0, 0.98
        ok = True
        for k in range(3):
            if abs(d[k]) < 1e-9:
                if p0[k] < lo[k] or p0[k] > hi[k]:
                    ok = False
                    break
                continue
            a, c = (lo[k] - p0[k]) / d[k], (hi[k] - p0[k]) / d[k]
            if a > c:
                a, c = c, a
            t0, t1 = max(t0, a), min(t1, c)
            if t0 > t1:
                ok = False
                break
        if ok:
            return True
    return False


# ------------------------------------------------------------------------------------------------ user
class User:
    """Scripted walker. Script steps: ("wait_flying", s), ("pause", s), ("walk", x, y), ("ask_find", cls)."""

    def __init__(self, x: float, y: float, heading: float, script: list, height: float = 1.72,
                 speed: float = 0.5, head_d: float = 0.23):
        self.x, self.y, self.heading = x, y, heading
        self.height, self.speed, self.head_d = height, speed, head_d
        self.script = collections.deque(script)
        self.v = 0.0
        self._t_step = 0.0
        self.events: list[tuple] = []

    @property
    def moving(self) -> bool:
        return self.v > 0.05

    def step(self, dt: float, flying_s: float) -> None:
        self._t_step += dt
        if not self.script:
            self.v = 0.0
            return
        st = self.script[0]
        kind = st[0]
        if kind == "wait_flying":
            self.v = 0.0
            if flying_s >= st[1]:
                self._next()
        elif kind == "pause":
            self.v = max(0.0, self.v - 1.0 * dt)
            if self._t_step >= st[1]:
                self._next()
        elif kind == "ask_find":
            self.events.append(("ask_find", st[1]))
            self._next()
        elif kind == "walk":
            dx, dy = st[1] - self.x, st[2] - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                self._next()
                return
            err = (math.atan2(dy, dx) - self.heading + math.pi) % (2 * math.pi) - math.pi
            turn = max(-math.radians(150) * dt, min(math.radians(150) * dt, err))
            self.heading += turn
            v_tgt = self.speed if abs(err) < math.radians(30) else 0.0
            v_tgt = min(v_tgt, dist / 0.8)  # slow down on arrival
            self.v += max(-1.0 * dt, min(1.0 * dt, v_tgt - self.v))
            step = min(self.v * dt, dist)
            self.x += step * math.cos(self.heading)
            self.y += step * math.sin(self.heading)

    def _next(self) -> None:
        self.script.popleft()
        self._t_step = 0.0

    def parts(self) -> list[tuple[np.ndarray, tuple]]:
        """Body as oriented boxes: (8x3 corners, color). Head is separate (head_center)."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        out = []

        def ob(fwd, left, z0, lx, ly, lz, col):
            pts = []
            for a in (-lx / 2, lx / 2):
                for b in (-ly / 2, ly / 2):
                    for z in (z0, z0 + lz):
                        f, l_ = fwd + a, left + b
                        pts.append([self.x + f * c - l_ * s, self.y + f * s + l_ * c, z])
            out.append((np.array(pts), col))

        k = self.height / 1.72
        swing = 0.12 * math.sin(self._t_step * 6.0) if self.moving else 0.0
        jeans, shirt, skin = (45, 60, 100), (200, 70, 60), (225, 185, 150)
        ob(swing, 0.1, 0.0, 0.14, 0.14, 0.85 * k, jeans)
        ob(-swing, -0.1, 0.0, 0.14, 0.14, 0.85 * k, jeans)
        ob(0.0, 0.0, 0.85 * k, 0.24, 0.42, 0.6 * k, shirt)
        ob(-swing, 0.26, 0.8 * k, 0.09, 0.09, 0.62 * k, shirt)
        ob(swing, -0.26, 0.8 * k, 0.09, 0.09, 0.62 * k, shirt)
        ob(0.0, 0.0, 1.45 * k, 0.08, 0.08, 0.06 * k, skin)
        return out

    def head_center(self) -> np.ndarray:
        return np.array([self.x, self.y, self.height - self.head_d / 2])

    def truth(self) -> dict:
        return {"x": round(self.x, 3), "y": round(self.y, 3), "heading_deg": round(_wrap180(math.degrees(self.heading)), 1),
                "height_m": self.height, "head_z_m": round(self.height - self.head_d / 2, 3), "moving": self.moving,
                "script_left": len(self.script)}


# ------------------------------------------------------------------------------------------------ camera geometry
class Cam:
    """Drone camera: level body frame pitched nose down by `pitch` (rad), like flyfollow.sim.camera.project."""

    def __init__(self, x, y, z, psi, pitch=0.0, fx=FX, fy=FY):
        self.p = np.array([x, y, z])
        self.c, self.s = math.cos(psi), math.sin(psi)
        self.cp, self.sp = math.cos(pitch), math.sin(pitch)
        self.pitch, self.fx, self.fy = pitch, fx, fy

    def to_cam(self, pts: np.ndarray) -> np.ndarray:
        """World points (N,3) -> camera (forward, left, up) (N,3)."""
        d = np.asarray(pts, dtype=float) - self.p
        f0 = d[:, 0] * self.c + d[:, 1] * self.s
        l0 = -d[:, 0] * self.s + d[:, 1] * self.c
        u0 = d[:, 2]
        return np.stack([f0 * self.cp - u0 * self.sp, l0, f0 * self.sp + u0 * self.cp], axis=1)

    def pix(self, q: np.ndarray) -> np.ndarray:
        return np.stack([CX0 - self.fx * q[:, 1] / q[:, 0], CY0 - self.fy * q[:, 2] / q[:, 0]], axis=1)

    def bbox(self, pts: np.ndarray) -> tuple[list[float] | None, float]:
        """Image box of a point set (edges of its convex hull clipped at the near plane) and the visible area fraction."""
        q = self.to_cam(pts)
        front = q[q[:, 0] > NEAR]
        if len(front) == 0:
            return None, 0.0
        if len(front) < len(q):  # add near-plane crossings of every point pair (small sets only)
            extra = []
            back = q[q[:, 0] <= NEAR]
            for a in front:
                for b in back:
                    t = (a[0] - NEAR) / (a[0] - b[0])
                    extra.append(a + t * (b - a))
            front = np.vstack([front, np.array(extra)])
        uv = self.pix(front)
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        full = max(1e-6, (x2 - x1) * (y2 - y1))
        cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(float(IMG_W), x2), min(float(IMG_H), y2)
        if cx2 - cx1 < 1.0 or cy2 - cy1 < 1.0:
            return None, 0.0
        return [cx1, cy1, cx2, cy2], (cx2 - cx1) * (cy2 - cy1) / full

    def head_box(self, head_c: np.ndarray, head_d: float) -> list[float] | None:
        """Head box with flyfollow.sim.camera.project (the same vertical-target geometry as PursuitEnv)."""
        q = self.to_cam(head_c[None, :])[0]
        # project() takes the drone-level frame and applies the pitch itself
        d = head_c - self.p
        fwd = d[0] * self.c + d[1] * self.s
        left = -d[0] * self.s + d[1] * self.c
        g = project(fwd, left, d[2], head_d, self.pitch, self.fx, self.fy, CX0, CY0)
        if g is None or q[0] <= NEAR:
            return None
        u, vt, vb = g
        h = vb - vt
        w = 0.85 * h
        x1, x2, y1, y2 = u - w / 2, u + w / 2, vt, vb
        if x2 < 0 or x1 > IMG_W or y2 < 0 or y1 > IMG_H:
            return None
        return [max(0.0, x1), max(0.0, y1), min(float(IMG_W), x2), min(float(IMG_H), y2)]


# ------------------------------------------------------------------------------------------------ renderer
BOX_FACES = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))  # x-, x+, y-, y+, z-, z+
FACE_SHADE = (0.78, 0.9, 0.84, 0.72, 0.5, 1.0)


def _clip_near(q: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a camera-frame polygon to forward >= NEAR."""
    out = []
    n = len(q)
    for i in range(n):
        a, b = q[i], q[(i + 1) % n]
        ia, ib = a[0] >= NEAR, b[0] >= NEAR
        if ia:
            out.append(a)
        if ia != ib:
            t = (a[0] - NEAR) / (a[0] - b[0])
            out.append(a + t * (b - a))
    return np.array(out)


def render(cam: Cam, furniture: list[Item], objects: list[Item], user: User | None) -> np.ndarray:
    """Flat-shaded painter's-algorithm view, 960x720 RGB uint8 (cv2 only for polygon fill)."""
    import cv2

    img = np.empty((IMG_H, IMG_W, 3), dtype=np.uint8)
    img[:] = (235, 235, 240)  # ceiling
    polys: list[tuple[float, np.ndarray, tuple]] = []

    def add(pts3: np.ndarray, color, shade=1.0, depth_bias=0.0):
        q = _clip_near(cam.to_cam(pts3))
        if len(q) < 3:
            return
        uv = cam.pix(q)
        if (uv[:, 0].max() < 0 or uv[:, 0].min() > IMG_W or uv[:, 1].max() < 0 or uv[:, 1].min() > IMG_H):
            return
        col = tuple(int(min(255, c * shade)) for c in color)
        polys.append((float(np.mean(q[:, 0])) + depth_bias, uv, col))

    W, D, H = ROOM_W, ROOM_D, ROOM_H
    add(np.array([[0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0]]), (150, 130, 110), depth_bias=1e3)  # floor, always first
    walls = [((0, 0), (W, 0), (205, 200, 185)), ((W, 0), (W, D), (190, 200, 205)), ((W, D), (0, D), (205, 195, 180)),
             ((0, D), (0, 0), (195, 205, 190))]
    for (x0, y0), (x1, y1), col in walls:
        add(np.array([[x0, y0, 0], [x1, y1, 0], [x1, y1, H], [x0, y0, H]]), col, depth_bias=1e3 - 1)
    for it in furniture + objects:
        for b in it.render_parts():
            cs = b.corners()
            for f, sh in zip(BOX_FACES, FACE_SHADE):
                add(cs[list(f)], b.color, sh)
    head = None
    if user is not None:
        for cs, col in user.parts():
            for f, sh in zip(BOX_FACES, FACE_SHADE):
                add(cs[list(f)], col, sh)
        hc = user.head_center()
        q = cam.to_cam(hc[None, :])[0]
        if q[0] > NEAR:
            uv = cam.pix(q[None, :])[0]
            head = (float(q[0]), uv, cam.fy * user.head_d / 2 / q[0])
    polys.sort(key=lambda p: -p[0])
    head_done = head is None
    for depth, uv, col in polys:
        if not head_done and depth < 1e2 and depth < head[0]:
            _draw_head(img, head, cv2)
            head_done = True
        cv2.fillPoly(img, [np.round(uv).astype(np.int32).clip(-10000, 10000)], col, lineType=cv2.LINE_AA)
    if not head_done:
        _draw_head(img, head, cv2)
    return img


def _draw_head(img, head, cv2) -> None:
    _, uv, r = head
    c = (int(round(uv[0])), int(round(uv[1])))
    r = int(max(1, min(4000, round(r))))
    cv2.circle(img, c, r, (225, 185, 150), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (c[0], c[1] - r // 3), (r, int(r * 0.75)), 0, 180, 360, (60, 40, 30), -1, lineType=cv2.LINE_AA)


# ------------------------------------------------------------------------------------------------ synthetic detector
class SynthDetector:
    """Ground truth -> det boxes with the flyfollow.sim.camera noise, dropout, burst, detection-probability and latency rules."""

    def __init__(self, cp: CameraParams, rng: np.random.Generator, small_full_px: float = 40.0):
        self.p, self.rng = cp, rng
        self.small_full_px = small_full_px  # COCO small objects need more pixels than a person
        self._burst_until = -1.0
        self._next_burst = self._draw_gap(0.0)

    def _draw_gap(self, t: float) -> float:
        return t + self.rng.exponential(1.0 / self.p.burst_rate_hz) if self.p.burst_rate_hz > 0 else float("inf")

    def _noisy(self, box: list[float], full_px: float, p_base: float) -> list[float] | None:
        x1, y1, x2, y2 = box
        h, w = y2 - y1, x2 - x1
        if self.rng.random() < self.p.dropout:
            return None
        hf = full_px
        if min(h, w * 2) < hf and self.rng.random() > (min(h, w * 2) / hf) ** 2:
            return None
        if self.rng.random() > p_base:
            return None
        k = 1.0 + self.rng.normal(0.0, self.p.h_noise_frac)
        cx = (x1 + x2) / 2 + self.rng.normal(0.0, self.p.center_sigma_px)
        cy = (y1 + y2) / 2 + self.rng.normal(0.0, self.p.center_sigma_px)
        hw, hh = max(0.5, w * k / 2), max(0.5, h * k / 2)
        return [round(max(0.0, cx - hw), 1), round(max(0.0, cy - hh), 1), round(min(IMG_W, cx + hw), 1), round(min(IMG_H, cy + hh), 1)]

    def detect(self, t: float, cam: Cam, user: User | None, furniture: list[Item], objects: list[Item]) -> list[dict]:
        if t >= self._next_burst:
            lo, hi = self.p.burst_len_s
            self._burst_until = t + self.rng.uniform(lo, hi)
            self._next_burst = self._draw_gap(self._burst_until)
        in_burst = t < self._burst_until
        blockers = [f.extent for f in furniture]
        out = []

        def conf(h):
            return round(float(np.clip(0.35 + 0.55 * min(1.0, h / 150.0) + self.rng.normal(0, 0.05), 0.26, 0.97)), 3)

        if user is not None:
            head_c = user.head_center()
            pts = np.vstack([cs for cs, _ in user.parts()] + [head_c + [0, 0, user.head_d / 2]])
            box, frac = cam.bbox(pts)
            head_vis = not ray_blocked(cam.p, head_c, blockers)
            if box is not None and frac > 0.2 and head_vis and not in_burst:
                b = self._noisy(box, self.p.h_full_det_px, 0.97)
                if b is not None:
                    out.append({"cls": "person", "conf": conf(b[3] - b[1]), "bbox": b, "track_id": USER_ID})
                    hb = cam.head_box(head_c, user.head_d)
                    if hb is not None:
                        bh = self._noisy(hb, self.p.h_full_det_px, 0.92)
                        if bh is not None:
                            out.append({"cls": "person_head", "conf": conf(3 * (bh[3] - bh[1])), "bbox": bh, "track_id": USER_ID})
        for it in objects + furniture:
            e = it.extent
            box, frac = cam.bbox(e.corners())
            if box is None or frac < 0.3:
                continue
            center = np.array([e.cx, e.cy, e.z0 + e.sz / 2])
            if it.kind == "object" and ray_blocked(cam.p, center, blockers):
                continue
            small = it.kind == "object" and it.cls != "backpack"
            b = self._noisy(box, self.small_full_px if small else self.p.h_full_det_px, 0.85 if small else 0.9)
            if b is not None:
                out.append({"cls": it.cls, "conf": conf(b[3] - b[1]), "bbox": b, "track_id": it.track_id})
        return out


# ------------------------------------------------------------------------------------------------ scenarios
WALK = [("walk", 5.2, 3.0), ("pause", 2.0), ("walk", 5.2, 1.5), ("pause", 3.0), ("walk", 3.0, 1.7), ("pause", 1.5),
        ("walk", 2.6, 3.2)]


def scenario(name: str, seed: int) -> dict:
    """Start poses and objects. Drone psi is counter-clockwise from +x (DroneModel convention)."""
    rng = np.random.default_rng(seed)
    objs = [make_object("bottle", 5.55, 4.4, 0.75, 11), make_object("cup", 1.25, 5.45, 0.75, 12),
            make_object("backpack", 3.3, 0.45, 0.0, 13)]
    if name in ("follow", "full"):
        script = [("wait_flying", 3.0)] + WALK
        if name == "full":
            script += [("pause", 4.0), ("ask_find", "bottle")]
        if seed:
            script = [(s[0], s[1] + rng.uniform(-0.2, 0.2), s[2] + rng.uniform(-0.2, 0.2)) if s[0] == "walk" else s for s in script]
        return {"user": (2.6, 3.0, 0.0, script), "drone": (0.8, 3.0, 0.0), "objects": objs}
    if name == "find":
        psi = math.pi  # facing the user, the table behind
        if seed:
            surf = rng.integers(3)
            spots = [(5.3 + rng.uniform(-0.45, 0.45), 4.3 + rng.uniform(-0.25, 0.25), 0.75),
                     (0.9 + rng.uniform(-0.45, 0.2), 5.4 + rng.uniform(-0.15, 0.15), 0.75),
                     (6.6 + rng.uniform(-0.1, 0.1), 1.6 + rng.uniform(-1.0, 1.0), 0.9)]
            objs[0] = make_object("bottle", *spots[surf], 11)
            psi = rng.uniform(-math.pi, math.pi)
        return {"user": (1.4, 3.0, 0.0, []), "drone": (3.2, 3.0, psi), "objects": objs}
    raise ValueError(f"unknown scenario {name!r}")


def demo_drone_params(cfg: dict | None = None) -> DroneParams:
    """configs/env.yaml profiles.demo (range entries at their midpoint), as DroneParams."""
    rp = profile_ranges(cfg or load_env_config(), "demo")
    kw = {}
    for k in DroneParams.__dataclass_fields__:
        if k in rp:
            v = rp[k]
            kw[k] = float(np.mean(v["choice"])) if isinstance(v, dict) else float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
    kw["drift_sigma_mps"] = min(kw.get("drift_sigma_mps", 0.0), 0.02)
    return DroneParams(**kw)


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------------------ the sim
class SimWorld:
    """tick(now) advances one DT of sim time. Everything a Tello would publish goes out through pub."""

    def __init__(self, pub, sub, *, scenario_name: str = "follow", seed: int = 0, det: str = "synthetic",
                 frames: bool = False, ring=None, battery: float = 90.0, auto_takeoff: bool = False,
                 script_ops: bool = True, limits: SafetyLimits | None = None, cam_params: CameraParams | None = None,
                 speed: float = 1.0):
        self.pub, self.sub, self.ring = pub, sub, ring
        self.scenario_name, self.seed, self.det_mode, self.frames = scenario_name, seed, det, frames
        self.speed = speed
        self.lim = limits or SafetyLimits()
        self.rng = np.random.default_rng([seed, 7])
        sc = scenario(scenario_name, seed)
        ux, uy, uh, script = sc["user"]
        self.user = User(ux, uy, uh, script, height=SETTINGS_DEFAULTS["user_height_m"])
        self.furniture = build_furniture()
        self.objects: list[Item] = sc["objects"]
        dx, dy, dpsi = sc["drone"]
        self.dp = demo_drone_params()
        n_noise = 2 * int(3600 / DT)
        self.dm = DroneModel(self.dp, self.rng.standard_normal(n_noise).tolist())
        self.dm.reset(dx, dy, 0.0, dpsi)
        self.psi0 = dpsi
        self.cp = cam_params or CameraParams(rate_hz=15.0, latency_s=0.30, det_time_s=0.05, center_sigma_px=2.0,
                                             h_noise_frac=0.05, dropout=0.03, burst_rate_hz=0.05)
        self.detector = SynthDetector(self.cp, np.random.default_rng([seed, 8]))
        self.arb = RcArbiter()
        self.phase = "ground"  # ground | takeoff | flying | move | landing
        self._ph: dict = {}
        self._jobs: collections.deque[dict] = collections.deque()
        self._ids = itertools.count(1)
        self.t = 0.0
        self.n_tick = 0
        self.bat = float(battery)
        self.temph = 62.0
        self.flight_s = 0.0
        self.flying_since: float | None = None
        self.collisions = collections.Counter()
        self._touching: set[str] = set()
        self.sticks = HOVER
        self.rc_src = "hover"
        self._next_cap = 0.0
        self._frame_id = 0
        self._det_q: collections.deque = collections.deque()
        self._frame_q: collections.deque = collections.deque()
        self._bus_seen: float | None = None
        self._t_wall0 = None
        self._last_cmd_wall = None
        self.safety_fired: str | None = None
        self.auto_takeoff = auto_takeoff
        self.script_ops = script_ops
        self.acks: list[dict] = []
        self._now: float | None = None

    # ------------------------------------------------------------------ bus
    def _publish(self, m: dict) -> None:
        if self._now is not None:
            m["t"] = self._now  # the tick's clock (wall time in the live loop, the test clock in tests)
        self.pub.publish(m)

    def _ack(self, job: dict, ok: bool, detail: str) -> None:
        m = msg("tello_ack", id=job.get("id"), cmd=job.get("cmd"), ok=bool(ok), detail=str(detail),
                elapsed_s=round((self.t - job.get("t_start", self.t)) / max(self.speed, 1e-6), 3), dry_run=False, sim=True)
        self.acks.append(m)
        self._publish(m)

    def on_msg(self, m: dict, now: float) -> None:
        tp = m.get("topic")
        if m.get("replayed") and tp != "tello_state":
            return  # never act on a replayed command (messages.py)
        if tp == "rc":
            if self.arb.offer(m, now):
                self._last_cmd_wall = now
        elif tp == "mode":
            self.arb.set_mode(str(m.get("to")))
        elif tp == "kill":
            self.kill(str(m.get("action", "land")), f"kill from {m.get('src_node', '?')}")
        elif tp == "tello_cmd":
            self._last_cmd_wall = now
            self.submit(m)
        elif tp == "tello_state" and m.get("src_node") == getattr(self.pub, "name", NAME):
            self._bus_seen = now

    def submit(self, m: dict) -> None:
        cmd = str(m.get("cmd", ""))
        job = {"id": m.get("id", f"auto-{next(self._ids)}"), "cmd": cmd, "args": m.get("args") or {}}
        if cmd in ("land", "emergency"):
            self.kill(cmd, f"tello_cmd from {m.get('src_node', '?')}", job)
            return
        if cmd == "stop":
            for j in list(self._jobs):
                self._ack(j, False, "cancelled by stop")
            self._jobs.clear()
        self._jobs.append(job)

    def kill(self, action: str, reason: str, job: dict | None = None) -> None:
        action = "emergency" if action == "emergency" else "land"
        job = dict(job or {"id": f"kill-{next(self._ids)}", "args": {}})
        job["cmd"] = action
        for j in list(self._jobs):
            self._ack(j, False, f"preempted by {action} ({reason})")
        self._jobs.clear()
        cur = self._ph.get("job")
        if self.phase in ("move", "takeoff") and cur is not None:
            self._ack(cur, False, f"preempted by {action} ({reason})")
        self._publish(msg("tello_event", kind="kill", action=action, reason=reason, sim=True))
        job["t_start"] = self.t
        if action == "emergency":
            self.dm.z = 0.0
            self._set_ground()
            self._ack(job, True, "emergency: motors off")
        elif self.phase in ("flying", "move", "takeoff"):
            self._start_landing(job)
        elif self.phase == "landing":
            self._ack(job, True, "already landing")
        else:
            self._ack(job, True, "not flying")

    # ------------------------------------------------------------------ phases
    def _set_ground(self) -> None:
        self.phase = "ground"
        self._ph = {}
        self.dm.vx = self.dm.vy = self.dm.vz = 0.0
        self.dm.yaw_rate_dps = 0.0
        self.flying_since = None

    def _start_landing(self, job: dict) -> None:
        self.phase = "landing"
        self._ph = {"job": job, "t0": self.t}

    def _start_job(self, job: dict) -> None:
        job["t_start"] = self.t
        cmd = job["cmd"]
        if cmd == "takeoff":
            if self.phase != "ground":
                self._ack(job, True, "already flying")
                return
            why = takeoff_block_reason(self.lim, bat_pct=int(self.bat), temph_c=int(self.temph), video_enabled=False, video_ok=True)
            if why:
                self._ack(job, False, f"refused: {why}")
                return
            self.phase = "takeoff"
            self._ph = {"job": job, "t0": self.t, "z_target": 0.85 + self.rng.uniform(0.0, 0.2)}
            self.safety_fired = None
        elif cmd == "move":
            if self.phase != "flying":
                self._ack(job, False, "not flying")
                return
            sdk, err = parse_move(job["args"])
            if sdk is None:
                self._ack(job, False, err)
                return
            d, v = sdk.split()
            self.phase = "move"
            self._ph = {"job": job, "t0": self.t, "dir": d, "val": float(v), "done": 0.0, "sdk": sdk}
            self.dm.vx = self.dm.vy = self.dm.vz = 0.0
            self.dm.yaw_rate_dps = 0.0
        elif cmd == "stop":
            self.arb.clear()
            self._ack(job, True, "queue cleared, hovering")
        else:
            self._ack(job, False, f"unknown cmd {cmd!r}")

    def _phase_step(self) -> None:
        ph, dt, dm = self._ph, DT, self.dm
        el = self.t - ph.get("t0", self.t)
        if self.phase == "takeoff":
            if el < 2.0:
                return  # motor spin-up
            if dm.z < ph["z_target"]:
                dm.z = min(ph["z_target"], dm.z + 0.3 * dt)
                dm.vz = 0.3
                return
            dm.vz = 0.0
            if el >= 2.0 + ph["z_target"] / 0.3 + 2.5:  # settle; the real "ok" came about 8 s after the command
                job = ph["job"]
                self.phase, self._ph = "flying", {}
                x, y, z, psi = dm.x, dm.y, dm.z, dm.psi
                dm.reset(x, y, z, psi)
                self.arb.clear()
                self.flying_since = self.t
                self._ack(job, True, "ok")
        elif self.phase == "landing":
            if dm.z > 0.0:
                dm.z = max(0.0, dm.z - 0.4 * dt)
                dm.vz = -0.4
                ph["t_down"] = self.t
                return
            if self.t - ph.get("t_down", self.t) >= 0.3:
                job = ph["job"]
                self._set_ground()
                self._ack(job, True, "ok")
        elif self.phase == "move":
            d, val = ph["dir"], ph["val"]
            if el < 0.4:
                return
            if d in ("cw", "ccw"):
                rate = 60.0 * dt
                step = min(rate, val - ph["done"])
                ph["done"] += step
                dm.psi += math.radians(-step if d == "cw" else step)
                dm.yaw_rate_dps = 60.0 if d == "cw" else -60.0
            else:
                v = 0.5 if d not in ("up", "down") else 0.3
                step = min(v * dt, val / 100.0 - ph["done"])
                ph["done"] += step
                c, s = math.cos(dm.psi), math.sin(dm.psi)
                vec = {"forward": (c, s, 0), "back": (-c, -s, 0), "left": (-s, c, 0), "right": (s, -c, 0),
                       "up": (0, 0, 1), "down": (0, 0, -1)}[d]
                dm.x += vec[0] * step
                dm.y += vec[1] * step
                dm.z = max(0.2, dm.z + vec[2] * step)
                dm.vx, dm.vy, dm.vz = (vec[0] * v, vec[1] * v, vec[2] * v) if step > 0 else (0.0, 0.0, 0.0)
                if self._collide(count=True):
                    job = ph["job"]
                    self._end_move()
                    self._ack(job, False, "collision")
                    return
            full = val if d in ("cw", "ccw") else val / 100.0
            if ph["done"] >= full - 1e-9:
                dm.vx = dm.vy = dm.vz = 0.0
                dm.yaw_rate_dps = 0.0
                ph.setdefault("t_end", self.t)
                if self.t - ph["t_end"] >= 0.8:
                    job = ph["job"]
                    self._end_move()
                    self._ack(job, True, "ok")

    def _end_move(self) -> None:
        dm = self.dm
        dm.reset(dm.x, dm.y, dm.z, dm.psi)
        self.phase, self._ph = "flying", {}
        self.arb.clear()

    # ------------------------------------------------------------------ collisions
    def _collide(self, count: bool = True) -> bool:
        """Push the drone out of walls, furniture and the user; count new contacts. True if touching anything."""
        dm = self.dm
        touching = set()
        for k, lo, hi in ((0, 0.0, ROOM_W), (1, 0.0, ROOM_D)):
            p = dm.x if k == 0 else dm.y
            if p < lo + DRONE_R or p > hi - DRONE_R:
                touching.add("wall")
                p = min(max(p, lo + DRONE_R), hi - DRONE_R)
                if k == 0:
                    dm.x, dm.vx = p, 0.0
                else:
                    dm.y, dm.vy = p, 0.0
        if dm.z > ROOM_H - 0.1:
            dm.z, dm.vz = ROOM_H - 0.1, 0.0
            touching.add("ceiling")
        for it in self.furniture:
            lo, hi = it.extent.lo(), it.extent.hi()
            if (lo[0] - DRONE_R < dm.x < hi[0] + DRONE_R and lo[1] - DRONE_R < dm.y < hi[1] + DRONE_R and dm.z < hi[2] + 0.05
                    and dm.z > lo[2] - 0.05):
                touching.add("furniture")
                pen = {"x-": dm.x - (lo[0] - DRONE_R), "x+": hi[0] + DRONE_R - dm.x, "y-": dm.y - (lo[1] - DRONE_R),
                       "y+": hi[1] + DRONE_R - dm.y, "z+": hi[2] + 0.05 - dm.z}
                side = min(pen, key=pen.get)
                if side == "x-":
                    dm.x, dm.vx = lo[0] - DRONE_R, 0.0
                elif side == "x+":
                    dm.x, dm.vx = hi[0] + DRONE_R, 0.0
                elif side == "y-":
                    dm.y, dm.vy = lo[1] - DRONE_R, 0.0
                elif side == "y+":
                    dm.y, dm.vy = hi[1] + DRONE_R, 0.0
                else:
                    dm.z, dm.vz = hi[2] + 0.05, 0.0
        u = self.user
        dx, dy = dm.x - u.x, dm.y - u.y
        r = math.hypot(dx, dy)
        if r < 0.35 and dm.z < u.height + 0.1:
            touching.add("person")
            if r > 1e-6:
                dm.x, dm.y = u.x + dx / r * 0.35, u.y + dy / r * 0.35
            dm.vx = dm.vy = 0.0
        if count:
            for k in touching - self._touching:
                self.collisions[k] += 1
                self._publish(msg("tello_event", kind="collision", what=k, sim=True))
            self._touching = touching
        return bool(touching)

    # ------------------------------------------------------------------ state
    def raw_state(self) -> dict:
        """What a real Tello state packet would carry (djitellopy keys), with the real sign conventions."""
        dm = self.dm
        c, s = math.cos(dm.psi), math.sin(dm.psi)
        vx, vy = dm.vx + dm.dvx, dm.vy + dm.dvy
        v_fwd = vx * c + vy * s
        v_right = vx * s - vy * c
        below = 0.0
        for it in self.furniture:
            lo, hi = it.extent.lo(), it.extent.hi()
            if lo[0] <= dm.x <= hi[0] and lo[1] <= dm.y <= hi[1] and hi[2] < dm.z:
                below = max(below, hi[2])
        return {
            "pitch": int(round(-math.degrees(dm.pitch))), "roll": 0,
            "yaw": int(round(_wrap180(-math.degrees(dm.psi - self.psi0)))),
            "vgx": int(round(-v_fwd * 10)), "vgy": int(round(-v_right * 10)), "vgz": int(round(-dm.vz * 10)),
            "templ": int(self.temph) - 3, "temph": int(self.temph),
            "tof": int(round(max(0.1, dm.z - below) * 100)) if self.phase != "ground" else 10,
            "h": int(10 * round(max(0.0, dm.z - 0.15) * 10)), "bat": int(self.bat), "baro": round(100.0 + dm.z, 2),
            "time": int(self.flight_s), "agx": 0.0, "agy": 0.0, "agz": -1000.0,
        }

    @property
    def flying(self) -> bool:
        return self.phase in ("flying", "move") or (self.phase in ("takeoff", "landing") and self.dm.z > 0.0)

    def cam(self) -> Cam:
        dm = self.dm
        return Cam(dm.x, dm.y, max(dm.z, 0.05), dm.psi, dm.pitch)

    def truth(self) -> dict:
        dm = self.dm
        return msg("sim_truth", sim_t=round(self.t, 3), scenario=self.scenario_name, seed=self.seed, room=[ROOM_W, ROOM_D, ROOM_H],
                   drone={"x": round(dm.x, 3), "y": round(dm.y, 3), "z": round(dm.z, 3), "psi_deg": round(math.degrees(dm.psi), 2),
                          "yaw_deg": round(_wrap180(-math.degrees(dm.psi - self.psi0)), 2), "vx": round(dm.vx, 3),
                          "vy": round(dm.vy, 3), "vz": round(dm.vz, 3), "phase": self.phase, "flying": self.flying},
                   user=self.user.truth(), objects=[o.truth() for o in self.objects],
                   furniture=[f.truth() for f in self.furniture], collisions=dict(self.collisions),
                   rc=list(self.sticks), rc_src=self.rc_src, mode=self.arb.mode, cam={"fx": FX, "fy": FY, "cx": CX0, "cy": CY0})

    # ------------------------------------------------------------------ tick
    def tick(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._now = now
        if self._t_wall0 is None:
            self._t_wall0 = now
            self._last_cmd_wall = now
        for m in self.sub.drain():
            self.on_msg(m, now)
        if self.auto_takeoff and self.n_tick == int(1.0 / DT):
            self.submit(msg("tello_cmd", id="auto-takeoff", cmd="takeoff"))
        dt = DT
        self.t += dt
        self.n_tick += 1
        flying_s = self.t - self.flying_since if self.flying_since is not None else -1.0
        self.user.step(dt, flying_s)
        for ev in self.user.events:
            if ev[0] == "ask_find" and self.script_ops:
                self._publish(msg("mode_cmd", mode="FIND", source="sim_script",
                                  target={"cls": ev[1], "prompt": "water bottle", "height_m": OBJ_SPECS[ev[1]][2]}))
                log(NAME, f"scripted operator: find the {ev[1]}")
        self.user.events.clear()
        if self.phase in ("ground", "flying") and self._jobs:
            self._start_job(self._jobs.popleft())
        if self.phase == "flying":
            self.sticks, self.rc_src = self.arb.select(now)
            self.dm.step(*self.sticks)
            self._collide()
            self.dm.z = max(self.dm.z, 0.1)
            self._publish(msg("rc_sent", lr=self.sticks[0], fb=self.sticks[1], ud=self.sticks[2], yaw=self.sticks[3],
                              src=self.rc_src, dry_run=False, mode=self.arb.mode, sim=True))
        else:
            self.sticks, self.rc_src = HOVER, "hold"
            self._phase_step()
        # battery and heat
        if self.flying:
            self.flight_s += dt
            self.bat = max(0.0, self.bat - 0.18 * dt)
            self.temph += (72.0 - self.temph) * 0.01 * dt
        else:
            self.bat = max(0.0, self.bat - 0.01 * dt)
            self.temph += (88.0 - self.temph) * 0.004 * dt
        self._camera(now)
        if self.n_tick % 2 == 0:
            f = state_fields(self.raw_state())
            self._publish(msg("tello_state", **f, flying=self.flying, video_ok=True, video_age_s=0.05, sending=True,
                              state_ok=True, state_age_s=0.0, mode=self.arb.mode, busy=self.phase if self.phase not in ("ground", "flying") else None,
                              queued=len(self._jobs), safety=self.safety_fired, sim=True))
            self._publish(self.truth())
        self._safety(now)

    def _safety(self, now: float) -> None:
        if self.safety_fired or self.phase not in ("flying", "move"):
            return
        bus_age = (now - self._bus_seen) if self._bus_seen is not None else (now - self._t_wall0 if now - self._t_wall0 > 5.0 else None)
        why = safety_reason(self.lim, flying=True, bat_pct=int(self.bat), temph_c=int(self.temph), video_enabled=False,
                            video_age_s=0.0, bus_age_s=bus_age, idle_s=now - (self._last_cmd_wall or now))
        if why:
            self.safety_fired = why
            self.kill("land", f"safety: {why}")

    def _camera(self, now: float) -> None:
        cp = self.cp
        vid_lat = cp.latency_s - cp.det_time_s
        if self.t >= self._next_cap:
            self._next_cap = self.t + 1.0 / cp.rate_hz
            self._frame_id += 1
            cam = self.cam()
            if self.det_mode == "synthetic":
                dets = self.detector.detect(self.t, cam, self.user, self.furniture, self.objects)
                self._det_q.append((self.t + cp.latency_s, self._frame_id, dets))
            if self.frames:
                img = render(cam, self.furniture, self.objects, self.user)
                self._frame_q.append((self.t + vid_lat, self._frame_id, img))
        while self._frame_q and self._frame_q[0][0] <= self.t:
            _, fid, img = self._frame_q.popleft()
            slot = self.ring.write(img, fid, now) if self.ring is not None else -1
            self._publish(msg("frame", frame_id=fid, t_decoded=now, slot=slot, w=IMG_W, h=IMG_H, src="sim"))
        while self._det_q and self._det_q[0][0] <= self.t:
            _, fid, dets = self._det_q.popleft()
            self._publish(msg("det", frame_id=fid, t_decoded=now - cp.det_time_s / max(self.speed, 1e-6), src="sim",
                              img_w=IMG_W, img_h=IMG_H, imgsz=IMG_W, dets=dets, latency_s=cp.det_time_s))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Simulated Tello + room + user (drop-in for tello_io)")
    ap.add_argument("--scenario", choices=("follow", "find", "full"), default="follow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--det", choices=("synthetic", "none"), default="synthetic")
    ap.add_argument("--speed", type=float, default=1.0, help="sim seconds per wall second")
    ap.add_argument("--frames", action="store_true", help="render 960x720 frames into the FrameRing (needs cv2)")
    ap.add_argument("--battery", type=float, default=90.0)
    ap.add_argument("--auto-takeoff", action="store_true", help="take off 1 s after start (no mission needed)")
    ap.add_argument("--no-script", action="store_true", help="full: do not send the scripted FIND mode_cmd")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after this many sim seconds (0 = run until killed)")
    a = ap.parse_args(argv)

    from flyfollow.runtime.bus import FrameRing, Publisher, Subscriber

    stop = stop_event()
    pub = Publisher(NAME)
    sub = Subscriber(TOPICS_IN)
    ring = FrameRing.create(h=IMG_H, w=IMG_W) if a.frames else None
    if a.det == "none" and not a.frames:
        log(NAME, "warning: --det none without --frames: nothing will produce detections")
    sim = SimWorld(pub, sub, scenario_name=a.scenario, seed=a.seed, det=a.det, frames=a.frames, ring=ring,
                   battery=a.battery, auto_takeoff=a.auto_takeoff, script_ops=not a.no_script, speed=a.speed)
    log(NAME, f"scenario {a.scenario} seed {a.seed} det {a.det} frames {a.frames} speed {a.speed}")
    rate = Rate(RC_HZ * a.speed)
    t_log = time.time()
    try:
        while not stop.is_set():
            sim.tick()
            if a.duration and sim.t >= a.duration:
                break
            if time.time() - t_log > 5.0:
                t_log = time.time()
                d = sim.dm
                log(NAME, f"t {sim.t:.1f} mode {sim.arb.mode} phase {sim.phase} drone ({d.x:.2f}, {d.y:.2f}, {d.z:.2f}) "
                          f"user ({sim.user.x:.2f}, {sim.user.y:.2f}) rc {sim.sticks} {sim.rc_src} collisions {dict(sim.collisions)}")
            rate.sleep()
    finally:
        if ring is not None:
            ring.close()
            ring.unlink()
        pub.close()
        sub.close()
        log(NAME, "stopped")


if __name__ == "__main__":
    main()
