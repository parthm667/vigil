"""Put the simulator together: world + drone + camera + oracle detectors + clock.

    sim = Sim(demo_world(), drone_xy=(0.2, 1.5))
    per = Perception(cfg, sim.person_detector, ColorBlobDetector(), sim.context_detector, sim.cam)
    while ...:
        sim.step(0.1)
        frame = sim.camera.read(); tel = sim.drone.telemetry(); ...

Oracle detectors stand in for YOLO (which cannot recognise rendered boxes): they project the ground
truth into the frame's camera pose, with occlusion, noise and dropouts. The TARGET is found by the real
colour-blob detector on the rendered image, so that path is exercised for real.
"""

from __future__ import annotations

import numpy as np

from ..detect.base import Detector
from ..geometry import CameraModel, tello_camera
from ..types import Detection
from .drone_sim import SimCamera, SimDrone, SimDroneParams
from .person_model import SimPerson, project
from .world import World


COCO_LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class SimPersonDetector(Detector):
    """Pose-model stand-in with realistic faults: pixel noise, i.i.d. and BURST dropouts (0.3-1 s), left/right
    keypoint swaps on back views (YOLO-pose does this), and faint hallucinated face points on back views."""

    name = "sim_person"

    def __init__(self, sim: Sim, px_noise: float = 2.0, dropout: float = 0.03, burst_prob: float = 0.01,
                 lr_swap_prob: float = 0.1, face_hallucination: float = 0.3, seed: int = 1):
        self.sim, self.px_noise, self.dropout = sim, px_noise, dropout
        self.burst_prob, self.lr_swap_prob, self.face_hallucination = burst_prob, lr_swap_prob, face_hallucination
        self.rng = np.random.default_rng(seed)
        self._burst_until = -1.0

    @property
    def classes(self) -> list[str]:
        return ["person"]

    def _faults(self, det: Detection) -> Detection:
        k = det.keypoints.copy()
        back_view = k[0, 2] < 0.2  # face invisible
        if back_view and self.face_hallucination > 0:
            k[[0, 1, 2], 2] = self.rng.uniform(0.0, self.face_hallucination, 3)
        if back_view and self.rng.random() < self.lr_swap_prob:
            for a, b in COCO_LR_PAIRS:
                k[[a, b]] = k[[b, a]]
        det.keypoints = k
        return det

    def detect(self, image: np.ndarray) -> list[Detection]:
        meta = self.sim.camera.meta_for(image)
        w = self.sim.world
        t = self.sim.t
        if meta is None or w.person is None or meta[2] is None:
            return []
        if t < self._burst_until:
            return []
        if self.rng.random() < self.burst_prob:
            self._burst_until = t + self.rng.uniform(0.3, 1.0)
            return []
        if self.rng.random() < self.dropout:
            return []
        pose, _, (px, py, ph) = meta
        person = SimPerson(px, py, ph, w.person.height_m)
        eye = (pose.x, pose.y, pose.z)
        s = person.height_m / 1.75
        if not any(w.line_of_sight(eye, (px, py, z * s), include_person=False) for z in (1.6, 1.2, 0.6)):
            return []  # hidden behind furniture
        det = person.detect(pose, self.sim.cam, self.rng, self.px_noise)
        return [self._faults(det)] if det is not None else []


class SimContextDetector(Detector):
    """Furniture boxes that carry a detector class (COCO names)."""

    name = "sim_context"

    def __init__(self, sim: Sim, exclude=("", "person", "bottle"), dropout: float = 0.05, seed: int = 2):
        self.sim, self.exclude, self.dropout = sim, set(exclude), dropout
        self.rng = np.random.default_rng(seed)

    @property
    def classes(self) -> list[str]:
        return sorted({b.cls for b in self.sim.world.boxes if b.cls not in self.exclude})

    def detect(self, image: np.ndarray) -> list[Detection]:
        meta = self.sim.camera.meta_for(image)
        if meta is None:
            return []
        pose, _, _ = meta
        cam = self.sim.cam
        out = []
        for b in self.sim.world.boxes:
            if b.cls in self.exclude or self.rng.random() < self.dropout:
                continue
            corners = np.array([[x, y, z] for x in (b.lo[0], b.hi[0]) for y in (b.lo[1], b.hi[1]) for z in (b.lo[2], b.hi[2])])
            px, front = project(corners, pose, cam)
            if not front.all():
                continue
            x1, y1 = px.min(axis=0)
            x2, y2 = px.max(axis=0)
            bx1, by1, bx2, by2 = max(0.0, x1), max(0.0, y1), min(float(cam.width), x2), min(float(cam.height), y2)
            if bx2 - bx1 < 6 or by2 - by1 < 6 or (bx2 - bx1) * (by2 - by1) < 0.3 * (x2 - x1) * (y2 - y1):
                continue
            c = b.center
            top = (c[0], c[1], b.hi[2] - 0.02)
            if not (self.sim.world.line_of_sight((pose.x, pose.y, pose.z), top, ignore=[b])
                    or self.sim.world.line_of_sight((pose.x, pose.y, pose.z), c, ignore=[b])):
                continue
            out.append(Detection(b.cls, 0.8, (float(bx1), float(by1), float(bx2), float(by2)), source=self.name))
        return out


class SimFreeSpace:
    """Ground-truth free space: horizontal rays at the drone's altitude from the frame's camera pose."""

    def __init__(self, sim: Sim, sectors: int = 5, max_m: float = 5.0):
        self.sim, self.n, self.max_m = sim, sectors, max_m

    def estimate(self, image, cam, altitude_m):
        from ..mapping.freespace import FreeSpace, sector_bearings

        meta = self.sim.camera.meta_for(image)
        if meta is None:
            return FreeSpace([(b, None, None) for b in sector_bearings(cam, self.n)])
        pose = meta[0]
        out = []
        for b in sector_bearings(cam, self.n):
            d = min(self.max_m, self.sim.world.ray_distance(pose.x, pose.y, pose.z, pose.heading_deg + b, self.max_m))
            out.append((b, d / self.max_m, d))
        return FreeSpace(out, metric=True)


class Sim:
    def __init__(self, world: World, drone_xy: tuple[float, float] = (0.3, 1.5), drone_heading: float = 0.0,
                 params: SimDroneParams | None = None, cam: CameraModel | None = None, fps: float = 15.0, seed: int = 0):
        self.world = world
        self.cam = cam or tello_camera().scaled(480, 360)
        self.drone = SimDrone(world, drone_xy[0], drone_xy[1], drone_heading, params or SimDroneParams(seed=seed), self.cam)
        self.camera = SimCamera(self.drone, world, self.cam, fps, seed)
        self.person_detector = SimPersonDetector(self, seed=seed + 1)
        self.context_detector = SimContextDetector(self, seed=seed + 2)
        self.world.update(0.0)
        self.camera.capture()

    @property
    def t(self) -> float:
        return self.drone.t

    def step(self, dt: float, substeps: int = 2) -> None:
        for _ in range(substeps):
            self.drone.step(dt / substeps)
            self.world.update(self.drone.t)
        self.camera.capture()

    def run(self, seconds: float, dt: float = 0.05) -> None:
        for _ in range(int(round(seconds / dt))):
            self.step(dt)
