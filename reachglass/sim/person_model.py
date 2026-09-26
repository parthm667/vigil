"""Synthetic person: a 3-D COCO-17 skeleton projected into a camera, with pose-model-like keypoint
confidences (face points fade when the person turns away, ears show on the side facing the camera).

Used by the simulator's oracle person detector and to test the person estimator against ground truth.
World frame as in types.py: x forward, y right, z up, headings clockwise positive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry import CameraModel
from ..types import Detection

# COCO-17 keypoints in the person frame, for a 1.75 m person: (forward, right, up). Left side = -right.
_H = 1.75
KEYPOINTS_1_75 = np.array([
    (0.10, 0.000, 1.63),  # 0 nose
    (0.08, -0.035, 1.66), (0.08, 0.035, 1.66),  # 1, 2 eyes (left, right)
    (0.00, -0.075, 1.64), (0.00, 0.075, 1.64),  # 3, 4 ears
    (0.00, -0.200, 1.45), (0.00, 0.200, 1.45),  # 5, 6 shoulders
    (0.00, -0.240, 1.17), (0.00, 0.240, 1.17),  # 7, 8 elbows
    (0.05, -0.240, 0.90), (0.05, 0.240, 0.90),  # 9, 10 wrists
    (0.00, -0.110, 0.92), (0.00, 0.110, 0.92),  # 11, 12 hips
    (0.00, -0.100, 0.48), (0.00, 0.100, 0.48),  # 13, 14 knees
    (0.00, -0.100, 0.08), (0.00, 0.100, 0.08),  # 15, 16 ankles
])
# body outline for the bounding box: head top, feet, and an elliptical torso/arms cross-section 0.48 m
# wide and 0.28 m deep (matches the estimator's body_width_m / BODY_DEPTH_M priors).
_ring = np.linspace(0, 2 * np.pi, 24, endpoint=False)
_OUTLINE_1_75 = np.vstack([
    np.array([(0, 0, 1.75), (0, -0.12, 0.0), (0, 0.12, 0.0)]),
    np.column_stack([0.14 * np.cos(_ring), 0.24 * np.sin(_ring), np.full(_ring.size, 1.2)]),
])



def _body_samples() -> np.ndarray:
    """Dense points on the body surface (1.75 m person), so the detection box covers only the VISIBLE part:
    close above someone only the head is in frame and a real detector's box is head-sized, not body-sized."""
    pts = []

    def rings(z0, z1, dz, n, cx, cy, rx, ry):
        for z in np.arange(z0, z1 + 1e-9, dz):
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            k = rx(z) if callable(rx) else rx
            j = ry(z) if callable(ry) else ry
            pts.append(np.column_stack([cx + k * np.cos(a), cy + j * np.sin(a), np.full(n, z)]))

    head = lambda z: np.sqrt(max(0.0, 1.0 - ((z - 1.64) / 0.11) ** 2))  # ellipsoid profile, top at 1.75
    rings(1.53, 1.75, 0.01, 16, 0.0, 0.0, lambda z: 0.10 * head(z), lambda z: 0.08 * head(z))  # 0.16 wide
    rings(1.45, 1.53, 0.02, 12, 0.0, 0.0, 0.06, 0.06)  # neck
    rings(0.92, 1.45, 0.02, 16, 0.0, 0.0, 0.14, 0.24)  # torso + arms: 0.48 wide, 0.28 deep
    for side in (-0.10, 0.10):
        rings(0.0, 0.92, 0.03, 8, 0.0, side, 0.07, 0.07)  # legs
    return np.vstack(pts)


_BODY_1_75 = _body_samples()


def visible_box(px: np.ndarray, front: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Box of the projected body points inside the image, extended to an image edge where the body
    continues beyond it (like a detector's box on a person cut by the frame)."""
    inside = front & (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    if not inside.any():
        return None
    x1, y1 = px[inside].min(axis=0)
    x2, y2 = px[inside].max(axis=0)
    out = front & ~inside
    cols = out & (px[:, 0] >= x1 - 2) & (px[:, 0] <= x2 + 2)
    rows = out & (px[:, 1] >= y1 - 2) & (px[:, 1] <= y2 + 2)
    if (cols & (px[:, 1] >= height)).any():
        y2 = float(height)
    if (cols & (px[:, 1] < 0)).any():
        y1 = 0.0
    if (rows & (px[:, 0] >= width)).any():
        x2 = float(width)
    if (rows & (px[:, 0] < 0)).any():
        x1 = 0.0
    return float(x1), float(y1), float(x2), float(y2)


@dataclass
class CameraPose:
    x: float
    y: float
    z: float  # height above the floor
    heading_deg: float
    pitch_deg: float = 0.0  # total pitch (camera tilt + body pitch), nose up positive


def world_to_camera(points: np.ndarray, pose: CameraPose) -> np.ndarray:
    """(N,3) world points -> (N,3) camera coords (right, up, forward)."""
    h, p = math.radians(pose.heading_deg), math.radians(pose.pitch_deg)
    fwd = np.array([math.cos(p) * math.cos(h), math.cos(p) * math.sin(h), math.sin(p)])
    right = np.array([-math.sin(h), math.cos(h), 0.0])
    up = np.array([-math.sin(p) * math.cos(h), -math.sin(p) * math.sin(h), math.cos(p)])
    v = points - np.array([pose.x, pose.y, pose.z])
    return np.stack([v @ right, v @ up, v @ fwd], axis=1)


def project(points: np.ndarray, pose: CameraPose, cam: CameraModel) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixels (N,2), in_front mask (N,))."""
    c = world_to_camera(points, pose)
    front = c[:, 2] > 0.05
    z = np.where(front, c[:, 2], 1.0)
    px = np.stack([cam.cx + cam.fx * c[:, 0] / z, cam.cy - cam.fy * c[:, 1] / z], axis=1)
    return px, front


@dataclass
class SimPerson:
    x: float
    y: float
    heading_deg: float  # direction the person faces
    height_m: float = 1.75

    def world_points(self, local: np.ndarray) -> np.ndarray:
        s = self.height_m / _H
        h = math.radians(self.heading_deg)
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([-math.sin(h), math.cos(h)])
        xy = np.array([self.x, self.y]) + (local[:, :1] * fwd + local[:, 1:2] * right) * s
        return np.column_stack([xy, local[:, 2] * s])

    def facing_rel_deg(self, cam_x: float, cam_y: float) -> float:
        """Ground-truth facing_deg as defined in types.PersonObs (0 = back to the camera)."""
        ray = math.degrees(math.atan2(self.y - cam_y, self.x - cam_x))
        return (self.heading_deg - ray + 180.0) % 360.0 - 180.0

    def detect(self, pose: CameraPose, cam: CameraModel, rng: np.random.Generator | None = None,
               px_noise: float = 0.0, min_visible_frac: float = 0.15, min_keypoints: int = 3) -> Detection | None:
        """What a pose model would report, or None if the person is not in view. Like YOLO, a person is
        found when enough of the body is in frame: >= `min_visible_frac` of the box, or at least
        `min_keypoints` keypoints (e.g. head and shoulders when the drone is close above them)."""
        pts = self.world_points(KEYPOINTS_1_75)
        kp_px, kp_front = project(pts, pose, cam)
        out_px, out_front = project(self.world_points(_OUTLINE_1_75), pose, cam)
        if not out_front.all():
            return None
        x1, y1 = out_px.min(axis=0)
        x2, y2 = out_px.max(axis=0)
        body_px, body_front = project(self.world_points(_BODY_1_75), pose, cam)
        vis = visible_box(body_px, body_front, cam.width, cam.height)
        if vis is None:
            return None
        bx1, by1, bx2, by2 = vis
        if bx2 - bx1 < 4 or by2 - by1 < 4:
            return None
        kp_in = kp_front & (kp_px[:, 0] >= 0) & (kp_px[:, 0] < cam.width) & (kp_px[:, 1] >= 0) & (kp_px[:, 1] < cam.height)
        if (bx2 - bx1) * (by2 - by1) < min_visible_frac * (x2 - x1) * (y2 - y1) and kp_in.sum() < min_keypoints:
            return None
        # visibility: face toward the camera -> face points; a side toward the camera -> that ear
        to_cam = math.atan2(pose.y - self.y, pose.x - self.x)
        alpha = math.radians(self.heading_deg) - to_cam  # 0 = faces the camera
        face = float(np.clip(0.5 + 0.8 * math.cos(alpha), 0.0, 0.95))
        # the person's right points along heading + 90; it faces the camera when that is toward to_cam
        right_to_cam = math.cos(math.radians(self.heading_deg + 90) - to_cam)
        ear_r = float(np.clip(0.35 + 0.8 * right_to_cam, 0.0, 0.95))
        ear_l = float(np.clip(0.35 - 0.8 * right_to_cam, 0.0, 0.95))
        conf = np.array([face, face, face, ear_l, ear_r] + [0.9] * 12)
        inside = kp_front & (kp_px[:, 0] >= 0) & (kp_px[:, 0] < cam.width) & (kp_px[:, 1] >= 0) & (kp_px[:, 1] < cam.height)
        conf = np.where(inside, conf, 0.0)
        if rng is not None and px_noise > 0:
            kp_px = kp_px + rng.normal(0, px_noise, kp_px.shape)
        kps = np.column_stack([kp_px, conf]).astype(np.float32)
        box_conf = float(np.clip(0.5 + 0.5 * inside.mean(), 0, 0.95))
        return Detection("person", box_conf, (float(bx1), float(by1), float(bx2), float(by2)), kps, source="sim")
