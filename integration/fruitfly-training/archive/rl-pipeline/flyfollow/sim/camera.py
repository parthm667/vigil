"""Pinhole camera plus a detector model: noise, dropouts, bursts, false boxes, detection rate and latency.

The controller never sees the true state, only boxes that arrive late and noisy,
the same way YOLO boxes arrive from the Tello video on the laptop.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def project(drone, target_x: float, target_y: float, target_z: float, size_m: float,
            fx: float, fy: float, width: int, height: int) -> dict | None:
    """True box (center x, center y, height in px) of a target, or None if it is out of view."""
    dx = target_x - drone.x
    dy = target_y - drone.y
    cos_p = math.cos(drone.psi)
    sin_p = math.sin(drone.psi)
    forward = dx * cos_p + dy * sin_p
    left = -dx * sin_p + dy * cos_p
    if forward < 0.15:
        return None
    up = target_z - drone.z
    cx = width / 2.0 - fx * left / forward
    cy = height / 2.0 - fy * up / forward
    h = fy * size_m / forward
    if cx < 0 or cx > width:
        return None
    if cy + h / 2.0 < 0 or cy - h / 2.0 > height:
        return None
    return {"cx": cx, "cy": cy, "h": h, "range": math.hypot(forward, left), "bearing": math.atan2(-left, forward)}


class DetectorModel:
    """Produces boxes at the detection rate; each becomes available after video latency plus detector time."""

    def __init__(self, scenario: dict, cam_cfg: dict, rng: np.random.Generator):
        s = scenario
        self.rng = rng
        self.width = cam_cfg["width"]
        self.height = cam_cfg["height"]
        self.fx = s["fx"]
        self.fy = s["fy"]
        self.period = 1.0 / s["detection_rate_hz"]
        self.latency = s["video_latency_s"] + s["detector_time_s"]
        self.center_sigma = s["box_center_sigma_px"]
        self.height_noise = s["box_height_noise"]
        self.dropout = s["dropout_iid"]
        self.burst_rate = s["burst_rate_per_s"]
        self.burst_len = s["burst_len_s"]
        self.false_rate = s["false_box_rate"]
        self.min_px = cam_cfg["min_detect_px"]
        self.next_capture = 0.0
        self.burst_until = -1.0
        self.pending = deque()

    def capture(self, t: float, drone, target, size_m: float) -> None:
        """Call every control tick; captures a frame whenever one is due."""
        while self.next_capture <= t:
            tc = self.next_capture
            self.next_capture += self.period
            if self.burst_until < tc and self.rng.random() < self.burst_rate * self.period:
                self.burst_until = tc + self.rng.uniform(self.burst_len[0], self.burst_len[1])
            box = project(drone, target.x, target.y, target.z, size_m, self.fx, self.fy, self.width, self.height)
            detected = box is not None
            if detected and tc < self.burst_until:
                detected = False
            if detected and self.rng.random() < self.dropout:
                detected = False
            if detected and box["h"] < self.min_px:
                # small boxes are found less often
                if self.rng.random() > box["h"] / self.min_px:
                    detected = False
            out = None
            if detected:
                out = {
                    "cx": box["cx"] + self.rng.normal(0.0, self.center_sigma),
                    "cy": box["cy"] + self.rng.normal(0.0, self.center_sigma),
                    "h": max(2.0, box["h"] * (1.0 + self.rng.normal(0.0, self.height_noise))),
                }
            elif self.rng.random() < self.false_rate:
                out = {
                    "cx": self.rng.uniform(0, self.width),
                    "cy": self.rng.uniform(0, self.height),
                    "h": self.rng.uniform(20, 150),
                }
            self.pending.append((tc + self.latency, tc, out))

    def available(self, t: float) -> list:
        """Detections (t_available, t_capture, box or None) that have arrived by time t."""
        ready = []
        while self.pending and self.pending[0][0] <= t:
            ready.append(self.pending.popleft())
        return ready
