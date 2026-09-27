"""Tello response to rc sticks: per-axis dead time plus a first-order lag (fit from the lag test).

World frame: x forward at takeoff, y left, z up. Yaw `psi` is counterclockwise
positive in the world frame, while a positive yaw stick turns the drone right
(clockwise), as on the real Tello.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class DroneModel:
    def __init__(self, scenario: dict, dt: float, rng: np.random.Generator):
        s = scenario
        self.dt = dt
        self.rng = rng
        self.x = 0.0
        self.y = 0.0
        self.z = s["drone_alt_m"]
        self.psi = 0.0
        self.v_fwd = 0.0
        self.yaw_rate = 0.0  # deg/s, positive = turning right
        self.v_z = 0.0
        self.drift_x = 0.0
        self.drift_y = 0.0

        self.yaw_gain = s["yaw_gain_dps_per_100"] / 100.0
        self.fwd_gain = s["fwd_gain_mps_per_100"] / 100.0
        self.vert_gain = s["vert_gain_mps_per_100"] / 100.0
        self.yaw_tau = s["yaw_tau_s"]
        self.fwd_tau = s["fwd_tau_s"]
        self.drift_sigma = s["hover_drift_sigma_mps"]

        # command delay lines, one slot per control tick
        self.yaw_queue = deque([0.0] * max(1, int(round(s["yaw_dead_time_s"] / dt))))
        self.fwd_queue = deque([0.0] * max(1, int(round(s["fwd_dead_time_s"] / dt))))
        self.vert_queue = deque([0.0] * max(1, int(round(s["vert_dead_time_s"] / dt))))

    def step(self, fb: float, yaw: float, ud: float) -> None:
        """Advance one control tick with sticks in final Tello units (-100..100)."""
        self.yaw_queue.append(yaw)
        self.fwd_queue.append(fb)
        self.vert_queue.append(ud)
        yaw_cmd = self.yaw_queue.popleft()
        fwd_cmd = self.fwd_queue.popleft()
        vert_cmd = self.vert_queue.popleft()

        dt = self.dt
        self.yaw_rate += (self.yaw_gain * yaw_cmd - self.yaw_rate) * min(1.0, dt / self.yaw_tau)
        self.v_fwd += (self.fwd_gain * fwd_cmd - self.v_fwd) * min(1.0, dt / self.fwd_tau)
        self.v_z = self.vert_gain * vert_cmd

        if self.drift_sigma > 0:
            self.drift_x += self.rng.normal(0.0, self.drift_sigma * math.sqrt(dt))
            self.drift_y += self.rng.normal(0.0, self.drift_sigma * math.sqrt(dt))
            self.drift_x *= 0.98
            self.drift_y *= 0.98

        self.psi -= math.radians(self.yaw_rate) * dt
        self.x += (self.v_fwd * math.cos(self.psi) + self.drift_x) * dt
        self.y += (self.v_fwd * math.sin(self.psi) + self.drift_y) * dt
        self.z = max(0.3, self.z + self.v_z * dt)
