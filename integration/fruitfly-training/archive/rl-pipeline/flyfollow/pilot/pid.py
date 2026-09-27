"""Classical follower (plan section 4.9): the PID baseline and the demo fallback."""

from __future__ import annotations

import math

PID_HAND = {"kp_yaw": 100.0, "kd_yaw": 20.0, "kp_fwd": 40.0, "deadband_m": 0.15}


class PIDController:
    def __init__(self, params: dict):
        self.p = dict(params)
        self.prev_ex = None

    def reset(self, scenario: dict) -> None:
        self.prev_ex = None

    def act(self, obs: dict, dt: float) -> tuple[float, float]:
        if not obs["valid"]:
            self.prev_ex = None
            return 0.0, 0.0
        ex = math.tan(obs["theta"]) * obs["fx"] / obs["width"]
        dex = 0.0
        if self.prev_ex is not None:
            dex = (ex - self.prev_ex) / dt
        self.prev_ex = ex
        yaw = self.p["kp_yaw"] * ex + self.p["kd_yaw"] * dex

        err = obs["range_est"] - obs["follow_distance_m"]
        if abs(err) < self.p["deadband_m"]:
            fb = 0.0
        else:
            fb = self.p["kp_fwd"] * err
        return yaw, fb
