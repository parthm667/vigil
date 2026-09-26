"""Constant-velocity Kalman filter on the target box, with latency compensation.

Each box is stamped with its estimated capture time (arrival time minus the
configured `video_latency_s`), and the filter predicts forward to "now". This is
shared preprocessing: every controller arm sees exactly the same observation.
"""

from __future__ import annotations

import math

import numpy as np


class _Axis:
    """1D constant-velocity filter with state [position, velocity]."""

    def __init__(self, accel_sigma: float, meas_sigma: float):
        self.q = accel_sigma ** 2
        self.r = meas_sigma ** 2
        self.x = np.zeros(2)
        self.P = np.eye(2) * 1e6
        self.t = None

    def predict_to(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        dt = t - self.t
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]])
        return F @ self.x, F @ self.P @ F.T + Q

    def update(self, t: float, z: float, r_scale: float = 1.0) -> float:
        """Returns the normalized innovation (in standard deviations) before updating."""
        if self.t is None:
            self.x = np.array([z, 0.0])
            self.P = np.diag([self.r * r_scale, 1e4])
            self.t = t
            return 0.0
        x, P = self.predict_to(t)
        S = P[0, 0] + self.r * r_scale
        innovation = z - x[0]
        K = P[:, 0] / S
        self.x = x + K * innovation
        self.P = P - np.outer(K, P[0, :])
        self.t = t
        return innovation / math.sqrt(S)


class BoxFilter:
    def __init__(self, video_latency_s: float, hold_s: float):
        self.latency = video_latency_s
        self.hold_s = hold_s
        self.reset()

    def reset(self) -> None:
        self.cx = _Axis(accel_sigma=400.0, meas_sigma=4.0)
        self.cy = _Axis(accel_sigma=300.0, meas_sigma=4.0)
        self.logh = _Axis(accel_sigma=1.5, meas_sigma=0.06)
        self.last_meas_t = None
        self.misses = 0

    def update(self, t_arrival: float, box: dict | None) -> None:
        if box is None:
            return
        t_capture = t_arrival - self.latency
        lost = self.last_meas_t is None or (t_capture - self.last_meas_t) > self.hold_s
        if lost:
            self.reset()
        else:
            # gate false boxes: reject a box far from the prediction
            x, P = self.cx.predict_to(t_capture)
            if abs(box["cx"] - x[0]) > 5.0 * math.sqrt(P[0, 0] + self.cx.r) + 40.0:
                self.misses += 1
                if self.misses < 4:
                    return
                self.reset()
        self.misses = 0
        self.cx.update(t_capture, box["cx"])
        self.cy.update(t_capture, box["cy"])
        self.logh.update(t_capture, math.log(max(box["h"], 1.0)))
        self.last_meas_t = t_capture

    def estimate(self, t: float) -> dict:
        """Box predicted to time t, plus age of the newest measurement."""
        if self.last_meas_t is None:
            # nothing seen yet: age counts from the start of the episode (t = 0)
            return {"valid": False, "age": t}
        age = t - self.last_meas_t
        cx, _ = self.cx.predict_to(t)
        cy, _ = self.cy.predict_to(t)
        logh, _ = self.logh.predict_to(t)
        return {
            "valid": age <= self.hold_s,
            "age": age,
            "cx": float(cx[0]),
            "cy": float(cy[0]),
            "h": float(math.exp(logh[0])),
            "dcx": float(cx[1]),
        }


def observation(est: dict, scenario: dict, cam_cfg: dict) -> dict:
    """Geometry shared by all controllers, using nominal (not true) camera numbers."""
    fx = cam_cfg["fx_nominal"]
    fy = cam_cfg["fy_nominal"]
    width = cam_cfg["width"]
    height = cam_cfg["height"]
    offset = math.radians(scenario["side_offset_deg"]) if scenario["kind"] == "follow" else 0.0
    obs = {
        "valid": est["valid"],
        "age": est["age"],
        "width": width,
        "height": height,
        "fx": fx,
        "follow_distance_m": scenario["follow_distance_m"],
        "max_fwd_stick": scenario["max_fwd_stick"],
        "side_offset": offset,
    }
    if not est["valid"]:
        obs["theta"] = 0.0
        obs["dtheta"] = 0.0
        obs["s"] = 0.0
        obs["range_est"] = float("inf")
        obs["cy"] = height / 2.0
        return obs
    raw = math.atan((est["cx"] - width / 2.0) / fx)
    obs["theta"] = raw - offset
    obs["dtheta"] = est["dcx"] * fx / (fx ** 2 + (est["cx"] - width / 2.0) ** 2)
    h_ref = fy * scenario["target_size_assumed_m"] / scenario["follow_distance_m"]
    obs["s"] = est["h"] / h_ref
    obs["range_est"] = fy * scenario["target_size_assumed_m"] / est["h"]
    obs["cy"] = est["cy"]
    return obs
