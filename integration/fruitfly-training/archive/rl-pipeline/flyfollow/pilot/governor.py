"""PersonGovernor: shared by every controller arm, never learned (plan section 4.1).

It owns up/down, left/right, lost-target behavior, minimum distance, speed
clamps and slew limits. Arms only produce yaw and forward, so comparisons
between arms measure the controller and nothing else.
All sticks are final Tello units (-100..100).
"""

from __future__ import annotations

import math


class PersonGovernor:
    def __init__(self, gov_cfg: dict, scenario: dict):
        self.cfg = gov_cfg
        self.s = scenario
        self.max_fwd = scenario["max_fwd_stick"]
        self.max_back = gov_cfg["max_back_stick"]
        self.max_yaw = gov_cfg["max_yaw_stick"]
        self.slew = gov_cfg["slew_stick_per_s"]
        self.min_dist = scenario["min_dist_m"]
        self.alt_floor = scenario["alt_floor_m"]
        self.alt_ceiling = 2.2
        self.prev_yaw = 0.0
        self.prev_fb = 0.0
        self.last_theta_sign = 1.0
        self.landed = False

    def filter(self, yaw: float, fb: float, obs: dict, altitude: float, dt: float) -> tuple[float, float, float, dict]:
        flags = {"safety": False, "clamped": False, "reasons": []}

        if obs["valid"]:
            if obs["theta"] > 0:
                self.last_theta_sign = 1.0
            elif obs["theta"] < 0:
                self.last_theta_sign = -1.0
        else:
            # lost target: stop and turn toward where it was last seen, then hover, then land
            flags["safety"] = True
            flags["reasons"].append("lost")
            fb = 0.0
            if obs["age"] < self.cfg["lost_hover_s"]:
                yaw = 20.0 * self.last_theta_sign
            else:
                yaw = 0.0
            if obs["age"] > self.cfg["lost_land_s"]:
                self.landed = True

        if obs["valid"] and obs["range_est"] < self.min_dist:
            flags["safety"] = True
            flags["reasons"].append("min_dist")
            fb = -self.max_back
        elif obs["valid"] and obs["range_est"] < self.min_dist + self.cfg["backoff_margin_m"] and fb > 0:
            fb = 0.0
            flags["clamped"] = True

        clamped_fb = max(-self.max_back, min(self.max_fwd, fb))
        clamped_yaw = max(-self.max_yaw, min(self.max_yaw, yaw))
        if clamped_fb != fb or clamped_yaw != yaw:
            flags["clamped"] = True
        fb = clamped_fb
        yaw = clamped_yaw

        # slew limit, but always allow moving toward zero at once
        step = self.slew * dt
        if abs(fb) > abs(self.prev_fb) or fb * self.prev_fb < 0:
            fb = max(self.prev_fb - step, min(self.prev_fb + step, fb))
        if abs(yaw) > abs(self.prev_yaw) or yaw * self.prev_yaw < 0:
            yaw = max(self.prev_yaw - step, min(self.prev_yaw + step, yaw))
        self.prev_fb = fb
        self.prev_yaw = yaw

        # up/down: keep the target slightly below image center, within altitude limits
        ud = 0.0
        if obs["valid"]:
            cy_ref = obs["height"] / 2.0 + self.s["cy_ref_frac"] * obs["height"]
            ud = max(-30.0, min(30.0, 60.0 * (cy_ref - obs["cy"]) / obs["height"] * 4.0))
        if altitude <= self.alt_floor and ud < 0:
            ud = 0.0
        if altitude < self.alt_floor - 0.05:
            ud = 15.0
        if altitude >= self.alt_ceiling and ud > 0:
            ud = 0.0
        if math.isnan(fb) or math.isnan(yaw):
            fb = 0.0
            yaw = 0.0
            flags["safety"] = True
            flags["reasons"].append("nan")
        return yaw, fb, ud, flags
