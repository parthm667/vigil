"""Classical follower (plan 4.9): PID-HAND and, with tuned gains, PID-CMA.

    e_x = (cx - cx_ref) / W          cx_ref includes the side-offset bearing
    Z   = fy * H_target / h_px       range from target size
    yaw = clip(Kp_y * e_x + Kd_y * de_x/dt, -60, 60)
    fb  = clip(Kp_f * (Z - Z_ref), -20, max_fwd_stick), zero inside the deadband d0

de_x/dt comes from the shared BoxFilter's velocity (vcx / W), not a finite difference.
ud, lr and lost-target handling belong to the PersonGovernor.
"""

from __future__ import annotations

import math

from flyfollow.interfaces import IMG_W, BoxState, Settings

HAND_GAINS = {"Kp_y": 100.0, "Kd_y": 20.0, "Kp_f": 40.0, "d0": 0.15}


class PIDController:
    """Satisfies flyfollow.interfaces.Controller. gains uses the keys of configs/controllers.yaml pid.params."""

    def __init__(self, gains: dict | None = None, name: str = "PID-HAND"):
        g = dict(HAND_GAINS)
        if gains:
            g.update({k: float(v) for k, v in gains.items()})
        self.gains = g
        self.name = name
        self._kp_y, self._kd_y, self._kp_f, self._d0 = g["Kp_y"], g["Kd_y"], g["Kp_f"], g["d0"]

    def reset(self, settings: Settings, seed: int) -> None:
        pass

    def warmup(self, seconds: float) -> None:
        pass

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        if not box.valid or box.h <= 0.0:
            return 0.0, 0.0
        st = settings
        cx_ref = st.cx0 + st.fx * math.tan(math.radians(st.side_offset_deg))
        e_x = (box.cx - cx_ref) / IMG_W
        yaw = self._kp_y * e_x + self._kd_y * box.vcx / IMG_W
        yaw = max(-60.0, min(60.0, yaw))
        e = st.fy * st.target_size_m / box.h - st.z_ref_m
        if abs(e) < self._d0:
            fb = 0.0
        else:
            fb = max(-20.0, min(st.max_fwd_stick, self._kp_f * e))
        return yaw, fb
