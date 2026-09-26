"""Drone pose in the mission frame from telemetry yaw + completed discrete moves.

The mission frame starts where the drone is when the search begins (x = its heading then, y = right).
Heading comes from the IMU yaw relative to that moment (clockwise +, sign checked at the first real
rotation: if the drone reports the opposite direction to what we commanded, yaw_sign is flipped).
Position advances by the COMMANDED distance of each completed move (the Tello closes that loop itself
with its optical-flow hold), times `move_scale` (calibrate on the venue floor).
"""

from __future__ import annotations

import logging

from ..types import Pose2D, Telemetry, wrap_deg

log = logging.getLogger("reachglass.odometry")


class Odometry:
    def __init__(self, yaw_sign: int = 1, move_scale: float = 1.0):
        self.yaw_sign = yaw_sign
        self.move_scale = move_scale
        self.pose = Pose2D()
        self._yaw0: float | None = None
        self._cmd_heading = 0.0  # heading from commanded rotations (fallback without yaw telemetry)
        self.sign_checked = False
        self.path: list[tuple[float, float]] = [(0.0, 0.0)]

    def reset(self, tel: Telemetry | None) -> None:
        """Start a new mission frame at the drone's current position and heading."""
        self._yaw0 = None if tel is None or tel.yaw_deg is None else tel.yaw_deg
        self._cmd_heading = 0.0
        self.pose = Pose2D()
        self.path = [(0.0, 0.0)]

    def heading_from(self, tel: Telemetry | None) -> float:
        if tel is None or tel.yaw_deg is None or self._yaw0 is None:
            return self._cmd_heading
        return wrap_deg(self.yaw_sign * (tel.yaw_deg - self._yaw0))

    def update(self, tel: Telemetry | None) -> Pose2D:
        """Refresh the heading from telemetry (call every loop)."""
        self.pose = Pose2D(self.pose.x, self.pose.y, self.heading_from(tel))
        return self.pose

    def on_rotation_done(self, commanded_deg: float, yaw_before: float | None, yaw_after: float | None) -> None:
        self._cmd_heading = wrap_deg(self._cmd_heading + commanded_deg)
        # Resolve the sign ONCE, on a clean rotation: near 180 deg the wrapped measurement is ambiguous
        # (a correct 180 deg turn that overshoots reads as -179), so only 20..135 deg turns are used.
        if self.sign_checked or yaw_before is None or yaw_after is None or not 20 <= abs(commanded_deg) <= 135:
            return
        raw = wrap_deg(yaw_after - yaw_before)
        e_same = abs(wrap_deg(self.yaw_sign * raw - commanded_deg))
        e_flip = abs(wrap_deg(-self.yaw_sign * raw - commanded_deg))
        if e_flip < 20 and e_same > 60:
            self.yaw_sign = -self.yaw_sign
            log.warning("telemetry yaw runs opposite to 'cw': flipping yaw_sign to %d (set drone.yaw_sign in the config)",
                        self.yaw_sign)
            self.sign_checked = True
        elif e_same < 20:
            self.sign_checked = True

    def on_move_done(self, direction: str, cm: float, tel: Telemetry | None = None) -> Pose2D:
        self.update(tel)  # heading from telemetry, or from the commanded rotations when there is none
        d = cm / 100.0 * self.move_scale
        fwd = {"forward": d, "back": -d}.get(direction, 0.0)
        right = {"right": d, "left": -d}.get(direction, 0.0)
        self.pose = self.pose.moved(forward_m=fwd, right_m=right)
        self.path.append((self.pose.x, self.pose.y))
        return self.pose
