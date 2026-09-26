"""Safety governor: wraps any Drone; limits always win over the mission.

    drone = SafetyGovernor(TelloDrone(...), cfg.safety)
    ...every loop:  action = drone.check(now, last_frame_t)   # 'ok' | 'hover' | 'land'

Rules (config.SafetyCfg):
  * every rc channel clamped to +-max_rc
  * altitude ceiling/floor: rc climb is cut above max_altitude_m (and pushed down), descent is cut below
    min_altitude_m; discrete up/down moves that would cross a limit are refused
  * battery below min_battery_pct, or flight time above max_flight_s -> land
  * no new video frame for frame_timeout_s, or stale telemetry -> hover (rc 0) until they come back
The mission code keeps working through the wrapper; refused commands report 'error: safety ...'.
"""

from __future__ import annotations

import time

from ..config import SafetyCfg
from .base import Drone, check_move, clamp_rc

SOFT_BAND_M = 0.4  # vertical speed is tapered to zero over this distance before a limit


class SafetyGovernor(Drone):
    name = "safety"

    def __init__(self, inner: Drone, cfg: SafetyCfg):
        self.inner = inner
        self.cfg = cfg
        self.events: list[tuple[float, str]] = []
        self._takeoff_t: float | None = None
        self._refused: str | None = None
        self.hovering_for_safety = False
        self.land_requested = False
        self._land_sent = -1e9

    def _event(self, msg: str) -> None:
        if not self.events or self.events[-1][1] != msg:
            self.events.append((time.time(), msg))

    # ------------------------------------------------------------------ periodic check
    def check(self, now: float, last_frame_t: float | None) -> str:
        """Call every loop. Returns 'land' (and starts landing), 'hover' (and holds rc 0) or 'ok'."""
        c = self.cfg
        tel = self.inner.telemetry()
        if self.inner.flying:
            if self._takeoff_t is None:
                self._takeoff_t = now
            if tel.battery_pct is not None and tel.battery_pct < c.min_battery_pct:
                return self._land(f"battery {tel.battery_pct:.0f}% < {c.min_battery_pct:.0f}%")
            if now - self._takeoff_t > c.max_flight_s:
                return self._land(f"flight time > {c.max_flight_s:.0f} s")
            frame_stale = last_frame_t is None or now - last_frame_t > c.frame_timeout_s
            tel_stale = now - tel.t > c.telemetry_timeout_s
            if frame_stale or tel_stale:
                self._event("video stale -> hover" if frame_stale else "telemetry stale -> hover")
                self.hovering_for_safety = True
                if not self.inner.busy():
                    self.inner.rc(0, 0, 0, 0)
                return "hover"
        else:
            self._takeoff_t = None
        self.hovering_for_safety = False
        return "ok"

    def _land(self, why: str) -> str:
        self._event(f"LAND: {why}")
        now = time.time()
        # (re)send land if we have not asked yet, or a previous land failed / was lost and it still flies
        if not self.land_requested or (not self.inner.busy() and self.inner.flying and now - self._land_sent > 2.0):
            self.land_requested = True
            self._land_sent = now
            self.inner.land()
        return "land"

    # ------------------------------------------------------------------ Drone interface (filtered)
    @property
    def flying(self) -> bool:
        return self.inner.flying

    def connect(self) -> None:
        self.inner.connect()

    def close(self) -> None:
        self.inner.close()

    def takeoff(self) -> None:
        self._refused = None
        self.inner.takeoff()

    def land(self) -> None:
        self.land_requested = True
        self._land_sent = time.time()
        self.inner.land()

    def emergency(self) -> None:
        self._event("EMERGENCY")
        self.inner.emergency()

    def stop(self) -> None:
        self.inner.stop()

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        if self.land_requested or self.hovering_for_safety:
            return
        m = self.cfg.max_rc
        lr, fb, ud, yaw = (clamp_rc(v, m) for v in (lr, fb, ud, yaw))
        tel = self.inner.telemetry()
        high, clear = tel.floor_altitude_m(), tel.altitude_m()  # above the floor / above what is below
        if self.inner.flying:
            hi, lo, band = self.cfg.max_altitude_m, self.cfg.min_altitude_m, SOFT_BAND_M
            if high is not None:
                if high > hi:
                    ud = min(ud, -20)
                    self._event(f"ceiling {high:.2f} m > {hi} m -> descend")
                elif ud > 0 and high > hi - band:  # taper the climb: the drone keeps rising ~0.2 m after rc 0
                    ud = int(ud * max(0.0, (hi - high) / band))
            if clear is not None:
                if clear < lo:
                    ud = max(ud, 0)
                    self._event(f"floor {clear:.2f} m < {lo} m -> no descent")
                elif ud < 0 and clear < lo + band:
                    ud = int(ud * max(0.0, (clear - lo) / band))
        self.inner.rc(lr, fb, ud, yaw)

    def move(self, direction: str, cm: int) -> None:
        cm = check_move(direction, cm)
        if self.land_requested:
            self._refuse("landing")
            return
        tel = self.inner.telemetry()
        high, clear = tel.floor_altitude_m(), tel.altitude_m()
        if direction == "up" and high is not None and high + cm / 100.0 > self.cfg.max_altitude_m:
            self._refuse(f"up {cm} would reach {high + cm / 100.0:.2f} m")
            return
        if direction == "down" and clear is not None and clear - cm / 100.0 < self.cfg.min_altitude_m:
            self._refuse(f"down {cm} would leave {clear - cm / 100.0:.2f} m below")
            return
        self._refused = None
        self.inner.move(direction, cm)

    def rotate(self, deg: int) -> None:
        if self.land_requested:
            self._refuse("landing")
            return
        self._refused = None
        self.inner.rotate(deg)

    def _refuse(self, why: str) -> None:
        self._refused = f"error: safety ({why})"
        self._event(f"refused: {why}")

    def busy(self) -> bool:
        return self.inner.busy()

    def last_result(self) -> str | None:
        return self._refused if self._refused else self.inner.last_result()

    def set_speed(self, cm_s: int) -> None:
        self.inner.set_speed(cm_s)

    def telemetry(self):
        return self.inner.telemetry()

    def frame_source(self):
        return self.inner.frame_source()
