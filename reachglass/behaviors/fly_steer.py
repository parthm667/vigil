"""Optional fruit fly steering for FOLLOW and APPROACH (follow.steering / approach.steering: fly).

The fly (flyfollow.steer.FlySteer: a 1,446-neuron connectome circuit from the fruit fly brain) decides ONLY the
yaw stick. Distance, altitude, orbit, lost-target search, the approach's discrete moves and the SafetyGovernor
are unchanged, and every command still goes through ctx.drone (the governor).

If flyfollow is not installed, the brain cannot be loaded, or FlySteer raises, the behaviour logs one warning
and uses its own yaw law instead (for the rest of that behaviour, or of the run if loading failed).
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger("reachglass.fly")
_SHARED: dict = {}  # one FlySteer per (params, brain, latency, viz) per process: the brain loads once
_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        log.warning(msg)


class FlyYaw:
    """Per-behaviour handle on the shared FlySteer. Angles in degrees here (the stack's unit), + = right."""

    def __init__(self, steer, max_rc_yaw: int, fwd_enabled: bool = False):
        self.steer = steer
        self.max_rc_yaw = max_rc_yaw
        self.fwd_enabled = fwd_enabled
        self.failed = False

    @classmethod
    def for_behavior(cls, ctx, kind: str, z_ref_m: float) -> FlyYaw | None:
        """A reset FlyYaw if `<kind>.steering` is "fly" and the fly loads, else None (use the PID law)."""
        cfg = ctx.cfg
        if getattr(cfg, kind).steering != "fly":
            return None
        fc = cfg.fly
        latency = cfg.drone.video_lag_s if fc.latency_s is None else fc.latency_s
        key = (fc.params_path, fc.brain_path, latency, fc.viz, fc.smoothing_ms, fc.deadband, fc.slew, fc.hysteresis)
        try:
            steer = _SHARED.get(key)
            if steer is None:
                from flyfollow.steer import FlySteer

                steer = FlySteer(params_path=fc.params_path or None, brain_path=fc.brain_path or None,
                                 max_yaw=fc.max_rc_yaw, latency_s=latency, viz=fc.viz,
                                 smoothing_ms=fc.smoothing_ms, deadband=fc.deadband, slew=fc.slew,
                                 hysteresis=fc.hysteresis)
                _SHARED[key] = steer
                log.info("fly steering loaded: %s", steer.stats)
            steer.reset(z_ref_m=z_ref_m, kind=kind)  # fresh brain state + 1 s warmup (~35 ms)
        except Exception as e:  # noqa: BLE001 (any failure: keep flying on the PID law)
            _warn_once(f"load{key}", f"fly steering unavailable ({type(e).__name__}: {e}); using the PID yaw law")
            return None
        return cls(steer, fc.max_rc_yaw, fwd_enabled=(kind == "follow" and fc.forward))

    @property
    def target_valid(self) -> bool:
        """The fly still has a target (a detection in the last 0.5 s)."""
        return (not self.failed) and bool(self.steer.target_valid)

    def yaw(self, now: float, bearing_deg: float | None = None, range_m: float | None = None,
            frame_t: float | None = None, fallback: float = 0.0) -> float:
        """Yaw stick (+ = clockwise). Pass the bearing only for a detection on a NEW frame; call with no
        bearing on other loops so the brain keeps ticking. Returns `fallback` if the fly has failed.
        range_m sets the size the fly sees (z_ref / range); None = its standoff size, which follow and approach
        use (in the sim: 6.4 deg RMS bearing error with None vs 9.4 with the person range)."""
        if self.failed:
            return fallback
        try:
            b = None if bearing_deg is None else math.radians(bearing_deg)
            y = self.steer.yaw(now, b, range_m=range_m, t_frame=frame_t)
            return float(max(-self.max_rc_yaw, min(self.max_rc_yaw, y)))
        except Exception as e:  # noqa: BLE001
            self.failed = True
            _warn_once(f"run{id(self)}", f"fly steering failed ({type(e).__name__}: {e}); back to the PID yaw law")
            return fallback

    def forward(self, fallback: float = 0.0) -> float:
        """The fly's forward/back stick from the last brain tick (fly.forward: true; the caller clamps and
        gates it). Returns `fallback` if the fly has failed or forward drive is off."""
        if self.failed or not self.fwd_enabled:
            return fallback
        try:
            return float(self.steer.forward)
        except Exception as e:  # noqa: BLE001
            self.failed = True
            _warn_once(f"run{id(self)}", f"fly forward failed ({type(e).__name__}: {e}); back to the PID laws")
            return fallback
