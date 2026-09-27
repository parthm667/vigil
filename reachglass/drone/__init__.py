"""Drones. All implement drone.base.Drone; wrap any of them in SafetyGovernor before flying."""

from .base import DIRECTIONS, Drone, check_move, check_rotate, clamp_rc
from .safety import SafetyGovernor


def make_tello(cfg, dry_run: bool = False):
    """Tello adapter from config (import deferred: djitellopy pulls in PyAV)."""
    from .tello import TelloDrone

    d = cfg.drone
    if d.kind not in ("tello", "dry_run"):
        raise ValueError(f"drone.kind '{d.kind}' cannot be used with a real Tello (use tello or dry_run)")
    dry = dry_run or d.kind == "dry_run"  # either the --dry-run flag or the config makes it a dry run
    return TelloDrone(dry_run=dry, move_speed_cm_s=d.move_speed_cm_s, yaw_sign=d.yaw_sign, video_fps=d.video_fps,
                      video_backend=d.video_backend)


__all__ = ["Drone", "SafetyGovernor", "DIRECTIONS", "check_move", "check_rotate", "clamp_rc", "make_tello"]
