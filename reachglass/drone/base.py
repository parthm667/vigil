"""Drone interface. The Tello adapter, the dry-run adapter and the simulator all implement it.

Two ways to move, never mixed at the same time:
  rc(lr, fb, ud, yaw)     continuous stick values -100..100 (follow, visual servoing). Send at 10-20 Hz.
  move(dir, cm) / rotate(deg)
                          discrete, closed-loop on the drone (the Tello's own optical-flow hold makes
                          these our odometry). NON-BLOCKING: poll busy(); last_result() is 'ok' or an error.
While busy(), rc() calls are ignored (the Tello would abort the move). stop() hovers immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..sources.base import FrameSource
from ..types import Telemetry

DIRECTIONS = ("forward", "back", "left", "right", "up", "down")


class Drone(ABC):
    name = "drone"

    def connect(self) -> None:  # noqa: B027 - optional
        pass

    @abstractmethod
    def takeoff(self) -> None:
        """Start a takeoff (busy() until the drone confirms)."""

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def emergency(self) -> None:
        """Motors off NOW. The drone falls."""

    @abstractmethod
    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        """Stick command, each -100..100 (+ = right / forward / up / clockwise)."""

    @abstractmethod
    def move(self, direction: str, cm: int) -> None:
        """Discrete move of 20..500 cm in the body frame (non-blocking)."""

    @abstractmethod
    def rotate(self, deg: int) -> None:
        """Discrete rotation, + clockwise, |deg| 1..360 (non-blocking)."""

    @abstractmethod
    def stop(self) -> None:
        """Hover in place now (also cancels a discrete move)."""

    @abstractmethod
    def busy(self) -> bool:
        """A discrete command (takeoff/move/rotate/land) is still running."""

    @abstractmethod
    def last_result(self) -> str | None:
        """Reply to the last discrete command: 'ok', an error string, or None while busy."""

    @abstractmethod
    def telemetry(self) -> Telemetry: ...

    @abstractmethod
    def frame_source(self) -> FrameSource: ...

    @property
    @abstractmethod
    def flying(self) -> bool: ...

    def set_speed(self, cm_s: int) -> None:  # noqa: B027 - optional
        pass

    def close(self) -> None:  # noqa: B027 - optional
        pass


def check_move(direction: str, cm: int) -> int:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    cm = int(round(cm))
    if not 20 <= cm <= 500:
        raise ValueError(f"move distance must be 20..500 cm, got {cm}")
    return cm


def check_rotate(deg: float) -> int:
    d = int(round(deg))
    if d == 0 or abs(d) > 360:
        raise ValueError(f"rotation must be 1..360 deg in magnitude, got {deg}")
    return d


def clamp_rc(v: float, limit: int = 100) -> int:
    return int(max(-limit, min(limit, round(v))))
