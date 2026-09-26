"""Shared data types.

Conventions used by every module
--------------------------------
World (map) frame: origin = drone position when the mission starts. x = forward (the drone's heading at
that moment), y = to the RIGHT, heading in degrees, CLOCKWISE positive (like a compass and like the
Tello's `cw` command). A point at distance d along world heading h is (d*cos h, d*sin h).

Camera: image x to the right, y down, pixels. Bearing = horizontal angle from the optical axis, positive to
the RIGHT. Elevation = vertical angle, positive UP. So the world heading of a seen object is
drone heading + bearing.

Times: time.time() seconds. Distances: metres. Angles: degrees unless a name ends in _rad.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


@dataclass
class Frame:
    image: np.ndarray  # H x W x 3, uint8, BGR
    t: float  # time the frame became available on the laptop
    seq: int  # increases by one for every new frame from the source
    source: str = ""

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    keypoints: np.ndarray | None = None  # (K, 3): x, y, confidence; COCO-17 order for people
    track_id: int | None = None
    source: str = ""  # detector that produced it

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def w(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def h(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.w * self.h


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Telemetry:
    """One state packet from the drone, already converted to SI units."""

    t: float
    yaw_deg: float | None = None  # IMU yaw as reported by the drone
    pitch_deg: float | None = None
    roll_deg: float | None = None
    height_m: float | None = None  # 'h': height above the takeoff point
    tof_m: float | None = None  # 'tof': downward range to whatever is below
    vx: float | None = None  # state-stream speeds, m/s (frame as reported by the drone)
    vy: float | None = None
    vz: float | None = None
    battery_pct: float | None = None
    flight_time_s: float | None = None

    def altitude_m(self, prefer_tof: bool = True, tof_max_m: float = 3.0) -> float | None:
        """Clearance above whatever is BELOW (downward ToF when in range, else 'h'). Over a table this is
        the height above the table: right for descending, wrong for 'how high above the floor am I'."""
        if prefer_tof and self.tof_m is not None and 0.1 <= self.tof_m <= tof_max_m:
            return self.tof_m
        return self.height_m

    def floor_altitude_m(self, agree_m: float = 0.2) -> float | None:
        """Height above the FLOOR: the ToF only when it agrees with 'h' (i.e. there is floor below),
        otherwise 'h' (height above the takeoff point, which was on the floor)."""
        if self.height_m is None:
            return self.altitude_m()
        if self.tof_m is not None and abs(self.tof_m - self.height_m) <= agree_m:
            return self.tof_m
        return self.height_m


@dataclass
class Pose2D:
    """Drone pose in the world frame (see module docstring)."""

    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0

    def point_at(self, distance_m: float, bearing_deg: float) -> tuple[float, float]:
        """World position of something seen at `distance_m` along camera bearing `bearing_deg`."""
        h = math.radians(self.heading_deg + bearing_deg)
        return self.x + distance_m * math.cos(h), self.y + distance_m * math.sin(h)

    def bearing_to(self, x: float, y: float) -> float:
        """Camera bearing (deg, + right) at which the world point (x, y) appears."""
        return wrap_deg(math.degrees(math.atan2(y - self.y, x - self.x)) - self.heading_deg)

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(x - self.x, y - self.y)

    def moved(self, forward_m: float = 0.0, right_m: float = 0.0, turn_deg: float = 0.0) -> Pose2D:
        """Pose after a body-frame translation followed by a clockwise turn."""
        h = math.radians(self.heading_deg)
        x = self.x + forward_m * math.cos(h) - right_m * math.sin(h)
        y = self.y + forward_m * math.sin(h) + right_m * math.cos(h)
        return Pose2D(x, y, wrap_deg(self.heading_deg + turn_deg))


@dataclass
class ObjectObs:
    """A detection interpreted geometrically from the drone's camera."""

    det: Detection
    bearing_deg: float
    elevation_deg: float
    range_m: float | None = None  # horizontal distance from the drone
    range_src: str = ""

    @property
    def cls(self) -> str:
        return self.det.cls


@dataclass
class PersonObs(ObjectObs):
    """facing_deg: direction the person faces, relative to the ray from the camera through them,
    clockwise positive seen from above. 0 = they face away from us (we see their back, i.e. we are
    behind them), +90 = they face toward image-right, +/-180 = they face the camera."""

    facing_deg: float | None = None
    facing_conf: float = 0.0
    range_lo_m: float | None = None  # smallest of the range cues: the safe value before moving toward them


@dataclass
class TargetObs(ObjectObs):
    confirmed: bool = False  # passed the target lock's confirmation rule


@dataclass
class PerceptionResult:
    """Everything perception knows about one frame. Behaviours read only this (plus telemetry/pose)."""

    t: float  # frame time
    seq: int
    latency_ms: float  # processing time for this frame
    image_size: tuple[int, int]  # (width, height)
    detections: list[Detection] = field(default_factory=list)
    persons: list[PersonObs] = field(default_factory=list)
    person: PersonObs | None = None  # the locked person being followed
    targets: list[TargetObs] = field(default_factory=list)
    target: TargetObs | None = None  # the locked target object
    context: list[ObjectObs] = field(default_factory=list)  # other objects (furniture...) for exploration
    ran: dict = field(default_factory=dict)  # role -> did that detector run on this frame
    # did the detector that reports the target run on this frame? (it skips frames; "no target" on a frame it
    # did not look at means nothing. True when there is no such detector: nothing to wait for)
    target_ran: bool = True
    person_unseen_s: float = float("inf")  # time since the locked person was last seen
    target_unseen_s: float = float("inf")
