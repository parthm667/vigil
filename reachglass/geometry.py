"""Pinhole camera model: pixels <-> angles, ranges from known object sizes, ground-plane distances.

The Tello's forward camera is fixed to the body: when the drone pitches forward to accelerate, the
image tilts with it. Functions that depend on the vertical angle take an optional `body_pitch_deg`
(telemetry pitch, nose-up positive) so callers can correct for it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    pitch_deg: float = 0.0  # camera tilt relative to the body, up positive (Tello: ~0, measure it)

    @classmethod
    def from_fov(cls, width: int, height: int, dfov_deg: float | None = None, hfov_deg: float | None = None,
                 pitch_deg: float = 0.0) -> CameraModel:
        """Square pixels, principal point at the image centre. Give either the diagonal or horizontal FOV."""
        if (dfov_deg is None) == (hfov_deg is None):
            raise ValueError("give exactly one of dfov_deg / hfov_deg")
        if hfov_deg is not None:
            f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        else:
            f = (math.hypot(width, height) / 2.0) / math.tan(math.radians(dfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0, pitch_deg)

    def scaled(self, width: int, height: int) -> CameraModel:
        """Same camera at another resolution (e.g. a downscaled frame)."""
        sx, sy = width / self.width, height / self.height
        return CameraModel(width, height, self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, self.pitch_deg)

    def for_frame(self, width: int, height: int) -> CameraModel:
        return self if (width, height) == (self.width, self.height) else self.scaled(width, height)

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.fy))

    # ------------------------------------------------------------------ angles
    def ray(self, x: float, y: float, body_pitch_deg: float = 0.0) -> tuple[float, float, float]:
        """Direction of pixel (x, y) in the drone's LEVEL frame: (forward, right, up), not normalised.
        Exact for off-axis pixels and includes camera + body pitch."""
        r, u = (x - self.cx) / self.fx, -(y - self.cy) / self.fy
        p = math.radians(self.pitch_deg + body_pitch_deg)
        return math.cos(p) - u * math.sin(p), r, math.sin(p) + u * math.cos(p)

    def bearing_deg(self, x: float) -> float:
        """Horizontal angle of pixel column x from the optical axis, + right (level camera)."""
        return math.degrees(math.atan((x - self.cx) / self.fx))

    def elevation_deg(self, y: float, body_pitch_deg: float = 0.0, x: float | None = None) -> float:
        """Angle of pixel (x, y) above the horizon, + up. x defaults to the image centre column."""
        f, r, u = self.ray(self.cx if x is None else x, y, body_pitch_deg)
        return math.degrees(math.atan2(u, math.hypot(f, r)))

    def x_of_bearing(self, bearing_deg: float) -> float:
        return self.cx + self.fx * math.tan(math.radians(bearing_deg))

    def y_of_elevation(self, elevation_deg: float, body_pitch_deg: float = 0.0) -> float:
        return self.cy - self.fy * math.tan(math.radians(elevation_deg - self.pitch_deg - body_pitch_deg))

    # ------------------------------------------------------------------ ranges
    def range_from_height(self, pixel_h: float, real_h_m: float, x: float | None = None) -> float | None:
        """Horizontal distance to an upright object of known height from its pixel height.

        pixel_h / fy = real_h / depth, and the horizontal distance is depth / cos(bearing).
        """
        if pixel_h <= 1.0 or real_h_m <= 0:
            return None
        depth = self.fy * real_h_m / pixel_h
        b = 0.0 if x is None else math.radians(self.bearing_deg(x))
        return depth / math.cos(b)

    def range_from_width(self, pixel_w: float, real_w_m: float, x: float | None = None) -> float | None:
        if pixel_w <= 1.0 or real_w_m <= 0:
            return None
        depth = self.fx * real_w_m / pixel_w
        b = 0.0 if x is None else math.radians(self.bearing_deg(x))
        return depth / math.cos(b)

    def range_from_elevation(self, elevation_deg: float, height_diff_m: float) -> float | None:
        """Horizontal distance to a point `height_diff_m` below the camera (negative = above) seen at
        `elevation_deg`. Returns None when the geometry is degenerate (point near the horizon)."""
        dep = -elevation_deg if height_diff_m > 0 else elevation_deg
        if dep < 1.5:  # below ~1.5 deg the estimate explodes
            return None
        return abs(height_diff_m) / math.tan(math.radians(dep))

    def range_to_height(self, x: float, y: float, height_diff_m: float, body_pitch_deg: float = 0.0,
                        min_angle_deg: float = 1.5) -> float | None:
        """Horizontal distance to the point imaged at (x, y), knowing it lies `height_diff_m` below the
        camera (negative = above). None if the ray does not reach that height at a usable angle."""
        f, r, u = self.ray(x, y, body_pitch_deg)
        horiz = math.hypot(f, r)
        if height_diff_m == 0 or u * height_diff_m >= 0:  # ray goes the wrong way
            return None
        if math.degrees(math.atan2(abs(u), horiz)) < min_angle_deg:
            return None
        return abs(height_diff_m) / abs(u) * horiz

    def ground_distance(self, y: float, camera_height_m: float, body_pitch_deg: float = 0.0,
                        x: float | None = None) -> float | None:
        """Horizontal distance to the floor point imaged at (x, y), or None if it is above the horizon."""
        return self.range_to_height(self.cx if x is None else x, y, camera_height_m, body_pitch_deg)


# The Tello's 960 x 720 VIDEO stream is cropped relative to the 82.6 deg stills spec: published calibrations
# give f ~ 920 px (~55 x 43 deg). Measure your own drone (see config.CameraCfg) and set it in the config.
TELLO_DFOV_DEG = 82.6
TELLO_STREAM_F = 920.0


def tello_camera(width: int = 960, height: int = 720, f: float | None = TELLO_STREAM_F, dfov_deg: float = TELLO_DFOV_DEG,
                 pitch_deg: float = 0.0) -> CameraModel:
    """Default Tello stream model at 960x720, scaled to (width, height). f=None -> from dfov_deg."""
    if f is None:
        base = CameraModel.from_fov(960, 720, dfov_deg=dfov_deg, pitch_deg=pitch_deg)
    else:
        base = CameraModel(960, 720, f, f, 480.0, 360.0, pitch_deg)
    return base.for_frame(width, height)


def camera_from_config(cc) -> CameraModel:
    """Build the camera model from a config.CameraCfg."""
    w, h = cc.ref_width, cc.ref_height
    if cc.fx is None:
        return CameraModel.from_fov(w, h, dfov_deg=cc.dfov_deg, pitch_deg=cc.pitch_deg)
    fy = cc.fy if cc.fy is not None else cc.fx
    cx = cc.cx if cc.cx is not None else w / 2.0
    cy = cc.cy if cc.cy is not None else h / 2.0
    return CameraModel(w, h, float(cc.fx), float(fy), float(cx), float(cy), cc.pitch_deg)
