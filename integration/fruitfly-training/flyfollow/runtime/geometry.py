"""Camera geometry for the drone runtime (plan 1b, 3.4, 5.3, 5.4): pure functions, no state, no bus.

Used by the control loop, the mission logic (FIND, APPROACH handoff, scene) and guidance.

Conventions (flyfollow.interfaces): pixels of the 960x720 Tello frame, x right, y down. Bearing + = right
of the camera axis, elevation + = up. drone_level frame: x forward, y left, z up, pitch and roll removed.
Meters and degrees at the API; radians only inside.

Attitude: functions take pitch_deg (+ = nose up) and roll_deg (+ = right side down), the aerospace
convention. tello_attitude() maps a tello_state message onto it with TELLO_PITCH_SIGN / TELLO_ROLL_SIGN.
UNVERIFIED: the Tello's own sign. Check at R0 by tilting the drone nose down by hand: pitch_deg must go
negative with TELLO_PITCH_SIGN = 1 (right side down: roll_deg positive). Flip the constant if not.

Range from size is depth along the optical axis, fy * size / h; for the small elevations of a head at follow
distance it equals the horizontal range to within a few percent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from flyfollow.interfaces import IMG_H, IMG_W, configs_dir
from flyfollow.runtime.messages import SETTINGS_DEFAULTS

TELLO_PITCH_SIGN = 1.0  # tello pitch_deg * this = nose-up degrees (unverified, see module doc)
TELLO_ROLL_SIGN = 1.0  # tello roll_deg * this = right-side-down degrees (unverified)
MAX_TILT_FOR_RANGE_DEG = 8.0  # plan 4.9: do not trust size range on frames with |pitch| above this

# ---------------------------------------------------------------------------------------------- size priors
# Vertical extent in the image (m) as the object usually stands, and the relative 1-sigma error of the prior.
# Research: class priors are off by about 20 to 25 %; the head prior (0.20 to 0.26 m) by about 12 %.
SIZE_SD_FRAC_DEFAULT = 0.25
CLASS_SIZE_M: dict[str, tuple[float, float]] = {
    # people
    "person_head": (0.23, 0.12),
    "person": (1.70, 0.08),
    # FIND targets (COCO names where they exist)
    "bottle": (0.22, 0.25),
    "cup": (0.10, 0.25),
    "cell phone": (0.15, 0.30),
    "backpack": (0.45, 0.25),
    "laptop": (0.25, 0.25),
    "book": (0.23, 0.30),
    "remote": (0.18, 0.25),
    "keys": (0.08, 0.35),
    "wallet": (0.10, 0.30),
    "glasses": (0.05, 0.35),
    "handbag": (0.30, 0.30),
    "wine glass": (0.18, 0.20),
    "bowl": (0.08, 0.30),
    "vase": (0.25, 0.35),
    "scissors": (0.18, 0.25),
    "teddy bear": (0.30, 0.35),
    "mouse": (0.04, 0.30),
    "keyboard": (0.04, 0.40),
    "sports ball": (0.22, 0.30),
    "clock": (0.30, 0.30),
    "umbrella": (0.90, 0.30),
    "toothbrush": (0.19, 0.20),
    "headphones": (0.18, 0.25),
    "apple": (0.08, 0.20),
    "orange": (0.08, 0.20),
    "banana": (0.18, 0.25),
    # furniture and landmarks (FIND hop ranking; mostly partly visible, so wide errors)
    "chair": (0.90, 0.20),
    "couch": (0.85, 0.20),
    "bed": (0.55, 0.30),
    "dining table": (0.75, 0.15),
    "table": (0.75, 0.15),
    "desk": (0.75, 0.15),
    "counter": (0.90, 0.10),
    "tv": (0.55, 0.35),
    "refrigerator": (1.75, 0.10),
    "toilet": (0.75, 0.15),
    "sink": (0.20, 0.40),
    "bench": (0.45, 0.25),
    "potted plant": (0.45, 0.40),
    "suitcase": (0.60, 0.25),
    "door": (2.00, 0.05),
    "cat": (0.25, 0.30),
    "dog": (0.50, 0.40),
}
CLASS_ALIASES = {
    "head": "person_head",
    "phone": "cell phone",
    "mobile phone": "cell phone",
    "cellphone": "cell phone",
    "smartphone": "cell phone",
    "water bottle": "bottle",
    "mug": "cup",
    "coffee cup": "cup",
    "glass": "wine glass",
    "remote control": "remote",
    "tv remote": "remote",
    "key": "keys",
    "car keys": "keys",
    "bag": "backpack",
    "sofa": "couch",
    "fridge": "refrigerator",
    "television": "tv",
    "tvmonitor": "tv",
    "plant": "potted plant",
    "sunglasses": "glasses",
    "eyeglasses": "glasses",
}


def canonical_class(cls: str | None) -> str:
    """Normalize a class or spoken prompt: lowercase, '_' -> ' ' (except person_head), aliases applied."""
    c = (cls or "").strip().lower()
    if c in CLASS_SIZE_M:
        return c
    c = c.replace("_", " ") if c != "person_head" else c
    if c in ("person head",):
        return "person_head"
    return CLASS_ALIASES.get(c, c)


def class_size(cls: str | None) -> tuple[float, float] | None:
    """(size_m, sd_frac) prior for a class or prompt, or None if unknown."""
    return CLASS_SIZE_M.get(canonical_class(cls))


def class_size_m(cls: str | None, default: float | None = None) -> float | None:
    p = class_size(cls)
    return p[0] if p else default


# ---------------------------------------------------------------------------------------------- intrinsics
@dataclass(frozen=True)
class Intrinsics:
    fx: float = float(SETTINGS_DEFAULTS["fx"])
    fy: float = float(SETTINGS_DEFAULTS["fy"])
    cx: float = float(SETTINGS_DEFAULTS["cx"])
    cy: float = float(SETTINGS_DEFAULTS["cy"])
    w: int = IMG_W
    h: int = IMG_H

    @property
    def hfov_deg(self) -> float:
        return math.degrees(math.atan(self.cx / self.fx) + math.atan((self.w - self.cx) / self.fx))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(math.atan(self.cy / self.fy) + math.atan((self.h - self.cy) / self.fy))

    @property
    def lower_half_vfov_deg(self) -> float:
        """Angle from the optical axis down to the bottom image edge."""
        return math.degrees(math.atan((self.h - self.cy) / self.fy))

    @classmethod
    def from_settings(cls, s: dict) -> Intrinsics:
        return cls(float(s.get("fx", cls.fx)), float(s.get("fy", cls.fy)), float(s.get("cx", cls.cx)), float(s.get("cy", cls.cy)))

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}


DEFAULT_K = Intrinsics()  # SETTINGS_DEFAULTS intrinsics (pass load_intrinsics() for the calibrated ones)


def camera_json_path() -> Path:
    return configs_dir() / "camera.json"


def load_camera_json(path: str | Path | None = None) -> dict:
    """fx, fy, cx, cy from configs/camera.json ({} if absent or unreadable).

    Accepts flat keys {"fx", "fy", "cx", "cy"} or an OpenCV "camera_matrix" / "K" 3x3 list. A calibration made
    at another resolution ("width"/"height" keys) is scaled to 960x720.
    """
    p = Path(path) if path else camera_json_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict = {}
    k = d.get("camera_matrix") or d.get("K")
    if k is not None:
        out = {"fx": k[0][0], "fy": k[1][1], "cx": k[0][2], "cy": k[1][2]}
    for key in ("fx", "fy", "cx", "cy"):
        if key in d:
            out[key] = d[key]
    sx = IMG_W / float(d.get("width", IMG_W))
    sy = IMG_H / float(d.get("height", IMG_H))
    for key, s in (("fx", sx), ("cx", sx), ("fy", sy), ("cy", sy)):
        if key in out:
            out[key] = float(out[key]) * s
    return out


def camera_defaults(path: str | Path | None = None) -> dict:
    """SETTINGS_DEFAULTS intrinsics overridden by configs/camera.json when present (operator settings win later)."""
    out = {k: float(SETTINGS_DEFAULTS[k]) for k in ("fx", "fy", "cx", "cy")}
    out.update(load_camera_json(path))
    return out


def load_intrinsics(path: str | Path | None = None, overrides: dict | None = None) -> Intrinsics:
    d = camera_defaults(path)
    d.update({k: float(v) for k, v in (overrides or {}).items() if k in ("fx", "fy", "cx", "cy") and v is not None})
    return Intrinsics.from_settings(d)


# ---------------------------------------------------------------------------------------------- attitude
def tello_attitude(state: dict | None) -> tuple[float, float]:
    """(pitch_deg nose-up, roll_deg right-down) from a tello_state message; (0, 0) when missing."""
    if not state:
        return 0.0, 0.0
    return TELLO_PITCH_SIGN * float(state.get("pitch_deg", 0.0) or 0.0), TELLO_ROLL_SIGN * float(state.get("roll_deg", 0.0) or 0.0)


def tilt_ok(pitch_deg: float, roll_deg: float = 0.0, max_deg: float = MAX_TILT_FOR_RANGE_DEG) -> bool:
    return abs(pitch_deg) <= max_deg and abs(roll_deg) <= max_deg


# ---------------------------------------------------------------------------------------------- rays
def pixel_ray(u: float, v: float, k: Intrinsics = DEFAULT_K, pitch_deg: float = 0.0, roll_deg: float = 0.0,
              cam_tilt_deg: float = 0.0) -> tuple[float, float, float]:
    """Unit ray through pixel (u, v) in the drone_level frame (x forward, y left, z up).

    cam_tilt_deg: fixed downward mount tilt of the camera relative to the body (0 for the Tello, unverified).
    """
    xb, yb, zb = 1.0, -(u - k.cx) / k.fx, -(v - k.cy) / k.fy
    if cam_tilt_deg:
        t = math.radians(-cam_tilt_deg)  # a down tilt acts like nose down
        ct, st = math.cos(t), math.sin(t)
        xb, zb = ct * xb - st * zb, st * xb + ct * zb
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    cr, sr, cp, sp = math.cos(r), math.sin(r), math.cos(p), math.sin(p)
    y1, z1 = cr * yb - sr * zb, sr * yb + cr * zb  # roll: body left axis -> (0, cos r, sin r)
    x2, z2 = cp * xb - sp * z1, sp * xb + cp * z1  # pitch: body forward axis -> (cos p, 0, sin p)
    n = math.sqrt(x2 * x2 + y1 * y1 + z2 * z2)
    return x2 / n, y1 / n, z2 / n


def bearing_elevation(u: float, v: float, k: Intrinsics = DEFAULT_K, pitch_deg: float = 0.0,
                      roll_deg: float = 0.0) -> tuple[float, float]:
    """(bearing_deg + right, elevation_deg + up) of pixel (u, v) in the drone_level frame."""
    x, y, z = pixel_ray(u, v, k, pitch_deg, roll_deg)
    return math.degrees(math.atan2(-y, x)), math.degrees(math.atan2(z, math.hypot(x, y)))


def bearing_deg(u: float, k: Intrinsics = DEFAULT_K) -> float:
    """Camera-frame bearing of image column u, no attitude correction (what the controllers use)."""
    return math.degrees(math.atan((u - k.cx) / k.fx))


def pixel_of(bearing_deg_: float, elevation_deg: float = 0.0, k: Intrinsics = DEFAULT_K) -> tuple[float, float]:
    """Inverse of bearing_elevation for a level camera: (u, v) of a direction."""
    b, e = math.radians(bearing_deg_), math.radians(elevation_deg)
    x, y, z = math.cos(e) * math.cos(b), -math.cos(e) * math.sin(b), math.sin(e)
    return k.cx - k.fx * y / x, k.cy - k.fy * z / x


# ---------------------------------------------------------------------------------------------- range
def range_from_size(h_px: float, size_m: float, fy: float = float(SETTINGS_DEFAULTS["fy"])) -> float:
    """Pinhole depth Z = fy * size / h. inf for a zero-height box."""
    return fy * size_m / h_px if h_px > 0 else math.inf


def size_range_sd(z_m: float, h_px: float, sd_frac: float, box_sd_px: float = 3.0) -> float:
    """1-sigma range error: size prior error and box height noise, added in quadrature."""
    if not math.isfinite(z_m) or h_px <= 0:
        return math.inf
    return z_m * math.hypot(sd_frac, box_sd_px / h_px)


def range_from_class(h_px: float, cls: str, fy: float = float(SETTINGS_DEFAULTS["fy"])) -> tuple[float, float] | None:
    """(range_m, range_sd_m) from the class size prior, None for an unknown class."""
    p = class_size(cls)
    if p is None:
        return None
    z = range_from_size(h_px, p[0], fy)
    return z, size_range_sd(z, h_px, p[1])


def ground_range(u: float, v_base: float, cam_height_m: float, k: Intrinsics = DEFAULT_K, pitch_deg: float = 0.0,
                 roll_deg: float = 0.0, min_depression_deg: float = 0.5) -> float | None:
    """Horizontal range to a floor point seen at pixel (u, v_base) (a foot or an object base) from camera height.

    None when the pixel is at or above the horizon (no floor intersection).
    """
    _, el = bearing_elevation(u, v_base, k, pitch_deg, roll_deg)
    if -el < min_depression_deg or cam_height_m <= 0:
        return None
    return cam_height_m / math.tan(math.radians(-el))


def user_height_range(u: float, v_head_top: float, v_feet: float, user_height_m: float, k: Intrinsics = DEFAULT_K,
                      pitch_deg: float = 0.0, roll_deg: float = 0.0) -> tuple[float, float] | None:
    """Plan 5.4: (range Z, camera height a) from the user's known height and the pixel rows of feet and head top.

    Z = H_p / (tan(alpha_f) - tan(alpha_h)), a = Z tan(alpha_f), alpha measured below the horizon (alpha_h < 0 when
    the head is above the camera). None when the rows are degenerate (feet not below the head).
    """
    _, el_f = bearing_elevation(u, v_feet, k, pitch_deg, roll_deg)
    _, el_h = bearing_elevation(u, v_head_top, k, pitch_deg, roll_deg)
    tf, th = math.tan(math.radians(-el_f)), math.tan(math.radians(-el_h))
    d = tf - th
    if d <= 1e-6:
        return None
    z = user_height_m / d
    return z, z * tf


def height_above_floor(u: float, v: float, range_m: float, cam_height_m: float, k: Intrinsics = DEFAULT_K,
                       pitch_deg: float = 0.0, roll_deg: float = 0.0) -> float:
    """Height of the point seen at (u, v) at horizontal range_m, for a camera cam_height_m above the floor."""
    _, el = bearing_elevation(u, v, k, pitch_deg, roll_deg)
    return cam_height_m + range_m * math.tan(math.radians(el))


def floor_visible_min_range(alt_m: float, k: Intrinsics = DEFAULT_K, pitch_deg: float = 0.0) -> float:
    """Closest floor distance in view: alt / tan(lower half VFOV + nose-down pitch) (plan 3.4). inf if none visible."""
    a = math.radians(k.lower_half_vfov_deg - pitch_deg)
    if a <= 0:
        return math.inf
    return max(0.0, alt_m) / math.tan(a)


def drone_level_xy(bearing_deg_: float, range_m: float) -> tuple[float, float]:
    """(x forward, y left) in meters of a point at bearing (+ right) and horizontal range."""
    b = math.radians(bearing_deg_)
    return range_m * math.cos(b), -range_m * math.sin(b)


def xy_to_bearing_range(x: float, y: float) -> tuple[float, float]:
    return math.degrees(math.atan2(-y, x)), math.hypot(x, y)


def wrap_deg(a: float) -> float:
    """Wrap to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


# ---------------------------------------------------------------------------------------------- boxes
def box_center_h(bbox) -> tuple[float, float, float]:
    """[x1, y1, x2, y2] -> (cx, cy, h)."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), max(0.0, y2 - y1)


def head_box_from_person(bbox, user_height_m: float = float(SETTINGS_DEFAULTS["user_height_m"]),
                         head_size_m: float = float(SETTINGS_DEFAULTS["head_size_m"]), img_h: int = IMG_H,
                         shoulder_w_m: float = 0.45) -> tuple[float, float, float]:
    """Head (cx, cy, h) guessed from a person box: the top of the box scaled to a head.

    Full body in view: head h = person h * head_size / user_height. Feet cut off by the bottom edge (usual at
    follow distance): head h = person width * head_size / shoulder width. Less accurate than a head detector
    (about 20 to 30 %), so callers flag it.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cut = y2 >= img_h - 3
    hh = w * head_size_m / shoulder_w_m if cut else h * head_size_m / user_height_m
    hh = min(hh, h)
    return 0.5 * (x1 + x2), y1 + 0.5 * hh, hh


def approach_z_ref(base_h_m: float, center_h_m: float, cfg: dict | None = None) -> float:
    """Plan 5.3 standoff for an object, with our config (configs/env.yaml approach: factor 3.0, 1.0 to 2.4 m)."""
    from flyfollow.sim.objects import approach_z_ref as _z

    if cfg is None:
        import yaml

        cfg = (yaml.safe_load((configs_dir() / "env.yaml").read_text(encoding="utf-8")) or {}).get("approach", {})
    return _z(base_h_m, center_h_m, cfg)


__all__ = [
    "CLASS_SIZE_M", "Intrinsics", "approach_z_ref", "bearing_deg", "bearing_elevation", "box_center_h", "camera_defaults",
    "canonical_class", "class_size", "class_size_m", "drone_level_xy", "floor_visible_min_range", "ground_range",
    "head_box_from_person", "height_above_floor", "load_intrinsics", "pixel_of", "pixel_ray", "range_from_class",
    "range_from_size", "size_range_sd", "tello_attitude", "tilt_ok", "user_height_range", "wrap_deg", "xy_to_bearing_range",
]
