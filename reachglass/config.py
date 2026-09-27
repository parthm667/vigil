"""Configuration: nested dataclasses with defaults, overridable from a YAML file or a dict.

    cfg = load_config("my_room.yaml")          # defaults + file
    cfg = load_config(overrides={"follow": {"distance_m": 1.8}})
    cfg = load_config("site.yaml", target="color")   # a target preset on top (TARGET_PRESETS)

Unknown keys raise, so a typo in a YAML file fails loudly instead of being ignored.
Each module reads only its own section.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints


# How the target (the team's blue Hydro Flask: 24 cm tall with its black cap, 9 cm wide) is found. The default
# is "yolo-world"; "color" is the earlier colour-blob dummy. Switch with `--target color` or load_config(target=).
# Measured on the team's photos placed at Tello-like distances (960 px input): YOLO-World finds it in 95-100 %
# of views at 1.3-4 m (60 % at 6 m), nothing else in the rooms scores even 0.05; "blue water bottle" alone
# still accepts a green or grey bottle, so the blue check on each box keeps only the team's bottle (>= 80 % of
# a box's middle is blue for it, <= 24 % for other colours). 13 ms/frame on a Mac GPU, ~100 ms on a CPU.
YOLO_WORLD_BOTTLE = {
    "weights": "yolov8s-worldv2.pt",
    "prompts": ["blue water bottle", "hydro flask water bottle"],
    "rename": {"blue water bottle": "bottle", "hydro flask water bottle": "bottle"},
    "classes": ["bottle"],
    "conf": 0.2,
    "imgsz": 960,  # the stream's own width: far (small) bottles keep their pixels
    "agnostic_nms": True,  # two prompts, one bottle: one box
    "require_color": {"hsv_ranges": [[95, 60, 40, 130, 255, 255]], "min_frac": 0.3},
}
# The colour dummy: the bottle's blue measured from a photo (hue 108-114, saturation ~165, brightness 85 on
# the shadow side to 230). Greyish blues (mesh chairs, windows) have S < 100; blue jeans/shirts can match, the
# person guard drops blobs inside a detected person's box.
COLOR_BOTTLE = {"label": "bottle", "hsv_ranges": [[100, 100, 70, 122, 255, 255]], "min_area_frac": 0.0003,
                "min_fill": 0.35, "max_aspect": 6.0}
TARGET_PRESETS = {
    "yolo-world": {"perception": {"target_detector": {"kind": "yolo", "params": YOLO_WORLD_BOTTLE},
                                  "object_heights_m": {"bottle": 0.24}, "object_widths_m": {"bottle": 0.09}},
                   "tracking": {"confirm_conf": 0.3}},
    # the colour blob sees only the blue body (not the cap / steel ring): 0.19 m
    "color": {"perception": {"target_detector": {"kind": "color_blob", "params": COLOR_BOTTLE},
                             "object_heights_m": {"bottle": 0.19}, "object_widths_m": {"bottle": 0.09}},
              "tracking": {"confirm_conf": 0.55}},
}


@dataclass
class ComponentSpec:
    """Which implementation to use for a pluggable part, and its constructor arguments."""

    kind: str = ""
    params: dict = field(default_factory=dict)


@dataclass
class CameraCfg:
    """Intrinsics of the VIDEO stream at ref_width x ref_height (scaled to whatever frame size arrives).

    The 82.6 deg spec is for stills; the 960x720 stream is cropped. Published calibrations of the stream
    (tello_ros camera_info: fx 921, fy 919; tello-ros2 ost.yaml: fx 919, fy 912) give f ~ 920 px, i.e.
    ~55 x 43 deg. Measure yours: object of known height h at a taped distance d -> f = pixel_height * d / h.
    Set fx to null to fall back to dfov_deg.
    """

    fx: float | None = 920.0
    fy: float | None = 920.0
    cx: float | None = None  # None = image centre
    cy: float | None = None
    ref_width: int = 960
    ref_height: int = 720
    dfov_deg: float = 82.6  # only used when fx is null
    pitch_deg: float = 0.0  # camera tilt relative to the body, up positive


@dataclass
class PerceptionCfg:
    # The target: the team's blue water bottle, reported as "bottle" so "find my water bottle" works.
    # Default = the "yolo-world" preset (see TARGET_PRESETS above); `--target color` = the colour blob.
    target_detector: ComponentSpec = field(default_factory=lambda: ComponentSpec("yolo", copy.deepcopy(YOLO_WORLD_BOTTLE)))
    person_detector: ComponentSpec = field(default_factory=lambda: ComponentSpec("yolo", {
        "weights": "yolo11n-pose.pt", "classes": ["person"], "conf": 0.4, "imgsz": 640}))
    # Furniture etc. for the exploration prior; set kind "" to disable.
    context_detector: ComponentSpec = field(default_factory=lambda: ComponentSpec("yolo", {
        "weights": "yolo11n.pt", "classes": ["chair", "couch", "bed", "dining table", "tv", "refrigerator",
                                             "sink", "oven", "microwave", "potted plant", "laptop", "bench",
                                             "toilet", "suitcase", "backpack"],
        "conf": 0.35, "imgsz": 640}))
    # Real heights (m) used to turn a pixel height into a distance. Measure your dummy/bottle!
    object_heights_m: dict = field(default_factory=lambda: {
        # bottle: the whole bottle incl. cap, as a YOLO box covers it (the colour preset uses 0.19, the blue part)
        "bottle": 0.24, "cup": 0.10, "backpack": 0.45, "chair": 0.85, "dining table": 0.75, "couch": 0.85,
        "bed": 0.60, "tv": 0.55, "laptop": 0.25, "refrigerator": 1.70, "potted plant": 0.60, "bench": 0.45,
        "sink": 0.20, "oven": 0.85, "microwave": 0.30, "toilet": 0.75, "suitcase": 0.60})
    # Widths (m) of ROUND objects (same width from every side). Used when the object's top or bottom is
    # hidden (e.g. behind a chair): then its pixel height shrinks but its width does not. Measure yours.
    object_widths_m: dict = field(default_factory=lambda: {"bottle": 0.09, "cup": 0.08})
    person_height_m: float = 1.75  # SET THIS to the wearer's real height (it sets the follow distance)
    person_height_sd_m: float = 0.05  # 0.02 once person_height_m is measured
    shoulder_width_m: float = 0.40
    body_width_m: float = 0.48  # bbox width of a person seen from behind
    # which detectors run in which mode (1 = every frame, N = every Nth frame, 0 = never). The target detector
    # skips frames: one detection is enough to confirm, and the scan waits until it has looked at each view.
    stride: dict = field(default_factory=lambda: {
        "follow": {"person": 1, "target": 0, "context": 0},
        "search": {"person": 5, "target": 3, "context": 2},
        "approach": {"person": 3, "target": 2, "context": 0},  # people: person guard (blue jeans) + safety
        "idle": {"person": 0, "target": 0, "context": 0},
    })
    # telemetry pitch sign so that nose-up is positive; 0 = do not use pitch (until checked on the drone)
    pitch_sign: int = 0
    target_range_m: tuple = (0.2, 8.0)  # plausible distances for a single-frame target confirmation


@dataclass
class TrackingCfg:
    iou_match: float = 0.2  # min IoU to associate a detection with a track...
    center_match_frac: float = 0.15  # ...or centre distance below this fraction of the image diagonal
    max_age_s: float = 1.5  # drop a track not seen for this long (>= lost_after_s, so re-locking works)
    confirm_hits: int = 3  # target lock: hits needed...
    confirm_window: int = 5  # ...within this many runs of the detector (it may skip frames)
    confirm_conf: float = 0.3  # or a single detection this confident that also looks right (size, distance)
    lost_after_s: float = 1.5  # locked target / person considered lost after this long unseen


@dataclass
class FollowCfg:
    # Geometry note (Tello camera is fixed, +-21 deg vertical view): at 2.0 m high and 1.0 m behind only the
    # person's HEAD is in frame (shoulders are 29 deg below the camera), so facing / orbit-behind cannot work
    # and detection relies on YOLO seeing a head from above. Shoulders come into view from ~1.6 m behind.
    distance_m: float = 1.0  # horizontal distance behind the person
    altitude_m: float = 2.0  # hover height above the floor: >= person height + 0.2 m (checked at load)
    yaw_gain: float = 1.2  # rc yaw per degree of bearing error
    yaw_deadband_deg: float = 4.0
    dist_gain: float = 40.0  # rc forward per metre of distance error
    dist_deadband_m: float = 0.1
    alt_gain: float = 60.0  # rc up per metre of altitude error
    alt_deadband_m: float = 0.05
    orbit_gain: float = 0.5  # rc lateral per degree of facing error (to get behind the person)
    orbit_deadband_deg: float = 25.0
    max_rc_yaw: int = 40
    max_rc_forward: int = 35  # ~0.35 m/s: keeps up with a slow walk
    max_rc_lateral: int = 30
    max_rc_up: int = 30
    max_rc_backoff: int = 50  # backing away from a person who comes closer (still clamped by safety.max_rc)
    approach_rate_mps: float = 0.2  # they come closer faster than this: back away at their speed right away
    min_range_m: float = 0.55  # the most conservative distance cue below this: back off (and never approach)
    max_box_width_frac: float = 0.75  # person box wider than this fraction of the frame: back off
    search_yaw_rc: int = 20  # yaw speed when the person is lost


@dataclass
class ExploreCfg:
    scan_altitude_m: float = 1.2
    scan_step_deg: int = 45
    dwell_s: float = 0.8  # wait after each rotation before using frames (video lag + settling)
    frames_per_dwell: int = 2  # frames the TARGET detector looked at, per view (it skips frames)
    hop_min_m: float = 0.5  # also the longest step into a direction with no free-space evidence
    hop_max_m: float = 1.5
    max_vantage_points: int = 5
    max_search_s: float = 150.0
    grid_cell_m: float = 0.25
    grid_size_m: float = 16.0
    view_range_m: float = 3.5  # how far a scan "clears" space for a bottle-sized target
    person_clearance_m: float = 1.5  # don't hop toward a person closer than this
    semantic_weights: dict = field(default_factory=lambda: {
        # target class -> {context class: weight}; where is the target likely to be?
        "bottle": {"dining table": 1.0, "desk": 1.0, "refrigerator": 0.6, "sink": 0.6, "couch": 0.3,
                   "chair": 0.3, "tv": 0.2},
        "cup": {"dining table": 1.0, "desk": 1.0, "sink": 0.8, "microwave": 0.5},
        "backpack": {"chair": 0.8, "bed": 0.8, "couch": 0.6, "dining table": 0.4},
        "laptop": {"dining table": 1.0, "desk": 1.0, "couch": 0.5, "bed": 0.4},
    })


@dataclass
class ApproachCfg:
    standoff_m: float = 1.3  # stop this far from the target (a bottle on a table stays in frame at 1.2 m altitude)
    tolerance_m: float = 0.25
    align_deg: float = 6.0  # rotate first if |bearing| is larger
    max_step_m: float = 1.2  # longest single forward move
    min_step_m: float = 0.2  # Tello minimum move is 20 cm
    reacquire_scan_deg: int = 30
    max_steps: int = 20  # moves + turns + sidesteps + searches before giving up


@dataclass
class SafetyCfg:
    max_rc: int = 50  # clamp on every rc channel
    min_altitude_m: float = 0.5
    max_altitude_m: float = 2.3
    min_battery_pct: float = 20.0
    max_flight_s: float = 420.0
    frame_timeout_s: float = 1.0  # hover if no new frame for this long
    telemetry_timeout_s: float = 1.0


@dataclass
class DroneCfg:
    kind: str = "tello"  # tello | dry_run | sim
    move_speed_cm_s: int = 50  # speed for discrete moves ('speed' command)
    yaw_sign: int = 1  # +1 if telemetry yaw grows with `cw`; auto-checked at the first scan rotation
    video_fps: int = 60  # reader retrieve cap: must sit above the ~30 fps stream (equal drops ~40 % of frames)
    video_backend: str = "pyav"  # Tello H.264 decoder: pyav (djitellopy's) or opencv (the team's VideoStream)
    # camera-to-laptop video delay (s). After a discrete move/rotation, only frames arriving later than
    # "command done + video_lag_s" show the new view. Measure it with tools/tello_latency_test.py.
    # 2026-09-26 run: video 6 +/- 78 ms (yaw) and 21 +/- 40 ms (forward move) behind telemetry, worst 85 ms
    video_lag_s: float = 0.25
    command_timeout_s: float = 12.0


@dataclass
class MissionCfg:
    takeoff: bool = True
    announce: bool = True  # print spoken-style messages (audio hook)


@dataclass
class Config:
    camera: CameraCfg = field(default_factory=CameraCfg)
    perception: PerceptionCfg = field(default_factory=PerceptionCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    follow: FollowCfg = field(default_factory=FollowCfg)
    explore: ExploreCfg = field(default_factory=ExploreCfg)
    approach: ApproachCfg = field(default_factory=ApproachCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
    mission: MissionCfg = field(default_factory=MissionCfg)


# ---------------------------------------------------------------------- merging
def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_update(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _coerce(value: Any, hint: Any, path: str) -> Any:
    """Check/convert a scalar against the field's type hint; raise with the key path on mismatch."""
    if hint is float or hint == (float | None):
        if value is None and hint != float:
            return None
        if isinstance(value, bool):
            raise TypeError(f"config '{path}' must be a number, got {value!r}")
        try:
            return float(value)  # also accepts "1e-3" that YAML leaves as a string
        except (TypeError, ValueError):
            raise TypeError(f"config '{path}' must be a number, got {value!r}") from None
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"config '{path}' must be an integer, got {value!r}")
        try:
            f = float(value)
        except ValueError:
            raise TypeError(f"config '{path}' must be an integer, got {value!r}") from None
        if f != int(f):
            raise TypeError(f"config '{path}' must be an integer, got {value!r}")
        return int(f)
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"config '{path}' must be true/false, got {value!r}")
        return value
    if hint is str:
        if value is None:
            return ""  # e.g. `kind: null` disables a component
        if not isinstance(value, str):
            raise TypeError(f"config '{path}' must be a string, got {value!r}")
        return value
    return copy.deepcopy(value)


def _merge(obj: Any, over: dict, path: str) -> Any:
    if not dataclasses.is_dataclass(obj):
        raise TypeError(f"{path} is not a section")
    if not isinstance(over, dict):
        raise TypeError(f"config section '{path.rstrip('.') or '<root>'}' must be a mapping, got {over!r}")
    hints = get_type_hints(type(obj))
    names = {f.name for f in dataclasses.fields(obj)}
    # Naming a kind describes the whole component: fresh params (e.g. a trained YOLO replacing YOLO-World must
    # not inherit its prompts). Only `params` given -> merged into the current ones.
    if isinstance(obj, ComponentSpec) and "kind" in over and (over["kind"] != obj.kind or "params" in over):
        obj.params = {}
    for k, v in over.items():
        if k not in names:
            raise KeyError(f"unknown config key '{path}{k}' (valid: {sorted(names)})")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur):
            if v is None:  # a YAML section with only commented-out keys: keep the defaults
                continue
            if isinstance(cur, ComponentSpec) and isinstance(v, str):
                v = {"kind": v}  # shorthand: target_detector: null_detector
            _merge(cur, v, f"{path}{k}.")
        elif isinstance(cur, dict):
            if v is None:
                continue
            if not isinstance(v, dict):
                raise TypeError(f"config '{path}{k}' must be a mapping, got {v!r}")
            setattr(obj, k, _deep_update(cur, v))
        else:
            setattr(obj, k, _coerce(v, hints.get(k), f"{path}{k}"))
    return obj


def validate(cfg: Config) -> Config:
    """Cross-field rules that would otherwise fail silently in flight."""
    p, f, s, t = cfg.perception, cfg.follow, cfg.safety, cfg.tracking
    if f.altitude_m < p.person_height_m + 0.2:
        raise ValueError(f"follow.altitude_m ({f.altitude_m}) must be >= perception.person_height_m + 0.2 "
                         f"({p.person_height_m + 0.2:.2f}): too close to head height, the person range estimate breaks")
    if f.min_range_m >= f.distance_m:
        raise ValueError(f"follow.min_range_m ({f.min_range_m}) must be below follow.distance_m ({f.distance_m})")
    if f.altitude_m > s.max_altitude_m:
        raise ValueError(f"follow.altitude_m ({f.altitude_m}) is above safety.max_altitude_m ({s.max_altitude_m})")
    if cfg.explore.scan_altitude_m < s.min_altitude_m:
        raise ValueError("explore.scan_altitude_m is below safety.min_altitude_m")
    if t.max_age_s < t.lost_after_s:
        raise ValueError("tracking.max_age_s must be >= tracking.lost_after_s (else re-locking a returning target fails)")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None, target: str | None = None) -> Config:
    """Defaults, then the YAML file, then the target preset (TARGET_PRESETS), then overrides."""
    cfg = Config()
    if path:
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        _merge(cfg, data, "")
    if target:
        if target not in TARGET_PRESETS:
            raise ValueError(f"unknown target preset {target!r}; choose one of {sorted(TARGET_PRESETS)}")
        _merge(cfg, copy.deepcopy(TARGET_PRESETS[target]), "")
    if overrides:
        _merge(cfg, overrides, "")
    return validate(cfg)


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
