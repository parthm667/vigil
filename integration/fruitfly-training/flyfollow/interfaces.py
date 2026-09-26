"""Frozen interfaces shared by the sim, the controllers and the trainer (plan item 0).

Every arm (FLY-HAND, FLY-CMA, FLY-SHUF, NOBRAIN, PID-HAND, PID-CMA) plugs into the
same loop through these types, so the arms differ only in how they turn a filtered
target box into (yaw, forward) sticks. See docs/DRONE_RL_PLAN.md Sections 4.1 to 4.7.

Conventions (use these everywhere, never redefine them):
- Image: 960x720 pixels of the Tello frame, never letterboxed. x right, y down.
- Bearing theta (rad): positive when the target is RIGHT of the image center.
- Tello rc sticks are final sent integers in -100..100 (never the FlyDrones [-1, 1] scale).
  yaw > 0 turns clockwise seen from above (to the right); fb > 0 flies forward;
  ud > 0 climbs; lr > 0 moves right.
- Drone-level frame: origin at the drone, x forward, y left, z up (pitch and roll removed).
- Control rate 20 Hz (DT = 0.05 s); one brain tick simulates 50 ms of LIF time.
- Normalized size s = h_px / h_ref with h_ref = fy * target_size_m / z_ref_m, so s = 1 at the
  standoff distance for any target. s > 1 means too close.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Where brains, calibration files and runs live: ./data locally, /data (the Modal Volume) on Modal."""
    return Path(os.environ.get("FLYFOLLOW_DATA", REPO_ROOT / "data"))


def brains_dir() -> Path:
    return data_root() / "brains"


def configs_dir() -> Path:
    return Path(os.environ.get("FLYFOLLOW_CONFIGS", REPO_ROOT / "configs"))


IMG_W = 960
IMG_H = 720
DT = 0.05
BRAIN_TICK_MS = 50.0

KINDS = ("follow", "approach")
PROFILES = ("train", "demo", "stress")
ARMS = ("FLY-HAND", "FLY-CMA", "FLY-SHUF", "NOBRAIN", "PID-HAND", "PID-CMA")
# Yaw-only arms (plan 4.3 outcome "yaw-only fly"; chosen 2026-09-26 after Parth's local runs showed the full
# fly fails on range): the controller sets ONLY yaw, and forward comes from PID-HAND's fixed range loop, so
# the comparison between these arms isolates steering. FLY-YAW-HAND is the untrained (hand-calibrated) fly.
YAW_ONLY_ARMS = ("FLY-YAW-HAND", "FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW")
ARMS = ARMS + YAW_ONLY_ARMS
TRAINED_ARMS = ("FLY-CMA", "FLY-SHUF", "NOBRAIN", "PID-CMA", "FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW")

# Seed sets (plan 4.4). Training seeds are drawn fresh per generation from TRAIN_SEED_BASE upward.
TRAIN_SEED_BASE = 10_000
SELECTION_SEEDS = tuple(range(1_000, 1_064))
TEST_SEEDS = tuple(range(2_000, 2_200))

# Neuron group names stored in every pursuit brain .npz (built by flyfollow.brain.build).
# Inputs: LC10a (steering target), LC9 and LC11 (extra visual inputs with their own gains),
# AROUSAL (P1 / pC1 candidates; may be empty if the audit cannot identify P1).
INPUT_GROUPS = ("LC10a_L", "LC10a_R", "LC9_L", "LC9_R", "LC11_L", "LC11_R", "AROUSAL_L", "AROUSAL_R")
# Steering DN readout set (FLYGUIDE_SPEC 6.6). Each type is one cell per side in MaleCNS v1.0.
DN_TYPES = ("DNa02", "DNa01", "DNb05", "DNg13", "DNb06")
OUTPUT_GROUPS = tuple(f"{t}_{s}" for t in DN_TYPES for s in "LR")
N_AZIMUTH_BINS = 8  # LC10a bins per side (plan 4.1)


@dataclass
class Settings:
    """Per-episode governor and controller settings (randomized in training, set by the operator at runtime).

    Everything here is known to the controller. Hidden simulator truth (true target size,
    true focal length, true latency, stick gains) lives only inside the env.
    """

    kind: str  # "follow" or "approach"
    z_ref_m: float  # standoff distance the controller should hold
    target_size_m: float  # size the controller ASSUMES (0.23 m head, class prior for objects)
    side_offset_deg: float = 0.0  # planned bearing offset; the controller centers theta - offset
    z_min_m: float = 1.0  # governor minimum distance (1.0 to 1.3 person, 0.5 object)
    max_fwd_stick: float = 35.0  # governor clamp on forward stick (20 to 50)
    max_back_stick: float = 20.0  # governor clamp on reverse stick
    max_yaw_stick: float = 60.0
    fx: float = 921.0  # intrinsics the controller assumes (nominal calibration)
    fy: float = 919.0
    cx0: float = IMG_W / 2
    cy0: float = IMG_H / 2
    cy_ref_frac: float = 0.55  # governor ud loop keeps the target at this fraction of image height
    video_latency_s: float = 0.3  # latency estimate the shared Kalman filter predicts forward by

    @property
    def h_ref_px(self) -> float:
        return self.fy * self.target_size_m / self.z_ref_m


@dataclass
class BoxState:
    """Latency-compensated target box from the shared BoxFilter (flyfollow.pilot.box_filter).

    valid is True while the filter has a measured box or is holding one through a dropout
    shorter than the hold time (0.5 s). When valid is False the governor's lost-target logic
    owns yaw and forward and the controller output is ignored for those axes.
    """

    valid: bool
    cx: float = IMG_W / 2
    cy: float = IMG_H / 2
    h: float = 0.0
    vcx: float = 0.0  # px/s
    vcy: float = 0.0
    vh: float = 0.0
    since_det_s: float = 0.0  # time since the last real detection was fused
    lost_s: float = 0.0  # time since valid went False (0 while valid)


@dataclass
class TargetFeatures:
    """The shared encoder feature vector. The FLY TargetEncoder and NOBRAIN both start from this."""

    valid: bool
    theta: float  # bearing error (rad), + = target right of the planned bearing
    s: float  # normalized apparent size, 1 at the standoff
    dtheta: float  # rad/s
    ds: float  # 1/s


def target_features(box: BoxState, st: Settings) -> TargetFeatures:
    """Box -> (bearing error, normalized size, their rates). Pure geometry, no trained parameters."""
    if not box.valid or box.h <= 0:
        return TargetFeatures(False, 0.0, 0.0, 0.0, 0.0)
    theta = math.atan((box.cx - st.cx0) / st.fx) - math.radians(st.side_offset_deg)
    dtheta = (box.vcx / st.fx) / (1.0 + ((box.cx - st.cx0) / st.fx) ** 2)
    s = box.h / st.h_ref_px
    ds = box.vh / st.h_ref_px
    return TargetFeatures(True, theta, s, dtheta, ds)


class Controller(Protocol):
    """One pursuit controller. Outputs are raw (pre-governor) yaw and forward sticks in -100..100.

    Lifecycle per episode: reset(settings, seed) -> warmup(seconds) -> act(...) every DT.
    The governor (flyfollow.pilot.governor.PersonGovernor) owns ud, lr, lost-target handling,
    min distance, clamps and slew limits for every arm.
    """

    name: str

    def reset(self, settings: Settings, seed: int) -> None: ...

    def warmup(self, seconds: float) -> None:
        """Brain arms run the LIF with no target for `seconds` (excluded from reward). PID: no-op."""
        ...

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        """Return (yaw_stick, fb_stick) before the governor."""
        ...


@dataclass
class EpisodeResult:
    """What one episode returns to the trainer and to eval. Plain data so it pickles across processes."""

    seed: int
    kind: str
    profile: str
    ret: float  # total return (reward sum incl. events); higher is better, usually negative
    terms: dict[str, float] = field(default_factory=dict)  # summed contribution of each reward term and event
    metrics: dict[str, float] = field(default_factory=dict)  # plan 4.7 metrics for this episode
    n_ticks: int = 0
    wall_s: float = 0.0
    brain_s: float = 0.0  # wall time spent inside the brain (0 for no-brain arms)
    trace: dict[str, list] | None = None  # per-tick arrays when run with record=True

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def azimuth_bins(group_idx: np.ndarray, n_bins: int = N_AZIMUTH_BINS) -> list[np.ndarray]:
    """Rank fallback for LC10a retinotopy: split one side's LC10a neurons into n_bins equal bins.

    Bin 0 is the most frontal (near the midline), bin n_bins-1 the most lateral. The brain
    builder may store a better assignment in the .npz meta["azimuth_bins"]; use that when present.
    """
    return [np.asarray(b, dtype=np.int64) for b in np.array_split(np.sort(np.asarray(group_idx, dtype=np.int64)), n_bins)]
