"""Frozen runtime message contract (docs/DRONE_RL_PLAN.md Section 6, extended for the full runtime).

Every drone-side process (Tello I/O, detector, controller, mission logic, guidance, operator console,
viz, recorder) talks through one local ZeroMQ broker with JSON messages. Change this file only with the
lead's agreement; everything else codes against it.

Transport
- A broker (flyfollow.runtime.bus) binds XSUB on PUB_ADDR and XPUB on SUB_ADDR. Publishers connect a
  PUB socket to PUB_ADDR, subscribers connect a SUB socket to SUB_ADDR. Loopback only.
- Each message is one JSON object with "topic" and "t" (laptop time.time(), seconds), sent as a
  two-frame multipart [topic bytes, json bytes] so subscribers can filter by topic prefix.
- Video frames do NOT go through JSON: the Tello I/O writes decoded RGB frames into a shared-memory ring
  (FRAME_RING_NAME, FRAME_RING_SLOTS slots of IMG_H x IMG_W x 3 uint8) and publishes a small "frame"
  message naming the slot. Consumers copy the slot out and must check frame_id in the slot header
  still matches (the producer may have overwritten it).

Units and frames (same as flyfollow.interfaces): pixels of the 960x720 Tello frame (never letterboxed),
bearing + = right, rc sticks are final integers -100..100 (yaw + = clockwise, fb + = forward,
ud + = up, lr + = right), drone_level frame x forward, y left, z up. Meters and degrees.

Who may command the drone
- Only flyfollow.runtime.tello_io talks to the Tello. It owns the single command queue.
- rc messages are accepted only from the owner of the current mode (RC_OWNER), plus the operator.
- The operator's "tello_cmd" land/emergency and the "kill" topic are always accepted, in every mode.
- If no accepted rc arrives for RC_TIMEOUT_S, Tello I/O sends rc 0 0 0 0 (hover) itself, and it keeps
  streaming rc at RC_HZ so the Tello never hits its 15 s no-command auto-land.
- Dry run (log only, never send) unless started with --send.
- Replayed messages carry "replayed": true; tello_io and sim_world must never act on a replayed command.
- Late joiners miss one-shot messages (ZeroMQ PUB/SUB): state-like topics (mode, target, settings, lock) are
  re-published about once a second by their owners.
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------------------------- transport
PUB_ADDR = "tcp://127.0.0.1:5550"  # publishers connect here (broker XSUB)
SUB_ADDR = "tcp://127.0.0.1:5551"  # subscribers connect here (broker XPUB)
FRAME_RING_NAME = "flyfollow_frames"
FRAME_RING_SLOTS = 6
RC_HZ = 20.0
RC_TIMEOUT_S = 0.5

# ---------------------------------------------------------------------------------------------- modes
MODES = ("FOLLOW", "FIND", "APPROACH", "FACE_PERSON", "OVERWATCH", "GUIDE", "RETURN", "HOLD", "LAND", "IDLE")
# IDLE = on the ground or before takeoff. HOLD = hover in place (operator pause or "stay").
RC_OWNER = {
    "FOLLOW": "controller",
    "APPROACH": "controller",
    "GUIDE": "controller",  # PID yaw hold on the user, forward disabled (plan 5.5)
    "FIND": "mission",
    "FACE_PERSON": "mission",
    "OVERWATCH": "mission",
    "RETURN": "mission",
    "HOLD": "mission",
    "LAND": None,
    "IDLE": None,
}
RC_SOURCES = ("controller", "mission", "operator")

# ---------------------------------------------------------------------------------------------- topics
# Each entry: topic -> (publisher, required fields). Optional fields are documented inline.
TOPICS: dict[str, tuple[str, tuple[str, ...]]] = {
    # Tello I/O
    "frame": ("tello_io", ("frame_id", "t_decoded", "slot", "w", "h")),
    "tello_state": ("tello_io", ("yaw_deg", "pitch_deg", "roll_deg", "vgx_dms", "vgy_dms", "vgz_dms", "h_cm", "tof_cm", "bat_pct", "temph_c", "flying", "video_ok", "video_age_s", "sending")),
    #   vg*_dms are decimeters per second (confirmed from the 2026-09-26 lag log: FlyDrones' cm/s assumption is wrong).
    #   Optional: "v_fwd_mps", "v_right_mps", "v_up_mps", "state_ok", "mode", "safety".
    "tello_ack": ("tello_io", ("id", "cmd", "ok", "detail")),  # reply to a tello_cmd; optional "elapsed_s", "dry_run"
    "tello_event": ("tello_io", ("kind", "detail")),  # kill, safety land (battery, heat, video/state/bus lost), collision (sim)
    "sim_truth": ("sim_world", ()),  # simulator only, 10 Hz: drone, user and object poses for tests and the viz top view
    "rc_sent": ("tello_io", ("lr", "fb", "ud", "yaw", "src", "dry_run")),  # what actually went out, 20 Hz
    # Perception (the perception owner's YOLO, or flyfollow.runtime.detector as the fallback)
    "det": ("detector", ("frame_id", "t_decoded", "src", "img_w", "img_h", "dets")),
    #   dets: list of {"cls": str, "conf": float, "bbox": [x1, y1, x2, y2], "track_id": int | None}
    #   classes include "person" and "person_head" (head box shares the person's track_id), COCO objects,
    #   and open-vocabulary prompts. Optional: "imgsz", "latency_s" (detector time).
    # Operator console, voice, settings
    "mode_cmd": ("operator|voice", ("mode", "source")),  # optional "target": {"cls", "prompt", "height_m"}
    "lock": ("operator|detector", ("track_id",)),  # which person track is the user; track_id null = unlock
    "settings": ("operator", ()),  # any subset of SETTINGS_DEFAULTS keys; receivers merge
    "kill": ("operator", ("action",)),  # action "land" or "emergency": always obeyed; optional "source"
    "health": ("launcher|recorder", ("key", "ok", "level", "text")),  # liveness and process status for the console
    # Controller (flyfollow.runtime.controller_runner)
    "rc": ("controller|mission|operator", ("lr", "fb", "ud", "yaw", "src", "mode")),
    #   optional: "gov": {"safety": bool, "clamped": bool, "reasons": [str]}, "brain_tick_ms": float
    "ctrl_status": ("controller", ("mode", "controller", "target_valid", "range_m", "bearing_deg", "in_band")),
    # Mission logic (flyfollow.runtime.mission)
    "tello_cmd": ("mission|operator", ("id", "cmd")),  # cmd: takeoff | land | emergency | move | stop; "args": {...}
    #   move args: {"direction": "forward"|"back"|"left"|"right"|"up"|"down"|"cw"|"ccw", "value": cm or deg}
    "mode": ("mission", ("from", "to", "reason")),
    "target": ("mission", ("mode", "kind")),
    #   kind "person" | "object" | "none"; optional: "cls", "track_id", "bearing_deg", "range_m",
    #   "size_m" (assumed target height), "z_ref_m", "z_min_m", "cy_ref_frac"
    "say": ("mission|guidance", ("text", "priority")),  # 0 = info, 1 = normal, 2 = important, 3 = safety
    "scene": ("mission", ("frame", "pose", "cam", "person", "object", "visible_floor_min_range_m")),  # plan 6
    "find_status": ("mission", ("state", "hops", "elapsed_s", "searched")),
    "det_cfg": ("mission|operator", ()),  # detector request: optional "imgsz" (640 | 960), "tiles" (1 | 2), "prompts" [str]
    # Guidance (flyfollow.runtime.guidance, minimal rules until a guidance owner exists)
    "cue": ("guidance", ("kind", "strength")),  # kind: left | right | aligned | stop | arrived | none
}

SETTINGS_DEFAULTS: dict[str, Any] = {
    "follow_distance_m": 2.0,
    "side_offset_deg": 8.0,
    "max_fwd_stick": 60,
    "max_back_stick": 40,
    "min_person_dist_m": 1.2,
    "search_alt_m": 1.0,
    "user_height_m": 1.72,
    "head_size_m": 0.23,
    "controller": "fly",  # "fly" (FLY-YAW: the fly steers, PID sets forward) or "pid" (PID-HAND fallback)
    "params_path": None,  # trained best.json; None = the arm's hand calibration
    "video_latency_s": 0.25,  # R0 stopwatch test; Parth's lag test suggests 0.2 to 0.3 s
    "fx": 921.0,  # R0 checkerboard calibration (flyfollow.tools.calibrate_camera writes configs/camera.json)
    "fy": 919.0,
    "cx": 480.0,
    "cy": 360.0,
    "find_budget_s": 120.0,
    "find_max_hops": 3,
}


def msg(topic: str, **fields: Any) -> dict:
    """Build a message dict with topic and t (now) filled in; fields override t if given."""
    out = {"topic": topic, "t": time.time()}
    out.update(fields)
    return out


def validate(m: dict) -> list[str]:
    """Missing required fields for a known topic (empty list = ok). Unknown topics are allowed."""
    if "topic" not in m or "t" not in m:
        return ["topic/t"]
    spec = TOPICS.get(m["topic"])
    if spec is None:
        return []
    return [f for f in spec[1] if f not in m]
