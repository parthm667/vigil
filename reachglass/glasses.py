"""Guide cues -> the glasses' haptic pads and the wearer's ears.

The mission's cue_fn gets -1 (turn left) / 0 (walk forward) / +1 (turn right) / 2 (arrived).
HapticCues forwards each cue to the Nano over GlassesLink.direction() -- the firmware crosses
sides on purpose (+1 "go right" presses the LEFT pad, a nudge from the far side pushing the
wearer the way they should go; see firmware/HARDWARE_INTERFACE.md section 4.1) -- 0 releases both,
and 2 (arrived, guiding over) releases both for good. Cue CHANGES are also spoken through the announce channel (the voice
app's TTS); the once-a-second repeats of the same cue are not, so the wearer is not nagged.

The firmware module (firmware/host/reachglass_glasses.py) is imported read-only from its own
folder. If the glasses are unreachable or glasses.enabled is false, cues still print and speak.

GlassesTargetView runs the target detector on the glasses' ESP32-CAM stream (the wearer's point of view)
for the last metres of GUIDE: the bottle's bearing (+ = to the wearer's right) and distance.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
import threading
from pathlib import Path
from typing import Callable

import cv2

log = logging.getLogger("reachglass.glasses")

SPOKEN = {-1: "Turn left.", 0: "Walk forward.", 1: "Turn right.",
          2: "You're there. It's right in front of you."}


def _firmware():
    """firmware/host/reachglass_glasses.py (imported from its file, once; firmware is read-only). It must be in
    sys.modules while it executes: its @dataclass looks its own module up there (without it the import raises,
    which HapticCues used to swallow as "glasses unavailable")."""
    if "reachglass_glasses" in sys.modules:
        return sys.modules["reachglass_glasses"]
    path = Path(__file__).resolve().parents[1] / "firmware" / "host" / "reachglass_glasses.py"
    spec = importlib.util.spec_from_file_location("reachglass_glasses", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reachglass_glasses"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules["reachglass_glasses"]
        raise
    return mod


def _glasses_link():
    return _firmware().GlassesLink


ROTATIONS = {"ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180}


class GlassesTargetView:
    """The target seen from the glasses camera. poll() is non-blocking: None when no new detector run, else
    (seen, bearing_deg, range_m): bearing + = right of the wearer's view; range None when the box is cut."""

    def __init__(self, gcfg, detector, target_cls: str, height_m: float | None, stride: int = 3):
        self.detector, self.cls, self.height_m, self.stride = detector, target_cls, height_m, max(1, stride)
        self.rotate = ROTATIONS.get(gcfg.camera_rotate)  # the camera is mounted turned: frames upright first
        self.f = gcfg.camera_f_px
        self.src = None
        self.closed = False
        self.seq, self.n = -1, 0
        # opening the MJPEG stream can block for seconds: never in the control loop
        threading.Thread(target=self._open, args=(gcfg.camera_url,), daemon=True).start()

    def _open(self, url: str) -> None:
        try:
            src = _firmware().GlassesVideoSource(url=url).start()
            if self.closed:
                src.stop()  # closed while it was connecting: the CAM serves one client, let it go
            else:
                self.src = src
        except Exception as e:  # noqa: BLE001 (no glasses camera: guiding goes on with the drone's camera)
            log.warning("glasses camera unavailable (%s: %s)", type(e).__name__, e)

    def poll(self):
        f = self.src.read() if self.src is not None else None
        if f is None or f.seq == self.seq:
            return None
        self.seq, self.n = f.seq, self.n + 1
        if self.n % self.stride:
            return None
        img = cv2.rotate(f.image, self.rotate) if self.rotate is not None else f.image
        dets = [d for d in self.detector.detect(img) if d.cls == self.cls]
        if not dets:
            return False, None, None
        d = max(dets, key=lambda x: x.conf)
        h, w = img.shape[:2]
        bearing = math.degrees(math.atan((d.cx - w / 2) / self.f))
        cut = d.bbox[1] <= 2 or d.bbox[3] >= h - 2
        rng = self.f * self.height_m / d.h if (self.height_m and not cut and d.h > 0) else None
        return True, bearing, rng

    def close(self) -> None:
        self.closed = True
        src, self.src = self.src, None
        if src is not None:
            src.stop()


class HapticCues:
    """A Mission cue_fn: pads + speech, falling back to print-only when the hardware is absent."""

    def __init__(self, cfg, say: Callable[[str], None] | None = None):
        self.say = say
        self.link = None
        self.last: int | None = None
        self._warned = False
        if cfg.enabled:
            try:
                self.link = _glasses_link()(host=cfg.host or None, max_press=cfg.max_press).start()
                log.info("glasses haptics up: %s", self.link.host)
            except Exception as e:  # noqa: BLE001 (no glasses must never stop a flight)
                log.warning("glasses unavailable (%s: %s): cues will print and speak only",
                            type(e).__name__, e)

    def __call__(self, cue: int, detail: str = "") -> None:
        print(f"guide cue: {cue:2d}   {detail}", flush=True)
        if self.link is not None:
            try:
                if cue == 2:
                    self.link.release()  # arrived: both pads off the face, and they stay off (guiding is over)
                else:
                    self.link.direction(cue)  # -1 presses the RIGHT pad, +1 the LEFT, 0 releases both
            except Exception as e:  # noqa: BLE001
                if not self._warned:
                    self._warned = True
                    log.warning("glasses cue failed (%s: %s): continuing without haptics",
                                type(e).__name__, e)
        if self.say is not None and cue != self.last and cue in SPOKEN:
            self.say(SPOKEN[cue])
        self.last = cue

    def close(self) -> None:
        if self.link is not None:
            try:
                self.link.stop()  # releases the pads explicitly, faster than the 600 ms failsafe
            except Exception:  # noqa: BLE001, S110
                pass
            self.link = None
