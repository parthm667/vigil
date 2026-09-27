"""Guide cues -> the glasses' haptic pads and the wearer's ears.

The mission's cue_fn gets -1 (turn left) / 0 (walk forward) / +1 (turn right) / 2 (arrived).
HapticCues forwards each cue to the Nano over GlassesLink.direction() -- the firmware crosses
sides on purpose (+1 "go right" presses the LEFT pad, a nudge from the far side pushing the
wearer the way they should go; see firmware/HARDWARE_INTERFACE.md section 4.1) -- and 2 becomes
a double buzz on both pads. Cue CHANGES are also spoken through the announce channel (the voice
app's TTS); the once-a-second repeats of the same cue are not, so the wearer is not nagged.

The firmware module (firmware/host/reachglass_glasses.py) is imported read-only from its own
folder. If the glasses are unreachable or glasses.enabled is false, cues still print and speak.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger("reachglass.glasses")

SPOKEN = {-1: "Turn left.", 0: "Walk forward.", 1: "Turn right.",
          2: "You're there. It's right in front of you."}


def _glasses_link():
    """The GlassesLink class from firmware/host (imported from its file; firmware is read-only)."""
    path = Path(__file__).resolve().parents[1] / "firmware" / "host" / "reachglass_glasses.py"
    spec = importlib.util.spec_from_file_location("reachglass_glasses", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GlassesLink


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
                    self.link.release()
                    self.link.pulse("B", count=2)  # arrived: a buzz on both pads, not a direction
                else:
                    self.link.direction(cue)
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
