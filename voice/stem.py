"""AirPods stem-press detection on Windows.

The AirPods decode presses on-device and send AVRCP media commands over
Bluetooth: single = Play/Pause, double = Next, triple = Previous. (Long squeeze
toggles ANC on-device and never reaches the PC; Siri gestures need an Apple
host.) Windows routes those commands to the ACTIVE media session -- so we
create a dummy MediaPlayer, disable its command manager, and pin its playback
status to PLAYING: our process becomes the active session, receives every
press via SystemMediaTransportControls.button_pressed, and CONSUMES it (Spotify
et al. never see it).

Do NOT listen for VK_MEDIA_* key events instead: the shell only synthesizes
those when no media session is active, and ours always is.

Known quirk: a double-press sometimes arrives as PAUSE followed by NEXT a few
ms apart, so gestures are classified over a debounce window, strongest wins
(PREVIOUS > NEXT > PLAY/PAUSE).
"""

from __future__ import annotations

import queue
import sys
import threading
import time

SINGLE, DOUBLE, TRIPLE = "single", "double", "triple"


class StemListener:
    def __init__(self, debounce_ms: int = 250):
        self.debounce_s = debounce_ms / 1000.0
        self._raw: queue.Queue[str] = queue.Queue()
        self._player = None  # keep refs alive or the session vanishes
        self._smtc = None
        self.ok = False
        try:
            self._start()
            self.ok = True
            print("[stem] SMTC session active: stem presses will be captured")
        except Exception as e:
            print(f"[stem] SMTC unavailable ({type(e).__name__}: {e}) -> keyboard fallback only")

    def _start(self) -> None:
        from winsdk.windows.media import (MediaPlaybackStatus,
                                          SystemMediaTransportControlsButton as Btn)
        from winsdk.windows.media.playback import MediaPlayer

        self._player = MediaPlayer()
        self._player.command_manager.is_enabled = False
        smtc = self._player.system_media_transport_controls
        smtc.is_play_enabled = True
        smtc.is_pause_enabled = True
        smtc.is_stop_enabled = True
        smtc.is_next_enabled = True
        smtc.is_previous_enabled = True
        smtc.playback_status = MediaPlaybackStatus.PLAYING
        self._smtc = smtc

        kind = {Btn.PLAY: "play", Btn.PAUSE: "pause", Btn.STOP: "stop",
                Btn.NEXT: "next", Btn.PREVIOUS: "previous"}

        def on_button(sender, args):
            self._raw.put(kind.get(args.button, "other"))

        smtc.add_button_pressed(on_button)

        def keep_alive():
            # re-assert PLAYING so another app starting playback can't steal
            # the AVRCP target away from us mid-demo
            while True:
                time.sleep(3.0)
                try:
                    smtc.playback_status = MediaPlaybackStatus.PLAYING
                except Exception:
                    return

        threading.Thread(target=keep_alive, daemon=True, name="smtc-keepalive").start()

    def get(self, timeout: float | None = None) -> str | None:
        """Next gesture: 'single' | 'double' | 'triple', or None on timeout.
        Collapses everything that arrives within the debounce window."""
        try:
            first = self._raw.get(timeout=timeout)
        except queue.Empty:
            return None
        events = {first}
        t_end = time.time() + self.debounce_s
        while True:
            left = t_end - time.time()
            if left <= 0:
                break
            try:
                events.add(self._raw.get(timeout=left))
            except queue.Empty:
                break
        if "previous" in events:
            return TRIPLE
        if "next" in events:
            return DOUBLE
        return SINGLE


class StdinFallback:
    """Always-on parallel trigger: Enter = single, r+Enter = double, s+Enter = triple.
    Saves the demo if SMTC or the AirPods misbehave."""

    KEYS = {"": SINGLE, "r": DOUBLE, "s": TRIPLE}

    def __init__(self):
        self._q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True, name="stdin-fallback").start()

    def _read(self) -> None:
        for line in sys.stdin:
            g = self.KEYS.get(line.strip().lower())
            if g:
                self._q.put(g)

    def get(self, timeout: float | None = None) -> str | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class Gestures:
    """StemListener + StdinFallback behind one poll() interface."""

    def __init__(self, debounce_ms: int = 250, stdin_fallback: bool = True):
        self.stem = StemListener(debounce_ms)
        self.stdin = StdinFallback() if stdin_fallback else None
        if self.stdin is not None:
            print("[stem] stdin fallback: Enter = talk, r = repeat, s = stop")

    def poll(self) -> str | None:
        """Non-blocking-ish: returns a pending gesture or None."""
        g = self.stem.get(timeout=0.02) if self.stem.ok else None
        if g:
            return g
        if self.stdin is not None:
            return self.stdin.get(timeout=0.02)
        return None
