"""VoiceApp: ties stem presses, push-to-talk, STT, TTS and the UDP ports together.

Gestures: single = push-to-talk, double = repeat / refresh guidance,
triple = send "stop". (Never mapped to "land": too easy to press by accident.)
"""

from __future__ import annotations

import time

from . import earcons
from .audio import Recorder
from .config import VoiceCfg
from .stem import DOUBLE, SINGLE, TRIPLE, Gestures
from .stt import Transcriber
from .tts import Speaker


class VoiceApp:
    def __init__(self, cfg: VoiceCfg, tts_backend: str = "auto"):
        from .__main__ import open_announce_rx, poll_announcements, send_query

        self._poll_announcements = poll_announcements
        self._send = lambda text: send_query(cfg, text)
        self.cfg = cfg
        self.speaker = Speaker(voice=cfg.tts_voice, backend=tts_backend)
        self.stt = Transcriber(cfg.stt_model)
        self.recorder = Recorder(cfg.input_substr, cfg.sample_rate, cfg.max_utterance_s,
                                 cfg.trailing_silence_s, cfg.silence_rms)
        self.gestures = Gestures(cfg.debounce_ms)
        self.rx = open_announce_rx(cfg)
        self.last_announcement: str | None = None
        self.last_find: str | None = None

    def run(self) -> int:
        print(f"[voice] ready: press the stem to talk "
              f"(-> {self.cfg.send_host}:{self.cfg.send_port}, <- :{self.cfg.announce_port})")
        self.speaker.speak("Voice control ready.")
        try:
            while True:
                for msg in self._poll_announcements(self.rx):
                    print(f"[voice] announce: {msg}")
                    self.last_announcement = msg
                    self.speaker.speak(msg)
                g = self.gestures.poll()
                if g == SINGLE:
                    self.push_to_talk()
                elif g == DOUBLE:
                    self.repeat()
                elif g == TRIPLE:
                    print("[voice] triple press -> stop")
                    self._send("stop")
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\n[voice] bye")
        finally:
            self.rx.close()
            self.speaker.close()
        return 0

    # ------------------------------------------------------------------ gestures
    def push_to_talk(self) -> None:
        """Open mic (A2DP->HFP switch starts), chirp once audio is live, record,
        blip, close mic, pad for the switch back, transcribe, send."""
        print("[voice] listening...")
        self.speaker.pause()
        try:
            audio = self.recorder.record(on_live=lambda: earcons.play(earcons.listen_chirp()))
            earcons.play(earcons.got_it_blip())
            time.sleep(self.cfg.post_mic_pad_s)  # let the AirPods settle back to A2DP
            text = self.stt.transcribe(audio)
        finally:
            self.speaker.resume()
        if not text:
            print("[voice] didn't catch that")
            earcons.play(earcons.error_buzz())
            return
        print(f"[voice] heard: {text!r}")
        self._send(text)
        if any(w in text for w in ("find", "where")):
            self.last_find = text

    # ------------------------------------------------------------------ hands-free
    KEYWORDS = ("find", "where", "stop", "cancel", "land", "takeoff", "take off",
                "follow", "come back", "what's around", "whats around")

    def run_hands_free(self) -> int:
        """Mic always on: one open stream, utterances segmented by silence, no stem needed.
        The mic is gated off while the TTS speaks (plus a short hangover) so the app never
        transcribes itself, and only text containing a command keyword is sent, so chatter
        near the wearer does not turn into drone commands. Stem presses still work.
        Note: the open mic keeps the AirPods in HFP for the whole session (mono audio)."""
        print(f"[voice] hands-free: mic is live, just talk "
              f"(-> {self.cfg.send_host}:{self.cfg.send_port}, <- :{self.cfg.announce_port})")
        self.speaker.speak("Voice control ready. Just talk.")
        self._quiet_until = 0.0

        def paused() -> bool:
            if self.speaker.busy():
                self._quiet_until = time.time() + 0.6  # hangover: don't catch our own tail
                return True
            return time.time() < self._quiet_until

        def tick() -> None:
            for msg in self._poll_announcements(self.rx):
                print(f"[voice] announce: {msg}")
                self.last_announcement = msg
                self.speaker.speak(msg)
            g = self.gestures.poll()
            if g == DOUBLE:
                self.repeat()
            elif g == TRIPLE:
                print("[voice] triple press -> stop")
                self._send("stop")

        def on_utterance(audio) -> None:
            text = self.stt.transcribe(audio)
            if not text:
                return
            low = text.lower()
            if not any(k in low for k in self.KEYWORDS):
                print(f"[voice] heard {text!r}: no command word, ignoring")
                return
            print(f"[voice] heard: {text!r}")
            earcons.play(earcons.got_it_blip())
            self._send(text)
            if any(w in low for w in ("find", "where")):
                self.last_find = text

        try:
            self.recorder.listen(on_utterance, paused=paused, tick=tick)
        except KeyboardInterrupt:
            print("\n[voice] bye")
        finally:
            self.rx.close()
            self.speaker.close()
        return 0

    def repeat(self) -> None:
        """Re-speak the last announcement now, and nudge the mission to refresh
        guidance for the current target (in ARRIVED it recomputes from the
        person's latest position instead of restarting the search)."""
        self.speaker.flush()
        if self.last_find:
            self._send(self.last_find)
        elif self.last_announcement:
            self.speaker.speak(self.last_announcement)
        else:
            earcons.play(earcons.error_buzz())
