"""Push-to-talk microphone capture, AirPods-profile aware.

The AirPods stay in A2DP (stereo, no mic) until an app opens their capture
endpoint, which flips them to HFP (mono 16 kHz) for exactly as long as the
stream is open. So the recorder opens the input stream ON DEMAND and closes it
the moment the utterance ends -- never hold it open, never rely on mute.

Devices are re-resolved by NAME on every open: WASAPI device indices shift when
the Bluetooth profile flips or the AirPods reconnect.
"""

from __future__ import annotations

import queue
import time

import numpy as np


def find_input_device(substr: str) -> tuple[int | None, str]:
    """Index and name of the first input device whose name contains substr
    (case-insensitive), or (None, <default device name>) to use the default mic."""
    import sounddevice as sd

    want = substr.lower()
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and want in dev["name"].lower():
            return i, dev["name"]
    try:
        default = sd.query_devices(kind="input")["name"]
    except Exception:
        default = "system default"
    return None, default


class Recorder:
    def __init__(self, input_substr: str = "AirPods", rate: int = 16000,
                 max_s: float = 6.0, trailing_silence_s: float = 0.8,
                 silence_rms: float = 0.010):
        self.input_substr = input_substr
        self.rate = rate
        self.max_s = max_s
        self.trailing_silence_s = trailing_silence_s
        self.silence_rms = silence_rms

    def record(self, on_live=None) -> np.ndarray:
        """Open the mic, capture one utterance, close the mic, return float32 mono.

        on_live() is called once, from the moment audio is actually flowing --
        that is when the A2DP->HFP switch has completed and the user should be
        cued to speak (play the chirp there, not before opening the stream).

        Endpointing: stop after trailing_silence_s of quiet FOLLOWING speech,
        or at max_s. If the user never speaks, returns after ~2.5s of silence.
        """
        import sounddevice as sd

        idx, name = find_input_device(self.input_substr)
        if idx is None:
            print(f"[audio] no '{self.input_substr}' input found -> using {name}")
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        blocks: list[np.ndarray] = []
        started_speaking = False
        quiet_s = 0.0
        cued = False
        with sd.InputStream(samplerate=self.rate, channels=1, dtype="float32",
                            device=idx, blocksize=int(self.rate * 0.05),
                            callback=callback):
            t0 = time.time()
            while time.time() - t0 < self.max_s:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if not cued:
                    cued = True
                    if on_live is not None:
                        on_live()
                    # everything before the cue is profile-switch garbage
                    blocks.clear()
                    continue
                blocks.append(block)
                dur = block.size / self.rate
                if float(np.sqrt(np.mean(block**2))) >= self.silence_rms:
                    started_speaking = True
                    quiet_s = 0.0
                else:
                    quiet_s += dur
                    if started_speaking and quiet_s >= self.trailing_silence_s:
                        break
                    if not started_speaking and quiet_s >= 2.5:
                        break
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


    def listen(self, on_utterance, paused=None, tick=None) -> None:
        """Hands-free: hold ONE input stream open (the AirPods stay in HFP for the whole
        session) and segment utterances by energy. on_utterance(audio) per utterance;
        paused() True drops audio (the TTS is speaking: don't transcribe ourselves);
        tick() runs every block (~50 ms) for the caller's housekeeping. Ctrl+C to stop."""
        import sounddevice as sd

        idx, name = find_input_device(self.input_substr)
        if idx is None:
            print(f"[audio] no '{self.input_substr}' input found -> using {name}")
        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        blocks: list[np.ndarray] = []
        speaking = False
        quiet_s = 0.0
        noise = 0.002  # adaptive floor: EMA of the RMS while nobody is talking. The fixed
        # push-to-talk threshold (silence_rms 0.010) needs a raised voice on the AirPods'
        # HFP mic; hands-free triggers a factor above the ACTUAL room floor instead.
        with sd.InputStream(samplerate=self.rate, channels=1, dtype="float32",
                            device=idx, blocksize=int(self.rate * 0.05),
                            callback=callback):
            while True:
                if tick is not None:
                    tick()
                try:
                    pending = [q.get(timeout=0.5)]
                except queue.Empty:
                    continue
                while True:  # drain: if tick() ever runs long, catch up instead of lagging behind
                    try:
                        pending.append(q.get_nowait())
                    except queue.Empty:
                        break
                if paused is not None and paused():
                    blocks.clear()  # our own TTS (or its tail): never an utterance
                    speaking, quiet_s = False, 0.0
                    continue
                block = np.concatenate(pending) if len(pending) > 1 else pending[0]
                dur = block.size / self.rate
                rms = float(np.sqrt(np.mean(block**2)))
                start_thr = max(3.0 * noise, 0.0035)  # to BEGIN an utterance
                keep_thr = max(1.6 * noise, 0.0020)  # to CONTINUE one (hysteresis: no mid-word cuts)
                loud = rms >= (keep_thr if speaking else start_thr)
                if not speaking and rms < start_thr:
                    noise += 0.05 * (rms - noise)  # learn the floor only from non-speech
                if loud:
                    speaking = True
                    quiet_s = 0.0
                elif speaking:
                    quiet_s += dur
                blocks.append(block)
                if speaking and (quiet_s >= self.trailing_silence_s
                                 or sum(b.size for b in blocks) >= self.rate * self.max_s):
                    on_utterance(np.concatenate(blocks))
                    blocks.clear()
                    speaking, quiet_s = False, 0.0
                elif not speaking and sum(b.size for b in blocks) > self.rate * 1.0:
                    del blocks[:-4]  # keep only a short pre-roll while nobody is talking


def play(data: np.ndarray, rate: int = 16000) -> None:
    """Blocking playback (used by the selftest to echo the recording)."""
    import sounddevice as sd

    sd.play(data, rate, blocking=True)
