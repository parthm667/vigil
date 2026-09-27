"""Text-to-speech with a worker thread and a queue.

Backend "edge" = edge-tts neural voices (needs internet), decoded with the
already-installed av and played through sounddevice as a NORMAL media stream --
never a Communications-category stream, which would force the AirPods into HFP.
Backend "sapi" = pyttsx3/Windows SAPI, fully offline. The backend is probed at
startup and every utterance falls back to SAPI on failure, so a mid-demo wifi
drop degrades to a robot voice instead of silence.
"""

from __future__ import annotations

import asyncio
import io
import queue
import threading
import time


def _synth_edge(text: str, voice: str) -> bytes:
    """Synthesize MP3 bytes with edge-tts (blocking wrapper)."""
    import edge_tts

    async def run() -> bytes:
        buf = io.BytesIO()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    return asyncio.run(asyncio.wait_for(run(), timeout=10.0))


def _decode_mp3(data: bytes):
    """MP3 bytes -> (float32 mono ndarray, rate) via av."""
    import av
    import numpy as np

    out = []
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        rate = stream.rate or 24000
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for rf in resampler.resample(frame):
                out.append(rf.to_ndarray().reshape(-1))
    if not out:
        raise ValueError("no audio frames decoded")
    return np.concatenate(out).astype(np.float32), rate


def _speak_sapi(text: str) -> None:
    """Offline fallback. A fresh engine per utterance dodges pyttsx3's stale-loop bugs."""
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", 185)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


class Speaker:
    """speak() enqueues; a single worker drains the queue. pause() holds speech
    (used while the mic is open -- never TTS into a live capture stream)."""

    def __init__(self, voice: str = "en-US-AriaNeural", backend: str = "auto"):
        self.voice = voice
        self._q: queue.Queue[str | None] = queue.Queue()
        self._enabled = threading.Event()
        self._enabled.set()
        self._speaking = threading.Event()
        self.backend = backend if backend != "auto" else self._probe()
        print(f"[tts] backend: {self.backend}")
        self._worker = threading.Thread(target=self._run, daemon=True, name="tts")
        self._worker.start()

    def _probe(self) -> str:
        try:
            _synth_edge("ready", self.voice)
            return "edge"
        except Exception as e:
            print(f"[tts] edge-tts unavailable ({type(e).__name__}), using offline SAPI")
            return "sapi"

    # ------------------------------------------------------------------ API
    def speak(self, text: str) -> None:
        text = text.strip()
        if text:
            self._q.put(text)

    def flush(self) -> None:
        """Drop everything queued but not yet spoken."""
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def pause(self) -> None:
        self._enabled.clear()

    def resume(self) -> None:
        self._enabled.set()

    @property
    def busy(self) -> bool:
        return self._speaking.is_set() or not self._q.empty()

    def wait(self, timeout: float = 30.0) -> None:
        t0 = time.time()
        while self.busy and time.time() - t0 < timeout:
            time.sleep(0.05)

    def close(self) -> None:
        self._q.put(None)
        self._worker.join(timeout=2.0)

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            self._enabled.wait()
            self._speaking.set()
            try:
                self._say(text)
            except Exception as e:
                print(f"[tts] failed to speak ({type(e).__name__}: {e}): {text!r}")
            finally:
                self._speaking.clear()

    def _say(self, text: str) -> None:
        if self.backend == "edge":
            try:
                import sounddevice as sd

                data, rate = _decode_mp3(_synth_edge(text, self.voice))
                sd.play(data, rate, blocking=True)
                return
            except Exception as e:
                print(f"[tts] edge failed ({type(e).__name__}), falling back to SAPI")
        _speak_sapi(text)
