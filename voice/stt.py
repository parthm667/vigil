"""Speech-to-text with faster-whisper (CTranslate2, int8 on CPU -- no CUDA needed).

The model is loaded once at startup so the first stem press doesn't pay the
load cost. initial_prompt biases decoding toward the command vocabulary, which
noticeably helps with HFP phone-call audio quality.
"""

from __future__ import annotations

import numpy as np

COMMAND_HINT = ("find my water bottle, find my blue water bottle, follow me, "
                "what's around me, where is, stop, cancel, land, takeoff")


class Transcriber:
    def __init__(self, model: str = "base.en"):
        from faster_whisper import WhisperModel

        print(f"[stt] loading whisper {model} (int8, cpu)...")
        self.model = WhisperModel(model, device="cpu", compute_type="int8")
        # warm up so the first real utterance is fast
        self.model.transcribe(np.zeros(1600, dtype=np.float32), language="en")
        print("[stt] ready")

    def transcribe(self, audio: np.ndarray) -> str:
        """16 kHz float32 mono in, lowercase text out ('' if nothing usable)."""
        if audio.size < 1600:  # <0.1 s
            return ""
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=3, initial_prompt=COMMAND_HINT,
            condition_on_previous_text=False, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return text.lower().strip(".,!? ")
