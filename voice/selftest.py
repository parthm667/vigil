"""Hardware smoke test: run this FIRST on the demo laptop with the AirPods in.

  python -m voice --selftest

Checks, in order: audio devices, TTS (both backends), stem press via SMTC,
push-to-talk recording + playback, transcription. Prints PASS/FAIL per step.
"""

from __future__ import annotations

import time

import numpy as np

from . import earcons
from .audio import Recorder, find_input_device, play
from .config import VoiceCfg
from .stem import StemListener


def run_selftest(cfg: VoiceCfg, tts_backend: str = "auto") -> int:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, note: str = "") -> None:
        results.append((name, ok, note))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{': ' + note if note else ''}")

    print("== 1/5 audio devices ==")
    idx, name = find_input_device(cfg.input_substr)
    check("input device", True, f"{'AirPods found: ' + name if idx is not None else 'no AirPods, default: ' + name}")
    if idx is None:
        print(f"  (connect the AirPods and re-run to test the real path)")

    print("== 2/5 text-to-speech ==")
    from .tts import Speaker

    for backend in (["edge", "sapi"] if tts_backend == "auto" else [tts_backend]):
        try:
            s = Speaker(voice=cfg.tts_voice, backend=backend)
            s.speak(f"This is the {backend} voice.")
            s.wait(20.0)
            s.close()
            check(f"tts {backend}", True)
        except Exception as e:
            check(f"tts {backend}", False, f"{type(e).__name__}: {e}")

    print("== 3/5 stem press (press an AirPod stem once, 10 s) ==")
    stem = StemListener(cfg.debounce_ms)
    if stem.ok:
        g = stem.get(timeout=10.0)
        check("stem press", g is not None, g or "nothing received (AirPods connected? pressed?)")
    else:
        check("stem press", False, "SMTC session could not be created")

    print("== 4/5 push-to-talk recording (speak after the chirp) ==")
    rec = Recorder(cfg.input_substr, cfg.sample_rate, max_s=4.0,
                   trailing_silence_s=cfg.trailing_silence_s, silence_rms=cfg.silence_rms)
    t0 = time.time()
    audio = rec.record(on_live=lambda: earcons.play(earcons.listen_chirp()))
    dur = audio.size / cfg.sample_rate
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    check("recording", audio.size > 0 and rms > 1e-4,
          f"{dur:.1f}s rms={rms:.4f} wall={time.time() - t0:.1f}s (profile-switch overhead included)")
    if audio.size:
        print("  playing it back...")
        play(audio, cfg.sample_rate)

    print("== 5/5 transcription ==")
    try:
        from .stt import Transcriber

        text = Transcriber(cfg.stt_model).transcribe(audio)
        check("stt", bool(text), repr(text))
    except Exception as e:
        check("stt", False, f"{type(e).__name__}: {e}")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" +
          (f" -- FAILED: {', '.join(r[0] for r in failed)}" if failed else " -- all good, fly it"))
    return 1 if failed else 0
