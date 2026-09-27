"""Synthesized cue tones -- no asset files. Each returns (float32 mono, rate)."""

from __future__ import annotations

import numpy as np

RATE = 24000


def _env(n: int, fade: int = 200) -> np.ndarray:
    e = np.ones(n, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, min(fade, n // 2), dtype=np.float32)
    e[: ramp.size] = ramp
    e[-ramp.size:] = ramp[::-1]
    return e


def _tone(freq_hz: float, dur_s: float, gain: float = 0.3) -> np.ndarray:
    t = np.arange(int(RATE * dur_s), dtype=np.float32) / RATE
    return (gain * np.sin(2 * np.pi * freq_hz * t) * _env(t.size)).astype(np.float32)


def listen_chirp() -> tuple[np.ndarray, int]:
    """Rising chirp: 'the mic is live, speak now' (also masks the HFP switch gap)."""
    t = np.arange(int(RATE * 0.25), dtype=np.float32) / RATE
    freq = 600.0 + (1200.0 - 600.0) * t / t[-1]
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    return (0.3 * np.sin(phase) * _env(t.size)).astype(np.float32), RATE


def got_it_blip() -> tuple[np.ndarray, int]:
    """Short double blip: 'utterance captured'."""
    gap = np.zeros(int(RATE * 0.05), dtype=np.float32)
    return np.concatenate([_tone(880, 0.08), gap, _tone(1100, 0.08)]), RATE


def error_buzz() -> tuple[np.ndarray, int]:
    """Low buzz: 'did not catch that'."""
    return _tone(220, 0.3, gain=0.35), RATE


def play(sound: tuple[np.ndarray, int], device=None) -> None:
    """Blocking playback. device None = the AirPods' live output endpoint if present, else
    the system default (the default may be a silent A2DP endpoint while the mic holds HFP)."""
    import sounddevice as sd

    if device is None:
        from .audio import find_output_device

        device = find_output_device()
    data, rate = sound
    sd.play(data, rate, device=device, blocking=True)
