"""Voice-layer settings. Everything time-critical here was measured or sourced:
the A2DP<->HFP profile switch on AirPods costs ~0.5-2 s of dropout each way, and a
stem double-press can arrive as PAUSE followed by NEXT within a couple hundred ms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VoiceCfg:
    # transport (reachglass side is --udp 5005 in, --announce-udp 5006 out)
    send_host: str = "127.0.0.1"
    send_port: int = 5005          # recognized text -> reachglass UdpInbox
    announce_port: int = 5006      # spoken announcements <- reachglass say()

    # audio devices
    input_substr: str = "AirPods"  # matched against input device names at every open
    sample_rate: int = 16000       # HFP wideband rate; whisper's native rate

    # stem presses
    debounce_ms: int = 250         # window to collapse PAUSE+NEXT into one double-press

    # push-to-talk recording
    max_utterance_s: float = 6.0
    trailing_silence_s: float = 0.8
    silence_rms: float = 0.010     # energy floor; below this counts as silence
    post_mic_pad_s: float = 0.4    # wait after closing mic before TTS (HFP->A2DP switch)

    # models
    stt_model: str = "base.en"     # --stt tiny.en if the laptop is slow
    tts_voice: str = "en-US-AriaNeural"
