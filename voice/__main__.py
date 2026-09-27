"""Voice app CLI. See voice/__init__.py for the modes."""

from __future__ import annotations

import argparse
import socket
import sys
import time

from .config import VoiceCfg


def open_announce_rx(cfg: VoiceCfg) -> socket.socket:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", cfg.announce_port))
    rx.setblocking(False)
    return rx


def poll_announcements(rx: socket.socket) -> list[str]:
    out = []
    while True:
        try:
            data, _ = rx.recvfrom(4096)
        except (BlockingIOError, OSError):
            break
        text = data.decode("utf-8", errors="ignore").strip()
        if text:
            out.append(text)
    return out


def send_query(cfg: VoiceCfg, text: str) -> None:
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(text.encode("utf-8"), (cfg.send_host, cfg.send_port))
    tx.close()


def cmd_text(cfg: VoiceCfg, text: str, listen_s: float, speak: bool) -> int:
    """Wiring test: no stem, no mic. Send one command, print (and optionally speak)
    every announcement the app publishes for the next listen_s seconds."""
    rx = open_announce_rx(cfg)
    speaker = None
    if speak:
        from .tts import Speaker

        speaker = Speaker(voice=cfg.tts_voice)
    send_query(cfg, text)
    print(f"[voice] sent {text!r} -> {cfg.send_host}:{cfg.send_port}; "
          f"listening on :{cfg.announce_port} for {listen_s:.0f}s")
    t_end = time.time() + listen_s
    n = 0
    try:
        while time.time() < t_end:
            for msg in poll_announcements(rx):
                n += 1
                print(f"[voice] announce: {msg}")
                if speaker is not None:
                    speaker.speak(msg)
            time.sleep(0.05)
    finally:
        if speaker is not None:
            speaker.wait()
            speaker.close()
        rx.close()
    print(f"[voice] done: {n} announcement(s) received")
    return 0 if n else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", help="wiring test: send this command once and print announcements")
    p.add_argument("--listen", type=float, default=30.0, help="--text: seconds to listen for announcements")
    p.add_argument("--speak", action="store_true", help="--text: also speak the announcements")
    p.add_argument("--selftest", action="store_true", help="hardware smoke test (devices, stem, record, STT, TTS)")
    p.add_argument("--stt", default=None, help="whisper model (default base.en; tiny.en if slow)")
    p.add_argument("--tts", choices=["auto", "edge", "sapi"], default="auto")
    p.add_argument("--device", default=None, help="input device name substring (default AirPods)")
    p.add_argument("--send-port", type=int, default=None)
    p.add_argument("--announce-port", type=int, default=None)
    args = p.parse_args(argv)

    cfg = VoiceCfg()
    if args.stt:
        cfg.stt_model = args.stt
    if args.device:
        cfg.input_substr = args.device
    if args.send_port:
        cfg.send_port = args.send_port
    if args.announce_port:
        cfg.announce_port = args.announce_port

    if args.text:
        return cmd_text(cfg, args.text, args.listen, args.speak)
    if args.selftest:
        from .selftest import run_selftest

        return run_selftest(cfg, tts_backend=args.tts)
    from .loop import VoiceApp

    return VoiceApp(cfg, tts_backend=args.tts).run()


if __name__ == "__main__":
    sys.exit(main())
