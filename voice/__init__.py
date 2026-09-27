"""AirPods Pro 2 voice layer for ReachGlass (Windows 11).

Runs as a separate process from the flight stack on purpose: if audio dies, the
drone keeps flying and typed queries still work.

    python -m voice                 # full loop: stem press -> STT -> UDP 5005; TTS <- UDP 5006
    python -m voice --text "..."    # wiring test: send one command, print announcements
    python -m voice --selftest      # hardware smoke test (devices, stem, record, STT, TTS)
"""
