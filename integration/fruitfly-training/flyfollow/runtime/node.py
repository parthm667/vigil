"""Small shared helpers for runtime processes: stop signal, fixed-rate loop, logging, liveness.

    stop = stop_event()                      # set by SIGINT / SIGTERM (the launcher stops children with SIGTERM)
    rate = Rate(20.0)
    while not stop.is_set():
        ...
        rate.sleep()
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field


def stop_event() -> threading.Event:
    """Event set on SIGINT or SIGTERM (main thread only). A second signal exits immediately."""
    ev = threading.Event()

    def handler(signum, frame):
        if ev.is_set():
            os._exit(130)
        ev.set()

    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except ValueError:  # not the main thread
                pass
    return ev


def log(name: str, *parts, file=None) -> None:
    """One timestamped, flushed line: HH:MM:SS.mmm [name] ..."""
    t = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"
    print(stamp, f"[{name}]", *parts, file=file or sys.stdout, flush=True)


@dataclass
class Rate:
    """Fixed-rate loop timer on the monotonic clock. Skips ahead (counting overruns) instead of bursting."""

    hz: float
    overruns: int = 0
    _next: float = field(default=0.0, repr=False)

    def sleep(self) -> None:
        period = 1.0 / self.hz
        now = time.monotonic()
        if self._next == 0.0:
            self._next = now
        self._next += period
        dt = self._next - now
        if dt > 0:
            time.sleep(dt)
        else:
            self.overruns += 1
            self._next = now


@dataclass
class Liveness:
    """Last-seen times per topic; stale() lists topics older than their limit (never seen counts after grace_s)."""

    limits: dict[str, float]
    grace_s: float = 10.0
    t0: float = field(default_factory=time.time)
    last: dict[str, float] = field(default_factory=dict)

    def see(self, topic: str, t: float | None = None) -> None:
        self.last[topic] = time.time() if t is None else t

    def age(self, topic: str, now: float | None = None) -> float | None:
        t = self.last.get(topic)
        return None if t is None else (now or time.time()) - t

    def stale(self, now: float | None = None) -> dict[str, float | None]:
        now = now or time.time()
        out: dict[str, float | None] = {}
        for tp, lim in self.limits.items():
            a = self.age(tp, now)
            if a is None:
                if now - self.t0 > self.grace_s:
                    out[tp] = None
            elif a > lim:
                out[tp] = a
        return out
