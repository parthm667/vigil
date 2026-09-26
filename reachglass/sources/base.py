"""Frame source interface. Every camera (Tello, webcam, file replay, simulator) implements it.

read() is non-blocking and returns the NEWEST frame (or None before the first one). Consumers compare
Frame.seq to know whether it is new. Frames are BGR uint8 and must be treated as read-only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..types import Frame


class FrameSource(ABC):
    name = "source"

    def start(self) -> FrameSource:
        return self

    @abstractmethod
    def read(self) -> Frame | None:
        """Newest frame, or None if nothing has arrived yet."""

    def stop(self) -> None:
        pass

    def wait_first(self, timeout_s: float = 10.0) -> Frame | None:
        """Block until the first frame arrives (or timeout)."""
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            f = self.read()
            if f is not None:
                return f
            time.sleep(0.01)
        return None

    def __enter__(self) -> FrameSource:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
