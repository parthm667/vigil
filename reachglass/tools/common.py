"""Open a frame source for the setup tools: the Tello (connect + streamon), a webcam index, or a file."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..sources import StaticImageSource, TelloVideoSource, VideoFileSource, WebcamSource


class Opened:
    def __init__(self, source, tello=None):
        self.source, self.tello = source, tello

    def close(self):
        self.source.stop()
        if self.tello is not None:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            self.tello.end()


def open_source(spec: str) -> Opened:
    """'tello' | a webcam index ('0') | an image or video path."""
    if spec == "tello":
        from djitellopy import Tello

        Tello.LOGGER.setLevel(logging.WARNING)
        t = Tello()
        t.connect()
        print(f"Tello battery {t.get_battery()}%")
        t.streamon()
        src = TelloVideoSource(t.get_udp_video_address()).start()
        return Opened(src, t)
    if spec.isdigit():
        return Opened(WebcamSource(int(spec)).start())
    img = cv2.imread(spec)
    if img is not None:
        return Opened(StaticImageSource(img))
    return Opened(VideoFileSource(spec, loop=True).start())


def latest(src, timeout_s: float = 10.0) -> np.ndarray | None:
    f = src.wait_first(timeout_s) if hasattr(src, "wait_first") else src.read()
    f = src.read() or f
    return None if f is None else f.image
