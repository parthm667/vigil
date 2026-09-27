"""Open a frame source for the setup tools: the Tello (video only, like the app's dry run), a webcam index, or a file."""

from __future__ import annotations

import cv2
import numpy as np

from ..sources import StaticImageSource, VideoFileSource, WebcamSource


class Opened:
    def __init__(self, source, drone=None):
        self.source, self.drone = source, drone

    def close(self):
        self.source.stop()
        if self.drone is not None:
            self.drone.close()  # stops the video, then djitellopy's end() sends streamoff


def open_source(spec: str) -> Opened:
    """'tello' | a webcam index ('0') | an image or video path."""
    if spec == "tello":
        from ..drone.tello import TelloDrone

        drone = TelloDrone(dry_run=True)  # same connect/streamon/video path as the app, never sends motion
        try:
            drone.connect()
        except Exception as e:
            drone.close()
            raise SystemExit(f"cannot start the Tello: {e}") from None
        print(f"Tello battery {drone.telemetry().battery_pct}%")
        return Opened(drone.frame_source(), drone)
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
