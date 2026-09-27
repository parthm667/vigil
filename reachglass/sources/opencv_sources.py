"""OpenCV-backed frame sources: the Tello UDP stream, a webcam, and a video file."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from ..types import Frame
from .base import FrameSource

TELLO_UDP_URL = "udp://@0.0.0.0:11111"


def _readonly(img: np.ndarray) -> np.ndarray:
    """Frames are shared between consumers: forbid in-place drawing (copy before annotating)."""
    img.flags.writeable = False
    return img


class TelloVideoSource(FrameSource):
    """The Tello's H.264 stream, decoded by PyAV (backend "pyav", default: djitellopy's decoder) or by the team's
    OpenCV VideoStream (backend "opencv").

    The drone must already have been told `streamon` (the drone adapter does that). Only one reader
    can bind UDP 11111: every consumer shares this source.

    start() never raises: one thread opens the stream, retrying until the drone's video decodes (FFmpeg
    gives up after its open timeout, e.g. when the drone starts streaming late), and reopens it when
    frames stop for longer than the read timeout (after one the capture never delivers again). seq keeps increasing across reopens.

    fps is the reader's retrieve cap. It must sit well above the ~30 fps stream: a cap equal to the
    stream rate drops ~40 % of frames because of arrival jitter (found in review).
    """

    name = "tello"

    def __init__(self, url: str = TELLO_UDP_URL, fps: int = 60, timeout_ms: int = 15000, reconnect_after_s: float = 20.0,
                 backend: str = "pyav"):
        if backend not in ("pyav", "opencv"):
            raise ValueError(f"video backend {backend!r}: use pyav or opencv")
        self.url, self.fps, self.backend = url, fps, backend
        # FFmpeg open/read timeout. It must outlast the wait for a clean keyframe: nothing decodes before one, and
        # the Tello sends them seconds apart (a lost packet costs a whole one). With 5 s / 3 s the capture was
        # aborted before its first keyframe on a weak link, and a timed-out capture never delivers again.
        self.timeout_ms = timeout_ms
        self.reconnect_after_s = reconnect_after_s  # > timeout: a shorter gap just pauses grab(), then it resumes
        self._stream = None
        self._lock = threading.Lock()
        self._offset = 0  # seq offset so seq stays monotonic across reopens
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.reconnects = 0

    def start(self) -> TelloVideoSource:
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        if self.backend == "pyav":
            from .pyav_stream import PyAVStream as Stream
        else:
            from .video_stream import VideoStream as Stream

        while not self._stop.is_set():
            try:
                stream = Stream(self.url, fps=self.fps, open_timeout_ms=self.timeout_ms, read_timeout_ms=self.timeout_ms).start()
            except RuntimeError:  # nothing decodable before the open timeout: try again
                self._stop.wait(0.5)
                continue
            with self._lock:
                self._stream = stream
            last_seq, last_new_t = -1, time.time()
            while not self._stop.wait(0.25):
                seq = stream.read_seq(copy=False)[0]
                if seq != last_seq:
                    last_seq, last_new_t = seq, time.time()
                elif time.time() - last_new_t > self.reconnect_after_s:
                    self.reconnects += 1
                    break
            with self._lock:
                self._stream = None
                self._offset += max(stream.read_seq(copy=False)[0], 0)
            stream.stop()  # its thread leaves grab() within the read timeout -> safe to release

    def read(self) -> Frame | None:
        with self._lock:
            if self._stream is None:
                return None
            seq, img, ts = self._stream.read_seq(copy=False)
            offset = self._offset
        return None if img is None else Frame(_readonly(img), ts, seq + offset, self.name)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)  # a stalled capture is released by the thread itself (daemon)
            self._thread = None
        with self._lock:
            self._stream = None


class _ThreadedCapture(FrameSource):
    """Grabs frames in a thread and keeps only the newest (no queue -> no growing latency)."""

    def __init__(self, name: str, pace_fps: float | None = None, loop: bool = False):
        self.name = name
        self._pace = pace_fps
        self._loop = loop
        self._cap = None
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.finished = False  # file sources: reached the end

    def _open(self):
        raise NotImplementedError

    def start(self):
        if self._thread is None:
            self._stop.clear()  # restartable after stop()
            self.finished = False
            with self._lock:
                self._frame = None  # never serve a frame from the previous run (seq stays monotonic)
            self._cap = self._open()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        period = 1.0 / self._pace if self._pace else 0.0
        next_t = time.time()
        failed_after_seek = False
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                if self._loop and not failed_after_seek:
                    self._cap.set(1, 0)  # CAP_PROP_POS_FRAMES: rewind
                    failed_after_seek = True
                    continue
                self.finished = True  # end of file, or unreadable even after a rewind
                break
            failed_after_seek = False
            if period:
                next_t += period
                delay = next_t - time.time()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.time()
            with self._lock:
                self._seq += 1
                self._frame = Frame(_readonly(img), time.time(), self._seq, self.name)

    def read(self) -> Frame | None:
        with self._lock:
            return self._frame

    def stop(self) -> None:
        self._stop.set()
        alive = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            alive = self._thread.is_alive()
            self._thread = None
        if self._cap is not None and not alive:  # never release under a blocked read
            self._cap.release()
        self._cap = None
        with self._lock:
            self._frame = None


class WebcamSource(_ThreadedCapture):
    def __init__(self, index: int = 0, width: int = 960, height: int = 720):
        super().__init__("webcam")
        self.index, self.width, self.height = index, width, height

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open webcam {self.index}")
        return cap


class VideoFileSource(_ThreadedCapture):
    """Replays a recorded video at its own frame rate (like a live camera: newest frame only)."""

    def __init__(self, path: str | Path, loop: bool = False, fps: float | None = None):
        super().__init__("file", loop=loop)
        self.path = str(path)
        self._fps_override = fps

    def _open(self):
        import cv2

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        self._pace = self._fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
        return cap


class SteppedVideoSource(FrameSource):
    """Deterministic replay for tests/offline analysis: every read() returns the NEXT frame."""

    name = "stepped"

    def __init__(self, path: str | Path, fps: float | None = None):
        import cv2

        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self._dt = 1.0 / (fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._seq = 0
        self.finished = False

    def read(self) -> Frame | None:
        ok, img = self._cap.read()
        if not ok:
            self.finished = True
            return None
        self._seq += 1
        return Frame(_readonly(img), self._seq * self._dt, self._seq, self.name)

    def stop(self) -> None:
        self._cap.release()


class StaticImageSource(FrameSource):
    """One image, re-served as a new frame on every read (useful to exercise detectors live)."""

    name = "image"

    def __init__(self, image: np.ndarray):
        self.image = _readonly(image.copy())
        self._seq = 0

    def read(self) -> Frame | None:
        self._seq += 1
        return Frame(self.image, time.time(), self._seq, self.name)
