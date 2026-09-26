"""Low-latency OpenCV/FFmpeg reader for the Tello's UDP video (the team's `video_stream.py`).

Kept as written by the team except (marked CHANGED):
  * one comment correction: with the FFmpeg backend grab() does decode each H.264 frame (it has to,
    frames depend on each other); retrieve() only converts the last decoded frame to BGR.
  * the capture is opened with open/read timeouts, and stop() never releases the capture while the
    reader thread may still be inside grab(): releasing under a blocked grab() on a stalled stream
    segfaults the process (found in review; reproduced by killing the UDP sender).
"""

import os
import threading
import time

import cv2

# Must be set before the first VideoCapture is opened. Tells ffmpeg's
# demuxer not to build up its own internal buffer and to drop frames it
# can't keep up with, rather than queueing them -- this is what actually
# stops latency from growing the longer the stream runs.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|framedrop;1",
)


class VideoStream:
    """
    Reads the Tello's UDP video feed directly via OpenCV's FFmpeg backend,
    instead of djitellopy's PyAV-based reader.

    A background thread continuously grab()s frames to keep the stream drained
    in real time. It only pays the cost of retrieve() (colour conversion) once
    per target interval, so consumers see at most `fps` new frames per second
    regardless of how fast frames arrive.
    """

    def __init__(self, url, fps=30, open_timeout_ms=5000, read_timeout_ms=3000):
        # CHANGED: bounded open/read so a stalled stream cannot block grab() forever
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
                                                          cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms])
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream at {url}")

        self._read_timeout_s = read_timeout_ms / 1000.0
        self._interval = 1 / fps
        self._frame = None
        self._seq = 0      # bumped per decoded frame so consumers can tell "new" from "same"
        self._ts = 0.0     # time.time() at retrieve(), i.e. when the frame became available
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _update(self):
        last_retrieve = 0.0
        while not self._stopped.is_set():
            if not self.cap.grab():
                time.sleep(0.005)
                continue

            now = time.time()
            if now - last_retrieve >= self._interval:
                ok, frame = self.cap.retrieve()
                if ok:
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                        self._ts = time.time()
                last_retrieve = now

    def read(self):
        """Returns a copy of the latest frame, or None if nothing yet."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def read_seq(self, copy=True):
        """Returns (seq, frame, timestamp). seq only changes when a new frame was decoded.

        copy=False hands back the internal array to skip a ~2.7 MB memcpy per frame; the
        caller must then treat it as read-only. _update never mutates a frame in place, it
        rebinds self._frame to whatever retrieve() allocated, so a stale reference stays valid.
        """
        with self._lock:
            if self._frame is None:
                return -1, None, 0.0
            return self._seq, (self._frame.copy() if copy else self._frame), self._ts

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._read_timeout_s + 0.5)  # a blocked grab() returns within the read timeout
        # CHANGED: only release once the reader thread is out of grab(); otherwise leak it (daemon)
        if not self._thread.is_alive():
            self.cap.release()
