"""PyAV reader for the Tello's UDP H.264 (the decoder djitellopy uses). Same interface as video_stream.VideoStream.

One thread demuxes and decodes every packet; a packet that fails to decode (lost UDP data) is skipped instead of
ending the stream. Nothing decodes before the first keyframe, so the timeouts must outlast the gap between them.
"""

import threading
import time

import av


class PyAVStream:
    def __init__(self, url, fps=60, open_timeout_ms=15000, read_timeout_ms=15000):
        try:
            self._container = av.open(url, format="h264", timeout=(open_timeout_ms / 1000.0, read_timeout_ms / 1000.0),
                                      options={"fflags": "nobuffer", "flags": "low_delay"})
        except (av.error.FFmpegError, OSError) as e:
            raise RuntimeError(f"Could not open video stream at {url}: {e}") from None
        self._interval = 1 / fps
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _update(self):
        last_convert = 0.0
        try:
            for packet in self._container.demux(video=0):
                if self._stopped.is_set():
                    break
                try:
                    frames = packet.decode()
                except av.error.FFmpegError:
                    continue  # corrupt packet: the decoder recovers at the next keyframe
                for f in frames:
                    now = time.time()
                    if now - last_convert >= self._interval:
                        img = f.to_ndarray(format="bgr24")
                        with self._lock:
                            self._frame, self._seq, self._ts = img, self._seq + 1, time.time()
                        last_convert = now
        except (av.error.FFmpegError, OSError):
            pass  # read timeout / stream ended: TelloVideoSource reopens when frames stop
        finally:
            self._container.close()

    def read_seq(self, copy=True):
        with self._lock:
            if self._frame is None:
                return -1, None, 0.0
            return self._seq, (self._frame.copy() if copy else self._frame), self._ts

    def stop(self):
        self._stopped.set()
        self._thread.join(timeout=2)  # a blocked demux ends at the read timeout and closes the container itself
