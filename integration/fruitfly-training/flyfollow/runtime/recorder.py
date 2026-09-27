"""Session recorder: every bus message to recordings/<session>/bus.jsonl, optionally the video to video.mp4.

Layout of recordings/<session>/:
- bus.jsonl    one message per line, exactly as received, plus "t_rec" (recorder receive time; replay clock)
- video.mp4    H.264 of the frames read from the FrameRing on each "frame" message (with --video)
- frames.jsonl one line per encoded frame: {"n": video frame number, "frame_id", "t_decoded", "t_rec"}
- meta.json    session, mode, argv, start/end, message counts per topic
- logs/        child process logs (written by the launcher)

The video is variable-rate in reality (Wi-Fi drops); frames.jsonl is the timing truth, the mp4 fps is nominal.
Encoding shells out to ffmpeg (imageio-ffmpeg), so PyAV and OpenCV are never loaded here.

    python -m flyfollow.runtime.recorder --session test1 [--video] [--root recordings]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flyfollow.interfaces import REPO_ROOT
from flyfollow.runtime.bus import FrameRing, Publisher, Subscriber, default_frame_ring, encode
from flyfollow.runtime.messages import msg
from flyfollow.runtime.node import log, stop_event

NAME = "recorder"


def default_session(suffix: str = "") -> str:
    return time.strftime("%Y%m%d_%H%M%S") + (f"_{suffix}" if suffix else "")


def recordings_root() -> Path:
    return Path(os.environ.get("FLYFOLLOW_RECORDINGS", REPO_ROOT / "recordings"))


def read_jsonl(path: str | Path) -> list[dict]:
    """Lines of a .jsonl file; a truncated last line (crash) is skipped."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


class VideoWriter:
    """Background ffmpeg encoder fed from a bounded queue (frames are dropped, never blocking the recorder)."""

    def __init__(self, session_dir: Path, fps: float = 30.0, crf: int = 20, max_queue: int = 60):
        self.dir = session_dir
        self.fps = fps
        self.crf = crf
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.n = 0
        self.dropped = 0
        self.error: str | None = None
        self._index = (session_dir / "frames.jsonl").open("a")
        self._thread = threading.Thread(target=self._run, name="video-writer", daemon=True)
        self._thread.start()

    def put(self, frame: np.ndarray, frame_id: int, t_decoded: float, t_rec: float) -> None:
        try:
            self.q.put_nowait((frame, frame_id, t_decoded, t_rec))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        import imageio_ffmpeg

        gen = None
        shape = None
        try:
            while True:
                item = self.q.get()
                if item is None:
                    break
                frame, fid, td, tr = item
                if gen is None:
                    shape = frame.shape
                    gen = imageio_ffmpeg.write_frames(
                        str(self.dir / "video.mp4"), (shape[1], shape[0]), fps=self.fps, codec="libx264", quality=None,
                        output_params=["-preset", "veryfast", "-crf", str(self.crf)], macro_block_size=1,
                        ffmpeg_log_level="error")
                    gen.send(None)
                if frame.shape != shape:
                    self.dropped += 1
                    continue
                gen.send(np.ascontiguousarray(frame))
                self._index.write(json.dumps({"n": self.n, "frame_id": fid, "t_decoded": td, "t_rec": tr}) + "\n")
                self.n += 1
                if self.n % 30 == 0:
                    self._index.flush()
        except Exception as e:  # keep recording the bus even if ffmpeg dies
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if gen is not None:
                try:
                    gen.close()
                except Exception as e:
                    self.error = self.error or f"{type(e).__name__}: {e}"
            self._index.close()

    def close(self, timeout_s: float = 10.0) -> None:
        self.q.put(None)
        self._thread.join(timeout=timeout_s)


class Recorder:
    """Writes every received message; on "frame" messages copies the ring slot into the video writer."""

    def __init__(self, session_dir: str | Path, sub: Subscriber, video: bool = False, ring_name: str | None = None,
                 fps: float = 30.0, meta: dict | None = None):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sub = sub
        self.video = VideoWriter(self.dir, fps=fps) if video else None
        self.ring_name = ring_name or default_frame_ring()
        self.ring: FrameRing | None = None
        self._ring_retry = 0.0
        self.counts: collections.Counter = collections.Counter()
        self.frame_misses = 0
        self._f = (self.dir / "bus.jsonl").open("a")
        self._last_flush = time.monotonic()
        self.meta = {"session": self.dir.name, "started": time.time(), "argv": sys.argv, "video": video,
                     "mode": os.environ.get("FLYFOLLOW_MODE"), **(meta or {})}
        self._write_meta()

    def _write_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=1, default=str))

    def _frame(self, m: dict, t_rec: float) -> None:
        if self.ring is None:
            if time.monotonic() < self._ring_retry:
                return
            try:
                self.ring = FrameRing.attach(self.ring_name)
            except (FileNotFoundError, RuntimeError):
                self._ring_retry = time.monotonic() + 1.0
                return
        img = self.ring.read(int(m.get("slot", -1)), int(m.get("frame_id", -1)))
        if img is None:
            self.frame_misses += 1
            return
        self.video.put(img, int(m["frame_id"]), float(m.get("t_decoded", m["t"])), t_rec)

    def handle(self, m: dict) -> None:
        t_rec = time.time()
        m["t_rec"] = t_rec
        self._f.write(encode(m).decode() + "\n")
        self.counts[m.get("topic", "?")] += 1
        if self.video is not None and m.get("topic") == "frame":
            self._frame(m, t_rec)
        if time.monotonic() - self._last_flush > 0.5:
            self._f.flush()
            self._last_flush = time.monotonic()

    def poll(self, timeout_s: float = 0.1) -> int:
        n = 0
        m = self.sub.recv(timeout_s)
        while m is not None:
            self.handle(m)
            n += 1
            if n >= 1000:
                break
            m = self.sub.recv(0.0)
        return n

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.poll(0.1)

    def close(self) -> dict:
        for m in self.sub.drain(10_000):
            self.handle(m)
        self._f.close()
        if self.video is not None:
            self.video.close()
        if self.ring is not None:
            self.ring.close()
        self.meta.update(ended=time.time(), n_msgs=sum(self.counts.values()), counts=dict(self.counts),
                         n_frames=self.video.n if self.video else 0,
                         frames_dropped=(self.video.dropped + self.frame_misses) if self.video else 0,
                         video_error=self.video.error if self.video else None)
        self._write_meta()
        return self.meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="record the runtime bus (and video) for replay")
    ap.add_argument("--session", default=None, help="session name (default: timestamp)")
    ap.add_argument("--root", default=None, help="recordings root (default: recordings/ or $FLYFOLLOW_RECORDINGS)")
    ap.add_argument("--video", action="store_true", help="also encode FrameRing frames to video.mp4")
    ap.add_argument("--fps", type=float, default=30.0, help="nominal mp4 fps (frames.jsonl has the real timing)")
    a = ap.parse_args(argv)
    stop = stop_event()
    session = a.session or os.environ.get("FLYFOLLOW_SESSION") or default_session()
    root = Path(a.root) if a.root else recordings_root()
    sub = Subscriber(None)
    rec = Recorder(root / session, sub, video=a.video, fps=a.fps)
    log(NAME, f"recording to {rec.dir}" + (" (bus + video)" if a.video else " (bus only)"))
    pub = Publisher(NAME)  # tells the launcher it may start the rest (this message is recorded too)
    pub.publish(msg("health", key="recorder", ok=True, level="ok", text=f"recorder ready: {rec.dir}"))
    try:
        rec.run(stop)
    finally:
        meta = rec.close()
        sub.close()
        pub.close()
        log(NAME, f"closed: {meta['n_msgs']} messages, {meta['n_frames']} frames"
            + (f", video error: {meta['video_error']}" if meta.get("video_error") else ""))


if __name__ == "__main__":
    main()

