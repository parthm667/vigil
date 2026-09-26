"""Replay a recorded session onto the bus (plan R1, open loop): develop detection, control and mission logic without flying.

Timing follows the recorder's receive clock ("t_rec"), at real time or N x speed. By default every time field
("t", "t_decoded") is shifted so messages look live (t_new = t + (replay wall time - t_rec)); --original-t keeps them.
With --frames the video is decoded into a FrameRing (the same name Tello I/O uses) and fresh "frame" messages are
published for it; recorded "frame" messages are never replayed (their slots point at a ring that no longer exists).
Command topics (kill, tello_cmd, rc, mode_cmd, settings, lock, det_cfg) and health are skipped unless named in
--topics, and every replayed message carries "replayed": true so a live Tello I/O can refuse it.

    python -m flyfollow.runtime.replay recordings/<session> [--speed 1] [--topics det,tello_state] [--frames]
        [--exclude rc,mode] [--start 10] [--duration 30] [--loop] [--original-t] [--no-broker]

A broker is started in-process unless one is already running (or --no-broker).
"""

from __future__ import annotations

import argparse
import heapq
import threading
import time
from pathlib import Path

from flyfollow.runtime.bus import Broker, FrameRing, Publisher
from flyfollow.runtime.node import log, stop_event
from flyfollow.runtime.recorder import read_jsonl

NAME = "replay"
TIME_FIELDS = ("t", "t_decoded")
# Never replayed unless named in --topics: they would command a live drone or reconfigure live processes.
COMMAND_TOPICS = frozenset(("kill", "tello_cmd", "rc", "mode_cmd", "settings", "lock", "det_cfg", "health"))


def resolve_session(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        p = p.parent
    if not (p / "bus.jsonl").exists():
        raise FileNotFoundError(f"no bus.jsonl in {p}")
    return p


class Replayer:
    """Merges bus.jsonl and frames.jsonl on t_rec and publishes them on a schedule."""

    def __init__(self, session: str | Path, pub: Publisher, topics: list[str] | None = None,
                 exclude: list[str] | None = None, frames: bool = False, speed: float = 1.0, retime: bool = True,
                 start_s: float = 0.0, duration_s: float | None = None, ring_name: str | None = None):
        self.dir = resolve_session(session)
        self.pub = pub
        self.speed = max(1e-3, speed)
        self.retime = retime
        self.frames = frames
        self.ring_name = ring_name
        self.ring: FrameRing | None = None
        topics_set = set(topics) if topics else None
        excl = set(exclude or []) | {"frame"} | (COMMAND_TOPICS - (topics_set or set()))
        msgs = [m for m in read_jsonl(self.dir / "bus.jsonl")
                if (topics_set is None or m.get("topic") in topics_set) and m.get("topic") not in excl]
        for m in msgs:
            m.setdefault("t_rec", m.get("t", 0.0))
        self.index = read_jsonl(self.dir / "frames.jsonl") if frames else []
        if frames and not (self.dir / "video.mp4").exists():
            raise FileNotFoundError(f"--frames needs {self.dir / 'video.mp4'} (record with --video)")
        t_all = [m["t_rec"] for m in msgs] + [f["t_rec"] for f in self.index]
        self.t0 = min(t_all) if t_all else 0.0
        lo = self.t0 + start_s
        hi = lo + duration_s if duration_s is not None else float("inf")
        self.msgs = [m for m in msgs if lo <= m["t_rec"] <= hi]
        self.index = [f for f in self.index if f["t_rec"] <= hi]  # frames before lo are decoded and skipped
        self.lo, self.hi = lo, hi
        self.max_frame_id = max([int(f["frame_id"]) for f in self.index] + [int(m.get("frame_id", 0) or 0)
                                                                              for m in self.msgs] + [0])
        self.published = 0
        self.frames_published = 0

    def _events(self):
        a = ((m["t_rec"], 1, i, m) for i, m in enumerate(self.msgs))
        b = ((f["t_rec"], 0, i, f) for i, f in enumerate(self.index))  # frames first on ties
        return heapq.merge(a, b, key=lambda e: (e[0], e[1], e[2]))

    def _open_video(self):
        import imageio_ffmpeg

        gen = imageio_ffmpeg.read_frames(str(self.dir / "video.mp4"), pix_fmt="rgb24")
        meta = next(gen)
        w, h = meta["size"]
        if self.ring is None:
            self.ring = FrameRing.create(self.ring_name, h=h, w=w) if self.ring_name else FrameRing.create(h=h, w=w)
        return gen, w, h

    def run_once(self, stop: threading.Event | None = None, loop_i: int = 0) -> None:
        import numpy as np

        stop = stop or threading.Event()
        gen = None
        w = h = 0
        n_decoded = 0
        if self.index:
            gen, w, h = self._open_video()
        id_off = loop_i * (self.max_frame_id + 1)
        wall0 = time.time()
        first = None
        for t_rec, kind, _, item in self._events():
            if stop.is_set():
                break
            if kind == 0 and t_rec < self.lo:  # decode frames before the start window, do not publish
                while n_decoded <= item["n"]:
                    next(gen)
                    n_decoded += 1
                continue
            if first is None:
                first = t_rec
            due = wall0 + (t_rec - first) / self.speed
            while (dt := due - time.time()) > 0 and not stop.is_set():
                time.sleep(min(dt, 0.1))
            off = due - t_rec
            if kind == 0:
                img = None
                while n_decoded <= item["n"]:
                    try:
                        raw = next(gen)
                    except StopIteration:
                        raw = None
                        break
                    n_decoded += 1
                    if n_decoded - 1 == item["n"]:
                        img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                if img is None:
                    continue
                fid = int(item["frame_id"]) + id_off
                td = float(item["t_decoded"]) + (off if self.retime else 0.0)
                slot = self.ring.write(img, fid, td)
                self.pub.publish({"topic": "frame", "t": due if self.retime else t_rec, "frame_id": fid,
                                  "t_decoded": td, "slot": slot, "w": w, "h": h, "replayed": True})
                self.frames_published += 1
            else:
                m = dict(item)
                m.pop("t_rec", None)
                m["replayed"] = True
                if self.retime:
                    for k in TIME_FIELDS:
                        if isinstance(m.get(k), (int, float)):
                            m[k] = m[k] + off
                if id_off and isinstance(m.get("frame_id"), int):
                    m["frame_id"] += id_off
                self.pub.publish(m)
                self.published += 1
        if gen is not None:
            gen.close()

    def run(self, stop: threading.Event | None = None, loop: bool = False) -> None:
        i = 0
        while True:
            self.run_once(stop, loop_i=i)
            i += 1
            if not loop or (stop is not None and stop.is_set()):
                break

    def close(self) -> None:
        if self.ring is not None:
            self.ring.close()
            self.ring.unlink()
            self.ring = None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="replay a recorded session onto the bus")
    ap.add_argument("session", help="recordings/<session> (or its bus.jsonl)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--topics", default="", help="comma-separated topics to replay (default: all recorded)")
    ap.add_argument("--exclude", default="", help="comma-separated topics to skip")
    ap.add_argument("--frames", action="store_true", help="decode video.mp4 into the FrameRing and publish frame")
    ap.add_argument("--original-t", action="store_true", help="keep recorded t / t_decoded instead of shifting to now")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the recording")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-broker", action="store_true", help="never start a broker (one must be running)")
    a = ap.parse_args(argv)
    stop = stop_event()
    broker = None
    if not a.no_broker:
        try:
            broker = Broker().start()
            log(NAME, f"started broker {broker.pub_addr} -> {broker.sub_addr}")
        except RuntimeError:
            broker = None  # one is already running (the launcher's)
    pub = Publisher(NAME)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    rp = Replayer(a.session, pub, topics=split(a.topics) or None, exclude=split(a.exclude), frames=a.frames,
                  speed=a.speed, retime=not a.original_t, start_s=a.start, duration_s=a.duration)
    log(NAME, f"{rp.dir}: {len(rp.msgs)} messages, {len(rp.index)} frames, speed {a.speed}x"
        + (", looping" if a.loop else ""))
    try:
        rp.run(stop, loop=a.loop)
    finally:
        rp.close()
        pub.close()
        if broker is not None:
            broker.stop()
        log(NAME, f"done: {rp.published} messages, {rp.frames_published} frames")


if __name__ == "__main__":
    main()
