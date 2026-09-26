"""Multi-object tracker: gives detections of the same object the same track_id across frames.

Greedy association per class, best pairs first: IoU >= iou_match, or centre distance below
center_match_frac * image diagonal (helps when a fast yaw shifts the box more than its own width).
Swap for ByteTrack/BoT-SORT behind the same `update()` if needed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

from ..types import Detection, iou


@dataclass
class Track:
    id: int
    cls: str
    bbox: tuple[float, float, float, float]
    conf: float
    first_t: float
    last_t: float
    hits: int = 1
    seen_frames: list[int] = field(default_factory=list)  # frame indices where it was matched


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], t: float, image_size: tuple[int, int]) -> list[Detection]:
        """Returns the detections with track_id set."""

    @abstractmethod
    def reset(self) -> None: ...


class SimpleTracker(Tracker):
    def __init__(self, iou_match: float = 0.2, center_match_frac: float = 0.15, max_age_s: float = 1.0):
        self.iou_match = iou_match
        self.center_match_frac = center_match_frac
        self.max_age_s = max_age_s
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self.frame_index = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_index = 0

    def _score(self, tr: Track, d: Detection, diag: float) -> float:
        """Association score in (0, 2]; 0 = no match."""
        o = iou(tr.bbox, d.bbox)
        if o >= self.iou_match:
            return 1.0 + o
        if not tr.seen_frames or tr.seen_frames[-1] != self.frame_index - 1:
            return 0.0  # the centre fallback is for fast motion between consecutive frames only
        tcx, tcy = 0.5 * (tr.bbox[0] + tr.bbox[2]), 0.5 * (tr.bbox[1] + tr.bbox[3])
        dist = math.hypot(d.cx - tcx, d.cy - tcy) / diag
        if dist < self.center_match_frac:
            # also require similar size so a small far object does not steal a big near one
            ta = (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1])
            ratio = min(ta, d.area) / max(ta, d.area, 1e-9)
            if ratio > 0.35:
                return 1.0 - dist / self.center_match_frac
        return 0.0

    def update(self, detections: list[Detection], t: float, image_size: tuple[int, int]) -> list[Detection]:
        self.frame_index += 1
        diag = math.hypot(*image_size)
        # expire old tracks first
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        pairs = []
        for di, d in enumerate(detections):
            for tid, tr in self.tracks.items():
                if tr.cls == d.cls:
                    s = self._score(tr, d, diag)
                    if s > 0:
                        pairs.append((s, di, tid))
        pairs.sort(reverse=True)
        used_d, used_t, assign = set(), set(), {}
        for s, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            assign[di] = tid
        out = []
        for di, d in enumerate(detections):
            tid = assign.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, d.cls, d.bbox, d.conf, t, t, 1, [self.frame_index])
            else:
                tr = self.tracks[tid]
                tr.bbox, tr.conf, tr.last_t = d.bbox, d.conf, t
                tr.hits += 1
                tr.seen_frames.append(self.frame_index)
                del tr.seen_frames[:-50]
            out.append(replace(d, track_id=tid))
        return out
