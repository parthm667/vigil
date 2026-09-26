"""Target lock: pick ONE object of a class, confirm it, stick to it, and say when it is lost.

    lock = TargetLock(tracker, confirm_hits=3, confirm_window=5)
    lock.set_class("bottle")
    state = lock.update(tracked_detections, t)     # detections already carry track_ids
    if state.confirmed and not state.lost: steer toward state.det

Rules
  * candidates: detections of the locked class; `select` picks one when nothing is locked
    (default: most confident; people use largest)
  * once locked, the lock follows that track_id only. It does NOT jump to another object of the
    same class while the locked one is still being tracked.
  * confirmed: the track was matched in >= confirm_hits of the last confirm_window frames, or one
    detection with conf >= confirm_conf that also passes `plausible(det)` (e.g. a size check)
  * lost: not seen for lost_after_s. While lost, a new candidate close to the last position is
    preferred for re-locking; after the lock is released any candidate can be locked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from ..types import Detection
from .tracker import SimpleTracker


def most_confident(cands: list[Detection]) -> Detection:
    return max(cands, key=lambda d: d.conf)


def largest(cands: list[Detection]) -> Detection:
    return max(cands, key=lambda d: d.area)


@dataclass
class LockState:
    cls: str | None = None
    det: Detection | None = None  # detection in THIS frame (None if not seen this frame)
    last_det: Detection | None = None  # last detection of the locked object (for display / memory)
    track_id: int | None = None
    confirmed: bool = False
    lost: bool = True
    unseen_s: float = math.inf
    hits: int = 0

    @property
    def visible(self) -> bool:
        return self.det is not None


class TargetLock:
    def __init__(self, tracker: SimpleTracker, confirm_hits: int = 3, confirm_window: int = 5, confirm_conf: float = 0.55,
                 lost_after_s: float = 1.5, select: Callable[[list[Detection]], Detection] = most_confident,
                 plausible: Callable[[Detection], bool] | None = None, relock_radius_frac: float = 0.25):
        self.tracker = tracker
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.confirm_conf = confirm_conf
        self.lost_after_s = lost_after_s
        self.select = select
        self.plausible = plausible
        self.relock_radius_frac = relock_radius_frac
        self.cls: str | None = None
        self.state = LockState()
        self._last_seen_t: float | None = None
        self._confirmed = False
        self._needs_hits = False

    def unseen_s(self, t: float) -> float:
        return math.inf if self._last_seen_t is None else t - self._last_seen_t

    def set_class(self, cls: str | None) -> None:
        if cls != self.cls:
            self.cls = cls
            self.release()

    def release(self) -> None:
        self.state = LockState(cls=self.cls)
        self._last_seen_t = None
        self._confirmed = False
        self._needs_hits = False

    def _hits_ok(self, tid: int | None) -> bool:
        tr = self.tracker.tracks.get(tid)
        if tr is None:
            return False
        recent = [f for f in tr.seen_frames if f > self.tracker.frame_index - self.confirm_window]
        return len(recent) >= self.confirm_hits

    def _is_confirmed(self, tid: int, det: Detection) -> bool:
        if self._hits_ok(tid):
            self._needs_hits = False
            return True
        if self._needs_hits:  # after a re-lock, one confident frame is not enough
            return False
        return det.conf >= self.confirm_conf and (self.plausible is None or self.plausible(det))

    def _pick(self, cands: list[Detection]) -> Detection:
        repeated = [d for d in cands if self._hits_ok(d.track_id)]
        return self.select(repeated or cands)

    @staticmethod
    def _jumped(a: Detection, b: Detection) -> bool:
        size = max(a.w, a.h, 1.0)
        return math.hypot(b.cx - a.cx, b.cy - a.cy) > 2.0 * size

    def update(self, detections: list[Detection], t: float, image_size: tuple[int, int] = (960, 720)) -> LockState:
        """Returns a snapshot (a copy): later updates do not change states already handed out."""
        return replace(self._update(detections, t, image_size))

    def _update(self, detections: list[Detection], t: float, image_size: tuple[int, int]) -> LockState:
        s = self.state
        if self.cls is None:
            return s
        cands = [d for d in detections if d.cls == self.cls]
        det = None
        if s.track_id is not None and not self._confirmed:
            # not committed yet: pick again every frame, so a one-frame false blob cannot block the real
            # target (tracks with repeated hits win over single sightings)
            if cands:
                det = self._pick(cands)
                if det.track_id != s.track_id:
                    s.track_id = det.track_id
        elif s.track_id is not None:
            det = next((d for d in cands if d.track_id == s.track_id), None)
            if det is not None and s.last_det is not None and self._jumped(s.last_det, det):
                self._confirmed, self._needs_hits = False, True  # our id glued onto something far away
            if det is None and cands and s.last_det is not None:
                tr = self.tracker.tracks.get(s.track_id)
                expired = tr is None
                brief = tr is None or len(tr.seen_frames) < self.confirm_hits
                repeated = [d for d in cands if self._hits_ok(d.track_id)]
                size = max(s.last_det.w, s.last_det.h, 1.0)
                diag = math.hypot(*image_size)

                def dist(d):
                    return math.hypot(d.cx - s.last_det.cx, d.cy - s.last_det.cy)

                # re-attach quickly only very close by; after a real loss, anywhere near where it was
                radius = self.relock_radius_frac * diag if s.lost else 3.0 * size
                near = [d for d in cands if dist(d) < radius]
                if brief and repeated:
                    # we had committed to something seen only briefly (a one-frame blob) while another
                    # candidate keeps being detected: take that one
                    det = self._pick(repeated)
                elif near or expired:
                    # back under a new track id near where we lost it, or our track is gone: take the
                    # nearest candidate, and confirm it again with REPEATED hits (not one confident frame)
                    det = min(near or cands, key=dist)
                    self._confirmed, self._needs_hits = False, True
                if det is not None:
                    s.track_id = det.track_id
        elif cands:
            det = self._pick(cands)
            s.track_id = det.track_id
            self._confirmed = False

        if det is not None:
            self._last_seen_t = t
            s.hits += 1
            self._confirmed = self._confirmed or self._is_confirmed(s.track_id, det)
            s.last_det = det
        s.det = det
        s.unseen_s = math.inf if self._last_seen_t is None else t - self._last_seen_t
        s.lost = s.unseen_s > self.lost_after_s or self._last_seen_t is None
        s.confirmed = self._confirmed and not s.lost
        if s.lost and self._last_seen_t is not None and s.unseen_s > 3 * self.lost_after_s:
            # gone for good: free the lock so any new candidate can be picked
            self.release()
            return self._update(detections, t, image_size) if cands else self.state
        return s
