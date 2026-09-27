"""Built-in person detector for the standalone runtime: FrameRing frames -> YOLO -> tracker -> det.

    python -m flyfollow.runtime.detector --selftest     # once, WITH internet: downloads the weights, times one image
    python -m flyfollow.runtime.detector                # started by launch (--detector yolo, the default for --dry/--send)
    options: --weights yolo11n-pose.pt  --conf 0.4  --imgsz 640  --device auto|cpu|mps|cuda:0  --threads N

Ported from the ReachGlass stack (jerkgt13 2302928): reachglass/detect/yolo.py (UltralyticsDetector, with the same
person detector settings: yolo11n-pose, class person, conf 0.4, imgsz 640) and reachglass/track/tracker.py
(SimpleTracker). Only "person" boxes are published; the controller takes the top of the box as the head it steers on
(geometry.head_box_from_person). Weights live in <repo>/models/. The Tello Wi-Fi has no internet, so run --selftest
once beforehand.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from flyfollow.interfaces import REPO_ROOT
from flyfollow.runtime.messages import msg

NAME = "detector"
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_WEIGHTS = "yolo11n-pose.pt"
LOG_EVERY_S = 10.0


def resolve_weights(weights: str) -> Path:
    """Bare names go to <repo>/models/ (Ultralytics downloads them there on first use)."""
    p = Path(weights)
    if p.is_absolute() or p.exists() or len(p.parts) > 1:
        return p
    return MODELS_DIR / p


def auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


@dataclass(frozen=True)
class Det:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels
    track_id: int | None = None

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def to_msg(self) -> dict:
        return {"cls": self.cls, "conf": round(self.conf, 3), "bbox": [round(v, 1) for v in self.bbox], "track_id": self.track_id}


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ------------------------------------------------------------------------------------------------ detector
class YoloDetector:
    """Ultralytics YOLO (detection or pose weights), filtered to `classes`, most confident first."""

    def __init__(self, weights: str = DEFAULT_WEIGHTS, classes: tuple[str, ...] = ("person",), conf: float = 0.4,
                 imgsz: int = 640, device: str = "auto", iou: float = 0.5, max_det: int = 20, warmup: bool = True):
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.weights.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(self.weights))
        names = {int(i): n for i, n in self.model.names.items()}
        unknown = [c for c in classes if c not in names.values()]
        if unknown:
            raise ValueError(f"{weights} has no class {unknown}")
        self._names = names
        self._ids = [i for i, n in names.items() if n in set(classes)]
        self.imgsz = int(imgsz)
        self.device = auto_device() if device == "auto" else device
        self._kw = dict(conf=conf, iou=iou, imgsz=self.imgsz, classes=self._ids, device=self.device, max_det=max_det, verbose=False)
        if warmup:  # pay model load / first-inference latency now, not in flight
            self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8), **self._kw)

    def detect(self, bgr: np.ndarray) -> list[Det]:
        r = self.model.predict(bgr, **self._kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        out = [Det(self._names[int(cls[i])], float(confs[i]), tuple(float(v) for v in xyxy[i])) for i in range(len(xyxy))]
        out.sort(key=lambda d: -d.conf)
        return out


# ------------------------------------------------------------------------------------------------ tracker
@dataclass
class Track:
    id: int
    cls: str
    bbox: tuple[float, float, float, float]
    last_t: float
    last_frame: int


class SimpleTracker:
    """Same track_id for the same object across frames (the controller's lock follows it).

    Greedy association per class, best pairs first: IoU >= iou_match, or, between consecutive frames, centre
    distance below center_match_frac of the image diagonal with a similar size (a fast yaw can shift a box by more
    than its own width). Tracks unseen for max_age_s are dropped.
    """

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

    def _score(self, tr: Track, d: Det, diag: float) -> float:
        """Association score in (0, 2]; 0 = no match."""
        o = iou(tr.bbox, d.bbox)
        if o >= self.iou_match:
            return 1.0 + o
        if tr.last_frame != self.frame_index - 1:
            return 0.0  # the centre fallback is for fast motion between consecutive frames only
        tcx, tcy = 0.5 * (tr.bbox[0] + tr.bbox[2]), 0.5 * (tr.bbox[1] + tr.bbox[3])
        dist = math.hypot(d.cx - tcx, d.cy - tcy) / diag
        if dist < self.center_match_frac:
            ta = (tr.bbox[2] - tr.bbox[0]) * (tr.bbox[3] - tr.bbox[1])
            ratio = min(ta, d.area) / max(ta, d.area, 1e-9)  # a small far box must not steal a big near one
            if ratio > 0.35:
                return 1.0 - dist / self.center_match_frac
        return 0.0

    def update(self, dets: list[Det], t: float, image_size: tuple[int, int]) -> list[Det]:
        self.frame_index += 1
        diag = math.hypot(*image_size)
        for tid in [k for k, tr in self.tracks.items() if t - tr.last_t > self.max_age_s]:
            del self.tracks[tid]
        pairs = sorted(((s, di, tid) for di, d in enumerate(dets) for tid, tr in self.tracks.items()
                        if tr.cls == d.cls and (s := self._score(tr, d, diag)) > 0), reverse=True)
        used_d, used_t, assign = set(), set(), {}
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            assign[di] = tid
        out = []
        for di, d in enumerate(dets):
            tid = assign.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, d.cls, d.bbox, t, self.frame_index)
            else:
                tr = self.tracks[tid]
                tr.bbox, tr.last_t, tr.last_frame = d.bbox, t, self.frame_index
            out.append(replace(d, track_id=tid))
        return out


# ------------------------------------------------------------------------------------------------ process
def det_message(dets: list[Det], frame_id: int, t_decoded: float, img_w: int, img_h: int, imgsz: int, latency_s: float) -> dict:
    return msg("det", frame_id=int(frame_id), t_decoded=float(t_decoded), src="yolo", img_w=int(img_w), img_h=int(img_h),
               imgsz=imgsz, latency_s=round(latency_s, 4), dets=[d.to_msg() for d in dets])


def run(detector: YoloDetector, pub, stop, ring=None, tracker: SimpleTracker | None = None) -> int:
    """Newest frame in the ring -> det on the bus, until stop is set. Skips frames it could not keep up with."""
    from flyfollow.runtime.bus import FrameRing
    from flyfollow.runtime.node import log

    tracker = tracker or SimpleTracker()
    last_id = None
    n = n_people = 0
    busy = 0.0
    t_log = time.monotonic()
    while not stop.is_set():
        if ring is None:
            try:
                ring = FrameRing.attach()
                log(NAME, "attached to the frame ring")
            except Exception:  # noqa: BLE001 (producer not up yet)
                stop.wait(0.2)
                continue
        try:
            got = ring.latest()
        except Exception:  # noqa: BLE001 (producer restarted: reattach)
            ring = None
            continue
        if got is None or got[1] == last_id:
            stop.wait(0.003)
            continue
        img, frame_id, t_dec = got
        last_id = frame_id
        t0 = time.perf_counter()
        dets = detector.detect(np.ascontiguousarray(img[..., ::-1]))  # the ring holds RGB; YOLO takes BGR arrays
        h, w = img.shape[:2]
        dets = tracker.update(dets, t_dec, (w, h))
        dt = time.perf_counter() - t0
        pub.publish(det_message(dets, frame_id, t_dec, w, h, detector.imgsz, dt))
        n += 1
        n_people += len(dets)
        busy += dt
        if time.monotonic() - t_log >= LOG_EVERY_S:
            span = time.monotonic() - t_log
            log(NAME, f"{n / span:.1f} frames/s, {1000 * busy / max(n, 1):.0f} ms per frame, "
                      f"{n_people / max(n, 1):.1f} people per frame")
            n = n_people = 0
            busy = 0.0
            t_log = time.monotonic()
    return 0


def selftest(detector: YoloDetector) -> int:
    import cv2
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))  # 4 people
    img = cv2.resize(img, (960, 720))
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        dets = detector.detect(img)
        times.append(1000 * (time.perf_counter() - t0))
    ms = float(np.median(times))
    print(f"weights {detector.weights} ({detector.device}): {len(dets)} people on bus.jpg, {ms:.0f} ms per frame")
    ok = len(dets) >= 3 and ms < 100
    print("SELFTEST OK: the detector runs offline now" if ok else
          "SELFTEST FAIL: expected >= 3 people and < 100 ms per frame (try --device cpu, or a faster laptop)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="built-in YOLO person detector: FrameRing -> det")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="bare name = <repo>/models/NAME (downloaded if missing)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda:0")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="torch CPU threads (default half the cores, so the brain keeps its 50 ms tick)")
    ap.add_argument("--selftest", action="store_true", help="download the weights, detect people in one image, exit")
    a = ap.parse_args(argv)
    try:
        import torch

        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    from flyfollow.runtime.node import log, stop_event

    t0 = time.perf_counter()
    det = YoloDetector(a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device)
    log(NAME, f"{det.weights.name} on {det.device}, conf {a.conf}, imgsz {a.imgsz}, loaded in {time.perf_counter() - t0:.1f} s")
    if a.selftest:
        return selftest(det)
    from flyfollow.runtime.bus import Publisher

    stop = stop_event()
    pub = Publisher(NAME)
    try:
        return run(det, pub, stop)
    finally:
        pub.close()


if __name__ == "__main__":
    sys.exit(main())
