"""The YOLO detector: one class that wraps an Ultralytics model, plus what it returns.

detect() takes a BGR uint8 image (OpenCV channel order). djitellopy frames are RGB: convert them first with
cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).

Weights are never downloaded here, because in flight the laptop is on the Tello's Wi-Fi with no internet.
A missing model raises FileNotFoundError that tells you to run `python -m yolo.prepare` once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class Detection:
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the image given to detect()
    keypoints: np.ndarray | None = None  # (17, 3) x, y, confidence in COCO order; pose models only

    @property
    def cx(self) -> float:
        return 0.5 * (self.bbox[0] + self.bbox[2])

    @property
    def cy(self) -> float:
        return 0.5 * (self.bbox[1] + self.bbox[3])

    @property
    def w(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.w * self.h


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_hsv_ranges(ranges) -> list[tuple[int, int, int, int, int, int]]:
    """[[h_lo, s_lo, v_lo, h_hi, s_hi, v_hi], ...] in OpenCV HSV (H 0-180, S and V 0-255).
    A range with h_lo > h_hi wraps around 180 (red) and is split in two."""
    out = []
    for r in ranges:
        if len(r) != 6:
            raise ValueError(f"an HSV range needs 6 numbers [h_lo, s_lo, v_lo, h_hi, s_hi, v_hi], got {r}")
        h0, s0, v0, h1, s1, v1 = (int(v) for v in r)
        if not (0 <= h0 <= 180 and 0 <= h1 <= 180 and 0 <= s0 <= s1 <= 255 and 0 <= v0 <= v1 <= 255):
            raise ValueError(f"HSV range out of bounds (H 0-180, S and V 0-255, lo <= hi): {r}")
        if h0 > h1:
            out += [(h0, s0, v0, 180, s1, v1), (0, s0, v0, h1, s1, v1)]
        else:
            out.append((h0, s0, v0, h1, s1, v1))
    return out


class Detector:
    """An Ultralytics YOLO model with fixed settings. Call detect(bgr) once per frame, from one thread.

    weights        file name in yolo/models/, or an absolute path (a relative path is taken inside yolo/models/)
    classes        keep only these class names (after rename); None keeps every class the model has
    rename         model class name -> name reported in Detection.cls, e.g. {"blue water bottle": "bottle"}
    agnostic_nms   merge overlapping boxes across classes (several prompts that describe one object)
    require_color  {"hsv_ranges": [[...]], "min_frac": 0.3}: keep a box only if at least min_frac of its middle
                   (the central 60 % of its width, full height) is inside the HSV ranges
    """

    def __init__(self, weights: str, classes: list[str] | None = None, conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 640, max_det: int = 50, device: str = "auto", rename: dict[str, str] | None = None,
                 agnostic_nms: bool = False, require_color: dict | None = None, warmup: bool = True):
        path = Path(weights)
        if not path.is_absolute():
            path = MODELS_DIR / path
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing. Run once, with internet: cd rewrite && python -m yolo.prepare")

        from ultralytics import YOLO

        self.weights = path
        self.model = YOLO(str(path))
        rename = dict(rename or {})
        self.names = {int(i): rename.get(n, n) for i, n in self.model.names.items()}  # class id -> reported name
        ids = None
        if classes is not None:
            unknown = sorted(set(classes) - set(self.names.values()))
            if unknown:
                raise ValueError(f"{path.name} has no class {unknown}; its classes are {sorted(set(self.names.values()))}")
            ids = sorted(i for i, n in self.names.items() if n in classes)
        self.classes = sorted({self.names[i] for i in (ids if ids is not None else self.names)})
        self.device = auto_device() if device == "auto" else device
        self.is_pose = self.model.task == "pose"
        self._predict_args = dict(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, classes=ids, device=self.device,
                                  agnostic_nms=agnostic_nms, verbose=False)
        self._color = None
        if require_color is not None:
            self._color = (parse_hsv_ranges(require_color["hsv_ranges"]), float(require_color.get("min_frac", 0.3)))
        if warmup:  # model load and first-inference cost now, not on the first flight frame (960x720 = Tello frame)
            self.model.predict(np.zeros((720, 960, 3), np.uint8), **self._predict_args)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detections in pixel coordinates of `image` (BGR uint8, H x W x 3), most confident first."""
        r = self.model.predict(image, **self._predict_args)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        kps = r.keypoints.data.cpu().numpy() if self.is_pose and r.keypoints is not None else None
        out = []
        for i in range(len(xyxy)):
            box = tuple(float(v) for v in xyxy[i])
            if self._color is not None and self.color_frac(image, box) < self._color[1]:
                continue
            k = None if kps is None else kps[i].astype(np.float32)
            out.append(Detection(self.names[int(cls_ids[i])], float(confs[i]), box, k))
        out.sort(key=lambda d: -d.conf)
        return out

    def color_frac(self, image: np.ndarray, box: tuple[float, float, float, float]) -> float:
        """Fraction of the box's middle (central 60 % of its width, full height) inside the required colour.
        The outer 20 % on each side of a round bottle's box is mostly background."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        xa, xb = int(max(0, x1 + 0.2 * w)), int(min(image.shape[1], x2 - 0.2 * w))
        ya, yb = int(max(0, y1)), int(min(image.shape[0], y2))
        if xb <= xa or yb <= ya:
            return 0.0
        hsv = cv2.cvtColor(image[ya:yb, xa:xb], cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for h0, s0, v0, h1, s1, v1 in self._color[0]:
            mask |= cv2.inRange(hsv, (h0, s0, v0), (h1, s1, v1))
        return float(np.count_nonzero(mask)) / mask.size
