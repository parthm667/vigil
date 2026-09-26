"""Ultralytics YOLO detector (detection or pose models, .pt or exported OpenVINO/ONNX).

    UltralyticsDetector(weights="yolo11n-pose.pt", classes=["person"])   # people + 17 COCO keypoints
    UltralyticsDetector(weights="yolo11n.pt", classes=["bottle"])        # COCO bottle
    UltralyticsDetector(weights="models/bottle.pt")                      # tomorrow's trained model

Bare weight names are resolved into <repo>/models/ (downloaded there on first use).
`rename` maps model class names to the names the rest of the stack uses, e.g. {"water_bottle": "bottle"}.
`prompts` (YOLO-World weights only) sets the open-vocabulary text classes, e.g.
    UltralyticsDetector(weights="yolov8s-worldv2.pt", prompts=["blue water bottle"],
                        rename={"blue water bottle": "bottle"}, classes=["bottle"])
`agnostic_nms` merges overlapping boxes across classes (several prompts for the same object).
`require_color` = {"hsv_ranges": [[...]], "min_frac": 0.3} keeps only boxes whose middle is mostly that
colour: YOLO-World finds bottles, this keeps the team's BLUE one (a red or clear bottle is dropped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import Detection
from .base import Detector
from .color_blob import hsv_mask, parse_hsv_ranges

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def resolve_weights(weights: str) -> str:
    p = Path(weights)
    if p.is_absolute() or p.exists() or len(p.parts) > 1:
        return str(p)
    MODELS_DIR.mkdir(exist_ok=True)
    return str(MODELS_DIR / p)


def auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class UltralyticsDetector(Detector):
    name = "yolo"

    def __init__(self, weights: str = "yolo11n.pt", classes: list[str] | None = None, conf: float = 0.35,
                 imgsz: int = 640, device: str = "auto", rename: dict | None = None, iou: float = 0.5,
                 max_det: int = 50, half: bool = False, warmup: bool = True, prompts: list[str] | None = None,
                 agnostic_nms: bool = False, require_color: dict | None = None):
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.model = YOLO(self.weights)
        if prompts:
            if not hasattr(self.model, "set_classes"):
                raise ValueError(f"{weights} is not an open-vocabulary (YOLO-World) model: it cannot take prompts")
            self.model.set_classes(list(prompts))  # encodes the text once (CLIP), then runs like any YOLO
        self.rename = dict(rename or {})
        names = self.model.names  # {id: name}
        self._all = {int(i): self.rename.get(n, n) for i, n in names.items()}
        if classes:
            unknown = [c for c in classes if c not in self._all.values()]
            if unknown:
                raise ValueError(f"{weights} has no class {unknown}; it knows {sorted(self._all.values())[:20]}...")
            self._ids = [i for i, n in self._all.items() if n in set(classes)]
        else:
            self._ids = None
        self.conf, self.imgsz, self.iou, self.max_det, self.half = conf, imgsz, iou, max_det, half
        self.device = auto_device() if device == "auto" else device
        self.is_pose = getattr(self.model, "task", "") == "pose"
        self._kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, classes=self._ids, device=self.device, max_det=max_det, verbose=False)
        if half:  # passing half=False makes ultralytics 8.4 print a deprecation line on every call
            self._kwargs["half"] = True
        if agnostic_nms:
            self._kwargs["agnostic_nms"] = True
        self._color = None
        if require_color:
            self._color = (parse_hsv_ranges(require_color["hsv_ranges"]), float(require_color.get("min_frac", 0.3)))
        if warmup:  # pay model load / first-inference latency now, not in flight
            self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), **self._kwargs)

    @property
    def classes(self) -> list[str]:
        ids = self._ids if self._ids is not None else sorted(self._all)
        return [self._all[i] for i in ids]

    def detect(self, image: np.ndarray) -> list[Detection]:
        r = self.model.predict(image, **self._kwargs)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        kps = None
        if self.is_pose and r.keypoints is not None and r.keypoints.data is not None:
            kps = r.keypoints.data.cpu().numpy()  # (N, 17, 3): x, y, conf
        out = []
        for i in range(len(xyxy)):
            k = None if kps is None else kps[i].astype(np.float32)
            box = tuple(float(v) for v in xyxy[i])
            if self._color is not None and self.color_frac(image, box) < self._color[1]:
                continue
            out.append(Detection(self._all[int(cls[i])], float(confs[i]), box, k, source=self.name))
        out.sort(key=lambda d: -d.conf)
        return out

    def color_frac(self, image: np.ndarray, box) -> float:
        """Fraction of the box's middle 60 % (width) in the required colour: the sides of a round bottle's box
        are background."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        xa, xb = int(max(0, x1 + 0.2 * w)), int(min(image.shape[1], x2 - 0.2 * w))
        ya, yb = int(max(0, y1)), int(min(image.shape[0], y2))
        if xb <= xa or yb <= ya:
            return 0.0
        return float(np.count_nonzero(hsv_mask(image[ya:yb, xa:xb], self._color[0]))) / ((xb - xa) * (yb - ya))
