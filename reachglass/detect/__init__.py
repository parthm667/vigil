"""Detectors, built by name from config: DETECTORS.build(cfg.perception.target_detector)."""

from ..registry import Registry
from .base import Detector, NullDetector
from .color_blob import ColorBlobDetector

DETECTORS = Registry("detector")
DETECTORS.register("color_blob")(ColorBlobDetector)
DETECTORS.register("null")(NullDetector)


@DETECTORS.register("yolo")
def _yolo(**params):
    from .yolo import UltralyticsDetector  # heavy import only when used

    return UltralyticsDetector(**params)


__all__ = ["DETECTORS", "Detector", "NullDetector", "ColorBlobDetector"]
