"""YOLO detection for the ReachGlass stack, as a standalone module: it imports nothing from reachglass/.

    from yolo import UltralyticsDetector, YOLO_WORLD_BOTTLE, PERSON_POSE

    bottle = UltralyticsDetector(**YOLO_WORLD_BOTTLE)   # the team's blue bottle (the stack's target default)
    people = UltralyticsDetector(**PERSON_POSE)         # people + 17 COCO keypoints
    for d in bottle.detect(bgr_image):
        print(d.cls, d.conf, d.bbox)

Layout (each file is a copy of the reachglass code named in its docstring):
    detector.py   UltralyticsDetector, weight resolution, device choice
    presets.py    the detector params the stack uses by default
    download.py   fetch + test-run the weights (python -m yolo.download)
    types.py      Detection + the Detector interface
    color.py      HSV helpers behind require_color

ultralytics is imported only when a detector is built, so `import yolo` works without it.
"""

from .detector import MODELS_DIR, UltralyticsDetector, auto_device, resolve_weights
from .presets import (BOTTLE_HEIGHT_M, BOTTLE_WIDTH_M, CONFIRM_CONF, CONTEXT_OBJECTS, PERSON_POSE,
                      YOLO_WORLD_BOTTLE)
from .types import Detection, Detector

__all__ = ["UltralyticsDetector", "Detection", "Detector", "MODELS_DIR", "auto_device", "resolve_weights",
           "YOLO_WORLD_BOTTLE", "PERSON_POSE", "CONTEXT_OBJECTS", "BOTTLE_HEIGHT_M", "BOTTLE_WIDTH_M",
           "CONFIRM_CONF"]
