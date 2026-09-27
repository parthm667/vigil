"""YOLO detection for the drone: people (with pose keypoints) and the team's blue water bottle.

    from yolo import person_detector, bottle_detector, draw

    people = person_detector()
    for d in people.detect(bgr):      # BGR uint8; djitellopy frames are RGB, so convert them first
        print(d.cls, d.conf, d.bbox)  # bbox = x1, y1, x2, y2 in pixels

See README.md in this folder.
"""

from .detector import MODELS_DIR, Detection, Detector, auto_device
from .draw import draw
from .presets import (BOTTLE, BOTTLE_HEIGHT_M, BOTTLE_PROMPTS, BOTTLE_WIDTH_M, DETECTORS, PERSON, bottle_detector,
                      person_detector)

__all__ = ["Detection", "Detector", "MODELS_DIR", "auto_device", "draw", "person_detector", "bottle_detector",
           "DETECTORS", "PERSON", "BOTTLE", "BOTTLE_PROMPTS", "BOTTLE_HEIGHT_M", "BOTTLE_WIDTH_M"]
