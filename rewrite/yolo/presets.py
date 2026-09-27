"""The two detectors the project uses, with the settings tuned in jerkgt13.

    people = person_detector()   # yolo11n-pose: boxes + 17 COCO keypoints
    bottle = bottle_detector()   # the team's blue Hydro Flask: YOLO-World prompts + a blue check on each box
    DETECTORS["person"]()        # the same, by name

Keyword arguments override a preset, e.g. bottle_detector(conf=0.3).
"""

from __future__ import annotations

from .detector import Detector

PERSON = {"weights": "yolo11n-pose.pt", "classes": ["person"], "conf": 0.4, "imgsz": 640}

# The team's blue Hydro Flask (0.24 m tall with its black cap, 0.09 m wide). YOLO-World finds it from these
# prompts (95-100 % of views at 1.3-4 m on the team's photos, 60 % at 6 m). The prompts also accept green or
# grey bottles, so the blue check keeps only boxes whose middle is at least 30 % blue: >= 80 % for the team's
# bottle, <= 24 % for other colours. The prompts are baked into bottle-world.pt by `python -m yolo.prepare`;
# re-run it with --force after changing them.
BOTTLE_PROMPTS = ["blue water bottle", "hydro flask water bottle"]
BOTTLE = {
    "weights": "bottle-world.pt",
    "rename": {p: "bottle" for p in BOTTLE_PROMPTS},
    "conf": 0.2,
    "imgsz": 960,  # the Tello frame's own width, so a small far bottle keeps its pixels
    "agnostic_nms": True,  # two prompts, one bottle: one box
    "require_color": {"hsv_ranges": [[95, 60, 40, 130, 255, 255]], "min_frac": 0.3},
}
BOTTLE_HEIGHT_M = 0.24  # physical size of the whole box YOLO draws (cap included), for range from box size
BOTTLE_WIDTH_M = 0.09


def person_detector(**overrides) -> Detector:
    return Detector(**{**PERSON, **overrides})


def bottle_detector(**overrides) -> Detector:
    return Detector(**{**BOTTLE, **overrides})


DETECTORS = {"person": person_detector, "bottle": bottle_detector}
