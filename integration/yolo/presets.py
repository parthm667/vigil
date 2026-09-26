"""The YOLO detector params the ReachGlass stack uses by default, as UltralyticsDetector keyword arguments.

Copied from reachglass/config.py (YOLO_WORLD_BOTTLE, PerceptionCfg.person_detector / context_detector and the
"yolo-world" entry of TARGET_PRESETS).

    UltralyticsDetector(**YOLO_WORLD_BOTTLE)   # target: the team's blue water bottle
    UltralyticsDetector(**PERSON_POSE)         # people + keypoints (follow, person guard)
    UltralyticsDetector(**CONTEXT_OBJECTS)     # furniture etc. for the exploration prior
"""

# How the target (the team's blue Hydro Flask: 24 cm tall with its black cap, 9 cm wide) is found.
# Measured on the team's photos placed at Tello-like distances (960 px input): YOLO-World finds it in 95-100 %
# of views at 1.3-4 m (60 % at 6 m), nothing else in the rooms scores even 0.05; "blue water bottle" alone
# still accepts a green or grey bottle, so the blue check on each box keeps only the team's bottle (>= 80 % of
# a box's middle is blue for it, <= 24 % for other colours). 13 ms/frame on a Mac GPU, ~100 ms on a CPU.
YOLO_WORLD_BOTTLE = {
    "weights": "yolov8s-worldv2.pt",
    "prompts": ["blue water bottle", "hydro flask water bottle"],
    "rename": {"blue water bottle": "bottle", "hydro flask water bottle": "bottle"},
    "classes": ["bottle"],
    "conf": 0.2,
    "imgsz": 960,  # the stream's own width: far (small) bottles keep their pixels
    "agnostic_nms": True,  # two prompts, one bottle: one box
    "require_color": {"hsv_ranges": [[95, 60, 40, 130, 255, 255]], "min_frac": 0.3},
}

# The rest of the "yolo-world" target preset: a YOLO box covers the whole bottle, cap included (the colour
# blob sees only the blue body, 0.19 m), and YOLO-World scores lower than the colour blob, so a single
# plausible detection confirms the target lock at 0.3 (0.55 for the colour blob).
BOTTLE_HEIGHT_M = 0.24
BOTTLE_WIDTH_M = 0.09
CONFIRM_CONF = 0.3

PERSON_POSE = {"weights": "yolo11n-pose.pt", "classes": ["person"], "conf": 0.4, "imgsz": 640}

CONTEXT_OBJECTS = {
    "weights": "yolo11n.pt", "classes": ["chair", "couch", "bed", "dining table", "tv", "refrigerator",
                                         "sink", "oven", "microwave", "potted plant", "laptop", "bench",
                                         "toilet", "suitcase", "backpack"],
    "conf": 0.35, "imgsz": 640}
