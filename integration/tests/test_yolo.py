"""The YOLO module on its own: copies of the YOLO tests in tests/test_detectors.py (built directly instead of
through reachglass's DETECTORS registry), plus checks that the module stands alone."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from yolo import (CONTEXT_OBJECTS, PERSON_POSE, YOLO_WORLD_BOTTLE, Detection, Detector, UltralyticsDetector,
                  resolve_weights)
from yolo.color import hsv_mask, parse_hsv_ranges

INTEGRATION = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ standalone (no ultralytics needed)
def test_imports_nothing_from_reachglass():
    code = ("import sys, yolo, yolo.download; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('reachglass', 'ultralytics')); "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=INTEGRATION, check=True)


def test_detector_interface_and_detection_geometry():
    assert issubclass(UltralyticsDetector, Detector) and UltralyticsDetector.name == "yolo"
    d = Detection("bottle", 0.9, (10.0, 20.0, 40.0, 100.0))
    assert (d.cx, d.cy, d.w, d.h, d.area) == (25.0, 60.0, 30.0, 80.0, 2400.0)


def test_bare_weight_names_resolve_into_integration_models():
    assert Path(resolve_weights("yolo11n.pt")) == INTEGRATION / "models" / "yolo11n.pt"
    assert resolve_weights("models/bottle.pt") == str(Path("models/bottle.pt"))  # a path is kept as given


def test_blue_check_ranges():
    blue = parse_hsv_ranges(YOLO_WORLD_BOTTLE["require_color"]["hsv_ranges"])
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :10] = (200, 60, 20)  # BGR blue
    img[:, 10:] = (20, 60, 200)  # BGR red
    m = hsv_mask(img, blue)
    assert m[:, :10].all() and not m[:, 10:].any()
    assert len(parse_hsv_ranges([[170, 100, 100, 10, 255, 255]])) == 2  # red wraps around: split in two
    with pytest.raises(ValueError):
        parse_hsv_ranges([[1, 2, 3]])


# ------------------------------------------------------------------ YOLO (real models)
@pytest.fixture(scope="module")
def assets():
    from ultralytics.utils import ASSETS

    return {n: cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")}


@pytest.mark.yolo
def test_yolo_person_detection_and_class_filter(assets):
    det = UltralyticsDetector(weights="yolo11n.pt", classes=["person"], conf=0.4)
    assert det.classes == ["person"]
    dets = det.detect(assets["bus.jpg"])
    assert len(dets) >= 3 and all(d.cls == "person" for d in dets)
    assert all(dets[i].conf >= dets[i + 1].conf for i in range(len(dets) - 1))
    h, w = assets["bus.jpg"].shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    all_cls = UltralyticsDetector(weights="yolo11n.pt")
    assert "bottle" in all_cls.classes and len(all_cls.classes) == 80
    assert "bus" in {d.cls for d in all_cls.detect(assets["bus.jpg"])}


@pytest.mark.yolo
def test_yolo_pose_keypoints(assets):
    det = UltralyticsDetector(weights="yolo11n-pose.pt", classes=["person"])
    dets = det.detect(assets["zidane.jpg"])
    assert len(dets) >= 2
    for d in dets:
        assert d.keypoints is not None and d.keypoints.shape == (17, 3)
        # shoulders (5, 6) of a person facing the camera: left shoulder appears on the image right
        ls, rs = d.keypoints[5], d.keypoints[6]
        if ls[2] > 0.5 and rs[2] > 0.5:
            assert ls[0] > rs[0]


@pytest.mark.yolo
def test_yolo_rename_and_unknown_class():
    det = UltralyticsDetector(weights="yolo11n.pt", classes=["flask"], rename={"bottle": "flask"})
    assert det.classes == ["flask"]
    with pytest.raises(ValueError, match="no class"):
        UltralyticsDetector(weights="yolo11n.pt", classes=["unicorn"])


@pytest.mark.yolo
def test_stack_default_presets_build(assets):
    """The person and context presets load with the vocabulary the stack expects (the target preset is
    checked on the team's photos below)."""
    person = UltralyticsDetector(**PERSON_POSE)
    assert person.classes == ["person"] and person.is_pose
    assert any(d.keypoints is not None for d in person.detect(assets["zidane.jpg"]))
    context = UltralyticsDetector(**CONTEXT_OBJECTS)
    assert sorted(context.classes) == sorted(CONTEXT_OBJECTS["classes"]) and "person" not in context.classes


@pytest.mark.yolo
def test_yolo_world_finds_the_team_bottle_and_only_a_blue_one():
    """The default target detector on the team's own photos (960x720 like the Tello): found with a confident
    box; the same scene with the bottle turned red is rejected by the blue check."""
    det = UltralyticsDetector(**YOLO_WORLD_BOTTLE)
    photos = sorted((Path(__file__).parent / "assets" / "bottle").glob("*.jpg"))
    assert photos
    for f in photos:
        img = cv2.imread(str(f))
        found = det.detect(img)
        assert len(found) == 1 and found[0].cls == "bottle" and found[0].conf >= 0.5, (f.name, found)
        x1, y1, x2, y2 = found[0].bbox
        assert 1.8 < (y2 - y1) / (x2 - x1) < 3.3, f.name  # the whole bottle, cap included (24 x 9 cm)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) - 110) % 180  # blue -> red, everything else shifts too
        assert det.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)) == [], f.name
