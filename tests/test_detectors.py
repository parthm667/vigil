"""Step 3: detectors."""

import cv2
import numpy as np
import pytest

from reachglass.config import ComponentSpec, load_config
from reachglass.detect import DETECTORS, ColorBlobDetector, NullDetector


def scene(w=960, h=720, seed=0):
    """Grey textured room-ish background with some clutter that is not the dummy's blue."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 120, np.uint8)
    img = cv2.add(img, rng.integers(0, 40, (h, w, 3), dtype=np.uint8))
    cv2.rectangle(img, (50, 400), (300, 700), (150, 130, 120), -1)  # greyish blue (mesh chair, window glare)
    cv2.rectangle(img, (600, 100), (900, 250), (40, 160, 40), -1)  # green box
    cv2.rectangle(img, (400, 500), (560, 560), (60, 200, 230), -1)  # yellow-ish (orange-free) box
    return img


BLUE = (167, 92, 52)  # the dummy water bottle's blue (BGR), measured from a photo
RED = (30, 30, 210)


def test_color_blob_finds_the_blue_bottle_with_accurate_box():
    img = scene()
    cv2.rectangle(img, (500, 300), (540, 388), BLUE, -1)  # a 40 x 88 px "bottle"
    dets = ColorBlobDetector().detect(img)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "bottle" and 0.5 < d.conf < 1.0
    assert d.bbox == pytest.approx((500, 300, 541, 389), abs=3)


def test_color_blob_rejects_clutter_specks_and_lines():
    det = ColorBlobDetector()
    assert det.detect(scene()) == []  # no dummy blue at all
    img = scene()
    cv2.rectangle(img, (100, 100), (103, 103), BLUE, -1)  # speck below min area
    cv2.rectangle(img, (200, 50), (205, 250), BLUE, -1)  # thin line: aspect > 6
    assert det.detect(img) == []


def test_color_blob_hue_wraparound_and_ordering():
    img = scene()
    cv2.rectangle(img, (100, 100), (160, 220), (60, 20, 220), -1)  # red-magenta, hue ~175 (wraps)
    cv2.rectangle(img, (700, 400), (716, 430), RED, -1)  # small far red object
    dets = ColorBlobDetector(hsv_ranges=[[172, 150, 140, 8, 255, 255]]).detect(img)  # a red dummy
    assert len(dets) == 2
    assert dets[0].bbox[0] == pytest.approx(100, abs=3)  # bigger blob first
    assert dets[0].conf > dets[1].conf


def test_color_blob_custom_label_and_range():
    img = scene()
    cv2.rectangle(img, (300, 300), (360, 400), (230, 60, 30), -1)  # vivid blue
    det = ColorBlobDetector(label="cup", hsv_ranges=[[100, 150, 100, 130, 255, 255]])
    dets = det.detect(img)
    assert det.classes == ["cup"] and len(dets) == 1 and dets[0].cls == "cup"
    with pytest.raises(ValueError):
        ColorBlobDetector(hsv_ranges=[[1, 2, 3]])


def test_color_blob_no_false_positives_on_noise():
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 255, (720, 960, 3), dtype=np.uint8)  # random colours incl. blue pixels
    assert ColorBlobDetector().detect(noise) == []


@pytest.mark.yolo
def test_color_blob_in_real_photos_and_dim_light(assets):
    det = ColorBlobDetector()
    # a blue bus IS blue (large blue surfaces show up; the shape/size checks and on-site tuning handle that),
    # but the two men's (darker, less saturated) jeans must not
    for d in det.detect(assets["bus.jpg"]):
        x1, _, x2, y2 = d.bbox
        assert not (x2 > 60 and x1 < 420 and y2 > 700), d.bbox
    assert len(det.detect(assets["zidane.jpg"])) <= 2
    img = scene()
    cv2.rectangle(img, (400, 300), (440, 390), (95, 52, 30), -1)  # the bottle's shadow side / dim room (V=95)
    assert len(det.detect(img)) == 1


def test_registry_builds_configured_target_detector():
    cfg = load_config()
    det = DETECTORS.build(cfg.perception.target_detector)
    assert isinstance(det, ColorBlobDetector) and det.classes == ["bottle"]
    assert isinstance(DETECTORS.build("null"), NullDetector) and DETECTORS.build("null").detect(scene()) == []
    assert DETECTORS.build(ComponentSpec("", {})) is None


# ------------------------------------------------------------------ YOLO (real models)
@pytest.fixture(scope="module")
def assets():
    from ultralytics.utils import ASSETS

    return {n: cv2.imread(str(ASSETS / n)) for n in ("bus.jpg", "zidane.jpg")}


@pytest.mark.yolo
def test_yolo_person_detection_and_class_filter(assets):
    det = DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n.pt", "classes": ["person"], "conf": 0.4}))
    assert det.classes == ["person"]
    dets = det.detect(assets["bus.jpg"])
    assert len(dets) >= 3 and all(d.cls == "person" for d in dets)
    assert all(dets[i].conf >= dets[i + 1].conf for i in range(len(dets) - 1))
    h, w = assets["bus.jpg"].shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
    all_cls = DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n.pt"}))
    assert "bottle" in all_cls.classes and len(all_cls.classes) == 80
    assert "bus" in {d.cls for d in all_cls.detect(assets["bus.jpg"])}


@pytest.mark.yolo
def test_yolo_pose_keypoints(assets):
    det = DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n-pose.pt", "classes": ["person"]}))
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
    det = DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n.pt", "classes": ["flask"], "rename": {"bottle": "flask"}}))
    assert det.classes == ["flask"]
    with pytest.raises(ValueError, match="no class"):
        DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n.pt", "classes": ["unicorn"]}))


def test_hsv_picker_range_covers_every_click_including_red_wraparound():
    from reachglass.tools.hsv_picker import range_from_samples

    def covered(r, h, s, v):
        h0, s0, v0, h1, _, _ = r
        in_h = (h0 <= h <= h1) if h0 <= h1 else (h >= h0 or h <= h1)
        return in_h and s >= s0 and v >= v0

    daylight, warm_dim = (2, 200, 200), (176, 170, 120)  # the same red dummy under two lights
    r = range_from_samples([daylight, warm_dim])
    assert r[0] > r[3]  # red: written wrap-around style
    assert covered(r, *daylight) and covered(r, *warm_dim)
    assert not covered(r, 30, 200, 200) and not covered(r, 150, 200, 200)  # yellow / magenta stay out
    ColorBlobDetector(hsv_ranges=[r])  # valid for the detector
    g = range_from_samples([(60, 200, 200), (70, 150, 150)])
    assert g[0] < g[3] and covered(g, 65, 160, 160) and not covered(g, 0, 200, 200)
