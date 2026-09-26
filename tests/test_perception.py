"""Step 7: perception pipeline (detectors + tracker + locks + estimators)."""

import math

import cv2
import numpy as np
import pytest

from reachglass.config import load_config
from reachglass.detect import ColorBlobDetector, Detector
from reachglass.geometry import tello_camera
from reachglass.perception import Perception
from reachglass.sim.person_model import CameraPose, SimPerson
from reachglass.types import Detection, Frame, Telemetry

CAM = tello_camera()


class Scripted(Detector):
    """Returns whatever `fn(call_index)` says; counts calls."""

    def __init__(self, classes, fn):
        self._classes, self.fn, self.calls = classes, fn, 0

    @property
    def classes(self):
        return self._classes

    def detect(self, image):
        self.calls += 1
        return self.fn(self.calls)


def frame(i, img=None):
    return Frame(img if img is not None else np.zeros((720, 960, 3), np.uint8), i * 0.1, i)


BOTTLE_H = load_config().perception.object_heights_m["bottle"]  # the dummy's configured size
BOTTLE_W = load_config().perception.object_widths_m["bottle"]
BLUE = (167, 92, 52)  # the dummy water bottle's blue (BGR)


def bottle_box(dist, bearing=0.0, height=BOTTLE_H, width=BOTTLE_W, hidden_bottom=0.0):
    """Box of the dummy bottle; `hidden_bottom` = fraction of its height hidden behind something."""
    depth = dist * math.cos(math.radians(bearing))
    ph, pw = CAM.fy * height / depth, CAM.fx * width / depth
    x = CAM.x_of_bearing(bearing)
    return Detection("bottle", 0.7, (x - pw / 2, 360 - ph / 2, x + pw / 2, 360 + ph / 2 - hidden_bottom * ph))


def test_mode_strides_decide_which_detectors_run():
    cfg = load_config()
    person = Scripted(["person"], lambda i: [])
    target = Scripted(["bottle"], lambda i: [])
    context = Scripted(["chair", "person"], lambda i: [])
    per = Perception(cfg, person, target, context)
    per.set_mode("follow")
    for i in range(10):
        per.update(frame(i))
    assert (person.calls, target.calls, context.calls) == (10, 0, 0)
    per.set_mode("search")
    for i in range(10, 20):
        r = per.update(frame(i))
    assert target.calls == 10 and context.calls == 5 and person.calls == 12
    assert set(r.ran) == {"person", "target", "context"}
    per.set_mode("idle")
    per.update(frame(21))
    assert target.calls == 10
    with pytest.raises(ValueError):
        per.set_mode("dance")


def test_vocabulary_is_target_plus_context_without_person():
    per = Perception(load_config(), Scripted(["person"], lambda i: []), Scripted(["bottle"], lambda i: []),
                     Scripted(["chair", "person", "dining table"], lambda i: []))
    assert per.vocabulary() == ["bottle", "chair", "dining table"]
    assert Perception(load_config()).vocabulary() == []


def test_target_obs_range_bearing_and_confirmation():
    cfg = load_config()
    per = Perception(cfg, None, Scripted(["bottle"], lambda i: [bottle_box(2.5, 12.0)]), None)
    per.set_mode("approach")
    per.set_target("bottle")
    rs = [per.update(frame(i)) for i in range(4)]
    t = rs[-1].target
    assert t is not None and t.range_m == pytest.approx(2.5, rel=0.02) and t.bearing_deg == pytest.approx(12.0, abs=0.2)
    assert [r.target.confirmed for r in rs] == [True, True, True, True]  # conf 0.7 >= 0.55 and plausible -> 1st frame
    assert rs[-1].target_unseen_s == 0.0


def test_partly_hidden_round_target_ranges_from_its_width():
    def run(box):
        per = Perception(load_config(), None, Scripted(["bottle"], lambda i: [box]), None)
        per.set_mode("approach")
        per.set_target("bottle")
        return per.update(frame(1)).target

    t = run(bottle_box(1.8, 5.0, hidden_bottom=0.4))  # 46 px wide: width is trustworthy
    assert t.range_src == "width_prior(occluded)" and t.range_m == pytest.approx(1.8, rel=0.03)  # height says 3.0
    assert run(bottle_box(1.8, 5.0)).range_src == "height_prior"
    far = run(bottle_box(4.0, 5.0, hidden_bottom=0.4))  # 21 px wide: too thin to trust, height is used
    assert far.range_src == "height_prior" and far.range_m > 4.0


def test_implausible_single_detection_needs_repeats():
    cfg = load_config()
    far = bottle_box(12.0)  # a "bottle" 12 m away: implausible in a room -> no single-frame confirmation
    per = Perception(cfg, None, Scripted(["bottle"], lambda i: [far]), None)
    per.set_mode("approach")
    per.set_target("bottle")
    rs = [per.update(frame(i)) for i in range(4)]
    assert [r.target.confirmed for r in rs] == [False, False, True, True]


def test_target_from_context_detector_when_asked_for_furniture():
    chair = Detection("chair", 0.8, (400, 300, 500, 500))
    per = Perception(load_config(), None, Scripted(["bottle"], lambda i: []), Scripted(["chair"], lambda i: [chair]))
    per.set_mode("search")
    per.set_target("chair")
    rs = [per.update(frame(i)) for i in range(4)]
    # the context detector owns "chair", so while searching for a chair it runs every frame
    assert [r.ran["context"] for r in rs] == [True, True, True, True]
    assert rs[3].target is not None and rs[3].target.cls == "chair" and rs[3].context == []
    per.set_target("bottle")  # back to a target the target detector owns: context detector at its stride
    rs = [per.update(frame(i)) for i in range(4, 8)]
    assert [r.ran["context"] for r in rs] == [True, False, True, False]


def test_person_obs_via_pose_model_output():
    p = SimPerson(1.8, 0.0, heading_deg=20.0)
    det = p.detect(CameraPose(0, 0, 2.0, 0), CAM)
    per = Perception(load_config(), Scripted(["person"], lambda i: [det]), None, None)
    per.set_mode("follow")
    rs = [per.update(frame(i), Telemetry(i * 0.1, tof_m=2.0)) for i in range(3)]
    o = rs[-1].person
    assert o is not None and o.range_m == pytest.approx(1.8, rel=0.05) and o.facing_deg == pytest.approx(20, abs=3)
    assert rs[-1].person_unseen_s == 0.0


def test_person_unseen_time_grows_when_they_leave():
    det = SimPerson(2.0, 0.0, 0.0).detect(CameraPose(0, 0, 2.0, 0), CAM)
    per = Perception(load_config(), Scripted(["person"], lambda i: [det] if i <= 3 else []), None, None)
    per.set_mode("follow")
    rs = [per.update(frame(i)) for i in range(10)]
    assert rs[2].person is not None and rs[5].person is None
    assert rs[9].person_unseen_s == pytest.approx(0.7, abs=1e-6)


def test_end_to_end_with_real_colour_blob_detector():
    # draw the blue dummy bottle 2 m away, 10 deg right, into a 960x720 image
    img = np.full((720, 960, 3), 110, np.uint8)
    b = bottle_box(2.0, 10.0)
    x1, y1, x2, y2 = (int(round(v)) for v in b.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), BLUE, -1)
    per = Perception(load_config(), None, ColorBlobDetector(), None)
    per.set_mode("approach")
    per.set_target("bottle")
    r = per.update(frame(1, img))
    assert r.target is not None and r.target.confirmed
    assert r.target.range_m == pytest.approx(2.0, rel=0.03) and r.target.bearing_deg == pytest.approx(10.0, abs=0.3)
    # the same image at half resolution gives the same geometry (camera model follows the frame size)
    small = cv2.resize(img, (480, 360), interpolation=cv2.INTER_NEAREST)
    per2 = Perception(load_config(), None, ColorBlobDetector(), None)
    per2.set_mode("approach")
    per2.set_target("bottle")
    r2 = per2.update(frame(1, small))
    assert r2.target.range_m == pytest.approx(2.0, rel=0.05) and r2.target.bearing_deg == pytest.approx(10.0, abs=0.5)


def test_red_shirt_inside_a_person_is_not_the_target():
    person = SimPerson(3.0, 0.0, 180.0).detect(CameraPose(0, 0, 1.2, 0), CAM)
    x1, y1, x2, y2 = person.bbox
    shirt = Detection("bottle", 0.9, (x1 + 10, y1 + 0.2 * (y2 - y1), x2 - 10, y1 + 0.5 * (y2 - y1)))
    real = bottle_box(2.0, -20.0)
    per = Perception(load_config(), Scripted(["person"], lambda i: [person]), Scripted(["bottle"], lambda i: [shirt, real]), None)
    per.set_mode("search")
    per.set_target("bottle")
    rs = [per.update(frame(i)) for i in range(6)]  # person detector runs only every 5th frame in search
    for r in rs:
        assert all(abs(t.bearing_deg + 20.0) < 1.0 for t in r.targets), [t.bearing_deg for t in r.targets]
    assert rs[-1].target is not None and rs[-1].target.bearing_deg == pytest.approx(-20.0, abs=0.5)


def test_from_config_builds_default_detectors():
    per = Perception.from_config(load_config(overrides={"perception": {"person_detector": {"kind": ""},
                                                                        "context_detector": {"kind": ""}}}))
    assert per.detectors["person"] is None and isinstance(per.detectors["target"], ColorBlobDetector)
    assert per.vocabulary() == ["bottle"]


@pytest.mark.yolo
def test_from_config_full_defaults_with_yolo():
    per = Perception.from_config(load_config())
    assert "bottle" in per.vocabulary() and "chair" in per.vocabulary() and "person" not in per.vocabulary()
