"""Built-in person detector (flyfollow.runtime.detector) and the fly's yaw shaping in the control loop."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from flyfollow.runtime.bus import InProcBus, Publisher, Subscriber
from flyfollow.runtime.controller_runner import YawShaper
from flyfollow.runtime.detector import DEFAULT_WEIGHTS, Det, SimpleTracker, resolve_weights, run
from flyfollow.runtime.launch import Launcher, build_parser, module_exists
from flyfollow.runtime.messages import validate


def person(x: float, w: float = 120.0, h: float = 400.0, conf: float = 0.9) -> Det:
    return Det("person", conf, (x - w / 2, 200.0, x + w / 2, 200.0 + h))


def test_tracker_keeps_ids_and_expires():
    tr = SimpleTracker(max_age_s=1.0)
    a, b = tr.update([person(300), person(700)], 0.0, (960, 720))
    ids = {a.track_id, b.track_id}
    assert len(ids) == 2
    for k in range(1, 20):  # both walk right 8 px per frame: same ids
        out = tr.update([person(300 + 8 * k), person(700 + 8 * k)], 0.05 * k, (960, 720))
        assert [d.track_id for d in out] == [a.track_id, b.track_id]
    fast = tr.update([person(300 + 8 * 19 + 130)], 1.0, (960, 720))  # one fast yaw frame: centre fallback
    assert fast[0].track_id == a.track_id
    late = tr.update([person(450)], 3.0, (960, 720))  # unseen > max_age_s: a new track
    assert late[0].track_id not in ids


class FakeRing:
    def __init__(self, n: int):
        self.n, self.k = n, 0

    def latest(self):
        if self.k >= self.n:
            return (np.zeros((720, 960, 3), np.uint8), self.k - 1, 100.0 + 0.033 * (self.k - 1))  # same frame again
        self.k += 1
        return np.zeros((720, 960, 3), np.uint8), self.k - 1, 100.0 + 0.033 * (self.k - 1)


class FakeDetector:
    imgsz = 640

    def __init__(self):
        self.calls = 0
        self.last_shape = None

    def detect(self, bgr):
        self.calls += 1
        self.last_shape = bgr.shape
        return [person(480 + 5 * self.calls)]


def test_run_publishes_det_per_new_frame():
    bus = InProcBus()
    sub = Subscriber(["det"], bus=bus)
    det = FakeDetector()
    stop = threading.Event()
    th = threading.Thread(target=run, args=(det, Publisher("detector", bus=bus), stop, FakeRing(10)))
    th.start()
    got = []
    for _ in range(200):
        m = sub.recv(0.05)
        if m is not None:
            got.append(m)
        if len(got) == 10:
            break
    stop.set()
    th.join(2)
    assert det.calls == 10 and det.last_shape == (720, 960, 3)  # each frame once, never the repeated one
    assert [m["frame_id"] for m in got] == list(range(10))
    for m in got:
        assert not validate(m)
        assert m["img_w"] == 960 and m["img_h"] == 720 and m["src"] == "yolo"
        (d,) = m["dets"]
        assert d["cls"] == "person" and d["track_id"] == got[0]["dets"][0]["track_id"]


def test_yaw_shaper_deadband_and_hysteresis():
    s = YawShaper(deadband=4, hysteresis=3)
    assert s(3.0) == 0.0 and s(-3.9) == 0.0  # inside the deadband
    assert s(10.0) == 6.0  # 10 - 4
    assert s(11.0) == 6.0 and s(12.9) == 6.0  # within 3 of the sent stick: hold
    assert s(14.0) == 10.0
    assert s(-20.0) == -16.0
    assert s(2.0) == 0.0  # back inside the deadband: zero at once
    s.reset()
    assert s.y == 0.0
    assert YawShaper()(7.4) == 7.0  # no shaping: only rounding to whole stick units


def test_launch_defaults_to_yolo_on_the_drone(tmp_path):
    root = ["--no-operator", "--record-root", str(tmp_path)]
    lz = Launcher(build_parser().parse_args(["--dry", *root]))
    assert lz.detector == "yolo"
    assert "detector" in [c.name for c in lz.plan()]
    sim = Launcher(build_parser().parse_args(["--sim", *root]))
    assert sim.detector == "synthetic" and "detector" not in [c.name for c in sim.plan()]


@pytest.mark.skipif(not module_exists("ultralytics") or not resolve_weights(DEFAULT_WEIGHTS).exists(),
                    reason="needs ultralytics and models/yolo11n-pose.pt (python -m flyfollow.runtime.detector --selftest)")
def test_yolo_finds_people_in_bus_image():
    import cv2
    from ultralytics.utils import ASSETS

    from flyfollow.runtime.detector import YoloDetector

    img = cv2.resize(cv2.imread(str(ASSETS / "bus.jpg")), (960, 720))
    dets = YoloDetector(device="cpu").detect(img)
    assert len(dets) >= 3 and all(d.cls == "person" and 0.4 <= d.conf <= 1 for d in dets)
    assert all(0 <= d.bbox[0] < d.bbox[2] <= 960 and 0 <= d.bbox[1] < d.bbox[3] <= 720 for d in dets)
