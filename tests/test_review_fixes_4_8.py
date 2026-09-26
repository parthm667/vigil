"""Regression tests for the confirmed findings of the steps 4-8 review."""

import math

import numpy as np
import pytest

from reachglass.config import load_config
from reachglass.detect import Detector
from reachglass.geometry import tello_camera
from reachglass.perception import Perception
from reachglass.person import PersonEstimator
from reachglass.query import KeywordQueryParser
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.person_model import CameraPose, SimPerson
from reachglass.sim.scenario import COCO_LR_PAIRS, Sim
from reachglass.sim.world import World
from reachglass.track import SimpleTracker, TargetLock
from reachglass.types import Detection, Frame, Telemetry

CAM = tello_camera()
SIZE = (960, 720)


class Scripted(Detector):
    def __init__(self, classes, fn):
        self._classes, self.fn, self.calls = classes, fn, 0

    @property
    def classes(self):
        return self._classes

    def detect(self, image):
        self.calls += 1
        return self.fn(self.calls)


def frame(i, dt=1 / 30):
    return Frame(np.zeros((720, 960, 3), np.uint8), i * dt, i)


def bottle(cx, conf=0.7, dist=3.0):
    ph, pw = CAM.fy * 0.22 / dist, CAM.fx * 0.07 / dist
    return Detection("bottle", conf, (cx - pw / 2, 360 - ph / 2, cx + pw / 2, 360 + ph / 2))


# ------------------------------------------------------------------ lock / tracker
def test_one_frame_false_blob_does_not_block_the_real_target():
    garbage = Detection("bottle", 0.9, (100, 330, 140, 400))  # a one-frame smear, plausible size
    per = Perception(load_config(), None, Scripted(["bottle"], lambda i: [garbage, bottle(712)] if i == 1 else [bottle(712)]), None)
    per.set_mode("search")
    per.set_target("bottle")
    rs = [per.update(frame(i)) for i in range(15)]  # 0.5 s at 30 fps
    assert rs[-1].target is not None and rs[-1].target.det.cx == pytest.approx(712, abs=1) and rs[-1].target.confirmed
    assert rs[-1].target_unseen_s < 0.1


def test_confirmed_lock_does_not_jump_via_centre_fallback():
    tr = SimpleTracker()
    lk = TargetLock(tr, confirm_hits=2)
    lk.set_class("bottle")
    for i in range(3):
        s = lk.update(tr.update([bottle(300)], i / 30, SIZE), i / 30, SIZE)
    assert s.confirmed
    s = lk.update(tr.update([], 3 / 30, SIZE), 3 / 30, SIZE)  # missed one frame...
    s = lk.update(tr.update([bottle(450)], 4 / 30, SIZE), 4 / 30, SIZE)  # ...and another bottle 150 px away
    assert s.det is None or not s.confirmed  # never silently a confirmed lock on the other object


def test_relock_after_a_gap_with_default_timing():
    cfg = load_config()
    seq = [[bottle(400)]] * 5 + [[]] * 36 + [[bottle(410)]] * 3  # 1.2 s gap at 30 fps
    per = Perception(cfg, None, Scripted(["bottle"], lambda i: seq[i - 1] if i <= len(seq) else []), None)
    per.set_mode("approach")
    per.set_target("bottle")
    rs = [per.update(frame(i)) for i in range(len(seq))]
    assert rs[-1].target is not None and rs[-1].target.det.cx == pytest.approx(410, abs=1)


def test_repeated_frame_is_not_a_new_sighting():
    per = Perception(load_config(), None, Scripted(["bottle"], lambda i: [bottle(400, conf=0.3)]), None)
    per.set_mode("approach")
    per.set_target("bottle")
    f = frame(1)
    r1 = per.update(f)
    r2 = per.update(f)
    r3 = per.update(f)
    assert r1 is r2 is r3 and per.detectors["target"].calls == 1


# ------------------------------------------------------------------ detector scheduling
def test_context_class_target_is_seen_in_approach_mode():
    chair = Detection("chair", 0.5, (400, 300, 520, 500))
    ctx = Scripted(["chair"], lambda i: [chair])
    per = Perception(load_config(), None, Scripted(["bottle"], lambda i: []), ctx)
    per.set_target("chair")
    per.set_mode("search")
    [per.update(frame(i)) for i in range(4)]
    per.set_mode("approach")
    rs = [per.update(frame(i)) for i in range(4, 20)]
    assert all(r.ran["context"] for r in rs)
    assert rs[-1].target is not None and rs[-1].target.confirmed


def test_red_shirt_suppressed_between_person_runs_and_in_approach():
    person = SimPerson(3.0, 0.0, 180.0).detect(CameraPose(0, 0, 1.2, 0), CAM)
    x1, y1, x2, y2 = person.bbox
    shirt = Detection("bottle", 0.9, (x1 + 20, y1 + 0.25 * (y2 - y1), x1 + 60, y1 + 0.55 * (y2 - y1)))
    for mode in ("search", "approach"):
        per = Perception(load_config(), Scripted(["person"], lambda i: [person]), Scripted(["bottle"], lambda i: [shirt]), None)
        per.set_mode(mode)
        per.set_target("bottle")
        rs = [per.update(frame(i)) for i in range(12)]
        assert all(r.target is None for r in rs), mode


def test_square_red_thing_is_not_confirmed_on_one_frame_but_a_bottle_is():
    square = Detection("bottle", 0.95, (400, 300, 460, 360))
    per = Perception(load_config(), None, Scripted(["bottle"], lambda i: [square]), None)
    per.set_mode("approach")
    per.set_target("bottle")
    assert not per.update(frame(1)).target.confirmed
    lying = Detection("bottle", 0.9, (400, 340, 400 + CAM.fx * 0.22 / 2.0, 340 + CAM.fy * 0.07 / 2.0))
    per2 = Perception(load_config(), None, Scripted(["bottle"], lambda i: [lying]), None)
    per2.set_mode("approach")
    per2.set_target("bottle")
    assert per2.update(frame(1)).target is not None


# ------------------------------------------------------------------ person estimator
def swapped(det):
    k = det.keypoints.copy()
    for a, b in COCO_LR_PAIRS:
        k[[a, b]] = k[[b, a]]
    det.keypoints = k
    return det


def test_left_right_swap_on_back_view_gives_low_confidence():
    d = SimPerson(1.8, 0.0, 0.0).detect(CameraPose(0, 0, 2.0, 0), CAM)
    o = PersonEstimator().estimate(swapped(d), CAM, 2.0)
    assert o.facing_conf <= 0.3  # "front view" without a face: do not trust it


def test_no_range_hint_caps_facing_confidence():
    d = SimPerson(3.0, 0.0, 60.0).detect(CameraPose(0, 0, 1.2, 0), CAM)
    o = PersonEstimator().estimate(d, CAM, altitude_m=None)
    assert o.facing_conf <= 0.3


def test_range_lo_is_a_lower_bound_when_the_wearer_height_is_configured():
    """With the wearer's height measured to +-5 cm and the default follow altitude, range_lo_m never
    overestimates by more than 10 % at 1.4-3 m, whatever way they face."""
    rng = np.random.default_rng(0)
    alt = load_config().follow.altitude_m
    for _ in range(300):
        true_h = 1.75 + rng.uniform(-0.05, 0.05)
        dist, phi = rng.uniform(1.4, 3.0), rng.uniform(-180, 180)
        d = SimPerson(dist, 0.0, phi, true_h).detect(CameraPose(0, 0, alt, 0), CAM, rng, 2.0)
        if d is None:
            continue
        o = PersonEstimator(person_height_m=1.75).estimate(d, CAM, alt)
        if o.range_lo_m is not None:
            assert o.range_lo_m <= dist * 1.10, (dist, phi, true_h, o.range_lo_m, o.range_src)


def test_close_follow_needs_the_wearer_height_to_2cm():
    """1 m behind at 2 m, only the head is in frame and range comes from its elevation, 0.25 m below the
    camera: every cm of height error is ~4 % of range. Measured to +-2 cm, range_lo_m still holds."""
    rng = np.random.default_rng(1)
    alt = load_config().follow.altitude_m
    checked = 0
    for _ in range(300):
        true_h = 1.75 + rng.uniform(-0.02, 0.02)
        dist, phi = rng.uniform(0.9, 1.5), rng.uniform(-180, 180)
        d = SimPerson(dist, 0.0, phi, true_h).detect(CameraPose(0, 0, alt, 0), CAM, rng, 2.0)
        if d is None:
            continue
        o = PersonEstimator(person_height_m=1.75).estimate(d, CAM, alt)
        if o.range_lo_m is not None:
            checked += 1
            assert o.range_lo_m <= dist * 1.10, (dist, phi, true_h, o.range_lo_m, o.range_src)
    assert checked > 200


def test_config_refuses_follow_altitude_at_head_height():
    with pytest.raises(ValueError, match="person_height_m"):
        load_config(overrides={"perception": {"person_height_m": 1.9}})  # 2.0 m follow altitude: too close
    assert load_config(overrides={"perception": {"person_height_m": 1.7}, "follow": {"altitude_m": 2.0}})
    with pytest.raises(ValueError, match="max_age_s"):
        load_config(overrides={"tracking": {"max_age_s": 0.5}})


def test_person_range_uses_floor_altitude_not_tof_over_a_table():
    d = SimPerson(2.0, 0.0, 0.0).detect(CameraPose(0, 0, 2.0, 0), CAM)
    per = Perception(load_config(), Scripted(["person"], lambda i: [d]), None, None)
    per.set_mode("follow")
    over_table = Telemetry(0.0, height_m=2.0, tof_m=1.25)  # ToF sees the table top
    assert over_table.floor_altitude_m() == 2.0 and over_table.altitude_m() == 1.25
    r = per.update(frame(1), over_table)
    assert r.person.range_m == pytest.approx(2.0, rel=0.1)


# ------------------------------------------------------------------ simulator contract
def flying_sim():
    sim = Sim(World(size_x=8, size_y=8, boxes=[]), (4.0, 4.0), params=SimDroneParams(hover_drift=0, rot_err_deg=0,
                                                                                     move_err_frac=0, move_drift_frac=0))
    sim.drone.takeoff()
    assert not sim.drone.flying  # only once takeoff completes, like the Tello adapter
    while sim.drone.busy():
        sim.step(0.05)
    assert sim.drone.flying
    return sim


def test_sim_long_rotations_turn_the_commanded_way():
    sim = flying_sim()
    h0 = sim.drone.heading
    sim.drone.rotate(270)
    t0 = sim.t
    while sim.drone.busy():
        sim.step(0.05)
    assert sim.t - t0 >= 270 / 75 - 0.1  # it really turned 270 deg
    assert (sim.drone.heading - h0) % 360 == pytest.approx(270, abs=1)
    sim.drone.rotate(-200)
    while sim.drone.busy():
        sim.step(0.05)
    assert (sim.drone.heading - h0) % 360 == pytest.approx(70, abs=1)


def test_sim_command_while_busy_raises_like_the_tello_adapter():
    sim = flying_sim()
    sim.drone.move("forward", 100)
    with pytest.raises(RuntimeError):
        sim.drone.rotate(45)


def test_sim_stop_overshoots_like_a_real_drone():
    sim = flying_sim()
    sim.drone.move("forward", 300)
    sim.run(1.5)
    x_stop = sim.drone.pos[0]
    sim.drone.stop()
    sim.run(2.0)
    assert 0.05 < sim.drone.pos[0] - x_stop < 0.35


# ------------------------------------------------------------------ parser
@pytest.mark.parametrize("text,target", [
    ("Look around for my water bottle", "bottle"),
    ("forget the cup, find my bottle", "bottle"),
    ("follow me to the chair", "chair"),
])
def test_parser_object_wins_over_describe_and_verb_order(text, target):
    q = KeywordQueryParser().parse(text, ["bottle", "cup", "chair"])
    assert (q.intent, q.target) == ("find", target), q.reason
