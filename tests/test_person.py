"""Step 5: person estimator (range, facing) against synthetic ground truth and a real pose model."""

import math

import cv2
import numpy as np
import pytest

from reachglass.geometry import tello_camera
from reachglass.person import PersonEstimator
from reachglass.sim.person_model import CameraPose, SimPerson

CAM = tello_camera()


def ang_err(a, b):
    return abs((a - b + 180) % 360 - 180)


def obs(dist, phi, alt, height=1.75, pitch=0.0, rng=None, px_noise=0.0, est_alt=None, est_pitch=None, bearing_deg=0.0):
    b = math.radians(bearing_deg)
    p = SimPerson(dist * math.cos(b), dist * math.sin(b), phi + bearing_deg, height)
    d = p.detect(CameraPose(0, 0, alt, 0, pitch), CAM, rng, px_noise)
    if d is None:
        return None, p
    o = PersonEstimator().estimate(d, CAM, alt if est_alt is None else est_alt, pitch if est_pitch is None else est_pitch)
    return o, p


# ------------------------------------------------------------------ synthetic model, no noise
def test_sim_model_conventions():
    p = SimPerson(2.0, 0.0, heading_deg=0.0)  # standing 2 m ahead, facing away from the camera at the origin
    assert p.facing_rel_deg(0, 0) == pytest.approx(0)
    assert SimPerson(2.0, 0.0, 90).facing_rel_deg(0, 0) == pytest.approx(90)  # faces +y = image right
    d = p.detect(CameraPose(0, 0, 1.2, 0), CAM)
    assert d.cx == pytest.approx(CAM.cx, abs=1)
    k = d.keypoints
    assert k[6, 0] > k[5, 0]  # back view: right shoulder on the image right
    assert k[0, 2] == 0.0  # face invisible from behind
    front = SimPerson(2.0, 0.0, 180).detect(CameraPose(0, 0, 1.2, 0), CAM).keypoints
    assert front[5, 0] > front[6, 0] and front[0, 2] > 0.9  # front view: left shoulder on the right, face visible


@pytest.mark.parametrize("alt,dist", [(2.0, 1.8), (2.0, 3.0), (1.2, 3.0), (1.2, 4.5)])
def test_exact_range_and_facing_all_directions(alt, dist):
    for phi in range(-180, 180, 15):
        o, _ = obs(dist, phi, alt)
        assert o is not None and o.range_m == pytest.approx(dist, rel=0.03), (phi, o.range_src)
        # straight front/back views: facing comes from acos(shoulder width / expected width), which is
        # ill-conditioned there (a 1 % range error, e.g. the feet's near edge lowering the box, gives ~8 deg)
        tol = 10 if abs(math.cos(math.radians(phi))) > 0.97 else 3
        assert o.facing_deg is not None and ang_err(o.facing_deg, phi) < tol, phi


def test_bearing_of_off_axis_person():
    for b in (-25, -10, 10, 25):
        o, _ = obs(3.0, 0, 1.2, bearing_deg=b)
        assert o.bearing_deg == pytest.approx(b, abs=1.0)
        assert o.range_m == pytest.approx(3.0, rel=0.05)


def test_follow_position_uses_head_elevation_when_feet_are_cut_off():
    o, _ = obs(1.8, 0, 2.0)
    assert o.det.bbox[3] == pytest.approx(CAM.height)  # feet out of frame
    assert "head_elev" in o.range_src and "full_height" not in o.range_src


def test_close_person_with_shoulder_out_of_frame_has_no_facing():
    o, _ = obs(1.2, -120, 2.0)
    assert o.facing_deg is None and o.range_m is not None


def test_no_keypoints_still_gives_range():
    o, p = obs(3.0, 0, 1.2)
    o.det.keypoints = None
    o2 = PersonEstimator().estimate(o.det, CAM, 1.2)
    assert o2.facing_deg is None and o2.range_m == pytest.approx(3.0, rel=0.05)


# ------------------------------------------------------------------ realistic noise
def noisy_stats(alt, dist, h_sd, n=300, seed=0):
    rng = np.random.default_rng(seed)
    rerr, ferr, sign_ok = [], [], []
    for _ in range(n):
        phi = rng.uniform(-180, 180)
        pitch = rng.normal(0, 3)
        o, _ = obs(dist, phi, alt, 1.75 + rng.normal(0, h_sd), pitch, rng, 2.0,
                   est_alt=alt + rng.normal(0, 0.05), est_pitch=pitch + rng.normal(0, 1))
        if o is None or o.range_m is None:
            continue
        rerr.append(abs(o.range_m - dist) / dist)
        if o.facing_deg is not None:
            ferr.append(ang_err(o.facing_deg, phi))
            if 40 < abs(phi) < 140:
                sign_ok.append(np.sign(o.facing_deg) == np.sign(phi))
    return np.median(rerr), np.percentile(rerr, 90), np.median(ferr), np.mean(sign_ok)


def test_noisy_follow_position_accuracy():
    r50, r90, f50, sign = noisy_stats(2.0, 1.8, h_sd=0.05)
    assert r50 < 0.2 and r90 < 0.5 and f50 < 15 and sign > 0.93


def test_noisy_exploration_height_accuracy():
    # with the real ~43 deg vertical FOV the feet of a person 3 m away are just out of frame at 1.2 m
    r50, r90, f50, sign = noisy_stats(1.2, 3.0, h_sd=0.05)
    assert r50 < 0.08 and r90 < 0.22 and f50 < 8 and sign > 0.95
    r50, r90, _, _ = noisy_stats(1.2, 4.5, h_sd=0.05)  # whole body in frame again: tight
    assert r50 < 0.06 and r90 < 0.15


def test_range_lo_is_never_above_fused_range():
    rng = np.random.default_rng(2)
    for _ in range(100):
        o, _ = obs(1.8, rng.uniform(-180, 180), 2.0, 1.75, 0, rng, 2.0)
        if o is not None and o.range_m is not None:
            assert o.range_lo_m <= o.range_m + 1e-9


# ------------------------------------------------------------------ real pose model
@pytest.mark.yolo
def test_real_pose_model_people_facing_camera():
    from ultralytics.utils import ASSETS

    from reachglass.detect import DETECTORS
    from reachglass.config import ComponentSpec

    img = cv2.imread(str(ASSETS / "zidane.jpg"))
    cam = tello_camera().for_frame(img.shape[1], img.shape[0])
    det = DETECTORS.build(ComponentSpec("yolo", {"weights": "yolo11n-pose.pt", "classes": ["person"]}))
    people = [d for d in det.detect(img) if d.area > 0.05 * img.shape[0] * img.shape[1]]
    assert len(people) >= 2
    for d in people:
        o = PersonEstimator().estimate(d, cam, None)
        assert o.facing_deg is not None and abs(o.facing_deg) > 90, o.facing_deg  # both face the camera
    # Zidane (left) leans and points toward image-right: facing must have a + sign
    zid = min(people, key=lambda d: d.cx)
    assert PersonEstimator().estimate(zid, cam, None).facing_deg > 0
