"""Step 1: types, geometry, config."""

import math

import numpy as np
import pytest

from reachglass.config import ComponentSpec, load_config, to_dict
from reachglass.geometry import CameraModel, tello_camera
from reachglass.types import Detection, Pose2D, Telemetry, iou, wrap_deg


# ------------------------------------------------------------------ types
@pytest.mark.parametrize("a,want", [(0, 0), (180, 180), (-180, 180), (190, -170), (-190, 170), (540, 180), (359, -1), (-721, -1)])
def test_wrap_deg(a, want):
    assert wrap_deg(a) == pytest.approx(want)


def test_pose_conventions_clockwise_right_positive():
    p = Pose2D(0, 0, 0)
    # straight ahead is +x, a bearing to the right is +y
    assert p.point_at(2.0, 0) == pytest.approx((2.0, 0.0))
    x, y = p.point_at(1.0, 90)
    assert x == pytest.approx(0.0, abs=1e-9) and y == pytest.approx(1.0)
    # turning clockwise 90 then moving forward goes to +y
    q = p.moved(turn_deg=90).moved(forward_m=1.0)
    assert (q.x, q.y, q.heading_deg) == pytest.approx((0.0, 1.0, 90.0), abs=1e-9)
    # moving right at heading 0 goes to +y; at heading 90 moving right goes to -x
    assert p.moved(right_m=1.0).y == pytest.approx(1.0)
    r = Pose2D(0, 0, 90).moved(right_m=1.0)
    assert (r.x, r.y) == pytest.approx((-1.0, 0.0), abs=1e-9)


def test_pose_bearing_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = Pose2D(*rng.uniform(-5, 5, 2), rng.uniform(-180, 180))
        d, b = rng.uniform(0.3, 6), rng.uniform(-179, 179)
        x, y = p.point_at(d, b)
        assert p.bearing_to(x, y) == pytest.approx(b, abs=1e-6)
        assert p.distance_to(x, y) == pytest.approx(d)


def test_detection_props_and_iou():
    d = Detection("bottle", 0.9, (10, 20, 30, 60))
    assert (d.cx, d.cy, d.w, d.h, d.area) == (20, 40, 20, 40, 800)
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_telemetry_altitude_prefers_valid_tof():
    assert Telemetry(0, height_m=1.5, tof_m=1.4).altitude_m() == 1.4
    assert Telemetry(0, height_m=1.5, tof_m=0.05).altitude_m() == 1.5  # tof out of range
    assert Telemetry(0, height_m=1.5, tof_m=6.0).altitude_m() == 1.5
    assert Telemetry(0, height_m=1.5).altitude_m() == 1.5
    assert Telemetry(0).altitude_m() is None


# ------------------------------------------------------------------ geometry
def test_tello_camera_intrinsics():
    cam = tello_camera()
    # the video stream (calibrated f ~ 920 px at 960x720 -> ~55 x 43 deg), not the 82.6 deg stills spec
    assert cam.fx == 920.0 and (cam.cx, cam.cy) == (480.0, 360.0)
    assert cam.hfov_deg == pytest.approx(55.1, abs=0.5)
    assert cam.vfov_deg == pytest.approx(42.7, abs=0.5)
    # stills-FOV construction still works: 82.6 deg diagonal over 1200 px -> f ~ 683 px
    assert CameraModel.from_fov(960, 720, dfov_deg=82.6).fx == pytest.approx(683, abs=2)
    assert CameraModel.from_fov(640, 480, hfov_deg=60).hfov_deg == pytest.approx(60)
    with pytest.raises(ValueError):
        CameraModel.from_fov(640, 480)


def test_bearing_elevation_roundtrip_and_signs():
    cam = tello_camera()
    assert cam.bearing_deg(cam.cx) == 0 and cam.bearing_deg(cam.width) > 0 and cam.bearing_deg(0) < 0
    assert cam.elevation_deg(0) > 0 and cam.elevation_deg(cam.height) < 0  # top of image is up
    for b in (-30, -5, 0, 12, 34):
        assert cam.bearing_deg(cam.x_of_bearing(b)) == pytest.approx(b)
    for e in (-25, 0, 20):
        assert cam.elevation_deg(cam.y_of_elevation(e, 3.0), 3.0) == pytest.approx(e)
    tilted = tello_camera(pitch_deg=-10)  # camera looks 10 deg down: centre row is 10 deg below horizon
    assert tilted.elevation_deg(tilted.cy) == pytest.approx(-10)


def test_range_from_size_matches_projection():
    cam = tello_camera()
    # a 0.22 m bottle 2 m ahead on the axis is fy*0.22/2 px tall
    px = cam.fy * 0.22 / 2.0
    assert cam.range_from_height(px, 0.22) == pytest.approx(2.0)
    # off-axis: object at bearing 30 deg, horizontal distance 2 m -> depth 2*cos30
    x = cam.x_of_bearing(30)
    px = cam.fy * 0.22 / (2.0 * math.cos(math.radians(30)))
    assert cam.range_from_height(px, 0.22, x) == pytest.approx(2.0)
    assert cam.range_from_width(cam.fx * 0.4 / 3.0, 0.4) == pytest.approx(3.0)
    assert cam.range_from_height(0.5, 0.22) is None


def test_ground_distance_and_elevation_range():
    cam = tello_camera()
    # floor point 3 m ahead, camera 1.2 m high -> depression atan(1.2/3)
    y = cam.y_of_elevation(-math.degrees(math.atan2(1.2, 3.0)))
    assert cam.ground_distance(y, 1.2) == pytest.approx(3.0)
    assert cam.ground_distance(cam.cy - 10, 1.2) is None  # above the horizon
    # a head 0.3 m below the camera seen 10 deg down
    assert cam.range_from_elevation(-10, 0.3) == pytest.approx(0.3 / math.tan(math.radians(10)))
    assert cam.range_from_elevation(-0.5, 0.3) is None  # degenerate


def test_range_to_height_exact_off_axis_and_pitched():
    """Cross-check against the simulator's independent 3-D projection (regression: off-axis rows used to
    give depth instead of horizontal distance, ~9 % short at 25 deg)."""
    from reachglass.sim.person_model import CameraPose, project

    cam = tello_camera()
    rng = np.random.default_rng(0)
    for _ in range(200):
        dist, bearing = rng.uniform(1.0, 6.0), rng.uniform(-30, 30)
        cam_h, pt_h, pitch = rng.uniform(0.8, 2.2), rng.uniform(0.0, 0.7), rng.uniform(-8, 8)
        b = math.radians(bearing)
        pt = np.array([[dist * math.cos(b), dist * math.sin(b), pt_h]])
        px, front = project(pt, CameraPose(0, 0, cam_h, 0, pitch), cam)
        x, y = px[0]
        if not (front[0] and 0 <= x < cam.width and 0 <= y < cam.height):
            continue
        r = cam.range_to_height(x, y, cam_h - pt_h, body_pitch_deg=pitch)
        if r is None:  # grazing angle
            continue
        assert r == pytest.approx(dist, rel=1e-6)
        f, rt, _ = cam.ray(x, y, pitch)
        assert math.degrees(math.atan2(rt, f)) == pytest.approx(bearing, abs=1e-6)  # level-frame bearing
        el = cam.elevation_deg(y, pitch, x)
        assert el == pytest.approx(math.degrees(math.atan2(pt_h - cam_h, dist)), abs=1e-6)
    assert cam.range_to_height(cam.cx, cam.cy - 50, 1.0) is None  # looking up cannot hit the floor


def test_scaled_camera_keeps_angles():
    cam = tello_camera()
    small = cam.scaled(480, 360)
    assert small.hfov_deg == pytest.approx(cam.hfov_deg)
    assert small.bearing_deg(small.x_of_bearing(20)) == pytest.approx(20)
    assert cam.for_frame(960, 720) is cam
    assert cam.for_frame(480, 360).width == 480


# ------------------------------------------------------------------ config
def test_config_defaults_and_overrides(tmp_path):
    cfg = load_config()
    assert cfg.follow.distance_m == 1.0 and cfg.follow.altitude_m == 2.0 and cfg.perception.target_detector.kind == "yolo"
    cfg = load_config(overrides={"follow": {"distance_m": 2, "altitude_m": 2.2}})
    assert cfg.follow.distance_m == 2.0 and isinstance(cfg.follow.distance_m, float)
    # dict fields merge
    cfg = load_config(overrides={"perception": {"object_heights_m": {"bottle": 0.26}}})
    assert cfg.perception.object_heights_m["bottle"] == 0.26 and "chair" in cfg.perception.object_heights_m
    # yaml file
    p = tmp_path / "room.yaml"
    p.write_text("explore:\n  scan_altitude_m: 1.1\n  max_vantage_points: 3\n")
    cfg = load_config(p)
    assert cfg.explore.scan_altitude_m == 1.1 and cfg.explore.max_vantage_points == 3
    assert to_dict(cfg)["explore"]["max_vantage_points"] == 3


def test_config_unknown_key_fails_loudly():
    with pytest.raises(KeyError, match="distnce_m"):
        load_config(overrides={"follow": {"distnce_m": 2}})
    with pytest.raises(KeyError):
        load_config(overrides={"nope": {}})


def test_component_params_replace_on_kind_change_merge_otherwise():
    cfg = load_config(overrides={"perception": {"target_detector": {"params": {"label": "cup"}}}}, target="color")
    p = cfg.perception.target_detector.params
    assert p["label"] == "cup" and "hsv_ranges" in p  # same kind: merged
    cfg = load_config(overrides={"perception": {"target_detector": {"kind": "yolo", "params": {"weights": "bottle.pt"}}}})
    assert cfg.perception.target_detector.params == {"weights": "bottle.pt"}  # new kind: replaced
    assert isinstance(cfg.perception.target_detector, ComponentSpec)


def test_configs_are_independent():
    a, b = load_config(), load_config()
    a.perception.object_heights_m["bottle"] = 9
    assert b.perception.object_heights_m["bottle"] == 0.24


def test_target_presets_switch_detector_sizes_and_confirmation_together():
    yw, col = load_config(), load_config(target="color")
    assert yw.perception.target_detector.kind == "yolo" and yw.perception.object_heights_m["bottle"] == 0.24
    assert col.perception.target_detector.kind == "color_blob" and col.perception.object_heights_m["bottle"] == 0.19
    assert "prompts" not in col.perception.target_detector.params  # a kind change replaces the params
    assert yw.tracking.confirm_conf < col.tracking.confirm_conf  # YOLO-World scores lower than the colour blob
    back = load_config(overrides={"perception": {"target_detector": {"kind": "color_blob"}}}, target="yolo-world")
    assert back.perception.target_detector.kind == "color_blob"  # explicit overrides still win
    with pytest.raises(ValueError, match="color"):
        load_config(target="colour")
