"""Step 8: simulator (drone kinematics, camera, oracle detectors) and perception on rendered frames."""

import math

import numpy as np
import pytest

from reachglass.config import load_config
from reachglass.detect import ColorBlobDetector
from reachglass.perception import Perception
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.scenario import Sim
from reachglass.sim.world import Box, World, demo_world
from reachglass.types import Pose2D


def empty_world():
    return World(size_x=8.0, size_y=8.0, boxes=[])


def fly(sim, dt=0.05):
    sim.drone.takeoff()
    assert run_until_idle(sim, 8.0, dt) == "ok" and sim.drone.flying


def run_until_idle(sim, max_s=15.0, dt=0.05):
    t0 = sim.t
    while sim.drone.busy() and sim.t - t0 < max_s:
        sim.step(dt)
    assert not sim.drone.busy(), "command did not finish"
    return sim.drone.last_result()


def exact(**kw):
    return SimDroneParams(move_err_frac=0.0, move_drift_frac=0.0, rot_err_deg=0.0, hover_drift=0.0, **kw)


# ------------------------------------------------------------------ kinematics
def test_takeoff_then_hover_height():
    sim = Sim(empty_world(), (2.0, 2.0), params=exact())
    assert not sim.drone.flying and sim.drone.last_result() == "ok"
    sim.drone.takeoff()
    assert sim.drone.busy() and sim.drone.last_result() is None
    assert run_until_idle(sim) == "ok"
    tel = sim.drone.telemetry()
    assert sim.drone.flying and tel.height_m == pytest.approx(0.8, abs=0.02) and tel.tof_m == pytest.approx(0.8, abs=0.02)


def test_discrete_move_and_rotation_conventions():
    sim = Sim(empty_world(), (2.0, 2.0), params=exact())
    fly(sim)
    sim.drone.move("forward", 100)
    assert run_until_idle(sim) == "ok"
    assert sim.drone.pos[:2] == pytest.approx([3.0, 2.0], abs=0.01)
    sim.drone.rotate(90)  # clockwise: now facing +y (right)
    assert run_until_idle(sim) == "ok"
    assert sim.drone.heading == pytest.approx(90, abs=0.5) and sim.drone.telemetry().yaw_deg == 90
    sim.drone.move("forward", 50)
    run_until_idle(sim)
    assert sim.drone.pos[:2] == pytest.approx([3.0, 2.5], abs=0.01)
    sim.drone.move("right", 50)  # heading 90: right is -x
    run_until_idle(sim)
    assert sim.drone.pos[:2] == pytest.approx([2.5, 2.5], abs=0.01)
    sim.drone.move("up", 40)
    run_until_idle(sim)
    assert sim.drone.pos[2] == pytest.approx(1.2, abs=0.01)
    sim.drone.rotate(-180)
    run_until_idle(sim)
    assert sim.drone.telemetry().yaw_deg == -90


def test_move_duration_is_realistic_and_rc_ignored_while_busy():
    sim = Sim(empty_world(), (2.0, 2.0), params=exact(move_speed=0.5))
    fly(sim)
    t0 = sim.t
    sim.drone.move("forward", 100)
    sim.step(0.2)
    sim.drone.rc(0, 0, 0, 100)  # must be ignored: no yaw during the move
    run_until_idle(sim)
    assert 2.0 < sim.t - t0 < 3.5  # 1 m at 0.5 m/s with accel/brake
    assert sim.drone.heading == pytest.approx(0, abs=0.01)


def test_odometry_noise_is_small_and_random():
    ends = []
    for seed in range(20):
        sim = Sim(empty_world(), (2.0, 2.0), params=SimDroneParams(seed=seed, hover_drift=0.0))
        fly(sim)
        sim.drone.move("forward", 200)
        run_until_idle(sim)
        ends.append(sim.drone.pos[:2].copy())
    ends = np.array(ends)
    assert np.abs(ends[:, 0] - 4.0).max() < 0.25 and ends[:, 0].std() > 0.01  # ~3 % of 2 m, not zero


def test_rc_dead_time_lag_and_speed():
    sim = Sim(empty_world(), (2.0, 4.0), params=exact())
    fly(sim)
    sim.drone.rc(0, 50, 0, 0)
    sim.step(0.05)
    assert np.linalg.norm(sim.drone.vel) < 1e-6  # still inside the dead time
    for _ in range(40):  # 2 s, re-sent like a real control loop
        sim.drone.rc(0, 50, 0, 0)
        sim.step(0.05)
    v = sim.drone.vel[0]
    assert 0.4 < v <= 0.5 and 0.6 < sim.drone.pos[0] - 2.0 < 1.0
    sim.drone.rc(0, 0, 0, 0)
    sim.run(1.5)
    assert abs(sim.drone.vel[0]) < 0.05


def test_rc_yaw_is_clockwise_and_pitch_nose_down_when_accelerating():
    sim = Sim(empty_world(), (4.0, 4.0), params=exact())
    fly(sim)
    pitches = []
    for _ in range(20):
        sim.drone.rc(0, 60, 0, 40)
        sim.step(0.05)
        pitches.append(sim.drone.pitch_deg())
    assert sim.drone.heading > 10  # yaw right = clockwise = heading grows
    assert min(pitches) < -1.0  # nose down while speeding up (sim convention: nose-up positive)


def test_collision_aborts_move_and_is_counted():
    sim = Sim(empty_world(), (7.0, 4.0), params=exact())
    fly(sim)
    sim.drone.move("forward", 300)  # the wall is 1 m ahead
    assert run_until_idle(sim) == "error: collision"
    assert sim.drone.collisions == 1 and sim.drone.pos[0] < 8.0 - 0.12


def test_stop_cancels_a_move_and_emergency_drops():
    sim = Sim(empty_world(), (2.0, 4.0), params=exact())
    fly(sim)
    sim.drone.move("forward", 300)
    sim.run(1.0)
    sim.drone.stop()
    assert not sim.drone.busy() and sim.drone.last_result() == "ok"
    x = sim.drone.pos[0]
    sim.run(2.0)
    assert 0.05 < sim.drone.pos[0] - x < 0.35 and 2.2 < x < 4.0  # brakes with its lag, like a real drone
    x = sim.drone.pos[0]
    sim.run(1.0)
    assert sim.drone.pos[0] == pytest.approx(x, abs=0.02)  # then holds
    sim.drone.emergency()
    assert not sim.drone.flying and sim.drone.pos[2] == 0.0


def test_auto_land_after_15_s_without_commands():
    sim = Sim(empty_world(), (2.0, 4.0), params=exact())
    fly(sim)
    sim.run(10.0)
    assert sim.drone.flying
    sim.drone.rc(0, 0, 0, 0)  # a keep-alive resets the watchdog
    sim.run(14.0)
    assert sim.drone.flying
    sim.run(4.0)
    assert not sim.drone.flying and any("auto-land" in e for _, e in sim.drone.events)


def test_tof_sees_tables_below():
    w = World(boxes=[Box((3.0, 1.0, 0.0), (4.0, 3.0, 0.75), cls="dining table")])
    sim = Sim(w, (2.0, 2.0), params=exact())
    fly(sim)
    sim.drone.move("up", 40)
    run_until_idle(sim)
    assert sim.drone.telemetry().tof_m == pytest.approx(1.2, abs=0.02)
    sim.drone.move("forward", 150)  # now above the table
    run_until_idle(sim)
    tel = sim.drone.telemetry()
    assert tel.tof_m == pytest.approx(0.45, abs=0.02) and tel.height_m == pytest.approx(1.2, abs=0.02)


# ------------------------------------------------------------------ camera + oracles
def test_camera_delay_and_frames():
    sim = Sim(demo_world(), (0.5, 1.5), params=exact())
    fly(sim)
    sim.run(1.0)
    f = sim.camera.read()
    pose, t_cap, _ = sim.camera.meta_for(f.image)
    assert f.image.shape == (360, 480, 3) and not f.image.flags.writeable
    assert sim.t - t_cap == pytest.approx(0.15, abs=1.0 / 15 + 1e-6)
    assert sim.camera.read() is f  # same frame until a new one is due
    sim.step(0.1)
    assert sim.camera.read().seq == f.seq + 1


def test_oracle_person_visible_from_behind_and_hidden_by_furniture():
    w = demo_world()
    w.person.x, w.person.y, w.person.heading_deg = 3.0, 1.5, 0.0
    sim = Sim(w, (1.2, 1.5), params=exact())
    fly(sim)
    sim.drone.move("up", 100)
    run_until_idle(sim)
    sim.step(0.3)
    d = sim.person_detector.detect(sim.camera.read().image)
    assert len(d) == 1 and d[0].keypoints is not None
    # a tall cupboard between drone and person hides them
    w.boxes.append(Box((2.0, 1.0, 0.0), (2.3, 2.0, 2.4), (90, 90, 90), ""))
    sim.step(0.3)
    assert sim.person_detector.detect(sim.camera.read().image) == []
    # a foreign image (not from this camera) -> no oracle answer
    assert sim.person_detector.detect(np.zeros((360, 480, 3), np.uint8)) == []


def test_takeoff_blocked_by_furniture_fails_cleanly():
    sim = Sim(demo_world(), (3.5, 2.5), params=exact())  # right next to the pouf
    sim.drone.takeoff()
    assert run_until_idle(sim) == "error: collision" and not sim.drone.flying


def test_oracle_context_sees_the_table():
    sim = Sim(demo_world(), (3.0, 2.2), drone_heading=math.degrees(math.atan2(5.0 - 2.2, 6.1 - 3.0)), params=exact())
    fly(sim)
    sim.drone.move("up", 40)
    run_until_idle(sim)
    sim.step(0.3)
    classes = {d.cls for d in sim.context_detector.detect(sim.camera.read().image)}
    assert "dining table" in classes and "person" not in classes


# ------------------------------------------------------------------ perception on rendered frames
def locate_dummy(start, tx=6.2, ty=5.0):
    w = demo_world()
    w.person = None
    heading = math.degrees(math.atan2(ty - start[1], tx - start[0])) + 10
    sim = Sim(w, start, drone_heading=heading, params=exact())
    fly(sim)
    sim.drone.move("up", 40)
    run_until_idle(sim)
    per = Perception(load_config(), None, ColorBlobDetector(), None, sim.cam)
    per.set_mode("approach")
    per.set_target("bottle")
    res = None
    for _ in range(10):
        sim.step(1.0 / 15)
        res = per.update(sim.camera.read(), sim.drone.telemetry())
    pose = Pose2D(sim.drone.pos[0], sim.drone.pos[1], sim.drone.heading)
    return res, pose


@pytest.mark.parametrize("start", [(3.0, 1.2), (4.2, 2.8), (1.0, 1.0), (4.0, 4.0), (5.0, 2.5), (3.8, 4.2), (5.2, 3.9)])
def test_perception_locates_rendered_dummy_in_world_coordinates(start):
    """Real colour-blob detector on rendered frames: world position within 10 % of the distance (1.5-6.6 m)."""
    tx, ty = 6.2, 5.0
    res, pose = locate_dummy(start)
    assert res.target is not None and res.target.confirmed, start
    x, y = pose.point_at(res.target.range_m, res.target.bearing_deg)
    true = math.hypot(tx - start[0], ty - start[1])
    assert math.hypot(x - tx, y - ty) < 0.10 * true, (start, x, y, res.target.range_m)


def test_far_partly_hidden_dummy_keeps_exact_bearing():
    """Known limitation: from 4.5 m with its bottom hidden behind a chair, the dummy looks shorter, so the
    range reads long; the bearing stays exact (the approach is closed-loop and re-measures up close)."""
    res, pose = locate_dummy((2.0, 3.5))
    assert res.target is not None
    assert res.target.bearing_deg == pytest.approx(pose.bearing_to(6.2, 5.0), abs=0.5)
    assert res.target.range_m > math.hypot(6.2 - 2.0, 5.0 - 3.5)
