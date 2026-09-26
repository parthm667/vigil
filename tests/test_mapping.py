"""Step 10: odometry, grid, semantic memory, free space."""

import math

import pytest

from reachglass.mapping import FREESPACE, Grid, NullFreeSpace, Odometry, SemanticMemory
from reachglass.geometry import tello_camera
from reachglass.types import Pose2D, Telemetry


def tel(yaw):
    return Telemetry(0.0, yaw_deg=yaw)


# ------------------------------------------------------------------ odometry
def test_odometry_heading_from_yaw_relative_to_mission_start():
    o = Odometry()
    o.reset(tel(37.0))
    assert o.update(tel(37.0)).heading_deg == 0
    assert o.update(tel(127.0)).heading_deg == 90
    assert o.update(tel(-150.0)).heading_deg == pytest.approx(173.0)  # wraps


def test_odometry_moves_follow_heading_and_body_axes():
    o = Odometry()
    o.reset(tel(0.0))
    o.on_move_done("forward", 100, tel(0.0))
    assert (o.pose.x, o.pose.y) == pytest.approx((1.0, 0.0))
    o.on_move_done("right", 50, tel(90.0))  # heading 90: right is -x
    assert (o.pose.x, o.pose.y) == pytest.approx((0.5, 0.0), abs=1e-9)
    o.on_move_done("forward", 200, tel(90.0))
    assert (o.pose.x, o.pose.y) == pytest.approx((0.5, 2.0), abs=1e-9)
    o.on_move_done("up", 40, tel(90.0))  # vertical moves do not change x/y
    assert (o.pose.x, o.pose.y) == pytest.approx((0.5, 2.0), abs=1e-9)
    assert o.path[-1] == pytest.approx((0.5, 2.0))


def test_odometry_detects_inverted_yaw_sign():
    o = Odometry(yaw_sign=1)
    o.reset(tel(0.0))
    o.on_rotation_done(90, 0.0, -88.0)  # we said cw 90, the drone reports yaw -88
    assert o.yaw_sign == -1 and o.sign_checked
    assert o.update(tel(-88.0)).heading_deg == pytest.approx(88.0)
    o2 = Odometry()
    o2.reset(tel(0.0))
    o2.on_rotation_done(90, 0.0, 91.0)
    assert o2.yaw_sign == 1 and o2.sign_checked
    o2.on_rotation_done(10, 91.0, 99.0)  # tiny rotations are not used for the check


def test_yaw_sign_not_flipped_by_ambiguous_180_turns():
    o = Odometry()
    o.reset(tel(0.0))
    o.on_rotation_done(180, 0.0, -178.0)  # a correct 180 that reads -178 after wrapping: must NOT flip
    o.on_rotation_done(-170, -178.0, 10.0)
    assert o.yaw_sign == 1 and not o.sign_checked
    o.on_rotation_done(45, 10.0, 56.0)  # the first clean rotation resolves it
    assert o.yaw_sign == 1 and o.sign_checked
    o.on_rotation_done(90, 56.0, -30.0)  # once checked, never flipped again
    assert o.yaw_sign == 1


def test_odometry_without_yaw_uses_commanded_rotations():
    o = Odometry()
    o.reset(None)
    o.on_rotation_done(90, None, None)
    o.on_move_done("forward", 100, None)
    assert (o.pose.x, o.pose.y, o.pose.heading_deg) == pytest.approx((0.0, 1.0, 90.0), abs=1e-9)
    o.move_scale = 0.9
    o.on_move_done("forward", 100, None)
    assert o.pose.y == pytest.approx(1.9)


# ------------------------------------------------------------------ grid
def test_grid_view_marks_frustum_and_novelty_drops():
    g = Grid(size_m=16, cell_m=0.25)
    p = Pose2D(0, 0, 0)
    assert g.novelty(p, 0, 3.0) == 1.0 and g.novelty(p, 180, 3.0) == 1.0
    g.mark_view(p, hfov_deg=55, max_range_m=3.5)
    assert g.novelty(p, 0, 3.0) == 0.0 and g.novelty(p, 20, 3.0) == 0.0
    assert g.novelty(p, 180, 3.0) == 1.0 and g.novelty(p, 90, 3.0) == 1.0
    assert 0 < g.coverage() < 0.05


def test_grid_clearance_needs_free_evidence_and_stops_before_obstacles():
    g = Grid()
    p = Pose2D(0, 0, 0)
    assert g.clear_distance(p, 0, 2.0, unknown_ok_m=0.5) == pytest.approx(0.5, abs=0.13)  # nothing known
    g.mark_free_ray(0, 0, 0, 3.0)  # we saw something 3.5 m ahead
    assert g.clear_distance(p, 0, 2.0) == pytest.approx(2.0)
    g.mark_blocked(1.5, 0.0, radius_m=0.3)  # a chair at 1.5 m
    d = g.clear_distance(p, 0, 2.0, margin_m=0.35)
    assert 0.6 < d < 1.0
    assert g.clear_distance(p, 90, 2.0, unknown_ok_m=0.5) <= 0.5 + 0.13  # sideways still unknown


def test_grid_metric_view_marks_free_and_blocked():
    g = Grid()
    p = Pose2D(0, 0, 0)
    g.mark_view(p, 55, 3.5, free=[(-20, 3.0), (0, 1.5), (20, None)])
    assert g.clear_distance(p, 0, 3.0, unknown_ok_m=0.2) < 1.5  # wall at 1.5 m straight ahead
    assert g.clear_distance(p, -22, 3.0, unknown_ok_m=0.2) > 2.0


# ------------------------------------------------------------------ semantic memory
def test_memory_merges_nearby_sightings_and_weights_close_views():
    m = SemanticMemory()
    a = m.add("bottle", 4.0, 1.0, 0.6, range_m=4.5, t=1.0)  # far view, off by 0.4 m
    b = m.add("bottle", 3.65, 1.05, 0.8, range_m=1.5, t=2.0)  # close view
    assert a is b and a.n == 2 and len(m.objects) == 1
    x, y = a.xy
    assert abs(x - 3.65) < 0.05  # dominated by the close view (weight ~ 1/range^2)
    c = m.add("bottle", 0.0, -3.0, 0.7, range_m=2.0, t=3.0)  # somewhere else: a second bottle
    assert c is not a and len(m.of_class("bottle")) == 2
    d = m.add("chair", 3.6, 1.0, 0.9, range_m=1.5, t=3.0)  # other class never merges
    assert d is not a


def test_memory_confirmation_and_best():
    m = SemanticMemory()
    o = m.add("bottle", 1, 1, 0.5, 3.0, 0.0)
    assert not o.confirmed and m.best("bottle") is None and m.best("bottle", confirmed_only=False) is o
    m.add("bottle", 1.1, 1.0, 0.5, 3.0, 1.0)
    m.add("bottle", 1.0, 0.9, 0.5, 3.0, 2.0)
    assert o.confirmed  # three consistent sightings
    p = m.add("bottle", 5, 5, 0.9, 2.0, 3.0, confirmed=True)
    assert p.confirmed and m.best("bottle") is p  # lock-confirmed and heavier weight
    assert {s["cls"] for s in m.summary()} == {"bottle"}


# ------------------------------------------------------------------ free space
def test_null_free_space_sectors():
    fs = NullFreeSpace(5).estimate(None, tello_camera(), 1.2)
    bs = [b for b, _, _ in fs.sectors]
    assert len(bs) == 5 and bs[2] == pytest.approx(0.0) and bs[0] == pytest.approx(-bs[4])
    assert all(o is None and d is None for _, o, d in fs.sectors) and not fs.metric
    assert isinstance(FREESPACE.build("null"), NullFreeSpace)


def test_sim_free_space_matches_geometry():
    from reachglass.sim.drone_sim import SimDroneParams
    from reachglass.sim.scenario import Sim, SimFreeSpace
    from reachglass.sim.world import World

    sim = Sim(World(size_x=6.0, size_y=6.0, boxes=[]), (1.0, 3.0), drone_heading=0.0, params=SimDroneParams(hover_drift=0))
    sim.drone.takeoff()
    while sim.drone.busy():
        sim.step(0.05)
    sim.step(0.3)
    f = sim.camera.read()
    fs = SimFreeSpace(sim, max_m=10).estimate(f.image, sim.cam, 0.8)
    for b, _, d in fs.sectors:
        assert d == pytest.approx(5.0 / math.cos(math.radians(b)), rel=0.02)  # wall at x = 6
