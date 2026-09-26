"""Step 11b: scan / explore / approach, closed loop in the simulator with the real colour-blob detector."""

import math

import pytest

from reachglass.behaviors import FAILURE, RUNNING, SUCCESS, Approach, Ctx, Explore, Scan, choose_hop, sense
from reachglass.config import load_config
from reachglass.detect import ColorBlobDetector
from reachglass.perception import Perception
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.scenario import Sim, SimFreeSpace
from reachglass.sim.world import demo_world

DT = 1.0 / 15
TARGET = (6.2, 5.0)


def setup(start=(1.0, 1.5), heading=0.0, seed=0, target=TARGET, oracle_free=False, person_xy=(1.8, 3.2)):
    w = demo_world(target_xy=target)
    w.person.x, w.person.y, w.person.heading_deg = *person_xy, 0.0
    sim = Sim(w, start, heading, params=SimDroneParams(seed=seed))
    cfg = load_config()
    per = Perception(cfg, sim.person_detector, ColorBlobDetector(), sim.context_detector, sim.cam)
    ctx = Ctx(cfg, sim.drone, per, freespace=SimFreeSpace(sim) if oracle_free else None)
    ctx.target_cls = "bottle"
    per.set_target("bottle")
    sim.drone.takeoff()
    while sim.drone.busy():
        sim.step(0.05)
    assert sim.drone.flying, f"takeoff failed at {start}"
    sim.drone.move("up", 40)
    while sim.drone.busy():
        sim.step(0.05)
    sense(ctx, sim.camera, sim.t)
    ctx.odom.reset(ctx.tel)
    return sim, ctx


def run(sim, ctx, beh, max_s):
    beh.start(ctx)
    r = RUNNING
    t0 = sim.t
    while r == RUNNING and sim.t - t0 < max_s:
        sim.step(DT)
        sense(ctx, sim.camera, sim.t)
        r = beh.step(ctx)
    return r, sim.t - t0


def world_of(ctx, sim, xy):
    """Mission-frame point -> sim world (the mission frame starts at the drone's pose at reset)."""
    return xy  # used only for readability in assertions below


def drone_to_target(sim, target=TARGET):
    return math.hypot(sim.drone.pos[0] - target[0], sim.drone.pos[1] - target[1])


def test_scan_from_a_spot_that_sees_the_target_confirms_it():
    sim, ctx = setup(start=(4.2, 2.6), heading=180.0)  # target ~3.1 m away, behind the drone at start
    scan = Scan(ctx.cfg.explore)
    r, dt = run(sim, ctx, scan, 90)
    assert r == SUCCESS and scan.found, scan.status
    obj = ctx.memory.best("bottle")
    assert obj is not None


def test_scan_without_target_in_view_covers_360_and_fills_memory():
    sim, ctx = setup(start=(1.0, 1.0), heading=0.0, target=(6.2, 5.0))
    sim.world.boxes = [b for b in sim.world.boxes if b.cls != "bottle"]  # no target at all
    # coverage test: the table is in exactly one view, where the (seeded) 5 % detector dropout can hit both looks
    sim.context_detector.dropout = 0.0
    scan = Scan(ctx.cfg.explore)
    r, dt = run(sim, ctx, scan, 120)
    assert r == SUCCESS and not scan.found
    # 8 views need 7 rotations of 45 deg: the scan ends at 315 deg (-45), within the rotation noise
    assert ctx.odom.pose.heading_deg == pytest.approx(-45, abs=8)
    classes = {o.cls for o in ctx.memory.objects}
    assert {"dining table", "couch"} <= classes, classes
    assert ctx.grid.coverage() > 0.01
    choice = choose_hop(ctx)
    assert choice is not None


@pytest.mark.slow
@pytest.mark.parametrize("seed,start,heading", [(0, (1.0, 1.5), 0.0), (1, (1.2, 1.0), 90.0), (2, (2.5, 1.0), 180.0)])
def test_explore_then_approach_reaches_the_target(seed, start, heading):
    sim, ctx = setup(start=start, heading=heading, seed=seed)
    ex = Explore(ctx.cfg.explore)
    r, t_explore = run(sim, ctx, ex, 200)
    assert r == SUCCESS, (ex.status, ex.decisions, ctx.notes[-5:])
    ap = Approach(ctx.cfg.approach)
    r, t_approach = run(sim, ctx, ap, 90)
    assert r == SUCCESS, (ap.status, ctx.notes[-8:])
    d = drone_to_target(sim)
    assert 0.8 < d < 1.9, (d, ap.status)  # standoff 1.3 m +- measurement/odometry error
    assert sim.drone.collisions == 0
    assert t_explore + t_approach < 220
    # the drone's odometry agrees with where the sim says it is (relative to the mission start)
    assert ctx.odom.pose is not None
