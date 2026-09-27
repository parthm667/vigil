"""Regression tests for the confirmed findings of the steps 9-12 review."""

import math
import time

import numpy as np
import pytest

from reachglass.behaviors import Ctx, Scan, choose_hop, observe, sense
from reachglass.config import load_config
from reachglass.detect import ColorBlobDetector
from reachglass.drone import SafetyGovernor, make_tello
from reachglass.drone.tello import TelloDrone
from reachglass.geometry import tello_camera
from reachglass.mission import ScriptedInbox
from reachglass.perception import Perception
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.runner import SimRunner
from reachglass.types import Detection, Frame, ObjectObs, PerceptionResult, Pose2D, Telemetry
from test_drone import FakeTello, wait


# ------------------------------------------------------------------ adapter / safety
def test_frozen_state_stream_is_detected_as_stale():
    fake = FakeTello()
    d = TelloDrone(tello=fake, video=False)
    d.connect()
    d.takeoff()
    wait(d)
    gov = SafetyGovernor(d, load_config().safety)
    now = time.time()
    assert gov.check(now, now) == "ok"
    time.sleep(1.2)  # the state dict is never replaced: djitellopy's receiver thread died
    now = time.time()
    assert gov.check(now, now) == "hover"
    fake.state = dict(fake.state)  # a new packet arrives
    now = time.time()
    assert gov.check(now, now) == "ok"


def test_dry_run_from_config_is_honoured_and_no_takeoff_live_refused():
    assert make_tello(load_config(overrides={"drone": {"kind": "dry_run"}})).dry_run
    assert not make_tello(load_config()).dry_run
    with pytest.raises(ValueError):
        make_tello(load_config(overrides={"drone": {"kind": "sim"}}))
    from reachglass.app import main

    assert main(["tello", "--no-takeoff", "--headless", "--udp", "0"]) == 4  # refused before connecting


def test_stop_never_cancels_takeoff_or_landing():
    d = TelloDrone(tello=FakeTello(delays={"takeoff": 0.3, "land": 0.3}), video=False)
    d.connect()
    d.takeoff()
    d.stop()
    assert d.busy() and "stop" not in d.tello.sent
    assert wait(d) == "ok" and d.flying
    d.land()
    d.stop()
    assert d.busy() and wait(d) == "ok" and not d.flying


def test_lost_land_reply_on_the_ground_counts_as_landed():
    fake = FakeTello()
    d = TelloDrone(tello=fake, video=False)
    d.connect()
    d.takeoff()
    wait(d)
    fake.error_for["land"] = "error"
    fake.state = dict(fake.state, h=0)  # it is down already
    d.land()
    assert wait(d) == "ok" and not d.flying


# ------------------------------------------------------------------ mission
def runner(queries, seed=11, **kw):
    said = []
    r = SimRunner(load_config(**kw) if kw else load_config(), inbox=ScriptedInbox(queries), seed=seed, announce=said.append)
    return r, said


def states(r):
    return [s for _, s, _ in r.mission.history]


def test_land_is_retried_when_the_first_one_is_lost():
    r, _ = runner([])
    r.run(8.0)
    real_land = r.sim.drone.land
    calls = []

    def lossy_land():
        calls.append(r.t)
        if len(calls) > 1:
            real_land()

    r.sim.drone.land = lossy_land
    r.mission.query("land")
    r.run(12.0, until=lambda rr: rr.mission.state == "LANDED")
    assert r.mission.state == "LANDED" and len(calls) >= 2 and calls[1] - calls[0] >= 2.0


def test_hold_is_ignored_during_takeoff():
    r, _ = runner([])
    r.step()
    assert r.mission.state == "TAKEOFF"
    r.mission.hold()
    assert r.mission.state == "TAKEOFF"
    r.run(8.0)
    assert r.mission.state == "FOLLOW" and r.sim.drone.flying


def test_repeated_find_does_not_restart_and_new_find_keeps_the_person():
    r, said = runner([(10.0, "find my water bottle")])
    r.run(14.0)
    assert r.mission.state in ("EXPLORE", "APPROACH")
    origin = r.ctx.person_origin
    n_states = len(r.mission.history)
    r.mission.query("find my water bottle")
    r.run(0.2)
    assert any("Still looking" in s for s in said) and len(r.mission.history) == n_states
    r.run(120.0, until=lambda rr: rr.mission.state == "ARRIVED")
    assert r.mission.state == "ARRIVED"
    r.sim.world.person.x, r.sim.world.person.y = 0.5, 0.5  # out of sight now
    r.mission.query("find the couch")
    r.run(0.3)
    assert r.ctx.target_cls == "couch" and r.ctx.person_origin is not None and origin is not None


def test_follow_me_while_following_a_lost_person_reacquires():
    r, _ = runner([])
    r.run(10.0)
    r.sim.world.person_script = None
    r.sim.world.person.x, r.sim.world.person.y = 0.4, 0.4  # walked out of view
    r.run(3.0)
    r.mission.query("follow me")
    r.run(0.2)
    assert "REACQUIRE" in states(r)


def test_find_while_taking_off_is_handled_once_following():
    r, said = runner([(0.5, "find my water bottle")])
    r.run(12.0)
    assert sum("still taking off" in s for s in said) == 1
    assert "EXPLORE" in states(r)


def test_find_after_landing_is_answered_not_looped():
    r, said = runner([])
    r.run(6.0)
    r.mission.query("land")
    r.run(8.0, until=lambda rr: rr.mission.state == "LANDED")
    r.mission.query("find my bottle")
    r.run(1.0)
    assert sum("landed" in s for s in said) == 1 and not r.mission._pending


# ------------------------------------------------------------------ exploration
def test_hop_chosen_even_with_nothing_detected():
    cfg = load_config()
    ctx = Ctx(cfg, None, Perception(cfg, None, None, None))
    ctx.vantages.append((0.0, 0.0))
    ctx.grid.mark_view(Pose2D(), 55.0, cfg.explore.view_range_m)  # we looked ahead, saw nothing
    choice = choose_hop(ctx)
    assert choice is not None and choice[1] >= cfg.explore.hop_min_m - 1e-6


def test_obstacle_top_comes_from_the_image_not_the_size_prior():
    cfg = load_config()
    cam = tello_camera()
    per = Perception(cfg, None, None, None, cam)
    ctx = Ctx(cfg, None, per)
    ctx.tel = Telemetry(0.0, height_m=1.2, tof_m=1.2)
    # a TV whose top edge is 5 deg above the horizon, 2.5 m away -> top ~ 1.2 + 2.5*tan(5) = 1.42 m
    y_top = cam.y_of_elevation(5.0)
    det = Detection("tv", 0.8, (430, y_top, 530, y_top + 60))
    ctx.res = PerceptionResult(0.0, 1, 0.0, (960, 720), [det], context=[ObjectObs(det, 0.0, 3.0, 2.5, "height_prior")])
    observe(ctx, mark_view=False)
    c = ctx.grid.idx(2.5, 0.0)
    assert ctx.grid.top[c] == pytest.approx(1.2 + 2.5 * math.tan(math.radians(5.0)), abs=0.05)
    assert ctx.grid.blocks(c, 1.2)  # it reaches flight level: an obstacle


@pytest.mark.slow
def test_target_out_of_detection_range_requires_hops_and_odometry_stays_accurate():
    class ShortSighted(ColorBlobDetector):
        """Only sees the dummy within ~3 m (like a small real object)."""

        def detect(self, image):
            f = 460.0 * image.shape[1] / 480
            return [d for d in super().detect(image) if d.h >= f * 0.22 / 3.0]

    from reachglass.behaviors import Approach, Explore, RUNNING, SUCCESS
    from reachglass.sim.scenario import Sim
    from reachglass.sim.world import demo_world

    w = demo_world()
    w.person.x, w.person.y = 1.0, 4.0
    sim = Sim(w, (1.0, 1.5), 20.0, params=SimDroneParams(seed=3))
    cfg = load_config()
    per = Perception(cfg, sim.person_detector, ShortSighted(), sim.context_detector, sim.cam)
    ctx = Ctx(cfg, sim.drone, per)
    ctx.target_cls = "bottle"
    per.set_target("bottle")
    sim.drone.takeoff()
    while sim.drone.busy():
        sim.step(0.05)
    sim.drone.move("up", 40)
    while sim.drone.busy():
        sim.step(0.05)
    sense(ctx, sim.camera, sim.t)
    ctx.odom.reset(ctx.tel)
    start = (sim.drone.pos[0], sim.drone.pos[1], sim.drone.heading)
    ex = Explore(cfg.explore)
    ex.start(ctx)
    r = RUNNING
    while r == RUNNING and sim.t < 260:
        sim.step(1 / 15)
        sense(ctx, sim.camera, sim.t)
        r = ex.step(ctx)
    assert r == SUCCESS and ex.hops >= 1, (ex.status, ex.decisions)
    # odometry vs ground truth (mission frame = drone pose at reset)
    h = math.radians(start[2])
    dx, dy = sim.drone.pos[0] - start[0], sim.drone.pos[1] - start[1]
    truth = (dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h))
    assert math.hypot(ctx.odom.pose.x - truth[0], ctx.odom.pose.y - truth[1]) < 0.4
    assert sim.drone.collisions == 0


@pytest.mark.slow
def test_mission_with_long_video_lag():
    cfg = load_config(overrides={"drone": {"video_lag_s": 1.0}})
    r = SimRunner(cfg, inbox=ScriptedInbox([(20.0, "find my water bottle")]), seed=2)
    r.sim.drone.p.video_delay_s = 0.8
    r.run(220.0, until=lambda rr: rr.mission.state in ("ARRIVED", "REACQUIRE", "HOLD"))
    assert r.mission.state == "ARRIVED" and r.sim.drone.collisions == 0, r.ctx.notes[-8:]
    assert "approach failed" not in r.mission.history[-1][2]


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 3, 5])
def test_guidance_turn_is_right_across_seeds(seed):
    r = SimRunner(load_config(), inbox=ScriptedInbox([(22.0, "find my water bottle")]), seed=seed)
    r.run(22.0 - 1e-6)
    ox, oy, oh = r.sim.drone.pos[0], r.sim.drone.pos[1], r.sim.drone.heading
    px, py, ph = r.truth_person()
    r.run(200.0, until=lambda rr: rr.mission.state in ("ARRIVED", "REACQUIRE", "HOLD"))
    g = r.mission.guidance
    assert r.mission.state == "ARRIVED" and g is not None and g.turn_deg is not None, r.ctx.notes[-6:]
    true_turn = math.degrees(math.atan2(5.0 - py, 6.2 - px)) - ph
    assert abs((g.turn_deg - true_turn + 180) % 360 - 180) < 30, (g.turn_deg, true_turn)
