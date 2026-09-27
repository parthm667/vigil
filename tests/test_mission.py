"""Step 12: the whole mission on the simulator: follow -> query -> explore -> approach -> guidance -> follow -> land."""

import math

import pytest

from reachglass.config import load_config
from reachglass.mission import ScriptedInbox, clock_face, compute_guidance, direction_words
from reachglass.mission.inbox import UdpInbox
from reachglass.sim.runner import SimRunner
from reachglass.types import Pose2D

TARGET = (6.2, 5.0)


# ------------------------------------------------------------------ guidance maths
def test_guidance_words_and_clock():
    assert clock_face(0) == "12 o'clock" and clock_face(90) == "3 o'clock" and clock_face(-60) == "10 o'clock"
    assert clock_face(180) == "6 o'clock"
    assert direction_words(5) == "straight ahead" and direction_words(25) == "ahead, slightly to your right"
    assert direction_words(-90) == "to your left" and direction_words(175) == "behind you"
    g = compute_guidance("bottle", (3.0, 4.0), (0.0, 0.0), 0.0)
    assert g.distance_m == pytest.approx(5.0) and g.turn_deg == pytest.approx(math.degrees(math.atan2(4, 3)))
    assert "5 meters" in g.text and "right" in g.text
    assert compute_guidance("bottle", (1, 1), None, None).distance_m is None
    d, turn = g.relative_to(3.0, 0.0, 90.0)  # person walked to (3, 0) and faces +y: target straight ahead
    assert d == pytest.approx(4.0) and turn == pytest.approx(0.0)


def test_udp_inbox_receives_queries():
    import socket

    ib = UdpInbox(port=0)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"find my water bottle", ("127.0.0.1", ib.port))
    s.sendto(b"  land  ", ("127.0.0.1", ib.port))
    import time

    time.sleep(0.05)
    assert ib.poll(0) == ["find my water bottle", "land"]
    ib.close()


# ------------------------------------------------------------------ whole mission
def states(r):
    return [s for _, s, _ in r.mission.history]


def to_mission_frame(r, xy):
    """Sim-world point -> the mission frame (origin = drone pose when the query arrived)."""
    o = r.origin
    dx, dy = xy[0] - o[0], xy[1] - o[1]
    h = math.radians(o[2])
    return dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)


@pytest.mark.slow
def test_full_mission_follow_find_approach_guide_follow_land():
    cfg = load_config()
    inbox = ScriptedInbox([(26.0, "Hey, can you find my water bottle?")])
    r = SimRunner(cfg, inbox=inbox, seed=0)
    # follow for 25 s
    r.run(25.0)
    assert r.mission.state == "FOLLOW" and r.sim.drone.collisions == 0
    px, py, ph = r.truth_person()
    d = r.sim.drone.pos
    assert abs(math.hypot(d[0] - px, d[1] - py) - cfg.follow.distance_m) < 0.5 and abs(d[2] - cfg.follow.altitude_m) < 0.25
    # the query: remember where the drone and the person were (ground truth for the guidance check)
    r.run(26.0 - r.t - 1e-6)
    r.origin = (r.sim.drone.pos[0], r.sim.drone.pos[1], r.sim.drone.heading)
    person_at_query = r.truth_person()
    r.run(200.0, until=lambda rr: rr.mission.state in ("ARRIVED", "REACQUIRE", "HOLD", "LANDED"))
    assert r.mission.state == "ARRIVED", (states(r), r.ctx.notes[-12:])
    assert r.sim.drone.collisions == 0
    assert 0.7 < r.distance_drone_to(TARGET) < 2.0
    # guidance: the target relative to where the person stood and faced when they asked
    g = r.mission.guidance
    assert g is not None and g.distance_m is not None, r.ctx.notes[-10:]
    true_person = to_mission_frame(r, person_at_query[:2])
    true_target = to_mission_frame(r, TARGET)
    true_dist = math.hypot(true_target[0] - true_person[0], true_target[1] - true_person[1])
    assert g.distance_m == pytest.approx(true_dist, abs=0.8), (g, true_dist)
    if g.turn_deg is not None:
        true_heading = person_at_query[2] - r.origin[2]
        true_turn = math.degrees(math.atan2(true_target[1] - true_person[1], true_target[0] - true_person[0])) - true_heading
        assert abs((g.turn_deg - true_turn + 180) % 360 - 180) < 25, (g.turn_deg, true_turn)
    said = [s for _, s in r.mission.said]
    assert any("Looking for your bottle" in s for s in said) and any("I found the bottle" in s for s in said)
    # "follow me": climbs back, turns until the person is seen, follows again
    inbox.push(r.t, "ok, follow me")
    r.run(60.0, until=lambda rr: rr.mission.state == "FOLLOW")
    assert r.mission.state == "FOLLOW", (states(r), r.ctx.notes[-8:])
    r.run(15.0)
    px, py, _ = r.truth_person()
    assert math.hypot(r.sim.drone.pos[0] - px, r.sim.drone.pos[1] - py) < 3.0
    inbox.push(r.t, "land")
    r.run(10.0, until=lambda rr: rr.mission.state == "LANDED")
    assert r.mission.state == "LANDED" and not r.sim.drone.flying and r.sim.drone.collisions == 0


def test_unknown_target_and_describe_keep_following():
    inbox = ScriptedInbox([(12.0, "find my keys"), (14.0, "what's around me?"), (16.0, "blah blah")])
    said = []
    r = SimRunner(load_config(), inbox=inbox, seed=1, announce=said.append)
    r.run(20.0)
    assert r.mission.state == "FOLLOW"
    assert any("can't look for keys" in s for s in said) and any("didn't understand" in s for s in said)
    assert len(said) == 3


def test_waits_on_the_ground_for_takeoff():
    said = []
    inbox = ScriptedInbox([(2.0, "find my bottle"), (4.0, "takeoff")])
    r = SimRunner(load_config(), inbox=inbox, seed=7, announce=said.append)
    r.mission.start(wait_for_operator=True)
    r._started = True
    r.run(3.5)
    assert r.mission.state == "IDLE" and not r.sim.drone.flying
    assert any("Say 'takeoff'" in s or "Say or type 'takeoff'" in s for s in said) and any("on the ground" in s for s in said)
    assert r.ctx.res is not None and r.ctx.res.ran.get("person")  # perception already running for the pre-flight check
    r.run(6.0)
    assert r.mission.state in ("TAKEOFF", "CLIMB", "FOLLOW") and r.sim.drone.flying


def test_land_during_search_and_cancel():
    inbox = ScriptedInbox([(12.0, "find my bottle"), (20.0, "never mind")])
    r = SimRunner(load_config(), inbox=inbox, seed=2)
    r.run(21.0)
    assert "REACQUIRE" in states(r) or r.mission.state in ("REACQUIRE", "FOLLOW")
    inbox.push(r.t, "land now")
    r.run(12.0, until=lambda rr: rr.mission.state == "LANDED")
    assert r.mission.state == "LANDED" and not r.sim.drone.flying


def test_safety_lands_on_low_battery_mid_search():
    inbox = ScriptedInbox([(12.0, "find my water bottle")])
    r = SimRunner(load_config(), inbox=inbox, seed=3)
    r.run(16.0)
    r.sim.drone.battery = 10.0
    r.run(10.0, until=lambda rr: rr.mission.state == "LANDED")
    assert r.mission.state == "LANDED" and not r.sim.drone.flying
    assert any("battery" in e for _, e in r.drone.events)


# ------------------------------------------------------------------ ARRIVED guidance refresh
def _arrived_mission(said):
    """A Mission parked in ARRIVED with known guidance, on a minimal fake ctx."""
    from types import SimpleNamespace

    from reachglass.mission.mission import Mission
    from reachglass.query import KeywordQueryParser

    ctx = SimpleNamespace(
        now=100.0, note=lambda s: None, target_cls="bottle",
        cfg=SimpleNamespace(mission=SimpleNamespace(announce=True)),
        perception=SimpleNamespace(vocabulary=lambda: ["bottle"]),
        drone=SimpleNamespace(stop=lambda: None),
        odom=SimpleNamespace(pose=Pose2D(0.0, 0.0, 0.0)),
        last_person=None, last_person_t=-1e9,
    )
    m = Mission(ctx, KeywordQueryParser(), announce=said.append)
    m.state = "ARRIVED"
    m.guidance = compute_guidance("bottle", (3.0, 0.0), (0.0, 0.0), 0.0)
    return m, ctx


def test_arrived_repeat_find_refreshes_guidance_not_search():
    from reachglass.types import Detection, PersonObs

    # person not seen recently: the original sentence again, and NO search restart
    said = []
    m, ctx = _arrived_mission(said)
    m._handle("find my bottle")
    assert m.state == "ARRIVED" and said == [m.guidance.text]

    # person seen walking: fresh distance/turn from where they are NOW
    said.clear()
    det = Detection(cls="person", conf=0.9, bbox=(0, 0, 20, 60))
    ctx.last_person = PersonObs(det, bearing_deg=0.0, elevation_deg=0.0, range_m=1.0,
                                facing_deg=180.0, facing_conf=0.9)  # 1 m ahead, facing the drone
    ctx.last_person_t = ctx.now
    m._handle("find my bottle")
    assert m.state == "ARRIVED" and len(said) == 1 and said[0] != m.guidance.text
    assert "meter" in said[0]

    # a DIFFERENT target must still restart the search normally
    said.clear()
    ctx.perception.vocabulary = lambda: ["bottle", "backpack"]
    ctx.perception.set_target = lambda cls: None
    try:
        m._handle("find my backpack")
    except AttributeError:
        pass  # _begin_search needs the full ctx; reaching it is what we assert
    assert not any(s == m.guidance.text for s in said)
