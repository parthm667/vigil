"""Harder whole-mission scenarios on the simulator (slow)."""

import math

import pytest

from reachglass.config import load_config
from reachglass.mission import ScriptedInbox
from reachglass.sim.person_model import SimPerson
from reachglass.sim.runner import SimRunner
from reachglass.sim.world import Box, PersonScript, World, dummy_bottle


def room(target_xy, target_z, extra=()):
    boxes = [
        Box((0.1, 4.6, 0.0), (2.2, 5.9, 0.85), (100, 140, 90), "couch"),
        Box((5.3, 4.2, 0.0), (6.9, 5.9, 0.75), (60, 90, 120), "dining table"),
        Box((6.6, 0.2, 0.0), (6.95, 1.6, 1.8), (150, 150, 150), ""),
        Box((0.2, 0.2, 0.0), (0.9, 0.9, 0.6), (80, 110, 140), "chair"),
        *extra,
    ]
    boxes += dummy_bottle(*target_xy, target_z)
    return World(size_x=7.0, size_y=6.0, boxes=boxes)


def run_mission(world, person_script, drone_xy, drone_heading, seed=0, query_at=15.0, max_s=220.0):
    world.person = SimPerson(*person_script.at(0.0))
    world.person_script = person_script
    inbox = ScriptedInbox([(query_at, "find my water bottle")])
    r = SimRunner(load_config(), world=world, drone_xy=drone_xy, drone_heading=drone_heading, inbox=inbox, seed=seed)
    r.run(query_at + max_s, until=lambda rr: rr.mission.state in ("ARRIVED", "HOLD", "LANDED")
          or (rr.mission.state == "REACQUIRE" and rr.t > query_at + 1))
    return r


@pytest.mark.slow
def test_target_behind_the_start_found_by_scanning():
    # side table behind-left of the drone's initial view
    side = Box((0.2, 2.3, 0.0), (0.8, 3.1, 0.7), (90, 100, 130), "dining table")
    w = room((0.5, 2.7), 0.7, extra=[side])
    script = PersonScript([(0.0, 3.4, 2.0, 0.0)])
    r = run_mission(w, script, drone_xy=(1.6, 2.0), drone_heading=0.0, seed=4)
    assert r.mission.state == "ARRIVED", r.ctx.notes[-10:]
    assert r.sim.drone.collisions == 0 and r.distance_drone_to((0.5, 2.7)) < 2.0


@pytest.mark.slow
def test_target_on_the_floor_needs_descent():
    w = room((4.5, 1.2), 0.0)
    script = PersonScript([(0.0, 2.6, 2.5, -30.0)])
    r = run_mission(w, script, drone_xy=(1.0, 3.2), drone_heading=-30.0, seed=5)
    assert r.mission.state == "ARRIVED", r.ctx.notes[-10:]
    assert r.sim.drone.collisions == 0 and r.distance_drone_to((4.5, 1.2)) < 2.0


@pytest.mark.slow
def test_person_walks_away_during_search_then_follow_me_finds_them():
    script = PersonScript([(0.0, 2.8, 1.6, 20.0), (18.0, 2.8, 1.6, 20.0), (28.0, 2.0, 3.8, 120.0), (200.0, 2.0, 3.8, 120.0)])
    w = room((6.2, 5.0), 0.75)
    r = run_mission(w, script, drone_xy=(1.1, 1.0), drone_heading=20.0, seed=6)
    assert r.mission.state == "ARRIVED", r.ctx.notes[-10:]
    r.inbox.push(r.t, "follow me")
    r.run(90.0, until=lambda rr: rr.mission.state == "FOLLOW")
    assert r.mission.state == "FOLLOW", r.ctx.notes[-10:]
    r.run(15.0)
    px, py, _ = r.truth_person()
    assert math.hypot(r.sim.drone.pos[0] - px, r.sim.drone.pos[1] - py) < 3.2 and r.sim.drone.collisions == 0
