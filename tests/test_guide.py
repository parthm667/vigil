"""GUIDE: walking path on the search map, cues -1/0/1/2, and the whole pipeline ending with the drone landing."""

import math

import pytest

from reachglass.config import load_config
from reachglass.mapping import Grid
from reachglass.mapping.walkpath import cue_from_error, plan_walk, segment_max_cost, steer, walk_costs
from reachglass.mission import ScriptedInbox
from reachglass.sim.runner import SimRunner

TARGET = (6.2, 5.0)  # the demo room's bottle (on the table)


def test_walk_path_open_floor_is_one_straight_leg():
    g = Grid(16.0, 0.25)
    p = plan_walk(g, walk_costs(g), (0.0, 0.0), (3.0, 1.0))
    assert len(p.points) == 2 and p.points[0] == (0.0, 0.0)
    assert math.dist(p.goal, (3.0, 1.0)) <= 0.5  # stops just short of the bottle, on the person's side
    assert p.length < math.dist((0.0, 0.0), (3.0, 1.0))


def test_walk_path_goes_around_a_chair_and_stops_at_the_table():
    g = Grid(16.0, 0.25)
    g.mark_blocked(1.5, 0.5, 0.3, top_m=0.85)  # a chair on the straight line (the drone flies over it, a person can't)
    costs = walk_costs(g)
    p = plan_walk(g, costs, (0.0, 0.0), (3.0, 1.0))
    assert len(p.points) >= 3 and p.length > math.dist((0.0, 0.0), (3.0, 1.0))
    assert all(segment_max_cost(g, costs, a, b) <= 1.5 for a, b in zip(p.points, p.points[1:]))  # never near the chair
    # a bottle on a table: the walk ends at the table's edge, on the person's side
    g = Grid(16.0, 0.25)
    g.mark_blocked(4.0, 0.0, 0.6, top_m=0.75)
    p = plan_walk(g, walk_costs(g), (0.0, 0.0), (4.1, 0.1))
    assert p.goal[0] < 4.0 - 0.6 and math.dist(p.goal, (4.1, 0.1)) < 1.5


def test_cues_left_right_forward():
    path = [(0.0, 0.0), (4.0, 0.0)]  # straight ahead along +x (y = to the right)
    for heading, cue in ((0.0, 0), (-45.0, 1), (45.0, -1), (180.0, 1)):
        err, _, _ = steer(path, (0.0, 0.0), heading)
        assert cue_from_error(err, None) == cue, (heading, err)
    err, _, _ = steer([(0.0, 0.0), (0.0, 3.0)], (0.0, 0.0), 0.0)  # the path turns to their right
    assert cue_from_error(err, None) == 1
    assert cue_from_error(20.0, 0) == 0 and cue_from_error(20.0, 1) == 1  # hysteresis: no flicker at the edge


@pytest.mark.slow
def test_whole_pipeline_guides_the_wearer_to_the_bottle_then_lands():
    inbox = ScriptedInbox([(26.0, "can you find my water bottle")])
    r = SimRunner(load_config(), inbox=inbox, seed=0)
    r.run(260.0, until=lambda rr: rr.mission.state in ("GUIDE", "REACQUIRE", "HOLD", "LANDED"))
    assert r.mission.state == "GUIDE", r.ctx.notes[-10:]
    person = r.sim.world.person
    r.sim.world.person_script = None  # from now on the wearer walks by the cues
    furniture = [b for b in r.sim.world.boxes if b.cls in ("chair", "dining table", "couch")]
    t_end = r.t + 150.0
    while r.t < t_end and r.mission.state == "GUIDE":
        r.step()
        cue = r.mission.cues[-1][1] if r.mission.cues else None
        if cue in (-1, 0, 1):  # -1: turn left and step that way, 0: walk forward, 1: turn right and step
            person.heading_deg += 60.0 * cue * r.dt
            v = 0.5 if cue == 0 else 0.15
            h = math.radians(person.heading_deg)
            person.x += v * math.cos(h) * r.dt
            person.y += v * math.sin(h) * r.dt
            assert not any(b.contains((person.x, person.y, 0.5), margin=0.1) for b in furniture), r.ctx.notes[-6:]
    r.run(15.0, until=lambda rr: rr.mission.state == "LANDED")
    cues = [c for _, c in r.mission.cues]
    assert r.mission.state == "LANDED" and not r.sim.drone.flying, r.ctx.notes[-10:]
    assert cues[-1] == 2 and cues.count(2) == 1 and set(cues[:-1]) <= {-1, 0, 1}
    assert math.hypot(person.x - TARGET[0], person.y - TARGET[1]) < 1.6  # at the table, next to the bottle
    assert r.sim.drone.collisions == 0
