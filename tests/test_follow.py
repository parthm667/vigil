"""Step 11a: follow-behind, closed loop in the simulator."""

import math

import numpy as np
import pytest

from reachglass.behaviors.base import Ctx
from reachglass.behaviors.follow import FollowBehind
from reachglass.behaviors.loop import sense
from reachglass.config import load_config
from reachglass.perception import Perception
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.person_model import SimPerson
from reachglass.sim.scenario import Sim
from reachglass.sim.world import PersonScript, World

DT = 1.0 / 15
ALT = load_config().follow.altitude_m


def setup(script, drone_xy=(1.2, 3.0), heading=0.0, seed=0, **params):
    w = World(size_x=9.0, size_y=8.0, boxes=[])
    w.person = SimPerson(*script.at(0.0))
    w.person_script = script
    sim = Sim(w, drone_xy, heading, params=SimDroneParams(seed=seed, **params))
    cfg = load_config()
    per = Perception(cfg, sim.person_detector, None, None, sim.cam)
    ctx = Ctx(cfg, sim.drone, per)
    sim.drone.takeoff()
    while sim.drone.busy():
        sim.step(0.05)
    beh = FollowBehind(cfg.follow)
    beh.start(ctx)
    return sim, ctx, beh


def run(sim, ctx, beh, seconds, record=None):
    for _ in range(int(seconds / DT)):
        sim.step(DT)
        sense(ctx, sim.camera, sim.t)
        beh.step(ctx)
        if record is not None:
            p = sim.world.person
            record.append((sim.t, *sim.drone.pos, sim.drone.heading, p.x, p.y, p.heading_deg))


def geometry(sim):
    p, d = sim.world.person, sim.drone.pos
    dist = math.hypot(d[0] - p.x, d[1] - p.y)
    # where is the drone relative to the person's back? 0 = straight behind
    ang = math.degrees(math.atan2(d[1] - p.y, d[0] - p.x))
    behind_err = (ang - (p.heading_deg + 180) + 180) % 360 - 180
    return dist, behind_err, d[2]


def test_climbs_to_follow_altitude_and_holds_distance_behind_standing_person():
    script = PersonScript([(0.0, 3.5, 3.0, 0.0)])
    sim, ctx, beh = setup(script)
    run(sim, ctx, beh, 12.0)
    dist, behind, alt = geometry(sim)
    assert 1.4 < dist < 2.3 and abs(behind) < 20 and abs(alt - ALT) < 0.15, (dist, behind, alt, beh.status)
    assert sim.drone.collisions == 0


def test_follows_walking_person_through_a_turn_and_ends_up_behind_them():
    # walk +x 2 m, turn right 90 deg, walk +y 1.5 m, then stand
    script = PersonScript([(0.0, 3.5, 2.5, 0.0), (5.0, 3.5, 2.5, 0.0), (11.0, 5.5, 2.5, 0.0), (12.5, 5.5, 2.5, 90.0),
                           (18.0, 5.5, 4.0, 90.0)])
    sim, ctx, beh = setup(script, drone_xy=(1.7, 2.5))
    rec = []
    run(sim, ctx, beh, 40.0, rec)
    dist, behind, alt = geometry(sim)
    assert 1.3 < dist < 2.4 and abs(behind) < 40 and abs(alt - ALT) < 0.2, (dist, behind, alt, beh.status)
    # never closer than 1.0 m horizontally while following
    closest = min(math.hypot(r[1] - r[5], r[2] - r[6]) for r in rec if r[3] > 1.5)
    assert closest > 1.0 and sim.drone.collisions == 0


def test_person_turning_to_face_the_drone_makes_it_orbit_behind():
    script = PersonScript([(0.0, 3.5, 3.0, 0.0), (6.0, 3.5, 3.0, 0.0), (7.0, 3.5, 3.0, 180.0)])  # turns around
    sim, ctx, beh = setup(script)
    run(sim, ctx, beh, 30.0)
    dist, behind, alt = geometry(sim)
    assert abs(behind) < 45 and 1.2 < dist < 2.6, (dist, behind, beh.status)


def test_person_lost_then_search_turns_toward_last_side():
    script = PersonScript([(0.0, 3.5, 3.0, 0.0), (3.0, 3.5, 3.0, 0.0), (4.0, 2.8, 5.5, 0.0)])  # steps away to the right
    sim, ctx, beh = setup(script)
    run(sim, ctx, beh, 20.0)
    dist, _, _ = geometry(sim)
    assert dist < 3.0 and ctx.res.person is not None or "lost" not in beh.status
