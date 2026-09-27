"""Optional fruit fly steering (follow.steering / approach.steering: fly): config, fallback, closed loop."""

import math
import sys

import pytest

from reachglass.behaviors import fly_steer
from reachglass.behaviors.base import Ctx
from reachglass.behaviors.fly_steer import FlyYaw
from reachglass.config import load_config

try:
    import flyfollow.steer  # noqa: F401

    HAVE_FLY = True
except Exception:  # noqa: BLE001
    HAVE_FLY = False


class _Ctx:
    def __init__(self, cfg):
        self.cfg = cfg


def test_steering_defaults_to_pid_and_rejects_typos():
    cfg = load_config()
    assert cfg.follow.steering == "pid" and cfg.approach.steering == "pid" and cfg.fly.params_path == ""
    cfg = load_config(overrides={"follow": {"steering": "fly"}, "fly": {"params_path": None, "latency_s": 0.3}})
    assert cfg.follow.steering == "fly" and cfg.fly.params_path == "" and cfg.fly.latency_s == 0.3
    assert cfg.fly.deadband > 0 and cfg.fly.hysteresis > 0 and cfg.fly.slew > 0  # stick smoothing on by default
    with pytest.raises(ValueError):
        load_config(overrides={"approach": {"steering": "brain"}})


def test_pid_config_never_touches_the_fly():
    assert FlyYaw.for_behavior(_Ctx(load_config()), "follow", 1.0) is None


def test_missing_flyfollow_falls_back_to_pid(monkeypatch):
    monkeypatch.setitem(sys.modules, "flyfollow.steer", None)  # import fails like an uninstalled package
    monkeypatch.setattr(fly_steer, "_SHARED", {})
    cfg = load_config(overrides={"follow": {"steering": "fly"}, "fly": {"brain_path": "not_installed.npz"}})
    assert FlyYaw.for_behavior(_Ctx(cfg), "follow", 1.0) is None


def test_fly_error_mid_flight_returns_the_pid_yaw():
    class Broken:
        target_valid = True

        def yaw(self, *a, **k):
            raise RuntimeError("boom")

    f = FlyYaw(Broken(), 40)
    assert f.yaw(1.0, 10.0, 1.0, 1.0, fallback=7.0) == 7.0 and f.failed
    assert f.yaw(1.1, fallback=-3.0) == -3.0 and not f.target_valid


@pytest.mark.skipif(not HAVE_FLY, reason="flyfollow (fruitfly-training) not installed")
def test_fly_steers_follow_closed_loop():
    from reachglass.behaviors.follow import FollowBehind
    from reachglass.behaviors.loop import sense
    from reachglass.perception import Perception
    from reachglass.sim.drone_sim import SimDroneParams
    from reachglass.sim.person_model import SimPerson
    from reachglass.sim.scenario import Sim
    from reachglass.sim.world import PersonScript, World

    script = PersonScript([(0.0, 3.5, 3.6, 0.0)])  # standing, ~17 deg right of the drone's heading
    w = World(size_x=9.0, size_y=8.0, boxes=[])
    w.person = SimPerson(*script.at(0.0))
    w.person_script = script
    sim = Sim(w, (1.2, 3.0), 0.0, params=SimDroneParams(seed=0))
    cfg = load_config(overrides={"follow": {"steering": "fly"}})
    ctx = Ctx(cfg, sim.drone, Perception(cfg, sim.person_detector, None, None, sim.cam))
    sim.drone.takeoff()
    while sim.drone.busy():
        sim.step(0.05)
    beh = FollowBehind(cfg.follow)
    beh.start(ctx)
    assert beh.fly is not None
    for _ in range(int(12.0 / (1.0 / 15))):
        sim.step(1.0 / 15)
        sense(ctx, sim.camera, sim.t)
        beh.step(ctx)
    p, d = sim.world.person, sim.drone.pos
    bearing = (math.degrees(math.atan2(p.y - d[1], p.x - d[0])) - sim.drone.heading + 180) % 360 - 180
    assert abs(bearing) < 12, (bearing, beh.status)  # turned onto the person (hand calibration holds ~5 deg left)
    assert sim.drone.collisions == 0 and beh.fly.steer.stats["ticks"] > 150 and not beh.fly.failed
