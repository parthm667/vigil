"""Tests for the pursuit sim, filter, governor, reward and rollout (plan items 4a, 4b, 5)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flyfollow.interfaces import DT, IMG_H, IMG_W, BoxState, Settings
from flyfollow.pilot.box_filter import BoxFilter
from flyfollow.pilot.governor import PersonGovernor
from flyfollow.pilot.pid import PIDController
from flyfollow.rl.env import PursuitEnv
from flyfollow.rl.reward import RewardConfig, e_z
from flyfollow.rl.rollout import run_episode
from flyfollow.sim.camera import project, visible_box


class Dummy:
    """Random sticks, seeded per episode (a stand-in for an untrained controller)."""

    name = "DUMMY"

    def reset(self, settings: Settings, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 12345)

    def warmup(self, seconds: float) -> None:
        pass

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        return float(self.rng.uniform(-100, 100)), float(self.rng.uniform(-100, 100))


class Const:
    name = "CONST"

    def __init__(self, yaw: float, fb: float):
        self.yaw, self.fb = yaw, fb

    def reset(self, settings, seed):
        pass

    def warmup(self, seconds):
        pass

    def act(self, box, settings, dt):
        return self.yaw, self.fb


@pytest.fixture(scope="module")
def env() -> PursuitEnv:
    return PursuitEnv()


# ---------------------------------------------------------------- determinism / common random numbers
def test_same_seed_same_result(env):
    for kind in ("follow", "approach"):
        a = run_episode(env, PIDController(), 10_003, kind)
        b = run_episode(env, PIDController(), 10_003, kind)
        assert a.ret == b.ret and a.terms == b.terms and a.n_ticks == b.n_ticks
        assert {k: v for k, v in a.metrics.items() if v == v} == {k: v for k, v in b.metrics.items() if v == v}


def test_same_world_for_different_controllers(env):
    for kind in ("follow", "approach"):
        for profile in ("train", "demo", "stress"):
            env.reset(10_007, kind, profile)
            w1 = env.world()
            run_episode(env, Dummy(), 10_007, kind, profile)
            env.reset(10_007, kind, profile)
            w2 = env.world()
            run_episode(env, PIDController(), 10_007, kind, profile)
            assert w1["params"] == w2["params"]
            assert w1["settings"] == w2["settings"]
            assert w1["camera_t"] == w2["camera_t"] and w1["camera_drop"] == w2["camera_drop"]
            if kind == "follow":
                assert w1["walker_xs"] == w2["walker_xs"] and w1["walker_ys"] == w2["walker_ys"]
            else:
                assert w1["object"] == w2["object"]


def test_worlds_differ_across_seeds(env):
    env.reset(10_000, "follow")
    a = env.world()
    env.reset(10_001, "follow")
    b = env.world()
    assert a["params"] != b["params"]


# ---------------------------------------------------------------- geometry signs
def test_projection_signs():
    fx = fy = 900.0
    # target 3 m ahead, 0.5 m to the RIGHT (level frame y is left, so left = -0.5)
    g = project(3.0, -0.5, 0.0, 0.23, 0.0, fx, fy, IMG_W / 2, IMG_H / 2)
    assert g[0] > IMG_W / 2  # right of center
    # above the camera -> upper half of the image (smaller v)
    g2 = project(3.0, 0.0, 0.5, 0.23, 0.0, fx, fy, IMG_W / 2, IMG_H / 2)
    assert 0.5 * (g2[1] + g2[2]) < IMG_H / 2
    # box height fy * H / Z
    g3 = project(2.0, 0.0, 0.0, 0.2, 0.0, fx, fy, IMG_W / 2, IMG_H / 2)
    assert abs((g3[2] - g3[1]) - fy * 0.2 / 2.0) < 1e-6
    # nose down pitch (accelerating) moves a level target UP in the image
    g4 = project(3.0, 0.0, 0.0, 0.23, math.radians(10), fx, fy, IMG_W / 2, IMG_H / 2)
    assert 0.5 * (g4[1] + g4[2]) < IMG_H / 2
    assert project(-1.0, 0.0, 0.0, 0.2, 0.0, fx, fy, 480, 360) is None
    assert visible_box(project(1.0, -2.0, 0.0, 0.2, 0.0, fx, fy, 480, 360)) is None  # far outside the FOV


def test_positive_yaw_reduces_positive_bearing(env):
    env.reset(10_010, "approach", "demo")
    # put the object 20 deg to the right of the drone heading
    r = math.hypot(env.tx, env.ty)
    env.tx, env.ty = r * math.cos(math.radians(-20)), r * math.sin(math.radians(-20))
    env._g = env._geom()
    b0 = math.atan2(-env._left, env._fwd)
    assert b0 > 0  # right = positive bearing
    for _ in range(10):
        env.drone.step(0, 0, 0, 40)
    env._geom()
    b1 = math.atan2(-env._left, env._fwd)
    assert b1 < b0 - math.radians(3)


def test_pid_yaw_sign():
    st = Settings(kind="follow", z_ref_m=2.0, target_size_m=0.23)
    pid = PIDController()
    pid.reset(st, 0)
    h = st.fy * 0.23 / 2.0
    yaw, fb = pid.act(BoxState(True, cx=700.0, cy=360.0, h=h), st, DT)
    assert yaw > 0 and fb == 0.0  # target right -> turn right; at the standoff -> no forward
    yaw, fb = pid.act(BoxState(True, cx=200.0, cy=360.0, h=h / 2), st, DT)
    assert yaw < 0 and fb > 0  # target left and far -> turn left, fly forward
    assert pid.act(BoxState(False), st, DT) == (0.0, 0.0)


def test_pid_tracks_in_closed_loop(env):
    """Closed loop: PID keeps a right-hand target near the center (sign convention end to end)."""
    res = run_episode(env, PIDController(), 10_020, "follow", "demo", record=True)
    berr = np.abs(np.asarray(res.trace["berr_deg"]))
    assert np.median(berr) < 10.0


# ---------------------------------------------------------------- governor
def _st(**kw) -> Settings:
    d = dict(kind="follow", z_ref_m=2.0, target_size_m=0.23, max_fwd_stick=35.0, z_min_m=1.2)
    d.update(kw)
    return Settings(**d)


def _box_at(st: Settings, z: float, cx: float = IMG_W / 2, cy: float | None = None) -> BoxState:
    return BoxState(True, cx=cx, cy=st.cy_ref_frac * IMG_H if cy is None else cy, h=st.fy * st.target_size_m / z)


def test_governor_clamps_and_slew():
    st = _st()
    gov = PersonGovernor()
    out = None
    for _ in range(20):
        out = gov.filter(100.0, 100.0, _box_at(st, 3.0), st, DT)
    assert out.fb == 35 and out.yaw == 60 and out.lr == 0 and out.clamped and not out.safety
    gov.reset()
    first = gov.filter(0.0, 100.0, _box_at(st, 3.0), st, DT)
    assert first.fb == 10  # slew 200/s -> 10 per tick
    assert gov.filter(0.0, 0.0, _box_at(st, 3.0), st, DT).fb == 0  # toward zero is instant
    gov.reset()
    assert gov.filter(0.0, -100.0, _box_at(st, 3.0), st, DT).fb == -10
    for _ in range(10):
        out = gov.filter(0.0, -100.0, _box_at(st, 3.0), st, DT)
    assert out.fb == -20  # reverse clamp


def test_governor_ud_loop_and_floor():
    st = _st()
    gov = PersonGovernor({"slew_ud_per_s": 1e9})
    up = gov.filter(0.0, 0.0, _box_at(st, 2.0, cy=100.0), st, DT, alt_m=1.5)
    assert up.ud > 0  # target above the reference row -> climb
    down = gov.filter(0.0, 0.0, _box_at(st, 2.0, cy=700.0), st, DT, alt_m=1.5)
    assert down.ud < 0
    assert abs(down.ud) <= 30
    st_a = _st(kind="approach", z_min_m=0.5)
    floor = gov.filter(0.0, 0.0, _box_at(st_a, 2.0, cy=700.0), st_a, DT, alt_m=0.8)
    assert floor.ud == 0 and "alt_floor" in floor.reasons


def test_governor_min_distance_uses_estimate():
    st = _st()
    gov = PersonGovernor({"slew_fb_per_s": 1e9})
    out = gov.filter(0.0, 30.0, _box_at(st, 1.0), st, DT)  # estimated 1.0 m < z_min 1.2
    assert out.fb < 0 and out.safety and "min_distance" in out.reasons
    out = gov.filter(0.0, 30.0, _box_at(st, 1.5), st, DT)
    assert out.fb > 0 and not out.safety


def test_governor_lost_target_logic():
    st = _st()
    gov = PersonGovernor({"slew_yaw_per_s": 1e9, "slew_fb_per_s": 1e9})
    gov.filter(0.0, 20.0, _box_at(st, 2.0, cx=800.0), st, DT)  # last seen on the right
    o = gov.filter(50.0, 50.0, BoxState(False, lost_s=0.1), st, DT)
    assert o.safety and o.fb == 0 and o.yaw == 20 and "lost_target" in o.reasons
    o = gov.filter(50.0, 50.0, BoxState(False, lost_s=5.5), st, DT)
    assert o.safety and o.hover and o.fb == 0 and o.yaw == 0 and not o.land
    o = gov.filter(50.0, 50.0, BoxState(False, lost_s=10.5), st, DT)
    assert o.land and (o.fb, o.yaw, o.ud) == (0, 0, 0)
    st_a = _st(kind="approach", z_min_m=0.5)  # no landing rule in APPROACH
    gov.reset()
    gov.filter(0.0, 0.0, _box_at(st_a, 2.0, cx=100.0), st_a, DT)
    o = gov.filter(0.0, 0.0, BoxState(False, lost_s=0.2), st_a, DT)
    assert o.yaw == -20
    assert not gov.filter(0.0, 0.0, BoxState(False, lost_s=30.0), st_a, DT).land


def test_governor_watchdog():
    st = _st()
    gov = PersonGovernor()
    o = gov.filter(40.0, 30.0, _box_at(st, 3.0), st, DT, brain_age_s=0.8)
    assert o.safety and o.fb == 0 and o.yaw == 0 and "watchdog" in o.reasons


# ---------------------------------------------------------------- filter
def test_filter_predicts_forward_and_holds():
    f = BoxFilter(latency_s=0.3, hold_s=0.5)
    # target moving right at 100 px/s, frames every 0.05 s, decoded 0.3 s after capture, detector 0.05 s
    for k in range(40):
        tc = k * 0.05
        f.update(tc + 0.35, tc + 0.3, 400.0 + 100.0 * tc, 360.0, 80.0)
    t_now = 39 * 0.05 + 0.35
    b = f.output(t_now)
    assert b.valid and abs(b.vcx - 100.0) < 10.0
    assert abs(b.cx - (400.0 + 100.0 * t_now)) < 10.0  # predicted to now, not to the capture time
    assert f.output(t_now + 0.4).valid
    lost = f.output(t_now + 0.7)
    assert not lost.valid and abs(lost.lost_s - 0.2) < 1e-9


def test_filter_gates_false_box():
    f = BoxFilter(latency_s=0.1)
    for k in range(20):
        f.update(k * 0.05 + 0.15, k * 0.05 + 0.1, 480.0, 360.0, 100.0)
    assert not f.update(20 * 0.05 + 0.15, 20 * 0.05 + 0.1, 50.0, 700.0, 30.0)
    assert abs(f.output(1.2).cx - 480.0) < 5.0


# ---------------------------------------------------------------- reward and termination
def test_e_z_shape():
    assert e_z(2.0, 2.0, 0.3) == 0.0
    assert e_z(2.29, 2.0, 0.3) == 0.0
    assert abs(e_z(1.5, 2.0, 0.3) - 0.2) < 1e-9
    assert abs(e_z(2.5, 2.0, 0.3) - (0.2 + 0.04)) < 1e-9  # far side adds the quadratic
    assert e_z(10.0, 2.0, 0.3) == 3.0


def test_collision_terminates_and_charges_tail(env):
    env.reset(10_030, "follow", "demo")
    # teleport the drone next to the person
    env.drone.x, env.drone.y = env.tx - 0.3, env.ty
    _, _r, done, info = env.step(0.0, 0.0)
    assert done and info["collision"]
    rc = RewardConfig.from_dict(env.cfg["reward"])
    tail = env.reward.terms()["ev_tail"]
    assert abs(tail + rc.max_rate * DT * (env.n_max - 1)) < 1e-6
    assert env.reward.terms()["ev_collision"] == rc.ev_collision
    res = env.result()
    assert res.metrics["collided"] == 1.0
    assert abs(res.ret - sum(res.terms.values())) < 1e-6


def test_return_equals_sum_of_terms(env):
    for kind in ("follow", "approach"):
        res = run_episode(env, PIDController(), 10_040, kind)
        assert abs(res.ret - sum(res.terms.values())) < 1e-6
        assert res.n_ticks <= (1200 if kind == "follow" else 500)


def test_approach_success_ends_episode(env):
    for s in range(10_050, 10_090):
        res = run_episode(env, PIDController(), s, "approach", "train")
        if res.metrics["success"]:
            assert res.terms["ev_success"] == 20.0 and res.n_ticks < 500
            assert res.metrics["time_to_standoff"] >= 0.0
            return
    pytest.fail("no PID-HAND approach success in 40 seeds")


# ---------------------------------------------------------------- bulk
@pytest.mark.parametrize("profile", ["train", "stress"])
def test_100_random_episodes_with_dummy_controller(env, profile):
    rng = np.random.default_rng(0)
    for k in range(100):
        kind = "follow" if k % 2 == 0 else "approach"
        res = run_episode(env, Dummy(), int(rng.integers(10_000, 1_000_000)), kind, profile)
        assert np.isfinite(res.ret)
        assert all(np.isfinite(v) for v in res.terms.values())
        assert res.n_ticks >= 1


def test_record_trace(env):
    res = run_episode(env, PIDController(), 10_060, "follow", record=True)
    n = len(res.trace["t"])
    assert n == res.n_ticks and all(len(v) == n for v in res.trace.values())
    assert res.wall_s > 0.0


def test_extras_are_active(env):
    """4b extras (false boxes, hover drift, pitch coupling) are on in the train profile."""
    seen_false = seen_drift = seen_pitch = False
    for s in range(10_000, 10_020):
        env.reset(s, "follow")
        p = env.params
        seen_false |= p["false_box_frac"] > 0.0
        seen_drift |= p["drift_sigma_mps"] > 0.0
        seen_pitch |= p["pitch_scale"] > 0.0
    assert seen_false and seen_drift and seen_pitch
    env.reset(10_000, "follow")
    for _ in range(40):
        env.step(0.0, 50.0)
    assert env.drone.pitch != 0.0 or env.drone.dvx != 0.0


# ---------------------------------------------------------------- lag-test dynamics
def test_per_axis_dead_time():
    from flyfollow.sim.drone_model import DroneModel, DroneParams

    d = DroneModel(DroneParams(yaw_dead_s=0.18, fwd_dead_s=0.47, tau_yaw_s=0.02, tau_fwd_s=0.45))
    d.reset(0.0, 0.0, 1.0, 0.0)
    t_yaw = t_fwd = None
    for i in range(1, 40):
        d.step(0.0, 30.0, 0.0, 30.0)
        if t_yaw is None and abs(d.yaw_rate_dps) > 1.0:
            t_yaw = i * DT
        if t_fwd is None and d.v_forward() > 0.02:
            t_fwd = i * DT
    assert 0.15 <= t_yaw <= 0.3 and 0.45 <= t_fwd <= 0.65
    assert abs(d.yaw_rate_dps - 55.0 * 0.3) < 0.5  # 16.5 deg/s at stick 30, as measured


def test_demo_profile_is_lag_test_medians(env):
    env.reset(10_000, "follow", "demo")
    p = env.params
    assert (p["yaw_gain_dps"], p["fwd_gain_mps"], p["fwd_dead_s"], p["yaw_dead_s"], p["fx_px"]) == (55.0, 0.96, 0.47, 0.18, 921.0)
    assert env.st.max_fwd_stick == 60.0 and abs(env.v_cap - 0.9 * 0.96 * 0.6) < 1e-9
    for s in range(10_000, 10_020):
        env.reset(s, "follow", "train")
        assert 40.0 <= env.st.max_fwd_stick <= 80.0
        assert abs(env.v_cap - min(1.4, 0.9 * env.params["fwd_gain_mps"] * env.st.max_fwd_stick / 100.0)) < 1e-9


# ---------------------------------------------------------------- obstacle scenarios
def _obstacle_env(obstacles, person_x: float = 6.0, avoid: bool = True, seed: int = 5_000, occlusion: bool = False):
    """Obstacles profile, drone at the origin facing +x, person parked person_x ahead, given footprints only.
    Occlusion is off unless asked for, so the user stays tracked behind the test obstacle."""
    from flyfollow.rl.env import load_env_config

    cfg = load_env_config()
    cfg["obstacles"]["avoid"] = avoid
    cfg["obstacles"]["occlusion"] = occlusion
    env = PursuitEnv(cfg)
    env.reset(seed, "follow", "obstacles")
    env.walker.xs = [person_x] * len(env.walker.xs)
    env.walker.ys = [0.0] * len(env.walker.ys)
    env.walker.step_i0 = env.walker.step_i1 = -1
    env.tx, env.ty = person_x, 0.0
    env.obstacles = list(obstacles)
    return env


def _tall_box(x: float, y: float, hx: float = 0.2, hy: float = 0.2, h: float = 2.5):
    from flyfollow.sim.obstacles import Obstacle

    return Obstacle(x, y, hx, hy, 0.0, h, "shelf")


def test_obstacles_off_by_default_and_same_world_as_train(env):
    env.reset(5_000, "follow", "train")
    w_train = env.world()
    res = run_episode(env, PIDController(), 5_000, "follow", "train")
    assert "obstacles" not in w_train and "obstacle_collision" not in res.metrics
    env.reset(5_000, "follow", "obstacles")
    w_obs = env.world()
    assert len(w_obs["obstacles"]) >= 1
    assert w_obs["params"] == w_train["params"] and w_obs["camera_drop"] == w_train["camera_drop"]
    assert w_obs["walker_xs"][:20] == w_train["walker_xs"][:20]  # detours never touch the start
    assert env.world()["obstacles"] == w_obs["obstacles"]  # deterministic placement


def test_obstacle_geometry():
    o = _tall_box(2.0, 0.0, 0.5, 0.25)
    assert o.dist(2.0, 0.0) == 0.0 and abs(o.dist(3.0, 0.0) - 0.5) < 1e-12 and abs(o.dist(2.0, 1.25) - 1.0) < 1e-12
    iv = o.segment_interval(0.0, 0.0, 4.0, 0.0)
    assert iv is not None and abs(iv[0] - 0.375) < 1e-12 and abs(iv[1] - 0.625) < 1e-12
    assert o.segment_interval(0.0, 1.0, 4.0, 1.0) is None
    assert o.overlaps_rect(1.25, 0.0, 1.0, 0.0, 0.5, 0.25) and not o.overlaps_rect(0.5, 0.0, 1.0, 0.0, 0.5, 0.25)


def test_user_path_keeps_clearance_from_obstacles(env):
    clear = env.cfg["obstacles"]["person_clear_m"]
    for s in range(5_000, 5_030):
        env.reset(s, "follow", "obstacles")
        for o in env.obstacles:
            assert min(o.dist(x, y) for x, y in zip(env.walker.xs, env.walker.ys)) >= clear - 1e-9
            assert o.dist(0.0, 0.0) >= env.cfg["obstacles"]["start_clear_m"] - 1e-9


def test_avoidance_brakes_and_never_touches_yaw():
    blocked = _obstacle_env([_tall_box(0.9, 0.0)])
    free = _obstacle_env([])
    for _ in range(12):
        blocked.step(30.0, 50.0)
        free.step(30.0, 50.0)
    assert blocked.braking and blocked.metrics()["blocked_s"] > 0.0
    assert blocked.drone.psi == free.drone.psi and blocked.drone.yaw_rate_dps == free.drone.yaw_rate_dps
    assert blocked.drone.v_forward() < 0.2 * free.drone.v_forward()


def test_avoidance_sidesteps_away_from_obstacle():
    env = _obstacle_env([_tall_box(0.8, 0.12)])  # slightly to the LEFT (+y) -> sidestep right (lr > 0)
    lrs = []
    for _ in range(80):
        _, _, done, info = env.step(0.0, 50.0)
        lrs.append(info["lr"])
        assert not done
    m = env.metrics()
    assert m["sidesteps"] >= 1 and max(lrs) > 0 and min(lrs) >= 0
    first = next(k for k, v in enumerate(lrs) if v != 0)
    assert first * DT >= env.cfg["obstacles"]["sidestep_after_s"] - DT
    assert env.drone.y < -0.2  # moved right (level frame y is left)


def test_sidestep_never_ends_closer_than_z_min():
    env = _obstacle_env([_tall_box(0.8, 0.12)], person_x=1.4)
    st = env.st
    # user estimated 1.4 m ahead and 0.4 m to the right: a 0.5 m sidestep right ends about 1.4 m away (ok if z_min small)
    b = math.atan2(0.4, 1.4)
    h = st.fy * st.target_size_m / math.hypot(1.4, 0.4)
    env._box = BoxState(True, cx=st.cx0 + st.fx * math.tan(b), cy=360.0, h=h)
    st.z_min_m = 1.45
    assert not env._sidestep_safe(+1)  # toward the user's side: vetoed
    assert env._sidestep_safe(-1)
    env._box = BoxState(False)
    assert env._sidestep_safe(+1)


def test_no_avoidance_layer_when_disabled():
    env = _obstacle_env([_tall_box(1.2, 0.0)], avoid=False)
    for _ in range(120):
        _, _, done, info = env.step(0.0, 50.0)
        if done:
            break
    assert env.metrics()["blocked_s"] == 0.0 and env.metrics()["sidesteps"] == 0.0
    assert env.obstacle_collision and env.collided and done


def test_occlusion_hides_the_box():
    tall = _obstacle_env([_tall_box(3.0, 0.0, 0.3, 0.3, h=2.6)], occlusion=True)
    short = _obstacle_env([_tall_box(3.0, 0.0, 0.3, 0.3, h=0.8)], occlusion=True)
    for e in (tall, short):
        for _ in range(60):
            e.step(0.0, 0.0)
    assert tall.occluded and tall.metrics()["frac_occluded"] > 0.9 and not tall._box.valid
    assert not short.occluded and short.metrics()["frac_occluded"] == 0.0 and short._box.valid
    assert tall.metrics()["frac_in_view"] < 0.1 and short.metrics()["frac_in_view"] > 0.9


def test_obstacle_collision_ends_episode_like_person_collision():
    env = _obstacle_env([_tall_box(3.0, 0.0)])
    env.drone.x, env.drone.y = 3.0, 0.25  # 0.05 m from the footprint, below its top
    _, _, done, info = env.step(0.0, 0.0)
    res = env.result()
    assert done and info["collision"]
    assert res.metrics["obstacle_collision"] == 1.0 and res.metrics["collided"] == 1.0
    assert res.terms["ev_collision"] == env.rcfg.ev_collision and res.terms["ev_tail"] < 0.0
    assert res.metrics["min_obstacle_dist"] < 0.12


def test_obstacle_metrics_and_bulk_episodes(env):
    keys = ("obstacle_collision", "min_obstacle_dist", "blocked_s", "sidesteps", "frac_occluded",
            "frac_in_view", "rms_bearing_err_deg", "loss_events_per_min", "frac_in_band")
    for s in range(5_100, 5_120):
        res = run_episode(env, Dummy() if s % 2 else PIDController(), s, "follow", "obstacles")
        assert np.isfinite(res.ret) and all(k in res.metrics for k in keys)
    res = run_episode(env, PIDController(), 5_100, "approach", "obstacles")  # approach: no obstacles
    assert "obstacle_collision" not in res.metrics
