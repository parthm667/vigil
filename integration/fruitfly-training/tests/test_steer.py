"""flyfollow.steer.FlySteer: the fly as a yaw-only controller for another stack (ReachGlass integration).

Fast: every test runs a few seconds of simulated time on the real pursuit_core1 brain (about 2 ms per tick).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from flyfollow.interfaces import brains_dir

CORE = brains_dir() / "pursuit_core1.npz"
pytestmark = pytest.mark.skipif(not CORE.exists() or not (brains_dir() / "init" / "FLY-HAND__pursuit_core1.json").exists(),
                                reason="needs data/brains/pursuit_core1.npz and its hand calibration (scripts/setup_data.sh)")


@pytest.fixture(scope="module")
def steer():
    from flyfollow.steer import FlySteer

    return FlySteer(z_ref_m=1.0)


def hold(fs, bearing_deg: float | None, seconds: float, t0: float = 0.0, dt: float = 1.0 / 15, range_m: float = 1.0):
    """Drive FlySteer at a host loop rate of 1/dt with a fixed target bearing (None: no detection)."""
    ys, t = [], t0
    for _ in range(round(seconds / dt)):
        t += dt
        b = None if bearing_deg is None else math.radians(bearing_deg)
        ys.append(fs.yaw(t, b, range_m=None if b is None else range_m, t_frame=t))
    return np.asarray(ys, float), t


def test_target_right_turns_right_and_left_turns_left(steer):
    steer.reset()
    right, _ = hold(steer, 15.0, 3.0)
    steer.reset()
    left, _ = hold(steer, -15.0, 3.0)
    assert right[-20:].mean() > 10, right[-20:]
    assert left[-20:].mean() < -10, left[-20:]
    assert all(isinstance(v, int) for v in (steer.yaw(100.0), steer.stats["yaw"]))


def test_box_input_uses_host_camera_and_image_size(steer):
    # a box right of center in a 480 x 360 frame (their simulator) and in the 960 x 720 Tello frame
    for size, box in (((480, 360), (330.0, 150.0, 370.0, 200.0)), ((960, 720), (660.0, 300.0, 740.0, 400.0))):
        steer.reset()
        ys, t = [], 0.0
        for _ in range(45):
            t += 1.0 / 15
            ys.append(steer.yaw(t, box=box, t_frame=t, image_size=size))
        assert np.mean(ys[-15:]) > 10, (size, ys[-15:])


def test_no_detection_never_crashes_and_yaw_decays(steer):
    steer.reset()
    y0 = steer.yaw(0.01)  # nothing ever seen
    assert isinstance(y0, int) and abs(y0) <= steer.max_yaw
    seen, t = hold(steer, 20.0, 2.0)
    assert seen[-10:].mean() > 10 and steer.target_valid
    lost, t = hold(steer, None, 2.0, t0=t)
    assert not steer.target_valid
    assert abs(lost[-10:]).max() <= 3, lost  # DNs relax: the readout's bias cancels the core's rest asymmetry
    assert abs(lost[-10:]).mean() < 0.3 * abs(seen[-10:].mean())


def test_params_hand_vs_best_json(tmp_path):
    from flyfollow.rl.controllers import init_x
    from flyfollow.rl.params import param_space
    from flyfollow.steer import FlySteer, load_params

    assert load_params(None)[:2] == (None, "FLY-YAW-HAND")
    assert load_params("none")[0] is None
    ps = param_space("FLY-YAW")
    p = ps.decode(init_x("FLY-YAW"))
    p["dec_b_yaw"] = 4.0  # a readout that always turns right hard: proves the file is what steers
    f_x = tmp_path / "best_x.json"
    f_x.write_text(json.dumps({"arm": "FLY-YAW", "x": ps.encode(p).tolist(), "brain": "pursuit_core1.npz", "gen": 3}))
    f_p = tmp_path / "best_params.json"
    f_p.write_text(json.dumps({"arm": "FLY-YAW", "params": p}))
    for f in (f_x, f_p):
        fs = FlySteer(params_path=f, z_ref_m=1.0)
        assert fs.stats["arm"] == "FLY-YAW" and fs.stats["params"] == str(f)
        ys, _ = hold(fs, -15.0, 2.0)  # target LEFT, but the bias wins
        assert ys[-10:].mean() > 30, ys[-10:]
    hand = FlySteer(z_ref_m=1.0)
    assert hand.stats["arm"] == "FLY-YAW-HAND" and hand.stats["params"] == "hand"
    ys, _ = hold(hand, -15.0, 2.0)
    assert ys[-10:].mean() < -10
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"arm": "FLY-YAW"}))
    with pytest.raises(ValueError):
        FlySteer(params_path=bad)


def test_timing_budget_and_never_catches_up(steer):
    steer.reset()
    hold(steer, 10.0, 10.0, dt=0.05)  # 200 ticks at the trained 20 Hz
    st = steer.stats
    assert st["ticks"] >= 195
    # about 2 ms mean on an idle M-series laptop; p95 bound = the runtime's brain limit (plan 3.5, 45 ms of the
    # 50 ms tick), so the test holds on a heavily loaded machine too
    assert st["tick_ms_mean"] < 15.0 and st["tick_ms_p95"] < 45.0, st
    n = st["ticks"]
    steer.yaw(10.0 + 30.0, math.radians(10.0))  # host stalled 30 s: at most 3 ticks, then resync
    assert steer.stats["ticks"] - n <= 3 and steer.stats["resyncs"] >= 1
    steer.yaw(5.0)  # clock went backwards (new run): starts over instead of stalling
    assert steer.stats["ticks"] <= 1


def test_smoothing_defaults_cut_jitter_and_keep_the_turn():
    from flyfollow.interfaces import brains_dir
    from flyfollow.steer import FlySteer

    trained = brains_dir() / "trained" / "FLY-YAW_best.json"
    for pp, need in ((None, 0.9), (trained, 0.7)):
        if pp is not None and not pp.exists():
            continue
        raw = FlySteer(params_path=pp, z_ref_m=1.0, smoothing_ms=0, deadband=0, slew=0, hysteresis=0)
        smooth = FlySteer(params_path=pp, z_ref_m=1.0)  # defaults: deadband + hysteresis + slew cap on
        assert smooth.deadband > 0 and smooth.hysteresis > 0 and smooth.slew > 0
        out = {}
        for name, fs in (("raw", raw), ("smooth", smooth)):
            jit, mean = [], []
            for seed in range(3):
                fs.reset(seed)
                ys, _ = hold(fs, 6.0, 4.0)
                jit.append(np.mean(np.abs(np.diff(ys[20:]))))
                mean.append(ys[20:].mean())
            out[name] = (np.mean(jit), np.mean(mean))
        assert out["smooth"][0] < need * out["raw"][0], (pp, out)  # less stick change per loop
        assert out["smooth"][1] > 5 and out["raw"][1] > 5, (pp, out)  # still turning right onto the target


def test_slew_and_deadband_limits():
    from flyfollow.steer import FlySteer

    fs = FlySteer(z_ref_m=1.0, smoothing_ms=0, deadband=0, hysteresis=0, slew=40.0)  # 2 stick per 50 ms tick
    ys, _ = hold(fs, 20.0, 2.0, dt=0.05)
    assert np.max(np.abs(np.diff(ys))) <= 3, ys  # 2 per tick, +1 for integer rounding
    lp = FlySteer(z_ref_m=1.0, smoothing_ms=250.0, deadband=0, hysteresis=0, slew=0)
    ys, _ = hold(lp, 20.0, 0.5, dt=0.05)
    assert abs(ys[2]) < abs(ys[-1])  # low-pass: rises gradually
    hy = FlySteer(z_ref_m=1.0, deadband=0, hysteresis=5.0, slew=0)
    ys, _ = hold(hy, 6.0, 3.0, dt=0.05)
    steps = np.abs(np.diff(ys))
    assert np.all((steps == 0) | (steps >= 5)), ys  # the sent stick only moves in steps of at least the band
    dead = FlySteer(z_ref_m=1.0, deadband=100.0)
    ys, _ = hold(dead, 20.0, 2.0)
    assert not np.any(ys), ys


def test_trained_params_copy_loads_and_is_centred():
    from flyfollow.interfaces import brains_dir
    from flyfollow.steer import FlySteer

    paths = [brains_dir() / "trained" / f"FLY-YAW{v}_best.json" for v in ("", "_smooth", "_smooth2")]
    paths = [p for p in paths if p.exists()]
    if not paths:
        pytest.skip("no trained parameters in data/brains/trained")
    for path in paths:
        fs = FlySteer(params_path=path, z_ref_m=1.0)
        assert fs.arm == "FLY-YAW"
        centre, _ = hold(fs, 0.0, 3.0)
        assert abs(centre[15:].mean()) < 3, (path, centre)  # the hand calibration sits at about +11 here
        fs.reset()
        right, _ = hold(fs, 10.0, 3.0)
        assert right[15:].mean() > 8, (path, right)
        fs.reset()
        left, _ = hold(fs, -10.0, 3.0)
        assert left[15:].mean() < -8, (path, left)
