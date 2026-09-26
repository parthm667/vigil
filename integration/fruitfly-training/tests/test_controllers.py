"""Controllers, encoder, decoder, parameter spaces and calibration (plan 4.1, 4.6, 4.7)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from flyfollow.interfaces import ARMS, OUTPUT_GROUPS, BoxState, Settings, TargetFeatures, brains_dir
from flyfollow.pilot.pursuit_decoder import PursuitDecoder
from flyfollow.rl.params import param_space
from flyfollow.senses.target import CH_AROUSAL, TargetEncoder

ST = Settings(kind="follow", z_ref_m=2.0, target_size_m=0.23)
REAL_BRAINS = brains_dir()  # captured before the synthetic fixture redirects FLYFOLLOW_DATA


def _params(arm: str = "FLY-HAND") -> dict:
    ps = param_space(arm)
    return ps.decode(ps.default_x())


def _box(theta_deg: float, s: float = 1.0) -> BoxState:
    cx = ST.cx0 + ST.fx * math.tan(math.radians(theta_deg))
    return BoxState(True, cx=cx, cy=ST.cy0, h=s * ST.h_ref_px)


@pytest.fixture(scope="module")
def syn_data(tmp_path_factory):
    """Synthetic brain + calibration files in an isolated data root."""
    from flyfollow.brain.synthetic import make_synthetic_pursuit_brain
    from flyfollow.rl.calibrate import calibrate_fly, calibrate_nobrain, write_pid
    from flyfollow.rl.params import controllers_config

    root = tmp_path_factory.mktemp("data")
    mp = pytest.MonkeyPatch()
    mp.setenv("FLYFOLLOW_DATA", str(root))
    path = make_synthetic_pursuit_brain(root / "brains" / "pursuit_coreS.npz")
    make_synthetic_pursuit_brain(root / "brains" / "pursuit_coreS_shuf1.npz", seed=1)  # stands in for a shuffle
    cfg = controllers_config({"calibration": {"repeats": 2, "measure_ms": 300, "settle_ms": 100}})
    res = calibrate_fly(path, cfg)
    calibrate_fly(root / "brains" / "pursuit_coreS_shuf1.npz", cfg)
    calibrate_nobrain(cfg)
    write_pid(cfg)
    yield path, res
    mp.undo()


# ---------------------------------------------------------------- parameter spaces
@pytest.mark.parametrize(
    "arm,dim",
    [("FLY-HAND", 47), ("FLY-CMA", 47), ("FLY-SHUF", 47), ("NOBRAIN", 47), ("PID-HAND", 4), ("PID-CMA", 4)]
    + [("FLY-YAW-HAND", 34), ("FLY-YAW", 34), ("FLY-SHUF-YAW", 34), ("NOBRAIN-YAW", 34)],
)
def test_param_dims(arm, dim):
    ps = param_space(arm)
    assert ps.dim == dim and len(set(ps.names)) == dim
    assert np.all(ps.lo < ps.hi)


@pytest.mark.parametrize("arm", ARMS)
def test_param_round_trip(arm):
    ps = param_space(arm)
    x = np.random.default_rng(0).random(ps.dim)
    np.testing.assert_allclose(ps.encode(ps.decode(x)), x, atol=1e-9)
    d = ps.decode(np.full(ps.dim, 2.0))  # out of range is clipped
    assert all(lo <= d[n] <= hi for n, lo, hi in zip(ps.names, ps.lo, ps.hi))
    assert ps.encode(ps.decode(ps.default_x())).shape == (ps.dim,)


def test_readout_bounds_match_plan():
    d = dict(zip(param_space("FLY-CMA").names, zip(param_space("FLY-CMA").lo, param_space("FLY-CMA").hi)))
    assert d["dec_tau_yaw_ms"][1] <= 150 and d["dec_tau_fwd_ms"][1] <= 150
    assert d["dec_g_yaw"] == (0.0, 1.0) and d["dec_g_fwd"] == (0.0, 1.0)
    assert d["enc_bin_gain_L0"] == (0.5, 2.0)


# ---------------------------------------------------------------- encoder
def test_encoder_sign_and_no_target():
    enc = TargetEncoder(_params(), arousal_mode="rate")
    right = enc.channels(TargetFeatures(True, math.radians(15), 1.0, 0.0, 0.0))
    left = enc.channels(TargetFeatures(True, math.radians(-15), 1.0, 0.0, 0.0))
    assert right[8:16].sum() > 5 * right[:8].sum()  # target right drives right LC10a
    assert left[:8].sum() > 5 * left[8:16].sum()
    assert right[16 + 1] > right[16] and right[18 + 1] > right[18]  # LC9_R > LC9_L, LC11_R > LC11_L
    assert right[16:20].max() <= 10.0  # LC9/LC11 capped (audit: stronger drive swamps LC10a)
    none = enc.channels(TargetFeatures(False, 0.0, 0.0, 0.0, 0.0))
    assert np.all(none[:CH_AROUSAL] == 0) and np.all(none[CH_AROUSAL:] > 0) and none.max() <= 10.0
    gain = TargetEncoder(_params(), arousal_mode="gain").channels(TargetFeatures(False, 0.0, 0.0, 0.0, 0.0))
    assert np.all(gain == 0)  # default p1_tonic_hz is 0


def test_encoder_size_widens_spot():
    enc = TargetEncoder(_params(), arousal_mode="rate")
    near = enc.channels(TargetFeatures(True, 0.0, 2.0, 0.0, 0.0))[:16]
    far = enc.channels(TargetFeatures(True, 0.0, 0.5, 0.0, 0.0))[:16]
    assert (near > 1.0).sum() > (far > 1.0).sum() and near.sum() > far.sum()


# ---------------------------------------------------------------- decoder
def test_decoder_bypass_and_signs():
    p = _params()
    p.update({f"dec_w_yaw_{t}": 1.0 for t in ("DNa02", "DNa01", "DNb05", "DNg13", "DNb06")})
    dec = PursuitDecoder(p, np.full(10, 20.0))
    r = np.full(10, 20.0)
    r[1::2] += 10.0  # right side more active
    y1 = dec.update(r, ST, 0.05)
    dec2 = PursuitDecoder(p, np.full(10, 20.0))
    assert dec2.update(r.copy(), ST, 0.05) == y1  # same rates, same output: nothing else enters
    assert y1[0] > 0
    assert abs(y1[0]) <= ST.max_yaw_stick and -ST.max_back_stick <= y1[1] <= ST.max_fwd_stick


def test_fly_controller_depends_only_on_dn_rates(syn_data):
    from flyfollow.rl.controllers import make_controller

    path, _ = syn_data
    outs = []
    for theta in (-20.0, 20.0):
        c = make_controller("FLY-HAND", None, str(path))
        c.reset(ST, 0)
        c.brain.tick_channels = lambda ch, ms=50.0: np.full(10, 30.0)  # brain output fixed
        outs.append([c.act(_box(theta), ST, 0.05) for _ in range(5)])
    assert outs[0] == outs[1]


# ---------------------------------------------------------------- brain-backed controllers
def test_fly_controller_steers_on_synthetic(syn_data):
    from flyfollow.rl.controllers import make_controller

    path, res = syn_data
    assert res["r2"]["yaw_init"] > 0.3
    c = make_controller("FLY-HAND", None, str(path))
    c.reset(ST, 3)
    c.warmup(0.5)
    assert c.brain_s > 0
    yaw_r = np.mean([c.act(_box(20.0), ST, 0.05)[0] for _ in range(20)])
    yaw_l = np.mean([c.act(_box(-20.0), ST, 0.05)[0] for _ in range(20)])
    assert yaw_r > 5 and yaw_l < -5


def test_lesion_clamps_dn_rates(syn_data):
    from flyfollow.rl.controllers import make_controller

    path, _ = syn_data
    c = make_controller("FLY-HAND", None, str(path))
    c.reset(ST, 1)
    c.warmup(0.2)
    for _ in range(10):
        c.act(_box(20.0), ST, 0.05)
    means = c.decoder.channel_means()
    assert set(means) == set(OUTPUT_GROUPS)
    audit = c.decoder.bias_audit()
    assert audit["n_ticks"] == 10 and audit["yaw_drive_abs"] > 0
    les = make_controller("FLY-HAND", None, str(path), lesion=means)
    les.reset(ST, 1)
    les.warmup(0.2)
    out_r = [les.act(_box(20.0), ST, 0.05) for _ in range(5)]
    out_l = [les.act(_box(-20.0), ST, 0.05) for _ in range(5)]
    assert len({o for o in out_r + out_l}) == 1  # every DN clamped: target no longer matters


def test_visualization_state(syn_data):
    from flyfollow.interfaces import INPUT_GROUPS
    from flyfollow.rl.controllers import make_controller

    path, _ = syn_data
    c = make_controller("FLY-HAND", None, str(path))
    c.reset(ST, 2)
    c.warmup(0.1)
    out = c.act(_box(20.0), ST, 0.05)
    assert c.last_sticks == out
    assert c.last_counts.shape == (c.connectome.n,) and c.last_counts.sum() > 0
    ir = c.last_input_rates
    assert set(ir) == set(INPUT_GROUPS)
    assert all(ir[g].shape == c.groups[g].shape for g in INPUT_GROUPS)
    assert ir["LC10a_R"].sum() > ir["LC10a_L"].sum()
    assert set(c.last_dn_rates) == set(OUTPUT_GROUPS)
    assert Path(c.brain_path).name == Path(path).name


def test_nobrain_and_pid_factory(syn_data):
    from flyfollow.rl.controllers import init_x, make_controller

    nb = make_controller("NOBRAIN", None)
    nb.reset(ST, 0)
    nb.warmup(0.2)
    assert nb.act(_box(20.0), ST, 0.05)[0] > nb.act(_box(-20.0), ST, 0.05)[0]
    pytest.importorskip("flyfollow.pilot.pid")
    pid = make_controller("PID-HAND", None)
    assert pid.name == "PID-HAND"
    np.testing.assert_allclose(param_space("PID-HAND").decode(init_x("PID-HAND"))["Kp_y"], 100.0)


def test_fly_shuf_requires_brain_path(syn_data):
    from flyfollow.rl.controllers import make_controller

    with pytest.raises(ValueError):
        make_controller("FLY-SHUF", None)


def test_empty_output_group_refused(tmp_path):
    from flydrones.brain.connectome import Connectome

    from flyfollow.brain.synthetic import make_synthetic_pursuit_brain
    from flyfollow.pilot.fly_brain import FlyBrain

    p = make_synthetic_pursuit_brain(tmp_path / "b.npz")
    c = Connectome.load(p)
    c.groups["DNg13_L"] = np.zeros(0, np.int64)
    c.save(tmp_path / "bad.npz")
    with pytest.raises(ValueError, match="DNg13_L"):
        FlyBrain(tmp_path / "bad.npz")
    c = Connectome.load(p)
    c.groups["AROUSAL_L"] = c.groups["AROUSAL_R"] = np.zeros(0, np.int64)
    c.save(tmp_path / "noarousal.npz")
    assert not FlyBrain(tmp_path / "noarousal.npz").has_arousal


# ---------------------------------------------------------------- yaw-only arms (plan 4.3 fallback)
def test_yaw_only_space_is_full_minus_forward():
    from flyfollow.rl.params import base_arm, is_forward_only

    for arm in ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "FLY-YAW-HAND"):
        full = param_space(base_arm(arm))
        assert param_space(arm).names == tuple(n for n in full.names if not is_forward_only(n))
        assert full.dim - param_space(arm).dim == 13


def _yaw_path(arm: str, path: Path) -> str | None:
    if arm == "NOBRAIN-YAW":
        return None
    return str(path.with_name("pursuit_coreS_shuf1.npz")) if arm == "FLY-SHUF-YAW" else str(path)


@pytest.mark.parametrize("arm", ["FLY-YAW-HAND", "FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW"])
def test_yaw_only_wrapper(arm, syn_data):
    pid_mod = pytest.importorskip("flyfollow.pilot.pid")
    from flyfollow.rl.controllers import init_x, make_controller
    from flyfollow.rl.params import base_arm

    path, _ = syn_data
    bp = _yaw_path(arm, path)
    base = base_arm(arm)
    full, sub = param_space(base), param_space(arm)
    ix = [full.names.index(n) for n in sub.names]
    np.testing.assert_allclose(init_x(arm, bp), init_x(base, bp)[ix])  # projection of the base init
    c = make_controller(arm, None, bp)
    assert c.name == arm and c.decoder is c.inner.decoder
    pid = pid_mod.PIDController({"Kp_y": 100.0, "Kd_y": 20.0, "Kp_f": 40.0, "d0": 0.15})
    c.reset(ST, 0)
    c.warmup(0.2)
    for theta, s, z_ref, vfwd in [(20.0, 0.5, 2.0, 35.0), (-10.0, 1.0, 1.5, 20.0), (5.0, 2.0, 2.5, 50.0), (0.0, 1.05, 2.0, 35.0)]:
        st = Settings(kind="follow", z_ref_m=z_ref, target_size_m=0.23, max_fwd_stick=vfwd)
        box = _box(theta, s)
        box.vcx = 30.0
        yaw, fb = c.act(box, st, 0.05)
        assert fb == pid.act(box, st, 0.05)[1]  # forward is exactly PID-HAND's range loop
        assert yaw == c.inner.last_sticks[0]  # yaw is the inner controller's
        assert c.last_sticks == (yaw, fb)
    if arm != "NOBRAIN-YAW":
        assert c.last_counts.shape == (c.connectome.n,) and set(c.last_dn_rates) == set(OUTPUT_GROUPS)
        assert c.brain_path.name == Path(bp).name and set(c.last_input_rates) >= {"LC10a_L", "LC10a_R"}


def test_yaw_only_x_sets_yaw_readout(syn_data):
    from flyfollow.rl.controllers import init_x, make_controller

    path, _ = syn_data
    ps = param_space("FLY-YAW")
    x = init_x("FLY-YAW", str(path))
    x[ps.names.index("dec_g_yaw")] = 0.0  # zero yaw gain: yaw must vanish, forward stays PID
    c = make_controller("FLY-YAW", x, str(path))
    c.reset(ST, 0)
    assert c.act(_box(20.0, 0.5), ST, 0.05)[0] == 0.0 and c.last_sticks[1] > 0
    assert c.inner.params["dec_b_fwd"] == pytest.approx(param_space("FLY-CMA").decode(init_x("FLY-CMA", str(path)))["dec_b_fwd"])


def test_yaw_only_lesion(syn_data):
    from flyfollow.rl.controllers import make_controller

    path, _ = syn_data
    c = make_controller("FLY-YAW", None, str(path))
    c.reset(ST, 1)
    c.warmup(0.2)
    for _ in range(10):
        c.act(_box(20.0), ST, 0.05)
    assert c.decoder.bias_audit()["yaw_drive_abs"] > 0
    les = make_controller("FLY-YAW", None, str(path), lesion=c.decoder.channel_means())
    les.reset(ST, 1)
    les.warmup(0.2)
    yaws = {les.act(_box(th), ST, 0.05)[0] for th in (20.0, -20.0, 20.0, -20.0)}
    assert len(yaws) == 1


def test_fly_shuf_yaw_requires_brain_path(syn_data):
    from flyfollow.rl.controllers import make_controller

    with pytest.raises(ValueError):
        make_controller("FLY-SHUF-YAW", None)


# ---------------------------------------------------------------- full episodes in A's env
def _env_or_skip():
    pytest.importorskip("flyfollow.rl.env")
    pytest.importorskip("flyfollow.rl.rollout")
    from flyfollow.rl.env import PursuitEnv

    return PursuitEnv()


@pytest.mark.parametrize("arm", ["FLY-HAND", "NOBRAIN", "PID-HAND", "FLY-YAW-HAND", "FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW"])
def test_full_episode(arm, syn_data):
    from flyfollow.rl.controllers import make_controller
    from flyfollow.rl.rollout import run_episode

    env = _env_or_skip()
    path, _ = syn_data
    bp = _yaw_path(arm, path) if "YAW" in arm else (str(path) if arm.startswith("FLY") else None)
    c = make_controller(arm, None, bp)
    res = run_episode(env, c, 1000, "follow")
    assert res.n_ticks > 100 and np.isfinite(res.ret)
    if arm.startswith("FLY"):
        assert c.brain_s > 0


def test_real_core1_episode():
    """FLY-HAND on the real pursuit subgraph with its own calibration file (skips until both exist)."""
    from flyfollow.rl.controllers import calibration_path, make_controller
    from flyfollow.rl.rollout import run_episode

    env = _env_or_skip()
    core = REAL_BRAINS / "pursuit_core1.npz"
    mp = pytest.MonkeyPatch()
    mp.setenv("FLYFOLLOW_DATA", str(REAL_BRAINS.parent))
    try:
        if not core.exists() or not calibration_path("FLY-HAND", core).exists():
            pytest.skip("pursuit_core1.npz or its calibration missing")
        res = run_episode(env, make_controller("FLY-HAND", None, str(core)), 1000, "follow")
    finally:
        mp.undo()
    assert res.n_ticks > 100 and np.isfinite(res.ret)


def test_init_files_consistent():
    """Every written init file decodes to its own x (catches bounds edited after calibration)."""
    d = REAL_BRAINS / "init"
    files = sorted(Path(d).glob("*.json")) if d.exists() else []
    if not files:
        pytest.skip("no calibration files")
    import json

    for f in files:
        rec = json.loads(f.read_text())
        ps = param_space(rec["arm"])
        np.testing.assert_allclose(ps.encode(rec["params"]), rec["x"], atol=1e-6, err_msg=f.name)
