"""Fast tests for flyfollow.viz: pose mapping signs, animator frames, frames/centroid, sink never blocks."""

from __future__ import annotations

import math
import pickle
import socket
import time

import numpy as np
import pytest

from flyfollow.interfaces import OUTPUT_GROUPS, brains_dir
from flyfollow.viz.fly_body import BodyMapping, MotorState, PoseSmoother, flybody_available, pose_targets
from flyfollow.viz.frames import default_bin_centers_deg, frame_from_controller, lc10a_bin_rates, look_bearing, motor_state

CORE1 = brains_dir() / "pursuit_core1.npz"
needs_core1 = pytest.mark.skipif(not CORE1.exists(), reason="data/brains/pursuit_core1.npz missing (scripts/setup_data.sh)")
needs_flybody = pytest.mark.skipif(not flybody_available(), reason="flybody assets missing (scripts/setup_viz.sh)")


# --------------------------------------------------------------------------- pose mapping (pure)
def test_right_turn_gives_larger_left_stroke_and_right_bank():
    p = pose_targets(MotorState(dna02_l=0.0, dna02_r=300.0, yaw_stick=40.0, fb_stick=10.0, look_bearing=math.radians(20)))
    assert p.amp_l > p.amp_r  # right turn = larger LEFT stroke amplitude, as in real flies
    assert p.bank > 0  # right wing down
    assert p.body_yaw < 0  # clockwise from above (world z counterclockwise positive)
    assert p.head_yaw > 0  # head toward the target on the right


def test_left_turn_mirrors():
    r = pose_targets(MotorState(dna02_l=0.0, dna02_r=300.0, yaw_stick=40.0, look_bearing=math.radians(20)))
    left = pose_targets(MotorState(dna02_l=300.0, dna02_r=0.0, yaw_stick=-40.0, look_bearing=math.radians(-20)))
    assert left.amp_r > left.amp_l
    assert left.bank == pytest.approx(-r.bank)
    assert left.body_yaw == pytest.approx(-r.body_yaw)
    assert left.head_yaw == pytest.approx(-r.head_yaw)
    assert left.amp_l == pytest.approx(r.amp_r)


def test_symmetric_input_is_symmetric_and_head_limited():
    p = pose_targets(MotorState(dna02_l=100.0, dna02_r=100.0, look_bearing=None))
    assert p.amp_l == pytest.approx(p.amp_r) and p.bank == 0 and p.body_yaw == 0 and p.head_yaw == 0
    far = pose_targets(MotorState(look_bearing=math.radians(80)))
    assert far.head_yaw == pytest.approx(math.radians(BodyMapping().head_max_deg))


def test_smoother_converges_without_overshoot():
    tgt = pose_targets(MotorState(dna02_r=400.0, yaw_stick=60.0, look_bearing=0.3))
    sm = PoseSmoother(pose_targets(MotorState()))
    hist = [sm.step(tgt, 1 / 30).bank for _ in range(90)]
    assert hist[-1] == pytest.approx(tgt.bank, abs=1e-3)
    assert max(hist) <= tgt.bank + 1e-6  # critically damped


# --------------------------------------------------------------------------- frames and centroid
class _FakeController:
    name = "FAKE"
    brain_path = "fake.npz"

    def __init__(self):
        self.last_counts = np.arange(10, dtype=np.int32)
        self.last_channels = np.zeros(22, np.float32)
        self.last_channels[8 + 5] = 120.0  # right bin 5 (lateral right)
        self.last_dn_rates = {g: (200.0 if g == "DNa02_R" else 0.0) for g in OUTPUT_GROUPS}
        self.last_input_rates = {"LC10a_R": np.array([0.0, 120.0])}
        self.last_sticks = (30.0, 5.0)


def test_frame_from_controller_and_motor_state():
    fr = frame_from_controller(_FakeController(), t=1.5, tick=30, sticks=(25.0, 4.0))
    assert fr["t"] == 1.5 and fr["counts"].dtype == np.int32 and fr["dn"]["DNa02_R"] == 200.0
    assert fr["sticks"]["yaw"] == 25.0 and fr["sticks"]["yaw_raw"] == 30.0
    assert fr["inputs"]["LC10a_R"] == pytest.approx(60.0)
    pickle.dumps(fr)  # must cross the process boundary
    ms = motor_state(fr)  # no bins: falls back to encoder channels
    assert ms.dna02_r > ms.dna02_l and ms.yaw_stick == 25.0
    assert ms.look_bearing is not None and ms.look_bearing > 0


def test_readout_drive_overrides_raw_dna02_bias():
    """core1 fires DNa02 R more at rest; the readout's per-side normalization decides the turn, and so do the wings."""
    fr = {"t": 0.0, "dn": {"DNa02_L": 245.0, "DNa02_R": 363.0}, "sticks": {"yaw": -8.0}, "readout": {"yaw_drive": -0.5}}
    p = pose_targets(motor_state(fr))
    assert p.amp_r > p.amp_l and p.bank < 0  # a left turn, as the drone does
    del fr["readout"]
    p_raw = pose_targets(motor_state(fr))
    assert p_raw.amp_l > p_raw.amp_r  # without a readout the raw DNa02 asymmetry is used


def test_look_bearing_sign_and_quiet():
    c = default_bin_centers_deg()
    r = np.zeros(16)
    r[8 + 3] = 100.0  # right eye bin 3
    assert look_bearing(r, c) > 0
    l = np.zeros(16)
    l[3] = 100.0
    assert look_bearing(l, c) < 0
    assert look_bearing(np.zeros(16), c) is None


def test_bin_rates_from_counts():
    counts = np.zeros(40, np.int32)
    counts[[20, 21]] = 5
    bins = {"LC10a_L": [np.array([0, 1])] * 8, "LC10a_R": [np.array([20, 21])] + [np.array([30])] * 7}
    r = lc10a_bin_rates({"counts": counts}, bins, dt=0.05)
    assert r[8] == pytest.approx(100.0) and r[:8].sum() == 0


# --------------------------------------------------------------------------- sink never blocks
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sink_drops_rather_than_blocks():
    zmq = pytest.importorskip("zmq")
    from flyfollow.viz.live import VizSink

    addr = f"tcp://127.0.0.1:{_free_port()}"
    sink = VizSink(addr, hwm=2)
    payload = np.zeros(100_000, np.uint8)  # 100 kB per frame, well past any socket buffer after a few frames
    # 1) nobody listening
    times = []
    for i in range(200):
        a = time.perf_counter()
        sink.publish({"t": i * 0.05, "blob": payload})
        times.append(time.perf_counter() - a)
    # 2) a subscriber that never reads (a stalled viewer)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 2)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(addr)
    time.sleep(0.3)
    n = 1500
    for i in range(n):
        a = time.perf_counter()
        sink.publish({"t": i * 0.05, "blob": payload})
        times.append(time.perf_counter() - a)
    got = 0
    time.sleep(0.2)
    while True:
        try:
            sub.recv(zmq.NOBLOCK)
            got += 1
        except zmq.Again:
            break
    sub.close(0)
    sink.close(end=False)
    t = np.array(times[1:]) * 1000
    assert t.max() < 50.0, f"publish blocked for {t.max():.1f} ms"
    assert np.percentile(t, 99) < 5.0
    assert got < n // 2  # most frames were dropped, not queued without bound


def test_sink_refuses_non_loopback():
    pytest.importorskip("zmq")
    from flyfollow.viz.live import VizSink

    with pytest.raises(ValueError):
        VizSink("tcp://0.0.0.0:5999")


# --------------------------------------------------------------------------- data-backed checks
@needs_core1
def test_brain_geometry_orientation_and_pathway():
    from flyfollow.viz.brain_view import BrainGeometry

    geo = BrainGeometry.load(CORE1, n_context=2000)
    assert geo.n == 1446 and geo.context.shape == (2000, 3)
    # display frame: +X is the fly's right
    assert geo.pos[geo.groups["LC10a_R"], 0].mean() > 0 > geo.pos[geo.groups["LC10a_L"], 0].mean()
    e = geo.edges
    k = geo.key
    inh = (e.pre == k["AOTU019_L"]) & (e.post == k["DNa02_R"])
    exc = (e.pre == k["AOTU025_L"]) & (e.post == k["DNa02_L"])
    assert inh.any() and e.weight[inh][0] < 0  # AOTU019 L inhibits the opposite DNa02
    assert exc.any() and e.weight[exc][0] > 0  # AOTU025 L excites the same-side DNa02
    lc_to_019 = (e.post == k["AOTU019_R"]) & (e.stage == 0)
    assert np.isin(e.pre[lc_to_019], geo.groups["LC10a_R"]).all()


@needs_core1
def test_synthetic_spot_on_right_drives_dna02_r():
    from flyfollow.viz.frames import synthetic_frames

    frames = list(synthetic_frames(CORE1, seconds=1.5, seed=1, script="left_right"))  # first 2 s: spot at +25 deg
    late = frames[10:]
    r = np.mean([f["dn"]["DNa02_R"] for f in late])
    l = np.mean([f["dn"]["DNa02_L"] for f in late])
    assert r > l + 50
    assert np.mean([f["sticks"]["yaw"] for f in late]) > 0


@needs_flybody
def test_animator_frames_and_head_direction():
    from flyfollow.viz.fly_body import FlyBodyAnimator, RenderStyle

    try:
        anim = FlyBodyAnimator(RenderStyle(width=160, height=120, ghosts=1), BodyMapping(max_body_yaw_deg=0.0, max_bank_deg=0.0))
    except Exception as exc:  # noqa: BLE001 (no GL context on this machine)
        pytest.skip(f"offscreen rendering unavailable: {exc}")
    try:
        anim.set_motor(MotorState(dna02_l=0.0, dna02_r=300.0, yaw_stick=40.0, look_bearing=math.radians(20)))
        img = None
        for _ in range(20):
            img = anim.step(1 / 30)
        assert img.shape == (120, 160, 3) and img.dtype == np.uint8
        assert img.std() > 5  # not blank
        m, d = anim.model, anim.data
        er = d.cam_xpos[m.camera("eye_right").id]
        el = d.cam_xpos[m.camera("eye_left").id]
        head = d.xpos[m.body("head").id]
        fwd = 0.5 * (er + el) - head
        assert fwd[1] < 0  # head turned toward the fly's right (-y in the flybody frame)
        assert anim.pose.amp_l > anim.pose.amp_r
    finally:
        anim.close()
