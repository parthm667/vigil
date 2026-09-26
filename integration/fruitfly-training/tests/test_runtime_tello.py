"""Drone I/O (tello_io), the simulated Tello (sim_world) and the R0 tools, with no hardware.

A fake djitellopy Tello records every packet; the runtime bus is the in-process InProcBus.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import yaml

from flyfollow.runtime.bus import InProcBus, Publisher, Subscriber
from flyfollow.runtime.drone_common import RcArbiter, SafetyLimits, clamp_stick, parse_move, safety_reason
from flyfollow.runtime.messages import RC_TIMEOUT_S, msg, validate
from flyfollow.runtime.tello_io import TOPICS_IN, TelloIO, TelloLink

MOVES = ("forward", "back", "left", "right", "up", "down", "cw", "ccw")


class FakeTello:
    """The djitellopy.Tello methods TelloLink uses. Moves block for move_s like the real "ok" wait."""

    def __init__(self):
        self.lock = threading.Lock()
        self.log: list[tuple[float, str]] = []
        self.state = {"bat": 80, "temph": 60, "h": 0, "yaw": 0, "vgx": 0, "vgy": 0, "vgz": 0, "tof": 10}
        self.udp = {"responses": []}
        self.move_s = 0.05

    def connect(self, wait_for_state=True):
        pass

    def get_current_state(self):
        return self.state

    def get_own_udp_object(self):
        return self.udp

    def _rec(self, c):
        with self.lock:
            self.log.append((time.time(), c))

    def send_command_with_return(self, c, timeout=7):
        assert isinstance(timeout, int)  # djitellopy's enforce_types rejects floats
        self._rec(c)
        if c.split()[0] in MOVES:
            time.sleep(self.move_s)
        if c == "takeoff":
            self.state = dict(self.state, h=80)
        if c == "land":
            self.state = dict(self.state, h=0)
        return "ok"

    def send_command_without_return(self, c):
        self._rec(c)

    def send_rc_control(self, a, b, c, d):
        assert all(isinstance(v, int) and -100 <= v <= 100 for v in (a, b, c, d))
        self._rec(f"rc {a} {b} {c} {d}")

    def cmds(self) -> list[str]:
        with self.lock:
            return [c for _, c in self.log]


def make_io(send=True, video=False):
    bus = InProcBus()
    ft = FakeTello()
    io = TelloIO(TelloLink(ft), Publisher("tello_io", bus=bus), Subscriber(TOPICS_IN, bus=bus), send=send, video=video)
    out = Subscriber(["tello_ack", "rc_sent", "tello_state", "tello_event", "frame"], bus=bus)
    op = Publisher("operator", bus=bus)
    return ft, io, out, op


def run(io, sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        io.step()
        time.sleep(0.005)


def rc(op, src, mode, lr=0, fb=0, ud=0, yaw=0, **kw):
    op.publish(msg("rc", lr=lr, fb=fb, ud=ud, yaw=yaw, src=src, mode=mode, **kw))


def mode(op, to):
    op.publish(msg("mode", **{"from": "?", "to": to, "reason": "test"}))


def takeoff(ft, io, op):
    op.publish(msg("tello_cmd", id="to", cmd="takeoff"))
    run(io, 0.25)
    assert io.flying


# ------------------------------------------------------------------------------------------------ arbitration
def test_arbiter_owner_operator_timeout():
    a = RcArbiter()
    now = 1000.0
    m = lambda src, md, yaw, t=now: {"topic": "rc", "t": t, "lr": 0, "fb": 0, "ud": 0, "yaw": yaw, "src": src, "mode": md}
    assert not a.offer(m("controller", "IDLE", 10), now)  # IDLE has no owner
    a.set_mode("FOLLOW")
    assert a.offer(m("controller", "FOLLOW", 10), now)
    assert not a.offer(m("mission", "FOLLOW", 50), now)  # not the owner
    assert not a.offer(m("controller", "FIND", 50), now)  # mode field mismatch
    assert not a.offer(m("controller", "FOLLOW", 50, t=now - 1.0), now)  # stale on arrival
    assert a.select(now + 0.1) == ((0, 0, 0, 10), "controller")
    assert a.select(now + RC_TIMEOUT_S + 0.01) == ((0, 0, 0, 0), "hover")
    assert a.offer(m("operator", "whatever", -20, t=now + 1), now + 1)
    a.offer(m("controller", "FOLLOW", 30, t=now + 1), now + 1)
    assert a.select(now + 1.1) == ((0, 0, 0, -20), "operator")  # operator wins
    assert a.select(now + 1.6) == ((0, 0, 0, 0), "hover")
    a.set_mode("FIND")  # mission owns FIND; the controller's rc is dropped on the change
    a.offer(m("controller", "FIND", 30, t=now + 2), now + 2)
    assert a.select(now + 2.1)[1] == "hover"
    assert a.offer(m("mission", "FIND", 40, t=now + 2), now + 2)
    assert a.select(now + 2.1) == ((0, 0, 0, 40), "mission")


def test_clamp_and_parse_move():
    assert clamp_stick(130.4) == 100 and clamp_stick(-20.6) == -21 and clamp_stick(float("nan")) == 0
    assert clamp_stick(None) == 0
    assert parse_move({"direction": "forward", "value": 50}) == ("forward 50", "")
    assert parse_move({"direction": "cw", "value": 90.2}) == ("cw 90", "")
    assert parse_move({"direction": "forward", "value": 10})[0] is None
    assert parse_move({"direction": "sideways", "value": 50})[0] is None


def test_safety_rules():
    lim = SafetyLimits()
    kw = dict(flying=True, bat_pct=60, temph_c=70, video_enabled=True, video_age_s=0.1, bus_age_s=0.1, idle_s=1.0)
    assert safety_reason(lim, **kw) is None
    assert "battery" in safety_reason(lim, **{**kw, "bat_pct": 24})
    assert "temperature" in safety_reason(lim, **{**kw, "temph_c": 90})
    assert "video" in safety_reason(lim, **{**kw, "video_age_s": 5.5})
    assert "bus" in safety_reason(lim, **{**kw, "bus_age_s": 3.0})
    assert safety_reason(lim, **{**kw, "flying": False, "bat_pct": 5}) is None


# ------------------------------------------------------------------------------------------------ tello_io
def test_dry_run_never_sends():
    ft, io, out, op = make_io(send=False)
    io.start()
    try:
        op.publish(msg("tello_cmd", id="t", cmd="takeoff"))
        mode(op, "FOLLOW")
        for _ in range(5):
            rc(op, "controller", "FOLLOW", fb=40, yaw=20)
            run(io, 0.05)
        flying_after_takeoff = io.flying
        op.publish(msg("tello_cmd", id="m", cmd="move", args={"direction": "forward", "value": 50}))
        op.publish(msg("kill", action="emergency"))
        run(io, 0.3)
    finally:
        io.shutdown()
    sent = ft.cmds()
    assert not [c for c in sent if c.split()[0] in ("takeoff", "land", "emergency", "rc") + MOVES], sent
    ms = out.drain(100000)
    rcs = [m for m in ms if m["topic"] == "rc_sent"]
    assert rcs and all(m["dry_run"] for m in rcs)
    assert any(m["fb"] == 40 and m["yaw"] == 20 and m["src"] == "controller" for m in rcs)  # what WOULD be sent
    acks = {m["id"]: m for m in ms if m["topic"] == "tello_ack"}
    assert acks["t"]["ok"] and acks["t"]["dry_run"] and "dry run" in acks["t"]["detail"]
    st = [m for m in ms if m["topic"] == "tello_state"]
    assert st and not st[-1]["sending"] and not validate(st[-1])
    assert flying_after_takeoff and not io.flying  # a dry takeoff pretends to fly (FOLLOW runs on the bench) until the emergency


def test_send_rc_stream_arbitration_and_hover():
    ft, io, out, op = make_io(send=True)
    io.start()
    try:
        takeoff(ft, io, op)
        mode(op, "FOLLOW")
        run(io, 0.05)
        t0 = time.time()
        while time.time() - t0 < 0.5:
            rc(op, "controller", "FOLLOW", fb=25.4, yaw=-130)
            rc(op, "mission", "FOLLOW", fb=90)  # not the owner in FOLLOW: ignored
            run(io, 0.05)
        run(io, RC_TIMEOUT_S + 0.3)  # nothing new: hover
    finally:
        io.shutdown()
    sent = [c for c in ft.cmds() if c.startswith("rc")]
    assert "rc 0 25 0 -100" in sent and "rc 0 90 0 0" not in sent
    assert sent[-1] == "rc 0 0 0 0"
    n = sum(1 for t, c in ft.log if c.startswith("rc") and t > t0)
    assert 20 <= n <= 30  # about 20 Hz over the 1.1 s
    ms = [m for m in out.drain(100000) if m["topic"] == "rc_sent"]
    assert all(not validate(m) and m["dry_run"] is False for m in ms)


def test_operator_rc_always_wins():
    ft, io, out, op = make_io(send=True)
    io.start()
    try:
        takeoff(ft, io, op)
        mode(op, "FIND")
        for _ in range(6):
            rc(op, "mission", "FIND", yaw=30)
            rc(op, "operator", "FIND", yaw=-15)
            run(io, 0.05)
    finally:
        io.shutdown()
    sent = [c for c in ft.cmds() if c.startswith("rc")]
    assert "rc 0 0 0 -15" in sent and "rc 0 0 0 30" not in sent


def test_move_stops_rc_then_hovers_and_resumes():
    ft, io, out, op = make_io(send=True)
    ft.move_s = 0.4
    io.start()
    try:
        takeoff(ft, io, op)
        mode(op, "FOLLOW")
        op.publish(msg("tello_cmd", id="mv", cmd="move", args={"direction": "forward", "value": 50}))
        t_end = time.time() + 1.0
        while time.time() < t_end:
            rc(op, "controller", "FOLLOW", fb=30)
            run(io, 0.03)
    finally:
        io.shutdown()
    log = list(ft.log)
    i = next(k for k, (_, c) in enumerate(log) if c == "forward 50")
    t_move = log[i][0]
    assert log[i + 1][1] == "rc 0 0 0 0"  # first packet after the move is a hover
    assert log[i + 1][0] - t_move >= 0.38  # nothing was sent while the move ran
    assert "rc 0 30 0 0" in [c for _, c in log[i + 2:]]  # rc resumed
    ack = [m for m in out.drain(100000) if m["topic"] == "tello_ack" and m["id"] == "mv"][0]
    assert ack["ok"] and ack["elapsed_s"] >= 0.38 and not validate(ack)


def test_kill_preempts_queue_and_inflight_move():
    ft, io, out, op = make_io(send=True)
    ft.move_s = 0.5
    io.start()
    try:
        takeoff(ft, io, op)
        mode(op, "FOLLOW")
        op.publish(msg("tello_cmd", id="m1", cmd="move", args={"direction": "forward", "value": 50}))
        op.publish(msg("tello_cmd", id="m2", cmd="move", args={"direction": "cw", "value": 90}))
        run(io, 0.15)
        t_kill = time.time()
        op.publish(msg("kill", action="land"))
        run(io, 0.1)
        assert ("land" in ft.cmds()) and time.time() - t_kill < 0.2  # raw land went out during the move
        for _ in range(8):
            rc(op, "controller", "FOLLOW", fb=50)
            run(io, 0.08)
    finally:
        io.shutdown()
    cmds = ft.cmds()
    assert "cw 90" not in cmds
    k = cmds.index("land")
    assert not [c for c in cmds[k:] if c.startswith("rc")]  # rc stream stopped for good
    assert not io.flying
    acks = {m["id"]: m for m in out.drain(100000) if m["topic"] == "tello_ack"}
    assert acks["m2"]["ok"] is False and "preempted" in acks["m2"]["detail"]


def test_safety_lands_on_battery_and_video_loss():
    ft, io, out, op = make_io(send=True)
    io.start()
    try:
        takeoff(ft, io, op)
        ft.state = dict(ft.state, bat=22)
        run(io, 0.2)
        assert "land" in ft.cmds() and not io.flying and "battery" in io.safety_fired
    finally:
        io.shutdown()
    ft, io, out, op = make_io(send=True, video=True)
    io.start()
    try:
        io._last_frame_t = time.time()
        takeoff(ft, io, op)
        io._last_frame_t = time.time() - 5.5  # video stopped 5.5 s ago
        run(io, 0.2)
        assert "land" in ft.cmds() and "video" in io.safety_fired
    finally:
        io.shutdown()


def test_takeoff_refused_without_video_and_exit_lands():
    ft, io, out, op = make_io(send=True, video=True)
    io.start()
    try:
        op.publish(msg("tello_cmd", id="to", cmd="takeoff"))
        run(io, 0.2)
        assert "takeoff" not in ft.cmds() and not io.flying
        io._last_frame_t = time.time()
        op.publish(msg("tello_cmd", id="to2", cmd="takeoff"))
        run(io, 0.2)
        assert io.flying
    finally:
        io.shutdown()  # process exit while flying: must land
    assert ft.cmds()[-1] == "land"


def test_replayed_commands_ignored():
    ft, io, out, op = make_io(send=True)
    io.start()
    try:
        op.publish(msg("tello_cmd", id="to", cmd="takeoff", replayed=True))
        run(io, 0.2)
    finally:
        io.shutdown()
    assert "takeoff" not in ft.cmds()


# ------------------------------------------------------------------------------------------------ sim_world
class SimRig:
    def __init__(self, scenario="follow", **kw):
        from flyfollow.runtime import sim_world as sw

        self.bus = InProcBus()
        self.sim = sw.SimWorld(Publisher("sim_world", bus=self.bus), Subscriber(sw.TOPICS_IN, bus=self.bus),
                               scenario_name=scenario, **kw)
        self.out = Subscriber(None, bus=self.bus)
        self.op = Publisher("operator", bus=self.bus)
        self.t0 = time.time()

    @property
    def now(self):
        return self.t0 + self.sim.t

    def pub(self, topic, **f):
        self.op.publish(msg(topic, t=self.now, **f))

    def tick(self, n=1, rc_=None):
        for _ in range(n):
            if rc_ is not None:
                self.pub("rc", **rc_)
            self.sim.tick(self.now)

    def takeoff(self):
        self.pub("tello_cmd", id="to", cmd="takeoff")
        self.tick(220)
        assert self.sim.phase == "flying"


def test_sim_takeoff_yaw_forward():
    r = SimRig("follow")
    r.pub("tello_cmd", id="to", cmd="takeoff")
    r.tick(100)
    assert not [a for a in r.sim.acks if a["id"] == "to"]  # takeoff takes about 8 s like the real one
    r.tick(120)
    ack = [a for a in r.sim.acks if a["id"] == "to"][0]
    assert ack["ok"] and 6.0 < ack["elapsed_s"] < 10.0
    assert 0.85 <= r.sim.dm.z <= 1.2 and r.sim.flying
    r.pub("mode", **{"from": "IDLE", "to": "FOLLOW", "reason": "t"})
    y0 = r.sim.raw_state()["yaw"]
    r.tick(40, dict(lr=0, fb=0, ud=0, yaw=50, src="controller", mode="FOLLOW"))
    assert r.sim.raw_state()["yaw"] - y0 > 30  # positive yaw stick turns clockwise, yaw grows
    x0, y0_, psi = r.sim.dm.x, r.sim.dm.y, r.sim.dm.psi
    r.tick(40, dict(lr=0, fb=60, ud=0, yaw=0, src="controller", mode="FOLLOW"))
    fwd = (r.sim.dm.x - x0) * math.cos(psi) + (r.sim.dm.y - y0_) * math.sin(psi)
    assert fwd > 0.3
    st = r.sim.raw_state()
    assert st["vgx"] < 0  # the real Tello reads forward as negative vgx
    r.tick(40, dict(lr=0, fb=0, ud=0, yaw=0, src="mission", mode="FOLLOW"))  # not the owner: hover
    assert r.sim.rc_src == "hover"


def test_sim_messages_valid_and_synthetic_det():
    r = SimRig("follow")
    r.takeoff()
    r.tick(60)
    ms = r.out.drain(100000)
    topics = {m["topic"] for m in ms}
    assert {"tello_state", "rc_sent", "tello_ack", "det", "sim_truth"} <= topics
    assert all(not validate(m) for m in ms), [(m["topic"], validate(m)) for m in ms if validate(m)][:3]
    st = [m for m in ms if m["topic"] == "tello_state"][-1]
    assert st["flying"] and st["sending"] and 70 <= st["h_cm"] <= 110 and st["bat_pct"] > 25
    dets = [m for m in ms if m["topic"] == "det"]
    assert dets and all(m["img_w"] == 960 and m["img_h"] == 720 and m["t"] >= m["t_decoded"] for m in dets)
    people = [d for m in dets for d in m["dets"] if d["cls"] == "person"]
    heads = [d for m in dets for d in m["dets"] if d["cls"] == "person_head"]
    assert len(people) > 0.7 * len(dets) and heads and all(d["track_id"] == 1 for d in people + heads)
    for d in people + heads:
        x1, y1, x2, y2 = d["bbox"]
        assert 0 <= x1 < x2 <= 960 and 0 <= y1 < y2 <= 720
    truth = [m for m in ms if m["topic"] == "sim_truth"][-1]
    assert {"drone", "user", "objects", "furniture"} <= set(truth)


def test_sim_move_kill_and_find_scenario():
    r = SimRig("find")
    r.takeoff()
    x0, y0, psi = r.sim.dm.x, r.sim.dm.y, r.sim.dm.psi
    r.pub("tello_cmd", id="mv", cmd="move", args={"direction": "forward", "value": 50})
    r.tick(80)
    ack = [a for a in r.sim.acks if a["id"] == "mv"][0]
    assert ack["ok"] and 1.5 < ack["elapsed_s"] < 3.0
    assert abs((r.sim.dm.x - x0) * math.cos(psi) + (r.sim.dm.y - y0) * math.sin(psi) - 0.5) < 0.05
    # the bottle is behind the drone in the default find scenario: never detected facing the user
    dets = [d for m in r.out.drain(100000) if m["topic"] == "det" for d in m["dets"]]
    assert not [d for d in dets if d["cls"] == "bottle"]
    r.pub("tello_cmd", id="m2", cmd="move", args={"direction": "cw", "value": 180})
    r.tick(2)
    r.pub("kill", action="land")
    r.tick(80)
    acks = {a["id"]: a for a in r.sim.acks}
    assert acks["m2"]["ok"] is False and "preempted" in acks["m2"]["detail"]
    assert r.sim.phase == "ground" and r.sim.dm.z == 0.0


def test_sim_rendered_frames_in_ring():
    from flyfollow.runtime.bus import FrameRing

    ring = FrameRing.create(name=f"ff_test_{int(time.time() * 1000) % 100000}", slots=3)
    try:
        r = SimRig("follow", frames=True, ring=ring, det="none")
        r.tick(20)
        fr = [m for m in r.out.drain(100000) if m["topic"] == "frame"]
        assert fr and not validate(fr[-1])
        img = ring.read(fr[-1]["slot"], fr[-1]["frame_id"])
        assert img is not None and img.shape == (720, 960, 3) and img.std() > 10
    finally:
        ring.close()
        ring.unlink()


# ------------------------------------------------------------------------------------------------ R0 tools
def test_stick_response_fit_recovers_known_parameters():
    from flyfollow.sim.drone_model import DroneParams
    from flyfollow.tools.stick_response import analyze, synth_log

    dp = DroneParams(fwd_gain_mps=1.1, yaw_gain_dps=62.0, vz_gain_mps=0.4, tau_fwd_s=0.35, tau_yaw_s=0.05,
                     tau_z_s=0.05, fwd_dead_s=0.4, yaw_dead_s=0.2, vz_dead_s=0.3)
    res = analyze(synth_log(dp, sticks=(30, 60, 100), seed=3))
    su, s = res["sim_update"], res["summary"]
    assert abs(su["fwd_gain_mps"] / 1.1 - 1) < 0.08
    assert abs(su["yaw_gain_dps"] / 62.0 - 1) < 0.05
    assert abs(su["vz_gain_mps"] / 0.4 - 1) < 0.2  # 0.12 m/s at stick 30 reads as 1 dm/s: quantization
    assert abs(su["fwd_dead_s"] - 0.4) < 0.07 and abs(su["yaw_dead_s"] - 0.2) < 0.07
    assert abs(su["tau_fwd_s"] - 0.35) < 0.1
    assert not any(v["nonlinear"] for v in s.values())  # DroneModel is linear in the stick
    assert all(0.8 < u["ratio"] < 1.2 for u in res["unit_check"])


def test_latency_analysis_recovers_known_latency():
    from flyfollow.tools.latency_test import analyze

    rng = np.random.default_rng(0)
    ft, lv = [1.0], [1]
    while ft[-1] < 25:
        ft.append(ft[-1] + rng.uniform(0.35, 0.9))
        lv.append(1 - lv[-1])
    tc = np.arange(0.5, 26, 1 / 30)
    idx = np.searchsorted(ft, tc, side="right") - 1
    level = np.where(idx >= 0, np.array(lv)[np.clip(idx, 0, None)], 0)
    lum = np.full((len(tc), 90, 120), 60.0)
    lum[:, 20:70, 30:90] = np.where(level == 1, 210.0, 25.0)[:, None, None] + rng.normal(0, 4, (len(tc), 50, 60))
    for true in (0.3, 0.8):
        res = analyze(tc + true, lum, np.c_[ft, lv])
        assert res["ok"] and abs(res["video_latency_s"] - true) < 0.02, res


def test_update_sim_from_r0_keeps_comments(tmp_path, monkeypatch):
    import json

    from flyfollow.interfaces import REPO_ROOT
    from flyfollow.tools import update_sim_from_r0 as up

    env = tmp_path / "env.yaml"
    env.write_text((REPO_ROOT / "configs" / "env.yaml").read_text())
    (tmp_path / "r0").mkdir()
    (tmp_path / "r0" / "stick_response_1.json").write_text(json.dumps(
        {"sim_update": {"fwd_gain_mps": 1.2, "yaw_gain_dps": 50.0}, "summary": {}}))
    (tmp_path / "r0" / "latency_1.json").write_text(json.dumps({"video_latency_s": 0.3}))
    monkeypatch.setenv("FLYFOLLOW_DATA", str(tmp_path))
    up.main(["--env", str(env), "--camera", str(tmp_path / "none.json"), "--write"])
    cfg = yaml.safe_load(env.read_text())
    assert cfg["profiles"]["demo"]["fwd_gain_mps"] == 1.2 and cfg["profiles"]["demo"]["video_latency_s"] == 0.3
    assert cfg["profiles"]["train"]["fwd_gain_mps"] == [0.84, 1.56]
    assert cfg["profiles"]["train"]["video_latency_s"] == [0.21, 0.39]
    assert "# m/s at sent stick 100" in env.read_text()  # comments survive
    need, lines = up.verdict({"video_latency_s": 0.9}, {"video_latency_s": [0.15, 0.45]})
    assert need and "OUTSIDE" in lines[0]


def test_calibrate_camera_synthetic():
    from flyfollow.tools.calibrate_camera import calibrate, synthetic_views

    res = calibrate(synthetic_views(9, 6, 0.025, n=10, seed=1), 9, 6, 0.025)
    assert abs(res["fx"] - 921) < 10 and abs(res["fy"] - 919) < 10 and res["rms_px"] < 0.5
    assert abs(res["hfov_deg"] - 55.0) < 0.6


def test_video_reader_decodes_udp_h264():
    """The PyAV reader on a raw H.264 UDP stream like the Tello's (libx264 encoded here)."""
    import io
    import socket

    import av

    from flyfollow.runtime.tello_io import VideoReader

    port = 11000 + int(time.time() * 1000) % 500
    buf = io.BytesIO()
    out = av.open(buf, "w", format="h264")
    st = out.add_stream("libx264", rate=30)
    st.width, st.height, st.pix_fmt = 960, 720, "yuv420p"
    st.options = {"tune": "zerolatency", "preset": "ultrafast", "g": "10"}
    pkts = []
    for i in range(45):
        img = np.zeros((720, 960, 3), np.uint8)
        img[100:300, 50 + 10 * i: 250 + 10 * i] = (255, 200, 0)
        pkts += [bytes(p) for p in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24"))]
    pkts += [bytes(p) for p in st.encode()]
    out.close()
    got = []
    r = VideoReader(lambda img, t: got.append(img.shape), url=f"udp://@0.0.0.0:{port}", warmup_s=0.0)
    r.start()
    time.sleep(0.3)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()
    for k, p in enumerate(pkts):
        for j in range(0, len(p), 1460):
            s.sendto(p[j: j + 1460], ("127.0.0.1", port))
        time.sleep(max(0.0, t0 + (k + 1) / 60 - time.time()))
    time.sleep(0.5)
    r.stop()
    s.close()
    assert len(got) >= 15 and got[0] == (720, 960, 3)
