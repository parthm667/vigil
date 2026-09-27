"""Bus, frame ring, recorder/replay and operator script tests (loopback only, no hardware)."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import textwrap
import time
import uuid

import numpy as np
import pytest

from flyfollow.runtime import bus as B


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def addrs(monkeypatch):
    pub, sub = f"tcp://127.0.0.1:{free_port()}", f"tcp://127.0.0.1:{free_port()}"
    monkeypatch.setenv("FLYFOLLOW_PUB_ADDR", pub)
    monkeypatch.setenv("FLYFOLLOW_SUB_ADDR", sub)
    return pub, sub


@pytest.fixture
def broker(addrs):
    b = B.Broker(*addrs).start()
    yield b
    b.stop()


def recv_until(sub: B.Subscriber, n: int, timeout_s: float = 2.0) -> list[dict]:
    out, deadline = [], time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < deadline:
        m = sub.recv(0.05)
        if m is not None:
            out.append(m)
    return out


# ------------------------------------------------------------------------------------------------ pub/sub
def test_broker_round_trip(broker):
    sub = B.Subscriber(["det"])
    pub = B.Publisher("tester")
    assert sub.connected and pub.connected
    pub.publish({"topic": "det", "frame_id": 1, "dets": [], "arr": np.arange(3), "x": np.float32(1.5)})
    got = recv_until(sub, 1)
    assert len(got) == 1
    m = got[0]
    assert m["topic"] == "det" and m["frame_id"] == 1 and m["src_node"] == "tester"
    assert m["arr"] == [0, 1, 2] and m["x"] == 1.5 and abs(m["t"] - time.time()) < 5
    pub.close()
    sub.close()


def test_exact_topic_filter_and_order(broker):
    sub = B.Subscriber(["det", "rc"])
    pub = B.Publisher("p")
    for i in range(50):
        pub.publish({"topic": "det_cfg", "i": i})  # prefix of nothing subscribed but "det" matches as prefix
        pub.publish({"topic": "det", "i": i})
        pub.publish({"topic": "rc", "i": i, "t": 123.0})
        pub.publish({"topic": "tello_state", "i": i})
    got = recv_until(sub, 100)
    assert len(got) == 100
    assert {m["topic"] for m in got} == {"det", "rc"}
    assert [m["i"] for m in got if m["topic"] == "det"] == list(range(50))  # in order per publisher
    assert all(m["t"] == 123.0 for m in got if m["topic"] == "rc")  # t kept when given
    assert sub.drain() == []
    pub.close()
    sub.close()


def test_all_topics_and_conflate(broker):
    sub_all = B.Subscriber()
    sub_c = B.Subscriber(["tello_state", "det"], conflate=True)
    pub = B.Publisher("p")
    for i in range(20):
        pub.publish({"topic": "tello_state", "i": i})
        pub.publish({"topic": "det", "i": i})
    assert len(recv_until(sub_all, 40)) == 40
    time.sleep(0.1)
    latest = sub_c.drain()
    assert sorted((m["topic"], m["i"]) for m in latest) == [("det", 19), ("tello_state", 19)]
    for s in (pub, sub_all, sub_c):
        s.close()


def test_publish_never_blocks_without_broker_or_subscriber(addrs):
    t0 = time.monotonic()
    pub = B.Publisher("lonely", wait_s=0.05)
    assert not pub.connected
    for i in range(20_000):  # beyond the high-water mark
        pub.publish({"topic": "rc", "i": i})
    pub.close()
    assert time.monotonic() - t0 < 2.0


def test_publish_requires_topic():
    with pytest.raises(ValueError):
        B.Publisher("x", bus=B.InProcBus()).publish({"t": 1.0})


def test_inproc_bus():
    bus = B.InProcBus()
    sub = B.Subscriber(["mode"], bus=bus)
    sub_all = B.Subscriber(bus=bus)
    pub = B.Publisher("mission", bus=bus)
    payload = {"topic": "mode", "from": "IDLE", "to": "FOLLOW", "reason": "test"}
    pub.publish(payload)
    pub.publish({"topic": "mode_x"})
    assert "src_node" not in payload  # caller's dict untouched
    m = sub.recv(0.1)
    assert m["to"] == "FOLLOW" and m["src_node"] == "mission"
    assert sub.recv() is None
    assert [x["topic"] for x in sub_all.drain()] == ["mode", "mode_x"]
    sub.close()
    pub.publish({"topic": "mode"})
    assert len(sub_all.drain()) == 1


def test_inproc_recv_timeout_wakes_on_publish():
    import threading

    bus = B.InProcBus()
    sub = B.Subscriber(["x"], bus=bus)
    pub = B.Publisher("p", bus=bus)
    threading.Timer(0.05, lambda: pub.publish({"topic": "x"})).start()
    t0 = time.monotonic()
    assert sub.recv(2.0) is not None
    assert time.monotonic() - t0 < 1.0


def test_broker_port_in_use(broker):
    with pytest.raises(RuntimeError, match="cannot bind"):
        B.Broker(broker.pub_addr, broker.sub_addr).start()


# ------------------------------------------------------------------------------------------------ frame ring
def ring_name() -> str:
    return "fft_" + uuid.uuid4().hex[:10]


def test_frame_ring_write_read_overwrite():
    name = ring_name()
    ring = B.FrameRing.create(name, slots=3, h=48, w=64)
    try:
        assert ring.latest() is None
        frames = [np.full((48, 64, 3), i, np.uint8) for i in range(5)]
        slots = [ring.write(f, frame_id=100 + i, t_decoded=10.0 + i) for i, f in enumerate(frames)]
        assert slots == [0, 1, 2, 0, 1]
        assert ring.read(0, 100) is None  # overwritten by frame 103
        assert ring.read(0, 103)[0, 0, 0] == 3
        img, fid, t = ring.latest()
        assert fid == 104 and t == 14.0 and img[5, 5, 2] == 4
        assert ring.read(7, 104) is None
        with pytest.raises(ValueError):
            ring.write(np.zeros((10, 10, 3), np.uint8), 1, 0.0)
    finally:
        ring.close()
        ring.unlink()


def test_frame_ring_stale_segment_and_long_name():
    name = ring_name() + "_this_name_is_far_too_long_for_macos"
    a = B.FrameRing.create(name, slots=2, h=8, w=8)
    a.write(np.ones((8, 8, 3), np.uint8), 1, 1.0)
    a.close()  # simulate a crash: closed but never unlinked
    b = B.FrameRing.create(name, slots=2, h=8, w=8)
    try:
        assert len(b.name) <= 30
        assert b.latest() is None  # fresh segment
    finally:
        b.close()
        b.unlink()
    with pytest.raises(FileNotFoundError):
        B.FrameRing.attach(name)


def test_frame_ring_across_processes():
    name = ring_name()
    ring = B.FrameRing.create(name, slots=4, h=72, w=96)
    try:
        for i in range(6):
            ring.write(np.full((72, 96, 3), i * 10, np.uint8), frame_id=i, t_decoded=float(i))
        code = textwrap.dedent(f"""
            import json
            from flyfollow.runtime.bus import FrameRing
            r = FrameRing.attach({name!r}, timeout_s=1.0)
            img, fid, t = r.latest()
            out = {{"fid": fid, "t": t, "px": int(img[0, 0, 0]), "shape": list(img.shape),
                    "old": r.read(1, 1) is None, "ok": int(r.read(1, 5)[3, 3, 1])}}
            r.close()
            print(json.dumps(out))
        """)
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
        assert p.returncode == 0, p.stderr
        assert "leaked" not in p.stderr and "Warning" not in p.stderr, p.stderr
        out = json.loads(p.stdout.strip().splitlines()[-1])
        assert out == {"fid": 5, "t": 5.0, "px": 50, "shape": [72, 96, 3], "old": True, "ok": 50}
        assert ring.latest()[1] == 5  # the consumer's exit did not unlink the segment
        B.FrameRing.attach(name).close()
    finally:
        ring.close()
        ring.unlink()


# ------------------------------------------------------------------------------------------------ recorder / replay
def test_recorder_replay_round_trip(tmp_path):
    from flyfollow.runtime.recorder import Recorder, read_jsonl
    from flyfollow.runtime.replay import Replayer

    bus = B.InProcBus()
    name = ring_name()
    ring = B.FrameRing.create(name, slots=4, h=48, w=64)
    rec = Recorder(tmp_path / "s1", B.Subscriber(bus=bus), video=True, ring_name=name, fps=30)
    pub = B.Publisher("tello_io", bus=bus)
    t0 = time.time()
    try:
        for i in range(12):
            img = np.zeros((48, 64, 3), np.uint8)
            img[:, : 4 + 4 * i] = 200  # a bar that grows, so frames are distinguishable after compression
            slot = ring.write(img, frame_id=i, t_decoded=t0 + i * 0.01)
            pub.publish({"topic": "frame", "frame_id": i, "t_decoded": t0 + i * 0.01, "slot": slot, "w": 64, "h": 48})
            pub.publish({"topic": "det", "frame_id": i, "t_decoded": t0 + i * 0.01, "src": "test", "img_w": 64,
                         "img_h": 48, "dets": []})
            if i % 3 == 0:
                pub.publish({"topic": "tello_state", "bat_pct": 90 - i})
            if i == 5:
                B.Publisher("operator", bus=bus).publish({"topic": "kill", "action": "emergency"})
            rec.poll(0.0)
            time.sleep(0.005)
        meta = rec.close()
    finally:
        ring.close()
        ring.unlink()
    assert meta["counts"] == {"frame": 12, "det": 12, "tello_state": 4, "kill": 1}
    assert meta["n_frames"] == 12 and meta["video_error"] is None
    lines = read_jsonl(tmp_path / "s1" / "bus.jsonl")
    assert len(lines) == 29 and all("t_rec" in m for m in lines)
    assert [f["frame_id"] for f in read_jsonl(tmp_path / "s1" / "frames.jsonl")] == list(range(12))

    out_bus = B.InProcBus()
    sub = B.Subscriber(bus=out_bus)
    out_ring = ring_name()
    rp = Replayer(tmp_path / "s1", B.Publisher("replay", bus=out_bus), topics=["det", "tello_state"], frames=True,
                  speed=20.0, ring_name=out_ring)
    t_start = time.time()
    try:
        rp.run()
        got = sub.drain()
        frames = [m for m in got if m["topic"] == "frame"]
        assert len(frames) == 12 and len([m for m in got if m["topic"] == "det"]) == 12
        assert len([m for m in got if m["topic"] == "tello_state"]) == 4
        assert all(m["t"] >= t_start - 0.01 and m["replayed"] for m in got)  # retimed to now, marked
        det3 = next(m for m in got if m["topic"] == "det" and m["frame_id"] == 3)
        rec3 = next(m for m in lines if m["topic"] == "det" and m["frame_id"] == 3)
        assert abs((det3["t"] - det3["t_decoded"]) - (rec3["t"] - rec3["t_decoded"])) < 1e-5  # latency kept
        # frames precede their det and the last one is still readable from the ring
        assert got.index(frames[0]) < got.index(next(m for m in got if m["topic"] == "det"))
        last = frames[-1]
        img = rp.ring.read(last["slot"], last["frame_id"])
        assert img is not None and img.shape == (48, 64, 3)
        assert abs(int(img[:, :48].mean()) - 200) < 15 and int(img[:, 56:].mean()) < 15
    finally:
        rp.close()

    safe = Replayer(tmp_path / "s1", B.Publisher("replay", bus=out_bus), speed=50.0)
    assert {m["topic"] for m in safe.msgs} == {"det", "tello_state"}  # no frame (stale slots), no command topics
    orig = Replayer(tmp_path / "s1", B.Publisher("replay", bus=out_bus), topics=["tello_state"], speed=50.0,
                    retime=False)
    orig.run()
    assert [m["t"] for m in sub.drain()] == [m["t"] for m in lines if m["topic"] == "tello_state"]


# ------------------------------------------------------------------------------------------------ operator
def test_operator_script_mode_emits_messages():
    from flyfollow.runtime import operator as op

    bus = B.InProcBus()
    sub = B.Subscriber(bus=bus)
    c = op.Console(B.Publisher("operator", bus=bus))
    op_sub = B.Subscriber(op.TOPICS, bus=bus)
    det = {"topic": "det", "frame_id": 1, "t_decoded": 0.0, "src": "t", "img_w": 960, "img_h": 720, "dets": [
        {"cls": "person", "conf": 0.9, "bbox": [0, 100, 100, 700], "track_id": 4},  # tall but at the edge
        {"cls": "person", "conf": 0.9, "bbox": [430, 150, 530, 690], "track_id": 7},  # central
        {"cls": "person_head", "conf": 0.9, "bbox": [455, 150, 505, 210], "track_id": 7}]}
    B.Publisher("detector", bus=bus).publish(det)
    steps = op.parse_script("""
        # comment
        0.00 takeoff
        0.01 follow
        0.02 lock auto
        0.03 cmd find my water bottle
        0.04 set follow_distance_m 2.5
        0.05 controller
        0.06 mode HOLD
        0.07 land
        0.08 emergency
        0.09 wait tello_ack 0.05
    """)
    fails = op.run_script(c, op_sub, steps, linger_s=0.0, echo=False)
    assert fails == 1  # nobody acks in this test
    got = [m for m in sub.drain() if m["src_node"] == "operator"]
    assert [m["topic"] for m in got] == ["tello_cmd", "mode_cmd", "lock", "mode_cmd", "settings", "settings", "mode_cmd",
                                         "kill", "kill"]
    assert got[0]["cmd"] == "takeoff" and got[0]["id"] == "op-1"
    assert got[1]["mode"] == "FOLLOW" and got[1]["source"] == "operator"
    assert got[2]["track_id"] == 7
    assert got[3]["mode"] == "FIND" and got[3]["target"]["cls"] == "bottle"
    assert got[4]["follow_distance_m"] == 2.5
    assert got[5]["controller"] == "pid"
    assert got[6]["mode"] == "HOLD"
    assert [m["action"] for m in got[7:]] == ["land", "emergency"]
    for m in got:
        assert not __import__("flyfollow.runtime.messages", fromlist=["validate"]).validate(m), m


def test_operator_keys_and_status():
    from flyfollow.runtime import operator as op

    bus = B.InProcBus()
    sub = B.Subscriber(bus=bus)
    c = op.Console(B.Publisher("operator", bus=bus))
    ks = op.KeyState()
    for ch in b" lt":
        op.handle_key(c, ks, ch)
    assert [(m["topic"], m.get("action")) for m in sub.drain()] == [("kill", "emergency"), ("kill", "land")]
    op.handle_key(c, ks, ord("t"))  # second press arms -> takeoff
    assert sub.drain()[0]["cmd"] == "takeoff"
    for ch in b"/stop\r":
        op.handle_key(c, ks, ch)
    assert sub.drain()[0]["mode"] == "HOLD"
    for ch in b"/fin":
        op.handle_key(c, ks, ch)
    op.handle_key(c, ks, 5)  # Ctrl+E while typing still kills
    assert sub.drain()[0]["action"] == "emergency"
    op.handle_key(c, ks, 27)
    op.handle_key(c, ks, 3)  # Ctrl+C: land then quit
    assert ks.quit and sub.drain()[0]["action"] == "land"
    c.ingest({"topic": "tello_state", "t": time.time(), "bat_pct": 20, "h_cm": 90, "video_age_s": 0.2, "flying": True})
    c.ingest({"topic": "mode", "t": time.time(), "from": "IDLE", "to": "FOLLOW", "reason": "operator"})
    text = "\n".join(t for t, _ in c.status_lines())
    assert "FOLLOW" in text and "BATTERY 20%" in text and "no rc from controller" in text
