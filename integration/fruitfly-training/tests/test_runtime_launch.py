"""Launcher smoke tests: whole process trees on random loopback ports, clean shutdown, nothing left behind."""

from __future__ import annotations

import json
import re
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from flyfollow.interfaces import REPO_ROOT
from flyfollow.runtime.launch import module_exists
from flyfollow.runtime.recorder import read_jsonl

SIM_MODULES = ("sim_world", "controller_runner")


def free_port_pair() -> int:
    for _ in range(50):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            p = s.getsockname()[1]
        if p < 65000:
            try:
                with socket.socket() as s2:
                    s2.bind(("127.0.0.1", p + 1))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port pair")


def launch(args: list[str], tmp: Path, timeout: float = 40.0) -> subprocess.CompletedProcess:
    env = dict(os.environ, FLYFOLLOW_RECORDINGS=str(tmp))
    cmd = [sys.executable, "-m", "flyfollow.runtime.launch", *args, "--port", str(free_port_pair()),
           "--record-root", str(tmp), "--quiet"]
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)


def assert_no_children_left(session: str) -> None:
    out = subprocess.run(["ps", "-Ao", "pid,command"], capture_output=True, text=True, check=False).stdout
    left = [ln for ln in out.splitlines() if session in ln and "flyfollow.runtime" in ln]
    assert not left, left


def make_recording(d: Path) -> None:
    d.mkdir(parents=True)
    t0 = time.time() - 100
    with (d / "bus.jsonl").open("w") as f:
        for i in range(30):
            t = t0 + i * 0.05
            f.write(json.dumps({"topic": "tello_state", "t": t, "t_rec": t, "bat_pct": 80, "flying": False}) + "\n")
            f.write(json.dumps({"topic": "det", "t": t, "t_rec": t, "t_decoded": t - 0.03, "frame_id": i, "src": "rec",
                                "img_w": 960, "img_h": 720, "dets": []}) + "\n")
            f.write(json.dumps({"topic": "rc", "t": t, "t_rec": t, "lr": 0, "fb": 0, "ud": 0, "yaw": 0, "src": "x",
                                "mode": "FOLLOW"}) + "\n")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_launch_replay_script_smoke(tmp_path):
    src = tmp_path / "src_session"
    make_recording(src)
    script = tmp_path / "ops.txt"
    script.write_text("0.3 wait tello_state 5\n0.5 follow\n0.6 cmd find my water bottle\n0.7 hold\n")
    session = "t_replay_" + uuid.uuid4().hex[:6]
    t0 = time.monotonic()
    p = launch(["--replay", str(src), "--script", str(script), "--linger", "0.3", "--session", session,
                "--no-controller", "--no-mission", "--no-guidance"], tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert time.monotonic() - t0 < 20
    lines = read_jsonl(tmp_path / session / "bus.jsonl")
    topics = [m["topic"] for m in lines]
    assert "tello_state" in topics and "det" in topics
    assert "rc" not in topics  # only drone-side topics are replayed
    cmds = [m for m in lines if m["topic"] == "mode_cmd"]
    assert [m["mode"] for m in cmds] == ["FOLLOW", "FIND", "HOLD"]
    assert cmds[1]["target"]["cls"] == "bottle"
    assert any(m["topic"] == "health" and m.get("key") == "recorder" for m in lines)
    replayed = [m for m in lines if m["topic"] == "tello_state"]
    assert all(m["t"] > time.time() - 60 for m in replayed)  # retimed to now
    meta = json.loads((tmp_path / session / "meta.json").read_text())
    assert meta["n_msgs"] == len(lines) and meta["mode"] == "REPLAY"
    assert (tmp_path / session / "logs" / "launch.log").exists()
    assert_no_children_left(session)


def test_launch_missing_required_backend(tmp_path, monkeypatch):
    p = launch(["--replay", str(tmp_path / "nope"), "--no-operator"], tmp_path, timeout=20)
    assert p.returncode == 2 and "no bus.jsonl" in p.stdout


@pytest.mark.skipif(not all(module_exists(f"flyfollow.runtime.{m}") for m in SIM_MODULES),
                    reason="sim_world / controller_runner not present")
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_launch_sim_smoke(tmp_path):
    """sim_world + controller + operator script; shutdown must land and get the land ack before stopping sim_world."""
    script = tmp_path / "ops.txt"
    script.write_text("0.5 wait tello_state 8\n1.0 takeoff\n1.2 wait tello_ack 8\n1.3 follow\n1.5 lock auto\n3.0 hold\n")
    session = "t_sim_" + uuid.uuid4().hex[:6]
    p = launch(["--sim", "--script", str(script), "--linger", "0.5", "--session", session,
                "--args", "sim_world=--speed 3"], tmp_path, timeout=60)  # 3x sim time: takeoff ok in about 3 s
    log = (tmp_path / session / "logs" / "launch.log").read_text() if (tmp_path / session).exists() else ""
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:] + log
    lines = read_jsonl(tmp_path / session / "bus.jsonl")
    topics = {m["topic"] for m in lines}
    assert {"tello_state", "mode_cmd", "tello_cmd", "kill"} <= topics, topics
    t_kill = next(m["t"] for m in lines if m["topic"] == "kill" and m.get("src_node") == "launch")  # land first
    acks = [m for m in lines if m["topic"] == "tello_ack"]
    assert any(m["cmd"] == "takeoff" and m["ok"] for m in acks), acks
    assert any(m["cmd"] == "land" and m["ok"] and m["t"] > t_kill for m in acks), acks
    assert "land: CONFIRMED by ack" in log, log
    assert any(m["topic"] == "lock" and m["track_id"] is not None for m in lines)
    assert "ERROR" not in log, log
    assert_no_children_left(session)


@pytest.mark.skipif(sys.platform == "win32", reason="pty")
def test_operator_curses_in_pty(monkeypatch):
    """The real curses console in a pseudo-terminal: space = emergency, l = land, q q = land + quit."""
    import pty

    from flyfollow.runtime import bus as B

    port = free_port_pair()
    pub_addr, sub_addr = f"tcp://127.0.0.1:{port}", f"tcp://127.0.0.1:{port + 1}"
    monkeypatch.setenv("FLYFOLLOW_PUB_ADDR", pub_addr)
    monkeypatch.setenv("FLYFOLLOW_SUB_ADDR", sub_addr)
    broker = B.Broker().start()
    sub = B.Subscriber(["kill", "mode_cmd"])
    feed = B.Publisher("tello_io")
    master, slave = pty.openpty()
    env = dict(os.environ, TERM="xterm-256color", LINES="40", COLUMNS="160")
    proc = subprocess.Popen([sys.executable, "-m", "flyfollow.runtime.operator"], cwd=REPO_ROOT, env=env,
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    screen = b""

    def read_for(s: float) -> None:
        nonlocal screen
        import select

        end = time.monotonic() + s
        while time.monotonic() < end:
            r, _, _ = select.select([master], [], [], 0.05)
            if r:
                try:
                    screen += os.read(master, 65536)
                except OSError:
                    return

    try:
        read_for(1.5)
        feed.publish({"topic": "tello_state", "bat_pct": 64, "h_cm": 100, "flying": True, "video_age_s": 0.2})
        read_for(0.3)
        for key in (b" ", b"l", b"f"):
            os.write(master, key)
            read_for(0.2)
        got = [(m["topic"], m.get("action") or m.get("mode")) for m in (sub.recv(1.0), sub.recv(1.0), sub.recv(1.0))]
        assert got == [("kill", "emergency"), ("kill", "land"), ("mode_cmd", "FOLLOW")]
        os.write(master, b"qq")
        read_for(0.5)
        assert proc.wait(timeout=5) == 0
        assert sub.recv(1.0)["action"] == "land"
        text = re.sub(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()=>][0-9A-B]?", b" ", screen)
        assert b"OPERATOR" in text and b"EMERGENCY" in text and b"64%" in text, text[-2000:]
    finally:
        if proc.poll() is None:
            proc.kill()
        os.close(master)
        for s in (sub, feed):
            s.close()
        broker.stop()


@pytest.mark.parametrize("behavior,path,n_kills", [
    ("ack", "ack", 1),                # normal: backend acks the land
    ("state_then_ack", "ack", 1),     # flying=False first, ack 0.1 s later: must keep waiting for the ack
    ("state_only", "state", 1),       # never acks: confirmed by tello_state after the grace period
    ("ignore_first", "ack", 2),       # first kill lost: re-sent after LAND_RESEND_S, then acked
    ("silent", "none", 2),            # nothing at all: re-sent once, then NOT CONFIRMED
    ("slow_landing", "ack", 1),       # kill received (tello_event), descending for 0.9 s: no re-send, no early exit
    ("already_landing", "ack", 1),    # an ack while states still say flying is not confirmation
])
def test_land_and_wait_paths(tmp_path, monkeypatch, behavior, path, n_kills):
    import threading

    from flyfollow.runtime import bus as B
    from flyfollow.runtime import launch as L

    port = free_port_pair()
    monkeypatch.setenv("FLYFOLLOW_PUB_ADDR", f"tcp://127.0.0.1:{port}")
    monkeypatch.setenv("FLYFOLLOW_SUB_ADDR", f"tcp://127.0.0.1:{port + 1}")
    monkeypatch.setenv("FLYFOLLOW_FRAME_RING", f"ff_frames_{port}")
    monkeypatch.setattr(L, "LAND_WAIT_S", 1.5)
    monkeypatch.setattr(L, "LAND_RESEND_S", 0.4)
    monkeypatch.setattr(L, "ACK_GRACE_S", 0.3)
    la = L.Launcher(L.build_parser().parse_args(["--sim", "--no-operator", "--quiet", "--port", str(port),
                                                 "--record-root", str(tmp_path), "--session", "land"]))
    la._print_ok = False
    broker = B.Broker().start()
    la.pub = B.Publisher("launch")
    la.sub = B.Subscriber(["tello_state", "det", "rc", "tello_ack", "tello_event", "mode", "health"])

    class Alive:
        def poll(self):
            return None

    la.children = [L.Child("sim_world", "x", [], proc=Alive())]
    kills = []
    done = threading.Event()

    def backend():
        sub, pub = B.Subscriber(["kill"]), B.Publisher("sim_world")
        pub.publish({"topic": "tello_state", "flying": True})
        while not done.is_set():
            m = sub.recv(0.02)
            if m is None:
                continue
            kills.append(m)
            if behavior == "silent" or (behavior == "ignore_first" and len(kills) == 1):
                continue
            if behavior in ("slow_landing", "already_landing"):
                pub.publish({"topic": "tello_event", "kind": "kill", "action": "land", "detail": ""})
                if behavior == "already_landing":
                    pub.publish({"topic": "tello_ack", "id": "k0", "cmd": "land", "ok": True, "detail": "already landing"})
                for _ in range(9):
                    time.sleep(0.1)
                    pub.publish({"topic": "tello_state", "flying": True})
                pub.publish({"topic": "tello_state", "flying": False})
                if behavior == "already_landing":
                    continue
            if behavior in ("state_then_ack", "state_only"):
                pub.publish({"topic": "tello_state", "flying": False})
                if behavior == "state_only":
                    continue
                time.sleep(0.1)
            pub.publish({"topic": "tello_ack", "id": f"kill-{len(kills)}", "cmd": "land", "ok": True, "detail": "ok"})
        sub.close()
        pub.close()

    th = threading.Thread(target=backend, daemon=True)
    th.start()
    time.sleep(0.6)  # backend connected (constructor handshakes)
    try:
        t0 = time.monotonic()
        assert la.land_and_wait() == path
        if behavior in ("slow_landing", "already_landing"):
            assert time.monotonic() - t0 > 0.85  # waited for touchdown
        assert len(kills) == n_kills and all(k["action"] == "land" for k in kills)
        text = (tmp_path / "land" / "logs" / "launch.log").read_text()
        assert {"ack": "CONFIRMED by ack", "state": "CONFIRMED by tello_state", "none": "NOT CONFIRMED"}[path] in text
        assert ("RE-SENT" in text) == (n_kills == 2)
    finally:
        done.set()
        th.join(2)
        la.pub.close()
        la.sub.close()
        broker.stop()


@pytest.mark.skipif(not os.environ.get("FLYFOLLOW_E2E"), reason="about 50 s: set FLYFOLLOW_E2E=1")
@pytest.mark.skipif(not all(module_exists(f"flyfollow.runtime.{m}") for m in SIM_MODULES), reason="sim modules missing")
def test_e2e_sim_takeoff_follow_40s(tmp_path):
    """launch --sim --no-operator --takeoff-follow --duration 40: the drone follows the walking user and lands."""
    import math

    session = "t_e2e_" + uuid.uuid4().hex[:6]
    p = launch(["--sim", "--no-operator", "--takeoff-follow", "--duration", "40", "--session", session], tmp_path,
               timeout=90)
    log = (tmp_path / session / "logs" / "launch.log").read_text()
    assert p.returncode == 0, p.stdout[-3000:] + log
    lines = read_jsonl(tmp_path / session / "bus.jsonl")
    trans = [(m["from"], m["to"]) for m in lines if m["topic"] == "mode" and m["reason"] != "periodic"]
    assert ("IDLE", "FOLLOW") in trans and trans[-2:] == [("FOLLOW", "LAND"), ("LAND", "IDLE")], trans
    truth = [m for m in lines if m["topic"] == "sim_truth"]
    fol = [m for m in truth if m["mode"] == "FOLLOW" and m["drone"]["flying"]]
    assert len(fol) > 250, len(fol)  # > 25 s of FOLLOW at 10 Hz
    hfov = math.atan(480 / 921)
    in_view, dmin, walked = 0, 99.0, 0.0
    for i, m in enumerate(fol):
        d, u = m["drone"], m["user"]
        psi = math.radians(d["psi_deg"])
        dx, dy = u["x"] - d["x"], u["y"] - d["y"]
        f, lft = dx * math.cos(psi) + dy * math.sin(psi), -dx * math.sin(psi) + dy * math.cos(psi)
        in_view += f > 0 and abs(math.atan2(lft, f)) < hfov
        dmin = min(dmin, math.hypot(dx, dy))
        if i:
            walked += math.hypot(u["x"] - fol[i - 1]["user"]["x"], u["y"] - fol[i - 1]["user"]["y"])
    assert walked > 3.0, walked  # the user really walked
    assert in_view / len(fol) > 0.8, in_view / len(fol)
    assert dmin > 0.8, dmin
    assert not truth[-1]["collisions"], truth[-1]["collisions"]
    assert sum(1 for m in lines if m["topic"] == "rc_sent" and m["src"] == "controller") > 400
    assert "land: CONFIRMED" in log, log
    assert truth[-1]["drone"]["flying"] is False
    assert_no_children_left(session)
