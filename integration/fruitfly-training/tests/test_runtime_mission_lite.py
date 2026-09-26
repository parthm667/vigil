"""mission_lite: the harness mode owner, driven step by step on an InProcBus with injected time."""

from __future__ import annotations

import pytest

from flyfollow.runtime import bus as B
from flyfollow.runtime.messages import validate
from flyfollow.runtime.mission_lite import MissionLite


class Rig:
    def __init__(self, **kw):
        self.bus = B.InProcBus()
        self.out = B.Subscriber(bus=self.bus)
        self.ml = MissionLite(B.Publisher("mission", bus=self.bus), B.Subscriber(
            ["mode_cmd", "tello_state", "tello_ack", "tello_cmd", "tello_event", "kill", "ctrl_status", "lock", "det"],
            bus=self.bus), **kw)
        self.ext = B.Publisher("test", bus=self.bus)
        self.now = 1000.0

    def send(self, topic: str, **f) -> None:
        self.ext.publish({"topic": topic, **f})

    def step(self, dt: float = 0.05) -> list[dict]:
        self.now += dt
        self.ml.step(self.now)
        return [m for m in self.out.drain() if m.get("src_node") == "mission"]

    def state(self, flying: bool, **f) -> list[dict]:
        self.send("tello_state", flying=flying, bat_pct=f.pop("bat_pct", 80), video_age_s=f.pop("video_age_s", 0.05), **f)
        return self.step()


def modes(out: list[dict]) -> list[str]:
    return [m["to"] for m in out if m["topic"] == "mode" and m["reason"] != "periodic"]


def flying_rig(mode: str = "FOLLOW") -> Rig:
    r = Rig()
    r.state(False)
    r.send("tello_cmd", id="op-1", cmd="takeoff")
    r.step()
    r.send("tello_ack", id="op-1", cmd="takeoff", ok=True, detail="ok")
    r.state(True)
    if mode != "HOLD":
        r.send("mode_cmd", mode=mode, source="operator")
        r.step()
    assert r.ml.mode == mode
    return r


def test_start_idle_and_refuse_on_ground():
    r = Rig()
    assert modes(r.state(False)) == ["IDLE"]
    r.send("mode_cmd", mode="FOLLOW", source="operator")
    out = r.step()
    assert r.ml.mode == "IDLE" and any("take off first" in m["text"] for m in out if m["topic"] == "say")


def test_operator_takeoff_goes_hold_with_hover_rc():
    r = Rig()
    r.state(False)
    r.send("tello_cmd", id="op-1", cmd="takeoff")
    r.step()
    r.send("tello_ack", id="op-1", cmd="takeoff", ok=True, detail="ok")
    out = r.state(True)
    assert modes(out) == ["HOLD"]
    rc = [m for m in out if m["topic"] == "rc"]
    assert rc and rc[0] == {**rc[0], "lr": 0, "fb": 0, "ud": 0, "yaw": 0, "src": "mission", "mode": "HOLD"}
    tg = [m for m in out if m["topic"] == "target"]
    assert tg[-1]["kind"] == "person" and tg[-1]["mode"] == "HOLD"
    for m in out:
        assert not validate(m), m


def test_takeoff_follow_success_and_auto_lock():
    r = Rig(takeoff_follow=True)
    r.send("det", frame_id=1, img_w=960, dets=[{"cls": "person", "conf": 0.9, "bbox": [400, 100, 520, 700],
                                                  "track_id": 1}])
    out = r.state(False)
    cmd = [m for m in out if m["topic"] == "tello_cmd"]
    assert [c["cmd"] for c in cmd] == ["takeoff"]
    r.state(True)
    assert r.ml.mode == "IDLE"  # waits for the ack, not for flying=True
    r.send("tello_ack", id=cmd[0]["id"], cmd="takeoff", ok=True, detail="ok")
    out = r.step()
    assert modes(out) == ["FOLLOW"]
    assert [m["track_id"] for m in out if m["topic"] == "lock"] == [1]
    assert [m.get("track_id") for m in out if m["topic"] == "target"][-1] == 1
    assert not [m for m in r.step() if m["topic"] == "rc"]  # FOLLOW: the controller owns rc


def test_takeoff_refused_and_timeout():
    r = Rig(takeoff_follow=True)
    cid = [m for m in r.state(False) if m["topic"] == "tello_cmd"][0]["id"]
    r.send("tello_ack", id=cid, cmd="takeoff", ok=False, detail="battery 10% below 25%")
    out = r.step()
    assert r.ml.mode == "IDLE" and any("refused" in m["text"] for m in out if m["topic"] == "say")
    r2 = Rig(takeoff_follow=True, takeoff_timeout_s=10.0)
    r2.state(False)
    out = r2.step(10.5)
    assert r2.ml.mode == "IDLE" and any("no ack" in m["text"] for m in out if m["topic"] == "say")


def test_not_in_harness_and_guide_stop():
    r = flying_rig("GUIDE")
    r.send("mode_cmd", mode="FIND", source="voice", target={"cls": "bottle"})
    out = r.step()
    assert r.ml.mode == "GUIDE" and any("not in this harness" in m["text"] for m in out if m["topic"] == "say")
    r.send("mode_cmd", mode="HOLD", source="voice", intent="stop")
    assert modes(r.step()) == ["FOLLOW"]
    r.send("mode_cmd", mode="HOLD", source="voice", intent="stay")
    assert modes(r.step()) == ["HOLD"]


@pytest.mark.parametrize("trigger,sends_land", [
    ("kill", False), ("event", False), ("battery", True), ("video", True), ("land_request", True), ("mode_cmd", True)])
def test_land_triggers(trigger, sends_land):
    r = flying_rig("FOLLOW")
    if trigger == "kill":
        r.send("kill", action="emergency")
        out = r.step()
    elif trigger == "event":
        r.send("tello_event", kind="kill", action="land", reason="safety: video lost", detail="")
        out = r.step()
    elif trigger == "battery":
        out = r.state(True, bat_pct=20)
    elif trigger == "video":
        out = r.state(True, video_age_s=6.0)
    elif trigger == "land_request":
        r.send("ctrl_status", mode="FOLLOW", controller="pid", target_valid=False, range_m=None, bearing_deg=None,
               in_band=False, land_request=True, land_reason="lost > 10 s")
        out = r.step()
    else:
        r.send("mode_cmd", mode="LAND", source="operator")
        out = r.step()
    assert modes(out) == ["LAND"]
    assert [m["cmd"] for m in out if m["topic"] == "tello_cmd"] == (["land"] if sends_land else [])
    assert modes(r.state(False)) == ["IDLE"]


def test_periodic_republish_and_lock_updates_target():
    r = flying_rig("FOLLOW")
    r.step()
    out = r.step(1.1)
    assert [m["reason"] for m in out if m["topic"] == "mode"] == ["periodic"]
    assert [m["kind"] for m in out if m["topic"] == "target"] == ["person"]
    r.send("lock", track_id=5)
    assert [m.get("track_id") for m in r.step() if m["topic"] == "target"] == [5]


def test_intent_parse_mode_cmd_accepted():
    from flyfollow.runtime.intent import parse

    r = flying_rig("HOLD")
    r.ext.publish(parse("follow me"))
    assert modes(r.step()) == ["FOLLOW"]
    r.ext.publish(parse("find my water bottle"))
    assert r.ml.mode == "FOLLOW" and any(m["topic"] == "say" for m in r.step())
