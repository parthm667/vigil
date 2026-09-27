"""Minimal mode owner for the fly-controller test harness (the ReachGlass stack owns the real mission).

Turns mode_cmd (operator console, flyfollow.runtime.intent.parse) into "mode" transitions for FOLLOW, HOLD, GUIDE,
LAND and IDLE only; FIND / APPROACH / RETURN / FACE_PERSON / OVERWATCH get a "say": not in this harness.

- Takes off / lands with tello_cmd. FOLLOW, HOLD and GUIDE need the drone flying (take off first: operator "t t",
  or --takeoff-follow). A successful takeoff (ours or the operator's) goes to HOLD, or to FOLLOW with --takeoff-follow.
  A refused takeoff (ack ok false) or no ack within TAKEOFF_TIMEOUT_S stays IDLE and says why.
- Publishes "target" kind person (with the locked track) in FOLLOW, GUIDE and HOLD, and re-publishes "mode" and
  "target" every REPUBLISH_S so late joiners (a restarted controller or tello_io) catch up.
- HOLD owns rc (RC_OWNER): rc 0 0 0 0 with src "mission" at RC_HZ.
- LAND on: any kill or tello_event kill (no extra command: the kill lands), ctrl_status.land_request, battery below
  MIN_BATTERY_PCT, or video age above VIDEO_LOST_S while flying (these send tello_cmd land). LAND -> IDLE once
  tello_state reports flying False. Landing by anything else while in a flying mode also goes IDLE.

    python -m flyfollow.runtime.mission_lite [--takeoff-follow] [--takeoff-timeout 10]
"""

from __future__ import annotations

import argparse
import itertools
import time

from flyfollow.runtime.bus import Publisher, Subscriber
from flyfollow.runtime.messages import MODES, RC_HZ, msg
from flyfollow.runtime.node import Rate, log, stop_event

NAME = "mission"
TOPICS_IN = ["mode_cmd", "tello_state", "tello_ack", "tello_cmd", "tello_event", "kill", "ctrl_status", "lock", "det"]
HARNESS_MODES = ("FOLLOW", "HOLD", "GUIDE", "LAND", "IDLE")
FLYING_MODES = ("FOLLOW", "HOLD", "GUIDE")
PERSON_MODES = ("FOLLOW", "GUIDE", "HOLD")
REPUBLISH_S = 1.0
TAKEOFF_TIMEOUT_S = 10.0  # sim takeoff "ok" after about 8 s (R2's lag test)
MIN_BATTERY_PCT = 25.0
VIDEO_LOST_S = 5.0
START_WAIT_S = 2.0  # wait this long for a first tello_state before announcing IDLE


class MissionLite:
    def __init__(self, pub: Publisher, sub: Subscriber, takeoff_follow: bool = False,
                 takeoff_timeout_s: float = TAKEOFF_TIMEOUT_S, rc_hz: float = RC_HZ):
        self.pub, self.sub = pub, sub
        self.takeoff_follow = takeoff_follow
        self.takeoff_timeout_s = takeoff_timeout_s
        self.rc_period = 1.0 / rc_hz
        self.mode = "IDLE"
        self.announced = False
        self.t0: float | None = None
        self.flying: bool | None = None
        self.lock_id: int | None = None
        self.last_det: dict | None = None
        self.takeoff: dict | None = None  # {"id", "t", "then", "ours"}
        self.auto_takeoff_sent = False
        self.t_mode_pub = -1e9
        self.t_rc = -1e9
        self._ids = itertools.count(1)
        self.transitions: list[tuple[str, str, str]] = []

    # ---------------------------------------------------------------------------------------------- output
    def say(self, text: str, priority: int = 1) -> None:
        self.pub.publish(msg("say", text=text, priority=priority))
        log(NAME, f"say: {text}")

    def set_mode(self, to: str, reason: str, now: float) -> None:
        prev, self.mode = self.mode, to
        self.announced = True
        self.transitions.append((prev, to, reason))
        self.pub.publish(msg("mode", **{"from": prev, "to": to, "reason": reason}))
        self.t_mode_pub = now
        self.publish_target()
        if prev != to:
            log(NAME, f"{prev} -> {to} ({reason})")
        if to == "HOLD":
            self.t_rc = -1e9  # hover rc right away

    def publish_target(self) -> None:
        if self.mode in PERSON_MODES:
            m = msg("target", mode=self.mode, kind="person")
            if self.lock_id is not None:
                m["track_id"] = self.lock_id
        else:
            m = msg("target", mode=self.mode, kind="none")
        self.pub.publish(m)

    def tello_cmd(self, cmd: str) -> str:
        cid = f"ml-{next(self._ids)}"
        self.pub.publish(msg("tello_cmd", id=cid, cmd=cmd))
        return cid

    def land(self, reason: str, now: float, send_cmd: bool = True) -> None:
        if self.mode == "LAND":
            return
        self.takeoff = None
        if send_cmd and self.flying is not False:
            self.tello_cmd("land")
        self.set_mode("LAND", reason, now)
        if self.flying is False:
            self.set_mode("IDLE", "on the ground", now)

    def start_takeoff(self, then: str, now: float) -> None:
        self.takeoff = {"id": self.tello_cmd("takeoff"), "t": now, "then": then, "ours": True}
        log(NAME, f"takeoff sent ({self.takeoff['id']}), then {then}")

    def after_takeoff(self, now: float) -> None:
        then = (self.takeoff or {}).get("then", "HOLD")
        self.takeoff = None
        if self.mode in ("IDLE", "HOLD"):
            if then == "FOLLOW" and self.lock_id is None:
                self.auto_lock()
            self.set_mode(then, "takeoff ok", now)

    def auto_lock(self) -> None:
        from flyfollow.runtime.operator import pick_user_track

        tid = pick_user_track(self.last_det)
        if tid is not None:
            self.lock_id = tid
            self.pub.publish(msg("lock", track_id=tid, source="mission"))
            log(NAME, f"auto lock track {tid}")

    # ---------------------------------------------------------------------------------------------- input
    def on_mode_cmd(self, m: dict, now: float) -> None:
        want = str(m.get("mode", "")).upper()
        if want not in MODES:
            self.say(f"unknown mode {want}")
            return
        if want not in HARNESS_MODES:
            self.say(f"{want} is not in this harness; use ReachGlass")
            return
        if want == "HOLD" and m.get("intent") == "stop" and self.mode == "GUIDE":
            want = "FOLLOW"  # "stop" in GUIDE means the user reached the object (intent.py)
        if want in ("LAND", "IDLE"):
            if self.flying is False and want == "IDLE":
                self.set_mode("IDLE", f"mode_cmd from {m.get('source', '?')}", now)
            else:
                self.land(f"mode_cmd {want} from {m.get('source', '?')}", now)
            return
        if not self.flying:
            self.say("on the ground: take off first")
            return
        if want != self.mode:
            self.set_mode(want, f"mode_cmd from {m.get('source', '?')}", now)

    def on_state(self, m: dict, now: float) -> None:
        was, self.flying = self.flying, m.get("flying")
        if not self.announced:
            if self.flying:
                self.set_mode("HOLD", "flying at start", now)
            else:
                self.set_mode("IDLE", "start", now)
        if self.takeoff_follow and not self.auto_takeoff_sent and self.flying is False and self.mode == "IDLE":
            self.auto_takeoff_sent = True
            self.start_takeoff("FOLLOW", now)
        if self.flying is False:
            if self.mode == "LAND" or (self.mode in FLYING_MODES and was):
                self.set_mode("IDLE", "on the ground", now)
            return
        if self.flying and self.mode == "IDLE" and self.takeoff is None and was is False:
            self.set_mode("HOLD", "flying (takeoff by someone else)", now)
        if not self.flying or self.mode == "LAND":
            return
        bat = m.get("bat_pct")
        if isinstance(bat, (int, float)) and bat < MIN_BATTERY_PCT:
            self.land(f"battery {bat}% < {MIN_BATTERY_PCT:g}%", now)
            return
        age = m.get("video_age_s")
        if isinstance(age, (int, float)) and age > VIDEO_LOST_S:
            self.land(f"video lost {age:.1f} s > {VIDEO_LOST_S:g} s", now)

    def on_ack(self, m: dict, now: float) -> None:
        if m.get("cmd") != "takeoff":
            return
        mine = self.takeoff is not None and m.get("id") == self.takeoff["id"]
        if m.get("ok"):
            if mine or self.takeoff is None:
                self.after_takeoff(now)
        elif mine:
            self.takeoff = None
            self.say(f"takeoff refused: {m.get('detail', '')}")

    def handle(self, m: dict, now: float) -> None:
        tp = m.get("topic")
        if m.get("replayed") and tp not in ("tello_state", "det"):
            return
        if tp == "mode_cmd":
            self.on_mode_cmd(m, now)
        elif tp == "tello_state":
            self.on_state(m, now)
        elif tp == "tello_ack":
            self.on_ack(m, now)
        elif tp == "tello_cmd" and m.get("cmd") == "takeoff" and m.get("src_node") != self.pub.name:
            self.takeoff = {"id": m.get("id"), "t": now, "then": "HOLD", "ours": False}  # the operator's takeoff
        elif tp == "kill":
            self.land(f"kill {m.get('action')} from {m.get('src_node', '?')}", now, send_cmd=False)
        elif tp == "tello_event" and m.get("kind") == "kill":
            self.land(f"backend {m.get('action', 'land')}: {m.get('reason', '')}", now, send_cmd=False)
        elif tp == "ctrl_status" and m.get("land_request") and self.mode in FLYING_MODES:
            self.land(f"controller land request: {m.get('land_reason')}", now)
        elif tp == "lock":
            tid = m.get("track_id")
            if tid != self.lock_id and m.get("src_node") != self.pub.name:
                self.lock_id = None if tid is None else int(tid)
                self.publish_target()
        elif tp == "det":
            self.last_det = m

    # ---------------------------------------------------------------------------------------------- loop
    def step(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.t0 is None:
            self.t0 = now
        for m in self.sub.drain(2000):
            self.handle(m, now)
        if not self.announced and now - self.t0 > START_WAIT_S:
            self.set_mode("IDLE", "start (no tello_state yet)", now)
        if self.takeoff is not None and now - self.takeoff["t"] > self.takeoff_timeout_s:
            if self.flying:
                self.after_takeoff(now)
            else:
                self.say(f"takeoff: no ack within {self.takeoff_timeout_s:g} s")
                self.takeoff = None
        if self.announced and now - self.t_mode_pub >= REPUBLISH_S:
            self.pub.publish(msg("mode", **{"from": self.mode, "to": self.mode, "reason": "periodic"}))
            self.publish_target()
            self.t_mode_pub = now
        if self.mode == "HOLD" and self.flying and now - self.t_rc >= self.rc_period * 0.9:
            self.pub.publish(msg("rc", lr=0, fb=0, ud=0, yaw=0, src="mission", mode="HOLD"))
            self.t_rc = now


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="minimal mode owner for the fly-controller harness")
    ap.add_argument("--takeoff-follow", action="store_true", help="take off at start, then FOLLOW (sim)")
    ap.add_argument("--takeoff-timeout", type=float, default=TAKEOFF_TIMEOUT_S)
    a = ap.parse_args(argv)
    stop = stop_event()
    sub = Subscriber(TOPICS_IN)
    pub = Publisher(NAME)
    ml = MissionLite(pub, sub, takeoff_follow=a.takeoff_follow, takeoff_timeout_s=a.takeoff_timeout)
    log(NAME, "mission_lite up" + (" (takeoff then FOLLOW)" if a.takeoff_follow else ""))
    rate = Rate(RC_HZ)
    try:
        while not stop.is_set():
            ml.step()
            rate.sleep()
    finally:
        log(NAME, f"down in {ml.mode}; transitions: " + ", ".join(f"{a}->{b}" for a, b, _ in ml.transitions))
        pub.close()
        sub.close()


if __name__ == "__main__":
    main()
