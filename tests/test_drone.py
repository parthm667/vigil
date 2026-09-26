"""Step 9: Tello adapter (against a fake djitellopy) and the safety governor."""

import threading
import time

import pytest

from reachglass.config import SafetyCfg
from reachglass.drone import SafetyGovernor
from reachglass.drone.tello import TelloDrone
from reachglass.sim.drone_sim import SimDroneParams
from reachglass.sim.scenario import Sim
from reachglass.sim.world import World


class FakeTello:
    """Mimics the djitellopy surface TelloDrone uses. Replies arrive asynchronously like the UDP thread's."""

    def __init__(self, delays=None, silent=()):
        self.sent = []
        self.udp = {"responses": [], "state": {}}
        self.is_flying = False
        self.delays = {"takeoff": 0.2, "land": 0.2, "move": 0.15, "rotate": 0.1, "stop": 0.02, "speed": 0.01}
        self.delays.update(delays or {})
        self.silent = set(silent)  # commands that never get a reply
        self.error_for = {}  # command prefix -> error text
        self.state = {"yaw": 10, "pitch": -2, "roll": 1, "h": 80, "tof": 85, "vgx": 3, "vgy": -1, "vgz": 0,
                      "bat": 87, "time": 5}
        self.ended = False

    def connect(self):
        self.sent.append("command")

    def send_control_command(self, cmd):
        self.sent.append(cmd)
        return True

    def streamon(self):
        self.sent.append("streamon")

    def get_udp_video_address(self):
        return "udp://@0.0.0.0:11111"

    def get_own_udp_object(self):
        return self.udp

    def get_current_state(self):
        return self.state

    def send_rc_control(self, *v):
        self.sent.append(("rc",) + tuple(v))

    def reply_later(self, text, delay):
        threading.Timer(delay, lambda: self.udp["responses"].append(text.encode())).start()

    def send_command_without_return(self, cmd):
        self.sent.append(cmd)
        word = cmd.split()[0]
        if word in self.silent or word == "emergency":
            return
        kind = "move" if word in ("forward", "back", "left", "right", "up", "down") else ("rotate" if word in ("cw", "ccw") else word)
        err = next((e for p, e in self.error_for.items() if cmd.startswith(p)), None)
        self.reply_later(err or "ok", self.delays.get(kind, 0.05))

    def end(self):
        self.ended = True


def wait(d, timeout=3.0):
    t0 = time.time()
    while d.busy() and time.time() - t0 < timeout:
        time.sleep(0.005)
    return d.last_result()


def make(**kw):
    fake = FakeTello(**{k: v for k, v in kw.items() if k in ("delays", "silent")})
    d = TelloDrone(tello=fake, video=False, **{k: v for k, v in kw.items() if k not in ("delays", "silent")})
    d.connect()
    return d, fake


# ------------------------------------------------------------------ Tello adapter
def test_connect_sets_speed_and_takeoff_is_non_blocking():
    d, fake = make(move_speed_cm_s=40)
    assert fake.sent[:2] == ["command", "speed 40"]
    t0 = time.time()
    d.takeoff()
    assert time.time() - t0 < 0.15 and d.busy() and d.last_result() is None and not d.flying
    assert wait(d) == "ok" and d.flying and fake.is_flying


def test_move_reply_and_rc_blocked_while_busy():
    d, fake = make()
    d.takeoff()
    wait(d)
    d.move("forward", 100)
    d.rc(0, 0, 0, 50)  # must not reach the drone: it would abort the move
    assert not any(isinstance(s, tuple) for s in fake.sent)
    assert wait(d) == "ok"
    d.rc(0, 30, 0, 0)
    assert fake.sent[-1] == ("rc", 0, 30, 0, 0)
    d.rotate(-90)
    assert fake.sent[-1] == "ccw 90" and wait(d) == "ok"
    with pytest.raises(RuntimeError):
        d.move("up", 20)
        d.move("up", 20)  # second command while the first runs is a programming error


def test_error_reply_and_timeout():
    d, fake = make()
    fake.error_for["forward"] = "error Not joystick"
    d.takeoff()
    wait(d)
    d.move("forward", 50)
    assert wait(d).startswith("error: error Not joystick")
    d2, fake2 = make(silent=("up",))
    d2._start("up 20", timeout_s=0.2, expected_s=0.1)
    assert wait(d2) == "error: timeout"


def test_stop_discards_the_late_reply_of_the_cancelled_move():
    d, fake = make(delays={"move": 0.4})
    d.takeoff()
    wait(d)
    d.move("forward", 200)
    time.sleep(0.05)
    d.stop()
    assert not d.busy() and d.last_result() == "ok" and fake.sent[-1] == "stop"
    fake.error_for["cw"] = "error No valid imu"
    t0 = time.time()
    d.rotate(45)  # sent right after the stop: waits out the settle window first (stale 'ok's arrive meanwhile)
    assert time.time() - t0 >= 0.8
    assert wait(d) == "error: error No valid imu"  # the stale "ok"s were not credited to the rotation


def test_stale_replies_are_cleared_before_a_new_command():
    d, fake = make(silent=("cw",))
    d.takeoff()
    wait(d)
    fake.udp["responses"].append(b"ok")  # leftover
    d._start("cw 30", timeout_s=0.3, expected_s=0.1)
    assert wait(d) == "error: timeout"


def test_min_gap_between_commands():
    d, fake = make(delays={"move": 0.0, "rotate": 0.0})
    d.takeoff()
    wait(d)
    t0 = time.time()
    for _ in range(3):
        d.rotate(10)
        wait(d)
    assert time.time() - t0 >= 0.2  # >= 0.1 s between sends


def test_telemetry_conversion():
    d, fake = make(yaw_sign=-1)
    tel = d.telemetry()
    assert tel.yaw_deg == -10 and tel.pitch_deg == -2 and tel.height_m == pytest.approx(0.8)
    assert tel.tof_m == pytest.approx(0.85) and tel.vx == pytest.approx(0.3) and tel.vy == pytest.approx(-0.1)
    assert tel.battery_pct == 87 and abs(tel.t - time.time()) < 2
    fake.state["tof"] = 6553  # out of range
    assert d.telemetry().tof_m is None
    fake.state = {}
    assert d.telemetry().yaw_deg is None


def test_dry_run_never_sends_motion():
    d, fake = make(dry_run=True)
    d.takeoff()
    assert d.busy()
    time.sleep(0.01)
    d.rc(10, 10, 10, 10)
    d.emergency()
    d.takeoff()
    d._pending.expected_s = 0.05
    assert wait(d) == "ok" and d.flying
    d.move("forward", 50)
    d._pending.expected_s = 0.05
    assert wait(d) == "ok"
    assert fake.sent == ["command"]  # not even 'speed'
    assert any("forward 50" in c for _, c in d.log) and not fake.is_flying


def test_land_is_idempotent_and_never_stops_a_landing():
    d, fake = make(delays={"land": 0.3})
    d.takeoff()
    wait(d)
    d.move("forward", 100)
    d.land()  # cancels the move with 'stop', then lands
    assert fake.sent[-2:] == ["stop", "land"]
    n = len(fake.sent)
    d.land()  # again while landing: nothing sent (a 'stop' would abort the landing)
    assert len(fake.sent) == n
    assert wait(d) == "ok" and not d.flying


def test_close_lands_and_ends():
    d, fake = make()
    d.takeoff()
    wait(d)
    d.close()
    assert "land" in fake.sent and fake.ended


# ------------------------------------------------------------------ safety governor (on the simulator)
def sim_drone(**cfg):
    sim = Sim(World(size_x=8, size_y=8, boxes=[]), (4.0, 4.0), params=SimDroneParams(hover_drift=0))
    gov = SafetyGovernor(sim.drone, SafetyCfg(**cfg))
    gov.takeoff()
    while gov.busy():
        sim.step(0.05)
    return sim, gov


def test_governor_clamps_rc():
    sim, gov = sim_drone(max_rc=30)
    gov.rc(100, -100, 0, 80)
    sim.step(0.2)
    assert sim.drone._rc == (30, -30, 0, 30)


def test_governor_ceiling_and_floor():
    sim, gov = sim_drone(max_altitude_m=1.2, min_altitude_m=0.5)
    for _ in range(80):  # climb hard for 4 s
        gov.rc(0, 0, 100, 0)
        sim.step(0.05)
    assert sim.drone.pos[2] < 1.3
    gov.move("up", 100)
    assert gov.last_result().startswith("error: safety") and not gov.busy()
    for _ in range(120):
        gov.rc(0, 0, -100, 0)
        sim.step(0.05)
    assert sim.drone.pos[2] > 0.4


def test_governor_lands_on_low_battery_and_flight_time():
    sim, gov = sim_drone(min_battery_pct=50)
    sim.drone.battery = 40
    assert gov.check(time.time(), time.time()) == "land" and gov.busy()
    gov.rc(0, 50, 0, 0)  # ignored while landing
    while gov.busy():
        sim.step(0.05)
    assert not gov.flying
    sim, gov = sim_drone(max_flight_s=10)
    now = time.time()
    gov.check(now, now)
    assert gov.check(now + 11, now + 11) == "land"


def test_governor_hovers_when_video_goes_stale_and_recovers():
    sim, gov = sim_drone(frame_timeout_s=1.0)
    now = sim.t  # one clock for everything (the sim's)
    assert gov.check(now, now - 0.5) == "ok"
    gov.rc(0, 40, 0, 0)
    sim.step(0.3)
    assert gov.check(now, now - 2.0) == "hover"
    gov.rc(0, 40, 0, 0)  # the mission keeps asking, but the governor holds position
    sim.step(0.3)
    assert sim.drone._rc == (0, 0, 0, 0)
    now = sim.t
    assert gov.check(now, now) == "ok"
    gov.rc(0, 40, 0, 0)
    sim.step(0.3)
    assert sim.drone._rc == (0, 40, 0, 0)
    # telemetry that stops updating also means hover
    assert gov.check(now + 5.0, now + 5.0) == "hover"
