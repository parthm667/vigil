"""Runtime control loop and geometry (flyfollow.runtime.controller_runner, flyfollow.runtime.geometry).

Fast and drone-free: the runner talks to an InProcBus, time is simulated (step(now)), dets are scripted.
Fly-controller tests need the real brain and calibration in data/brains (skipped otherwise).
"""

from __future__ import annotations

import itertools
import json
import math

import pytest

from flyfollow.interfaces import brains_dir
from flyfollow.runtime import geometry as geo
from flyfollow.runtime.bus import InProcBus, Publisher, Subscriber
from flyfollow.runtime.controller_runner import ControllerRunner
from flyfollow.runtime.messages import validate

FX, FY, CX, CY = 921.0, 919.0, 480.0, 360.0
HAVE_BRAIN = (brains_dir() / "pursuit_core1.npz").exists() and any((brains_dir() / "init").glob("FLY-*__pursuit_core1.json"))
needs_brain = pytest.mark.skipif(not HAVE_BRAIN, reason="real brain + calibration not in data/brains")
CONTROLLERS = [pytest.param("pid"), pytest.param("fly", marks=needs_brain)]


# ============================================================================================== geometry
def test_bearing_and_elevation_level():
    assert geo.bearing_deg(CX + FX) == pytest.approx(45.0)
    b, e = geo.bearing_elevation(CX + FX, CY)
    assert (b, e) == (pytest.approx(45.0), pytest.approx(0.0, abs=1e-9))
    b, e = geo.bearing_elevation(CX, CY + FY)
    assert (b, e) == (pytest.approx(0.0, abs=1e-9), pytest.approx(-45.0))
    u, v = geo.pixel_of(-20.0, 5.0)
    b, e = geo.bearing_elevation(u, v)
    assert (b, e) == (pytest.approx(-20.0), pytest.approx(5.0))


def test_attitude_correction():
    # nose up 10 deg: the image center looks 10 deg up
    assert geo.bearing_elevation(CX, CY, pitch_deg=10.0)[1] == pytest.approx(10.0)
    # right side down 10 deg, pixel 20 deg right on the center row: ray (1, -tan20, 0) rolled ->
    # y = -cos10 tan20 = -0.35844, z = -sin10 tan20 = -0.063204 -> bearing 19.72, elevation -3.405
    b, e = geo.bearing_elevation(CX + FX * math.tan(math.radians(20)), CY, roll_deg=10.0)
    assert b == pytest.approx(19.72, abs=0.01)
    assert e == pytest.approx(-3.405, abs=0.01)
    assert geo.tello_attitude({"pitch_deg": -3, "roll_deg": 2}) == (-3.0 * geo.TELLO_PITCH_SIGN, 2.0 * geo.TELLO_ROLL_SIGN)
    assert geo.tello_attitude(None) == (0.0, 0.0)
    assert geo.tilt_ok(5, -7) and not geo.tilt_ok(9, 0)


def test_range_from_size_and_class():
    assert geo.range_from_size(919 * 0.23 / 2.0, 0.23) == pytest.approx(2.0)  # head 105.7 px at 2 m
    z, sd = geo.range_from_class(919 * 0.22 / 3.0, "water bottle")
    assert z == pytest.approx(3.0)
    assert sd == pytest.approx(3.0 * math.hypot(0.25, 3.0 / 67.393), rel=1e-3)  # 0.762 m
    assert geo.range_from_class(50, "unicorn") is None
    assert geo.range_from_size(0, 0.2) == math.inf


def test_class_priors_and_aliases():
    assert geo.class_size_m("bottle") == 0.22
    assert geo.class_size_m("Cell_Phone") == 0.15
    assert geo.class_size_m("phone") == 0.15
    assert geo.class_size_m("keys") == 0.08
    assert geo.class_size_m("mug") == 0.10
    assert geo.class_size_m("person_head") == 0.23
    assert geo.class_size_m("dining table") == 0.75
    assert geo.class_size_m("unicorn", 0.2) == 0.2
    assert geo.canonical_class("person_head") == "person_head"


def test_ground_range_and_height():
    # camera 1.0 m up, floor point 2.6 m ahead: v = 360 + 919 / 2.6 = 713.46
    assert geo.ground_range(CX, CY + FY / 2.6, 1.0) == pytest.approx(2.6)
    assert geo.ground_range(CX, CY - 10, 1.0) is None  # above the horizon
    # a point 0.5 m above the camera at 2 m: v = 360 - 919 * 0.25 = 130.25 -> 1.5 m above the floor
    assert geo.height_above_floor(CX, CY - FY * 0.25, 2.0, 1.0) == pytest.approx(1.5)


def test_user_height_method():
    # plan 5.4: user 1.72 m at 3.5 m, camera 1.2 m: feet v = 360 + 919 * 1.2 / 3.5, head v = 360 - 919 * 0.52 / 3.5
    vf, vh = CY + FY * 1.2 / 3.5, CY - FY * 0.52 / 3.5
    z, a = geo.user_height_range(CX, vh, vf, 1.72)
    assert z == pytest.approx(3.5)
    assert a == pytest.approx(1.2)
    assert geo.user_height_range(CX, vf, vh, 1.72) is None


def test_floor_visible_min_range():
    # plan 3.4: alt / tan(21.4 deg) = alt * 919 / 360: 1.28 m at 0.5, 2.55 at 1.0, 3.06 at 1.2
    assert geo.floor_visible_min_range(0.5) == pytest.approx(0.5 * 919 / 360)
    assert geo.floor_visible_min_range(1.0) == pytest.approx(2.5528, abs=1e-3)
    assert geo.floor_visible_min_range(1.2) == pytest.approx(3.0633, abs=1e-3)
    k = geo.DEFAULT_K
    assert k.vfov_deg == pytest.approx(42.8, abs=0.1) and k.hfov_deg == pytest.approx(55.0, abs=0.1)


def test_drone_level_xy():
    x, y = geo.drone_level_xy(30.0, 2.0)
    assert (x, y) == (pytest.approx(math.sqrt(3)), pytest.approx(-1.0))  # right = negative y
    assert geo.xy_to_bearing_range(x, y) == (pytest.approx(30.0), pytest.approx(2.0))
    assert geo.wrap_deg(190.0) == pytest.approx(-170.0) and geo.wrap_deg(-180.0) == 180.0


def test_head_box_from_person():
    cx, cy, h = geo.head_box_from_person([400, 100, 520, 700], 1.72, 0.23)  # full body: 600 * 0.23 / 1.72
    assert (cx, h) == (460.0, pytest.approx(80.23, abs=0.01))
    assert cy == pytest.approx(100 + 40.12, abs=0.01)
    _, _, h = geo.head_box_from_person([400, 100, 520, 720], 1.72, 0.23)  # feet cut: width 120 * 0.23 / 0.45
    assert h == pytest.approx(61.33, abs=0.01)


def test_approach_z_ref_rule():
    # our config: Z_ref = clip(3.0 x (max(0.8, center) - base), 1.0, 2.4)
    assert geo.approach_z_ref(0.0, 0.11) == pytest.approx(2.4)  # floor bottle: 3.0 x 0.8 = 2.4
    assert geo.approach_z_ref(0.75, 0.86) == pytest.approx(1.0)  # table top
    assert geo.approach_z_ref(0.4, 0.5) == pytest.approx(1.2)


def test_camera_json(tmp_path):
    p = tmp_path / "camera.json"
    p.write_text(json.dumps({"camera_matrix": [[1842, 0, 962], [0, 1838, 718], [0, 0, 1]], "width": 1920, "height": 1440}))
    k = geo.load_intrinsics(p)
    assert (k.fx, k.fy, k.cx, k.cy) == (921.0, 919.0, 481.0, 359.0)
    assert geo.load_intrinsics(tmp_path / "missing.json") == geo.DEFAULT_K
    assert geo.load_intrinsics(p, {"fx": 900.0}).fx == 900.0


# ============================================================================================== runner rig
def head(bearing_deg: float, range_m: float, tid: int | None = 3, cy: float = 300.0, person: bool = True, with_head: bool = True):
    h = FY * 0.23 / range_m
    cx = CX + FX * math.tan(math.radians(bearing_deg))
    out = []
    if with_head:
        out.append({"cls": "person_head", "conf": 0.9, "bbox": [cx - 0.4 * h, cy - h / 2, cx + 0.4 * h, cy + h / 2], "track_id": tid})
    if person:
        out.append({"cls": "person", "conf": 0.9, "bbox": [cx - 0.9 * h, cy - h / 2, cx + 0.9 * h, 720.0], "track_id": tid})
    return out


def obj(cls: str, bearing_deg: float, range_m: float, size_m: float, tid: int | None = None, cy: float = 420.0):
    h = FY * size_m / range_m
    cx = CX + FX * math.tan(math.radians(bearing_deg))
    return {"cls": cls, "conf": 0.8, "bbox": [cx - h / 3, cy - h / 2, cx + h / 3, cy + h / 2], "track_id": tid}


class Rig:
    def __init__(self, controller: str = "pid", mode: str | None = "FOLLOW", **kw):
        self.bus = InProcBus()
        self.pub = Publisher("test", bus=self.bus)
        self.out = Subscriber(["rc", "ctrl_status"], bus=self.bus)
        self.r = ControllerRunner(bus=self.bus, controller=controller, **kw)
        self.t = 1000.0
        self.frame = 0
        self.rcs: list[dict] = []
        self.status: list[dict] = []
        self.yaw_deg = 0.0
        if mode:
            self.mode(mode)

    def send(self, topic: str, **f) -> None:
        self.pub.publish({"topic": topic, "t": self.t, **f})

    def mode(self, to: str) -> None:
        self.send("mode", **{"from": self.r.mode, "to": to, "reason": "test"})

    def state(self, h_cm: int = 150, bat: int = 80) -> None:
        self.send("tello_state", yaw_deg=self.yaw_deg, pitch_deg=0, roll_deg=0, vgx_dms=0, vgy_dms=0, vgz_dms=0, h_cm=h_cm,
                  tof_cm=h_cm, bat_pct=bat, temph_c=50, flying=True, video_ok=True, video_age_s=0.0, sending=False)

    def det(self, dets: list[dict], t_decoded: float | None = None, t: float | None = None) -> None:
        self.frame += 1
        self.pub.publish({"topic": "det", "t": self.t if t is None else t, "frame_id": self.frame,
                          "t_decoded": self.t - 0.03 if t_decoded is None else t_decoded, "src": "test", "img_w": 960,
                          "img_h": 720, "dets": dets})

    def run(self, seconds: float, dets=None, h_cm: int = 150) -> list[dict]:
        """Advance simulated time in 50 ms ticks; dets(t) returns a det list (or None for no message)."""
        new = []
        for _ in range(round(seconds / 0.05)):
            self.t = round(self.t + 0.05, 6)
            self.state(h_cm)
            if dets is not None:
                d = dets(self.t)
                if d is not None:
                    self.det(d)
            self.r.step(self.t)
            for m in self.out.drain():
                (self.rcs if m["topic"] == "rc" else self.status).append(m)
                if m["topic"] == "rc":
                    new.append(m)
        return new

    def close(self) -> None:
        self.r.close()


@pytest.fixture
def rig():
    rigs = []

    def make(*a, **kw):
        r = Rig(*a, **kw)
        rigs.append(r)
        return r

    yield make
    for r in rigs:
        r.close()


# ============================================================================================== runner tests
@pytest.mark.parametrize("ctrl", CONTROLLERS)
def test_target_right_turns_right(rig, ctrl):
    g = rig(ctrl)
    rcs = g.run(2.0, lambda t: head(20.0, 2.0))
    assert rcs, "no rc in FOLLOW"
    assert all(m["yaw"] > 0 for m in rcs[-10:]), [m["yaw"] for m in rcs[-10:]]
    g2 = rig(ctrl)
    rcs = g2.run(2.0, lambda t: head(-25.0, 2.0))
    assert all(m["yaw"] < 0 for m in rcs[-10:])
    assert g.status[-1]["controller"] == ctrl and g.status[-1]["target_valid"]
    assert g.status[-1]["range_m"] == pytest.approx(2.0, rel=0.02)


@pytest.mark.parametrize("ctrl", CONTROLLERS)
def test_range_loop_far_forward_close_back(rig, ctrl):
    far = rig(ctrl).run(2.0, lambda t: head(8.0, 3.5))  # 8 deg = the side offset: steering is centered
    assert all(m["fb"] > 0 for m in far[-10:]) and max(m["fb"] for m in far) <= 60  # max_fwd_stick
    close = rig(ctrl).run(1.0, lambda t: head(8.0, 1.0))  # inside min_person_dist_m 1.2
    assert all(m["fb"] < 0 for m in close[-5:])
    assert all(m["gov"]["safety"] and "min_distance" in m["gov"]["reasons"] for m in close[-5:])
    assert min(m["fb"] for m in close) >= -40  # max_back_stick


def test_no_rc_in_mission_modes(rig):
    g = rig("pid", mode=None)
    assert g.run(0.5, lambda t: head(20.0, 2.0)) == []  # IDLE
    for mode in ("FIND", "FACE_PERSON", "OVERWATCH", "RETURN", "HOLD", "LAND"):
        g.mode(mode)
        assert g.run(0.5, lambda t: head(20.0, 2.0)) == [], mode
    assert g.status and g.status[-1]["mode"] == "LAND" and not g.status[-1]["rc_owner"]
    g.mode("FOLLOW")
    assert g.run(0.2, lambda t: head(20.0, 2.0))


def test_kill_stops_rc_until_next_mode(rig):
    g = rig("pid")
    assert g.run(0.5, lambda t: head(0.0, 2.0))
    g.send("kill", action="land")
    assert g.run(0.5, lambda t: head(0.0, 2.0)) == []
    assert g.status[-1]["killed"]
    g.mode("GUIDE")
    assert g.run(0.2, lambda t: head(0.0, 2.0))


@needs_brain
def test_hot_swap_fly_pid(rig):
    g = rig("fly")
    g.run(2.0, lambda t: head(20.0, 2.5))
    assert g.r.active == "fly"
    g.send("settings", controller="pid")
    g.run(1.0, lambda t: head(20.0, 2.5))
    assert g.r.active == "pid" and g.rcs[-1]["controller"] == "pid" and g.status[-1]["arm"] == "PID-HAND"
    g.send("settings", controller="fly")
    g.run(1.0, lambda t: head(20.0, 2.5))
    assert g.r.active == "fly" and g.status[-1]["arm"] == "FLY-YAW-HAND"
    ys = [m["yaw"] for m in g.rcs]
    fs = [m["fb"] for m in g.rcs]
    # no jump: moves away from zero stay inside the governor slew (20 yaw, 10 fb per tick), including both swaps
    for a, b in itertools.pairwise(ys):
        assert abs(b) <= abs(a) or abs(b - a) <= 20
    for a, b in itertools.pairwise(fs):
        assert abs(b) <= abs(a) or abs(b - a) <= 10
    assert all(y > 0 for y in ys[-10:])


def test_guide_holds_yaw_forward_zero(rig):
    g = rig("pid", mode="GUIDE")
    calls = []
    real_act = g.r.ctrl.act
    g.r.ctrl.act = lambda *a: calls.append(1) or real_act(*a)
    rcs = g.run(2.0, lambda t: head(20.0, 3.5))  # far and to the right: yaw right, no forward
    assert all(m["fb"] == 0 for m in rcs)
    assert all(m["yaw"] > 0 for m in rcs[-10:])
    assert rcs[-1]["controller"] == "guide_pid" and not calls  # the brain / fly controller is idle
    back = g.run(1.0, lambda t: head(0.0, 1.0))  # user inside 1.2 m: back off
    assert all(m["fb"] < 0 for m in back[-5:]) and "min_distance" in back[-1]["gov"]["reasons"]


class Boom:
    name = "BOOM"

    def reset(self, st, seed):
        pass

    def warmup(self, s):
        pass

    def act(self, box, st, dt):
        raise RuntimeError("brain died")


def test_brain_watchdog(rig):
    g = rig("pid")
    g.run(1.0, lambda t: head(20.0, 3.0))
    g.r.ctrl = Boom()
    rcs = g.run(1.0, lambda t: head(20.0, 3.0))
    dog = [m for m in rcs if "watchdog" in m["gov"]["reasons"]]
    assert len(dog) >= 8  # brain age passes 0.5 s after 10 failed ticks
    assert all(m["yaw"] == 0 and m["fb"] == 0 and m["gov"]["safety"] for m in dog)
    assert "act_error" in g.status[-1]["warnings"]


def test_slow_act_triggers_watchdog_and_fallback(rig):
    from flyfollow.pilot.pid import PIDController

    ticks = itertools.count(0.0, 0.06)  # every perf() call advances 60 ms: each act() measures 60 ms
    g = rig("pid", auto_fallback=True, perf=lambda: next(ticks))
    g.r._ctrls[("fly", None)] = PIDController(name="FAKE-FLY")  # stands in for the fly (the rule only watches "fly")
    g.send("settings", controller="fly")
    g.run(4.0, lambda t: head(0.0, 2.0))
    assert g.r.active == "fly" and not g.r.fallback  # the fly must be active 5 s before the rule judges it
    g.run(2.0, lambda t: head(0.0, 2.0))
    assert "brain_over_budget" in g.r.warnings and g.r.fallback
    assert g.r.status(g.t)["brain_tick_ms_p95"] == pytest.approx(60.0)
    g.run(0.5, lambda t: head(0.0, 2.0))
    assert g.status[-1]["fallback"] and g.status[-1]["controller"] == "pid"
    slow = itertools.count(0.0, 0.6)  # 600 ms acts: the watchdog hovers
    g.r._perf = lambda: next(slow)
    rcs = g.run(0.3, lambda t: head(20.0, 2.0))
    assert all("watchdog" in m["gov"]["reasons"] and m["yaw"] == 0 for m in rcs)


@needs_brain
def test_auto_fallback_off_only_warns(rig):
    ticks = itertools.count(0.0, 0.03)  # 30 ms per call pair gap -> measured 30 ms, then raise it
    g = rig("fly", perf=lambda: next(ticks))
    g.run(6.0, lambda t: head(0.0, 2.0))
    assert "brain_over_budget" not in g.r.warnings
    ticks2 = itertools.count(0.0, 0.05)
    g.r._perf = lambda: next(ticks2)
    g.run(6.0, lambda t: head(0.0, 2.0))
    assert "brain_over_budget" in g.status[-1]["warnings"] and g.r.active == "fly" and not g.r.fallback


@pytest.mark.parametrize("det_time", [0.05, 0.25])
def test_late_det_is_latency_compensated(rig, det_time):
    g = rig("pid")
    lat = g.r.cfg["video_latency_s"]
    v = 120.0  # px/s, constant

    def cx_at(t):
        return 300.0 + v * (t - 1000.0)

    for _ in range(60):
        g.t = round(g.t + 0.05, 6)
        g.state()
        tc = g.t - lat - det_time  # captured, decoded after the video latency, boxed det_time later (arrives now)
        h = FY * 0.23 / 2.0
        x = cx_at(tc)
        g.det([{"cls": "person_head", "conf": 0.9, "bbox": [x - 0.4 * h, 250, x + 0.4 * h, 250 + h], "track_id": 3}],
              t_decoded=tc + lat)
        g.r.step(g.t)
    est = g.r.filter.output(g.t).cx
    raw_err = abs(cx_at(g.t - lat - det_time) - cx_at(g.t))
    assert raw_err == pytest.approx(v * (lat + det_time))
    assert abs(est - cx_at(g.t)) < 0.1 * raw_err


def test_messages_validate(rig):
    g = rig("pid")
    g.run(1.0, lambda t: head(10.0, 2.0))
    assert g.rcs and g.status
    for m in g.rcs + g.status:
        assert validate(m) == [], m
        json.dumps(m, allow_nan=False)  # strict JSON: no NaN / inf
    rc = g.rcs[-1]
    assert rc["src"] == "controller" and rc["mode"] == "FOLLOW" and rc["lr"] == 0
    assert all(isinstance(rc[k], int) and -100 <= rc[k] <= 100 for k in ("lr", "fb", "ud", "yaw"))
    assert set(rc["gov"]) == {"safety", "clamped", "reasons"}
    assert 9 <= len(g.status) <= 11  # 10 Hz


def test_head_missing_falls_back_to_person_box(rig):
    g = rig("pid")
    g.run(1.0, lambda t: head(15.0, 2.0, with_head=False))
    s = g.status[-1]
    assert s["target_valid"] and s["target"]["src"].endswith("person_top")
    assert g.rcs[-1]["yaw"] > 0


def test_track_id_change_is_associated(rig):
    g = rig("pid")
    g.send("lock", track_id=3)
    g.run(1.0, lambda t: head(10.0, 2.0, tid=3) + head(-20.0, 2.5, tid=9))
    # the detector re-IDs the user as 7 (same place); the stranger 9 stays where it was
    g.run(1.0, lambda t: head(10.5, 2.0, tid=7) + head(-20.0, 2.5, tid=9))
    s = g.status[-1]
    assert s["target_valid"] and s["target"]["track_id"] == 7 and s["target"]["lock_id"] == 3
    assert s["bearing_deg"] == pytest.approx(10.5, abs=0.5)
    # no track ids at all: association by position and size keeps the user
    g.run(1.0, lambda t: head(11.0, 2.0, tid=None) + head(-20.0, 2.5, tid=None))
    assert g.status[-1]["bearing_deg"] == pytest.approx(11.0, abs=0.5)


def test_unlocked_needs_single_person(rig):
    g = rig("pid")
    g.run(1.0, lambda t: head(10.0, 2.0, tid=None) + head(-20.0, 2.5, tid=None))
    assert not g.status[-1]["target_valid"]  # two people, no lock: follow nobody
    g2 = rig("pid")
    g2.run(1.0, lambda t: head(10.0, 2.0, tid=None))
    assert g2.status[-1]["target_valid"] and g2.status[-1]["target"]["lock_id"] is None
    assert g2.status[0]["target"]["src"] == "single:head"  # then kept by association


def test_lost_target_requests_land_but_never_lands(rig):
    g = rig("pid")
    g.run(1.0, lambda t: head(10.0, 2.0))
    rcs = g.run(11.0, lambda t: None)
    assert any("lost_target" in m["gov"]["reasons"] for m in rcs)
    assert any("lost_hover" in m["gov"]["reasons"] for m in rcs)
    assert g.status[-1]["land_request"] and g.status[-1]["land_reason"] == "lost_land"
    assert rcs[-1]["yaw"] == rcs[-1]["fb"] == rcs[-1]["ud"] == 0  # hover; landing is the mission's call
    assert any(s["lost_2s"] for s in g.status)


def test_approach_object_target(rig):
    g = rig("pid", mode="APPROACH")
    g.send("target", mode="APPROACH", kind="object", cls="bottle", z_ref_m=1.2)
    dets = lambda t: [obj("bottle", -15.0, 2.0, 0.22), obj("cup", 10.0, 1.5, 0.10)] + head(5.0, 3.0)
    rcs = g.run(1.5, dets)
    s = g.status[-1]
    assert s["target"]["cls"] == "bottle" and s["range_m"] == pytest.approx(2.0, rel=0.03)
    assert s["z_ref_m"] == 1.2 and s["z_min_m"] == 0.5 and s["side_offset_deg"] == 0.0
    assert all(m["yaw"] < 0 and m["fb"] > 0 for m in rcs[-5:])
    g2 = rig("pid", mode="APPROACH")
    g2.send("target", mode="APPROACH", kind="object", cls="bottle", z_ref_m=1.2)
    close = g2.run(1.0, lambda t: [obj("bottle", 0.0, 0.4, 0.22)])  # inside the 0.5 m object z_min
    assert close[-1]["fb"] < 0 and "min_distance" in close[-1]["gov"]["reasons"]


def test_approach_picks_bottle_nearest_expected_bearing(rig):
    g = rig("pid", mode="APPROACH")
    g.state()
    g.send("target", mode="APPROACH", kind="object", cls="bottle", bearing_deg=-10.0, size_m=0.22)
    g.run(1.0, lambda t: [obj("bottle", 25.0, 2.0, 0.22), obj("bottle", -12.0, 3.0, 0.22)])
    s = g.status[-1]
    assert s["bearing_deg"] == pytest.approx(-12.0, abs=0.5) and s["range_m"] == pytest.approx(3.0, rel=0.03)
    assert s["z_ref_m"] >= 1.0  # plan 5.3 standoff estimated from the box (no z_ref_m sent)


def test_run_real_time(rig):
    g = rig("pid")
    g.r.run(duration_s=1.0)
    assert 18 <= g.r.n_ticks <= 21
    s = g.r.status(g.r._t_last_tick)
    assert s["loop_hz"] == pytest.approx(20.0, rel=0.1)


def test_cli_over_zmq(monkeypatch):
    """main() against a real broker on spare ports: rc flows in the initial mode, ctrl_status at 10 Hz."""
    import socket

    from flyfollow.runtime import bus as busmod
    from flyfollow.runtime.controller_runner import main

    ports = []
    for _ in range(2):
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            ports.append(so.getsockname()[1])
    pub_a, sub_a = (f"tcp://127.0.0.1:{p}" for p in ports)
    monkeypatch.setenv("FLYFOLLOW_PUB_ADDR", pub_a)
    monkeypatch.setenv("FLYFOLLOW_SUB_ADDR", sub_a)
    with busmod.Broker(pub_a, sub_a):
        sub = busmod.Subscriber(["rc", "ctrl_status"])
        assert main(["--controller", "pid", "--mode", "FOLLOW", "--duration", "1.0"]) == 0
        got = sub.drain()
        sub.close()
    rcs = [m for m in got if m["topic"] == "rc"]
    assert len(rcs) >= 15 and all(validate(m) == [] for m in got)
    assert all(m["yaw"] == 0 and m["fb"] == 0 for m in rcs)  # no target: lost-target handling, nothing to turn to


@pytest.mark.parametrize("ctrl", CONTROLLERS)
@pytest.mark.parametrize("seed", [0, 1])
def test_closed_loop_follow_with_sim_world(monkeypatch, ctrl, seed):
    """sim_world (synthetic det, the demo-profile Tello) + this runner on one InProcBus, 30 s of FOLLOW after takeoff.

    Both processes stamp messages with time.time(), so the test swaps in a simulated wall clock.
    """
    import time as _time

    sw = pytest.importorskip("flyfollow.runtime.sim_world")
    clock = [2_000_000.0]
    monkeypatch.setattr(_time, "time", lambda: clock[0])
    bus = InProcBus()
    sim = sw.SimWorld(Publisher("sim_world", bus=bus), Subscriber(sw.TOPICS_IN, bus=bus), scenario_name="follow", seed=seed)
    r = ControllerRunner(bus=bus, controller=ctrl)
    mission = Publisher("mission", bus=bus)
    status = Subscriber(["ctrl_status"], bus=bus)
    mission.publish({"topic": "tello_cmd", "id": "takeoff-1", "cmd": "takeoff"})
    mission.publish({"topic": "mode", "from": "IDLE", "to": "FOLLOW", "reason": "test"})
    n = n_view = n_valid = 0
    d_min = math.inf
    for _ in range(int(60 / 0.05)):
        clock[0] += 0.05
        sim.tick(clock[0])
        r.step(clock[0])
        if sim.phase != "flying":
            continue
        dm, u = sim.dm, sim.user
        dx, dy = u.x - dm.x, u.y - dm.y
        d_min = min(d_min, math.hypot(dx, dy))
        bearing = math.degrees((math.atan2(dy, dx) - dm.psi + math.pi) % (2 * math.pi) - math.pi)
        n += 1
        n_view += abs(bearing) < 27.5  # HFOV 55 deg
        n_valid += r.filter.output(clock[0]).valid
        if n >= 600:
            break
    st = status.drain()[-1]
    r.close()
    assert n == 600, "never reached 30 s of flight"
    assert n_view / n >= 0.9 and n_valid / n >= 0.85, (n_view / n, n_valid / n)
    assert not sim.collisions and d_min > 0.8, (dict(sim.collisions), d_min)
    assert len(sim.user.script) <= 2  # the user actually walked the path
    assert not st["land_request"] and st["controller"] == ctrl


@pytest.mark.parametrize("ctrl", CONTROLLERS)
def test_viz_frames(rig, ctrl):
    """--viz: one frame per owned tick through VizSink (non-blocking), with the brain hooks and the latest camera frame."""
    import pickle
    import socket
    import time as _time

    import numpy as np
    import zmq

    from flyfollow.runtime.bus import FrameRing

    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        addr = f"tcp://127.0.0.1:{so.getsockname()[1]}"
    ring = FrameRing.create(name=f"ff_test_{addr.rsplit(':', 1)[1]}", slots=2)
    try:
        g = rig(ctrl, viz=True, viz_address=addr, frame_ring=ring)
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.connect(addr)
        _time.sleep(0.2)  # slow joiner
        img = np.full((720, 960, 3), 77, np.uint8)
        ring.write(img, 1, _time.time())
        g.run(0.5, lambda t: head(20.0, 2.0))
        frames = []
        while sub.poll(200):
            frames.append(pickle.loads(sub.recv()))
        sub.close(0)
        assert frames, "no viz frames"
        fr = frames[-1]
        assert fr["meta"]["source"] == "drone" and fr["meta"]["mode"] == "FOLLOW"
        assert fr["target"]["valid"] and fr["target"]["bearing_deg"] == pytest.approx(20.0, abs=0.5)
        assert fr["sticks"]["yaw"] == g.rcs[-1]["yaw"]
        if ctrl == "fly":
            assert fr["counts"].size > 1000 and "DNa02_R" in fr["dn"] and "yaw_drive" in fr["readout"]
        assert fr["image"].shape == (360, 480, 3) and int(fr["image"][0, 0, 0]) == 77  # downscaled in the sink
        assert g.r.sink.stats()["publish_ms_max"] < 20.0
    finally:
        ring.close()
        ring.unlink()


def test_bad_settings_are_ignored_and_good_ones_apply(rig):
    g = rig("pid")
    g.send("settings", fx=None, max_fwd_stick="fast", controller="banana", follow_distance_m=3.0, video_latency_s=0.3)
    g.run(1.0, lambda t: head(8.0, 2.0))
    s = g.status[-1]
    assert {"bad_setting_fx", "bad_setting_max_fwd_stick", "bad_setting_controller"} <= set(s["warnings"])
    assert s["z_ref_m"] == 3.0 and g.r.filter.latency_s == 0.3 and g.r.active == "pid"
    assert all(m["fb"] < 0 for m in g.rcs[-5:])  # 2 m is now too close for a 3 m standoff


def test_unlock_and_user_tracking_in_mission_modes(rig):
    g = rig("pid")
    g.send("lock", track_id=5)
    g.run(1.0, lambda t: head(10.0, 2.0, tid=5) + head(-20.0, 2.5, tid=9))
    assert g.status[-1]["target"]["lock_id"] == 5 and g.status[-1]["bearing_deg"] == pytest.approx(10.0, abs=0.5)
    g.send("lock", track_id=None)  # null = unlock: with two people in view, follow nobody
    g.run(1.0, lambda t: head(10.0, 2.0, tid=5) + head(-20.0, 2.5, tid=9))
    assert g.status[-1]["target"]["lock_id"] is None and not g.status[-1]["target_valid"]
    g.send("lock", track_id=9)
    g.mode("FACE_PERSON")  # mission owns rc; the loop still reports where the user is
    rcs = g.run(1.0, lambda t: head(10.0, 2.0, tid=5) + head(-20.0, 2.5, tid=9))
    s = g.status[-1]
    assert rcs == [] and s["target_valid"] and s["bearing_deg"] == pytest.approx(-20.0, abs=0.5)
    assert s["range_m"] == pytest.approx(2.5, rel=0.03)
