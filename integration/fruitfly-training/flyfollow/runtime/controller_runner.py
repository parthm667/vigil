"""Control loop process (plan 4.1 loop, 4.9, 5.3, 5.5): det -> BoxFilter -> controller -> PersonGovernor -> rc.

    python -m flyfollow.runtime.controller_runner [--controller fly|pid] [--params PATH|none] [--viz] [--hz 20]
        [--auto-fallback] [--mode FOLLOW] [--brain data/brains/pursuit_core1.npz] [--deadband 4] [--hysteresis 3]

Per tick (20 Hz):
    box = BoxFilter.output(now)                    shared latency filter, fed by every det of the selected target
    yaw, fb = controller.act(box, settings, dt)    FOLLOW / APPROACH: FLY-YAW (fly steers, PID-HAND range loop sets
                                                   forward) or PID-HAND; GUIDE: PID yaw law, fb = 0, brain idle
    yaw = YawShaper(yaw)                           fly only: deadband + hysteresis, as FlySteer (flyfollow/steer.py)
    go = PersonGovernor.filter(yaw, fb, box, ...)  ud, lost target, min distance, clamps, slew, watchdog, battery
    publish rc                                     only when RC_OWNER[mode] == "controller" and no kill since the last mode
Other modes: the brain is idle and nothing is sent, but the target is still tracked and ctrl_status still flows.
The governor's land request is never executed here: ctrl_status carries land_request and the mission lands.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from flyfollow.interfaces import IMG_H, IMG_W, BoxState, Settings
from flyfollow.pilot.box_filter import BoxFilter
from flyfollow.pilot.governor import PersonGovernor
from flyfollow.pilot.pid import PIDController
from flyfollow.runtime import geometry as geo
from flyfollow.runtime.messages import RC_OWNER, SETTINGS_DEFAULTS, msg

NAME = "controller"
TOPICS_IN = ["det", "tello_state", "settings", "lock", "target", "mode", "kill"]
BRAIN_MODES = ("FOLLOW", "APPROACH")
PERSON_MODES = ("FOLLOW", "GUIDE")
# modes whose default target (no target message) is the user: the mission reads their bearing / range from ctrl_status
USER_TRACK_MODES = ("FOLLOW", "GUIDE", "FACE_PERSON", "OVERWATCH", "RETURN", "HOLD")
BRAIN_P95_LIMIT_MS = 45.0  # plan 3.5
BRAIN_WINDOW_S = 5.0
STATS_WINDOW_S = 10.0
STATUS_HZ = 10.0
WARMUP_S = 1.0
BAND_FRAC = 0.15  # configs/env.yaml reward.band_frac
LOST_REPORT_S = 2.0  # plan 5.3: APPROACH target lost > 2 s (mission rescans)
IN_BAND_REPORT_S = 2.0  # plan 2.1: in the standoff band 2 s -> FACE_PERSON
TELLO_STATE_STALE_S = 2.0
OBJECT_Z_MIN_M = 0.5
GUIDE_CY_BAND = (0.25, 0.75)  # GUIDE holds altitude while the head stays inside this band of image rows
REACQUIRE_SINGLE_S = 3.0  # locked track unseen this long and exactly one person in view: take them
# the shipped fly (trained, then fine-tuned for a smoother stick); --params none = the untrained hand calibration
DEFAULT_PARAMS = "trained/FLY-YAW_smooth_best.json"


class YawShaper:
    """FlySteer's output conditioning (flyfollow/steer.py): deadband, then the sent stick moves only when the new
    value is more than `hysteresis` away. Cuts stick dither without adding lag (a low-pass added lag and wobble)."""

    def __init__(self, deadband: float = 0.0, hysteresis: float = 0.0):
        self.deadband = max(0.0, float(deadband))
        self.hysteresis = max(0.0, float(hysteresis))
        self.y = 0.0

    def reset(self) -> None:
        self.y = 0.0

    def __call__(self, yaw: float) -> float:
        y = math.copysign(max(0.0, abs(yaw) - self.deadband), yaw)
        if y == 0.0 or self.hysteresis <= 0 or abs(y - self.y) > self.hysteresis:
            self.y = float(round(y))
        return self.y


def load_env_config() -> dict:
    from flyfollow.rl.env import load_env_config as _load

    return _load()


def load_params_file(path: str | None) -> tuple[np.ndarray | None, str | None, str | None]:
    """(x, arm, brain) from a trainer best.json ('x' or 'params'); (None, None, None) for no file."""
    if not path or str(path).lower() == "none":
        return None, None, None
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = rec.get("arm") or "FLY-YAW"
    if rec.get("x") is not None:
        x = np.asarray(rec["x"], np.float64)
    elif rec.get("params"):
        from flyfollow.rl.params import param_space

        x = param_space(arm).encode(rec["params"])
    else:
        raise ValueError(f"{path}: expected 'x' or 'params'")
    return x, arm, rec.get("brain")


# ------------------------------------------------------------------------------------------------ small helpers
class Window:
    """(t, value) samples over a sliding time window; percentiles on demand."""

    def __init__(self, span_s: float):
        self.span = span_s
        self.q: collections.deque[tuple[float, float]] = collections.deque()

    def add(self, t: float, v: float) -> None:
        self.q.append((t, v))
        while self.q and self.q[0][0] < t - self.span:
            self.q.popleft()

    def values(self, now: float, span_s: float | None = None) -> list[float]:
        lo = now - (span_s or self.span)
        return [v for t, v in self.q if t >= lo]

    def pct(self, now: float, p: float, span_s: float | None = None) -> float | None:
        v = self.values(now, span_s)
        return float(np.percentile(v, p)) if v else None

    def clear(self) -> None:
        self.q.clear()


def _f(v, nd: int = 3):
    """JSON-safe rounded float (None for NaN / inf / None)."""
    if v is None:
        return None
    v = float(v)
    return round(v, nd) if math.isfinite(v) else None


@dataclass
class Target:
    """What the loop pursues in one mode (from the mission's target message, or the mode default)."""

    mode: str
    kind: str = "none"  # person | object | none
    cls: str | None = None
    track_id: int | None = None
    bearing_deg: float | None = None  # camera bearing at message time t (+ right)
    yaw_abs_deg: float | None = None  # bearing pinned to Tello yaw at receipt (None if yaw unknown then)
    range_m: float | None = None
    size_m: float | None = None
    z_ref_m: float | None = None
    z_min_m: float | None = None
    cy_ref_frac: float | None = None

    def key(self) -> tuple:
        return (self.kind, geo.canonical_class(self.cls) if self.cls else None, self.track_id)


@dataclass
class Cand:
    cx: float
    cy: float
    h: float
    track_id: int | None
    src: str  # head | person_top | object
    conf: float = 0.0
    cls: str = ""
    bbox: tuple = ()
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ runner
class ControllerRunner:
    """The control loop. Call step(now) at the loop rate (run() does that in real time).

    bus: an InProcBus for tests (None = the ZeroMQ broker). perf: the clock act() is timed with (tests may fake it).
    """

    def __init__(self, bus=None, controller: str | None = None, params_path: str | None = None, hz: float = 20.0,
                 viz: bool = False, viz_address: str | None = None, auto_fallback: bool = False, initial_mode: str = "IDLE",
                 brain_path: str | None = None, env_cfg: dict | None = None, camera_json: str | None = None,
                 reacquire_single: bool = True, frame_ring=None, verbose: bool = False, perf=time.perf_counter,
                 deadband: float = 0.0, hysteresis: float = 0.0):
        from flyfollow.runtime.bus import Publisher, Subscriber

        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.env_cfg = env_cfg if env_cfg is not None else load_env_config()
        self.auto_fallback = auto_fallback
        self.reacquire_single = reacquire_single
        self.verbose = verbose
        self._perf = perf
        self.brain_path = brain_path
        # settings: defaults <- configs/camera.json <- CLI <- settings messages
        self.cfg: dict = dict(SETTINGS_DEFAULTS)
        self.cfg.update(geo.camera_defaults(camera_json))
        if controller:
            self.cfg["controller"] = controller
        if params_path:
            self.cfg["params_path"] = params_path
        # pipeline pieces (shared with the sim)
        self.filter = BoxFilter.from_config(self.env_cfg.get("filter"), float(self.cfg["video_latency_s"]))
        self.gov = PersonGovernor(self.env_cfg.get("governor"))
        self.guide_pid = PIDController(name="GUIDE-PID")
        self.shaper = YawShaper(deadband, hysteresis)
        self._ctrls: dict[tuple, object] = {}
        self.ctrl = None
        self.ctrl_key: tuple | None = None
        self.active: str | None = None  # "fly" | "pid"
        self.fallback = False
        self.warnings: dict[str, float] = {}  # warning -> time raised
        # world state
        self.mode = "IDLE"
        self.killed = False
        self.state: dict | None = None
        self.state_t = -math.inf
        self.lock_id: int | None = None
        self.lock_t = -math.inf
        self.assoc_id: int | None = None  # track id the association currently follows (may differ from lock)
        self.targets: dict[str, Target] = {}
        self.target = Target("IDLE")
        self.last_det_t = -math.inf  # any det message
        self.last_sel: Cand | None = None
        self.last_sel_t = -math.inf
        self.last_lock_seen_t = -math.inf
        self.target_src = None
        self.n_det = self.n_sel = 0
        self._obj_zref: float | None = None
        # timing and status
        self.tick_ms = Window(STATS_WINDOW_S)
        self.loop_ms = Window(STATS_WINDOW_S)
        self.work_ms = Window(STATS_WINDOW_S)  # whole tick (filter, controller, governor, publish, viz)
        self.fly_since: float | None = None
        self._t_act_ok = -math.inf
        self._t_last_tick: float | None = None
        self._t_status = -math.inf
        self._in_band_since: float | None = None
        self._land_req: str | None = None
        self.last_rc: dict | None = None
        self.last_go = None
        self.n_ticks = 0
        self.n_rc = 0
        self.act_errors = 0
        self.tick_errors = 0
        self.t0: float | None = None
        # bus
        self.sub = Subscriber(TOPICS_IN, bus=bus)
        self.pub = Publisher(NAME, bus=bus)
        # viz
        self.sink = None
        self.ring = frame_ring
        self._ring_try = -math.inf
        if viz:
            from flyfollow.viz.live import DEFAULT_ADDRESS, VizSink

            self.sink = VizSink(viz_address or DEFAULT_ADDRESS)
        self._ensure_controller(time.time())
        if initial_mode != "IDLE":
            self._set_mode(initial_mode, time.time())

    # -------------------------------------------------------------------------------------------- messages
    def handle(self, m: dict, now: float | None = None) -> None:
        now = time.time() if now is None else now
        tp = m.get("topic")
        try:
            if tp == "det":
                self._on_det(m)
            elif tp == "tello_state":
                self.state = m
                self.state_t = float(m.get("t", now))
            elif tp == "settings":
                self._on_settings(m, now)
            elif tp == "lock":
                tid = m.get("track_id")
                tid = None if tid is None else int(tid)
                if tid != self.lock_id:
                    self.lock_id = self.assoc_id = tid
                    self.lock_t = now
                    if self.target.kind == "person":
                        self._reset_track()
            elif tp == "target":
                self._on_target(m, now)
            elif tp == "mode":
                self._set_mode(str(m.get("to", "")), now)
            elif tp == "kill":
                self.killed = True
                self._warn("killed", now)
        except (TypeError, ValueError, KeyError) as e:  # a malformed message must never stop the loop
            self._warn(f"bad_{tp}", now)
            if self.verbose:
                print(f"[{NAME}] bad {tp} message: {e!r}", flush=True)

    def _on_settings(self, m: dict, now: float) -> None:
        upd = {}
        for k, v in m.items():
            if k in ("topic", "t", "src_node"):
                continue
            d = SETTINGS_DEFAULTS.get(k)
            if isinstance(d, (int, float)) and not isinstance(d, bool):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = math.nan
                if not math.isfinite(v):
                    self._warn(f"bad_setting_{k}", now)
                    continue
            elif k == "controller" and str(v).lower() not in ("fly", "pid"):
                self._warn("bad_setting_controller", now)
                continue
            upd[k] = v
        if not upd:
            return
        self.cfg.update(upd)
        if "video_latency_s" in upd:
            self.filter.latency_s = float(self.cfg["video_latency_s"])
        if "controller" in upd or "params_path" in upd:
            self.fallback = False  # an explicit operator choice overrides the auto fallback
            self.warnings.pop("fallback_pid", None)
            self._ensure_controller(now)

    def _on_target(self, m: dict, now: float) -> None:
        def num(k):
            v = m.get(k)
            return None if v is None else float(v)

        tid = m.get("track_id")
        tg = Target(mode=str(m.get("mode", self.mode)), kind=str(m.get("kind", "none")), cls=m.get("cls"),
                    track_id=None if tid is None else int(tid), bearing_deg=num("bearing_deg"), range_m=num("range_m"),
                    size_m=num("size_m"), z_ref_m=num("z_ref_m"), z_min_m=num("z_min_m"), cy_ref_frac=num("cy_ref_frac"))
        yaw = self._yaw_now(now)
        if tg.bearing_deg is not None and yaw is not None:
            tg.yaw_abs_deg = yaw + tg.bearing_deg
        self.targets[tg.mode] = tg
        if tg.mode == self.mode:
            self._activate_target(tg)

    def _default_target(self, mode: str) -> Target:
        return Target(mode, kind="person") if mode in USER_TRACK_MODES else Target(mode)

    def _activate_target(self, tg: Target) -> None:
        if tg.key() != self.target.key():
            self._reset_track()
        self.target = tg
        self._obj_zref = None
        if tg.kind == "person" and tg.track_id is not None and tg.track_id != self.lock_id:
            self.lock_id = self.assoc_id = tg.track_id
            self.lock_t = time.time()

    def _reset_track(self) -> None:
        self.filter.reset()
        self.last_sel = None
        self.last_sel_t = -math.inf
        self._in_band_since = None
        self._obj_zref = None
        if self.target.kind == "object":
            self.assoc_id = None

    def _set_mode(self, mode: str, now: float) -> None:
        if mode not in RC_OWNER:
            self._warn("bad_mode", now)
            return
        self.killed = False  # a mode message after a kill re-arms the loop
        if mode == self.mode:
            return
        prev, self.mode = self.mode, mode
        self._activate_target(self.targets.get(mode) or self._default_target(mode))
        was_owned = RC_OWNER.get(prev) == "controller"
        if RC_OWNER.get(mode) == "controller" and not was_owned:
            self.gov.reset()  # the previous owner's sticks are gone (tello_io drops them on a mode change)
        if mode in BRAIN_MODES and prev not in BRAIN_MODES:
            self._warm(now)
        if mode not in BRAIN_MODES:
            self.fly_since = None
        self._in_band_since = None
        self._land_req = None

    # -------------------------------------------------------------------------------------------- controllers
    def _want(self) -> tuple:
        name = str(self.cfg.get("controller") or "fly").lower()
        if name not in ("fly", "pid"):
            name = "fly"
        if name == "fly" and self.fallback:
            name = "pid"
        return (name, self.cfg.get("params_path") if name == "fly" else None)

    def _build(self, key: tuple):
        from flyfollow.rl.controllers import make_controller

        name, path = key
        if name == "pid":
            return make_controller("PID-HAND")
        x, arm, brain = load_params_file(path)
        if x is None:
            return make_controller("FLY-YAW-HAND", brain_path=self.brain_path)
        if not arm.startswith(("FLY", "NOBRAIN")):
            raise ValueError(f"{path}: arm {arm} is not a fly arm")
        return make_controller(arm, x=x, brain_path=self.brain_path or brain)

    def _ensure_controller(self, now: float) -> None:
        key = self._want()
        if key == self.ctrl_key:
            return
        ctrl = self._ctrls.get(key)
        if ctrl is None:
            try:
                ctrl = self._build(key)
            except Exception as e:  # noqa: BLE001 (missing brain / calibration / bad params: fly on the PID)
                self._warn("fly_build_failed", now)
                print(f"[{NAME}] cannot build the fly controller ({e!r}); using PID-HAND", flush=True)
                key = ("pid", None)
                ctrl = self._ctrls.get(key) or self._build(key)
            self._ctrls[key] = ctrl
        prev = self.active
        self.ctrl, self.ctrl_key, self.active = ctrl, key, key[0]
        self.fly_since = None
        self._warm(now)
        # the governor keeps its slew state across a swap, so the sent sticks ramp from where they were
        if self.verbose or prev is not None:
            print(f"[{NAME}] controller {prev} -> {self.active} ({getattr(ctrl, 'name', '?')})", flush=True)

    def _warm(self, now: float) -> None:
        self.shaper.reset()
        st = self._settings(self.filter.output(now))
        self.ctrl.reset(st, 0)
        self.ctrl.warmup(WARMUP_S)
        self._t_act_ok = now
        self.fly_since = None

    # -------------------------------------------------------------------------------------------- detections
    def _fresh_state(self, now: float) -> dict | None:
        return None if self.state is None or now - self.state_t > TELLO_STATE_STALE_S else self.state

    def _yaw_now(self, now: float) -> float | None:
        s = self._fresh_state(now)
        y = None if s is None else s.get("yaw_deg")
        return None if y is None else float(y)

    def _intr(self) -> geo.Intrinsics:
        return geo.Intrinsics.from_settings(self.cfg)

    def _on_det(self, m: dict) -> None:
        t = float(m["t"])
        t_dec = float(m.get("t_decoded", t))
        t_dec = min(t_dec, t)  # clocks disagree (should not happen on one laptop): trust arrival
        self.last_det_t = max(self.last_det_t, t)
        self.n_det += 1
        if self.target.kind == "none":
            return
        sx = IMG_W / float(m.get("img_w") or IMG_W)
        sy = IMG_H / float(m.get("img_h") or IMG_H)
        dets = []
        for d in m.get("dets") or []:
            bb = d.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in bb)
            dets.append({"cls": geo.canonical_class(d.get("cls")), "conf": float(d.get("conf") or 0.0),
                         "bbox": (x1 * sx, y1 * sy, x2 * sx, y2 * sy), "track_id": d.get("track_id")})
        c = self._select_person(dets, t) if self.target.kind == "person" else self._select_object(dets, t)
        if c is None:
            return
        if self.filter.update(t, t_dec, c.cx, c.cy, c.h):
            self.last_sel, self.last_sel_t = c, t
            self.n_sel += 1
            if c.track_id is not None:
                self.assoc_id = c.track_id
            if self.target.kind == "object":
                self._update_obj_zref(c)

    def _predicted(self, t: float) -> tuple[float, float, float] | None:
        """Where the tracked box should be now: the filter while valid, else the last selection for 2 s."""
        b = self.filter.output(t)
        if b.valid:
            return b.cx, b.cy, b.h
        if self.last_sel is not None and t - self.last_sel_t < 2.0:
            return self.last_sel.cx, self.last_sel.cy, self.last_sel.h
        return None

    @staticmethod
    def _gate(c: Cand, p: tuple[float, float, float], t_gap: float) -> float | None:
        """Association cost of candidate c against prediction p (None = outside the gate)."""
        px, py, ph = p
        ratio = c.h / max(ph, 1.0)
        if not 0.5 < ratio < 2.0:
            return None
        d = math.hypot(c.cx - px, c.cy - py) / max(ph, 20.0)
        if d > 2.5 + 2.0 * max(0.0, t_gap):
            return None
        return d + abs(math.log(ratio))

    def _person_cands(self, dets: list[dict]) -> list[Cand]:
        persons = [d for d in dets if d["cls"] == "person"]
        out: list[Cand] = []
        covered: set[int] = set()
        for d in dets:
            if d["cls"] != "person_head":
                continue
            cx, cy, h = geo.box_center_h(d["bbox"])
            tid = d["track_id"]
            for i, p in enumerate(persons):  # head inside the top of a person box: same person
                x1, y1, x2, y2 = p["bbox"]
                inside = x1 <= cx <= x2 and y1 - 0.5 * h <= cy <= y1 + 0.4 * (y2 - y1)
                if inside and (tid is None or tid == p["track_id"]):
                    covered.add(i)
                    tid = p["track_id"] if tid is None else tid
                    break
            out.append(Cand(cx, cy, h, tid, "head", d["conf"], "person_head", d["bbox"]))
        for i, p in enumerate(persons):
            if i in covered or (p["track_id"] is not None and any(c.track_id == p["track_id"] for c in out)):
                continue
            cx, cy, h = geo.head_box_from_person(p["bbox"], float(self.cfg["user_height_m"]), float(self.cfg["head_size_m"]))
            out.append(Cand(cx, cy, h, p["track_id"], "person_top", p["conf"], "person", p["bbox"]))
        return out

    def _select_person(self, dets: list[dict], t: float) -> Cand | None:
        cands = self._person_cands(dets)
        if not cands:
            return None
        rank = lambda c: (c.src != "head", -c.conf)
        ids = {i for i in (self.lock_id, self.assoc_id) if i is not None}
        if ids:
            hit = sorted((c for c in cands if c.track_id in ids), key=rank)
            if hit:
                locked = self.lock_id is not None and hit[0].track_id == self.lock_id
                if locked:
                    self.last_lock_seen_t = t
                self.target_src = f"{'lock' if locked else 'track'}:{hit[0].src}"
                return hit[0]
        p = self._predicted(t)
        if p is not None:
            scored = [(s, c) for c in cands if (s := self._gate(c, p, t - self.last_sel_t)) is not None]
            if scored:
                c = min(scored, key=lambda sc: (sc[0], rank(sc[1])))[1]
                self.target_src = f"assoc:{c.src}"
                return c
        n_people = len({c.track_id if c.track_id is not None else id(c) for c in cands})
        unseen = t - max(self.lock_t, self.last_lock_seen_t, self.last_sel_t)
        if n_people == 1 and (self.lock_id is None or (self.reacquire_single and unseen > REACQUIRE_SINGLE_S)):
            self.target_src = "single:" + cands[0].src if self.lock_id is None else "reacquire:" + cands[0].src
            return min(cands, key=rank)
        return None

    def _select_object(self, dets: list[dict], t: float) -> Cand | None:
        want = geo.canonical_class(self.target.cls)
        cands = [Cand(*geo.box_center_h(d["bbox"]), d["track_id"], "object", d["conf"], d["cls"], d["bbox"])
                 for d in dets if d["cls"] == want]
        if not cands:
            return None
        ids = {i for i in (self.target.track_id, self.assoc_id) if i is not None}
        hit = [c for c in cands if c.track_id in ids]
        if hit:
            self.target_src = "track"
            return max(hit, key=lambda c: c.conf)
        p = self._predicted(t)
        if p is not None:
            scored = [(s, c) for c in cands if (s := self._gate(c, p, t - self.last_sel_t)) is not None]
            if scored:
                self.target_src = "assoc"
                return min(scored, key=lambda sc: sc[0])[1]
            return None  # a different box of the same class: do not jump to it while tracking
        k = self._intr()
        exp = self._expected_bearing(t)
        if exp is not None:
            scored = [(abs(geo.wrap_deg(geo.bearing_deg(c.cx, k) - exp)), c) for c in cands]
            scored = [sc for sc in scored if sc[0] <= 20.0]
            if not scored:
                return None
            self.target_src = "bearing"
            return min(scored, key=lambda sc: (sc[0], -sc[1].conf))[1]
        self.target_src = "conf"
        return max(cands, key=lambda c: c.conf)

    def _expected_bearing(self, t: float) -> float | None:
        tg = self.target
        yaw = self._yaw_now(t)
        if tg.yaw_abs_deg is not None and yaw is not None:
            return geo.wrap_deg(tg.yaw_abs_deg - yaw)
        return tg.bearing_deg

    def _object_size(self) -> float:
        if self.target.size_m:
            return float(self.target.size_m)
        return geo.class_size_m(self.target.cls, 0.20)

    def _alt_m(self, now: float) -> float | None:
        s = self._fresh_state(now)
        return None if s is None or s.get("h_cm") is None else float(s["h_cm"]) / 100.0

    def _update_obj_zref(self, c: Cand) -> None:
        """Plan 5.3 standoff when the mission did not send z_ref_m: object base and center heights from the box."""
        if self.target.z_ref_m is not None:
            return
        alt = self._alt_m(self.last_sel_t)
        if alt is None:
            return
        k = self._intr()
        pitch, roll = geo.tello_attitude(self.state)
        z = geo.range_from_size(c.h, self._object_size(), k.fy)
        if not math.isfinite(z):
            return
        y2 = c.bbox[3]
        base = max(0.0, geo.height_above_floor(c.cx, y2, z, alt, k, pitch, roll))
        center = max(base, geo.height_above_floor(c.cx, c.cy, z, alt, k, pitch, roll))
        zr = geo.approach_z_ref(base, center, self.env_cfg.get("approach", {}))
        self._obj_zref = zr if self._obj_zref is None else self._obj_zref + 0.2 * (zr - self._obj_zref)

    # -------------------------------------------------------------------------------------------- settings
    def _settings(self, box: BoxState) -> Settings:
        s, tg, mode = self.cfg, self.target, self.mode
        common = {"max_fwd_stick": float(s["max_fwd_stick"]), "max_back_stick": float(s["max_back_stick"]),
                  "max_yaw_stick": 60.0, "fx": float(s["fx"]), "fy": float(s["fy"]), "cx0": float(s["cx"]), "cy0": float(s["cy"]),
                  "video_latency_s": float(s["video_latency_s"])}
        if mode == "APPROACH" or (tg.kind == "object" and mode not in PERSON_MODES):
            ac = self.env_cfg.get("approach", {})
            z_ref = tg.z_ref_m or self._obj_zref or float(ac.get("z_ref_min_m", 1.0))
            return Settings(kind="approach", z_ref_m=float(z_ref), target_size_m=self._object_size(), side_offset_deg=0.0,
                            z_min_m=float(tg.z_min_m if tg.z_min_m is not None else OBJECT_Z_MIN_M),
                            cy_ref_frac=float(tg.cy_ref_frac if tg.cy_ref_frac is not None else 0.55), **common)
        z_ref = float(tg.z_ref_m or s["follow_distance_m"])
        size = float(tg.size_m or s["head_size_m"])
        z_min = float(tg.z_min_m if tg.z_min_m is not None else s["min_person_dist_m"])
        if mode == "GUIDE":
            # hold altitude: the ud loop only acts when the head leaves the middle band of rows
            lo, hi = GUIDE_CY_BAND
            cy_ref = min(hi, max(lo, box.cy / IMG_H)) if box.valid else 0.55
            return Settings(kind="follow", z_ref_m=z_ref, target_size_m=size, side_offset_deg=0.0, z_min_m=z_min,
                            cy_ref_frac=cy_ref, **common)
        return Settings(kind="follow", z_ref_m=z_ref, target_size_m=size, side_offset_deg=float(s["side_offset_deg"]),
                        z_min_m=z_min, cy_ref_frac=float(tg.cy_ref_frac if tg.cy_ref_frac is not None else 0.55), **common)

    # -------------------------------------------------------------------------------------------- tick
    def _warn(self, w: str, now: float) -> None:
        if w not in self.warnings and self.verbose:
            print(f"[{NAME}] warning: {w}", flush=True)
        self.warnings.setdefault(w, now)

    def step(self, now: float | None = None) -> dict | None:
        """Drain the bus, then run one tick. Returns the rc message published (None if not the rc owner)."""
        now = time.time() if now is None else now
        for m in self.sub.drain():
            self.handle(m, now)
        return self.tick(now)

    def tick(self, now: float) -> dict | None:
        w0 = time.perf_counter()
        rc = self._tick(now)
        self.work_ms.add(now, 1000.0 * (time.perf_counter() - w0))
        return rc

    def _tick(self, now: float) -> dict | None:
        if self.t0 is None:
            self.t0 = now
        if self._t_last_tick is not None:
            self.loop_ms.add(now, 1000.0 * (now - self._t_last_tick))
        dt = self.dt if self._t_last_tick is None else min(0.2, max(0.01, now - self._t_last_tick))
        self._t_last_tick = now
        self.n_ticks += 1
        box = self.filter.output(now)
        st = self._settings(box)
        owned = RC_OWNER.get(self.mode) == "controller" and not self.killed
        rc = None
        tick_ms = None
        law = "idle"
        if owned:
            if self.mode == "GUIDE":
                law = "guide_yaw_hold"
                yaw = self.guide_pid.act(box, st, dt)[0]
                fb = 0.0
                brain_age = 0.0
            else:
                law = "fly_yaw+pid_fwd" if self.active == "fly" else "pid"
                a = self._perf()
                try:
                    yaw, fb = self.ctrl.act(box, st, self.dt)
                    ok = True
                except Exception as e:  # noqa: BLE001 (the watchdog takes over; never crash the loop)
                    yaw = fb = 0.0
                    ok = False
                    self.act_errors += 1
                    self._warn("act_error", now)
                    if self.act_errors <= 3:
                        print(f"[{NAME}] controller.act failed: {e!r}", flush=True)
                tick_ms = 1000.0 * (self._perf() - a)
                if ok and self.active == "fly":
                    yaw = self.shaper(yaw)
                if ok:
                    self._t_act_ok = now
                    brain_age = tick_ms / 1000.0
                else:
                    brain_age = now - self._t_act_ok
                self.tick_ms.add(now, tick_ms)
                self._check_budget(now)
            alt = self._alt_m(now)
            fs = self._fresh_state(now)
            bat = None if fs is None else fs.get("bat_pct")
            go = self.gov.filter(yaw, fb, box, st, dt, alt_m=alt, brain_age_s=brain_age,
                                 battery_pct=None if bat is None else float(bat))
            self.last_go = go
            rc = msg("rc", t=now, lr=go.lr, fb=go.fb, ud=go.ud, yaw=go.yaw, src="controller", mode=self.mode,
                     gov={"safety": go.safety, "clamped": go.clamped, "reasons": list(go.reasons)},
                     brain_tick_ms=_f(tick_ms, 2), controller=self.active if self.mode != "GUIDE" else "guide_pid", law=law)
            self.pub.publish(rc)
            self.last_rc = rc
            self.n_rc += 1
            req = ("battery" if "battery" in go.reasons else "lost_land") if go.land else None
            if req is not None and self._land_req is None:
                self._t_status = -math.inf  # report a new land request now
            self._land_req = req  # the mission latches it; this process never lands
        else:
            self.last_go = None
        self._track_band(box, st, now)
        if now - self._t_status >= 1.0 / STATUS_HZ - 1e-6:
            self._t_status = now
            self.pub.publish(self.status(now, box, st, law))
        if self.sink is not None and owned:
            self._viz(now, box, st, rc)
        return rc

    def _check_budget(self, now: float) -> None:
        """Plan 3.5: brain tick p95 above 45 ms over 5 s with the fly -> warn, and switch to the PID with --auto-fallback."""
        if self.active != "fly":
            self.fly_since = None
            return
        if self.fly_since is None:
            self.fly_since = now
            return
        if now - self.fly_since < BRAIN_WINDOW_S:
            return
        p95 = self.tick_ms.pct(now, 95, BRAIN_WINDOW_S)
        if p95 is None or p95 <= BRAIN_P95_LIMIT_MS:
            self.warnings.pop("brain_over_budget", None)
            return
        self._warn("brain_over_budget", now)
        if self.auto_fallback and not self.fallback:
            print(f"[{NAME}] brain tick p95 {p95:.1f} ms > {BRAIN_P95_LIMIT_MS:.0f} ms for {BRAIN_WINDOW_S:.0f} s: "
                  "switching to PID-HAND (plan 3.5)", flush=True)
            self.fallback = True
            self._warn("fallback_pid", now)
            self._ensure_controller(now)

    def _range_bearing(self, box: BoxState, st: Settings) -> tuple[float | None, float | None]:
        if not box.valid or box.h <= 0:
            return None, None
        return geo.range_from_size(box.h, st.target_size_m, st.fy), geo.bearing_deg(box.cx, self._intr())

    def _track_band(self, box: BoxState, st: Settings, now: float) -> None:
        z, _ = self._range_bearing(box, st)
        if z is not None and abs(z - st.z_ref_m) <= BAND_FRAC * st.z_ref_m:
            if self._in_band_since is None:
                self._in_band_since = now
        else:
            self._in_band_since = None

    def status(self, now: float, box: BoxState | None = None, st: Settings | None = None, law: str = "idle") -> dict:
        box = self.filter.output(now) if box is None else box
        st = self._settings(box) if st is None else st
        z, b = self._range_bearing(box, st)
        k = self._intr()
        pitch, roll = geo.tello_attitude(self._fresh_state(now))
        el = xy = None
        if box.valid:
            b_lvl, el = geo.bearing_elevation(box.cx, box.cy, k, pitch, roll)
            if z is not None:
                xy = [_f(v) for v in geo.drone_level_xy(b_lvl, z)]
        sd_frac = 0.12 if self.target.kind == "person" else (geo.class_size(self.target.cls) or (0.2, 0.25))[1]
        in_band_s = 0.0 if self._in_band_since is None else now - self._in_band_since
        go = self.last_go
        sel = self.last_sel
        p50 = self.tick_ms.pct(now, 50)
        p95 = self.tick_ms.pct(now, 95)
        return msg(
            "ctrl_status", t=now, mode=self.mode, controller=self.active, arm=getattr(self.ctrl, "name", None), law=law,
            rc_owner=RC_OWNER.get(self.mode) == "controller", killed=self.killed,
            target_valid=bool(box.valid), range_m=_f(z), bearing_deg=_f(b, 2),
            in_band=bool(z is not None and abs(z - st.z_ref_m) <= BAND_FRAC * st.z_ref_m),
            in_band_s=_f(in_band_s, 2), in_band_2s=in_band_s >= IN_BAND_REPORT_S,
            range_sd_m=_f(None if z is None else geo.size_range_sd(z, box.h, sd_frac)), elevation_deg=_f(el, 2), xy_m=xy,
            z_ref_m=_f(st.z_ref_m), z_min_m=_f(st.z_min_m), side_offset_deg=_f(st.side_offset_deg, 2),
            target={"kind": self.target.kind, "cls": self.target.cls, "lock_id": self.lock_id, "track_id": self.assoc_id,
                    "src": self.target_src if sel is not None else None, "conf": _f(sel.conf if sel else None)},
            box={"cx": _f(box.cx, 1), "cy": _f(box.cy, 1), "h": _f(box.h, 1)} if box.valid else None,
            box_age_s=_f(box.since_det_s), lost_s=_f(box.lost_s), lost_2s=box.since_det_s > LOST_REPORT_S,
            det_age_s=_f(now - self.last_det_t if math.isfinite(self.last_det_t) else None),
            brain_tick_ms_p50=_f(p50, 2), brain_tick_ms_p95=_f(p95, 2), loop_hz=_f(self._loop_hz(now), 1),
            tick_work_ms_p95=_f(self.work_ms.pct(now, 95), 2),
            gov=None if go is None else {"safety": go.safety, "clamped": go.clamped, "hover": go.hover, "reasons": list(go.reasons)},
            land_request=self._land_req is not None, land_reason=self._land_req,
            fallback=self.fallback, warnings=sorted(self.warnings), alt_m=_f(self._alt_m(now), 2),
            visible_floor_min_range_m=_f(None if self._alt_m(now) is None else geo.floor_visible_min_range(self._alt_m(now), k, pitch)),
        )

    def _loop_hz(self, now: float) -> float | None:
        v = self.loop_ms.values(now)
        return 1000.0 / float(np.mean(v)) if v else None

    # -------------------------------------------------------------------------------------------- viz
    def _viz(self, now: float, box: BoxState, st: Settings, rc: dict | None) -> None:
        from flyfollow.viz.frames import frame_from_controller

        try:
            z, b = self._range_bearing(box, st)
            target = {"valid": bool(box.valid), "bearing_deg": b, "range_m": z, "in_view": bool(box.valid), "z_ref_m": st.z_ref_m}
            img = self._camera_image(now)
            ctrl = self.guide_pid if self.mode == "GUIDE" else self.ctrl
            sticks = (rc["yaw"], rc["fb"]) if rc else None
            fr = frame_from_controller(ctrl, t=now - (self.t0 or now), tick=self.n_ticks, box=box, sticks=sticks, target=target,
                                       image=img, meta={"source": "drone", "mode": self.mode, "controller": self.active})
            self.sink.publish(fr)
        except Exception as e:  # noqa: BLE001 (the viz never stops the loop)
            self._warn("viz_error", now)
            if self.verbose:
                print(f"[{NAME}] viz error {e!r}", flush=True)

    def _camera_image(self, now: float):
        if self.ring is None:
            if now - self._ring_try < 2.0:
                return None
            self._ring_try = now
            try:
                from flyfollow.runtime.bus import FrameRing

                self.ring = FrameRing.attach()
            except Exception:  # noqa: BLE001 (no producer yet)
                return None
        try:
            got = self.ring.latest()
        except Exception:  # noqa: BLE001 (producer restarted: reattach later)
            self.ring = None
            return None
        if got is None:
            return None
        img, _, t_dec = got
        return img if now - t_dec < 1.0 else None

    # -------------------------------------------------------------------------------------------- loop
    def run(self, stop=None, duration_s: float | None = None) -> None:
        """Real-time loop: tick every 1/hz on the monotonic clock, handle messages while waiting (never blocks on I/O)."""
        period = self.dt
        t_next = time.monotonic()
        t_end = None if duration_s is None else t_next + duration_s
        while not (stop is not None and stop.is_set()):
            m0 = time.monotonic()
            if t_end is not None and m0 >= t_end:
                break
            now = time.time()
            for m in self.sub.drain():
                self.handle(m, now)
            try:
                self.tick(now)
            except Exception as e:  # noqa: BLE001 (no rc this tick: tello_io hovers after RC_TIMEOUT_S; keep running)
                self.tick_errors += 1
                self._warn("tick_error", now)
                if self.tick_errors <= 5:
                    import traceback

                    print(f"[{NAME}] tick failed: {e!r}\n{traceback.format_exc()}", flush=True)
            t_next += period
            t_next = max(t_next, time.monotonic())  # overrun: skip ahead instead of bursting
            while True:
                left = t_next - time.monotonic()
                if left <= 0:
                    break
                m = self.sub.recv(min(left, 0.01))
                if m is not None:
                    self.handle(m)

    def close(self) -> None:
        if self.sink is not None:
            self.sink.close()
            self.sink = None
        if self.ring is not None:
            self.ring.close()
            self.ring = None
        self.sub.close()
        self.pub.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="flyfollow control loop: det -> filter -> fly/PID -> governor -> rc")
    ap.add_argument("--controller", choices=("fly", "pid"), default=None, help="initial controller (settings may switch it)")
    ap.add_argument("--params", default=None,
                    help=f"trainer best.json for the fly (default: <brains>/{DEFAULT_PARAMS}; none = FLY-YAW-HAND hand calibration)")
    ap.add_argument("--brain", default=None, help="brain .npz (default: the params file's brain or controllers.yaml)")
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--viz", action="store_true", help="publish live viz frames (VizSink); run flyfollow.viz.live to see them")
    ap.add_argument("--viz-address", default=None)
    ap.add_argument("--viz-spawn", action="store_true", help="with --viz: also start the Rerun viewer process")
    ap.add_argument("--auto-fallback", action="store_true", help="switch to PID if the brain tick p95 > 45 ms for 5 s")
    ap.add_argument("--mode", default="IDLE", help="initial mode before the mission's first mode message (bench tests)")
    ap.add_argument("--no-reacquire-single", action="store_true", help="never take a lone unlocked person after losing the lock")
    ap.add_argument("--deadband", type=float, default=4.0, help="fly yaw deadband in stick units (FlySteer default)")
    ap.add_argument("--hysteresis", type=float, default=3.0, help="fly yaw hysteresis in stick units (FlySteer default)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.params is None:
        from flyfollow.interfaces import brains_dir

        p = brains_dir() / DEFAULT_PARAMS
        a.params = str(p) if p.exists() else None
    from flyfollow.brain.build import ensure_flydrones
    from flyfollow.runtime.node import log, stop_event

    ensure_flydrones()
    stop = stop_event()
    viewer = None
    if a.viz and a.viz_spawn:
        from flyfollow.viz.live import DEFAULT_ADDRESS, spawn_viewer

        viewer = spawn_viewer(a.brain or "data/brains/pursuit_core1.npz", a.viz_address or DEFAULT_ADDRESS)
    r = ControllerRunner(controller=a.controller, params_path=a.params, hz=a.hz, viz=a.viz, viz_address=a.viz_address,
                         auto_fallback=a.auto_fallback, initial_mode=a.mode, brain_path=a.brain,
                         reacquire_single=not a.no_reacquire_single, verbose=a.verbose, deadband=a.deadband,
                         hysteresis=a.hysteresis)
    log(NAME, f"up: {r.active} ({getattr(r.ctrl, 'name', '?')}, params {a.params or 'hand calibration'}, deadband "
        f"{a.deadband:g}, hysteresis {a.hysteresis:g}) at {a.hz:g} Hz, mode {r.mode}, "
        f"auto-fallback {'on' if a.auto_fallback else 'off'}, viz {'on' if a.viz else 'off'}")
    try:
        r.run(stop, a.duration)
    finally:
        s = r.status(time.time())
        log(NAME, f"down: {r.n_ticks} ticks, {r.n_rc} rc, brain p50/p95 {s['brain_tick_ms_p50']}/{s['brain_tick_ms_p95']} ms, "
            f"loop {s['loop_hz']} Hz")
        r.close()
        if viewer is not None:
            viewer.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
