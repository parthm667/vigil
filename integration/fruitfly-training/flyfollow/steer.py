"""FlySteer: the fruit fly pursuit brain as a drop-in YAW (steering) controller for another drone stack.

The host stack keeps perception, distance keeping, altitude, safety and the mission. Every control loop it
hands FlySteer the target's bearing (and, if known, its range) and gets back one Tello yaw stick.

    from flyfollow.steer import FlySteer

    fly = FlySteer(params_path=None)           # None: hand calibration (FLY-YAW-HAND); or a trainer best.json
    fly.reset()                                # new behavior / episode: 1 s brain warmup with no target
    ...every loop (any rate from 10 to 30 Hz):
    yaw = fly.yaw(now, bearing_rad=b, range_m=r, t_frame=frame_t)   # target seen on this frame
    yaw = fly.yaw(now)                                              # no new detection (between frames or lost)

Conventions (the Tello's, same as flyfollow.interfaces):
    bearing_rad   horizontal angle of the target from the optical axis, POSITIVE = target RIGHT of center
    return value  int yaw stick in -max_yaw..max_yaw, POSITIVE = turn clockwise (to the right), i.e. the
                  value for rc(lr, fb, ud, yaw). Target right -> positive yaw.
    times         seconds on one clock (time.time(), a sim clock, ...); `t_frame` is when the frame the
                  detection came from reached the host (their Frame.t). Capture time = t_frame - latency_s.
    box           (x1, y1, x2, y2) pixels in an image of `image_size`, only needed when the caller has no bearing.
    range_m       distance to the target (m), gives the normalized size s = z_ref_m / range_m (1 at the standoff).

Inside, per 50 ms brain tick (the rate it was trained at, whatever the caller's loop rate):
    detection -> BoxFilter (latency compensation: predicts the box forward from capture to now)
              -> TargetEncoder (LC10a / LC9 / LC11 rates) -> 1,446-neuron LIF pursuit circuit (MaleCNS)
              -> steering DN rates (DNa02, DNa01, ...) -> readout -> yaw stick
The fly only steers: the controller's forward output is discarded. With no detection for more than
0.5 s the fly gets no target input (its DNs relax and the yaw decays); the host's lost-target logic
should own yaw from then on. Each call runs at most 3 ticks (about 2 ms each on a laptop CPU), so it never
blocks the host loop; after a long pause it resynchronizes instead of catching up.
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np

from flyfollow.interfaces import DT, IMG_H, IMG_W, Settings
from flyfollow.pilot.box_filter import BoxFilter

HAND_ARM = "FLY-YAW-HAND"
DEFAULT_TRAINED_ARM = "FLY-YAW"
# The virtual camera FlySteer feeds the fly: the nominal Tello calibration the controller was trained with.
# Bearings are converted to this frame exactly, so the host's own camera model never leaks into the fly.
FX_NOM = 921.0
FY_NOM = 919.0
TRAINED_MAX_YAW = 60.0  # Settings.max_yaw_stick in training: the readout's output scale (not the output clamp)
MAX_TICKS_PER_CALL = 3
# Output conditioning defaults, chosen on the ReachGlass sim sweep (docs/integration/REACHGLASS.md): deadband +
# hysteresis cut the stick change per loop without adding lag; any low-pass (40 to 250 ms) added lag and cost
# bearing error, so it is off by default. The slew cap never bound in the sim: a safety net only. Deadband 4 /
# hysteresis 3 with the smooth fine-tune (FLY-YAW_smooth2_best.json) was the smoothest setting that still
# tracks better than the host's PID (RMS bearing error); deadband 3 / hysteresis 2 tracks tighter.
SMOOTHING_MS = 0.0
DEADBAND = 4.0
SLEW = 300.0
HYSTERESIS = 3.0
SIZE_HOLD_S = 2.0  # a detection without range / box reuses the last known size for this long
WARMUP_S = 1.0


def load_params(path: str | Path | None) -> tuple[np.ndarray | None, str, str | None]:
    """(normalized x or None, arm, brain) from a trainer best.json ('x' or 'params'); None: hand calibration."""
    if path is None or str(path).strip().lower() in ("", "none", "hand"):
        return None, HAND_ARM, None
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    arm = rec.get("arm") or DEFAULT_TRAINED_ARM
    if rec.get("x") is not None:
        x = np.asarray(rec["x"], np.float64)
    elif rec.get("params"):
        from flyfollow.rl.params import param_space

        x = param_space(arm).encode(rec["params"])
    else:
        raise ValueError(f"{path}: expected 'x' or 'params' in the trainer best.json")
    return x, arm, rec.get("brain")


class FlySteer:
    """The fly brain as a yaw controller. See the module docstring for units and signs.

    params_path   trainer best.json (fields "arm", "x" or "params", "brain"); None = hand calibration.
                  Trained copies: data/brains/trained/FLY-YAW_smooth2_best.json (recommended, smooth stick),
                  FLY-YAW_smooth_best.json, FLY-YAW_best.json (v1, tightest tracking)
    brain_path    pursuit subgraph .npz; None = the one named in best.json, else configs' default (core1)
    fx, cx        the HOST camera's focal length and principal point (px) at `image_size`; used only to turn
                  a pixel box into a bearing when the caller passes box= instead of bearing_rad=
    max_yaw       clamp on the returned stick (the host's safety layer clamps again)
    latency_s     capture -> frame-arrival delay the latency filter predicts across (Tello about 0.25 s)
    z_ref_m       the standoff the host holds; the fly sees size 1 there (set per behavior with reset())
    viz           publish one frame per tick to the fly body + brain viewer (flyfollow.viz.live); never
                  raises: if the viz extra is missing it switches itself off and says why in stats
    """

    def __init__(self, params_path: str | Path | None = None, brain_path: str | Path | None = None, fx: float = FX_NOM,
                 cx: float = IMG_W / 2, max_yaw: int = 60, latency_s: float = 0.25, z_ref_m: float = 2.0,
                 target_size_m: float = 0.23, image_size: tuple[int, int] = (IMG_W, IMG_H), viz: bool = False,
                 viz_address: str | None = None, seed: int = 0, smoothing_ms: float = SMOOTHING_MS,
                 deadband: float = DEADBAND, slew: float = SLEW, hysteresis: float = HYSTERESIS, dn_avg_ticks: int = 1):
        from flyfollow.rl.controllers import make_controller

        x, self.arm, brain_rec = load_params(params_path)
        self.params_path = None if x is None else str(params_path)
        self.brain_path = brain_path or brain_rec
        self.ctrl = make_controller(self.arm, x=x, brain_path=self.brain_path)
        self.fx, self.cx = float(fx), float(cx)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.max_yaw = int(max_yaw)
        # output conditioning (spike noise -> stick jitter), applied once per 50 ms brain tick, in this order:
        # DN rates averaged over dn_avg_ticks ticks -> readout -> low-pass (smoothing_ms) -> slew (stick/s) -> deadband
        # -> hysteresis (the sent stick moves only when the value is more than `hysteresis` away: removes dither
        # without lag for real turns)
        self.smoothing_ms = max(0.0, float(smoothing_ms))
        self.deadband = max(0.0, float(deadband))
        self.slew = max(0.0, float(slew))
        self.hysteresis = max(0.0, float(hysteresis))
        self.dn_avg_ticks = max(1, int(dn_avg_ticks))
        self._rates: deque = deque(maxlen=self.dn_avg_ticks)
        self.latency_s = float(latency_s)
        self.filter = BoxFilter(latency_s=self.latency_s)
        self.settings = Settings(kind="follow", z_ref_m=float(z_ref_m), target_size_m=float(target_size_m),
                                 max_yaw_stick=TRAINED_MAX_YAW, fx=FX_NOM, fy=FY_NOM, cx0=IMG_W / 2, cy0=IMG_H / 2,
                                 video_latency_s=self.latency_s)
        self._tick_ms: deque[float] = deque(maxlen=200)
        self._s: deque[float] = deque(maxlen=600)  # normalized size of fused detections (1 = at z_ref_m)
        self._viz = None
        self._viz_error: str | None = None
        if viz:
            try:
                from flyfollow.viz.live import DEFAULT_ADDRESS, VizSink

                self._viz = VizSink(viz_address or DEFAULT_ADDRESS)
            except Exception as e:  # noqa: BLE001 (viz is optional: never take the controller down)
                self._viz_error = f"viz off: {type(e).__name__}: {e}"
        self.reset(seed)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int = 0, z_ref_m: float | None = None, kind: str | None = None) -> None:
        """Start a new pursuit: fresh brain state, 1 s warmup with no target, empty latency filter."""
        if z_ref_m is not None:
            self.settings.z_ref_m = float(z_ref_m)
        if kind is not None:
            self.settings.kind = kind
        self.ctrl.reset(self.settings, seed)
        self.ctrl.warmup(WARMUP_S)
        self.filter.reset()
        self._s.clear()
        self._h_last: tuple[float, float] | None = None
        self._t_next: float | None = None
        self._t_last: float | None = None
        self._t0: float | None = None
        self._yaw_raw = 0.0
        self._yaw_lp = 0.0
        self._yaw_out = 0.0
        self._rates.clear()
        self._yaw = 0
        self._valid = False
        self._ticks = 0
        self._resyncs = 0
        self._last_bearing: float | None = None
        self._last_range: float | None = None

    def close(self) -> None:
        if self._viz is not None:
            try:
                self._viz.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self._viz = None

    # ------------------------------------------------------------------ control
    def yaw(self, t: float, bearing_rad: float | None = None, *, range_m: float | None = None,
            box: tuple[float, float, float, float] | None = None, t_frame: float | None = None,
            image_size: tuple[int, int] | None = None) -> int:
        """One host loop: fuse the detection (if any), run the due 50 ms brain ticks, return the yaw stick.

        Pass bearing_rad and/or box for a detection made on a NEW frame; pass neither when there is no new
        detection (the filter predicts through short gaps, then the fly sees no target).
        """
        t = float(t)
        if self._t_last is not None and t < self._t_last:  # clock went backwards (new sim run): start over
            self.reset()
        self._t_last = t
        if self._t0 is None:
            self._t0 = t
        if bearing_rad is not None or box is not None:
            self._fuse(t, bearing_rad, range_m, box, t_frame, image_size)
        if self._t_next is None:
            self._t_next = t
        n = 0
        while t >= self._t_next and n < MAX_TICKS_PER_CALL:
            self._tick(t)
            self._t_next += DT
            n += 1
        if t >= self._t_next:  # fell behind (host paused): drop the backlog, never catch up in one call
            self._t_next = t + DT
            self._resyncs += 1
        return self._yaw

    def _fuse(self, t, bearing_rad, range_m, box, t_frame, image_size) -> None:
        w, h = image_size or self.image_size
        if bearing_rad is None:
            sx = w / IMG_W if w else 1.0  # fx / cx are given at the reference image size
            bx = 0.5 * (box[0] + box[2])
            bearing_rad = math.atan((bx - self.cx * sx) / (self.fx * sx))
        bearing_rad = max(-1.2, min(1.2, float(bearing_rad)))
        st = self.settings
        if range_m is not None and range_m > 0.05:
            h_px = FY_NOM * st.target_size_m / float(range_m)  # s = z_ref / range
        elif box is not None and h:
            h_px = max(1.0, (box[3] - box[1]) * IMG_H / h)  # the host's box height in the 720-row frame
        elif self._h_last is not None and t - self._h_last[0] < SIZE_HOLD_S:
            h_px = self._h_last[1]  # no range on this frame: keep the last size (a jump would be gated as an outlier)
        else:
            h_px = st.h_ref_px  # size unknown: the fly sees it at the standoff
        if range_m is not None or box is not None:
            self._h_last = (t, h_px)
        cx = IMG_W / 2 + FX_NOM * math.tan(bearing_rad)
        self.filter.update(t, t if t_frame is None else float(t_frame), cx, IMG_H / 2, h_px)
        self._s.append(h_px / st.h_ref_px)
        self._last_bearing, self._last_range = bearing_rad, range_m

    def _tick(self, t: float) -> None:
        box = self.filter.output(t)
        t0 = time.perf_counter()
        y = self._act(box)
        self._tick_ms.append(1000.0 * (time.perf_counter() - t0))
        self._ticks += 1
        self._valid = bool(box.valid)
        self._yaw_raw = y if math.isfinite(y) else 0.0
        if self.smoothing_ms > 0:  # first-order low-pass, time constant smoothing_ms
            self._yaw_lp += (1.0 - math.exp(-DT * 1000.0 / self.smoothing_ms)) * (self._yaw_raw - self._yaw_lp)
        else:
            self._yaw_lp = self._yaw_raw
        if self.slew > 0:  # at most `slew` stick per second
            step = self.slew * DT
            self._yaw_out += max(-step, min(step, self._yaw_lp - self._yaw_out))
        else:
            self._yaw_out = self._yaw_lp
        y = math.copysign(max(0.0, abs(self._yaw_out) - self.deadband), self._yaw_out)  # like the host's bearing deadband
        y = max(-self.max_yaw, min(self.max_yaw, y))
        if y == 0.0 or self.hysteresis <= 0 or abs(y - self._yaw) > self.hysteresis:  # else keep the same stick
            self._yaw = round(y)
        if self._viz is not None:
            self._publish(t, box)

    def _act(self, box) -> float:
        """Raw yaw from the fly for one tick; with dn_avg_ticks > 1 the readout sees DN rates averaged over the
        last ticks (the fly's own path otherwise: FlyController.act, forward output ignored)."""
        fly = getattr(self.ctrl, "inner", self.ctrl)
        if self.dn_avg_ticks == 1 or not hasattr(fly, "brain"):
            return float(self.ctrl.act(box, self.settings, DT)[0])
        from flyfollow.interfaces import target_features

        fly.last_channels = ch = fly.encoder.channels(target_features(box, self.settings))
        fly.last_rates = rates = fly.brain.tick_channels(ch, DT * 1000.0)
        self._rates.append(rates)
        fly.last_sticks = fly.decoder.update(np.mean(self._rates, axis=0), self.settings, DT)
        return float(fly.last_sticks[0])

    def _publish(self, t: float, box) -> None:
        try:
            from flyfollow.viz.frames import frame_from_controller

            target = {"valid": bool(box.valid), "cx": float(box.cx), "cy": float(box.cy), "h": float(box.h),
                      "bearing_deg": math.degrees(math.atan((box.cx - IMG_W / 2) / FX_NOM)) if box.valid else None,
                      "range_m": self._last_range, "z_ref_m": self.settings.z_ref_m}
            self._viz.publish(frame_from_controller(self.ctrl, t=t - (self._t0 or t), tick=self._ticks, box=box,
                                                    sticks=(self._yaw, 0.0), target=target,
                                                    meta={"arm": self.arm, "source": "drone", "kind": self.settings.kind}))
        except Exception as e:  # noqa: BLE001 (viz must never stop steering)
            self._viz_error = f"viz off: {type(e).__name__}: {e}"
            self._viz = None

    # ------------------------------------------------------------------ dashboard
    @property
    def target_valid(self) -> bool:
        """The fly has a target now: a detection was fused less than 0.5 s ago (BoxFilter hold time)."""
        return self._t_last is not None and self.filter.output(self._t_last).valid

    @property
    def stats(self) -> dict:
        """Plain numbers for a dashboard / run log. brain tick times are wall ms per 50 ms tick."""
        ms = np.asarray(self._tick_ms, float)
        sz = np.asarray(self._s, float)
        try:
            dn = {k: round(float(v), 1) for k, v in self.ctrl.last_dn_rates.items()}
        except Exception:  # noqa: BLE001
            dn = {}
        return {
            "arm": self.arm,
            "params": self.params_path or "hand",
            "yaw": self._yaw,
            "yaw_raw": round(self._yaw_raw, 2),  # before smoothing / slew / deadband
            "smoothing": {"ms": self.smoothing_ms, "deadband": self.deadband, "slew": self.slew, "hysteresis": self.hysteresis,
                          "dn_avg_ticks": self.dn_avg_ticks},
            "target_valid": self._valid,
            "ticks": self._ticks,
            "tick_ms_last": round(float(ms[-1]), 3) if ms.size else None,
            "tick_ms_mean": round(float(ms.mean()), 3) if ms.size else None,
            "tick_ms_p95": round(float(np.percentile(ms, 95)), 3) if ms.size else None,
            "resyncs": self._resyncs,
            # normalized target size s = z_ref_m / range seen by the encoder (trained around 1; clipped to 0.2..5)
            "s_last": round(float(sz[-1]), 3) if sz.size else None,
            "s_p05_p50_p95": [round(float(v), 3) for v in np.percentile(sz, [5, 50, 95])] if sz.size else None,
            "dn_hz": dn,
            "fused": self.filter.n_fused,
            "rejected": self.filter.n_rejected,
            "viz": "on" if self._viz is not None else (self._viz_error or "off"),
        }
