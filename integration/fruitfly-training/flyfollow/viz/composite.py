"""Slide-ready 1920x1080 composite: fly body | fly brain on top; camera, top view, fly's eye, traces below.

Compositor.feed(frame) for every 20 Hz brain tick, Compositor.render(dt) for every video frame (30 fps).
"""

from __future__ import annotations

import math

import numpy as np

from flyfollow.viz import draw
from flyfollow.viz.brain_view import BrainActivity, BrainCamera, BrainGeometry, BrainRenderer
from flyfollow.viz.fly_body import FlyBodyAnimator, RenderStyle
from flyfollow.viz.frames import motor_state
from flyfollow.viz.panels import CameraPanel, FlyEyePanel, TopDownPanel, TracePanel

W, H = 1920, 1080
HEADER = 64
MAIN_H = 720
LOW_H = H - HEADER - MAIN_H  # 296

TITLE = "A fruit fly's brain is steering this drone"
SUBTITLE = ("Real MaleCNS wiring (1,446-neuron pursuit circuit, spiking LIF).  Biology: wiring, cell types, spiking model.  "
            "Engineering: camera-to-neuron encoding, readout to yaw, PID forward speed, safety governor, body animation.")


class Compositor:
    def __init__(self, geo: BrainGeometry, body: FlyBodyAnimator | None = None, sway_deg: float = 12.0, sway_period_s: float = 24.0):
        self.geo = geo
        self.activity = BrainActivity(geo)
        self.brain = BrainRenderer(geo, 960, MAIN_H, BrainCamera())
        self.body = body or FlyBodyAnimator(RenderStyle(width=960, height=MAIN_H))
        self.cam = CameraPanel(480, LOW_H)
        self.eye = FlyEyePanel(480, LOW_H)
        self.top = TopDownPanel(480, LOW_H)
        self.traces = TracePanel(480, LOW_H)
        self.sway_deg, self.sway_period = sway_deg, sway_period_s
        self.frame: dict | None = None
        self.video_t = 0.0
        self._lows: np.ndarray | None = None
        self._lows_dirty = True

    def feed(self, frame: dict, dt: float = 0.05) -> None:
        """One brain tick: spike traces, body targets, trails and traces."""
        if frame.get("counts") is not None:
            self.activity.update(frame["counts"], dt)
        else:
            self.activity.update_from_rates(frame, dt)
        self.body.set_motor(motor_state(frame, self.geo.lc10a_bins))
        self.top.push(frame)
        self.traces.push(frame)
        self.frame = frame
        self._lows_dirty = True

    def render(self, dt: float = 1 / 30) -> np.ndarray:
        self.video_t += dt
        fr = self.frame or {"t": 0.0}
        out = np.empty((H, W, 3), np.uint8)
        out[:] = draw.BG
        body = self.body.step(dt)
        self.brain.cam.azimuth_deg = self.sway_deg * math.sin(2 * math.pi * self.video_t / self.sway_period)
        brain = self.brain.render(self.activity.glow(), t=self.video_t, rates=self.activity.label_rates())
        out[HEADER : HEADER + MAIN_H, 0:960] = body
        out[HEADER : HEADER + MAIN_H, 960:1920] = brain
        if self._lows_dirty or self._lows is None:
            lows = np.empty((LOW_H, W, 3), np.uint8)
            lows[:, 0:480] = self.cam.render(fr)
            lows[:, 480:960] = self.top.render(fr)
            lows[:, 960:1440] = self.eye.render(fr, self.geo.lc10a_bins)
            lows[:, 1440:1920] = self.traces.render()
            self._lows = lows
            self._lows_dirty = False
        out[HEADER + MAIN_H :, :] = self._lows
        # separators
        out[HEADER : HEADER + MAIN_H, 958:962] = draw.BG
        out[HEADER + MAIN_H : HEADER + MAIN_H + 3, :] = draw.BG
        for x in (479, 959, 1439):
            out[HEADER + MAIN_H :, x : x + 2] = draw.BG
        self._header(out, fr)
        return out

    def _header(self, out: np.ndarray, fr: dict) -> None:
        meta = fr.get("meta") or {}
        with draw.TextBatch(out) as tb:
            tb.text((22, 12), TITLE, size=24, bold=True)
            tb.text((22, 42), SUBTITLE, size=13, color=draw.MUTED)
            src = {"sim": "simulation", "synthetic": "scripted stimulus", "drone": "live flight"}.get(meta.get("source", ""), meta.get("source", ""))
            right = f"t = {fr.get('t', 0.0):5.1f} s"
            if meta.get("arm"):
                right += f"   {meta['arm']}"
            if src:
                right += f"   {src}"
            if meta.get("seed") is not None and meta.get("source") == "sim":
                right += f" seed {meta['seed']}"
            tb.text((W - 22, 16), right, size=15, color=draw.TEXT, anchor="ra", kind="mono")
