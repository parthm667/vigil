"""Camera and detector model (plan 4.4): pinhole projection to a box, noise, dropouts and latency.

All per-frame randomness (noise, iid dropout, burst schedule, detection draws, false boxes) is
pre-sampled at construction from the camera's own RNG stream and indexed by frame number, so
the detector sees the same noise sequence whatever the controller does.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from flyfollow.interfaces import IMG_H, IMG_W

Geom = tuple[float, float, float]  # (u, v_top, v_bottom) in pixels, unclipped


@dataclass
class CameraParams:
    fx: float = 921.0  # TRUE focal length (hidden from the controller)
    fy: float = 919.0
    cx0: float = IMG_W / 2
    cy0: float = IMG_H / 2
    rate_hz: float = 15.0
    latency_s: float = 0.3  # capture to box delivered
    det_time_s: float = 0.05  # detector share of latency_s; the rest is video latency
    center_sigma_px: float = 2.0
    h_noise_frac: float = 0.05
    dropout: float = 0.0  # iid per frame
    burst_rate_hz: float = 0.0
    burst_len_s: tuple[float, float] = (0.2, 1.5)
    false_frac: float = 0.0
    false_h_px: tuple[float, float] = (15.0, 150.0)
    h_full_det_px: float = 20.0  # detection probability (h / this)^2 below it

    @property
    def half_hfov(self) -> float:
        return math.atan(IMG_W / 2 / self.fx)


def project(fwd: float, left: float, up: float, size: float, pitch: float,
            fx: float, fy: float, cx0: float, cy0: float) -> Geom | None:
    """Vertical target of height `size` centered at (fwd, left, up) in the drone-level frame.

    The camera is fixed to the body, pitched nose down by `pitch` (rad). Returns None when
    either end of the target is behind the camera.
    """
    cp, sp = math.cos(pitch), math.sin(pitch)
    half = 0.5 * size
    ut, ub = up + half, up - half
    d_t = fwd * cp - ut * sp
    d_b = fwd * cp - ub * sp
    if d_t < 0.05 or d_b < 0.05:
        return None
    d_c = fwd * cp - up * sp
    return (cx0 - fx * left / d_c, cy0 - fy * (fwd * sp + ut * cp) / d_t, cy0 - fy * (fwd * sp + ub * cp) / d_b)


def visible_box(g: Geom | None) -> tuple[float, float, float] | None:
    """(cx, cy, h) of the box clipped to the image, or None when the target is out of view."""
    if g is None:
        return None
    u, vt, vb = g
    if u < 0.0 or u > IMG_W:
        return None
    vc = 0.5 * (vt + vb)
    if vc < 0.0 or vc > IMG_H:
        return None
    vt = max(0.0, vt)
    vb = min(float(IMG_H), vb)
    h = vb - vt
    if h < 2.0:
        return None
    return u, 0.5 * (vt + vb), h


Sink = Callable[[float, float, float, float, float, bool], None]  # (t, t_decoded, cx, cy, h, true_target)


class Camera:
    """Captures frames at rate_hz, applies the detector model, delivers boxes after latency_s."""

    def __init__(self, p: CameraParams, rng: np.random.Generator, t0: float, t1: float):
        self.p = p
        period = 1.0 / p.rate_hz
        n = int((t1 - t0) * p.rate_hz) + 3
        phase = float(rng.uniform(0.0, period))
        self._t = (t0 + phase + np.arange(n) * period).tolist()
        z = rng.standard_normal((3, n))
        uu = rng.random((4, n))
        self._nx = (z[0] * p.center_sigma_px).tolist()
        self._ny = (z[1] * p.center_sigma_px).tolist()
        self._nh = (1.0 + z[2] * p.h_noise_frac).tolist()
        self._udet = uu[1].tolist()
        # bursts: Poisson process with a fixed number of draws so the stream does not shift with the rate
        m = 64
        gaps = rng.exponential(1.0, m)
        lens = rng.uniform(p.burst_len_s[0], p.burst_len_s[1], m)
        drop = uu[0] < p.dropout
        if p.burst_rate_hz > 0.0:
            starts = t0 + np.cumsum(gaps) / p.burst_rate_hz
            tt = np.asarray(self._t)
            for s, ln in zip(starts, lens):
                if s > t1:
                    break
                drop |= (tt >= s) & (tt < s + ln)
        self._drop = drop.tolist()
        self._false = (uu[2] < p.false_frac).tolist()
        fb = rng.random((3, n))
        self._fu = (fb[0] * IMG_W).tolist()
        self._fv = (fb[1] * IMG_H).tolist()
        self._fh = (p.false_h_px[0] + fb[2] * (p.false_h_px[1] - p.false_h_px[0])).tolist()
        self._k = 0
        self._q: deque = deque()
        self._lat = p.latency_s
        self._vid = p.latency_s - p.det_time_s
        self.n_frames = 0  # frames captured so far
        self.n_true = 0  # true-target boxes produced

    def step(self, t_prev: float, t_now: float, g_prev: Geom | None, g_now: Geom | None, occluded: bool = False) -> None:
        """Capture every frame with capture time in (t_prev, t_now]; geometry is interpolated between ticks.

        occluded: the target is hidden behind an obstacle this tick, so its frames give no true box
        (false boxes still happen).
        """
        ts = self._t
        k = self._k
        while ts[k] <= t_now:
            tk = ts[k]
            self.n_frames += 1
            if self._drop[k]:
                k += 1
                continue
            if self._false[k]:
                self._q.append((tk + self._lat, tk + self._vid, self._fu[k], self._fv[k], self._fh[k], False))
                k += 1
                continue
            if occluded:
                k += 1
                continue
            g = g_now
            if g_prev is not None and g_now is not None and t_now > t_prev:
                w = (tk - t_prev) / (t_now - t_prev)
                if 0.0 <= w < 1.0:
                    a = 1.0 - w
                    g = (a * g_prev[0] + w * g_now[0], a * g_prev[1] + w * g_now[1], a * g_prev[2] + w * g_now[2])
            box = visible_box(g)
            if box is not None:
                cx, cy, h = box
                hf = self.p.h_full_det_px
                if h >= hf or self._udet[k] < (h / hf) ** 2:
                    hn = h * self._nh[k]
                    if hn < 1.0:
                        hn = 1.0
                    self._q.append((tk + self._lat, tk + self._vid, cx + self._nx[k], cy + self._ny[k], hn, True))
                    self.n_true += 1
            k += 1
        self._k = k

    def deliver(self, t_now: float, sink: Sink) -> None:
        """Hand every box whose delivery time has come to sink(t, t_decoded, cx, cy, h, true_target)."""
        q = self._q
        while q and q[0][0] <= t_now:
            d = q.popleft()
            sink(d[0], d[1], d[2], d[3], d[4], d[5])
