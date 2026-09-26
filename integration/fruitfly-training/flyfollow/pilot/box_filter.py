"""Shared latency filter (plan 4.1): constant-velocity Kalman filter on the target box (cx, cy, h).

Used by every arm, in sim and at runtime, so it knows nothing about the simulator. Feed it every
`det` box of the locked track with update(t, t_decoded, cx, cy, h), where t is when the box
arrived and t_decoded when its frame was decoded. The frame was captured about latency_s
(the R0 video latency, Settings.video_latency_s) before t_decoded. output(t) predicts the box
to time t, i.e. forward by video latency plus detector time plus the age of the last box.
Three independent 1D filters (position, velocity), so the math stays scalar and cheap.
"""

from __future__ import annotations

import math

from flyfollow.interfaces import BoxState


class BoxFilter:
    def __init__(self, latency_s: float = 0.3, hold_s: float = 0.5, q_c: float = 1e5, q_h_rel: float = 0.1,
                 r_c_px: float = 3.0, r_h_frac: float = 0.07, gate_sigma: float = 5.0, gate_reinit: int = 3):
        self.latency_s = latency_s
        self.hold_s = hold_s
        self.q_c = q_c
        self.q_h_rel = q_h_rel
        self.r_c = r_c_px * r_c_px
        self.r_h_frac = r_h_frac
        self.gate2 = gate_sigma * gate_sigma
        self.gate_reinit = gate_reinit
        self.reset()

    @classmethod
    def from_config(cls, cfg: dict | None, latency_s: float) -> BoxFilter:
        c = dict(cfg or {})
        return cls(latency_s=latency_s, **{k: c[k] for k in ("hold_s", "q_c", "q_h_rel", "r_c_px", "r_h_frac",
                                                              "gate_sigma", "gate_reinit") if k in c})

    def reset(self) -> None:
        self._init = False
        self._t0: float | None = None  # first output time, for lost_s before any detection
        self._t_fused = -1e9
        self._ts = 0.0  # capture time the state refers to
        self._n_rej = 0
        self.n_fused = 0
        self.n_rejected = 0
        # per axis: position, velocity, covariance (a = var p, b = cov pv, c = var v)
        self._x = [0.0] * 6
        self._P = [0.0] * 9

    def _start(self, tc: float, cx: float, cy: float, h: float) -> None:
        self._x = [cx, 0.0, cy, 0.0, h, 0.0]
        vc = 400.0 ** 2
        rh = (self.r_h_frac * h) ** 2
        self._P = [self.r_c, 0.0, vc, self.r_c, 0.0, vc, rh, 0.0, (0.5 * h) ** 2]
        self._ts = tc
        self._init = True
        self._n_rej = 0

    def update(self, t: float, t_decoded: float, cx: float, cy: float, h: float) -> bool:
        """Fuse one box. Returns False if it was gated out as an outlier (e.g. a false box)."""
        tc = t_decoded - self.latency_s
        if not self._init or t - self._t_fused >= self.hold_s:
            self._start(tc, cx, cy, h)
            self._t_fused = t
            self.n_fused += 1
            return True
        dt = tc - self._ts
        if dt < 0.0:
            dt = 0.0
        x, P = self._x, self._P
        dt2 = dt * dt
        pred = []
        z = (cx, cy, h)
        big = False
        for j in range(3):
            q = self.q_c if j < 2 else self.q_h_rel * x[4] * x[4]
            p, v = x[2 * j], x[2 * j + 1]
            a, b, c = P[3 * j], P[3 * j + 1], P[3 * j + 2]
            p += v * dt
            a += 2.0 * b * dt + c * dt2 + q * dt2 * dt / 3.0
            b += c * dt + q * dt2 / 2.0
            c += q * dt
            r = self.r_c if j < 2 else (self.r_h_frac * p) ** 2
            s = a + r
            y = z[j] - p
            if y * y > self.gate2 * s:
                big = True
            pred.append((p, v, a, b, c, s, y))
        if big:
            self._n_rej += 1
            self.n_rejected += 1
            if self._n_rej >= self.gate_reinit:
                self._start(tc, cx, cy, h)
                self._t_fused = t
                self.n_fused += 1
                return True
            return False
        for j in range(3):
            p, v, a, b, c, s, y = pred[j]
            k1, k2 = a / s, b / s
            x[2 * j] = p + k1 * y
            x[2 * j + 1] = v + k2 * y
            P[3 * j] = (1.0 - k1) * a
            P[3 * j + 1] = (1.0 - k1) * b
            P[3 * j + 2] = c - k2 * b
        self._ts = tc
        self._t_fused = t
        self._n_rej = 0
        self.n_fused += 1
        return True

    def output(self, t: float) -> BoxState:
        """Box predicted to time t. valid while the last fused box arrived less than hold_s ago."""
        if self._t0 is None:
            self._t0 = t
        if not self._init:
            return BoxState(False, since_det_s=t - self._t0, lost_s=t - self._t0)
        since = t - self._t_fused
        if since >= self.hold_s:
            return BoxState(False, since_det_s=since, lost_s=since - self.hold_s)
        x = self._x
        dt = t - self._ts
        h = x[4] + x[5] * dt
        # CV extrapolation of h can cross zero on long horizons; keep the box physical
        h = max(h, 0.25 * x[4], 1.0)
        return BoxState(True, x[0] + x[1] * dt, x[2] + x[3] * dt, h, x[1], x[3], x[5], since, 0.0)

    @property
    def last_capture_t(self) -> float:
        return self._ts if self._init else -math.inf
