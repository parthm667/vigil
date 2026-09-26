"""PursuitDecoder (plan 4.1): steering DN rates -> raw (pre-governor) yaw and forward sticks.

Input is a 10-vector of rates ordered like interfaces.OUTPUT_GROUPS (type-major, side-minor:
DNa02_L, DNa02_R, DNa01_L, ...). NOBRAIN feeds its 5 pooled features per side in the same slots.

    n     = lowpass_tau(rate) / norm                       (tau <= 150 ms, one per axis)
    yaw   = max_yaw * G_yaw * tanh(sum_t w_t (n_tR - n_tL) + b_yaw)
    fwd   = S * G_fwd * tanh(sum_t (u_tL n_tL + u_tR n_tR) + b_fwd),  S = max_fwd or max_back by sign
    out   = sign(out) * max(0, |out| - deadzone)

Bypass rule: update() sees only the rate vector and the governor's stick limits, never the box.
Lesion mode clamps chosen channels to fixed rates before filtering (plan 4.7). Running statistics
feed the bias audit: mean |DN-driven term| inside each tanh versus |b|.
"""

from __future__ import annotations

import math

import numpy as np

from flyfollow.interfaces import DN_TYPES, Settings


def channel_names(types: tuple[str, ...] = DN_TYPES) -> tuple[str, ...]:
    return tuple(f"{t}_{s}" for t in types for s in "LR")


class PursuitDecoder:
    def __init__(
        self,
        params: dict[str, float],
        norm: np.ndarray,
        types: tuple[str, ...] = DN_TYPES,
        lesion: dict[str, float] | None = None,
    ):
        self.types = tuple(types)
        self.names = channel_names(self.types)
        self.norm = np.asarray(norm, dtype=np.float64)
        if self.norm.shape != (len(self.names),) or np.any(self.norm <= 0):
            raise ValueError(f"norm must be {len(self.names)} positive rates, got {self.norm}")
        self._lesion_mask = np.zeros(len(self.names), dtype=bool)
        self._lesion_val = np.zeros(len(self.names))
        if lesion:
            unknown = set(lesion) - set(self.names)
            if unknown:
                raise ValueError(f"lesion names {sorted(unknown)} not in {self.names}")
            for i, n in enumerate(self.names):
                if n in lesion:
                    self._lesion_mask[i] = True
                    self._lesion_val[i] = float(lesion[n])
        self.set_params(params)
        self.reset()

    def set_params(self, p: dict[str, float]) -> None:
        self.p = dict(p)
        w = np.array([p[f"dec_w_yaw_{t}"] for t in self.types])
        u = np.array([p[f"dec_u_fwd_{t}_{s}"] for t in self.types for s in "LR"])
        wy = np.empty(2 * len(self.types))
        wy[0::2], wy[1::2] = -w, w  # sum_t w_t (n_R - n_L)
        self.wy = wy / self.norm  # fold the normalization into the weights
        self.wf = u / self.norm
        self.b_yaw = float(p["dec_b_yaw"])
        self.b_fwd = float(p["dec_b_fwd"])
        self.g_yaw = float(p["dec_g_yaw"])
        self.g_fwd = float(p["dec_g_fwd"])
        self.tau_yaw = min(float(p["dec_tau_yaw_ms"]), 150.0) / 1000.0
        self.tau_fwd = min(float(p["dec_tau_fwd_ms"]), 150.0) / 1000.0
        self.deadzone = float(p["dec_deadzone"])
        self._dt_cached = -1.0

    def reset(self) -> None:
        self.f_yaw: np.ndarray | None = None
        self.f_fwd: np.ndarray | None = None
        self.reset_stats()

    def reset_stats(self) -> None:
        self._n = 0
        self._sum_abs_dy = 0.0
        self._sum_abs_df = 0.0
        self._sum_rate = np.zeros(len(self.names))

    def _alphas(self, dt: float) -> tuple[float, float]:
        if dt != self._dt_cached:
            self._a_yaw = 1.0 - math.exp(-dt / self.tau_yaw)
            self._a_fwd = 1.0 - math.exp(-dt / self.tau_fwd)
            self._dt_cached = dt
        return self._a_yaw, self._a_fwd

    def filter(self, rates: np.ndarray, dt: float) -> None:
        """Advance the low-pass filters only (warmup). `rates` in Hz, OUTPUT_GROUPS order."""
        r = np.asarray(rates, dtype=np.float64)
        if self._lesion_mask.any():
            r = np.where(self._lesion_mask, self._lesion_val, r)
        if self.f_yaw is None:
            self.f_yaw, self.f_fwd = r.copy(), r.copy()
            return
        a_y, a_f = self._alphas(dt)
        self.f_yaw += a_y * (r - self.f_yaw)
        self.f_fwd += a_f * (r - self.f_fwd)

    def update(self, rates: np.ndarray, settings: Settings, dt: float) -> tuple[float, float]:
        """Rates (Hz) -> (yaw_stick, fb_stick). Records bias-audit statistics on the raw rates."""
        r = np.asarray(rates, dtype=np.float64)
        self._sum_rate += r
        self.filter(r, dt)
        dy = float(self.wy @ self.f_yaw)
        df = float(self.wf @ self.f_fwd)
        self._n += 1
        self._sum_abs_dy += abs(dy)
        self._sum_abs_df += abs(df)
        yaw = settings.max_yaw_stick * self.g_yaw * math.tanh(dy + self.b_yaw)
        tf = math.tanh(df + self.b_fwd)
        fwd = (settings.max_fwd_stick if tf > 0 else settings.max_back_stick) * self.g_fwd * tf
        dz = self.deadzone
        yaw = math.copysign(max(0.0, abs(yaw) - dz), yaw)
        fwd = math.copysign(max(0.0, abs(fwd) - dz), fwd)
        return yaw, fwd

    # ------------------------------------------------------------ plan 4.7 checks
    def channel_means(self) -> dict[str, float]:
        """Mean raw rate per channel since reset (feed to lesion= for the lesion rerun)."""
        m = self._sum_rate / max(self._n, 1)
        return dict(zip(self.names, m.tolist()))

    def bias_audit(self) -> dict[str, float]:
        """Typical |channel-driven term| inside each tanh versus the bias magnitude."""
        n = max(self._n, 1)
        dy, df = self._sum_abs_dy / n, self._sum_abs_df / n
        return {
            "n_ticks": float(self._n),
            "yaw_drive_abs": dy,
            "b_yaw_abs": abs(self.b_yaw),
            "yaw_bias_ratio": abs(self.b_yaw) / max(dy, 1e-9),
            "fwd_drive_abs": df,
            "b_fwd_abs": abs(self.b_fwd),
            "fwd_bias_ratio": abs(self.b_fwd) / max(df, 1e-9),
        }
