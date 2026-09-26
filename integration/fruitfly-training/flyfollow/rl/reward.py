"""Reward (plan 4.5). Uses simulator truth; the controller only ever sees the filtered box.

    r_t = -dt * (w_r e_Z + w_x e_x^2 + w_j j_t + w_g o_t + w_l l_t + w_c c_t) + events

Every term and event is summed separately (terms()) so we can see what the optimizer buys.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

TERM_KEYS = ("e_z", "e_x", "jerk", "safety", "lost", "too_close")
EVENT_KEYS = ("ev_loss", "ev_too_close", "ev_collision", "ev_tail", "ev_success")


@dataclass
class RewardConfig:
    w_r: float = 1.0
    w_x: float = 2.0
    w_j: float = 5.0
    w_g: float = 1.0
    w_l: float = 5.0
    w_c: float = 10.0
    ev_loss: float = -10.0
    ev_too_close: float = -20.0
    ev_collision: float = -200.0
    ev_success: float = 20.0
    band_frac: float = 0.15
    e_z_cap: float = 3.0
    far_quad_m: float = 1.0
    lost_after_s: float = 1.0
    collide_person_m: float = 0.5
    collide_object_m: float = 0.2

    @classmethod
    def from_dict(cls, d: dict | None) -> RewardConfig:
        d = d or {}
        return cls(**{f.name: float(d[f.name]) for f in fields(cls) if f.name in d})

    @property
    def max_rate(self) -> float:
        """Maximum non-jerk cost per second, charged for every tick a collision (or landing) cuts off."""
        return self.w_r * self.e_z_cap + self.w_x + self.w_g + self.w_l + self.w_c


def e_z(z: float, z_ref: float, band: float, cap: float = 3.0, far_quad_m: float = 1.0) -> float:
    """Standoff band error: max(0, |d| - b) + max(0, d - b)^2 / (1 m), capped."""
    d = z - z_ref
    ad = d if d >= 0.0 else -d
    e = ad - band if ad > band else 0.0
    if d > band:
        e += (d - band) * (d - band) / far_quad_m
    return e if e < cap else cap


class RewardTracker:
    """Per-episode reward state and per-term sums (each sum is the signed contribution to the return)."""

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg
        self.reset(2.0, 1.0, 0.5)

    def reset(self, z_ref: float, z_min: float, half_fov: float) -> None:
        self.z_ref = z_ref
        self.z_min = z_min
        self.band = self.cfg.band_frac * z_ref
        self.half_fov = half_fov
        self.s_ez = self.s_ex = self.s_j = self.s_g = self.s_l = self.s_c = 0.0
        self.s_evl = self.s_evc = self.s_col = self.s_tail = self.s_succ = 0.0
        self._lost = False
        self._close = False
        self.n_loss = 0
        self.n_close = 0

    def tick(self, dt: float, z: float, bearing_err: float, d_yaw: float, d_fb: float, safety: bool, lost: bool) -> float:
        """One tick. bearing_err in rad relative to the planned bearing; d_yaw, d_fb are sent-stick changes."""
        c = self.cfg
        ez = e_z(z, self.z_ref, self.band, c.e_z_cap, c.far_quad_m)
        ex = abs(bearing_err) / self.half_fov
        if ex > 1.0:
            ex = 1.0
        j = (d_yaw * d_yaw + d_fb * d_fb) * 0.002  # / 100^2 * 20
        close = z < self.z_min
        k = -dt
        a = k * c.w_r * ez
        b = k * c.w_x * ex * ex
        jj = k * c.w_j * j
        g = k * c.w_g if safety else 0.0
        lo = k * c.w_l if lost else 0.0
        cl = k * c.w_c if close else 0.0
        self.s_ez += a
        self.s_ex += b
        self.s_j += jj
        self.s_g += g
        self.s_l += lo
        self.s_c += cl
        r = a + b + jj + g + lo + cl
        if lost and not self._lost:
            self.s_evl += c.ev_loss
            self.n_loss += 1
            r += c.ev_loss
        if close and not self._close:
            self.s_evc += c.ev_too_close
            self.n_close += 1
            r += c.ev_too_close
        self._lost = lost
        self._close = close
        return r

    def collision(self, remaining_ticks: int, dt: float) -> float:
        self.s_col += self.cfg.ev_collision
        return self.cfg.ev_collision + self.tail(remaining_ticks, dt)

    def tail(self, remaining_ticks: int, dt: float) -> float:
        t = -self.cfg.max_rate * dt * remaining_ticks
        self.s_tail += t
        return t

    def success(self) -> float:
        self.s_succ += self.cfg.ev_success
        return self.cfg.ev_success

    def terms(self) -> dict[str, float]:
        return {"e_z": self.s_ez, "e_x": self.s_ex, "jerk": self.s_j, "safety": self.s_g, "lost": self.s_l,
                "too_close": self.s_c, "ev_loss": self.s_evl, "ev_too_close": self.s_evc, "ev_collision": self.s_col,
                "ev_tail": self.s_tail, "ev_success": self.s_succ}

    @property
    def total(self) -> float:
        return sum(self.terms().values())
