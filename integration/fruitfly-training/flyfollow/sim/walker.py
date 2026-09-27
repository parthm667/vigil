"""Person walker (plan 4.4): unicycle kinematics over a random segment sequence.

The whole base path is generated at construction from the walker's own RNG stream, so it is the
same for every controller. The only part that reads the drone is the optional single step toward
the drone: its time, distance and speed are pre-sampled, only its direction is taken from where
the drone is when the step starts. The step is added as an offset on top of the base path.
"""

from __future__ import annotations

import math

import numpy as np

from flyfollow.interfaces import DT


def _u(rng: np.random.Generator, r) -> float:
    if isinstance(r, (list, tuple)):
        return float(rng.uniform(r[0], r[1]))
    return float(r)


class Walker:
    """Head position of the person per tick. Head height and head size are fixed per episode."""

    def __init__(self, rng: np.random.Generator, x0: float, y0: float, heading: float, v_cap: float,
                 n_ticks: int, cfg: dict, speed_frac: list | tuple = (0.0, 1.0), dt: float = DT):
        self.dt = dt
        self.v_cap = v_cap
        n = n_ticks + 2
        xs = [0.0] * n
        ys = [0.0] * n
        x, y, phi = x0, y0, heading
        amax = cfg.get("accel_mps2", 1.0) * dt
        p_str = cfg.get("p_straight", 0.5)
        p_turn = cfg.get("p_turn", 0.3)
        p_new = cfg.get("p_new_speed", 0.6)
        do_step = rng.random() < cfg.get("step_prob", 0.5)
        step_after = int(rng.uniform(0.15, 0.8) * n_ticks)
        step_dist = _u(rng, cfg.get("step_dist_m", (0.4, 1.0)))
        self.step_speed = _u(rng, cfg.get("step_speed_mps", (0.4, 0.8)))
        self.step_stop = float(cfg.get("step_stop_dist_m", 0.8))
        self.step_i0 = self.step_i1 = -1
        v = 0.0
        v_tgt = v_cap * _u(rng, speed_frac) if rng.random() < 0.5 else 0.0
        self.segments: list[tuple[str, int, int]] = []  # (type, first tick, end tick), for plots and tests
        i = 0
        while i < n:
            u = rng.random()
            dur_turn = 0.0
            rate = 0.0
            if do_step and self.step_i0 < 0 and i >= step_after:
                kind = "step"
                n_move = max(1, int(round(step_dist / self.step_speed / dt)))
                self.step_i0 = i + 10  # stop for 0.5 s, then step
                self.step_i1 = self.step_i0 + n_move
                dur = (n_move + 20) * dt
                v_tgt = 0.0
            elif u < p_str:
                kind = "straight"
                dur = _u(rng, cfg.get("straight_s", (2.0, 8.0)))
                if rng.random() < p_new or v_tgt <= 0.0:
                    v_tgt = v_cap * _u(rng, speed_frac)
            elif u < p_str + p_turn:
                kind = "turn"
                ang = math.radians(_u(rng, cfg.get("turn_deg", (30.0, 180.0))))
                rate = math.radians(min(90.0, _u(rng, cfg.get("turn_rate_dps", (30.0, 90.0)))))
                if rng.random() < 0.5:
                    rate = -rate
                dur = ang / abs(rate)
                dur_turn = dur
            else:
                kind = "stop"
                dur = _u(rng, cfg.get("stop_s", (1.0, 6.0)))
                v_tgt = 0.0
            m = max(1, int(round(dur / dt)))
            self.segments.append((kind, i, min(n, i + m)))
            for _ in range(m):
                if i >= n:
                    break
                if v < v_tgt:
                    v = min(v_tgt, v + amax)
                elif v > v_tgt:
                    v = max(v_tgt, v - amax)
                if dur_turn > 0.0:
                    phi += rate * dt
                x += v * math.cos(phi) * dt
                y += v * math.sin(phi) * dt
                xs[i] = x
                ys[i] = y
                i += 1
        self.xs, self.ys = xs, ys
        self._ox = self._oy = 0.0
        self._ux = self._uy = 0.0

    def position(self, i: int, drone_x: float, drone_y: float) -> tuple[float, float]:
        """Head (x, y) at tick i. Call once per tick in order: the step offset integrates here."""
        if self.step_i0 <= i < self.step_i1:
            bx, by = self.xs[i] + self._ox, self.ys[i] + self._oy
            dx, dy = drone_x - bx, drone_y - by
            d = math.hypot(dx, dy)
            if i == self.step_i0:
                self._ux, self._uy = (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)
            if d > self.step_stop:
                self._ox += self._ux * self.step_speed * self.dt
                self._oy += self._uy * self.step_speed * self.dt
        return self.xs[i] + self._ox, self.ys[i] + self._oy
