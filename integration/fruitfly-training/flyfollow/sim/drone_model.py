"""Tello response model (plan 4.4): first-order lag from sent sticks to velocity and yaw rate.

Like FlyDrones' SimDrone (third_party/FlyDrones/src/flydrones/drones/sim.py) but in final
stick units, with a per-axis command dead time (the Tello lag test measured 0.18 s on yaw but
0.47 s on forward), an OU hover drift and pitch during acceleration.
World frame: x, y on the ground plane, z up; heading psi is counter-clockwise from +x.
A positive yaw stick turns clockwise, so it decreases psi.
Pure Python floats on purpose: this runs every tick of millions of episodes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from flyfollow.interfaces import DT

G = 9.81


@dataclass
class DroneParams:
    fwd_gain_mps: float = 0.96  # forward and lateral speed at stick 100 (defaults: lag test medians)
    yaw_gain_dps: float = 55.0
    vz_gain_mps: float = 0.32
    tau_fwd_s: float = 0.45  # also used for lateral
    tau_yaw_s: float = 0.02
    tau_z_s: float = 0.02
    fwd_dead_s: float = 0.47  # command dead time, forward and lateral
    yaw_dead_s: float = 0.18
    vz_dead_s: float = 0.41
    drift_sigma_mps: float = 0.0  # stationary std of the OU drift velocity per horizontal axis
    drift_tau_s: float = 2.0
    pitch_scale: float = 0.0  # pitch = pitch_scale * atan(a_fwd / g); 0 disables pitch coupling
    pitch_max_deg: float = 25.0


class DroneModel:
    """Velocity-mode quadcopter. State: x, y, z (m), psi (rad), world velocity, yaw rate, pitch.

    drift_noise holds 2 standard normals per tick, pre-sampled from the episode's own RNG
    stream, so the drift is identical for every controller (common random numbers).
    The hover drift is modeled as an Ornstein-Uhlenbeck velocity (stationary std drift_sigma,
    time constant drift_tau) rather than a pure random walk, which would grow without bound.
    """

    def __init__(self, p: DroneParams, drift_noise: list[float] | None = None, dt: float = DT):
        self.p = p
        self.dt = dt
        self._noise = drift_noise or []
        self._k_f = 1.0 - math.exp(-dt / p.tau_fwd_s)
        self._k_y = 1.0 - math.exp(-dt / p.tau_yaw_s)
        self._k_z = 1.0 - math.exp(-dt / p.tau_z_s)
        a = math.exp(-dt / p.drift_tau_s)
        self._d_a = a
        self._d_b = p.drift_sigma_mps * math.sqrt(max(0.0, 1.0 - a * a))
        # per axis (lr, fb, ud, yaw): dead time L = (m + r) dt
        self._mr = []
        for L in (p.fwd_dead_s, p.fwd_dead_s, p.vz_dead_s, p.yaw_dead_s):
            m = max(0.0, L) / dt
            self._mr.append((int(m), m - int(m)))
        self._nh = max(m for m, _ in self._mr) + 2
        self._pmax = math.radians(p.pitch_max_deg)
        self.reset(0.0, 0.0, 1.0, 0.0)

    def reset(self, x: float, y: float, z: float, psi: float) -> None:
        self.x, self.y, self.z, self.psi = x, y, z, psi
        self.vx = self.vy = self.vz = 0.0  # commanded-response velocity (world)
        self.dvx = self.dvy = 0.0  # drift velocity (world)
        self.yaw_rate_dps = 0.0  # clockwise positive, like the stick
        self.pitch = 0.0  # rad, nose down positive (accelerating forward)
        self._hist = [(0.0, 0.0, 0.0, 0.0)] * self._nh
        self._i = 0

    def step(self, lr: float, fb: float, ud: float, yaw: float) -> None:
        """Advance one tick with the sticks sent at the start of it."""
        h = self._hist
        h.insert(0, (lr, fb, ud, yaw))
        h.pop()
        # Dead time L = (m + r) dt: over this tick the drone sees u[n-m-1] for r of it and u[n-m] for the rest.
        (m0, r0), (m1, r1), (m2, r2), (m3, r3) = self._mr
        c_lr = (1.0 - r0) * h[m0][0] + r0 * h[m0 + 1][0]
        c_fb = (1.0 - r1) * h[m1][1] + r1 * h[m1 + 1][1]
        c_ud = (1.0 - r2) * h[m2][2] + r2 * h[m2 + 1][2]
        c_yaw = (1.0 - r3) * h[m3][3] + r3 * h[m3 + 1][3]
        p = self.p
        dt = self.dt
        # yaw first, then translate along the new heading
        self.yaw_rate_dps += (p.yaw_gain_dps * c_yaw * 0.01 - self.yaw_rate_dps) * self._k_y
        self.psi -= math.radians(self.yaw_rate_dps) * dt
        c, s = math.cos(self.psi), math.sin(self.psi)
        vf = p.fwd_gain_mps * c_fb * 0.01
        vl = -p.fwd_gain_mps * c_lr * 0.01  # lr > 0 moves right, the level frame's y is left
        tx = vf * c - vl * s
        ty = vf * s + vl * c
        ox, oy = self.vx, self.vy
        k = self._k_f
        self.vx += (tx - ox) * k
        self.vy += (ty - oy) * k
        self.vz += (p.vz_gain_mps * c_ud * 0.01 - self.vz) * self._k_z
        if self._d_b > 0.0:
            i = self._i
            n = self._noise
            if 2 * i + 1 < len(n):
                self.dvx = self._d_a * self.dvx + self._d_b * n[2 * i]
                self.dvy = self._d_a * self.dvy + self._d_b * n[2 * i + 1]
        self._i += 1
        self.x += (self.vx + self.dvx) * dt
        self.y += (self.vy + self.dvy) * dt
        self.z += self.vz * dt
        if self.z < 0.0:
            self.z = 0.0
            self.vz = 0.0
        if p.pitch_scale > 0.0:
            a_fwd = ((self.vx - ox) * c + (self.vy - oy) * s) / dt
            pt = p.pitch_scale * math.atan(a_fwd / G)
            self.pitch = max(-self._pmax, min(self._pmax, pt))

    def v_forward(self) -> float:
        """Body forward speed without drift (m/s)."""
        return self.vx * math.cos(self.psi) + self.vy * math.sin(self.psi)
