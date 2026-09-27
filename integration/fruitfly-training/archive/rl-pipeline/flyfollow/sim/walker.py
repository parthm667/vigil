"""Walking person (unicycle kinematics) and static object targets."""

from __future__ import annotations

import math

import numpy as np


class Walker:
    """A person walking a random sequence of straight, turn, stop and speed-change segments."""

    def __init__(self, scenario: dict, person_cfg: dict, dt: float, rng: np.random.Generator):
        self.rng = rng
        self.dt = dt
        self.cfg = person_cfg
        self.x = scenario["target_x0"]
        self.y = scenario["target_y0"]
        self.z = scenario["target_z_m"]
        self.heading = math.atan2(self.y, self.x)  # start walking away from the drone
        self.speed_max = max(0.0, scenario["person_speed_max"])
        self.speed = 0.0
        self.target_speed = 0.0
        self.turn_rate = 0.0
        self.segment_left = 0.0
        self.step_toward_done = rng.random() >= person_cfg["step_toward_prob"]
        self.drone_xy = (0.0, 0.0)
        self._new_segment()

    def _new_segment(self) -> None:
        rng = self.rng
        cfg = self.cfg
        choice = rng.random()
        self.turn_rate = 0.0
        if choice < 0.45:
            self.target_speed = rng.uniform(0.3, 1.0) * self.speed_max
            self.segment_left = rng.uniform(cfg["straight_s"][0], cfg["straight_s"][1])
        elif choice < 0.7:
            angle = math.radians(rng.uniform(cfg["turn_deg"][0], cfg["turn_deg"][1]))
            rate = math.radians(rng.uniform(cfg["turn_rate_dps"][0], cfg["turn_rate_dps"][1]))
            if rng.random() < 0.5:
                rate = -rate
            self.turn_rate = rate
            self.segment_left = angle / abs(rate)
            self.target_speed = rng.uniform(0.2, 0.8) * self.speed_max
        elif choice < 0.9:
            self.target_speed = 0.0
            self.segment_left = rng.uniform(cfg["stop_s"][0], cfg["stop_s"][1])
        else:
            self.target_speed = rng.uniform(0.0, 1.0) * self.speed_max
            self.segment_left = rng.uniform(1.0, 3.0)

    def step(self) -> None:
        dt = self.dt
        self.segment_left -= dt
        if self.segment_left <= 0:
            if not self.step_toward_done and self.rng.random() < 0.15:
                # a short step back toward the drone, which tests the back-off behavior
                self.step_toward_done = True
                dx = self.drone_xy[0] - self.x
                dy = self.drone_xy[1] - self.y
                self.heading = math.atan2(dy, dx)
                self.turn_rate = 0.0
                self.target_speed = 0.3 * self.speed_max
                self.segment_left = 1.0
            else:
                self._new_segment()
        self.speed += (self.target_speed - self.speed) * min(1.0, dt / 0.5)
        self.heading += self.turn_rate * dt
        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt


class StaticObject:
    def __init__(self, scenario: dict):
        self.x = scenario["target_x0"]
        self.y = scenario["target_y0"]
        self.z = scenario["target_z_m"]
        self.speed = 0.0
        self.drone_xy = (0.0, 0.0)

    def step(self) -> None:
        pass
