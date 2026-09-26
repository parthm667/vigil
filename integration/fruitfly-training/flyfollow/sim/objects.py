"""Static object targets for APPROACH episodes (plan 4.4 and 5.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectTarget:
    x: float
    y: float
    base_h_m: float  # height of the surface it stands on (0 = floor)
    size_m: float  # true vertical size
    prior_size_m: float  # class size prior the controller assumes
    z_ref_m: float  # approach standoff from the altitude rule

    @property
    def center_h_m(self) -> float:
        return self.base_h_m + 0.5 * self.size_m

    @property
    def floor(self) -> bool:
        return self.base_h_m <= 0.0


def approach_z_ref(base_h_m: float, center_h_m: float, cfg: dict) -> float:
    """Plan 5.3: Z_ref = max(1.0, 2.5 x (altitude - object height)), capped at 2.0 m.

    Altitude is where the governor settles: the ud loop keeps the object near the image center,
    so about the object's center height, but never below the 0.8 m floor. Object height is its
    base (the bottom leaves the lower half of the FOV first), which is the conservative choice.
    """
    alt = max(cfg.get("alt_floor_m", 0.8), center_h_m)
    z = cfg.get("z_ref_factor", 2.5) * (alt - base_h_m)
    return min(cfg.get("z_ref_max_m", 2.0), max(cfg.get("z_ref_min_m", 1.0), z))


def sample_object(rng: np.random.Generator, cfg: dict, approach_cfg: dict, bearing_max_deg: float,
                  drone_x: float = 0.0, drone_y: float = 0.0, drone_psi: float = 0.0) -> ObjectTarget:
    """Draws a fixed number of uniforms so the stream never shifts between profiles."""
    u = rng.random(6).tolist()
    lo, hi = cfg.get("size_m", (0.08, 0.30))
    size = lo + u[0] * (hi - lo)
    lo, hi = cfg.get("prior_ratio", (0.75, 1.25))
    ratio = lo + u[1] * (hi - lo)
    lo, hi = cfg.get("base_h_m", (0.0, 1.0))
    base = 0.0 if u[2] < cfg.get("floor_prob", 0.35) else lo + u[3] * (hi - lo)
    z_ref = approach_z_ref(base, base + 0.5 * size, approach_cfg)
    r0 = z_ref + cfg.get("start_margin_m", 0.5)
    rng_m = r0 + u[4] * (cfg.get("start_range_max_m", 6.0) - r0)
    bearing = math.radians((2.0 * u[5] - 1.0) * bearing_max_deg)  # + = right of the drone heading
    ang = drone_psi - bearing
    return ObjectTarget(drone_x + rng_m * math.cos(ang), drone_y + rng_m * math.sin(ang), base, size, size / ratio, z_ref)
