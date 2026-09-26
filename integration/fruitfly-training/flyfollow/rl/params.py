"""Parameter spaces for every arm (plan 4.6): named, bounded, normalized to [0, 1] for CMA-ES.

FLY arms and NOBRAIN: 47 = 25 encoder + 22 readout. PID arms: 4. Yaw-only arms (plan 4.3 fallback:
the fly steers, PID-HAND's range loop sets forward): the 47 minus the 13 forward-only readout
parameters = 34, same bounds and order. Bounds and defaults live in configs/controllers.yaml;
x in [0, 1]^n maps to a value linearly or on a log scale.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from flyfollow.interfaces import ARMS, DN_TYPES, N_AZIMUTH_BINS, configs_dir

N_POOLS = 5
POOL_TYPES = tuple(f"P{i}" for i in range(N_POOLS))  # NOBRAIN stand-ins for the 5 DN types

# Yaw-only arm -> the full arm whose controller it wraps and whose calibration init it projects.
YAW_BASE = {"FLY-YAW-HAND": "FLY-HAND", "FLY-YAW": "FLY-CMA", "FLY-SHUF-YAW": "FLY-SHUF", "NOBRAIN-YAW": "NOBRAIN"}
FWD_ONLY = ("dec_b_fwd", "dec_g_fwd", "dec_tau_fwd_ms")  # plus every dec_u_fwd_*


def base_arm(arm: str) -> str:
    """FLY-YAW -> FLY-CMA, NOBRAIN-YAW -> NOBRAIN, ...; full arms map to themselves."""
    return YAW_BASE.get(arm, arm)


def is_forward_only(name: str) -> bool:
    return name.startswith("dec_u_fwd_") or name in FWD_ONLY


@lru_cache(maxsize=1)
def _base_config() -> dict:
    return yaml.safe_load((configs_dir() / "controllers.yaml").read_text(encoding="utf-8"))


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def controllers_config(overrides: dict | None = None) -> dict:
    """configs/controllers.yaml deep-merged with `overrides` (a fresh copy every call)."""
    return _merge(_base_config(), overrides or {})


@dataclass(frozen=True)
class ParamSpace:
    names: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray
    log: np.ndarray  # bool per parameter
    default: np.ndarray  # hand-set values (readout weights are overwritten by calibration)

    @property
    def dim(self) -> int:
        return len(self.names)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)

    def to_values(self, x: np.ndarray) -> np.ndarray:
        x = self.clip(x)
        if x.shape != (self.dim,):
            raise ValueError(f"expected x of shape ({self.dim},), got {x.shape}")
        lin = self.lo + x * (self.hi - self.lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = self.lo * (self.hi / self.lo) ** x
        return np.where(self.log, lg, lin)

    def from_values(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=np.float64), self.lo, self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(v / self.lo) / np.log(self.hi / self.lo)
            lin = (v - self.lo) / (self.hi - self.lo)
        return self.clip(np.where(self.log, lg, lin))

    def decode(self, x: np.ndarray) -> dict[str, float]:
        """x in [0, 1]^n (clipped) -> {name: value}."""
        return dict(zip(self.names, self.to_values(x).tolist()))

    def encode(self, d: dict[str, float]) -> np.ndarray:
        """{name: value} -> x in [0, 1]^n. Values are clipped to the bounds; missing names use the default."""
        v = np.array([float(d.get(n, dv)) for n, dv in zip(self.names, self.default)])
        return self.from_values(v)

    def default_x(self) -> np.ndarray:
        return self.from_values(self.default)

    def subset(self, names: list[str] | tuple[str, ...]) -> ParamSpace:
        ix = [self.names.index(n) for n in names]
        return ParamSpace(tuple(names), self.lo[ix], self.hi[ix], self.log[ix], self.default[ix])

    def table(self) -> list[tuple[str, float, float, str, float]]:
        return [(n, float(a), float(b), "log" if g else "lin", float(d)) for n, a, b, g, d in zip(self.names, self.lo, self.hi, self.log, self.default)]


def encoder_names() -> list[str]:
    names = ["enc_r_max_hz", "enc_width_deg", "enc_overlap_deg", "enc_size_width_exp", "enc_size_amp_exp", "enc_vel_gain"]
    names += [f"enc_bin_gain_{s}{k}" for s in "LR" for k in range(N_AZIMUTH_BINS)]
    names += ["enc_arousal", "enc_lc9_hz", "enc_lc11_hz"]
    return names


def readout_names(types: tuple[str, ...]) -> list[str]:
    names = [f"dec_w_yaw_{t}" for t in types]
    names += [f"dec_u_fwd_{t}_{s}" for t in types for s in "LR"]
    names += ["dec_b_yaw", "dec_b_fwd", "dec_g_yaw", "dec_g_fwd", "dec_tau_yaw_ms", "dec_tau_fwd_ms", "dec_deadzone"]
    return names


def _spec_key(name: str) -> str:
    """Parameter name -> its bounds entry in the yaml (per-bin gains and per-type weights share one)."""
    for prefix in ("enc_bin_gain", "dec_w_yaw", "dec_u_fwd"):
        if name.startswith(prefix + "_"):
            return prefix
    return name


def _space(names: list[str], specs: dict) -> ParamSpace:
    rows = [specs[_spec_key(n)] for n in names]
    lo = np.array([float(r[0]) for r in rows])
    hi = np.array([float(r[1]) for r in rows])
    log = np.array([str(r[2]) == "log" for r in rows])
    default = np.array([float(r[3]) for r in rows])
    if np.any(log & (lo <= 0)):
        raise ValueError("log-scale parameters need lo > 0")
    return ParamSpace(tuple(names), lo, hi, log, default)


def readout_types(arm: str) -> tuple[str, ...]:
    return POOL_TYPES if base_arm(arm) == "NOBRAIN" else DN_TYPES


def param_space(arm: str, cfg: dict | None = None) -> ParamSpace:
    """FLY-HAND / FLY-CMA / FLY-SHUF / NOBRAIN: 47; PID-HAND / PID-CMA: 4;
    FLY-YAW-HAND / FLY-YAW / FLY-SHUF-YAW / NOBRAIN-YAW: 34 (no forward-only readout parameters)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
    c = controllers_config(cfg)
    if arm.startswith("PID"):
        return _space(list(c["pid"]["params"]), c["pid"]["params"])
    specs = {**c["encoder"]["params"], **c["decoder"]["params"]}
    full = _space(encoder_names() + readout_names(readout_types(arm)), specs)
    if arm in YAW_BASE:
        return full.subset([n for n in full.names if not is_forward_only(n)])
    return full
