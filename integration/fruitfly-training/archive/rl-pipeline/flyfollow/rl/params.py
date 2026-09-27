"""Trainable parameter specs per arm, and the mapping between CMA-ES space [0, 1] and real values.

FLY and NOBRAIN share the same 47 parameters (25 encoder + 22 readout), so the
only difference between them is whether the fixed connectome sits in the middle.
PID has 4 parameters.
"""

from __future__ import annotations

import math

import numpy as np

from ..pilot.pid import PID_HAND


class Spec:
    def __init__(self, name: str, low: float, high: float, init: float, log: bool = False):
        self.name = name
        self.low = low
        self.high = high
        self.init = init
        self.log = log

    def to_unit(self, value: float) -> float:
        value = min(max(value, self.low), self.high)
        if self.log:
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)

    def from_unit(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        if self.log:
            return math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        return self.low + u * (self.high - self.low)


def encoder_specs() -> list[Spec]:
    # starting values from the G0 audit (docs/audit_result.md): 200 Hz max rate, 10 degree tuning
    specs = [
        Spec("r_max", 20.0, 300.0, 200.0, log=True),
        Spec("sigma_deg", 5.0, 40.0, 10.0),
        Spec("az0_deg", -10.0, 10.0, 0.0),
        Spec("size_exp", 0.0, 2.0, 1.0),
        Spec("size_widen", 0.0, 1.0, 0.3),
        Spec("vel_gain", 0.0, 2.0, 0.3),
    ]
    for side in ("L", "R"):
        for k in range(8):
            specs.append(Spec(f"bin_gain_{side}{k}", 0.5, 2.0, 1.0, log=True))
    # audit: 5 mV arousal had no effect and 10 mV drove the DNs with no target, so start at 0 and cap at 10
    specs.append(Spec("arousal_mv", 0.0, 10.0, 0.0))
    specs.append(Spec("lc9_gain", 0.0, 1.0, 0.2))
    specs.append(Spec("lc11_gain", 0.0, 1.0, 0.2))
    return specs


def readout_specs(dn_types: list[str]) -> list[Spec]:
    specs = []
    for dn in dn_types:
        specs.append(Spec(f"w_{dn}", -4.0, 4.0, 0.0))
    for dn in dn_types:
        for side in ("L", "R"):
            specs.append(Spec(f"u_{dn}_{side}", -4.0, 4.0, 0.0))
    specs.append(Spec("b_yaw", -1.0, 1.0, 0.0))
    specs.append(Spec("b_fwd", -1.0, 1.0, 0.0))
    specs.append(Spec("g_yaw", 0.0, 1.0, 1.0))
    specs.append(Spec("g_fwd", 0.0, 1.0, 0.8))
    specs.append(Spec("tau_yaw", 0.02, 0.15, 0.08))
    specs.append(Spec("tau_fwd", 0.02, 0.15, 0.12))
    specs.append(Spec("deadzone", 0.0, 0.2, 0.02))
    return specs


def pid_specs() -> list[Spec]:
    return [
        Spec("kp_yaw", 20.0, 300.0, PID_HAND["kp_yaw"], log=True),
        Spec("kd_yaw", 0.0, 60.0, PID_HAND["kd_yaw"]),
        Spec("kp_fwd", 5.0, 120.0, PID_HAND["kp_fwd"], log=True),
        Spec("deadband_m", 0.0, 0.4, PID_HAND["deadband_m"]),
    ]


def specs_for_arm(arm: str, dn_types: list[str]) -> list[Spec]:
    if arm in ("fly", "fly_shuf", "nobrain"):
        return encoder_specs() + readout_specs(dn_types)
    if arm == "pid":
        return pid_specs()
    raise ValueError(f"unknown arm {arm}")


def decode(specs: list[Spec], x) -> dict:
    params = {}
    for i, spec in enumerate(specs):
        params[spec.name] = spec.from_unit(float(x[i]))
    return params


def encode(specs: list[Spec], params: dict) -> np.ndarray:
    x = np.zeros(len(specs))
    for i, spec in enumerate(specs):
        x[i] = spec.to_unit(params.get(spec.name, spec.init))
    return x


def defaults(specs: list[Spec]) -> dict:
    params = {}
    for spec in specs:
        params[spec.name] = spec.init
    return params
