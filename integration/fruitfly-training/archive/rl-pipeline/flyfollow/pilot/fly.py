"""Fly pursuit controller: box -> LC10a encoder -> fixed connectome -> steering DN readout.

Also the NOBRAIN control arm, which uses the same encoder and readout with the
brain replaced by a fixed pooling of the encoder rates.
Only the encoder and readout parameters are trained. The connectome is never changed.
"""

from __future__ import annotations

import math

import numpy as np

from .pid import PID_HAND, PIDController

N_BINS = 8
BIN_SPACING_DEG = 10.0
TICK_MS = 50.0


class Encoder:
    """Turns the observation into Poisson input rates (Hz) for the fly's visual neurons."""

    def __init__(self, p: dict):
        self.p = p
        self.bin_gain = {}
        for side in ("L", "R"):
            gains = []
            for k in range(N_BINS):
                gains.append(p[f"bin_gain_{side}{k}"])
            self.bin_gain[side] = np.array(gains)
        centers = []
        for k in range(N_BINS):
            centers.append(p["az0_deg"] + 5.0 + BIN_SPACING_DEG * k)
        self.centers = np.array(centers)  # degrees into each hemifield

    def rates(self, obs: dict) -> dict:
        inputs = {}
        if not obs["valid"]:
            for side in ("L", "R"):
                for k in range(N_BINS):
                    inputs[f"LC10a_{side}_b{k}"] = 0.0
                inputs[f"LC9_{side}"] = 0.0
                inputs[f"LC11_{side}"] = 0.0
            return inputs
        p = self.p
        theta_deg = math.degrees(obs["theta"])
        s = max(obs["s"], 0.0)
        size_drive = min(s, 3.0) ** p["size_exp"]
        sigma = p["sigma_deg"] * (1.0 + p["size_widen"] * (min(s, 3.0) - 1.0))
        sigma = max(sigma, 2.0)
        motion = 1.0 + p["vel_gain"] * abs(obs["dtheta"])
        for side in ("L", "R"):
            # right optic lobe sees the right hemifield (positive theta), left sees the left
            if side == "R":
                az = self.centers
            else:
                az = -self.centers
            tuning = np.exp(-((az - theta_deg) ** 2) / (2.0 * sigma ** 2))
            bins = p["r_max"] * self.bin_gain[side] * tuning * size_drive * motion
            for k in range(N_BINS):
                inputs[f"LC10a_{side}_b{k}"] = float(bins[k])
            peak = float(bins.max())
            inputs[f"LC9_{side}"] = p["lc9_gain"] * peak
            inputs[f"LC11_{side}"] = p["lc11_gain"] * peak
        return inputs


class Readout:
    """Low-pass filtered, normalized channel rates -> yaw and forward sticks (tanh readout)."""

    def __init__(self, p: dict, dn_types: list[str], norm: dict):
        self.p = p
        self.dn_types = dn_types
        self.norm = norm
        self.filtered = {}

    def reset(self) -> None:
        self.filtered = {}

    def step(self, rates: dict, dt: float, max_fwd_stick: float, max_back_stick: float) -> tuple[float, float]:
        p = self.p
        alpha_yaw = min(1.0, dt / p["tau_yaw"])
        alpha_fwd = min(1.0, dt / p["tau_fwd"])
        yaw_sum = p["b_yaw"]
        fwd_sum = p["b_fwd"]
        for dn in self.dn_types:
            values = {}
            for side in ("L", "R"):
                name = f"{dn}_{side}"
                raw = rates.get(name, 0.0) / max(self.norm.get(name, 20.0), 1.0)
                old_yaw = self.filtered.get(name + ":y", raw)
                old_fwd = self.filtered.get(name + ":f", raw)
                self.filtered[name + ":y"] = old_yaw + alpha_yaw * (raw - old_yaw)
                self.filtered[name + ":f"] = old_fwd + alpha_fwd * (raw - old_fwd)
                values[side] = (self.filtered[name + ":y"], self.filtered[name + ":f"])
            yaw_sum += p[f"w_{dn}"] * (values["R"][0] - values["L"][0])
            fwd_sum += p[f"u_{dn}_L"] * values["L"][1] + p[f"u_{dn}_R"] * values["R"][1]
        yaw = math.tanh(yaw_sum)
        if abs(yaw) < p["deadzone"]:
            yaw = 0.0
        yaw = 60.0 * p["g_yaw"] * yaw
        fwd = math.tanh(fwd_sum)
        if fwd >= 0:
            fb = max_fwd_stick * p["g_fwd"] * fwd
        else:
            fb = max_back_stick * p["g_fwd"] * fwd
        return yaw, fb


class FlyController:
    def __init__(self, params: dict, brain_base, dn_types: list[str], norm: dict, fly_forward: str = "brain",
                 warmup_s: float = 1.0, max_back_stick: float = 20.0):
        self.params = params
        self.brain_base = brain_base
        self.dn_types = dn_types
        self.outputs = []
        for dn in dn_types:
            self.outputs.append(f"{dn}_L")
            self.outputs.append(f"{dn}_R")
        self.encoder = Encoder(params)
        self.readout = Readout(params, dn_types, norm)
        self.fly_forward = fly_forward
        self.warmup_s = warmup_s
        self.max_back = max_back_stick
        self.range_pid = PIDController(PID_HAND)
        self.brain = None
        self.last_rates = {}

    def reset(self, scenario: dict) -> None:
        self.brain = self.brain_base.fresh(scenario["brain_seed"])
        self.brain.set_bias("AROUSAL_L", self.params["arousal_mv"])
        self.brain.set_bias("AROUSAL_R", self.params["arousal_mv"])
        self.readout.reset()
        self.range_pid.reset(scenario)
        blank = {"valid": False}
        warm_ticks = int(round(self.warmup_s * 1000.0 / TICK_MS))
        for i in range(warm_ticks):
            self.brain.tick(self.encoder.rates(blank), self.outputs, TICK_MS)

    def act(self, obs: dict, dt: float) -> tuple[float, float]:
        rates = self.brain.tick(self.encoder.rates(obs), self.outputs, TICK_MS)
        self.last_rates = rates
        yaw, fb = self.readout.step(rates, dt, obs["max_fwd_stick"], self.max_back)
        if self.fly_forward == "pid":
            _, fb = self.range_pid.act(obs, dt)
        return yaw, fb


class NoBrainController:
    """Same encoder and readout, brain replaced by fixed pooling (5 channels per side)."""

    def __init__(self, params: dict, dn_types: list[str], norm: dict, max_back_stick: float = 20.0, fly_forward: str = "brain"):
        self.params = params
        self.dn_types = dn_types
        self.encoder = Encoder(params)
        self.readout = Readout(params, dn_types, norm)
        self.max_back = max_back_stick
        self.fly_forward = fly_forward
        self.range_pid = PIDController(PID_HAND)

    def reset(self, scenario: dict) -> None:
        self.readout.reset()
        self.range_pid.reset(scenario)

    def pooled(self, inputs: dict) -> dict:
        rates = {}
        for side in ("L", "R"):
            channels = []
            for k in range(0, N_BINS, 2):
                channels.append(0.5 * (inputs[f"LC10a_{side}_b{k}"] + inputs[f"LC10a_{side}_b{k + 1}"]))
            channels.append(0.5 * (inputs[f"LC9_{side}"] + inputs[f"LC11_{side}"]))
            for i, dn in enumerate(self.dn_types):
                rates[f"{dn}_{side}"] = channels[i % len(channels)]
        return rates

    def act(self, obs: dict, dt: float) -> tuple[float, float]:
        rates = self.pooled(self.encoder.rates(obs))
        yaw, fb = self.readout.step(rates, dt, obs["max_fwd_stick"], self.max_back)
        if self.fly_forward == "pid":
            _, fb = self.range_pid.act(obs, dt)
        return yaw, fb
