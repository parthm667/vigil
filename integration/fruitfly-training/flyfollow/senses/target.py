"""TargetEncoder (plan 4.1): TargetFeatures -> Poisson rates for the pursuit brain's visual inputs.

Output is a fixed 22-channel rate vector (see CHANNELS): 8 LC10a azimuth bins per side, then
LC9, LC11 and AROUSAL per side. FlyBrain maps channels to neurons; NOBRAIN pools the bins.

Retinotopy is ours: right LC10a covers bearings from -overlap to +theta_max (bin 0 frontal,
bin 7 lateral), left LC10a mirrors it. Per bin:

    rate = r_max * gain_bin * exp(-0.5 ((theta - c) / (sigma * s^a))^2) * s^b * (1 + k_v |dtheta|)

LC9 and LC11 get absolute rates, capped at lc_cap_hz (10 Hz): the G0 audit found that 50 Hz LC9/LC11
drive swamps LC10a and breaks steering on core1.

Arousal (docs/audit_result.md: P1 has no pathway onto LC10a or AOTU019):
- mode "gain" (default): enc_arousal multiplies all LC10a rates (plan 4.1 fallback); AROUSAL
  neurons get the fixed, untrained p1_tonic_hz (0 by default, capped at p1_cap_hz).
- mode "rate": P1 rate = enc_arousal * arousal_base_hz, capped at p1_cap_hz.
With no target (features.valid False) only the tonic P1 channel can be on.
"""

from __future__ import annotations

import math

import numpy as np

from flyfollow.interfaces import N_AZIMUTH_BINS, TargetFeatures

NB = N_AZIMUTH_BINS
CHANNELS = (
    tuple(f"LC10a_L_b{k}" for k in range(NB))
    + tuple(f"LC10a_R_b{k}" for k in range(NB))
    + ("LC9_L", "LC9_R", "LC11_L", "LC11_R", "AROUSAL_L", "AROUSAL_R")
)
N_CHANNELS = len(CHANNELS)
CH_LC9 = 2 * NB  # LC9_L, LC9_R, LC11_L, LC11_R, AROUSAL_L, AROUSAL_R follow the bins
CH_LC11 = CH_LC9 + 2
CH_AROUSAL = CH_LC9 + 4

_DEFAULT_FIXED = {
    "theta_max_deg": 40.0,
    "s_clip": [0.2, 5.0],
    "dtheta_clip": 3.0,
    "rate_cap_hz": 400.0,
    "lc_cap_hz": 10.0,
    "arousal_base_hz": 2.5,
    "p1_tonic_hz": 0.0,
    "p1_cap_hz": 10.0,
}


class TargetEncoder:
    """Stateless per tick. `params` holds the 25 enc_* values; `fixed` the encoder section of the config."""

    def __init__(self, params: dict[str, float], fixed: dict | None = None, arousal_mode: str = "rate"):
        if arousal_mode not in ("rate", "gain"):
            raise ValueError(f"arousal_mode must be 'rate' or 'gain', got {arousal_mode!r}")
        f = {**_DEFAULT_FIXED, **(fixed or {})}
        self.arousal_mode = arousal_mode
        self.theta_max = math.radians(float(f["theta_max_deg"]))
        self.s_lo, self.s_hi = (float(v) for v in f["s_clip"])
        self.dtheta_clip = float(f["dtheta_clip"])
        self.rate_cap = float(f["rate_cap_hz"])
        self.arousal_base = float(f["arousal_base_hz"])
        self.lc_cap = float(f["lc_cap_hz"])
        self.p1_tonic = min(float(f["p1_tonic_hz"]), float(f["p1_cap_hz"]))
        self.p1_cap = float(f["p1_cap_hz"])
        self.set_params(params)

    def set_params(self, p: dict[str, float]) -> None:
        self.p = dict(p)
        self.r_max = float(p["enc_r_max_hz"])
        self.sigma = math.radians(float(p["enc_width_deg"]))
        ov = math.radians(float(p["enc_overlap_deg"]))
        self.a_w = float(p["enc_size_width_exp"])
        self.a_amp = float(p["enc_size_amp_exp"])
        self.k_v = float(p["enc_vel_gain"])
        self.lc9 = float(p["enc_lc9_hz"])
        self.lc11 = float(p["enc_lc11_hz"])
        self.arousal = float(p["enc_arousal"])
        gains = np.array([p[f"enc_bin_gain_{s}{k}"] for s in "LR" for k in range(NB)], dtype=np.float64)
        span = self.theta_max + ov
        c_r = -ov + (np.arange(NB) + 0.5) * span / NB  # right side: frontal (slightly left of 0) to lateral
        self.centers = np.concatenate([-c_r, c_r])  # [L0..L7, R0..R7]
        lc10_gain = self.arousal if self.arousal_mode == "gain" else 1.0
        self.amp_bins = self.r_max * gains * lc10_gain
        self._rest = np.zeros(N_CHANNELS, dtype=np.float32)
        p1 = min(self.arousal * self.arousal_base, self.p1_cap) if self.arousal_mode == "rate" else self.p1_tonic
        self._rest[CH_AROUSAL : CH_AROUSAL + 2] = p1

    def channels(self, f: TargetFeatures) -> np.ndarray:
        """Features -> rates (Hz) per channel, float32 of shape (22,). Returns a fresh array."""
        out = self._rest.copy()
        if not f.valid:
            return out
        s = min(max(f.s, self.s_lo), self.s_hi)
        sig = self.sigma * s**self.a_w
        amp = s**self.a_amp * (1.0 + self.k_v * min(abs(f.dtheta), self.dtheta_clip))
        z = (f.theta - self.centers) / sig
        tun = np.exp(-0.5 * z * z)
        out[: 2 * NB] = np.minimum(self.amp_bins * tun * amp, self.rate_cap)
        act_l, act_r = tun[:NB].max() * amp, tun[NB:].max() * amp  # how strongly the spot falls in each side's field
        cap = self.lc_cap
        out[CH_LC9] = min(self.lc9 * act_l, cap)
        out[CH_LC9 + 1] = min(self.lc9 * act_r, cap)
        out[CH_LC11] = min(self.lc11 * act_l, cap)
        out[CH_LC11 + 1] = min(self.lc11 * act_r, cap)
        return out

    def encode(self, f: TargetFeatures) -> dict[str, np.ndarray]:
        """Channel name -> rate, for inspection and the dict API."""
        return dict(zip(CHANNELS, self.channels(f).tolist()))


def encoder_from_params(params: dict[str, float], cfg: dict, has_arousal: bool) -> TargetEncoder:
    """Config encoder.arousal_mode: "gain", "rate" or "auto" (rate when the brain has AROUSAL neurons)."""
    fixed = {k: v for k, v in cfg["encoder"].items() if k not in ("params", "arousal_mode")}
    mode = cfg["encoder"].get("arousal_mode", "gain")
    if mode == "auto":
        mode = "rate" if has_arousal else "gain"
    if mode == "rate" and not has_arousal:
        mode = "gain"
    return TargetEncoder(params, fixed, mode)
