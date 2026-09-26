"""Hand calibration: the starting point for CMA-ES (plan section 4.6).

We show the controller a fixed set of target stimuli (9 bearings x 3 sizes),
record its channel rates (DN rates for the fly, pooled encoder rates for
NOBRAIN), and ridge-regress the desired yaw and forward commands on them.
The same procedure is used for FLY, FLY-SHUF and NOBRAIN, so they start from
equally fair initial readouts.
"""

from __future__ import annotations

import math

import numpy as np

from ..rl.params import defaults, encoder_specs, readout_specs
from .fly import TICK_MS, Encoder, NoBrainController

BEARINGS_DEG = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
SIZES = [0.5, 1.0, 2.0]
WIDTH = 960.0
FX = 921.0


def _stimulus(theta_deg: float, s: float) -> dict:
    return {"valid": True, "theta": math.radians(theta_deg), "dtheta": 0.0, "s": s}


def measure(arm: str, params: dict, brain_base, dn_types: list[str], seed: int = 7) -> list[dict]:
    """Channel rates for every stimulus, in stimulus order (bearing-major)."""
    encoder = Encoder(params)
    outputs = []
    for dn in dn_types:
        outputs.append(f"{dn}_L")
        outputs.append(f"{dn}_R")
    rows = []
    for theta in BEARINGS_DEG:
        for s in SIZES:
            obs = _stimulus(theta, s)
            if arm == "nobrain":
                pooler = NoBrainController(params, dn_types, {})
                rows.append(pooler.pooled(encoder.rates(obs)))
                continue
            brain = brain_base.fresh(seed)
            brain.set_bias("AROUSAL_L", params["arousal_mv"])
            brain.set_bias("AROUSAL_R", params["arousal_mv"])
            for i in range(10):
                brain.tick(encoder.rates({"valid": False}), outputs, TICK_MS)
            total = {}
            for name in outputs:
                total[name] = 0.0
            n_measure = 10
            for i in range(15):
                rates = brain.tick(encoder.rates(obs), outputs, TICK_MS)
                if i >= 15 - n_measure:
                    for name in outputs:
                        total[name] += rates[name] / n_measure
            rows.append(total)
    return rows


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Ridge regression with an unpenalized intercept."""
    x_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - x_mean
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y - y_mean))
    b = y_mean - x_mean @ w
    return w, float(b)


def hand_init(arm: str, brain_base, dn_types: list[str], lam: float = 0.5) -> tuple[dict, dict, dict]:
    """Returns (initial params, channel normalization rates, calibration report)."""
    params = defaults(encoder_specs() + readout_specs(dn_types))
    rows = measure(arm, params, brain_base, dn_types)

    norm = {}
    for name in rows[0]:
        mean_rate = float(np.mean([row[name] for row in rows]))
        norm[name] = max(mean_rate, 2.0)

    yaw_features = []
    fwd_features = []
    yaw_targets = []
    fwd_targets = []
    i = 0
    for theta in BEARINGS_DEG:
        for s in SIZES:
            row = rows[i]
            i += 1
            yaw_row = []
            fwd_row = []
            for dn in dn_types:
                left = row[f"{dn}_L"] / norm[f"{dn}_L"]
                right = row[f"{dn}_R"] / norm[f"{dn}_R"]
                yaw_row.append(right - left)
                fwd_row.append(left)
                fwd_row.append(right)
            yaw_features.append(yaw_row)
            fwd_features.append(fwd_row)
            yaw_stick = max(-60.0, min(60.0, 100.0 * math.tan(math.radians(theta)) * FX / WIDTH))
            yaw_targets.append(math.atanh(0.9 * yaw_stick / 60.0))
            fwd_targets.append(math.atanh(0.9 * max(-1.0, min(1.0, 1.0 - s))))

    yaw_w, yaw_b = _ridge(np.array(yaw_features), np.array(yaw_targets), lam)
    fwd_w, fwd_b = _ridge(np.array(fwd_features), np.array(fwd_targets), lam)

    for k, dn in enumerate(dn_types):
        params[f"w_{dn}"] = float(np.clip(yaw_w[k], -4.0, 4.0))
        params[f"u_{dn}_L"] = float(np.clip(fwd_w[2 * k], -4.0, 4.0))
        params[f"u_{dn}_R"] = float(np.clip(fwd_w[2 * k + 1], -4.0, 4.0))
    params["b_yaw"] = float(np.clip(yaw_b, -1.0, 1.0))
    params["b_fwd"] = float(np.clip(fwd_b, -1.0, 1.0))

    yaw_pred = np.array(yaw_features) @ yaw_w + yaw_b
    fwd_pred = np.array(fwd_features) @ fwd_w + fwd_b
    report = {
        "yaw_r2": _r2(np.array(yaw_targets), yaw_pred),
        "fwd_r2": _r2(np.array(fwd_targets), fwd_pred),
        "yaw_weights": {dn: float(yaw_w[k]) for k, dn in enumerate(dn_types)},
        "mean_rates": {name: float(np.mean([row[name] for row in rows])) for name in rows[0]},
    }
    return params, norm, report


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
