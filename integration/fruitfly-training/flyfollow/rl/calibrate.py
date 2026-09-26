"""Hand calibration and ridge initialization of the readout (plan 4.6).

Encoder values are hand-set (configs/controllers.yaml defaults). A spot is shown at 9 bearings x
3 sizes; we record the 10 readout channels (DN rates for FLY arms, pooled encoder features for
NOBRAIN), normalize each by its mean over all stimuli, and ridge-regress the desired tanh
arguments (yaw: proportional to bearing; forward: proportional to 1 - s) on them. Writes
data/brains/init/<ARM>__<brain_stem>.json with the param dict, normalized x, norm rates and R^2.

    python -m flyfollow.rl.calibrate --brain data/brains/pursuit_core1.npz
    python -m flyfollow.rl.calibrate --all          # core1 and its shuffles (config), NOBRAIN and PID
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from flyfollow.interfaces import BRAIN_TICK_MS, DN_TYPES, TargetFeatures, brains_dir
from flyfollow.pilot.fly_brain import FlyBrain, resolve_brain_path
from flyfollow.pilot.pursuit_decoder import channel_names
from flyfollow.rl.controllers import NO_TARGET, init_dir, nobrain_feature_matrix
from flyfollow.rl.params import POOL_TYPES, controllers_config, param_space
from flyfollow.senses.target import encoder_from_params


def stimuli(cal: dict) -> list[tuple[float, float]]:
    return [(float(b), float(s)) for s in cal["sizes"] for b in cal["bearings_deg"]]


def record_fly(brain_path: Path, params: dict, cfg: dict, seed: int = 0, repeats: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mean DN rates per (repeat, stimulus). Returns X (R*S, 10), stim rows (R*S, 3: theta_deg, s, repeat), info."""
    cal = cfg["calibration"]
    repeats = int(repeats or cal["repeats"])
    brain = FlyBrain(brain_path, seed=seed, lif=cfg.get("lif"))
    enc = encoder_from_params(params, cfg, brain.has_arousal)
    rest = enc.channels(NO_TARGET)
    ms = BRAIN_TICK_MS
    n_settle = max(1, round(cal["settle_ms"] / ms))
    n_meas = max(1, round(cal["measure_ms"] / ms))
    for _ in range(round(cal["warmup_ms"] / ms)):
        brain.tick_channels(rest, ms)
    rest_rates = np.mean([brain.tick_channels(rest, ms) for _ in range(n_meas)], axis=0)
    rng = np.random.default_rng(seed)
    stim = stimuli(cal)
    X, rows = [], []
    t0 = time.perf_counter()
    for r in range(repeats):
        for i in rng.permutation(len(stim)):
            b, s = stim[i]
            ch = enc.channels(TargetFeatures(True, math.radians(b), s, 0.0, 0.0))
            for _ in range(n_settle):
                brain.tick_channels(ch, ms)
            X.append(np.mean([brain.tick_channels(ch, ms) for _ in range(n_meas)], axis=0))
            rows.append((b, s, r))
    info = {
        "arousal_mode": enc.arousal_mode,
        "n_neurons": brain.n,
        "group_sizes": brain.t.group_sizes,
        "rest_rates": dict(zip(channel_names(DN_TYPES), np.round(rest_rates, 3).tolist())),
        "brain_ms_per_tick": 1000.0 * brain.brain_s / ((repeats * len(stim) * (n_settle + n_meas)) + n_meas + round(cal["warmup_ms"] / ms)),
        "wall_s": time.perf_counter() - t0,
    }
    return np.asarray(X), np.asarray(rows), info


def record_nobrain(params: dict, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    enc = encoder_from_params(params, cfg, False)
    M = nobrain_feature_matrix()
    X, rows = [], []
    for b, s in stimuli(cfg["calibration"]):
        X.append(M @ enc.channels(TargetFeatures(True, math.radians(b), s, 0.0, 0.0)))
        rows.append((b, s, 0))
    return np.asarray(X), np.asarray(rows), {"arousal_mode": "gain"}


def _ridge(F: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Unbounded ridge on standardized columns, unpenalized intercept (reference fit for R^2)."""
    mu, sd = F.mean(axis=0), F.std(axis=0)
    live = sd > 1e-9
    w = np.zeros(F.shape[1])
    if live.any():
        Fs = (F[:, live] - mu[live]) / sd[live]
        ws = np.linalg.solve(Fs.T @ Fs + lam * np.eye(Fs.shape[1]), Fs.T @ (y - y.mean()))
        w[live] = ws / sd[live]
    return w, float(y.mean() - mu @ w)


def _bounded_ridge(F: np.ndarray, y: np.ndarray, lam: float, wb: tuple[float, float], bb: tuple[float, float]) -> tuple[np.ndarray, float]:
    """Same ridge objective as _ridge, solved with the weights and intercept inside their parameter bounds.

    Clipping an unbounded fit afterwards can wreck it (saturated DNs have tiny variance, so their
    standardized-ridge weights are huge); a bounded solve redistributes the fit instead.
    """
    k = F.shape[1]
    sd = F.std(axis=0)
    A = np.vstack([np.hstack([F, np.ones((F.shape[0], 1))]), np.hstack([np.sqrt(lam) * np.diag(sd), np.zeros((k, 1))])])
    t = np.concatenate([y, np.zeros(k)])
    lo = np.array([wb[0]] * k + [bb[0]])
    hi = np.array([wb[1]] * k + [bb[1]])
    res = lsq_linear(A, t, bounds=(lo, hi), method="bvls")
    return res.x[:k], float(res.x[k])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")


def fit_readout(X: np.ndarray, rows: np.ndarray, cfg: dict, types: tuple[str, ...]) -> tuple[dict, dict, dict]:
    """Bounded ridge fit of both tanh arguments. Returns (readout params, norm {channel: hz}, fit report).

    r2 yaw/fwd: unbounded ridge; *_init: the bounded fit actually written; *_holdout: bounded fit on
    all repeats but the last, scored on the last.
    """
    cal, dec = cfg["calibration"], cfg["decoder"]["params"]
    names = channel_names(types)
    norm = np.maximum(X.mean(axis=0), float(cfg["decoder"]["norm_floor_hz"]))
    N = X / norm
    Fy = N[:, 1::2] - N[:, 0::2]  # (R - L) per type
    Ff = N
    zy = float(cal["yaw_z_per_deg"]) * rows[:, 0]
    zf = float(cal["fwd_z_per_unit"]) * (1.0 - rows[:, 1])
    lam = float(cal["ridge"])

    wy, by = _ridge(Fy, zy, lam)
    wf, bf = _ridge(Ff, zf, lam)
    yb = (tuple(dec["dec_w_yaw"][:2]), tuple(dec["dec_b_yaw"][:2]))
    fb = (tuple(dec["dec_u_fwd"][:2]), tuple(dec["dec_b_fwd"][:2]))
    wy_c, by_c = _bounded_ridge(Fy, zy, lam, *yb)
    wf_c, bf_c = _bounded_ridge(Ff, zf, lam, *fb)

    rep = rows[:, 2]
    hold = {}
    if np.unique(rep).size > 1:
        tr, te = rep < rep.max(), rep == rep.max()
        a, b0 = _bounded_ridge(Fy[tr], zy[tr], lam, *yb)
        c, d0 = _bounded_ridge(Ff[tr], zf[tr], lam, *fb)
        hold = {"yaw_holdout": _r2(zy[te], Fy[te] @ a + b0), "fwd_holdout": _r2(zf[te], Ff[te] @ c + d0)}

    D = X[:, 1::2] - X[:, 0::2]
    corr = {t: (float(np.corrcoef(rows[:, 0], D[:, k])[0, 1]) if D[:, k].std() > 0 else 0.0) for k, t in enumerate(types)}
    report = {
        "r2": {
            "yaw": _r2(zy, Fy @ wy + by),
            "fwd": _r2(zf, Ff @ wf + bf),
            "yaw_init": _r2(zy, Fy @ wy_c + by_c),
            "fwd_init": _r2(zf, Ff @ wf_c + bf_c),
            **hold,
        },
        "corr_bearing_vs_R_minus_L": corr,
        "at_bound": int(sum(np.isclose(v, lim).sum() for v, lims in ((wy_c, yb[0]), (by_c, yb[1]), (wf_c, fb[0]), (bf_c, fb[1])) for lim in lims)),
        "raw_fit": {"w_yaw": wy.tolist(), "b_yaw": by, "u_fwd": wf.tolist(), "b_fwd": bf},
    }
    params = {f"dec_w_yaw_{t}": float(wy_c[k]) for k, t in enumerate(types)}
    params.update({f"dec_u_fwd_{n}": float(wf_c[k]) for k, n in enumerate(names)})
    params.update({"dec_b_yaw": by_c, "dec_b_fwd": bf_c})
    return params, dict(zip(names, norm.tolist())), report


def _hand_params(arm: str, cfg: dict) -> dict:
    ps = param_space(arm)
    return ps.decode(ps.default_x())


def _write(arm: str, stem: str, params: dict, norm: dict | None, extra: dict, out: Path) -> Path:
    ps = param_space(arm)
    x = ps.encode(params)
    rec = {
        "arm": arm,
        "brain": stem,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "params": ps.decode(x),
        "x": x.tolist(),
        "norm": norm,
        **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{arm}__{stem}.json"
    p.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return p


def calibrate_fly(brain_path: str | Path, cfg: dict, seed: int = 0, repeats: int | None = None, out: Path | None = None) -> dict:
    path = resolve_brain_path(brain_path)
    arms = ("FLY-SHUF",) if re.search(r"_shuf\d*$", path.stem) else ("FLY-HAND", "FLY-CMA")
    params = _hand_params(arms[0], cfg)
    X, rows, info = record_fly(path, params, cfg, seed, repeats)
    ro, norm, report = fit_readout(X, rows, cfg, DN_TYPES)
    params.update(ro)
    extra = {
        "brain_path": str(path),
        **report,
        **info,
        "stimuli": {k: cfg["calibration"][k] for k in ("bearings_deg", "sizes", "settle_ms", "measure_ms")},
        "rates": {"rows": rows.tolist(), "X": np.round(X, 3).tolist()},
    }
    written = [str(_write(a, path.stem, params, norm, extra, out or init_dir())) for a in arms]
    return {"brain": path.stem, "arms": arms, "files": written, **report, **{k: info[k] for k in ("arousal_mode", "n_neurons", "brain_ms_per_tick", "wall_s")}}


def calibrate_nobrain(cfg: dict, out: Path | None = None) -> dict:
    params = _hand_params("NOBRAIN", cfg)
    X, rows, info = record_nobrain(params, cfg)
    ro, norm, report = fit_readout(X, rows, cfg, POOL_TYPES)
    params.update(ro)
    f = _write("NOBRAIN", "none", params, norm, {**report, **info, "rates": {"rows": rows.tolist(), "X": np.round(X, 3).tolist()}}, out or init_dir())
    return {"brain": "none", "arms": ("NOBRAIN",), "files": [str(f)], **report}


def write_pid(cfg: dict, out: Path | None = None) -> list[str]:
    files = []
    for arm in ("PID-HAND", "PID-CMA"):
        files.append(str(_write(arm, "none", _hand_params(arm, cfg), None, {"source": "plan 4.9 defaults"}, out or init_dir())))
    return files


def _summary(res: dict) -> str:
    r = res["r2"]
    fmt = lambda k: f"{r[k]:.3f}" if k in r and r[k] == r[k] else "n/a"
    s = f"{res['brain']:>28s} {'/'.join(res['arms']):18s} R2 yaw {fmt('yaw')} (init {fmt('yaw_init')}, holdout {fmt('yaw_holdout')})  "
    s += f"fwd {fmt('fwd')} (init {fmt('fwd_init')}, holdout {fmt('fwd_holdout')})  at bound {res['at_bound']}"
    if "corr_bearing_vs_R_minus_L" in res:
        s += "\n" + " " * 30 + "corr(bearing, R-L): " + ", ".join(f"{k} {v:+.2f}" for k, v in res["corr_bearing_vs_R_minus_L"].items())
    if "brain_ms_per_tick" in res:
        s += f"\n{' ' * 30}arousal {res['arousal_mode']}, {res['n_neurons']} neurons, brain {res['brain_ms_per_tick']:.1f} ms/tick, {res['wall_s']:.0f} s"
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", action="append", default=[], help="pursuit subgraph .npz (repeatable); shuffles (*_shufN) calibrate FLY-SHUF")
    ap.add_argument("--nobrain", action="store_true", help="calibrate NOBRAIN")
    ap.add_argument("--pid", action="store_true", help="write PID-HAND / PID-CMA inits from plan 4.9")
    ap.add_argument("--all", action="store_true", help="calibration.brains from configs/controllers.yaml, plus --nobrain --pid")
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="output dir (default data/brains/init)")
    a = ap.parse_args(argv)
    cfg = controllers_config()
    brains = list(a.brain)
    if a.all:
        brains += [str(brains_dir() / b) for b in cfg["calibration"]["brains"] if (brains_dir() / b).exists()]
    if not (brains or a.nobrain or a.pid or a.all):
        ap.error("nothing to do: pass --brain, --nobrain, --pid or --all")
    for b in dict.fromkeys(brains):
        print(_summary(calibrate_fly(b, cfg, a.seed, a.repeats, a.out)), flush=True)
    if a.nobrain or a.all:
        print(_summary(calibrate_nobrain(cfg, a.out)), flush=True)
    if a.pid or a.all:
        print("PID inits:", ", ".join(write_pid(cfg, a.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
