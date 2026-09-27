"""Controllers for every arm (plan 4.7) and the factory the trainer calls.

    FLY-HAND / FLY-CMA / FLY-SHUF: TargetEncoder -> FlyBrain (LIF) -> PursuitDecoder
    NOBRAIN: same encoder; per-side bin rates pooled into 5 features per side, fed to the same
             PursuitDecoder in place of the 5 DN types (no spiking)
    PID-HAND / PID-CMA: flyfollow.pilot.pid.PIDController
    FLY-YAW-HAND / FLY-YAW / FLY-SHUF-YAW / NOBRAIN-YAW (plan 4.3 yaw-only fallback): YawOnlyController,
             yaw from the wrapped fly or no-brain controller, forward from PID-HAND's range loop

make_controller(arm, x) builds one from a normalized parameter vector; x=None means the arm's
hand-calibration init (data/brains/init/<ARM>__<brain_stem>.json, written by flyfollow.rl.calibrate).
Yaw-only arms have no init files of their own: they project their base arm's init (params.base_arm).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from flyfollow.interfaces import (
    BRAIN_TICK_MS,
    DN_TYPES,
    DT,
    N_AZIMUTH_BINS,
    BoxState,
    Controller,
    Settings,
    TargetFeatures,
    brains_dir,
    target_features,
)
from flyfollow.pilot.fly_brain import FlyBrain, resolve_brain_path
from flyfollow.pilot.pursuit_decoder import PursuitDecoder, channel_names
from flyfollow.rl.params import N_POOLS, POOL_TYPES, YAW_BASE, base_arm, controllers_config, param_space
from flyfollow.senses.target import N_CHANNELS, encoder_from_params

NO_TARGET = TargetFeatures(False, 0.0, 0.0, 0.0, 0.0)
FLY_ARMS = ("FLY-HAND", "FLY-CMA", "FLY-SHUF")
SHUF_ARMS = ("FLY-SHUF", "FLY-SHUF-YAW")


def is_fly_arm(arm: str) -> bool:
    return base_arm(arm) in FLY_ARMS


# ---------------------------------------------------------------- calibration files
def init_dir() -> Path:
    return brains_dir() / "init"


def brain_stem(arm: str, brain_path: str | Path | None) -> str:
    if is_fly_arm(arm):
        return resolve_brain_path(brain_path or controllers_config()["calibration"]["default_brain"]).stem
    return "none"


def calibration_path(arm: str, brain_path: str | Path | None = None) -> Path:
    return init_dir() / f"{base_arm(arm)}__{brain_stem(arm, brain_path)}.json"


@lru_cache(maxsize=64)
def _read_json(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_calibration(arm: str, brain_path: str | Path | None = None) -> dict:
    """Calibration record for (arm, brain). FLY arms fall back to another FLY arm's file on the same brain.

    Yaw-only arms read their base arm's file (FLY-YAW: FLY-CMA / FLY-HAND, FLY-SHUF-YAW: FLY-SHUF, ...).
    """
    stem = brain_stem(arm, brain_path)
    arm = base_arm(arm)
    order = [arm] + ([a for a in FLY_ARMS if a != arm] if arm in FLY_ARMS else []) + (["PID-HAND"] if arm == "PID-CMA" else [])
    for a in order:
        p = init_dir() / f"{a}__{stem}.json"
        if p.exists():
            return _read_json(str(p), p.stat().st_mtime_ns)
    raise FileNotFoundError(
        f"no calibration for {arm} on {stem} in {init_dir()}; run: python -m flyfollow.rl.calibrate "
        + ("--pid" if arm.startswith("PID") else "--nobrain" if arm == "NOBRAIN" else f"--brain {brain_path or stem}")
    )


def init_x(arm: str, brain_path: str | Path | None = None, cfg: dict | None = None) -> np.ndarray:
    """Normalized init vector for the arm (hand calibration). PID arms fall back to the 4.9 defaults.

    Yaw-only arms get their base arm's init projected onto their 34 parameters.
    """
    ps = param_space(arm, cfg)
    try:
        rec = load_calibration(arm, brain_path)
    except FileNotFoundError:
        if arm.startswith("PID"):
            return ps.default_x()
        raise
    return ps.encode(rec["params"])


def _norm(arm: str, brain_path, cfg: dict) -> np.ndarray:
    names = channel_names(POOL_TYPES if base_arm(arm) == "NOBRAIN" else DN_TYPES)
    if cfg.get("norm") is not None:
        n = cfg["norm"]
        v = np.array([n[k] for k in names] if isinstance(n, dict) else n, dtype=np.float64)
    else:
        rec = load_calibration(arm, brain_path)
        v = np.array([rec["norm"][k] for k in names], dtype=np.float64)
    return np.maximum(v, float(cfg["decoder"]["norm_floor_hz"]))


# ---------------------------------------------------------------- controllers
class FlyController:
    """Encoder -> LIF pursuit subgraph -> readout. brain_s accumulates brain wall time since reset.

    Read-only live state for visualization, refreshed by every act() (and warmup tick) without copies;
    the dict views are built only when read:
        last_counts       np.ndarray (n,) spike counts per neuron over the last tick (subgraph indexing)
        last_input_rates  {input group: per-neuron rate (Hz) fed to the LIF, group index order}
        last_dn_rates     {output group: raw rate (Hz) before filtering and lesion}
        last_sticks       (yaw, fb) as returned by act()
        connectome, groups, brain_path
    """

    def __init__(self, name: str, params: dict[str, float], brain_path: str | Path, norm: np.ndarray, cfg: dict, lesion: dict | None = None):
        self.name = name
        self.params = params
        self.brain = FlyBrain(brain_path, seed=0, lif=cfg.get("lif"))
        self.encoder = encoder_from_params(params, cfg, self.brain.has_arousal)
        self.decoder = PursuitDecoder(params, norm, DN_TYPES, lesion)
        self._rest = self.encoder.channels(NO_TARGET)
        self.last_channels = self._rest
        self.last_rates = np.zeros(len(self.decoder.names))
        self.last_sticks = (0.0, 0.0)

    @property
    def brain_s(self) -> float:
        return self.brain.brain_s

    @property
    def brain_path(self) -> Path:
        return self.brain.path

    @property
    def connectome(self):
        return self.brain.connectome

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return self.brain.groups

    @property
    def last_counts(self) -> np.ndarray:
        return self.brain.last_counts

    @property
    def last_input_rates(self) -> dict[str, np.ndarray]:
        return self.brain.input_rates_by_group(self.last_channels)

    @property
    def last_dn_rates(self) -> dict[str, float]:
        return dict(zip(self.decoder.names, self.last_rates.tolist()))

    def reset(self, settings: Settings, seed: int) -> None:
        self.brain.reset(seed)
        self.decoder.reset()
        self.last_sticks = (0.0, 0.0)

    def warmup(self, seconds: float) -> None:
        self.last_channels = self._rest
        for _ in range(round(seconds / DT)):
            self.last_rates = self.brain.tick_channels(self._rest, BRAIN_TICK_MS)
            self.decoder.filter(self.last_rates, DT)

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        self.last_channels = ch = self.encoder.channels(target_features(box, settings))
        self.last_rates = self.brain.tick_channels(ch, dt * 1000.0)
        self.last_sticks = self.decoder.update(self.last_rates, settings, dt)
        return self.last_sticks


def pool_matrix(n_pools: int = N_POOLS, n_bins: int = N_AZIMUTH_BINS) -> np.ndarray:
    """(n_pools, n_bins) triangular weights over bins, each row sums to 1; pool 0 frontal."""
    c = np.linspace(0, n_bins - 1, n_pools)
    half = (n_bins - 1) / (n_pools - 1)
    w = np.maximum(0.0, 1.0 - np.abs(np.arange(n_bins)[None, :] - c[:, None]) / half)
    return w / w.sum(axis=1, keepdims=True)


def nobrain_feature_matrix() -> np.ndarray:
    """(10, 22) map from encoder channels to pooled features in [P0_L, P0_R, P1_L, ...] order."""
    P = pool_matrix()
    M = np.zeros((2 * N_POOLS, N_CHANNELS))
    M[0::2, :N_AZIMUTH_BINS] = P
    M[1::2, N_AZIMUTH_BINS : 2 * N_AZIMUTH_BINS] = P
    return M


class NoBrainController:
    """Same encoder and readout form as the fly, with pooled LC10a bin rates standing in for DN rates."""

    brain_s = 0.0

    def __init__(self, name: str, params: dict[str, float], norm: np.ndarray, cfg: dict, lesion: dict | None = None):
        self.name = name
        self.params = params
        self.encoder = encoder_from_params(params, cfg, False)
        self.decoder = PursuitDecoder(params, norm, POOL_TYPES, lesion)
        self.M = nobrain_feature_matrix()
        self._rest = self.M @ self.encoder.channels(NO_TARGET)
        self.last_rates = np.zeros(len(self.decoder.names))
        self.last_sticks = (0.0, 0.0)

    @property
    def last_dn_rates(self) -> dict[str, float]:
        """Pooled features standing in for DN rates (same slots as the fly readout)."""
        return dict(zip(self.decoder.names, self.last_rates.tolist()))

    def features(self, f: TargetFeatures) -> np.ndarray:
        return self.M @ self.encoder.channels(f)

    def reset(self, settings: Settings, seed: int) -> None:
        self.decoder.reset()

    def warmup(self, seconds: float) -> None:
        for _ in range(round(seconds / DT)):
            self.decoder.filter(self._rest, DT)

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        self.last_rates = self.features(target_features(box, settings))
        self.last_sticks = self.decoder.update(self.last_rates, settings, dt)
        return self.last_sticks


class YawOnlyController:
    """Plan 4.3 yaw-only fly: yaw from the inner controller, forward from PID-HAND's range loop.

    The forward stick is exactly PIDController(PID-HAND gains).act(...)[1] (Kp_f 40 per m, d0 0.15 m,
    not trained). The inner controller runs unchanged (its own forward output is ignored), so lesion,
    bias_audit (yaw side) and the visualization hooks work through .inner / the forwarded attributes.
    last_sticks is this wrapper's final (yaw, fb).
    """

    def __init__(self, name: str, inner: FlyController | NoBrainController, forward_gains: dict | None = None):
        from flyfollow.pilot.pid import PIDController

        ps = param_space("PID-HAND")
        gains = dict(zip(ps.names, ps.default.tolist()))  # exact 4.9 values
        if forward_gains:  # e.g. PID-CMA's tuned Kp_f and d0 (cfg yaw_only.forward_gains); yaw gains are unused here
            gains.update({k: float(v) for k, v in forward_gains.items() if k in gains})
        self.name = name
        self.inner = inner
        self.pid = PIDController(gains, name=f"{name}/fwd")
        self.last_sticks = (0.0, 0.0)

    # forwarded state (read-only)
    brain_s = property(lambda self: self.inner.brain_s)
    decoder = property(lambda self: self.inner.decoder)
    encoder = property(lambda self: self.inner.encoder)
    params = property(lambda self: self.inner.params)
    last_dn_rates = property(lambda self: self.inner.last_dn_rates)
    last_rates = property(lambda self: self.inner.last_rates)

    def __getattr__(self, attr: str):
        # last_counts, last_input_rates, connectome, groups, brain_path, brain, ... of a FlyController
        if attr in ("inner", "pid", "__getstate__", "__setstate__"):
            raise AttributeError(attr)
        return getattr(self.inner, attr)

    def reset(self, settings: Settings, seed: int) -> None:
        self.inner.reset(settings, seed)
        self.pid.reset(settings, seed)
        self.last_sticks = (0.0, 0.0)

    def warmup(self, seconds: float) -> None:
        self.inner.warmup(seconds)
        self.pid.warmup(seconds)

    def act(self, box: BoxState, settings: Settings, dt: float) -> tuple[float, float]:
        yaw = self.inner.act(box, settings, dt)[0]
        fb = self.pid.act(box, settings, dt)[1]
        self.last_sticks = (yaw, fb)
        return self.last_sticks


# ---------------------------------------------------------------- factory
def make_controller(
    arm: str,
    x: np.ndarray | None = None,
    brain_path: str | None = None,
    cfg: dict | None = None,
    lesion: dict | None = None,
) -> Controller:
    """Build the controller for `arm` from a normalized parameter vector (None: hand-calibration init).

    FLY arms need brain_path (default configs calibration.default_brain; FLY-SHUF and FLY-SHUF-YAW
    must pass their shuffled file). cfg deep-merges over configs/controllers.yaml; cfg["norm"]
    overrides the calibration normalization rates. lesion clamps readout channels ({"DNa02_L": hz, ...}).
    Yaw-only arms take a 34-dim x; the unused forward readout parameters come from the base arm's init.
    """
    c = controllers_config(cfg)
    ps = param_space(arm, cfg)
    if arm in SHUF_ARMS and brain_path is None:
        raise ValueError(f"{arm} needs brain_path of a shuffled subgraph")
    if x is None:
        x = init_x(arm, brain_path, cfg)
    params = ps.decode(np.asarray(x, dtype=np.float64))
    if arm in YAW_BASE:
        base = YAW_BASE[arm]
        full = param_space(base, cfg).decode(init_x(base, brain_path, cfg))
        inner = make_controller(base, param_space(base, cfg).encode({**full, **params}), brain_path, cfg, lesion)
        inner.name = arm
        return YawOnlyController(arm, inner, ((cfg or {}).get("yaw_only") or {}).get("forward_gains"))
    if arm.startswith("PID"):
        from flyfollow.pilot.pid import PIDController

        return PIDController(params, name=arm)
    if arm == "NOBRAIN":
        return NoBrainController(arm, params, _norm(arm, None, c), c, lesion)
    if arm in FLY_ARMS:
        path = resolve_brain_path(brain_path or c["calibration"]["default_brain"])
        return FlyController(arm, params, path, _norm(arm, path, c), c, lesion)
    raise ValueError(f"unknown arm {arm!r}")
