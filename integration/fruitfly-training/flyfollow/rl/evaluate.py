"""Evaluate episodes for the trainer and for eval (plan item 7).

One *item* is one episode request, a plain dict so it pickles to worker processes and to Modal:

    {"arm": "FLY-CMA", "x": [...] or None, "brain": "pursuit_core1.npz" or None,
     "seed": 10123, "kind": "follow", "profile": "train", "record": False,
     "lesion": None, "overrides": None, "warmup_s": 1.0, "tag": <opaque, echoed back>}

`x` is the normalized [0, 1] parameter vector (None means the arm's hand-calibrated init),
`brain` is a file name relative to `brains_dir()`, `overrides` is a nested dict deep-merged into
the env config (for example the smoke-test episode length), `lesion` clamps readout channels
({"DNa02_L": hz, ...}), and `audit: True` adds the readout's per-channel mean rates
(`channel_means`, the input of a lesion rerun) and `bias_audit` to the result (plan 4.7).
`evaluate_item` never raises: an
episode that fails returns ok=False with the error text, so one bad episode cannot kill a
generation. The trainer scores those as a large penalty and logs them.

Everything here runs identically in a local process pool and inside a Modal container
(`modal_app.evaluate_chunk` calls `evaluate_chunk_local`).
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker: every episode is single-threaded Python, and the pools run
# one worker per core (or hyperthread). Must happen before numpy is imported in a fresh worker.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import numpy as np

from flyfollow.interfaces import REPO_ROOT, brains_dir, configs_dir

ITEM_KEYS = ("arm", "x", "brain", "seed", "kind", "profile", "record", "lesion", "overrides", "warmup_s", "audit", "stick_stats", "tag")
ENV_CONFIG_NAME = "env.yaml"
MAX_TRACEBACK_CHARS = 4000
CONTROLLER_CACHE_SIZE = 4


def ensure_flydrones() -> None:
    """Import flydrones, falling back to the vendored source tree.

    The venv's editable-install .pth files can carry the macOS "hidden" flag, which makes
    Python 3.12 skip them. Workers are fresh interpreters, so they need this too.
    """
    try:
        import flydrones  # noqa: F401
    except ImportError:
        src = str(REPO_ROOT / "third_party" / "FlyDrones" / "src")
        if src not in sys.path:
            sys.path.insert(0, src)


ensure_flydrones()


def make_item(
    arm: str,
    seed: int,
    kind: str,
    *,
    x: Any = None,
    brain: str | None = None,
    profile: str = "train",
    record: bool = False,
    lesion: dict | None = None,
    overrides: dict | None = None,
    warmup_s: float = 1.0,
    audit: bool = False,
    stick_stats: bool = False,
    tag: Any = None,
) -> dict:
    """Build a well-formed item (x becomes a plain list of floats)."""
    if x is not None:
        x = [float(v) for v in np.asarray(x, dtype=float).ravel()]
    return {
        "arm": arm,
        "x": x,
        "brain": brain,
        "seed": int(seed),
        "kind": kind,
        "profile": profile,
        "record": bool(record),
        "lesion": lesion,
        "overrides": overrides,
        "warmup_s": float(warmup_s),
        "audit": bool(audit),
        "stick_stats": bool(stick_stats),
        "tag": tag,
    }


# --------------------------------------------------------------------------------------------
# Env config and overrides
# --------------------------------------------------------------------------------------------


def deep_merge(base: dict, over: dict | None) -> dict:
    """Return base with over merged in recursively (over wins). Inputs are not modified."""
    out = copy.deepcopy(base) if base else {}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_env_config() -> dict:
    import yaml

    path = configs_dir() / ENV_CONFIG_NAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _overrides_key(overrides: dict | None) -> str:
    return json.dumps(overrides or {}, sort_keys=True, default=str)


# --------------------------------------------------------------------------------------------
# Per-process caches
# --------------------------------------------------------------------------------------------

_ENV_BASE: dict | None = None
_ENV_CACHE: dict[str, Any] = {}
_INIT_CACHE: dict[tuple, np.ndarray] = {}
_CTRL_CACHE: OrderedDict = OrderedDict()
_STATS = {"items": 0, "env_builds": 0, "ctrl_builds": 0, "ctrl_hits": 0, "started": time.time()}


def clear_caches() -> None:
    global _ENV_BASE
    _ENV_BASE = None
    _ENV_CACHE.clear()
    _INIT_CACHE.clear()
    _CTRL_CACHE.clear()


def get_env(overrides: dict | None = None):
    """One PursuitEnv per distinct override set, reused across episodes (reset re-randomizes)."""
    global _ENV_BASE
    key = _overrides_key(overrides)
    env = _ENV_CACHE.get(key)
    if env is None:
        from flyfollow.rl.env import PursuitEnv

        if _ENV_BASE is None:
            _ENV_BASE = load_env_config()
        cfg = apply_overrides(_ENV_BASE, overrides)
        env = PursuitEnv(cfg if cfg else None)
        _ENV_CACHE[key] = env
        _STATS["env_builds"] += 1
    return env


def apply_overrides(env_cfg: dict, overrides: dict | None) -> dict:
    """Merge item overrides into the env config.

    `episode_s` is a shorthand (the trainer's --episode-s): it caps every episode length key the
    env config knows. Everything else is deep-merged as is.
    """
    overrides = dict(overrides or {})
    episode_s = overrides.pop("episode_s", None)
    cfg = deep_merge(env_cfg, overrides)
    if episode_s is not None:
        cfg = set_episode_length(cfg, float(episode_s))
    return cfg


# Episode length keys of configs/env.yaml (Engineer A's PursuitEnv reads cfg["episode"][...]).
EPISODE_LENGTH_KEYS = ("follow_s", "approach_s")


def set_episode_length(cfg: dict, seconds: float) -> dict:
    """Cap episode.follow_s and episode.approach_s to `seconds` (smoke tests)."""
    cfg = copy.deepcopy(cfg)
    ep = cfg.setdefault("episode", {})
    for k in EPISODE_LENGTH_KEYS:
        ep[k] = min(float(ep[k]), seconds) if isinstance(ep.get(k), (int, float)) else seconds
    return cfg


def resolve_brain(brain: str | None) -> str | None:
    if not brain:
        return None
    p = Path(brain)
    if not p.is_absolute():
        p = brains_dir() / p
    return str(p)


def get_init_x(arm: str, brain_path: str | None) -> np.ndarray:
    key = (arm, brain_path)
    x = _INIT_CACHE.get(key)
    if x is None:
        from flyfollow.rl.controllers import init_x

        x = np.asarray(init_x(arm, brain_path), dtype=float)
        _INIT_CACHE[key] = x
    return x


def get_controller(arm: str, x: np.ndarray, brain_path: str | None, lesion: dict | None):
    """Small LRU of controllers keyed by (arm, x, brain, lesion): the K episodes of one candidate
    that land in the same worker reuse the built brain. Controller.reset(settings, seed) resets
    all episode state, so reuse does not change results (checked in the tests)."""
    key = (arm, np.asarray(x, dtype=np.float64).tobytes(), brain_path, _overrides_key(lesion))
    ctrl = _CTRL_CACHE.get(key)
    if ctrl is not None:
        _CTRL_CACHE.move_to_end(key)
        _STATS["ctrl_hits"] += 1
        return ctrl
    from flyfollow.rl.controllers import make_controller

    ctrl = make_controller(arm, np.asarray(x, dtype=float), brain_path=brain_path, lesion=lesion)
    _CTRL_CACHE[key] = ctrl
    _STATS["ctrl_builds"] += 1
    while len(_CTRL_CACHE) > CONTROLLER_CACHE_SIZE:
        _CTRL_CACHE.popitem(last=False)
    return ctrl


def run_one(item: dict):
    """Run the episode for one item and return the EpisodeResult (raises on failure)."""
    from flyfollow.rl.rollout import run_episode

    brain_path = resolve_brain(item.get("brain"))
    x = item.get("x")
    x = get_init_x(item["arm"], brain_path) if x is None else np.asarray(x, dtype=float)
    env = get_env(item.get("overrides"))
    ctrl = get_controller(item["arm"], x, brain_path, item.get("lesion"))
    probe = StickProbe(ctrl) if item.get("stick_stats") else None
    res = run_episode(
        env,
        probe or ctrl,
        seed=int(item["seed"]),
        kind=item["kind"],
        profile=item.get("profile", "train"),
        record=bool(item.get("record", False)),
        warmup_s=float(item.get("warmup_s", 1.0)),
    )
    if probe is not None:
        res.metrics.update(probe.stats())
    if not item.get("audit"):
        return res
    d = res.to_dict()
    dec = getattr(ctrl, "decoder", None)
    d["channel_means"] = dec.channel_means() if dec is not None else None
    d["bias_audit"] = dec.bias_audit() if dec is not None else None
    return d


class StickProbe:
    """Pass-through Controller that records the controller's own (raw, pre-governor) yaw stick.

    Adds per-episode stick smoothness metrics, per 50 ms control loop (item flag stick_stats):
    yaw_raw_step_abs = mean |yaw_t - yaw_(t-1)|, yaw_raw_step_rms, fb_raw_step_abs, and
    yaw_raw_abs_mean. The env's yaw_jerk is the same idea on the SENT stick (after the governor's
    clamps and slew limits), squared and scaled by 0.002 like the reward's jerk term.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "probe")
        self.yaw: list[float] = []
        self.fb: list[float] = []

    def reset(self, settings, seed: int) -> None:
        self.yaw.clear()
        self.fb.clear()
        self.inner.reset(settings, seed)

    def warmup(self, seconds: float) -> None:
        self.inner.warmup(seconds)

    def act(self, box, settings, dt: float):
        yaw, fb = self.inner.act(box, settings, dt)
        self.yaw.append(float(yaw))
        self.fb.append(float(fb))
        return yaw, fb

    def __getattr__(self, name: str):  # brain_s, decoder, ... of the wrapped controller
        return getattr(self.inner, name)

    def stats(self) -> dict[str, float]:
        if len(self.yaw) < 2:
            return {}
        dy = np.diff(np.asarray(self.yaw))
        dfb = np.diff(np.asarray(self.fb))
        return {
            "yaw_raw_step_abs": float(np.mean(np.abs(dy))),
            "yaw_raw_step_rms": float(np.sqrt(np.mean(dy * dy))),
            "fb_raw_step_abs": float(np.mean(np.abs(dfb))),
            "yaw_raw_abs_mean": float(np.mean(np.abs(self.yaw))),
        }


# Tests swap this for a fake episode runner (same signature as run_one).
EPISODE_RUNNER: Callable[[dict], Any] = run_one


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(u) for u in v]
    return v


def evaluate_item(item: dict) -> dict:
    """Run one episode. Never raises: failures come back with ok=False and the error text."""
    t0 = time.perf_counter()
    base = {
        "arm": item.get("arm"),
        "tag": item.get("tag"),
        "seed": item.get("seed"),
        "kind": item.get("kind"),
        "profile": item.get("profile", "train"),
    }
    try:
        res = EPISODE_RUNNER(item)
        d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        d = _jsonable(d)
        ret = d.get("ret")
        if ret is None or not math.isfinite(float(ret)):
            raise ValueError(f"non-finite episode return {ret!r}")
        d.update(base)
        d["ok"] = True
        d["error"] = None
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        d = dict(base)
        d.update(
            ok=False,
            ret=None,
            terms={},
            metrics={},
            n_ticks=0,
            wall_s=0.0,
            brain_s=0.0,
            trace=None,
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc()[-MAX_TRACEBACK_CHARS:],
        )
    d["item_wall_s"] = time.perf_counter() - t0
    d["pid"] = os.getpid()
    _STATS["items"] += 1
    return d


def error_result(item: dict, message: str) -> dict:
    """The result dict for an item that never ran (lost worker, failed container)."""
    return {
        "arm": item.get("arm"),
        "tag": item.get("tag"),
        "seed": item.get("seed"),
        "kind": item.get("kind"),
        "profile": item.get("profile", "train"),
        "ok": False,
        "ret": None,
        "terms": {},
        "metrics": {},
        "n_ticks": 0,
        "wall_s": 0.0,
        "brain_s": 0.0,
        "trace": None,
        "error": message,
        "traceback": None,
        "item_wall_s": 0.0,
        "pid": None,
    }


# --------------------------------------------------------------------------------------------
# Process pool (persistent, so worker caches survive across generations)
# --------------------------------------------------------------------------------------------

_POOL: ProcessPoolExecutor | None = None
_POOL_KEY: tuple | None = None


def default_start_method() -> str:
    # forkserver on Linux (Modal): children fork from a clean single-threaded server, never from a
    # process running Modal's client threads, and no __main__ re-import. spawn on macOS.
    return "forkserver" if sys.platform.startswith("linux") else "spawn"


def get_pool(processes: int, start_method: str | None = None) -> ProcessPoolExecutor:
    global _POOL, _POOL_KEY
    start_method = start_method or default_start_method()
    key = (processes, start_method)
    if _POOL is None or _POOL_KEY != key:
        shutdown_pool()
        ctx = mp.get_context(start_method)
        if start_method == "forkserver":
            ctx.set_forkserver_preload(["flyfollow.rl.evaluate"])
        _POOL = ProcessPoolExecutor(max_workers=processes, mp_context=ctx)
        _POOL_KEY = key
    return _POOL


def shutdown_pool() -> None:
    global _POOL, _POOL_KEY
    if _POOL is not None:
        _POOL.shutdown(wait=False, cancel_futures=True)
    _POOL = None
    _POOL_KEY = None


def evaluate_items(items: list[dict], processes: int = 1, start_method: str | None = None) -> list[dict]:
    """Evaluate items in order. processes > 1 uses a persistent process pool.

    A crashed worker (segfault, OOM kill) turns its items into error results and the pool is
    rebuilt for the next call; it never raises.
    """
    if processes <= 1 or len(items) <= 1:
        return [evaluate_item(it) for it in items]
    pool = get_pool(processes, start_method)
    # Longest-first submission (follow before approach, recorded before plain) packs better.
    order = sorted(range(len(items)), key=lambda i: (items[i].get("kind") != "follow", i))
    futures = {}
    broken = False
    try:
        for i in order:
            futures[i] = pool.submit(evaluate_item, items[i])
    except BrokenProcessPool:
        broken = True
    out: list[dict] = []
    for i, it in enumerate(items):
        fut = futures.get(i)
        if fut is None:
            out.append(error_result(it, "BrokenProcessPool: item not submitted"))
            continue
        try:
            out.append(fut.result())
        except BrokenProcessPool as e:
            broken = True
            out.append(error_result(it, f"BrokenProcessPool: {e}"))
        except Exception as e:  # noqa: BLE001
            out.append(error_result(it, f"{type(e).__name__}: {e}"))
    if broken:
        shutdown_pool()
    return out


def evaluate_chunk_local(items: list[dict], workers: int = 1, start_method: str | None = None) -> dict:
    """Evaluate one chunk and time it. Shared by the local backend and Modal's evaluate_chunk."""
    t_start = time.time()
    t0 = time.perf_counter()
    results = evaluate_items(items, processes=workers, start_method=start_method)
    return {
        "results": results,
        "wall_s": time.perf_counter() - t0,
        "t_start": t_start,
        "workers": workers,
        "n_items": len(items),
    }


def worker_stats() -> dict:
    return dict(_STATS)
