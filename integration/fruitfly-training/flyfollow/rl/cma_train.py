"""CMA-ES driver for the trained arms (plan 4.4 to 4.6, item 7).

    python -m flyfollow.rl.cma_train --arm FLY-CMA --seed 1 --gens 150 --backend local
    python -m flyfollow.rl.cma_train --arm NOBRAIN --seed 1 --gens 2 --pop 4 --k-follow 1 --k-approach 0 \
        --episode-s 10 --backend local --tag smoke
    python -m flyfollow.rl.cma_train --arm FLY-YAW --seed 3 --gens 40 --tag smooth --sigma0 0.02 --select-every 5 \
        --init-from data/runs/FLY-YAW_s3_v1/best.json --env-override reward.w_j=26.4      # smoothness fine-tune

What one generation does:
1. ask pycma for `popsize` candidates on the normalized [0, 1] vector (bounded);
2. evaluate every candidate on the same K training seeds (common random numbers; K_follow follow
   plus K_approach approach episodes), plus PID-HAND on any of those seeds not yet cached;
3. score each episode as ret / max(|PID-HAND ret on the same seed|, floor), clipped; fitness =
   w_follow * mean(follow scores) + w_approach * mean(approach scores); errors score `error_score`;
4. tell pycma -fitness (pycma minimizes);
5. every `selection.every` generations, score the distribution mean on SELECTION_SEEDS the same
   way and keep the best mean ever seen as best.json (the "best checkpoint");
6. append one JSON line to log.jsonl and atomically rewrite state.pkl. A restarted driver resumes
   from state.pkl and continues exactly as if it had never stopped.

Backends: `local` (a process pool on this machine) and `modal` (modal_app.evaluate_chunk.map).
Both evaluate the same item dicts with flyfollow.rl.evaluate.evaluate_item.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
import json
import math
import pickle
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from flyfollow.interfaces import (
    ARMS,
    REPO_ROOT,
    SELECTION_SEEDS,
    TEST_SEEDS,
    TRAIN_SEED_BASE,
    configs_dir,
    data_root,
)
from flyfollow.rl import evaluate as ev

STATE_VERSION = 1
TRAIN_CONFIG_NAME = "train.yaml"
SEEDS_PER_RUN_SEED = 1_000_000  # training seed = BASE + run_seed * this + gen * SEEDS_PER_GEN + i
SEEDS_PER_GEN = 100
# Arm families. Yaw-only arms (interfaces.YAW_ONLY_ARMS: the fly steers, the PID sets forward) run
# the same brain as their full twins, so they share brain files, chunking and cost.
FLY_ARMS = ("FLY-HAND", "FLY-CMA", "FLY-SHUF", "FLY-YAW-HAND", "FLY-YAW", "FLY-SHUF-YAW")
SHUFFLE_ARMS = ("FLY-SHUF", "FLY-SHUF-YAW")  # run seed i trains on shuffle i
DEFAULT_RUN_ARMS = ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA")
# Same per-episode compute as another arm, for cost projections without a smoke test of their own.
DEFAULT_SAME_COST_AS = {
    "FLY-YAW": "FLY-CMA",
    "FLY-SHUF-YAW": "FLY-CMA",
    "FLY-YAW-HAND": "FLY-CMA",
    "FLY-SHUF": "FLY-CMA",
    "FLY-HAND": "FLY-CMA",
    "NOBRAIN-YAW": "NOBRAIN",
}

# Fallback when configs/train.yaml is missing (tests); the YAML file is the source of truth.
DEFAULT_CONFIG: dict = {
    "tag": "v1",
    "env_overrides": {},
    "cma": {"sigma0": 0.2, "popsize": 32, "generations": 150, "bounds": [0.0, 1.0]},
    "episodes": {"k_follow": 5, "k_approach": 3, "profile": "train", "warmup_s": 1.0, "episode_s": None},
    "fitness": {"pid_arm": "PID-HAND", "denom_floor": 1.0, "w_follow": 0.5, "w_approach": 0.5, "error_score": -20.0},
    "selection": {"every": 10, "at_start": True, "at_end": True, "n_follow": 40, "n_approach": 24},
    "brain": {"core": 1, "file": "pursuit_core{core}.npz", "shuffle_file": "pursuit_core{core}_shuf{seed}.npz"},
    "runs": {"arms": list(DEFAULT_RUN_ARMS), "seeds": [1, 2, 3], "gens_later_seeds": None},
    "local": {"processes": os.cpu_count() or 1, "start_method": None},
    "chunking": {"items_per_chunk": 16, "by_arm": {"NOBRAIN": 96, "NOBRAIN-YAW": 96, "PID-CMA": 96}},
    "eval": {
        "n_follow": 125,
        "n_approach": 75,
        "sets": {"test": "train", "demo": "demo", "stress": "stress"},
        "lesion_n": 40,
        "lesion_arms": ["FLY-YAW", "FLY-CMA"],
        "references": ["FLY-YAW-HAND", "FLY-HAND", "PID-HAND"],
    },
    "modal": {
        "app_name": "flyfollow",
        "volume": "flyfollow-data",
        "eval": {
            "cpu": 8.0,
            "workers_per_container": 16,
            "memory_mib_per_worker": 1024,
            "max_containers": 80,
            "scaledown_window_s": 60,
            "timeout_s": 1800,
            "retries": 2,
            "start_method": "forkserver",
        },
        "train": {"cpu": 0.5, "memory_mib": 1024, "timeout_s": 86400, "retries": 3},
        "orchestrator": {"cpu": 0.125, "memory_mib": 512, "timeout_s": 86400, "max_restarts_per_run": 3},
        "commit_every": 1,
    },
    "prices": {"cpu_core_hour_usd": 0.0473, "mem_gib_hour_usd": 0.008, "billing_overhead_factor": 1.2},
    "budget": {
        "total_usd": 800.0,
        "per_run_usd": None,
        "share": {"FLY-YAW": 0.4, "FLY-SHUF-YAW": 0.3, "NOBRAIN-YAW": 0.1, "PID-CMA": 0.1, "FLY-CMA": 0.1},
    },
    "projection": {
        "packing_efficiency": 0.8,
        "gen_overhead_s": 3.0,
        "episode_wall_s": {"FLY-YAW": 5.5, "FLY-SHUF-YAW": 5.5, "NOBRAIN-YAW": 0.1, "FLY-CMA": 5.5, "FLY-SHUF": 5.5, "NOBRAIN": 0.1, "PID-CMA": 0.05},
        "same_cost_as": dict(DEFAULT_SAME_COST_AS),
    },
    "chart": {},
}


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """DEFAULT_CONFIG, then configs/train.yaml, then `overrides` (all deep-merged)."""
    import yaml

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    path = Path(path) if path else configs_dir() / TRAIN_CONFIG_NAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            cfg = ev.deep_merge(cfg, yaml.safe_load(f) or {})
    return ev.deep_merge(cfg, overrides or {})


def brain_for(arm: str, run_seed: int, cfg: dict) -> str | None:
    """Brain file (relative to brains_dir()) for an arm: the core for FLY arms, shuffle i for
    FLY-SHUF and FLY-SHUF-YAW run seed i, none for NOBRAIN and PID arms."""
    if arm not in FLY_ARMS:
        return None
    b = cfg["brain"]
    if arm in SHUFFLE_ARMS:
        return b["shuffle_file"].format(core=b["core"], seed=run_seed)
    return b["file"].format(core=b["core"])


def run_name(arm: str, run_seed: int, tag: str) -> str:
    return f"{arm}_s{run_seed}_{tag}"


def runs_dir() -> Path:
    return data_root() / "runs"


def env_overrides(cfg: dict) -> dict | None:
    """The `overrides` of every item a run evaluates: candidates, PID-HAND denominators and selection.

    cfg["env_overrides"] (for example {"reward": {"w_j": 26.4}} for a smoothness fine-tune) is
    deep-merged into configs/env.yaml by evaluate_item; episodes.episode_s adds the smoke-test
    length cap. The same dict goes into the PID-HAND cache key, so the fitness normalization always
    uses denominators computed under the same env.
    """
    o = copy.deepcopy(cfg.get("env_overrides") or {})
    s = cfg["episodes"].get("episode_s")
    if s:
        o["episode_s"] = float(s)
    return o or None


def parse_env_overrides(pairs: list[str] | str | None) -> dict:
    """["reward.w_j=26.4", "episode.follow_s=30"] (or one comma-separated string) -> nested dict."""
    import yaml

    if isinstance(pairs, str):
        pairs = [p for p in pairs.split(",") if p.strip()]
    out: dict = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"env override {pair!r} is not KEY=VALUE (dotted key, for example reward.w_j=26.4)")
        d = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = yaml.safe_load(raw.strip())
    return out


def resolve_data_path(path: str | Path) -> Path:
    """A path given as in the repo ("data/runs/X/best.json") that also works under FLYFOLLOW_DATA (Modal: /data)."""
    p = Path(path)
    if p.is_absolute():
        return p
    cands = [data_root() / p]
    if p.parts and p.parts[0] == "data":
        cands.append(data_root() / Path(*p.parts[1:]))
    cands.append(REPO_ROOT / p)
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"{path} not found (tried {', '.join(str(c) for c in cands)})")


def load_init(path: str | Path) -> dict:
    """Read a best.json as the starting point of a fine-tune: its normalized x plus provenance."""
    import hashlib

    p = resolve_data_path(path)
    raw = p.read_bytes()
    b = json.loads(raw)
    if "x" not in b:
        raise ValueError(f"{p} has no normalized 'x'")
    return {
        "x": [float(v) for v in b["x"]],
        "source": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "arm": b.get("arm"),
        "run_seed": b.get("run_seed"),
        "tag": b.get("tag"),
        "gen": b.get("gen"),
        "sel_fitness": b.get("sel_fitness"),
        "brain": b.get("brain"),
        "calibration": b.get("calibration"),
    }


# --------------------------------------------------------------------------------------------
# Seeds (plan 4.4)
# --------------------------------------------------------------------------------------------


def train_seeds(run_seed: int, gen: int, k_follow: int, k_approach: int) -> list[tuple[int, str]]:
    """The K common training seeds of generation `gen` (1-based) for run seed `run_seed`.

    Deterministic, distinct across generations and run seeds, and always >= TRAIN_SEED_BASE, so
    they never overlap SELECTION_SEEDS or TEST_SEEDS. Arms that share a run seed see the same
    training episodes. First k_follow are follow episodes, the rest approach.
    """
    k = k_follow + k_approach
    if not 0 < k <= SEEDS_PER_GEN:
        raise ValueError(f"K = {k} must be in 1..{SEEDS_PER_GEN}")
    if gen < 1 or gen * SEEDS_PER_GEN >= SEEDS_PER_RUN_SEED:
        raise ValueError(f"generation {gen} out of range")
    base = TRAIN_SEED_BASE + int(run_seed) * SEEDS_PER_RUN_SEED + int(gen) * SEEDS_PER_GEN
    return [(base + i, "follow" if i < k_follow else "approach") for i in range(k)]


def fixed_plan(seeds: tuple[int, ...], n_follow: int, n_approach: int) -> list[tuple[int, str]]:
    """Fixed kind assignment on a fixed seed set: the first n_follow seeds follow, the next n_approach approach."""
    if n_follow + n_approach > len(seeds):
        raise ValueError(f"{n_follow} + {n_approach} exceeds the {len(seeds)} seeds")
    return [(s, "follow") for s in seeds[:n_follow]] + [(s, "approach") for s in seeds[n_follow : n_follow + n_approach]]


def selection_plan(cfg: dict) -> list[tuple[int, str]]:
    s = cfg["selection"]
    return fixed_plan(SELECTION_SEEDS, int(s["n_follow"]), int(s["n_approach"]))


def test_plan(cfg: dict) -> list[tuple[int, str]]:
    e = cfg["eval"]
    return fixed_plan(TEST_SEEDS, int(e["n_follow"]), int(e["n_approach"]))


# --------------------------------------------------------------------------------------------
# Fitness (plan 4.5)
# --------------------------------------------------------------------------------------------


def normalize_return(
    ret: float | None, pid_ret: float | None, floor: float = 1.0, error_score: float = -20.0, clip: list | tuple | None = None
) -> float:
    """ret / max(|pid_ret|, floor), clipped to `clip` if given. An episode that failed (ret None) scores error_score.

    The clip keeps one catastrophic episode (collision tail charge, landing) from deciding a candidate's
    rank: measured split-half rank reliability at K = 32 rose from 0.80 to 0.91 with clip [-4, 2].
    """
    if ret is None or not math.isfinite(ret):
        return float(error_score)
    denom = floor if pid_ret is None or not math.isfinite(pid_ret) else max(abs(pid_ret), floor)
    v = float(ret) / denom
    if clip is not None:
        v = min(max(v, float(clip[0])), float(clip[1]))
    return v


def fitness(scores: list[float], kinds: list[str], w_follow: float = 0.5, w_approach: float = 0.5) -> float:
    """w_follow * mean(follow scores) + w_approach * mean(approach scores); a kind with no episodes
    is dropped and the weights renormalized. Higher is better."""
    parts, weights = [], []
    for kind, w in (("follow", w_follow), ("approach", w_approach)):
        v = [s for s, k in zip(scores, kinds) if k == kind]
        if v and w > 0:
            parts.append(float(np.mean(v)))
            weights.append(float(w))
    if not parts:
        return float("nan")
    return float(np.dot(parts, weights) / np.sum(weights))


# --------------------------------------------------------------------------------------------
# Chunking and cost
# --------------------------------------------------------------------------------------------


def chunk_size(cfg: dict, arm: str | None = None) -> int:
    """Items per evaluate_chunk call: chunking.by_arm[arm] if set, else chunking.items_per_chunk."""
    c = cfg["chunking"]
    return int((c.get("by_arm") or {}).get(arm) or c["items_per_chunk"])


def make_chunks(items: list, size: int) -> list[list]:
    """Consecutive chunks of at most `size` items, order preserved (flattening gives back items)."""
    size = max(1, int(size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def container_usd_per_s(cpu: float, memory_mib: float, prices: dict) -> float:
    """Modal list price per second of one container with this request (billed max(request, usage))."""
    return (float(cpu) * prices["cpu_core_hour_usd"] + float(memory_mib) / 1024.0 * prices["mem_gib_hour_usd"]) / 3600.0


def eval_container_spec(cfg: dict) -> dict:
    m = cfg["modal"]["eval"]
    workers = int(m["workers_per_container"])
    return {"cpu": float(m["cpu"]), "memory_mib": workers * int(m["memory_mib_per_worker"]), "workers": workers}


def planned_runs(cfg: dict, gens: int | None = None) -> list[dict]:
    """The run list in launch order: seed 1 of every arm, then seed 2 of every arm, and so on."""
    r = cfg["runs"]
    gens = int(gens or cfg["cma"]["generations"])
    later = r.get("gens_later_seeds")
    out = []
    for i, seed in enumerate(r["seeds"]):
        for arm in r["arms"]:
            out.append({"arm": arm, "seed": int(seed), "gens": int(later) if (later and i > 0) else gens})
    return out


def episodes_per_gen(cfg: dict) -> float:
    """Episodes one run evaluates per generation, selection and PID denominators amortized."""
    e, s = cfg["episodes"], cfg["selection"]
    k = int(e["k_follow"]) + int(e["k_approach"])
    n = int(cfg["cma"]["popsize"]) * k + k
    every = int(s.get("every") or 0)
    if every > 0:
        n += (int(s["n_follow"]) + int(s["n_approach"])) / every
    return float(n)


def project(cfg: dict, measured: dict | None = None, runs: list[dict] | None = None) -> dict:
    """Project wall time and cost of the planned runs on Modal.

    measured: {arm: {"container_s_per_gen": s, "chunk_wall_s": s, "n_items": n}} from smoke tests.
    An arm without its own smoke uses the smoke of projection.same_cost_as[arm] (same brain, for
    example FLY-YAW uses FLY-CMA's), else projection.episode_wall_s from the config. Wall time is the larger of
    (a) the slowest run doing one chunk wave per generation and (b) all container-seconds spread
    over max_containers (queueing when the runs together need more containers than the cap).
    """
    measured = measured or {}
    runs = runs or planned_runs(cfg)
    spec = eval_container_spec(cfg)
    prices = cfg["prices"]
    over = float(prices.get("billing_overhead_factor", 1.0))
    usd_s = container_usd_per_s(spec["cpu"], spec["memory_mib"], prices)
    pj = cfg.get("projection") or {}
    pack = float(pj.get("packing_efficiency", 0.8))
    same = pj.get("same_cost_as") or DEFAULT_SAME_COST_AS
    per_arm = {}
    for arm in {r["arm"] for r in runs}:
        m, src = measured.get(arm), "smoke"
        if not m and measured.get(same.get(arm, "")):
            m, src = measured[same[arm]], f"smoke of {same[arm]}"
        if m:
            cs, cw = float(m["container_s_per_gen"]), float(m["chunk_wall_s"])
            if m.get("n_items"):  # the smoke may have used a different K or popsize: scale per episode
                cs *= episodes_per_gen(cfg) / float(m["n_items"])
        else:
            ep = float((pj.get("episode_wall_s") or {}).get(arm, 60.0))
            cs = episodes_per_gen(cfg) * ep / spec["workers"] / pack
            cw = ep / pack
            src = "config guess"
        per_arm[arm] = {"container_s_per_gen": cs, "chunk_wall_s": cw, "source": src}
    total_cs = sum(r["gens"] * per_arm[r["arm"]]["container_s_per_gen"] for r in runs)
    max_c = int(cfg["modal"]["eval"]["max_containers"])
    overhead = float(pj.get("gen_overhead_s", 3.0))  # map submission, result transfer, driver, commit
    wall_run = max(r["gens"] * (per_arm[r["arm"]]["chunk_wall_s"] + overhead) for r in runs)
    wall_cap = total_cs / max_c
    wall = max(wall_run, wall_cap)
    t = cfg["modal"]["train"]
    driver_usd = sum(r["gens"] * (per_arm[r["arm"]]["chunk_wall_s"] + overhead) for r in runs) * container_usd_per_s(t["cpu"], t["memory_mib"], prices)
    eval_usd = total_cs * usd_s * over
    return {
        "runs": len(runs),
        "per_arm": per_arm,
        "container_s_total": total_cs,
        "container_h_total": total_cs / 3600.0,
        "eval_usd": eval_usd,
        "driver_usd": driver_usd,
        "total_usd": eval_usd + driver_usd,
        "wall_h": wall / 3600.0,
        "wall_h_one_wave_per_gen": wall_run / 3600.0,
        "wall_h_container_cap": wall_cap / 3600.0,
        "max_containers": max_c,
        "eval_container": spec,
        "usd_per_eval_container_h": usd_s * 3600.0,
    }


def run_budget_usd(cfg: dict, arm: str) -> float | None:
    """Per-run budget: budget.per_run_usd, else total_usd x share[arm] / (number of seeds of the arm)."""
    b = cfg.get("budget") or {}
    if b.get("per_run_usd"):
        return float(b["per_run_usd"])
    total = b.get("total_usd")
    share = (b.get("share") or {}).get(arm)
    if not total or share is None:
        return None
    n = len(cfg["runs"]["seeds"]) if arm in cfg["runs"]["arms"] else 1
    return float(total) * float(share) / max(1, n)


# --------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------


class Backend(Protocol):
    name: str

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        """Return (results in item order, stats with container_s and cost_usd)."""
        ...


class LocalBackend:
    """Process pool on this machine. Cost is what the same episodes would cost packed into Modal eval containers."""

    name = "local"

    def __init__(self, cfg: dict, processes: int | None = None, start_method: str | None = None):
        self.cfg = cfg
        self.processes = int(processes or cfg["local"]["processes"] or os.cpu_count() or 1)
        self.start_method = start_method or cfg["local"].get("start_method")

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        results = ev.evaluate_items(items, processes=self.processes, start_method=self.start_method)
        spec = eval_container_spec(self.cfg)
        ep_s = sum(float(r.get("item_wall_s") or 0.0) for r in results)
        container_s = ep_s / spec["workers"]
        usd = container_s * container_usd_per_s(spec["cpu"], spec["memory_mib"], self.cfg["prices"])
        usd *= float(self.cfg["prices"].get("billing_overhead_factor", 1.0))
        return results, {
            "wall_s": time.perf_counter() - t0,
            "container_s": container_s,
            "cost_usd": usd,
            "n_chunks": 1,
            "cost_basis": "local-estimate",
        }

    def close(self) -> None:
        ev.shutdown_pool()


class ModalBackend:
    """Fan chunks out with `evaluate_chunk.map` (the Modal function from modal_app)."""

    name = "modal"

    def __init__(self, cfg: dict, fn: Any, arm: str | None = None):
        self.cfg = cfg
        self.fn = fn
        self.chunk_size = chunk_size(cfg, arm)

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        t_submit = time.time()
        chunks = make_chunks(items, self.chunk_size)
        outs = list(self.fn.map(chunks, return_exceptions=True, order_outputs=True))
        results: list[dict] = []
        container_s = 0.0
        chunk_walls, queue_s, errors = [], [], 0
        for chunk, out in zip(chunks, outs):
            if isinstance(out, BaseException) or not isinstance(out, dict):
                errors += 1
                results.extend(ev.error_result(it, f"chunk failed: {type(out).__name__}: {out}") for it in chunk)
                continue
            results.extend(out["results"])
            container_s += float(out.get("wall_s", 0.0))
            chunk_walls.append(float(out.get("wall_s", 0.0)))
            if out.get("t_start"):
                queue_s.append(float(out["t_start"]) - t_submit)
        spec = eval_container_spec(self.cfg)
        usd = container_s * container_usd_per_s(spec["cpu"], spec["memory_mib"], self.cfg["prices"])
        usd *= float(self.cfg["prices"].get("billing_overhead_factor", 1.0))
        return results, {
            "wall_s": time.perf_counter() - t0,
            "container_s": container_s,
            "cost_usd": usd,
            "n_chunks": len(chunks),
            "chunk_errors": errors,
            "chunk_wall_s_max": max(chunk_walls) if chunk_walls else None,
            "chunk_wall_s_mean": float(np.mean(chunk_walls)) if chunk_walls else None,
            "queue_s_max": max(queue_s) if queue_s else None,
            "queue_s_min": min(queue_s) if queue_s else None,
            "chunk_meta": [
                {k: o.get(k) for k in ("wall_s", "t_start", "cold", "container_age_s", "workers", "n_items")}
                for o in outs
                if isinstance(o, dict)
            ],
            "cost_basis": "modal-chunk-wall",
        }

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------------------------
# CMA-ES helpers
# --------------------------------------------------------------------------------------------


class SeededRandn:
    """Normal sampler for pycma, reseeded every generation.

    pycma's default `randn` is a bound method of numpy's global RandomState; pickling the ES then
    copies that RandomState, so a resumed run would no longer follow np.random.seed. With this
    object (shared by es.opts and the sampler, and pickled with them) generation g always draws
    the same candidates, before and after a resume.
    """

    def __init__(self, seed: int = 0):
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self.rng.standard_normal(shape)


def cma_stds(arm: str, cma_cfg: dict) -> np.ndarray | None:
    """Per-coordinate step multipliers from cma.stds_by_prefix (first matching name prefix wins).

    The readout weights and biases have bounds about 10x wider than the scale they act on
    (tanh arguments of normalized rates), so the same normalized step that is small for the
    encoder flips the readout. Measured 2026-09-26 on NOBRAIN: a 0.05 step on the yaw weights
    made the median candidate 6x worse than the init; 0.003 made it 1.1x worse.
    """
    by_prefix = cma_cfg.get("stds_by_prefix") or {}
    if not by_prefix:
        return None
    from flyfollow.rl.params import param_space

    names = list(param_space(arm).names)
    return np.array([next((float(m) for p, m in by_prefix.items() if n.startswith(p)), 1.0) for n in names])


def ask_seed(run_seed: int, gen: int) -> list[int]:
    return [int(run_seed), int(gen), 0xC3A]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_bytes(path, (json.dumps(obj, indent=2, default=_json_default) + "\n").encode())


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
        f.flush()


def _truncate_jsonl(path: Path, max_gen: int) -> None:
    """Drop log lines of generations that were logged but never checkpointed (crash in between)."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if int(rec.get("gen", 0)) <= max_gen:
            keep.append(line)
    _atomic_write_bytes(path, ("\n".join(keep) + ("\n" if keep else "")).encode())


def _mean_dicts(dicts: list[dict]) -> dict:
    acc: dict[str, list[float]] = {}
    for d in dicts:
        for k, v in (d or {}).items():
            if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(float(v)) or isinstance(v, bool):
                acc.setdefault(k, []).append(float(v))
    return {k: float(np.mean(v)) for k, v in sorted(acc.items())}


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("trace", "traceback")}


# --------------------------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------------------------

# Config fields that must match to resume a run (anything else may change between restarts).
RESUME_KEYS = (
    ("cma", "popsize"),
    ("cma", "sigma0"),
    ("episodes", "k_follow"),
    ("episodes", "k_approach"),
    ("episodes", "profile"),
    ("episodes", "episode_s"),
    ("fitness", "pid_arm"),
)


class Trainer:
    """One CMA-ES run (arm x run seed), checkpointed every generation in its run directory."""

    def __init__(
        self,
        arm: str,
        run_seed: int,
        cfg: dict,
        backend: Backend,
        *,
        tag: str | None = None,
        brain: str | None = None,
        run_dir: str | Path | None = None,
        x0: np.ndarray | None = None,
        init: dict | None = None,
        driver_usd_per_s: float = 0.0,
        on_checkpoint: Callable[[], None] | None = None,
        budget_usd: float | None = None,
        log: Callable[[str], None] = print,
    ):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
        self.arm = arm
        self.run_seed = int(run_seed)
        self.cfg = cfg
        self.backend = backend
        self.tag = tag or cfg.get("tag") or "v1"
        self.brain = brain if brain is not None else brain_for(arm, run_seed, cfg)
        self.run_dir = Path(run_dir) if run_dir else runs_dir() / run_name(arm, run_seed, self.tag)
        self.x0 = None if x0 is None else np.asarray(x0, dtype=float)
        self.init = init  # from load_init(): fine-tune from another run's best checkpoint
        self.driver_usd_per_s = float(driver_usd_per_s)
        self.on_checkpoint = on_checkpoint
        self.budget_usd = budget_usd if budget_usd is not None else run_budget_usd(cfg, arm)
        self.log = log
        self.overrides = env_overrides(cfg)
        self.state: dict = {}

    # ---- paths
    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.pkl"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "log.jsonl"

    @property
    def sel_path(self) -> Path:
        return self.run_dir / "selection.jsonl"

    @property
    def best_path(self) -> Path:
        return self.run_dir / "best.json"

    @property
    def err_path(self) -> Path:
        return self.run_dir / "errors.jsonl"

    # ---- init / resume
    def _initial_x(self) -> np.ndarray:
        if self.x0 is not None:
            return self.x0
        if self.init is not None:
            return self._init_x_checked()
        from flyfollow.rl.controllers import init_x

        return np.asarray(init_x(self.arm, ev.resolve_brain(self.brain)), dtype=float)

    def _init_x_checked(self) -> np.ndarray:
        """x0 from an init record, refused if it cannot mean the same controller here."""
        from flyfollow.rl.params import param_space

        x = np.asarray(self.init["x"], dtype=float)
        dim = param_space(self.arm).dim
        if x.size != dim:
            raise ValueError(f"init {self.init.get('source')} has {x.size} parameters; {self.arm} has {dim}")
        if self.init.get("brain") not in (None, self.brain):
            raise ValueError(f"init was trained on brain {self.init['brain']!r}; this run uses {self.brain!r}")
        then, now = self.init.get("calibration"), calibration_fingerprint(self.arm, self.brain)
        if then and now and then.get("sha256") != now.get("sha256"):
            raise ValueError(f"calibration {now['file']} changed since the init run: its x would decode to a different controller")
        if self.init.get("arm") not in (None, self.arm):
            self.log(f"[{self.name}] note: init comes from arm {self.init['arm']}")
        return x

    def load_or_init(self) -> bool:
        """Resume from state.pkl if present (returns True), else start a new run."""
        import cma

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            with open(self.state_path, "rb") as f:
                st = pickle.load(f)
            self._check_resumable(st)
            self.state = st
            _truncate_jsonl(self.log_path, st["gen"])
            _truncate_jsonl(self.sel_path, st["gen"])
            self.log(f"[{self.name}] resumed at generation {st['gen']} from {self.state_path}")
            return True
        x0 = np.clip(self._initial_x(), 0.0, 1.0)
        c = self.cfg["cma"]
        opts = {
            "popsize": int(c["popsize"]),
            "bounds": list(c["bounds"]),
            "seed": np.nan,  # sampling is controlled by SeededRandn, not numpy's global RNG
            "randn": SeededRandn(0),
            "verbose": -9,
            "tolfun": 0,
            "tolfunhist": 0,
            "tolflatfitness": 10**9,
            "tolx": 0,
            "tolstagnation": 10**9,
            "maxiter": 10**9,
        }
        try:
            stds = cma_stds(self.arm, c)
        except Exception:
            if self.x0 is None:
                raise  # a real run must get its per-coordinate steps
            stds = None  # synthetic test problem with an explicit x0
        if stds is not None and stds.size == x0.size:  # size differs only for synthetic test problems
            opts["CMA_stds"] = stds.tolist()
        if c.get("maxstd"):
            opts["maxstd"] = float(c["maxstd"])
        es = cma.CMAEvolutionStrategy(x0.tolist(), float(c["sigma0"]), opts)
        self.state = {
            "version": STATE_VERSION,
            "arm": self.arm,
            "run_seed": self.run_seed,
            "tag": self.tag,
            "brain": self.brain,
            "dim": int(x0.size),
            "x0": x0.tolist(),
            "cfg": copy.deepcopy(self.cfg),
            "es": es,
            "gen": 0,
            "pid_cache": {},
            "best": None,
            "sel_done": [],
            "cost_usd_total": 0.0,
            "container_s_total": 0.0,
            "wall_s_total": 0.0,
            "episodes_total": 0,
            "errors_total": 0,
            "stopped": None,
            "created": _now(),
            "calibration": calibration_fingerprint(self.arm, self.brain),
            "init_from": self._init_meta(),
        }
        _atomic_write_json(
            self.run_dir / "config.json",
            {
                "arm": self.arm,
                "run_seed": self.run_seed,
                "brain": self.brain,
                "calibration": self.state["calibration"],
                "init_from": self.state["init_from"],
                "env_overrides": self.overrides,
                "cfg": self.cfg,
            },
        )
        self.save()
        self.log(f"[{self.name}] new run, dim {x0.size}, popsize {opts['popsize']}, dir {self.run_dir}")
        return False

    def _init_meta(self) -> dict | None:
        return None if self.init is None else {k: v for k, v in self.init.items() if k != "x"}

    def _check_resumable(self, st: dict) -> None:
        if st.get("version") != STATE_VERSION or st.get("arm") != self.arm or st.get("run_seed") != self.run_seed:
            raise RuntimeError(f"{self.state_path} belongs to a different run ({st.get('arm')}, seed {st.get('run_seed')})")
        if st.get("brain") != self.brain:
            raise RuntimeError(f"{self.state_path} was trained on brain {st.get('brain')!r}, not {self.brain!r}; use --fresh or another --tag")
        for sec, key in RESUME_KEYS:
            old, new = st["cfg"][sec].get(key), self.cfg[sec].get(key)
            if old != new:
                raise RuntimeError(f"cannot resume {self.run_dir}: {sec}.{key} was {old!r}, now {new!r}; use --fresh or another --tag")
        old, new = st["cfg"].get("env_overrides") or {}, self.cfg.get("env_overrides") or {}
        if old != new:
            raise RuntimeError(f"cannot resume {self.run_dir}: env_overrides was {old!r}, now {new!r}; use --fresh or another --tag")
        if self.init is not None and (st.get("init_from") or {}).get("sha256") != self.init.get("sha256"):
            raise RuntimeError(f"cannot resume {self.run_dir}: it was started from {st.get('init_from')!r}, not {self.init.get('source')!r}")

    @property
    def name(self) -> str:
        return run_name(self.arm, self.run_seed, self.tag)

    def save(self) -> None:
        _atomic_write_bytes(self.state_path, pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL))
        if self.on_checkpoint:
            self.on_checkpoint()

    # ---- evaluation helpers
    def _item(self, arm: str, seed: int, kind: str, x: Any, tag: dict, brain: str | None) -> dict:
        e = self.cfg["episodes"]
        return ev.make_item(
            arm,
            seed,
            kind,
            x=x,
            brain=brain,
            profile=e["profile"],
            overrides=self.overrides,
            warmup_s=float(e["warmup_s"]),
            tag=tag,
        )

    def _pid_key(self, seed: int, kind: str) -> str:
        return f"{seed}|{kind}|{self.cfg['episodes']['profile']}|{ev._overrides_key(self.overrides)}"

    def _pid_items(self, plan: list[tuple[int, str]]) -> list[dict]:
        pid_arm = self.cfg["fitness"]["pid_arm"]
        out = []
        for seed, kind in plan:
            if self._pid_key(seed, kind) not in self.state["pid_cache"]:
                out.append(self._item(pid_arm, seed, kind, None, {"role": "pid", "seed": seed}, None))
        return out

    def _store_pid(self, results: list[dict]) -> int:
        n_err = 0
        for r in results:
            if r.get("tag", {}).get("role") != "pid":
                continue
            ok = bool(r.get("ok"))
            n_err += not ok
            # A failed denominator is cached as None (floor 1 is used) so it is not retried forever.
            self.state["pid_cache"][self._pid_key(r["seed"], r["kind"])] = float(r["ret"]) if ok else None
        return n_err

    def _scores(self, results: list[dict]) -> list[float]:
        f = self.cfg["fitness"]
        out = []
        for r in results:
            pid = self.state["pid_cache"].get(self._pid_key(r["seed"], r["kind"]))
            out.append(normalize_return(r.get("ret") if r.get("ok") else None, pid, float(f["denom_floor"]), float(f["error_score"]), f.get("clip")))
        return out

    def _fitness(self, scores: list[float], kinds: list[str]) -> float:
        f = self.cfg["fitness"]
        return fitness(scores, kinds, float(f["w_follow"]), float(f["w_approach"]))

    def _log_errors(self, gen: int, results: list[dict], limit: int = 20) -> list[str]:
        errs = [r for r in results if not r.get("ok")]
        for r in errs[:limit]:
            _append_jsonl(self.err_path, {"gen": gen, **{k: r.get(k) for k in ("arm", "tag", "seed", "kind", "error", "traceback")}})
        return [str(r.get("error"))[:300] for r in errs[:3]]

    def _account(self, stats: dict, wall_s: float) -> float:
        usd = float(stats.get("cost_usd") or 0.0) + wall_s * self.driver_usd_per_s
        self.state["cost_usd_total"] += usd
        self.state["container_s_total"] += float(stats.get("container_s") or 0.0)
        self.state["wall_s_total"] += wall_s
        return usd

    # ---- selection (plan 4.6: distribution mean on SELECTION_SEEDS, keep the best ever seen)
    def mean_x(self) -> np.ndarray:
        es = self.state["es"]
        return np.clip(np.asarray(es.result.xfavorite, dtype=float), 0.0, 1.0)

    def select(self, gen: int, x: np.ndarray | None = None) -> dict:
        t0 = time.perf_counter()
        x = self.mean_x() if x is None else np.asarray(x, dtype=float)
        plan = selection_plan(self.cfg)
        items = self._pid_items(plan) + [
            self._item(self.arm, s, k, x, {"role": "sel", "j": j}, self.brain) for j, (s, k) in enumerate(plan)
        ]
        results, stats = self.backend.evaluate(items)
        self._store_pid(results)
        sel = [r for r in results if r.get("tag", {}).get("role") == "sel"]
        scores = self._scores(sel)
        kinds = [r["kind"] for r in sel]
        fit = self._fitness(scores, kinds)
        wall = time.perf_counter() - t0
        usd = self._account(stats, wall)
        self.state["episodes_total"] += len(results)
        n_err = sum(not r.get("ok") for r in results)
        self.state["errors_total"] += n_err
        err_examples = self._log_errors(gen, results)
        best = self.state["best"]
        improved = best is None or (math.isfinite(fit) and fit > best["sel_fitness"])
        rec = {
            "gen": gen,
            "time": _now(),
            "sel_fitness": fit,
            "follow_mean": _mean_or_none([s for s, k in zip(scores, kinds) if k == "follow"]),
            "approach_mean": _mean_or_none([s for s, k in zip(scores, kinds) if k == "approach"]),
            "n_errors": n_err,
            "error_examples": err_examples,
            "metrics_mean": {k: _mean_dicts([r["metrics"] for r in sel if r["kind"] == k and r.get("ok")]) for k in ("follow", "approach")},
            "terms_mean": _mean_dicts([r["terms"] for r in sel if r.get("ok")]),
            "improved": improved,
            "wall_s": wall,
            "cost_usd": usd,
            "x": x.tolist(),
        }
        if improved:
            self.state["best"] = {"gen": gen, "sel_fitness": fit, "x": x.tolist()}
            self._write_best(rec)
        rec["best_gen"] = self.state["best"]["gen"]
        rec["best_sel_fitness"] = self.state["best"]["sel_fitness"]
        _append_jsonl(self.sel_path, rec)
        self.state["sel_done"].append(gen)
        self.log(f"[{self.name}] selection gen {gen}: fitness {fit:.4f} (best {rec['best_sel_fitness']:.4f} at gen {rec['best_gen']})")
        return rec

    def _write_best(self, sel_rec: dict) -> None:
        params = None
        try:
            from flyfollow.rl.params import param_space

            params = _jsonable_params(param_space(self.arm).decode(np.asarray(sel_rec["x"])))
        except Exception:  # noqa: BLE001
            params = None
        _atomic_write_json(
            self.best_path,
            {
                "arm": self.arm,
                "run_seed": self.run_seed,
                "tag": self.tag,
                "brain": self.brain,
                "gen": sel_rec["gen"],
                "sel_fitness": sel_rec["sel_fitness"],
                "metrics_mean": sel_rec["metrics_mean"],
                "x": sel_rec["x"],
                "params": params,
                "calibration": self.state.get("calibration"),
                "init_from": self.state.get("init_from"),
                "env_overrides": self.overrides,
                "episode_s": self.cfg["episodes"].get("episode_s"),
                "time": sel_rec["time"],
            },
        )

    # ---- one generation
    def step(self) -> dict:
        st = self.state
        es = st["es"]
        gen = st["gen"] + 1
        t0 = time.perf_counter()
        e = self.cfg["episodes"]
        plan = train_seeds(self.run_seed, gen, int(e["k_follow"]), int(e["k_approach"]))
        es.opts["randn"].reseed(ask_seed(self.run_seed, gen))
        X_raw = es.ask()
        X = [np.clip(np.asarray(x, dtype=float), 0.0, 1.0) for x in X_raw]
        items = self._pid_items(plan)
        for i, x in enumerate(X):
            for j, (s, k) in enumerate(plan):
                items.append(self._item(self.arm, s, k, x, {"role": "cand", "i": i, "j": j}, self.brain))
        results, stats = self.backend.evaluate(items)
        n_pid_err = self._store_pid(results)
        cand = [r for r in results if r.get("tag", {}).get("role") == "cand"]
        by_i: dict[int, list[dict]] = {}
        for r in cand:
            by_i.setdefault(int(r["tag"]["i"]), []).append(r)
        fits = []
        for i in range(len(X)):
            rs = sorted(by_i.get(i, []), key=lambda r: r["tag"]["j"])
            fit = self._fitness(self._scores(rs), [r["kind"] for r in rs]) if rs else float("nan")
            fits.append(fit if math.isfinite(fit) else float(self.cfg["fitness"]["error_score"]))
        es.tell(X_raw, [-f for f in fits])
        wall = time.perf_counter() - t0
        usd = self._account(stats, wall)
        n_err = sum(not r.get("ok") for r in results)
        st["episodes_total"] += len(results)
        st["errors_total"] += n_err
        st["gen"] = gen
        ok = [r for r in cand if r.get("ok")]
        f = np.asarray(fits)
        rec = {
            "gen": gen,
            "time": _now(),
            "fit_best": float(f.max()),
            "fit_median": float(np.median(f)),
            "fit_mean": float(f.mean()),
            "fit_worst": float(f.min()),
            "sigma": float(es.sigma),
            "n_items": len(items),
            "n_pid": sum(1 for it in items if it["tag"]["role"] == "pid"),
            "n_errors": n_err,
            "n_pid_errors": n_pid_err,
            "error_examples": self._log_errors(gen, results),
            "terms_mean": _mean_dicts([r["terms"] for r in ok]),
            "metrics_mean": {k: _mean_dicts([r["metrics"] for r in ok if r["kind"] == k]) for k in ("follow", "approach")},
            "wall_s": wall,
            "episode_wall_s_sum": float(sum(float(r.get("wall_s") or 0.0) for r in results)),
            "item_wall_s_sum": float(sum(float(r.get("item_wall_s") or 0.0) for r in results)),
            "brain_s_sum": float(sum(float(r.get("brain_s") or 0.0) for r in results)),
            "container_s": float(stats.get("container_s") or 0.0),
            "cost_usd_gen": usd,
            "cost_usd_total": st["cost_usd_total"],
            "cost_basis": stats.get("cost_basis"),
            "backend": self.backend.name,
            "n_chunks": stats.get("n_chunks"),
            "queue_s_max": stats.get("queue_s_max"),
            "train_seeds": [s for s, _ in plan],
            "stopped": None,
        }
        return rec

    def _selection_due(self, gen: int, last_gen: int) -> bool:
        s = self.cfg["selection"]
        every = int(s.get("every") or 0)
        if gen in self.state["sel_done"]:
            return False
        return (every > 0 and gen % every == 0) or (bool(s.get("at_end")) and gen == last_gen)

    def _over_budget(self, next_gen_usd: float) -> bool:
        if not self.budget_usd:
            return False
        return self.state["cost_usd_total"] + next_gen_usd > self.budget_usd

    def run(self, gens: int, stop_after: int | None = None) -> dict:
        """Run until `gens` generations are done (resuming if a checkpoint exists).

        stop_after: return after this many new generations (tests use it to simulate a crash).
        """
        self.load_or_init()
        st = self.state
        if st.get("stopped") == "budget":
            self.log(f"[{self.name}] stopped earlier on budget; delete state.pkl or raise the budget to continue")
            return self.summary()
        if bool(self.cfg["selection"].get("at_start")) and 0 not in st["sel_done"]:
            self.select(0, np.asarray(st["x0"]))
            self.save()
        done_now = 0
        last_usd = 0.0
        while st["gen"] < gens:
            if stop_after is not None and done_now >= stop_after:
                break
            if self._over_budget(last_usd):
                st["stopped"] = "budget"
                _append_jsonl(self.log_path, {"gen": st["gen"], "time": _now(), "event": "budget stop", "cost_usd_total": st["cost_usd_total"], "budget_usd": self.budget_usd})
                self.save()
                self.log(f"[{self.name}] budget stop at gen {st['gen']}: ${st['cost_usd_total']:.2f} of ${self.budget_usd:.2f}")
                break
            rec = self.step()
            if self._selection_due(rec["gen"], gens):
                sel = self.select(rec["gen"])
                rec["sel_fitness"] = sel["sel_fitness"]
                rec["cost_usd_total"] = st["cost_usd_total"]
                rec["cost_usd_gen"] += sel["cost_usd"]
            if st["best"] is not None:
                rec["best_gen"] = st["best"]["gen"]
                rec["best_sel_fitness"] = st["best"]["sel_fitness"]
            last_usd = rec["cost_usd_gen"]
            _append_jsonl(self.log_path, rec)
            self.save()
            done_now += 1
            self.log(
                f"[{self.name}] gen {rec['gen']}/{gens} fit best {rec['fit_best']:.4f} median {rec['fit_median']:.4f} "
                f"sigma {rec['sigma']:.4f} err {rec['n_errors']} wall {rec['wall_s']:.1f}s cost ${rec['cost_usd_total']:.3f}"
            )
        if st["gen"] >= gens and st.get("stopped") is None:
            st["stopped"] = "done"
            self.save()
        return self.summary()

    def summary(self) -> dict:
        st = self.state
        return {
            "run": self.name,
            "run_dir": str(self.run_dir),
            "gen": st.get("gen"),
            "best": st.get("best"),
            "cost_usd_total": st.get("cost_usd_total"),
            "container_s_total": st.get("container_s_total"),
            "wall_s_total": st.get("wall_s_total"),
            "episodes_total": st.get("episodes_total"),
            "errors_total": st.get("errors_total"),
            "stopped": st.get("stopped"),
        }


def _mean_or_none(v: list[float]) -> float | None:
    return float(np.mean(v)) if v else None


def _jsonable_params(p: Any) -> Any:
    return json.loads(json.dumps(p, default=_json_default))


def calibration_fingerprint(arm: str, brain: str | None) -> dict | None:
    """sha256 of the hand-calibration file the controller reads its init and readout norm from.

    The readout normalization comes from this file at controller build time, so a checkpoint only
    means the same thing while the file is unchanged. Recorded per run and checked by eval.
    """
    import hashlib

    try:
        from flyfollow.rl.controllers import calibration_path, load_calibration

        p = calibration_path(arm, ev.resolve_brain(brain))
        if not p.exists():
            load_calibration(arm, ev.resolve_brain(brain))  # raises if nothing usable exists
            return None
        return {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    except Exception:  # noqa: BLE001
        return None


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def cli_overrides(args: argparse.Namespace) -> dict:
    o: dict = {"cma": {}, "episodes": {}, "local": {}}
    if args.pop is not None:
        o["cma"]["popsize"] = args.pop
    if args.sigma0 is not None:
        o["cma"]["sigma0"] = args.sigma0
    if args.k_follow is not None:
        o["episodes"]["k_follow"] = args.k_follow
    if args.k_approach is not None:
        o["episodes"]["k_approach"] = args.k_approach
    if args.episode_s is not None:
        o["episodes"]["episode_s"] = args.episode_s
    if args.select_every is not None:
        o["selection"] = {"every": args.select_every}
    if args.processes is not None:
        o["local"]["processes"] = args.processes
    if args.budget is not None:
        o["budget"] = {"per_run_usd": args.budget}
    if args.tag is not None:
        o["tag"] = args.tag
    if args.env_override:
        o["env_overrides"] = parse_env_overrides(args.env_override)
    return o


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m flyfollow.rl.cma_train", description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, required=True, help="run seed (1, 2, 3); FLY-SHUF and FLY-SHUF-YAW seed i train on shuffle i")
    p.add_argument("--gens", type=int, default=None, help="total generations (default cma.generations)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--sigma0", type=float, default=None)
    p.add_argument("--k-follow", type=int, default=None)
    p.add_argument("--k-approach", type=int, default=None)
    p.add_argument("--episode-s", type=float, default=None, help="cap episode length (smoke tests only)")
    p.add_argument("--select-every", type=int, default=None)
    p.add_argument("--brain", default=None, help="brain file under data/brains (default from config)")
    p.add_argument("--tag", default=None, help="run dir suffix (default config tag)")
    p.add_argument("--processes", type=int, default=None, help="local backend pool size")
    p.add_argument("--budget", type=float, default=None, help="per-run budget in USD (default from config)")
    p.add_argument("--config", default=None, help="train config (default configs/train.yaml)")
    p.add_argument("--fresh", action="store_true", help="move an existing run dir aside and start over")
    p.add_argument("--init-from", default=None, help="best.json to start from (its normalized x becomes the CMA mean)")
    p.add_argument("--env-override", action="append", default=[], metavar="KEY=VALUE",
                   help="env config override for every episode of the run, dotted key, repeatable (e.g. reward.w_j=26.4)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, cli_overrides(args))
    gens = int(args.gens or cfg["cma"]["generations"])
    tag = cfg["tag"]
    rdir = runs_dir() / run_name(args.arm, args.seed, tag)
    if args.fresh and rdir.exists():
        bak = rdir.with_name(rdir.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.move(str(rdir), str(bak))
        print(f"moved {rdir} to {bak}")
    init = load_init(args.init_from) if args.init_from else None
    if args.backend == "local":
        backend = LocalBackend(cfg)
        trainer = Trainer(args.arm, args.seed, cfg, backend, tag=tag, brain=args.brain, init=init)
        t0 = time.perf_counter()
        try:
            summary = trainer.run(gens)
        finally:
            backend.close()
    else:
        from flyfollow.rl import modal_app

        with modal_app.app.run():
            backend = ModalBackend(cfg, modal_app.evaluate_chunk, args.arm)
            trainer = Trainer(args.arm, args.seed, cfg, backend, tag=tag, brain=args.brain, init=init)
            t0 = time.perf_counter()
            summary = trainer.run(gens)
    summary["driver_wall_s"] = time.perf_counter() - t0
    if summary.get("best"):
        summary["best"] = {k: v for k, v in summary["best"].items() if k != "x"}
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0 if summary.get("errors_total", 0) < max(1, summary.get("episodes_total", 0)) else 1


if __name__ == "__main__":
    # Run through the importable module, not __main__: state.pkl then refers to
    # flyfollow.rl.cma_train.SeededRandn and loads anywhere (Modal train, eval, tests).
    from flyfollow.rl import cma_train as _module  # noqa: PLW0406

    sys.exit(_module.main())
