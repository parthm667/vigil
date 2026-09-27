"""One evaluation job = one episode of one candidate. Used by every backend (serial, local pool, Modal).

A job is a plain dict so it can be pickled to a local process or sent to Modal:
    {"arm", "x" (CMA unit vector) or "params" (dict), "seed", "kind", "cfg", "norm", "shuffle_seed"}
"""

from __future__ import annotations

from pathlib import Path

from ..config import ROOT
from ..pilot.fly import FlyController, NoBrainController
from ..pilot.pid import PID_HAND, PIDController
from ..sim.scenario import sample_scenario
from .env import run_episode
from .params import decode, specs_for_arm

_BRAINS = {}
_PID_RETURNS = {}


def get_brain(cfg: dict, shuffle_seed: int | None = None):
    from ..brain.runner import BrainRunner, load_connectome

    path = cfg["brain"]["path"]
    key = (path, shuffle_seed)
    if key in _BRAINS:
        return _BRAINS[key]
    full = ROOT / path
    if not full.exists():
        full = Path(path)
    if not full.exists():
        raise FileNotFoundError(f"brain file {path} not found; build it with `python -m flyfollow.brain.build`")
    connectome = load_connectome(str(full))
    if shuffle_seed is not None:
        from ..brain.shuffle import degree_preserving_shuffle

        connectome = degree_preserving_shuffle(connectome, shuffle_seed)
    brain = BrainRunner(connectome, lif=cfg["brain"].get("lif"), seed=0)
    _BRAINS[key] = brain
    return brain


def make_controller(arm: str, params: dict, cfg: dict, norm: dict, shuffle_seed: int | None = None):
    dn_types = cfg["brain"]["dn_types"]
    max_back = cfg["governor"]["max_back_stick"]
    if arm == "pid":
        return PIDController(params)
    if arm == "nobrain":
        return NoBrainController(params, dn_types, norm, max_back_stick=max_back,
                                 fly_forward=cfg["brain"].get("fly_forward", "brain"))
    if arm in ("fly", "fly_shuf"):
        brain = get_brain(cfg, shuffle_seed if arm == "fly_shuf" else None)
        return FlyController(params, brain, dn_types, norm, fly_forward=cfg["brain"].get("fly_forward", "brain"),
                             warmup_s=cfg["episode"]["brain_warmup_s"], max_back_stick=max_back)
    raise ValueError(f"unknown arm {arm}")


def pid_baseline_return(cfg: dict, seed: int, kind: str) -> float:
    key = (seed, kind, cfg["episode"]["follow_s"], cfg["episode"]["approach_s"])
    if key not in _PID_RETURNS:
        scenario = sample_scenario(cfg, seed, kind)
        _PID_RETURNS[key] = run_episode(PIDController(PID_HAND), scenario, cfg)["return"]
    return _PID_RETURNS[key]


def score_from(episode_return: float, pid_return: float) -> float:
    """Return normalized by PID-HAND on the same seed (floored and clipped so one easy seed cannot dominate)."""
    ratio = episode_return / max(abs(pid_return), 10.0)
    return max(ratio, -20.0)


def run_job(job: dict) -> dict:
    cfg = job["cfg"]
    arm = job["arm"]
    if "params" in job and job["params"] is not None:
        params = job["params"]
    else:
        params = decode(specs_for_arm(arm, cfg["brain"]["dn_types"]), job["x"])
    scenario = sample_scenario(cfg, job["seed"], job["kind"])
    controller = make_controller(arm, params, cfg, job.get("norm") or {}, job.get("shuffle_seed"))
    result = run_episode(controller, scenario, cfg)
    result["pid_return"] = pid_baseline_return(cfg, job["seed"], job["kind"])
    result["score"] = score_from(result["return"], result["pid_return"])
    result["seed"] = job["seed"]
    result["candidate"] = job.get("candidate")
    return result
