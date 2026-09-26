# Written against Modal 1.5.5 (latest on PyPI, September 2026). Modal APIs used here were checked against:
#   https://modal.com/docs/guide/images              debian_slim(python_version), pip_install, run_commands,
#                                                    add_local_dir(copy=True) before later build steps, add_local_python_source
#   https://modal.com/docs/sdk/py/latest/Image       add_local_dir(local_path, remote_path, *, copy, ignore);
#                                                    add_local_python_source puts packages in /root, which is on PYTHONPATH
#   https://modal.com/docs/sdk/py/latest/App         function(cpu, memory, timeout, retries, max_containers), local_entrypoint
#   https://modal.com/docs/sdk/py/latest/Function    map(*inputs, order_outputs=True, return_exceptions=False)
#   https://modal.com/docs/reference/cli/run         modal run file.py::name; parameter underscores become dashes in flags
#   https://modal.com/docs/guide/apps                multiple local entrypoints, str/int/float/bool parameters
#   https://modal.com/docs/guide/resources           cpu in physical cores, memory in MiB, billed on max(request, usage)
#   https://modal.com/docs/guide/timeouts            timeout in seconds, counted per attempt when retrying
#   https://modal.com/docs/guide/retries             retries=int, each map input retried independently
#   https://modal.com/docs/guide/scale               max_containers; workspace concurrency depends on the plan
#   https://modal.com/pricing                        CPU and memory prices, Starter plan 100 containers, Team 5000
"""CMA-ES training with the episodes fanned out to Modal CPU containers.

The training loop (flyfollow.rl.train.Trainer) runs on the laptop that starts the run, so runs/<run_name>/
checkpoints and --push-every publishing to the rl-results worktree happen locally. Only the episodes
(flyfollow.rl.worker.run_job, one per Modal input) run on Modal.

Setup, once:
    pip install modal
    modal setup

Run from the repo root. The same lines work in PowerShell and in macOS/Linux shells. This file has two
entrypoints, so always name one with ::smoke or ::main. Parameter names become dashed flags
(init_from -> --init-from, push_every -> --push-every).

Smoke test first (1 generation at popsize 4, no publishing). It prints the per-episode time and the projected
wall time and container-hours for the full configs/train_modal.yaml run:
    modal run flyfollow/rl/modal_app.py::smoke --arm fly --init-from ../fruitfly-training-results/checkpoints/fly/latest.json

Full run of the fly arm from the latest checkpoint, publishing every 5 generations:
    modal run flyfollow/rl/modal_app.py::main --arm fly --init-from ../fruitfly-training-results/checkpoints/fly/latest.json --push-every 5

Other examples:
    modal run flyfollow/rl/modal_app.py::main --arm nobrain --seed 2
    modal run flyfollow/rl/modal_app.py::main --arm fly --generations 20 --popsize 16 --run-name fly-modal-short
    modal run flyfollow/rl/modal_app.py::main --arm fly --resume runs/fly-s1-modal-0926-1400 --run-name fly-s1-modal-0926-1400

Keep the laptop awake and online for the whole run. `modal run --detach` does not help, because the training
loop itself is local. Stop cleanly with Ctrl+C or by creating runs/<run_name>/STOP.
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from pathlib import Path

import modal

# `modal run flyfollow/rl/modal_app.py` imports this file as a script, so put the repo root on sys.path to make
# `import flyfollow` work from any directory. In the container this file is /root/modal_app.py and this does nothing.
REPO = Path(__file__).resolve().parent.parent.parent
if (REPO / "flyfollow").is_dir() and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from flyfollow.rl.backends import ModalBackend  # noqa: E402
from flyfollow.rl.train import Trainer, parse_args  # noqa: E402
from flyfollow.rl.worker import run_job  # noqa: E402

MAX_CONTAINERS = 256  # one generation of configs/train_modal.yaml (32 candidates x 8 episodes) at once
CPU = 1.0
MEMORY_MB = 2048
COST_PER_CONTAINER_HOUR = 3600 * (0.0000131 * CPU + 0.00000222 * MEMORY_MB / 1024)  # modal.com/pricing, Sep 2026

# Container layout. add_local_python_source("flyfollow") puts the package at /root/flyfollow, and
# flyfollow.config.ROOT is the parent of the package, so ROOT is /root in the container. configs/ and data/brains/
# therefore go to /root/configs and /root/data/brains, the same layout relative to ROOT as in the repo.
# FlyDrones is copied into the image (copy=True, needed before the pip install build step) at /opt so its source
# tree is not on PYTHONPATH next to the installed package.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.3",
        "scipy==1.18.1",
        "pyyaml==6.0.3",
        "cma==4.5.0",
        "pandas==3.0.6",
        "pyarrow==25.0.1",
        "matplotlib==3.11.2",
    )
    .add_local_dir(
        REPO / "third_party" / "FlyDrones",
        "/opt/FlyDrones",
        copy=True,
        ignore=["**/__pycache__", "**/*.egg-info", "build", ".git"],
    )
    .run_commands("python -m pip install /opt/FlyDrones")
    .add_local_python_source("flyfollow")
    .add_local_dir(REPO / "configs", "/root/configs")
    .add_local_dir(REPO / "data" / "brains", "/root/data/brains")
)

app = modal.App("flyfollow-train", image=image)


@app.function(cpu=CPU, memory=MEMORY_MB, timeout=15 * 60, retries=2, max_containers=MAX_CONTAINERS)
def evaluate(job: dict) -> dict:
    start = time.perf_counter()
    result = run_job(job)
    result["wall_s"] = time.perf_counter() - start
    return result


class TimedBackend(ModalBackend):
    """ModalBackend that also records how long each batch and each episode took (used by smoke)."""

    def __init__(self, evaluate_fn):
        super().__init__(evaluate_fn)
        self.batches = []
        self.episode_s = []

    def map(self, jobs: list[dict]) -> list[dict]:
        start = time.time()
        results = super().map(jobs)
        wall = time.time() - start

        slowest = 0.0
        for r in results:
            self.episode_s.append(r["wall_s"])
            slowest = max(slowest, r["wall_s"])
        self.batches.append((len(jobs), wall, slowest))
        return results


def build_argv(arm: str, seed: int, config: str, init_from: str, push_every: int, generations: int, popsize: int,
               episodes: int, results_worktree: str, run_name: str, resume: str, label: str = "") -> list[str]:
    argv = ["--arm", arm, "--seed", str(seed), "--config", config, "--backend", "modal", "--push-every", str(push_every)]
    if init_from != "":
        argv.extend(["--init-from", init_from])
    if generations > 0:
        argv.extend(["--generations", str(generations)])
    if popsize > 0:
        argv.extend(["--popsize", str(popsize)])
    if episodes > 0:
        argv.extend(["--episodes", str(episodes)])
    if results_worktree != "":
        argv.extend(["--results-worktree", results_worktree])
    if run_name != "":
        argv.extend(["--run-name", run_name])
    if resume != "":
        argv.extend(["--resume", resume])
    if label != "":
        argv.extend(["--label", label])
    return argv


@app.local_entrypoint()
def main(arm: str, seed: int = 1, config: str = "configs/train_modal.yaml", init_from: str = "", push_every: int = 5,
         generations: int = 0, popsize: int = 0, episodes: int = 0, results_worktree: str = "", run_name: str = "",
         resume: str = "", label: str = ""):
    """Full training run. generations, popsize and episodes of 0 mean the config value."""
    argv = build_argv(arm, seed, config, init_from, push_every, generations, popsize, episodes, results_worktree,
                      run_name, resume, label)
    print("train args: " + " ".join(argv), flush=True)
    Trainer(parse_args(argv), backend=ModalBackend(evaluate)).run()


@app.local_entrypoint()
def smoke(arm: str, seed: int = 1, config: str = "configs/train_modal.yaml", init_from: str = "", popsize: int = 4,
          episodes: int = 0, workspace_containers: int = 100):
    """One generation at a small popsize, then a projection for the full configured run.

    workspace_containers is how many containers the Modal workspace may run at once (Starter 100, Team 5000).
    """
    run_name = f"smoke-{arm}-{datetime.now().strftime('%m%d-%H%M%S')}"
    argv = build_argv(arm, seed, config, init_from, 0, 1, popsize, episodes, "", run_name, "")
    print("train args: " + " ".join(argv), flush=True)
    backend = TimedBackend(evaluate)
    trainer = Trainer(parse_args(argv), backend=backend)
    trainer.run()
    print_projection(trainer.cfg, config, backend, workspace_containers)


def print_projection(cfg: dict, config: str, backend: TimedBackend, workspace_containers: int) -> None:
    if len(backend.episode_s) == 0:
        print("smoke: no episodes finished, nothing to project")
        return

    # what the smoke generation measured
    mean_ep = sum(backend.episode_s) / len(backend.episode_s)
    slowest = max(backend.episode_s)
    overhead = None
    for n_jobs, wall, batch_slowest in backend.batches:
        print(f"smoke batch: {n_jobs} episodes in {wall:.1f} s wall (slowest episode {batch_slowest:.1f} s)")
        if overhead is None or wall - batch_slowest < overhead:
            overhead = max(wall - batch_slowest, 0.0)
    print(f"smoke: {len(backend.episode_s)} episodes, mean {mean_ep:.1f} s, slowest {slowest:.1f} s of container time each")

    # size of the full run in the config
    tcfg = cfg["train"]
    popsize = tcfg["popsize"]
    generations = tcfg["generations"]
    k = tcfg["episodes_per_candidate"]
    lo, hi = tcfg["selection_seeds"]
    n_sel = hi - lo
    n_selections = 0
    for g in range(1, generations + 1):
        if g % tcfg["selection_every"] == 0 or g == 1:
            n_selections += 1
    total_episodes = generations * popsize * k + n_selections * n_sel

    # every wave of containers waits for its slowest episode; each map call also pays the measured overhead
    containers = min(MAX_CONTAINERS, workspace_containers)
    train_waves = math.ceil(popsize * k / containers)
    sel_waves = math.ceil(n_sel / containers)
    wall_s = generations * (train_waves * slowest + overhead) + n_selections * (sel_waves * slowest + overhead)
    busy_hours = total_episodes * mean_ep / 3600
    alive_hours = min(popsize * k, containers) * wall_s / 3600

    print(f"full run ({config}): {generations} generations x {popsize} candidates x {k} episodes"
          f" + {n_selections} selections x {n_sel} = {total_episodes} episodes")
    print(f"projected wall time: {wall_s / 3600:.1f} h with up to {containers} containers at once"
          f" ({train_waves} wave(s) per generation, {overhead:.1f} s overhead per batch)")
    print(f"container-hours: {busy_hours:.1f} busy, up to {alive_hours:.1f} if containers stay up between waves"
          f" (about ${busy_hours * COST_PER_CONTAINER_HOUR:.0f} to ${alive_hours * COST_PER_CONTAINER_HOUR:.0f}"
          f" at {CPU:g} core + {MEMORY_MB} MiB)")
