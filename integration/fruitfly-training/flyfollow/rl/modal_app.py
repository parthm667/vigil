"""Modal app for CMA-ES training (plan 4.6, items 1b, 7, 8, 9, 10).

Run with the module path so the container imports the same module names as locally:

    .venv/bin/modal run -m flyfollow.rl.modal_app::upload                 # brains + init JSON to the Volume
    .venv/bin/modal run -m flyfollow.rl.modal_app::probe --arms PID-HAND,NOBRAIN --episode-s 10
    .venv/bin/modal run -m flyfollow.rl.modal_app::smoke --arm NOBRAIN    # one full-size generation
    .venv/bin/modal run -m flyfollow.rl.modal_app::launch                 # prints the projection only
    .venv/bin/modal run --detach -m flyfollow.rl.modal_app::launch --confirm
    .venv/bin/modal run -m flyfollow.rl.modal_app::status
    .venv/bin/modal run -m flyfollow.rl.modal_app::fetch

Layout: Volume `flyfollow-data` at /data with brains/, brains/init/ and runs/. `train` is the
CMA-ES driver (flyfollow.rl.cma_train.Trainer) running in a small container; it fans each
generation out with `evaluate_chunk.map`. `orchestrate` spawns every `train` call in launch order
and re-spawns any that dies; `launch` triggers only `orchestrate`, because a detached
`modal run` keeps only the last triggered function alive. Everything is preemptible: a restarted
`train` reloads the Volume and resumes from its last pickled generation, and a restarted
`orchestrate` re-attaches to the train calls it recorded on the Volume instead of spawning twins.
"""

from __future__ import annotations

import builtins
import datetime as _dt
import json
import math
import os
import time
from pathlib import Path

import modal

from flyfollow.interfaces import REPO_ROOT, TRAINED_ARMS, data_root
from flyfollow.rl import cma_train as ct

CFG = ct.load_config()
M = CFG["modal"]
EV = M["eval"]
TR = M["train"]
OR = M["orchestrator"]
APP_NAME = M["app_name"]
DATA_MOUNT = "/data"
REMOTE_CONFIGS = "/root/configs"
LAUNCH_DIR = "runs/_launch"
LEASE_TTL_S = 45 * 60

THREAD_ENV = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
IGNORE = ["**/__pycache__", "**/*.pyc", "**/.DS_Store", "**/*.egg-info", "**/.pytest_cache"]
# Pinned to the local venv so Modal and the laptop compute the same numbers.
PACKAGES = [
    "numpy==2.5.3",
    "scipy==1.18.1",
    "pandas==3.0.6",
    "pyarrow==25.0.1",
    "pyyaml==6.0.3",
    "cma==4.5.0",
    "matplotlib==3.11.2",
    "pillow==12.3.0",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*PACKAGES)
    .env({"FLYFOLLOW_DATA": DATA_MOUNT, "FLYFOLLOW_CONFIGS": REMOTE_CONFIGS, "PYTHONPATH": "/root", **THREAD_ENV})
    .add_local_dir(REPO_ROOT / "flyfollow", "/root/flyfollow", ignore=IGNORE)
    .add_local_dir(REPO_ROOT / "third_party" / "FlyDrones" / "src" / "flydrones", "/root/flydrones", ignore=IGNORE)
    .add_local_dir(REPO_ROOT / "configs", REMOTE_CONFIGS, ignore=IGNORE)
    # Lets `modal run flyfollow/rl/modal_app.py::...` (file mode, module "modal_app") work too.
    .add_local_file(Path(__file__), "/root/modal_app.py")
)

# include_source=False: the packages above are added explicitly, so the automatic source mount
# would only duplicate /root/flyfollow.
app = modal.App(APP_NAME, include_source=False)
volume = modal.Volume.from_name(M["volume"], create_if_missing=True)

_EVAL_SPEC = ct.eval_container_spec(CFG)
_T_IMPORT = time.time()
_CHUNKS_SERVED = 0


def _retries(n: int) -> modal.Retries:
    return modal.Retries(max_retries=int(n), backoff_coefficient=2.0, initial_delay=2.0, max_delay=60.0)


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    cpu=_EVAL_SPEC["cpu"],
    memory=_EVAL_SPEC["memory_mib"],
    max_containers=int(EV["max_containers"]),
    scaledown_window=int(EV["scaledown_window_s"]),
    timeout=int(EV["timeout_s"]),
    retries=_retries(EV["retries"]),
)
def evaluate_chunk(items: list[dict]) -> dict:
    """Evaluate a chunk of episode items with a process pool of workers_per_container workers."""
    global _CHUNKS_SERVED
    from flyfollow.rl import evaluate as ev

    cold = _CHUNKS_SERVED == 0
    age = time.time() - _T_IMPORT
    out = ev.evaluate_chunk_local(items, workers=_EVAL_SPEC["workers"], start_method=EV.get("start_method"))
    _CHUNKS_SERVED += 1
    out.update(cold=cold, container_age_s=age, container_id=os.environ.get("MODAL_TASK_ID"))
    return out


def _driver_usd_per_s() -> float:
    return ct.container_usd_per_s(TR["cpu"], TR["memory_mib"], CFG["prices"])


def _check_lease(run_dir: Path, call_id: str | None) -> None:
    """Refuse to run a second live driver on the same run directory (for example a double launch)."""
    lease = run_dir / "lease.json"
    if not lease.exists() or call_id is None:
        return
    try:
        d = json.loads(lease.read_text())
    except (OSError, json.JSONDecodeError):
        return
    if d.get("call_id") not in (None, call_id) and time.time() - float(d.get("heartbeat", 0)) < LEASE_TTL_S:
        raise RuntimeError(f"{run_dir.name} is being trained by call {d['call_id']} (heartbeat {time.time() - d['heartbeat']:.0f} s ago)")


def _write_lease(run_dir: Path, call_id: str | None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "lease.json").write_text(json.dumps({"call_id": call_id, "heartbeat": time.time()}))


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    cpu=float(TR["cpu"]),
    memory=int(TR["memory_mib"]),
    timeout=int(TR["timeout_s"]),
    retries=_retries(TR["retries"]),
)
def train(
    arm: str,
    seed: int,
    gens: int | None = None,
    tag: str | None = None,
    overrides: dict | None = None,
    brain: str | None = None,
    init: dict | None = None,
) -> dict:
    """One CMA-ES run on Modal. Idempotent: resumes from the Volume after preemption or a retry.

    overrides: deep-merged into configs/train.yaml (cma, episodes, selection, budget, env_overrides).
    init: a cma_train.load_init() record (x and provenance), passed by value from the launching laptop.
    """
    volume.reload()
    cfg = ct.load_config(overrides=overrides)
    tag = tag or cfg["tag"]
    gens = int(gens or cfg["cma"]["generations"])
    call_id = modal.current_function_call_id()
    run_dir = ct.runs_dir() / ct.run_name(arm, seed, tag)
    _check_lease(run_dir, call_id)
    _write_lease(run_dir, call_id)
    volume.commit()
    every = max(1, int(cfg["modal"].get("commit_every", 1)))
    n_ckpt = {"n": 0}

    def on_checkpoint() -> None:
        n_ckpt["n"] += 1
        if n_ckpt["n"] % every == 0:
            _write_lease(run_dir, call_id)
            volume.commit()

    backend = ct.ModalBackend(cfg, evaluate_chunk, arm)
    trainer = ct.Trainer(
        arm, seed, cfg, backend, tag=tag, brain=brain, init=init, driver_usd_per_s=_driver_usd_per_s(), on_checkpoint=on_checkpoint
    )
    summary = trainer.run(gens)
    (run_dir / "lease.json").unlink(missing_ok=True)
    volume.commit()
    return summary


@app.function(
    image=image,
    volumes={DATA_MOUNT: volume},
    cpu=float(OR["cpu"]),
    memory=int(OR["memory_mib"]),
    timeout=int(OR["timeout_s"]),
    retries=_retries(2),
)
def orchestrate(launch_id: str, runs: list[dict], tag: str | None = None, overrides: dict | None = None) -> list[dict]:
    """Spawn train for every run in order, watch them, and re-spawn any call that failed.

    The spawned call ids are stored in /data/runs/_launch/<launch_id>.json, so a preempted
    orchestrator re-attaches to them instead of launching duplicates.
    """
    volume.reload()
    rec_path = Path(DATA_MOUNT) / LAUNCH_DIR / f"{launch_id}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"launch_id": launch_id, "runs": {}}

    def save() -> None:
        rec_path.write_text(json.dumps(rec, indent=2))
        volume.commit()

    calls: dict[str, modal.FunctionCall] = {}
    for r in runs:
        name = ct.run_name(r["arm"], r["seed"], tag or CFG["tag"])
        entry = rec["runs"].get(name)
        if entry and entry.get("status") == "running":
            calls[name] = modal.FunctionCall.from_id(entry["call_id"])
            print(f"re-attached {name} -> {entry['call_id']}")
            continue
        if entry and entry.get("status") in ("done", "failed"):
            continue
        fc = train.spawn(r["arm"], int(r["seed"]), int(r["gens"]), tag, overrides, None, r.get("init"))
        calls[name] = fc
        rec["runs"][name] = {"call_id": fc.object_id, "status": "running", "restarts": 0, "run": r, "spawned": time.time()}
        print(f"spawned {name} -> {fc.object_id}")
        save()
    while calls:
        time.sleep(30)
        for name in list(calls):
            entry = rec["runs"][name]
            try:
                result = calls[name].get(timeout=0)
            except builtins.TimeoutError:
                continue  # still running
            except Exception as e:  # noqa: BLE001
                entry["last_error"] = f"{type(e).__name__}: {e}"[:500]
                if entry["restarts"] >= int(OR["max_restarts_per_run"]):
                    entry["status"] = "failed"
                    del calls[name]
                    print(f"{name} failed for good: {entry['last_error']}")
                else:
                    entry["restarts"] += 1
                    r = entry["run"]
                    fc = train.spawn(r["arm"], int(r["seed"]), int(r["gens"]), tag, overrides, None, r.get("init"))
                    calls[name] = fc
                    entry["call_id"] = fc.object_id
                    print(f"{name} died ({entry['last_error']}); re-spawned as {fc.object_id}")
                save()
                continue
            entry["status"] = "done"
            entry["summary"] = result
            del calls[name]
            print(f"{name} done: {json.dumps(result)[:300]}")
            save()
    return list(rec["runs"].values())


# --------------------------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------------------------


@app.local_entrypoint()
def upload(include_full: bool = False) -> None:
    """Push data/brains/*.npz and data/brains/init/*.json to the Volume (brains/, brains/init/)."""
    bdir = REPO_ROOT / "data" / "brains"
    files = sorted(p for p in bdir.glob("*.npz") if include_full or p.name != "malecns_full.npz")
    files += sorted((bdir / "init").glob("*.json"))
    if not files:
        print(f"nothing to upload in {bdir}")
        return
    with volume.batch_upload(force=True) as batch:
        for p in files:
            remote = "/brains/" + str(p.relative_to(bdir)).replace(os.sep, "/")
            batch.put_file(p, remote)
            print(f"{p.relative_to(REPO_ROOT)} -> {M['volume']}:{remote} ({p.stat().st_size / 1e6:.2f} MB)")
    print(f"uploaded {len(files)} files")


@app.local_entrypoint()
def probe(arms: str = "PID-HAND,NOBRAIN", episode_s: float = 10.0, n: int = 2, chunks: int = 2, brain: str = "") -> None:
    """Cheap end-to-end check: n short episodes per arm, sent as `chunks` evaluate_chunk.map inputs
    through ModalBackend (the trainer's code path). Prints image/cold-start and per-episode timings."""
    from flyfollow.rl import evaluate as ev

    items = []
    for arm in [a.strip() for a in arms.split(",") if a.strip()]:
        b = brain or ct.brain_for(arm, 1, CFG)
        for i in range(n):
            kind = "follow" if i % 2 == 0 else "approach"
            items.append(ev.make_item(arm, 10_000 + i, kind, brain=b, overrides={"episode_s": episode_s} if episode_s else None, tag={"i": i}))
    cfg = ct.load_config(overrides={"chunking": {"items_per_chunk": max(1, math.ceil(len(items) / max(1, chunks)))}})
    t0 = time.time()
    res, stats = ct.ModalBackend(cfg, evaluate_chunk).evaluate(items)
    print(f"map round trip {time.time() - t0:.1f} s over {stats['n_chunks']} chunks, container-s {stats['container_s']:.2f}, "
          f"est. cost ${stats['cost_usd']:.5f} (+ startup and scaledown idle)")
    for c in stats["chunk_meta"]:
        print(f"  chunk: submit to start {c['t_start'] - t0:.1f} s, cold {c['cold']}, container age at start {c['container_age_s']:.1f} s, "
              f"wall {c['wall_s']:.2f} s, {c['n_items']} items on {c['workers']} workers")
    for r in res:
        print(f"  {r['arm']:9s} seed {r['seed']} {r['kind']:8s} ok={r['ok']} ret={r.get('ret')} ticks={r.get('n_ticks')} "
              f"episode {r.get('wall_s', 0):.3f} s, item {r.get('item_wall_s', 0):.3f} s, brain {r.get('brain_s', 0):.3f} s, pid {r.get('pid')} "
              f"err={r.get('error')}")


def _smoke_file(arm: str) -> Path:
    return data_root() / "runs" / "smoke" / f"{arm}.json"


def _load_measured() -> dict:
    out = {}
    d = data_root() / "runs" / "smoke"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            m = json.loads(p.read_text())
            out[m["arm"]] = {"container_s_per_gen": m["container_s_per_gen"], "chunk_wall_s": m["chunk_wall_s_max"], "n_items": m.get("n_items")}
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return out


def _fmt_projection(p: dict) -> str:
    lines = [
        f"runs: {p['runs']}, eval container {p['eval_container']} at ${p['usd_per_eval_container_h']:.3f}/h, max_containers {p['max_containers']}",
        f"container-hours: {p['container_h_total']:.1f}; cost: eval ${p['eval_usd']:.2f} + drivers ${p['driver_usd']:.2f} = ${p['total_usd']:.2f}",
        f"wall: {p['wall_h']:.2f} h (one chunk wave per generation: {p['wall_h_one_wave_per_gen']:.2f} h; "
        f"container cap: {p['wall_h_container_cap']:.2f} h)",
    ]
    for arm, a in sorted(p["per_arm"].items()):
        lines.append(f"  {arm:13s} container-s/gen {a['container_s_per_gen']:.0f}, chunk wall {a['chunk_wall_s']:.1f} s ({a['source']})")
    return "\n".join(lines)


@app.local_entrypoint()
def smoke(arm: str = "NOBRAIN", seed: int = 1, pop: int = 0, k_follow: int = -1, k_approach: int = -1, episode_s: float = 0.0, write_log: bool = True) -> None:
    """One full-size generation for an arm on Modal, driven from this laptop. Prints and records
    per-episode wall time, container startup, measured container-seconds and cost, and the
    projection for all planned runs."""
    import numpy as np

    o: dict = {"cma": {}, "episodes": {}, "selection": {"at_start": False, "at_end": False, "every": 0}, "budget": {"per_run_usd": 1e9}}
    if pop > 0:
        o["cma"]["popsize"] = pop
    if k_follow >= 0:
        o["episodes"]["k_follow"] = k_follow
    if k_approach >= 0:
        o["episodes"]["k_approach"] = k_approach
    if episode_s > 0:
        o["episodes"]["episode_s"] = episode_s
    cfg = ct.load_config(overrides=o)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = data_root() / "runs" / "smoke" / f"{arm}_s{seed}_{stamp}"
    captured: dict = {}

    class _Capture(ct.ModalBackend):
        def evaluate(self, items):
            res, stats = super().evaluate(items)
            captured["results"], captured["stats"] = res, stats
            return res, stats

    backend = _Capture(cfg, evaluate_chunk, arm)
    trainer = ct.Trainer(arm, seed, cfg, backend, tag=f"smoke{stamp}", run_dir=run_dir)
    trainer.load_or_init()
    t0 = time.time()
    rec = trainer.step()
    gen_wall = time.time() - t0
    trainer.save()
    res, stats = captured["results"], captured["stats"]
    ok = [r for r in res if r.get("ok")]
    walls = {k: [float(r["wall_s"]) for r in ok if r["kind"] == k] for k in ("follow", "approach")}
    cold_q = [c["t_start"] - t0 for c in stats["chunk_meta"] if c.get("cold") and c.get("t_start")]
    ages = [c["container_age_s"] for c in stats["chunk_meta"] if c.get("cold")]
    eval_usd = stats["container_s"] * ct.container_usd_per_s(_EVAL_SPEC["cpu"], _EVAL_SPEC["memory_mib"], cfg["prices"])
    m = {
        "arm": arm,
        "time": ct._now(),
        "popsize": int(cfg["cma"]["popsize"]),
        "k": int(cfg["episodes"]["k_follow"]) + int(cfg["episodes"]["k_approach"]),
        "episode_s": cfg["episodes"].get("episode_s"),
        "n_items": len(res),
        "n_errors": len(res) - len(ok),
        "error_examples": rec["error_examples"],
        "gen_wall_s": gen_wall,
        "n_chunks": stats["n_chunks"],
        "items_per_chunk": cfg["chunking"]["items_per_chunk"],
        "episode_wall_s": {k: _stats(v) for k, v in walls.items()},
        "brain_s_mean": float(np.mean([r["brain_s"] for r in ok])) if ok else None,
        "cold_chunks": len(cold_q),
        "startup_s": _stats(cold_q),
        "container_age_at_first_chunk_s": _stats(ages),
        "chunk_wall_s_mean": stats["chunk_wall_s_mean"],
        "chunk_wall_s_max": stats["chunk_wall_s_max"],
        "container_s_per_gen": stats["container_s"],
        "eval_usd_per_gen_list_price": eval_usd,
        "eval_usd_per_gen_with_overhead": eval_usd * float(cfg["prices"]["billing_overhead_factor"]),
        "fit": {k: rec[k] for k in ("fit_best", "fit_median", "fit_mean")},
    }
    _smoke_file(arm).parent.mkdir(parents=True, exist_ok=True)
    _smoke_file(arm).write_text(json.dumps(m, indent=2))
    measured = _load_measured()
    proj = ct.project(CFG, measured)
    m["projection"] = proj
    print(json.dumps({k: v for k, v in m.items() if k != "projection"}, indent=2))
    print("projection for all planned runs:\n" + _fmt_projection(proj))
    if write_log:
        _append_training_log(m, proj)


def _stats(v: list[float]) -> dict | None:
    import numpy as np

    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def _append_training_log(m: dict, proj: dict) -> None:
    path = REPO_ROOT / "docs" / "training_log.md"
    ew = m["episode_wall_s"]
    lines = [
        "",
        f"## Modal smoke test: {m['arm']} ({_dt.datetime.now().strftime('%Y-%m-%d %H:%M')})",  # noqa: DTZ005
        "",
        f"- One generation: population {m['popsize']}, K = {m['k']}, {m['n_items']} episodes incl. PID-HAND denominators, "
        f"{m['n_chunks']} chunks of {m['items_per_chunk']}, episode_s override {m['episode_s']}; {m['n_errors']} errors.",
        f"- Generation wall time {m['gen_wall_s']:.1f} s; chunk wall mean {_f(m['chunk_wall_s_mean'])} s, max {_f(m['chunk_wall_s_max'])} s.",
        f"- Episode wall time (s): follow {_fs(ew.get('follow'))}; approach {_fs(ew.get('approach'))}; brain time mean {_f(m['brain_s_mean'])} s.",
        f"- Container startup (submit to first chunk start, cold containers): {_fs(m['startup_s'])}.",
        f"- Container-seconds per generation {m['container_s_per_gen']:.0f}; cost per generation ${m['eval_usd_per_gen_list_price']:.3f} at list price, "
        f"${m['eval_usd_per_gen_with_overhead']:.3f} with the {CFG['prices']['billing_overhead_factor']}x startup/idle allowance.",
        "- Projection for all planned runs:",
        "",
        "```",
        _fmt_projection(proj),
        "```",
    ]
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("# Training log\n")
        f.write("\n".join(lines) + "\n")
    print(f"appended to {path}")


def _f(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _fs(s: dict | None) -> str:
    if not s:
        return "n/a"
    return f"mean {s['mean']:.2f}, median {s['median']:.2f}, p90 {s['p90']:.2f}, max {s['max']:.2f} (n={s['n']})"


@app.local_entrypoint()
def launch(
    confirm: bool = False,
    arms: str = "",
    seeds: str = "",
    gens: int = 0,
    tag: str = "",
    deployed: bool = False,
    wait: bool = True,
    init_from: str = "",
    env_override: str = "",
    sigma0: float = 0.0,
    pop: int = 0,
    k_follow: int = -1,
    k_approach: int = -1,
    select_every: int = -1,
    budget: float = 0.0,
) -> None:
    """Spawn the planned runs (seed 1 of every arm first) through `orchestrate`.

    Prints the projected cost and does nothing else unless --confirm is given. Use with
    `modal run --detach` so the runs survive this laptop, or deploy first and pass --deployed.

    Fine-tune options (sent to every train call as config overrides): --init-from a best.json
    (read here, passed by value), --env-override "reward.w_j=26.4[,key=value]", --sigma0, --pop,
    --k-follow, --k-approach, --select-every, --budget (per run, USD).
    """
    o: dict = {}
    if sigma0 > 0:
        o.setdefault("cma", {})["sigma0"] = sigma0
    if pop > 0:
        o.setdefault("cma", {})["popsize"] = pop
    if k_follow >= 0:
        o.setdefault("episodes", {})["k_follow"] = k_follow
    if k_approach >= 0:
        o.setdefault("episodes", {})["k_approach"] = k_approach
    if select_every >= 0:
        o["selection"] = {"every": select_every}
    if budget > 0:
        o["budget"] = {"per_run_usd": budget}
    if env_override:
        o["env_overrides"] = ct.parse_env_overrides(env_override)
    if tag:
        o["tag"] = tag
    overrides = o or None
    base = ct.load_config(overrides=overrides)  # exactly what each train driver loads (budget, K, env overrides)
    cfg = ct.load_config(overrides=overrides)
    if arms:
        cfg["runs"]["arms"] = [a.strip() for a in arms.split(",") if a.strip()]
        bad = [a for a in cfg["runs"]["arms"] if a not in TRAINED_ARMS]
        if bad:
            raise SystemExit(f"not trainable arms: {bad}; choose from {TRAINED_ARMS}")
    if seeds:
        cfg["runs"]["seeds"] = [int(s) for s in seeds.split(",") if s.strip()]
    runs = ct.planned_runs(cfg, gens or None)
    init = ct.load_init(init_from) if init_from else None
    if init is not None:
        from flyfollow.rl.params import param_space

        for r in runs:
            dim = param_space(r["arm"]).dim
            if dim != len(init["x"]):
                raise SystemExit(f"--init-from has {len(init['x'])} parameters but {r['arm']} has {dim}")
            r["init"] = init
        print(f"init from {init['source']} ({init['arm']} s{init['run_seed']} {init['tag']}, gen {init['gen']}, "
              f"selection fitness {init['sel_fitness']}, sha256 {init['sha256'][:12]})")
    if overrides:
        print(f"config overrides for every run: {json.dumps(overrides)}")
    proj = ct.project(cfg, _load_measured(), runs)
    print("planned runs (launch order):")
    for r in runs:
        brain = ct.brain_for(r["arm"], r["seed"], cfg) or "-"
        print(f"  {r['arm']:13s} seed {r['seed']} gens {r['gens']} brain {brain:26s} budget ${ct.run_budget_usd(base, r['arm']) or math.inf:.0f}")
    print("projection:\n" + _fmt_projection(proj))
    print(f"budget total ${cfg['budget']['total_usd']}; drivers stop cleanly when their share is spent")
    containers = int(EV["max_containers"]) + len(runs) + 1
    print(f"peak containers: {EV['max_containers']} eval + {len(runs)} train + 1 orchestrator = {containers}")
    if not confirm:
        print("\nnot launching: re-run with --confirm (and `modal run --detach`) to spawn the runs")
        return
    launch_id = time.strftime("launch-%Y%m%d-%H%M%S")
    fn = modal.Function.from_name(APP_NAME, "orchestrate") if deployed else orchestrate
    call = fn.spawn(launch_id, runs, tag or None, overrides)
    print(f"launched {launch_id}: orchestrate call {call.object_id}")
    _append_launch_log(launch_id, call.object_id, runs, proj)
    if wait:
        print("waiting for all runs (Ctrl-C is safe under --detach; check progress with ::status)")
        for r in call.get():
            print(json.dumps(r, default=str)[:400])


def _append_launch_log(launch_id: str, call_id: str, runs: list[dict], proj: dict) -> None:
    path = REPO_ROOT / "docs" / "training_log.md"
    lines = [
        "",
        f"## Launch {launch_id} ({_dt.datetime.now().strftime('%Y-%m-%d %H:%M')})",  # noqa: DTZ005
        "",
        f"- orchestrate call `{call_id}` (train call ids are in the Volume at `{LAUNCH_DIR}/{launch_id}.json`)",
        f"- runs: {', '.join(f'{r['arm']} s{r['seed']} ({r['gens']} gens)' for r in runs)}",
        f"- projected ${proj['total_usd']:.0f}, {proj['wall_h']:.1f} h",
    ]
    with open(path, "a", encoding="utf-8") as f:
        if f.tell() == 0:
            f.write("# Training log\n")
        f.write("\n".join(lines) + "\n")


def _read_volume_text(path: str) -> str | None:
    try:
        return b"".join(volume.read_file(path)).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


@app.local_entrypoint()
def status(tag: str = "") -> None:
    """Table of every run on the Volume (last generation, fitness, selection, errors, cost), then
    Modal's own billing summary for this month."""
    try:
        entries = volume.listdir("/runs")
    except Exception as e:  # noqa: BLE001
        print(f"no runs on the Volume yet ({e})")
        entries = []
    rows = []
    for e in sorted(entries, key=lambda e: e.path):
        name = e.path.rstrip("/").split("/")[-1]
        if name.startswith("_") or name == "smoke" or (tag and not name.endswith("_" + tag)):
            continue
        text = _read_volume_text(f"/runs/{name}/log.jsonl")
        recs = [json.loads(line) for line in (text or "").splitlines() if line.strip()]
        gens = [r for r in recs if "fit_best" in r]
        last = gens[-1] if gens else {}
        stop = next((r for r in reversed(recs) if r.get("event")), {})
        rows.append(
            [
                name,
                str(last.get("gen", 0)),
                _f(last.get("fit_best")),
                _f(last.get("fit_mean")),
                _f(last.get("sigma")),
                f"{_f(last.get('best_sel_fitness'))}@{last.get('best_gen', '-')}",
                str(sum(r.get("n_errors", 0) for r in gens)),
                f"{last.get('cost_usd_total', 0.0):.2f}",
                f"{last.get('wall_s', 0.0):.0f}",
                (last.get("time") or "-")[11:19] + (f" {stop['event']}" if stop else ""),
            ]
        )
    head = ["run", "gen", "fit_best", "fit_mean", "sigma", "best_sel@gen", "errors", "est_usd", "gen_s", "last (UTC)"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
    print("  ".join(h.ljust(w) for h, w in zip(head, widths)))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    est = sum(float(r[7]) for r in rows)
    print(f"\nour estimate across listed runs: ${est:.2f}")
    try:
        s = modal.Workspace.from_context().billing.summary()
        print(f"Modal billing summary this cycle: metered ${float(s.metered_cost):.2f}, billed ${float(s.billed_cost):.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"Modal billing summary unavailable: {type(e).__name__}: {e}")


@app.local_entrypoint()
def fetch(run: str = "", include_state: bool = True) -> None:
    """Download /runs from the Volume into local data/runs/ (optionally one run dir)."""
    dest = REPO_ROOT / "data" / "runs"
    root = f"/runs/{run}" if run else "/runs"
    n = 0
    for e in volume.iterdir(root, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        rel = e.path.lstrip("/")
        rel = rel.removeprefix("runs/")
        if not include_state and rel.endswith("state.pkl"):
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.writelines(volume.read_file(e.path))
        n += 1
    print(f"fetched {n} files into {dest}")
