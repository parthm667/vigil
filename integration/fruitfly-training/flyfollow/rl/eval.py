"""Final evaluation of every arm (plan 4.7, item 7b).

    python -m flyfollow.rl.eval --name final                       # local pool, default arms, tag from config
    python -m flyfollow.rl.eval --name final --backend modal       # same, fanned out on Modal
    python -m flyfollow.rl.eval --name quick --n-follow 10 --n-approach 6 --sets test --episode-s 20

Entries: the best checkpoint (best.json) of every trained run `<arm>_s<seed>_<tag>` found under
data_root()/runs (any of interfaces.TRAINED_ARMS), plus the untrained references in eval.references
(FLY-YAW-HAND, FLY-HAND, PID-HAND) from their hand-calibrated init. Each entry is scored
on TEST_SEEDS (fixed kinds: the first n_follow follow, the next n_approach approach) under each
evaluation set: test (train profile), demo and stress. Fitness uses the same normalization as
training (return / |PID-HAND return on the same seed and profile|).

Brain-use checks on eval.lesion_arms (FLY-YAW first, FLY-CMA if trained; plan 4.7): lesion
(rerun each episode with every DN rate clamped to its own episode mean from a first pass; for the
yaw-only fly the bearing error and time in view are the telling metrics, since the PID sets
forward) and the bias audit (|b_yaw|, |b_fwd| against the typical
DN-driven term inside each tanh, from PursuitDecoder.bias_audit()).

Writes data_root()/runs/eval/<name>.json (everything, per-episode records included) and <name>.md
(summary tables). flyfollow.rl.chart turns the JSON into the comparison chart.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from flyfollow.interfaces import ARMS, TEST_SEEDS, TRAINED_ARMS, YAW_ONLY_ARMS, data_root
from flyfollow.rl import cma_train as ct
from flyfollow.rl import evaluate as ev

HAND_ARMS = ("FLY-YAW-HAND", "FLY-HAND", "PID-HAND")  # untrained: evaluated from their hand-calibrated init
ARM_ORDER = (
    "FLY-YAW-HAND",
    "FLY-YAW",
    "FLY-SHUF-YAW",
    "NOBRAIN-YAW",
    "FLY-HAND",
    "FLY-CMA",
    "FLY-SHUF",
    "NOBRAIN",
    "PID-HAND",
    "PID-CMA",
)
# Per-kind metrics reported in the tables (keys of PursuitEnv.metrics()).
FOLLOW_METRICS = (
    "frac_in_band",
    "mean_signed_range_err",
    "rms_range_err",
    "rms_bearing_err_deg",
    "loss_events_per_min",
    "min_dist",
    "safety_interventions_per_min",
    "yaw_jerk",
    "yaw_raw_step_abs",
    "yaw_raw_step_rms",
    "frac_in_view",
    "collided",
)
APPROACH_METRICS = ("success", "time_to_standoff", "overshoot", "min_dist", "collided")
N_BOOT = 2000


def eval_dir() -> Path:
    return data_root() / "runs" / "eval"


# --------------------------------------------------------------------------------------------
# Entries
# --------------------------------------------------------------------------------------------


def discover(cfg: dict, tag: str, arms: list[str]) -> list[dict]:
    """Best checkpoints of trained runs plus the hand-calibrated arms, in ARM_ORDER."""
    out = []
    for arm in [a for a in ARM_ORDER if a in arms]:
        if arm in HAND_ARMS:
            out.append({"name": arm, "arm": arm, "run_seed": None, "x": None, "brain": ct.brain_for(arm, 1, cfg), "source": "init"})
            continue
        for seed in cfg["runs"]["seeds"]:
            d = ct.runs_dir() / ct.run_name(arm, seed, tag)
            best = d / "best.json"
            if not best.exists():
                if d.exists():
                    print(f"skip {d.name}: no best.json yet")
                continue
            b = json.loads(best.read_text())
            now = ct.calibration_fingerprint(arm, b.get("brain"))
            then = b.get("calibration")
            if then and now and then["sha256"] != now["sha256"]:
                print(f"WARNING {d.name}: calibration {now['file']} changed since training; the readout norm differs, results are not comparable")
            out.append(
                {
                    "name": d.name,
                    "arm": arm,
                    "run_seed": seed,
                    "x": b["x"],
                    "brain": b.get("brain"),
                    "source": "best.json",
                    "best_gen": b.get("gen"),
                    "sel_fitness": b.get("sel_fitness"),
                    "params": b.get("params"),
                }
            )
    return out


def entries_for_runs(run_names: list[str]) -> list[dict]:
    """Explicit run directories (for example FLY-YAW_s3_v1,FLY-YAW_s3_smooth), one entry each."""
    out = []
    for name in run_names:
        best = ct.runs_dir() / name / "best.json"
        if not best.exists():
            raise SystemExit(f"{best} not found")
        b = json.loads(best.read_text())
        out.append(
            {
                "name": name,
                "arm": b["arm"],
                "run_seed": b.get("run_seed"),
                "x": b["x"],
                "brain": b.get("brain"),
                "source": "best.json",
                "best_gen": b.get("gen"),
                "sel_fitness": b.get("sel_fitness"),
                "params": b.get("params"),
            }
        )
    return out


def plan_for(n_follow: int, n_approach: int) -> list[tuple[int, str]]:
    return ct.fixed_plan(TEST_SEEDS, n_follow, n_approach)


# --------------------------------------------------------------------------------------------
# Scoring and aggregation
# --------------------------------------------------------------------------------------------


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))


def _mean(v: list[float]) -> float | None:
    v = [float(a) for a in v if _finite(a)]
    return float(np.mean(v)) if v else None


def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float] | None:
    """95 % percentile bootstrap interval of the mean over episodes."""
    v = np.asarray([float(a) for a in values if _finite(a)])
    if v.size < 2:
        return None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(records: list[dict], pid_ret: dict[tuple[int, str], float | None], cfg: dict) -> dict:
    """One entry on one evaluation set: fitness, per-kind metric means, errors, bootstrap CIs."""
    f = cfg["fitness"]
    scores, kinds = [], []
    for r in records:
        scores.append(ct.normalize_return(r["ret"] if r["ok"] else None, pid_ret.get((r["seed"], r["kind"])), float(f["denom_floor"]), float(f["error_score"]), f.get("clip")))
        kinds.append(r["kind"])
    fol = [r for r in records if r["kind"] == "follow" and r["ok"]]
    app = [r for r in records if r["kind"] == "approach" and r["ok"]]
    out = {
        "n": len(records),
        "n_errors": sum(not r["ok"] for r in records),
        "fitness": ct.fitness(scores, kinds, float(f["w_follow"]), float(f["w_approach"])),
        "ret_mean": _mean([r["ret"] for r in records if r["ok"]]),
        "follow": {k: _mean([r["metrics"].get(k) for r in fol]) for k in FOLLOW_METRICS},
        "approach": {k: _mean([r["metrics"].get(k) for r in app]) for k in APPROACH_METRICS},
        "ci95": {
            "follow_frac_in_band": bootstrap_ci([r["metrics"].get("frac_in_band") for r in fol]),
            "approach_success": bootstrap_ci([r["metrics"].get("success") for r in app]),
        },
    }
    floor = [r for r in app if r["metrics"].get("floor_object")]
    out["approach"]["success_floor_objects"] = _mean([r["metrics"].get("success") for r in floor])
    out["approach"]["n_floor_objects"] = len(floor)
    out["approach"]["collision_episodes"] = int(sum(bool(r["metrics"].get("collided")) for r in app))
    out["follow"]["collision_episodes"] = int(sum(bool(r["metrics"].get("collided")) for r in fol))
    return out


def aggregate_arm(summaries: list[dict]) -> dict:
    """Mean across run seeds with the range (min, max) across seeds, for every numeric leaf."""

    def leaves(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(leaves(v, f"{prefix}{k}."))
            elif _finite(v) and not isinstance(v, bool):
                out[prefix + k] = float(v)
        return out

    flat = [leaves({k: s[k] for k in ("fitness", "ret_mean", "follow", "approach", "n_errors")}) for s in summaries]
    keys = sorted(set().union(*flat)) if flat else []
    out = {}
    for k in keys:
        v = [fl[k] for fl in flat if k in fl]
        out[k] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n_seeds": len(v)}
    return out


# --------------------------------------------------------------------------------------------
# Evaluation passes
# --------------------------------------------------------------------------------------------


def make_backend(kind: str, cfg: dict, processes: int | None):
    if kind == "local":
        return ct.LocalBackend(cfg, processes=processes), None
    from flyfollow.rl import modal_app

    ctx = modal_app.app.run()
    ctx.__enter__()
    return ct.ModalBackend(cfg, modal_app.evaluate_chunk), ctx


def evaluate_sets(entries: list[dict], sets: dict[str, str], plan, backend, cfg: dict, overrides: dict | None) -> dict:
    """{set: {entry name: [records]}} for every entry on every set. PID-HAND is always included."""
    items = []
    for set_name, profile in sets.items():
        for e in entries:
            for j, (seed, kind) in enumerate(plan):
                items.append(
                    ev.make_item(e["arm"], seed, kind, x=e["x"], brain=e["brain"], profile=profile, overrides=overrides,
                                 warmup_s=float(cfg["episodes"]["warmup_s"]), stick_stats=True,
                                 tag={"set": set_name, "entry": e["name"], "j": j})
                )
    t0 = time.perf_counter()
    results, stats = backend.evaluate(items)
    print(f"evaluated {len(items)} episodes in {time.perf_counter() - t0:.1f} s ({stats.get('cost_basis')}, est ${stats.get('cost_usd', 0):.3f})")
    out: dict = {s: {e["name"]: [] for e in entries} for s in sets}
    for r in results:
        t = r["tag"]
        out[t["set"]][t["entry"]].append(_record(r))
    for s in out:
        for name in out[s]:
            out[s][name].sort(key=lambda r: r["j"])
    return out


def _record(r: dict) -> dict:
    return {
        "j": r["tag"]["j"],
        "seed": r["seed"],
        "kind": r["kind"],
        "ok": bool(r.get("ok")),
        "ret": r.get("ret"),
        "metrics": r.get("metrics") or {},
        "terms": r.get("terms") or {},
        "wall_s": r.get("wall_s"),
        "error": r.get("error"),
    }


def lesion_and_bias(entries: list[dict], plan, backend, cfg: dict, overrides: dict | None) -> dict:
    """Plan 4.7 brain-use checks on the given FLY entries, on the test set profile.

    Pass 1 records each episode's mean readout rate per DN channel (and the bias audit); pass 2
    reruns the same episode with every channel clamped to that mean.
    """
    if not entries:
        return {}
    profile = cfg["eval"]["sets"].get("test", "train")
    warm = float(cfg["episodes"]["warmup_s"])
    items1 = [
        ev.make_item(e["arm"], s, k, x=e["x"], brain=e["brain"], profile=profile, overrides=overrides, warmup_s=warm, audit=True,
                     tag={"entry": e["name"], "j": j})
        for e in entries
        for j, (s, k) in enumerate(plan)
    ]
    res1, _ = backend.evaluate(items1)
    items2 = []
    for it, r in zip(items1, res1):
        if r.get("ok") and r.get("channel_means"):
            lesioned = dict(it, lesion=r["channel_means"], audit=False)
            items2.append(lesioned)
    res2, _ = backend.evaluate(items2) if items2 else ([], {})
    by2 = {(r["tag"]["entry"], r["tag"]["j"]): r for r in res2}
    out = {}
    for e in entries:
        r1 = [r for r in res1 if r["tag"]["entry"] == e["name"]]
        pairs = [(a, by2.get((e["name"], a["tag"]["j"]))) for a in r1 if a.get("ok")]
        pairs = [(a, b) for a, b in pairs if b is not None and b.get("ok")]
        fol = [(a, b) for a, b in pairs if a["kind"] == "follow"]
        app = [(a, b) for a, b in pairs if a["kind"] == "approach"]
        audits = [a["bias_audit"] for a in r1 if a.get("ok") and a.get("bias_audit")]
        out[e["name"]] = {
            "arm": e["arm"],
            "n_pairs": len(pairs),
            "n_errors": len(r1) - len(pairs),
            "intact": _lesion_metrics([a for a, _ in fol], [a for a, _ in app], [a for a, _ in pairs]),
            "lesioned": _lesion_metrics([b for _, b in fol], [b for _, b in app], [b for _, b in pairs]),
            "bias_audit": {k: _mean([a.get(k) for a in audits]) for k in (audits[0].keys() if audits else [])},
            "biases": {k: (e.get("params") or {}).get(k) for k in ("dec_b_yaw", "dec_b_fwd")},
        }
    return out


def _lesion_metrics(fol: list[dict], app: list[dict], both: list[dict]) -> dict:
    return {
        "follow_frac_in_band": _mean([r["metrics"].get("frac_in_band") for r in fol]),
        "follow_rms_bearing_err_deg": _mean([r["metrics"].get("rms_bearing_err_deg") for r in fol]),
        "follow_frac_in_view": _mean([r["metrics"].get("frac_in_view") for r in fol]),
        "approach_success": _mean([r["metrics"].get("success") for r in app]),
        "ret_mean": _mean([r["ret"] for r in both]),
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _fmt(v, nd: int = 3) -> str:
    return "n/a" if not _finite(v) else f"{float(v):.{nd}f}"


def _fmt_agg(a: dict | None, nd: int = 3) -> str:
    if not a:
        return "n/a"
    if a["n_seeds"] > 1:
        return f"{a['mean']:.{nd}f} [{a['min']:.{nd}f}, {a['max']:.{nd}f}]"
    return f"{a['mean']:.{nd}f}"


TABLE_COLS = (
    ("fitness", "fitness", 3),
    ("follow.frac_in_band", "follow in band", 3),
    ("follow.mean_signed_range_err", "signed range err (m)", 2),
    ("follow.rms_bearing_err_deg", "RMS bearing (deg)", 1),
    ("follow.loss_events_per_min", "losses/min", 2),
    ("follow.min_dist", "follow min dist (m)", 2),
    ("follow.safety_interventions_per_min", "safety/min", 2),
    ("follow.yaw_jerk", "yaw jerk (sent)", 3),
    ("follow.yaw_raw_step_abs", "raw yaw step/loop", 2),
    ("approach.success", "approach success", 3),
    ("approach.time_to_standoff", "time to standoff (s)", 1),
    ("approach.overshoot", "overshoot (m)", 2),
    ("n_errors", "errors", 0),
)


def _fitness_note(f: dict) -> str:
    if not f:
        return "Fitness is the training normalization (return over the PID-HAND return on the same seed)."
    clip = f.get("clip")
    return (
        f"Fitness is each episode's return / max(|PID-HAND return on the same seed and profile|, {float(f['denom_floor']):g})"
        + (f", clipped to [{clip[0]:g}, {clip[1]:g}]" if clip else "")
        + f", weighted {f['w_follow']:g} follow + {f['w_approach']:g} approach, as in training; higher is better."
    )


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        f"# Evaluation: {report['name']}",
        "",
        f"{report['time']}. Tag `{report['tag']}`, {report['n_follow']} follow + {report['n_approach']} approach episodes on TEST_SEEDS "
        f"per set, episode_s override {report['episode_s']}. Trained arms: best checkpoint (by selection fitness) of each run seed; "
        "values are the mean across run seeds with [min, max] across seeds. "
        f"{_fitness_note(report.get('fitness') or {})}",
    ]
    for set_name in report["sets"]:
        lines += ["", f"## Set: {set_name} (profile {report['sets'][set_name]})", ""]
        lines.append("| arm | " + " | ".join(c[1] for c in TABLE_COLS) + " |")
        lines.append("|" + "---|" * (len(TABLE_COLS) + 1))
        for arm, agg in report["arms"][set_name].items():
            lines.append(f"| {arm} | " + " | ".join(_fmt_agg(agg.get(k), nd) for k, _, nd in TABLE_COLS) + " |")
    if report.get("lesion"):
        lines += ["", "## Brain-use checks: lesion and bias audit (test set subset)", "",
                  "| run | RMS bearing (deg) intact | lesioned | in view intact | lesioned | follow in band intact | lesioned | "
                  "approach success intact | lesioned | |b_yaw| / yaw drive | |b_fwd| / fwd drive |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, L in report["lesion"].items():
            b, i, x = L["bias_audit"], L["intact"], L["lesioned"]
            fwd = "n/a (PID forward)" if L["arm"] in YAW_ONLY_ARMS else _fmt(b.get("fwd_bias_ratio"), 2)
            lines.append(
                f"| {name} | {_fmt(i.get('follow_rms_bearing_err_deg'), 1)} | {_fmt(x.get('follow_rms_bearing_err_deg'), 1)} | "
                f"{_fmt(i.get('follow_frac_in_view'))} | {_fmt(x.get('follow_frac_in_view'))} | "
                f"{_fmt(i.get('follow_frac_in_band'))} | {_fmt(x.get('follow_frac_in_band'))} | "
                f"{_fmt(i.get('approach_success'))} | {_fmt(x.get('approach_success'))} | "
                f"{_fmt(b.get('yaw_bias_ratio'), 2)} | {fwd} |"
            )
        lines += ["", "Lesion: every DN readout channel clamped to its own episode mean (from an intact first pass). "
                  "If the brain does the steering, bearing error should rise and time in view and in band collapse. "
                  "Bias ratio: |b| over the mean |DN-driven term| inside the tanh (above about 1, the bias does more than the brain)."]
    lines += ["", "## Entries", "", "| entry | arm | source | best gen | selection fitness |", "|---|---|---|---|---|"]
    for e in report["entries"]:
        lines.append(f"| {e['name']} | {e['arm']} | {e['source']} | {e.get('best_gen', '')} | {_fmt(e.get('sel_fitness'))} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m flyfollow.rl.eval", description="Final evaluation of every arm (plan 4.7).")
    p.add_argument("--name", default="final", help="output name: data/runs/eval/<name>.json and .md")
    p.add_argument("--tag", default=None, help="run tag to evaluate (default config tag)")
    p.add_argument("--arms", default=None, help="comma-separated arms (default: every trained arm plus eval.references)")
    p.add_argument("--sets", default=None, help="comma-separated evaluation sets (default all in config eval.sets)")
    p.add_argument("--n-follow", type=int, default=None)
    p.add_argument("--n-approach", type=int, default=None)
    p.add_argument("--episode-s", type=float, default=None, help="cap episode length (quick checks only)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--runs", default=None, help="comma-separated run dirs to compare row by row (e.g. FLY-YAW_s3_v1,FLY-YAW_s3_smooth)")
    p.add_argument("--no-lesion", action="store_true")
    p.add_argument("--lesion-arms", default=None, help="comma-separated arms for the lesion and bias checks (default eval.lesion_arms)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = ct.load_config()
    tag = args.tag or cfg["tag"]
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    else:
        arms = list(TRAINED_ARMS) + [a for a in cfg["eval"].get("references", HAND_ARMS) if a not in TRAINED_ARMS]
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}")
    if "PID-HAND" not in arms:
        arms.append("PID-HAND")  # the normalization reference
    all_sets = dict(cfg["eval"]["sets"])
    sets = {s: all_sets[s] for s in (args.sets.split(",") if args.sets else all_sets)}
    n_f = int(args.n_follow if args.n_follow is not None else cfg["eval"]["n_follow"])
    n_a = int(args.n_approach if args.n_approach is not None else cfg["eval"]["n_approach"])
    plan = plan_for(n_f, n_a)
    overrides = {"episode_s": float(args.episode_s)} if args.episode_s else None
    if args.runs:
        # Explicit runs: each is its own row (not pooled by arm), plus the references asked for.
        refs = [a for a in arms if a in HAND_ARMS] if args.arms else ["PID-HAND"]
        entries = entries_for_runs([r.strip() for r in args.runs.split(",") if r.strip()]) + discover(cfg, tag, refs)
    else:
        entries = discover(cfg, tag, arms)
    print(f"{len(entries)} entries: {', '.join(e['name'] for e in entries)}")
    backend, ctx = make_backend(args.backend, cfg, args.processes)
    t0 = time.perf_counter()
    try:
        recs = evaluate_sets(entries, sets, plan, backend, cfg, overrides)
        lesion = {}
        if not args.no_lesion:
            l_arms = [a.strip() for a in args.lesion_arms.split(",") if a.strip()] if args.lesion_arms else list(cfg["eval"].get("lesion_arms", []))
            l_entries = [e for e in entries if e["arm"] in l_arms]
            n_l = int(cfg["eval"].get("lesion_n", 40))
            lesion = lesion_and_bias(l_entries, plan[: max(1, min(n_l, len(plan)))] if n_l < len(plan) else plan, backend, cfg, overrides)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
        if hasattr(backend, "close"):
            backend.close()
    report: dict = {
        "name": args.name,
        "time": ct._now(),
        "tag": tag,
        "n_follow": n_f,
        "n_approach": n_a,
        "episode_s": args.episode_s,
        "sets": sets,
        "fitness": cfg["fitness"],
        "entries": [{k: v for k, v in e.items() if k not in ("x", "params")} for e in entries],
        "summaries": {},
        "arms": {},
        "lesion": lesion,
        "records": recs,
        "wall_s": time.perf_counter() - t0,
    }
    for set_name in sets:
        pid = {(r["seed"], r["kind"]): (r["ret"] if r["ok"] else None) for r in recs[set_name].get("PID-HAND", [])}
        report["summaries"][set_name] = {e["name"]: summarize(recs[set_name][e["name"]], pid, cfg) for e in entries}
        report["arms"][set_name] = {}
        if args.runs:  # one row per entry
            for e in entries:
                report["arms"][set_name][e["name"]] = aggregate_arm([report["summaries"][set_name][e["name"]]])
            continue
        for arm in [a for a in ARM_ORDER if a in arms]:
            ss = [report["summaries"][set_name][e["name"]] for e in entries if e["arm"] == arm]
            if ss:
                report["arms"][set_name][arm] = aggregate_arm(ss)
    out = eval_dir()
    out.mkdir(parents=True, exist_ok=True)
    jpath, mpath = out / f"{args.name}.json", out / f"{args.name}.md"
    jpath.write_text(json.dumps(report, default=ct._json_default), encoding="utf-8")
    write_markdown(report, mpath)
    print(f"wrote {jpath} and {mpath} in {report['wall_s']:.1f} s")
    print(mpath.read_text())
    return 0


if __name__ == "__main__":
    from flyfollow.rl import eval as _module  # noqa: PLW0406

    sys.exit(_module.main())
