"""Obstacle scenarios: does the runtime avoidance layer conflict with the learned steering?

Runs follow episodes for each arm on the same seeds under three conditions:
    train    the normal train profile (no obstacles), same random world, for reference
    avoid    profile "obstacles" with the avoidance layer on (brake + sidestep; yaw untouched)
    noavoid  profile "obstacles" with the avoidance layer off (the controller alone)
and prints one table row per (arm, condition). Local multiprocessing via flyfollow.rl.evaluate.

    .venv/bin/python scripts/eval_obstacles.py                       # hand calibrations
    .venv/bin/python scripts/eval_obstacles.py --params FLY-YAW=data/runs/FLY-YAW_s1_v1/best.json \\
        --params NOBRAIN-YAW=data/runs/NOBRAIN-YAW_s1_v1/best.json --n 64

--params ARM=path may repeat (also for one arm with several run seeds); arms without --params use
their hand-calibration init. Results (per-episode metrics too) go to --out as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing the package

from flyfollow.rl import evaluate as ev  # noqa: E402  (also puts the vendored flydrones on sys.path)

CONDITIONS = {
    "train": ("train", None),
    "avoid": ("obstacles", None),
    "noavoid": ("obstacles", {"obstacles": {"avoid": False}}),
}
COLUMNS = (  # (metric, header, format)
    ("frac_in_view", "in_view", "{:.3f}"),
    ("frac_in_band", "in_band", "{:.3f}"),
    ("rms_bearing_err_deg", "brg_rms", "{:.1f}"),
    ("loss_events_per_min", "loss/min", "{:.2f}"),
    ("collided", "coll", "{:.3f}"),
    ("obstacle_collision", "obs_coll", "{:.3f}"),
    ("min_obstacle_dist", "min_obs", "{:.2f}"),
    ("blocked_s", "blocked_s", "{:.2f}"),
    ("sidesteps", "sidesteps", "{:.2f}"),
    ("frac_occluded", "occluded", "{:.3f}"),
    ("landed", "landed", "{:.3f}"),
)


def entries_from_args(arms: list[str], params: list[str]) -> list[dict]:
    given: dict[str, list[str]] = {}
    for p in params:
        arm, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"--params expects ARM=path, got {p!r}")
        given.setdefault(arm.strip(), []).append(path.strip())
    out = []
    for arm in arms + [a for a in given if a not in arms]:
        if arm not in given:
            out.append({"label": f"{arm} (hand)", "arm": arm, "x": None, "brain": None})
            continue
        for path in given[arm]:
            b = json.loads(Path(path).read_text())
            if b.get("arm") and b["arm"] != arm:
                print(f"WARNING {path}: best.json is for {b['arm']}, used as {arm}")
            out.append({"label": f"{arm}@{Path(path).parent.name} g{b.get('gen')}", "arm": arm, "x": b["x"], "brain": b.get("brain")})
    return out


def _mean(vals: list[float]) -> float:
    v = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    return sum(v) / len(v) if v else math.nan


def summarize(results: list[dict]) -> dict:
    rows: dict[tuple[str, str], dict] = {}
    for r in results:
        label, cond = r["tag"]
        row = rows.setdefault((label, cond), {"n": 0, "errors": 0, "metrics": [], "ret": []})
        if not r.get("ok"):
            row["errors"] += 1
            print(f"ERROR {label} {cond} seed {r.get('seed')}: {r.get('error')}")
            continue
        row["n"] += 1
        row["metrics"].append(r["metrics"])
        row["ret"].append(r["ret"])
    out = {}
    for key, row in rows.items():
        agg = {m: _mean([x.get(m) for x in row["metrics"]]) for m, _, _ in COLUMNS}
        agg.update(n=row["n"], errors=row["errors"], ret=_mean(row["ret"]))
        out[key] = agg
    return out


def print_table(summary: dict, labels: list[str]) -> None:
    head = f"{'arm':34s} {'cond':8s} {'n':>4s} " + " ".join(f"{h:>9s}" for _, h, _ in COLUMNS) + f" {'ret':>8s}"
    print(head)
    print("-" * len(head))
    for label in labels:
        for cond in CONDITIONS:
            a = summary.get((label, cond))
            if a is None:
                continue
            cells = " ".join(f"{(fmt.format(a[m]) if math.isfinite(a[m]) else '-'):>9s}" for m, _, fmt in COLUMNS)
            print(f"{label:34s} {cond:8s} {a['n']:>4d} {cells} {a['ret']:>8.0f}")
        base, av, no = (summary.get((label, c)) for c in CONDITIONS)
        if base and av and no:
            print(f"{'':34s} {'delta':8s}      in_view avoid-train {av['frac_in_view'] - base['frac_in_view']:+.3f}, "
                  f"noavoid-train {no['frac_in_view'] - base['frac_in_view']:+.3f}; bearing rms avoid-train "
                  f"{av['rms_bearing_err_deg'] - base['rms_bearing_err_deg']:+.1f} deg, noavoid-train "
                  f"{no['rms_bearing_err_deg'] - base['rms_bearing_err_deg']:+.1f} deg")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="FLY-YAW,NOBRAIN-YAW,PID-HAND", help="comma list; hand calibration unless --params")
    ap.add_argument("--params", action="append", default=[], help="ARM=path/to/best.json (repeatable)")
    ap.add_argument("--n", type=int, default=32, help="follow seeds per condition")
    ap.add_argument("--seed-base", type=int, default=5_000, help="seeds seed_base .. seed_base + n - 1 (not training seeds)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None, help="JSON output (default runs/obstacles/eval_<time>.json)")
    a = ap.parse_args(argv)
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c!r}; choose from {list(CONDITIONS)}")
    entries = entries_from_args(arms, a.params)
    items = []
    for e in entries:
        for c in conds:
            profile, overrides = CONDITIONS[c]
            for s in range(a.seed_base, a.seed_base + a.n):
                items.append(ev.make_item(e["arm"], s, "follow", x=e["x"], brain=e["brain"], profile=profile,
                                          overrides=overrides, tag=(e["label"], c)))
    print(f"{len(items)} follow episodes ({len(entries)} entries x {len(conds)} conditions x {a.n} seeds), {a.workers} workers")
    t0 = time.perf_counter()
    results = ev.evaluate_items(items, processes=a.workers)
    ev.shutdown_pool()
    for r in results:
        r["tag"] = tuple(r["tag"])
    print(f"done in {time.perf_counter() - t0:.0f} s\n")
    summary = summarize(results)
    print_table(summary, [e["label"] for e in entries])
    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "runs" / "obstacles" / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "args": vars(a),
        "entries": [{k: v for k, v in e.items() if k != "x"} for e in entries],
        "summary": [{"label": k[0], "condition": k[1], **v} for k, v in summary.items()],
        "episodes": [{"label": r["tag"][0], "condition": r["tag"][1], "seed": r.get("seed"), "ok": r.get("ok"),
                      "ret": r.get("ret"), "metrics": r.get("metrics")} for r in results],
    }, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
