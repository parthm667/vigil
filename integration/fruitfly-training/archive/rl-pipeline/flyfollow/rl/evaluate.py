"""Evaluate trained checkpoints on the held-out test seeds and compare arms (plan section 4.7).

    python -m flyfollow.rl.evaluate --checkpoints ../fruitfly-training-results/checkpoints --workers 6 --n-test 100

Uses `best_params` from each checkpoint (the best mean on the selection seeds) plus the
hand-tuned PID as the reference. Writes runs/eval/<stamp>/metrics.csv and chart.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import ROOT, load_config
from ..pilot.pid import PID_HAND
from .backends import make_backend
from .train import DEFAULT_RESULTS_WORKTREE


def load_arms(checkpoint_root: Path) -> list[dict]:
    arms = [{"label": "pid_hand", "arm": "pid", "params": dict(PID_HAND), "norm": {}}]
    for folder in sorted(checkpoint_root.iterdir()):
        f = folder / "latest.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        params = d.get("best_params") or d.get("mean_params")
        arms.append({"label": folder.name, "arm": d["arm"], "params": params, "norm": d.get("norm", {}),
                     "shuffle_seed": d.get("run_seed") if d["arm"] == "fly_shuf" else None})
        calib = d.get("calibration") or {}
        if d["arm"] in ("fly", "nobrain") and "hand_params" in calib:
            arms.append({"label": folder.name + "_hand", "arm": d["arm"], "params": calib["hand_params"], "norm": d.get("norm", {})})
    return arms


def metrics_for(results: list[dict]) -> dict:
    follow = [r for r in results if r["kind"] == "follow"]
    approach = [r for r in results if r["kind"] == "approach"]
    out = {}
    if follow:
        out["follow_in_band"] = float(np.mean([r["in_band_frac"] for r in follow]))
        out["follow_mean_range_err_m"] = float(np.mean([r["mean_range_err"] for r in follow]))
        out["follow_rms_range_err_m"] = float(np.mean([r["rms_range_err"] for r in follow]))
        out["follow_rms_bearing_deg"] = float(np.mean([r["rms_bearing_deg"] for r in follow]))
        out["follow_loss_per_min"] = float(np.sum([r["loss_events"] for r in follow]) / (np.sum([r["duration"] for r in follow]) / 60.0))
        out["follow_collision_rate"] = float(np.mean([1.0 if r["outcome"] == "collision" else 0.0 for r in follow]))
        out["follow_min_dist_p5_m"] = float(np.percentile([r["min_dist"] for r in follow], 5))
        out["follow_yaw_jerk"] = float(np.mean([r["yaw_jerk_per_s"] for r in follow]))
        out["follow_score"] = float(np.mean([r["score"] for r in follow]))
    if approach:
        out["approach_success"] = float(np.mean([1.0 if r["outcome"] == "success" else 0.0 for r in approach]))
        times = [r["success_time"] for r in approach if r["success_time"] is not None]
        out["approach_time_s"] = float(np.mean(times)) if times else float("nan")
        out["approach_score"] = float(np.mean([r["score"] for r in approach]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default=str(DEFAULT_RESULTS_WORKTREE / "checkpoints"))
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=100, help="test seeds per kind (follow and approach)")
    ap.add_argument("--backend", default="local", choices=["serial", "local"])
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    lo = cfg["train"]["test_seeds"][0]
    arms = load_arms(Path(args.checkpoints))
    backend = make_backend(args.backend, workers=args.workers)
    out_dir = ROOT / "runs" / "eval" / datetime.now().strftime("%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for entry in arms:
        jobs = []
        for kind in ("follow", "approach"):
            for i in range(args.n_test):
                jobs.append({"arm": entry["arm"], "params": entry["params"], "seed": lo + i, "kind": kind, "cfg": cfg,
                             "norm": entry["norm"], "shuffle_seed": entry.get("shuffle_seed")})
        results = backend.map(jobs)
        m = metrics_for(results)
        m["label"] = entry["label"]
        rows.append(m)
        print(entry["label"], {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}, flush=True)
    backend.close()

    keys = ["label"] + sorted({k for r in rows for k in r if k != "label"})
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    try:
        chart(rows, out_dir / "chart.png")
    except ImportError:
        pass
    print(f"wrote {out_dir}")


def chart(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    panels = [("follow_in_band", "Follow: time in band"), ("follow_collision_rate", "Follow: collision rate"),
              ("approach_success", "Approach: success rate"), ("follow_rms_bearing_deg", "Follow: RMS bearing error (deg)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.5))
    for ax, (key, title) in zip(axes, panels):
        values = [r.get(key, float("nan")) for r in rows]
        ax.bar(range(len(labels)), values, color="#4a78a8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
