"""CMA-ES training of one controller arm (plan section 4.6).

Local run (this laptop):
    python -m flyfollow.rl.train --arm fly --config configs/train_local.yaml --backend local --workers 6 --push-every 5

Modal run: see flyfollow/rl/modal_app.py (same loop, Modal backend).

Run state lives in runs/<run_name>/ (gitignored): CMA pickle every generation, log.csv, best.json.
With --push-every N, every N generations checkpoints/<arm>/latest.json and the status block in
HANDOFF.md are committed and pushed to the rl-results branch from a separate worktree.
Stop cleanly with Ctrl+C or by creating runs/<run_name>/STOP; both push one last time.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cma
import numpy as np

from ..config import ROOT, load_config
from .backends import make_backend
from .params import decode, encode, specs_for_arm
from .publish import Publisher

DEFAULT_RESULTS_WORKTREE = ROOT.parent / "fruitfly-training-results"


def episode_plan(cfg: dict, seeds: list[int]) -> list[tuple[int, str]]:
    """Split seeds into follow and approach episodes in the training proportion."""
    k = cfg["train"]["episodes_per_candidate"]
    n_approach = cfg["train"]["approach_episodes"]
    plan = []
    for i, seed in enumerate(seeds):
        if (i % k) >= k - n_approach:
            plan.append((seed, "approach"))
        else:
            plan.append((seed, "follow"))
    return plan


def combine(results: list[dict]) -> float:
    """0.5 x mean follow score + 0.5 x mean approach score (plan section 4.5)."""
    follow = []
    approach = []
    for r in results:
        if r["kind"] == "follow":
            follow.append(r["score"])
        else:
            approach.append(r["score"])
    if follow and approach:
        return 0.5 * float(np.mean(follow)) + 0.5 * float(np.mean(approach))
    if follow:
        return float(np.mean(follow))
    return float(np.mean(approach))


def summarize(results: list[dict]) -> dict:
    follow = [r for r in results if r["kind"] == "follow"]
    approach = [r for r in results if r["kind"] == "approach"]
    out = {}
    if follow:
        out["follow_in_band"] = float(np.mean([r["in_band_frac"] for r in follow]))
        out["follow_collisions"] = int(sum(1 for r in follow if r["outcome"] == "collision"))
        out["follow_rms_range"] = float(np.mean([r["rms_range_err"] for r in follow]))
    if approach:
        out["approach_success"] = float(np.mean([1.0 if r["outcome"] == "success" else 0.0 for r in approach]))
    return out


def center(es) -> np.ndarray:
    """The distribution mean in [0, 1] parameter space.

    With bounds, pycma keeps an internal mean that can sit outside [0, 1] and folds candidates back inside,
    so np.clip(es.mean) is NOT where the candidates are centered. result.xfavorite is.
    """
    return np.clip(np.array(es.result.xfavorite, dtype=float), 0.0, 1.0)


def step_size(es) -> float:
    """Mean per-parameter standard deviation actually used for sampling (sigma times sqrt of the diagonal of C).

    es.sigma alone can drift to huge values while C shrinks (it did with maxstd on), so we publish this instead.
    """
    return float(np.mean(es.stds))


class Trainer:
    def __init__(self, args, backend=None):
        self.args = args
        self.cfg = load_config(args.config)
        if args.follow_s:
            self.cfg["episode"]["follow_s"] = args.follow_s
        if args.episodes:
            share = self.cfg["train"]["approach_episodes"] / self.cfg["train"]["episodes_per_candidate"]
            self.cfg["train"]["episodes_per_candidate"] = args.episodes
            self.cfg["train"]["approach_episodes"] = max(1, round(args.episodes * share))
        self.arm = args.arm
        self.dn_types = self.cfg["brain"]["dn_types"]
        self.specs = specs_for_arm(self.arm, self.dn_types)
        self.run_name = args.run_name or f"{self.arm}-s{args.seed}-{args.backend}-{datetime.now().strftime('%m%d-%H%M')}"
        self.run_dir = ROOT / "runs" / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if args.label:
            self.folder = args.label
        elif args.seed == 1:
            self.folder = self.arm
        else:
            self.folder = f"{self.arm}_s{args.seed}"
        self.backend = backend if backend is not None else make_backend(args.backend, workers=args.workers)
        self.publisher = None
        if args.push_every > 0:
            self.publisher = Publisher(args.results_worktree, log_path=self.run_dir / "publish.log")
        self.norm = {}
        self.calibration = {}
        self.best_score = None
        self.best_gen = None
        self.best_params = None
        self.selection_history = []
        self.generation = 0
        self.latest_train_score = float("nan")
        self.latest_summary = {}
        self.started = time.time()
        self.shuffle_seed = args.seed if self.arm == "fly_shuf" else None

    # ------------------------------------------------------------------ setup
    def log(self, msg: str) -> None:
        line = f"[{self.run_name} {datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.run_dir / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def initial_point(self) -> tuple[np.ndarray, float]:
        args = self.args
        sigma0 = self.cfg["train"]["sigma0"]
        if args.init_from:
            data = json.loads(Path(args.init_from).read_text(encoding="utf-8"))
            if data["arm"] != self.arm:
                raise SystemExit(f"--init-from checkpoint is for arm {data['arm']}, not {self.arm}")
            self.norm = data.get("norm", {})
            self.calibration = data.get("calibration", {})
            x0 = np.array(data["mean_unit"], dtype=float)
            sigma = max(float(data.get("sigma", sigma0)), 0.05)
            self.log(f"initialized from {args.init_from} (gen {data.get('generation')}, sigma {sigma:.3f})")
            return x0, min(sigma, sigma0)
        if self.arm == "pid":
            from ..pilot.pid import PID_HAND

            return encode(self.specs, PID_HAND), sigma0
        from ..pilot.calibrate import hand_init

        brain = None
        if self.arm in ("fly", "fly_shuf"):
            from .worker import get_brain

            brain = get_brain(self.cfg, self.shuffle_seed)
        t0 = time.time()
        params, norm, report = hand_init(self.arm, brain, self.dn_types)
        self.norm = norm
        self.calibration = dict(report)
        self.calibration["hand_params"] = params
        (self.run_dir / "calibration.json").write_text(json.dumps({"params": params, "norm": norm, "report": report}, indent=1))
        self.log(f"hand calibration in {time.time() - t0:.1f}s: yaw R2 {report['yaw_r2']:.3f}, fwd R2 {report['fwd_r2']:.3f}")
        return encode(self.specs, params), sigma0

    # ------------------------------------------------------------------ evaluation
    def jobs_for(self, candidates: list[np.ndarray], plan: list[tuple[int, str]]) -> list[dict]:
        jobs = []
        for c, x in enumerate(candidates):
            for seed, kind in plan:
                jobs.append({"arm": self.arm, "x": [float(v) for v in x], "seed": seed, "kind": kind, "cfg": self.cfg,
                             "norm": self.norm, "shuffle_seed": self.shuffle_seed, "candidate": c})
        return jobs

    def evaluate_candidates(self, candidates: list[np.ndarray], plan: list[tuple[int, str]]) -> tuple[list[float], list[dict]]:
        results = self.backend.map(self.jobs_for(candidates, plan))
        scores = []
        for c in range(len(candidates)):
            mine = [r for r in results if r["candidate"] == c]
            scores.append(combine(mine))
        return scores, results

    def selection_plan(self) -> list[tuple[int, str]]:
        lo, hi = self.cfg["train"]["selection_seeds"]
        return episode_plan(self.cfg, list(range(lo, hi)))

    # ------------------------------------------------------------------ checkpoint
    def checkpoint_data(self, es, finished: bool = False) -> dict:
        mean = center(es)
        try:
            cov = np.array(es.sm.C, dtype=float).round(6).tolist()
        except AttributeError:
            cov = None
        return {
            "arm": self.arm,
            "run_name": self.run_name,
            "run_seed": self.args.seed,
            "backend": self.args.backend,
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "config": str(self.args.config),
            "generation": self.generation,
            "evaluations": int(es.countevals),
            "wall_hours": round((time.time() - self.started) / 3600.0, 3),
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "finished": finished,
            "sigma": step_size(es),
            "cma_sigma_raw": float(es.sigma),
            "latest_train_score": self.latest_train_score,
            "latest_summary": self.latest_summary,
            "best_selection_score": self.best_score,
            "best_generation": self.best_gen,
            "best_params": self.best_params,
            "selection_history": self.selection_history,
            "mean_unit": mean.tolist(),
            "mean_params": decode(self.specs, mean),
            "param_names": [s.name for s in self.specs],
            "norm": self.norm,
            "calibration": self.calibration,
            "dn_types": self.dn_types,
            "brain_path": self.cfg["brain"]["path"],
            "covariance": cov,
        }

    def save_local(self, es) -> None:
        tmp = self.run_dir / "state.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"es": es, "trainer": self.state_dict()}, f)
        tmp.replace(self.run_dir / "state.pkl")
        (self.run_dir / "latest.json").write_text(json.dumps(self.checkpoint_data(es), indent=1))

    def state_dict(self) -> dict:
        return {"norm": self.norm, "calibration": self.calibration, "best_score": self.best_score, "best_gen": self.best_gen,
                "best_params": self.best_params, "selection_history": self.selection_history, "generation": self.generation}

    def publish(self, es, finished: bool = False) -> None:
        if self.publisher is None:
            return
        data = self.checkpoint_data(es, finished=finished)
        best = "n/a" if self.best_score is None else f"{self.best_score:.3f}"
        state = "final" if finished else f"gen {self.generation}"
        self.publisher.publish(self.folder, data, f"results: {self.folder} {state}, best selection {best}")

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        args = self.args
        tcfg = self.cfg["train"]
        if args.resume:
            with open(Path(args.resume) / "state.pkl", "rb") as f:
                saved = pickle.load(f)
            es = saved["es"]
            for key, value in saved["trainer"].items():
                setattr(self, key, value)
            self.log(f"resumed from {args.resume} at generation {self.generation}")
        else:
            x0, sigma0 = self.initial_point()
            popsize = args.popsize or tcfg["popsize"]
            options = {"popsize": popsize, "bounds": [0, 1], "seed": args.seed, "verbose": -9, "maxstd": tcfg.get("maxstd", 0.3)}
            es = cma.CMAEvolutionStrategy(list(x0), sigma0, options)
            (self.run_dir / "config.json").write_text(json.dumps({"args": vars(args), "cfg": self.cfg}, indent=1, default=str))

        generations = args.generations or tcfg["generations"]
        k = tcfg["episodes_per_candidate"]
        log_path = self.run_dir / "log.csv"
        new_log = not log_path.exists()
        stop_file = self.run_dir / "STOP"
        self.log(f"arm {self.arm}, {len(self.specs)} params, popsize {es.popsize}, {k} episodes each, backend {self.backend.name}")
        finished = False
        try:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_log:
                    writer.writerow(["generation", "time", "wall_s", "best_train", "mean_train", "sigma", "selection", "in_band", "collisions", "approach_success"])
                while self.generation < generations:
                    t0 = time.time()
                    candidates = es.ask()
                    base = tcfg["train_seed_base"] + args.seed * 1_000_000 + self.generation * 100
                    plan = episode_plan(self.cfg, list(range(base, base + k)))
                    scores, results = self.evaluate_candidates(candidates, plan)
                    es.tell(candidates, [-s for s in scores])
                    self.generation += 1
                    self.latest_train_score = float(np.mean(scores))
                    self.latest_summary = summarize(results)

                    selection = ""
                    if self.generation % tcfg["selection_every"] == 0 or self.generation == 1:
                        mean = center(es)
                        sel_scores, sel_results = self.evaluate_candidates([mean], self.selection_plan())
                        sel = sel_scores[0]
                        selection = f"{sel:.4f}"
                        self.selection_history.append([self.generation, sel, summarize(sel_results)])
                        if self.best_score is None or sel > self.best_score:
                            self.best_score = sel
                            self.best_gen = self.generation
                            self.best_params = decode(self.specs, mean)
                            (self.run_dir / "best.json").write_text(json.dumps({"generation": self.generation, "selection_score": sel,
                                                                              "params": self.best_params, "norm": self.norm}, indent=1))

                    wall = time.time() - t0
                    s = self.latest_summary
                    writer.writerow([self.generation, datetime.now().strftime("%H:%M:%S"), round(wall, 1), round(max(scores), 4),
                                     round(self.latest_train_score, 4), round(step_size(es), 5), selection,
                                     round(s.get("follow_in_band", float("nan")), 3), s.get("follow_collisions", ""),
                                     round(s.get("approach_success", float("nan")), 3)])
                    f.flush()
                    self.log(f"gen {self.generation}: best {max(scores):.3f} mean {self.latest_train_score:.3f} step {step_size(es):.4f} "
                             f"{'sel ' + selection if selection else ''} ({wall:.1f}s)")
                    self.save_local(es)
                    if args.push_every > 0 and self.generation % args.push_every == 0:
                        self.publish(es)
                    if stop_file.exists():
                        self.log("STOP file found; stopping")
                        break
            finished = self.generation >= generations
        except KeyboardInterrupt:
            self.log("interrupted (Ctrl+C); saving and pushing one last time")
        finally:
            self.save_local(es)
            self.publish(es, finished=True)
            self.backend.close()
            self.log(f"done at generation {self.generation} ({'completed' if finished else 'stopped'})")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train a pursuit controller arm with CMA-ES")
    ap.add_argument("--arm", required=True, choices=["fly", "fly_shuf", "nobrain", "pid"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--config", default="configs/train_local.yaml")
    ap.add_argument("--backend", default="local", choices=["serial", "local", "modal"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run-name")
    ap.add_argument("--resume", help="runs/<run_name> directory to continue exactly (same machine)")
    ap.add_argument("--init-from", help="checkpoints/<arm>/latest.json from rl-results: start a new CMA-ES at its mean and sigma")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--popsize", type=int)
    ap.add_argument("--follow-s", type=float, help="override follow episode length (s)")
    ap.add_argument("--episodes", type=int, help="override episodes per candidate (approach share kept)")
    ap.add_argument("--label", help="checkpoint folder name on rl-results (default: arm, or arm_s<seed>)")
    ap.add_argument("--push-every", type=int, default=0, help="publish to rl-results every N generations (0 = never)")
    ap.add_argument("--results-worktree", default=str(DEFAULT_RESULTS_WORKTREE))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    Trainer(args).run()


if __name__ == "__main__":
    sys.exit(main())
