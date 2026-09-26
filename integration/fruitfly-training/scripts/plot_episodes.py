"""Sanity plots of PID-HAND episodes in PursuitEnv: top-down paths, range with the band, bearing.

    .venv/bin/python scripts/plot_episodes.py [--seeds 10000 10001 10002] [--profile train] [--out runs/sim_check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing the package

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flyfollow.interfaces import REPO_ROOT
from flyfollow.pilot.pid import PIDController
from flyfollow.rl.env import PursuitEnv
from flyfollow.rl.rollout import run_episode


def plot(env: PursuitEnv, seed: int, kind: str, profile: str, out: Path, suffix: str = "") -> Path:
    res = run_episode(env, PIDController(), seed, kind, profile, record=True)
    tr = {k: np.asarray(v) for k, v in res.trace.items()}
    t = tr["t"]
    z_ref = env.z_ref_eval
    band = env.rcfg.band_frac * z_ref
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    a.plot(tr["x"], tr["y"], color="tab:blue", label="drone")
    a.plot(tr["tx"], tr["ty"], color="tab:orange", label="person" if kind == "follow" else "object")
    step = max(1, len(t) // 25)
    for i in range(0, len(t), step):  # heading ticks
        psi = np.radians(tr["psi_deg"][i])
        a.plot([tr["x"][i], tr["x"][i] + 0.3 * np.cos(psi)], [tr["y"][i], tr["y"][i] + 0.3 * np.sin(psi)], color="tab:blue", lw=0.8)
    for o in env.world().get("obstacles", []):  # footprints; darker = reaches flight height
        c, sn = np.cos(o["yaw"]), np.sin(o["yaw"])
        corners = [(sx * o["hx"], sy * o["hy"]) for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]
        px = [o["x"] + c * u - sn * v for u, v in corners]
        py = [o["y"] + sn * u + c * v for u, v in corners]
        tall = o["height"] > float(np.median(tr["alt"])) - 0.3
        a.fill(px, py, color="0.3" if tall else "0.8", alpha=0.8)
        a.text(o["x"], o["y"], f"{o['kind']}\n{o['height']:.1f}", fontsize=6, ha="center", va="center", color="w" if tall else "k")
    a.plot(tr["x"][0], tr["y"][0], "o", color="tab:blue")
    a.plot(tr["tx"][0], tr["ty"][0], "o", color="tab:orange")
    a.set_aspect("equal", "datalim")
    m = res.metrics
    extra = f", obst coll {m['obstacle_collision']:.0f}, sidesteps {m['sidesteps']:.0f}" if "obstacle_collision" in m else ""
    a.set_title(f"top-down, {kind} seed {seed}{extra}", fontsize=9)
    a.legend(loc="best", fontsize=8)
    a = ax[1]
    a.fill_between(t, z_ref - band, z_ref + band, color="tab:green", alpha=0.2, label="band")
    a.axhline(env.st.z_min_m, color="tab:red", ls="--", lw=0.8, label="z_min")
    a.plot(t, tr["z"], color="k", lw=1, label="true range")
    valid = tr["valid"].astype(bool)
    z_est = np.where(valid, env.st.fy * env.st.target_size_m / np.maximum(tr["box_h"], 1e-3), np.nan)
    a.plot(t, z_est, color="tab:purple", lw=0.7, alpha=0.7, label="range est (box)")
    a.set_ylim(0, max(5.0, float(np.nanmax(tr["z"])) + 0.5))
    a.set_xlabel("t (s)")
    a.set_title(f"range: in band {res.metrics['frac_in_band']:.2f}, ret {res.ret:.0f}")
    a.legend(loc="best", fontsize=8)
    a = ax[2]
    a.plot(t, tr["berr_deg"], color="k", lw=1, label="true bearing err (deg)")
    a.plot(t, tr["yaw"], color="tab:blue", lw=0.7, alpha=0.7, label="yaw stick")
    a.plot(t, tr["fb"], color="tab:orange", lw=0.7, alpha=0.7, label="fb stick")
    lost = np.where(~valid, 0.0, np.nan)
    a.plot(t, lost, "r|", ms=8, label="box invalid")
    if "occluded" in tr:
        a.plot(t, np.where(tr["occluded"].astype(bool), -5.0, np.nan), "|", color="0.3", ms=8, label="occluded")
    if "lr" in tr and np.any(tr["lr"] != 0):
        a.plot(t, tr["lr"], color="tab:green", lw=0.8, label="lr (sidestep)")
    a.set_ylim(-70, 70)
    a.set_xlabel("t (s)")
    p = env.params
    a.set_title(f"gain {p['fwd_gain_mps']:.2f} m/s, fwd dead {p['fwd_dead_s']:.2f} s, video lat {p['video_latency_s']:.2f} s, fx {p['fx_px']:.0f}, walk cap {env.v_cap:.2f}", fontsize=8)
    a.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path = out / f"{profile}_{kind}_{seed}{suffix}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[10_000, 10_001, 10_002])
    ap.add_argument("--profile", default="train")
    ap.add_argument("--kinds", nargs="+", default=["follow", "approach"])
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "sim_check"))
    ap.add_argument("--no-avoid", action="store_true", help="obstacles profile with the avoidance layer off")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = None
    if a.no_avoid:
        from flyfollow.rl.env import load_env_config

        cfg = load_env_config()
        cfg["obstacles"]["avoid"] = False
    env = PursuitEnv(cfg)
    for kind in a.kinds:
        for s in a.seeds:
            print(plot(env, s, kind, a.profile, out, "_noavoid" if a.no_avoid else ""))


if __name__ == "__main__":
    main()
