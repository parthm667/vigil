"""Quick simulator sanity check with the hand-tuned PID follower.

    python -m flyfollow.tools.sanity [n_follow] [n_approach]
"""

from __future__ import annotations

import sys

import numpy as np

from ..config import load_config
from ..pilot.pid import PID_HAND, PIDController
from ..rl.env import run_episode
from ..sim.scenario import sample_scenario


def main():
    n_follow = 60
    n_approach = 40
    if len(sys.argv) > 1:
        n_follow = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_approach = int(sys.argv[2])
    cfg = load_config()
    for kind, n in (("follow", n_follow), ("approach", n_approach)):
        results = []
        for seed in range(n):
            results.append(run_episode(PIDController(PID_HAND), sample_scenario(cfg, 100 + seed, kind), cfg))
        outcomes = {}
        for r in results:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        print(kind, outcomes,
              "in_band", round(float(np.mean([r["in_band_frac"] for r in results])), 3),
              "return", round(float(np.mean([r["return"] for r in results])), 1),
              "rms_range", round(float(np.mean([r["rms_range_err"] for r in results])), 2),
              "loss_events", round(float(np.mean([r["loss_events"] for r in results])), 2),
              "min_dist_p5", round(float(np.percentile([r["min_dist"] for r in results], 5)), 2))
        terms = {}
        for key in results[0]["terms"]:
            terms[key] = round(float(np.mean([r["terms"][key] for r in results])), 1)
        print("   terms", terms, "events", round(float(np.mean([r["events"] for r in results])), 1))


if __name__ == "__main__":
    main()
