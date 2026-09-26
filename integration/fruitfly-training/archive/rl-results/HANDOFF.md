# Drone RL: training handoff (rl-results branch)

This branch holds only training results. Code lives on `rl-pipeline`. The trainer on Parth's
laptop pushes here every few generations (`--push-every`), so `git pull` on this branch
always gives you the newest checkpoints.

## Live status

<!-- STATUS:BEGIN -->
_Last updated 2026-09-26 16:15 UTC. Written automatically by the trainer (`--push-every`)._

Score = episode return divided by |return of the hand-tuned PID| on the same seed, averaged half follow, half approach. About -1.0 means as good as the hand-tuned PID; higher is better.

| Checkpoint | Run | Backend | Generation | Best selection score | Latest train score | Sigma | State | Updated |
|---|---|---|---|---|---|---|---|---|
| fly | fly-s1-local | local | 400 | -2.268 (gen 280) | -3.366 | 0.0435 | finished | 2026-09-26 15:15 UTC |
| fly_shuf | fly_shuf-s1-local | local | 109 | -1.031 (gen 80) | -3.451 | 0.0916 | finished | 2026-09-26 16:15 UTC |
| fly_shuf_yaw_only | fly_shuf_yaw_only-s1-local | local | 5 | -1.308 (gen 1) | -2.828 | 0.1725 | finished | 2026-09-26 16:14 UTC |
| fly_yaw_only | fly_yaw_only-s1-local | local | 78 | -0.515 (gen 60) | -0.005 | 0.0201 | finished | 2026-09-26 16:14 UTC |
| nobrain | nobrain-s1-local | local | 800 | -0.371 (gen 430) | 0.333 | 0.0068 | finished | 2026-09-26 10:34 UTC |
| nobrain_yaw_only | nobrain_yaw_only-s1-local | local | 400 | -0.489 (gen 50) | -0.324 | 0.0174 | finished | 2026-09-26 15:56 UTC |
| pid | pid-s1-local-v2 | serial | 400 | -0.523 (gen 140) | nan | 0.0552 | finished | 2026-09-26 10:45 UTC |
<!-- STATUS:END -->

## What is in `checkpoints/<arm>/latest.json`

- `mean_unit`: the CMA-ES mean in normalized [0, 1] space. `--init-from` starts a new run here.
- `sigma`, `covariance`: CMA-ES step size and covariance at that generation.
- `best_params`, `best_selection_score`, `best_generation`: the best mean so far on the fixed selection seeds.
- `norm`, `calibration`: the readout normalization from hand calibration (needed to reuse the params).
- `latest_summary`: in-band fraction, collisions and approach success of the last generation.

Arms: `fly` (fixed MaleCNS connectome with a trained encoder and readout), `nobrain` (same encoder and
readout, no brain), `pid` (classical follower with CMA-tuned gains), `fly_shuf` (degree-preserving
shuffled connectome). See `docs/DRONE_RL_PLAN.md` section 4.7 on `rl-pipeline`.

## Start the Modal run from the newest checkpoint (the teammate)

```bash
# code
git clone https://github.com/parthm667/fruitfly-training.git
cd fruitfly-training
git checkout rl-pipeline
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e third_party/FlyDrones

# results, in a sibling folder (the trainer's default --results-worktree)
git worktree add ../fruitfly-training-results rl-results

# Modal (only needed once)
pip install modal
modal setup
```

Then see the "Modal" section of `HANDOFF.md` on `rl-pipeline` for the exact `modal run` command
(it starts from `../fruitfly-training-results/checkpoints/<arm>/latest.json`). When your Modal run
is going, tell Parth so he can stop the local runs (Ctrl+C, or create `runs/<run_name>/STOP`).
Your run will push its own checkpoints here with the same `--push-every` mechanism.
