# Drone RL handoff (code branch `rl-pipeline`)

Plan: [docs/DRONE_RL_PLAN.md](docs/DRONE_RL_PLAN.md). Live training status and checkpoints: the
`HANDOFF.md` on the **`rl-results`** branch, which the trainer updates automatically.

## Current state (12:15, handed to the teammate; all local runs stopped)

- **Read first: [docs/results/LOCAL_RESULTS_0926.md](docs/results/LOCAL_RESULTS_0926.md)** (overnight results and what to run next).
- **All local runs are stopped** and their final checkpoints are pushed to `rl-results`
  (`git -C ../fruitfly-training-results pull`). Nothing is running on Parth's laptop.
- **G0 connectome audit: full** ([docs/audit_result.md](docs/audit_result.md)). LC10a reaches DNa02 through
  AOTU019/AOTU025 as a clean push-pull; DNa02 R minus L flips sign with bearing. Shuffled wiring loses this.
- **Main finding:** the full fly controller fails on **range** (the audit's weak size signal). With the brain
  only steering and the PID setting forward speed, the **yaw-only fly** is level with the controls.
  Final local best selection scores (16 seeds, -1.0 = hand-tuned PID, higher is better):

  | Checkpoint on rl-results | Generations | Best selection |
  |---|---|---|
  | `fly_yaw_only` | 78 | -0.515 |
  | `nobrain_yaw_only` | 400 | -0.489 |
  | `pid` (CMA-tuned) | 400 | -0.523 |
  | `nobrain` (full) | 800 | -0.371 |
  | `fly_shuf` (full) | 109 | -1.031 |
  | `fly` (full) | 400 | -2.268 |
  | `fly_shuf_yaw_only` | 5 | -1.308 (barely started) |

- **Next (the teammate, Modal):** the smoke test, then the yaw-only fly and its yaw-only controls at full size, extra
  seeds, then `flyfollow.rl.evaluate` with 100 test seeds per kind for the final chart. Commands below.

## Branches

| Branch | What | Who writes it |
|---|---|---|
| `rl-pipeline` | Code, configs, the pursuit brain files, audit results | People (commits by hand) |
| `rl-results` | `checkpoints/<arm>/latest.json` and the status block in its `HANDOFF.md` | The trainer only (`--push-every`), from a separate worktree |

The trainer never commits on `rl-pipeline` and never runs `git add -A`. It stages only the checkpoint
file and `HANDOFF.md` inside `../fruitfly-training-results` (a `git worktree` of `rl-results`).
A failed push (no internet, laptop on the Tello Wi-Fi) is logged to `runs/<run>/publish.log` and retried
at the next push; training keeps going.

## Setup

```bash
git clone https://github.com/parthm667/fruitfly-training.git
cd fruitfly-training
git checkout rl-pipeline
python -m venv .venv
source .venv/bin/activate                      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e third_party/FlyDrones           # vendored FlyDrones at 3e26934 with the MaleCNS side fix
git worktree add ../fruitfly-training-results rl-results
python -m pytest -q                            # fast checks, no drone or Modal needed
```

## What is being trained

One controller that steers toward a box and holds a standoff (FOLLOW the user's head, APPROACH an
object). Arms, all with the same governor, Kalman filter, simulator, seeds and CMA-ES budget:

| Arm | Trainable | Brain |
|---|---|---|
| `fly` | 47 (25 encoder + 22 readout) | Fixed MaleCNS pursuit subgraph (`data/brains/pursuit_core1.npz`), never trained |
| `fly_shuf` | 47 | Same subgraph after a degree-preserving shuffle (control) |
| `nobrain` | 47 | None: encoder rates pooled straight into the readout (control) |
| `pid` | 4 | Classical follower gains (baseline and demo fallback) |

The simulator's drone response comes from Parth's lag test on 2026-09-26
(`data/lag_test`, refit with `python -m flyfollow.tools.analyze_lag data/lag_test/drone_fulllogs_20260926_052637.json`):
yaw dead time 0.17 s and about 55 deg/s at stick 100, forward dead time 0.48 s, time constant 0.44 s and
about 0.96 m/s at stick 100 (only stick 30 was measured, so larger sticks assume linearity).

## Local training (Parth's laptop)

```bash
python -m flyfollow.rl.train --arm fly     --backend local --workers 5 --push-every 5  --run-name fly-s1-local
python -m flyfollow.rl.train --arm nobrain --backend local --workers 2 --push-every 50 --run-name nobrain-s1-local
python -m flyfollow.rl.train --arm pid     --backend local --workers 1 --push-every 50 --run-name pid-s1-local
```

Stop a run with Ctrl+C or by creating `runs/<run_name>/STOP`; either way it saves and pushes once more.
Resume the exact CMA state on the same machine with `--resume runs/<run_name>`.

## Modal (the teammate)

`flyfollow/rl/modal_app.py` fans the episodes out to Modal CPU containers (the fly brain sim is numpy,
so no GPU). The CMA-ES loop itself runs on your laptop, so checkpoints and `--push-every` publishing to
`rl-results` work the same as locally. Keep the laptop awake and online during the run; `--detach` does not
help because the loop is local. Written against Modal 1.5.5 and checked against the docs, but **never run on
Modal yet** (nobody here had credentials), so start with the smoke test.

```bash
pip install modal
modal setup
git -C ../fruitfly-training-results pull          # newest checkpoints from Parth's laptop

# 1. smoke test: one small generation, prints per-episode time and the projected hours and cost
modal run flyfollow/rl/modal_app.py::smoke --arm fly --config configs/train_modal_yaw_only.yaml --init-from ../fruitfly-training-results/checkpoints/fly_yaw_only/latest.json

# 2. main jobs: yaw-only fly and its yaw-only controls (see docs/results/LOCAL_RESULTS_0926.md)
modal run flyfollow/rl/modal_app.py::main --arm fly      --config configs/train_modal_yaw_only.yaml --label fly_yaw_only      --init-from ../fruitfly-training-results/checkpoints/fly_yaw_only/latest.json     --push-every 5
modal run flyfollow/rl/modal_app.py::main --arm nobrain  --config configs/train_modal_yaw_only.yaml --label nobrain_yaw_only  --init-from ../fruitfly-training-results/checkpoints/nobrain_yaw_only/latest.json --push-every 5
modal run flyfollow/rl/modal_app.py::main --arm fly_shuf --config configs/train_modal_yaw_only.yaml --label fly_shuf_yaw_only --init-from ../fruitfly-training-results/checkpoints/fly_shuf_yaw_only/latest.json --push-every 5

# 3. extra seeds for spread (repeat with --seed 3)
modal run flyfollow/rl/modal_app.py::main --arm fly --config configs/train_modal_yaw_only.yaml --label fly_yaw_only_s2 --seed 2 --init-from ../fruitfly-training-results/checkpoints/fly_yaw_only/latest.json --push-every 5

# 4. optional: full fly (brain forward) and PID at full size
modal run flyfollow/rl/modal_app.py::main --arm fly --init-from ../fruitfly-training-results/checkpoints/fly/latest.json --push-every 5
modal run flyfollow/rl/modal_app.py::main --arm pid --init-from ../fruitfly-training-results/checkpoints/pid/latest.json --push-every 5
```

Notes:
- Always name the entrypoint (`::smoke` or `::main`). Flags use dashes (`--init-from`, `--push-every`).
- `--init-from` starts a fresh CMA-ES at the checkpoint's mean and step size, so the bigger Modal popsize is fine.
- The Starter plan runs 100 containers at once, so a 256-episode generation runs in about 3 waves.
  Pass `--workspace-containers` to `::smoke` to match your plan.
- Use a different `--seed` for extra seeds of the same arm (they publish to `checkpoints/<arm>_s<seed>/`).
- If an episode fails all retries the run stops, but it saves and publishes first. Resume with
  `--resume runs/<run_name> --run-name <run_name>`.
- Once your runs are going, tell Parth so he can stop the local runs (Ctrl+C or `runs/<run_name>/STOP`).

## Evaluate and compare arms

```bash
python -m flyfollow.rl.evaluate --checkpoints ../fruitfly-training-results/checkpoints --workers 6 --n-test 100
```

Writes `runs/eval/<stamp>/metrics.csv` and `chart.png` (time in band, collisions, approach success,
bearing error) for every checkpoint plus the hand-tuned PID.
