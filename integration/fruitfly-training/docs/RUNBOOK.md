# Runbook: training the fly pursuit controller on Modal

How to set up, launch, watch, stop and evaluate the CMA-ES training in `docs/DRONE_RL_PLAN.md` Section 4.
Everything below was run end to end on 2026-09-26 (Apple M5 laptop, Modal Starter plan).

## 0. What exists

| Piece | Where | Status |
|---|---|---|
| FlyDrones (brain sim, MaleCNS loader) | `third_party/FlyDrones` (upstream `3e26934` + somaSide patch) | vendored |
| Brains | `data/brains/pursuit_core1.npz` (1,446 neurons) + 3 shuffles | built, audited (G0 pass, `docs/audit_result.md`) |
| Simulator, governor, Kalman filter, PID, reward | `flyfollow/sim`, `flyfollow/pilot`, `flyfollow/rl/env.py` | 22 tests |
| Fly and no-brain controllers, 47 parameters, calibration | `flyfollow/senses`, `flyfollow/pilot`, `flyfollow/rl/controllers.py` | 28 tests |
| CMA-ES trainer, Modal app, eval, chart | `flyfollow/rl/cma_train.py`, `modal_app.py`, `eval.py`, `chart.py` | 25 tests, Modal smoke passed |
| Configs | `configs/env.yaml`, `controllers.yaml`, `train.yaml`, `pursuit_malecns.yaml` | frozen for launch |

`data/` is gitignored; rebuild it with step 1.

## 1. One-time setup (any teammate's machine)

```bash
scripts/setup_env.sh            # Python 3.12 venv + all deps; also fixes the macOS hidden-.pth quirk
.venv/bin/modal setup           # browser login, once per machine (only the Modal account owner)
scripts/setup_data.sh           # download MaleCNS (1.2 GB), build brains, shuffles, bench (about 10 min)
.venv/bin/python -m flyfollow.audit.audit --brain data/brains/pursuit_core1.npz   # G0 audit, 15 s
.venv/bin/python -m flyfollow.rl.calibrate --all                                   # init params for every arm
.venv/bin/python -m pytest -q                                                      # all tests
```

Recalibrating changes what saved parameter vectors mean. Never recalibrate between launch and eval.

## 2. Before launch (5 minutes)

1. **Set a hard cap in Modal**: modal.com, Settings, Usage & Billing, workspace budget (usage limit) at $1,000 or lower.
   This is the only cap Modal itself enforces. Our code also stops each run at its share of `budget.total_usd` ($800) in `configs/train.yaml`.
2. Push brains and calibration to the Volume (again after any recalibration):
   `.venv/bin/modal run -m flyfollow.rl.modal_app::upload`
3. Preview the plan and projected cost (spawns nothing):
   `.venv/bin/modal run -m flyfollow.rl.modal_app::launch`

## 3. Launch (only with the owner's go-ahead)

```bash
.venv/bin/modal run --detach -m flyfollow.rl.modal_app::launch --confirm
```

This spawns one `orchestrate` call that starts the 12 runs (seed 1 of every arm first) and restarts any that die.
Default arms (`configs/train.yaml` `runs`): **FLY-YAW** (the fly steers, PID-HAND's range loop sets forward speed; the demo
controller), FLY-SHUF-YAW (shuffled wiring steers), NOBRAIN-YAW (no brain steers) and PID-CMA, each with seeds 1 to 3.
The full fly (brain also sets forward speed) is optional: `--arms FLY-CMA --seeds 1`.
With `--detach` the runs keep going if the laptop closes. Launch ids are appended to `docs/training_log.md`.
Subsets: `--arms FLY-CMA,NOBRAIN --seeds 1 --gens 150 --tag v1`.

## 4. Watch

```bash
.venv/bin/modal run -m flyfollow.rl.modal_app::status     # per-run generation, fitness, cost estimate, and Modal's billed cost
.venv/bin/modal billing summary                           # actual spend
.venv/bin/modal app list                                  # running apps
```

Healthy: fitness finite and rising, `err 0`, cost per FLY generation about $0.06 (K = 32).

## 5. Stop or resume

- Stop everything: `.venv/bin/modal app stop <app-id> -y` (ids from `modal app list`).
- Resume: re-run the same `launch --confirm` command with the same `--tag`. Every run resumes from its last generation
  (state is pickled and committed to the Volume every generation; preemptions resume automatically).

## 6. Evaluate and chart

```bash
.venv/bin/modal run -m flyfollow.rl.modal_app::fetch                  # runs -> data/runs/
.venv/bin/python -m flyfollow.rl.eval --name final --backend modal    # test, demo, stress sets; lesion and bias checks
.venv/bin/python -m flyfollow.rl.chart --name final                   # slide PNG in data/runs/eval/
```

Gate G2 (plan Section 8) is read off the demo-profile rows.

## 7. Measured numbers and lead decisions (2026-09-26)

| Quantity | Value |
|---|---|
| Brain tick, core1 (1,446 neurons) | 1.7 ms per 50 ms tick on the M5 |
| One 60 s follow episode, FLY arm | 2.2 s on the M5; 2.5 s mean on Modal |
| One FLY-CMA generation at the plan's K = 8 on Modal (smoke) | 21.5 s wall, 91 container-seconds, $0.015 |
| One FLY-YAW generation at K = 32 on Modal (smoke) | 24 s wall, 520 container-seconds, about $0.09 |
| Projection, 12 default runs x 150 generations at K = 32 | about $70 to $75, about 1.4 h wall (Starter container cap bound) |
| Container cap | Starter plan 100; config uses 80 eval + 12 drivers + 1 orchestrator |

Changes from the plan, each backed by a measurement (details in the config comments):

| Setting | Plan | Now | Why |
|---|---|---|---|
| Pursuit core | hops 1 or 2 | hops 1 only | core2 has runaway activity (DNb05 pinned near 380 Hz), fails the audit |
| `cma.sigma0` | 0.2 | 0.05, readout weights x0.08, biases x0.2, `maxstd` 0.3 | readout bounds are ~10x their useful scale: at 0.2 almost every candidate was far worse than the init and local runs never improved |
| K per candidate | 5 + 3 | 20 + 12 | split-half rank reliability of candidate scores 0.80 -> 0.91 (NOBRAIN), 0.55 -> 0.75 (PID) |
| Episode score | ret / max(abs(PID ret), 1) | floor 20, clipped to [-4, 2] | near-zero PID approach returns and single crashes dominated whole generations |
| Approach reward band | around Z_ref | around the achievable standoff (true size / prior x Z_ref) | the size-prior error is invisible to the controller; plan 4.2 already accepts it |
| Low-object standoff | 2.5 x height difference, max 2.0 m | 3.0 x, max 2.4 m | 2.5 put floor objects exactly on the lower image edge |
| What the fly controls | yaw and forward | yaw only; PID-HAND's range loop sets forward (plan 4.3 "yaw-only fly") | Parth's 400-generation local run: the full fly fails on range (13 % in band, hangs back 0.67 m) while the yaw-only fly matches PID and no-brain |
| Drone dynamics in the sim | wide guesses | centered on Parth's Tello lag test (yaw 55 deg/s, 0.18 s dead; forward 0.96 m/s, 0.47 s dead, tau 0.45 s; at stick 30) | plan Section 0 rule: center on measured values +-30 % |
| `max_fwd_stick` | 20 to 50 | 40 to 80 (demo 60) | at ~1 m/s per 100 stick, 20 to 50 is slower than a person walks |
| `max_back_stick` | 20 | 20 to 40 randomized (demo 40) | 20 is 0.19 m/s: a person stepping back toward the drone hits it (demo collisions 5 -> 0 at 40). No rear sensing: confirm at R2/R3 with a spotter |

## 8. Visualization (fly body + brain, for the demo screen)

```bash
scripts/setup_viz.sh                                                  # flybody model into data/flybody/ (140 MB, gitignored)
.venv/bin/python -m flyfollow.viz.demo --seed 1000 --kind follow      # live Rerun window, simulated episode
.venv/bin/python -m flyfollow.viz.demo --seed 1000 --seconds 30 --no-live --record runs/viz/demo.mp4
.venv/bin/python -m flyfollow.viz.demo --params data/runs/<run>/best.json ...   # show a trained controller
```

Details, the brain-to-body mapping and the live-drone hook (`VizSink`) are in `docs/VIZ.md`.

## 9. Known issues

- Evaluation containers keep their worker pool until exit, so Modal waits up to 30 s at shutdown (a few cents in total).
  The clean fix is an `@app.cls` with `@modal.exit`.
- Demo-profile dynamics in `configs/env.yaml` (`profiles.demo`) are Parth's lag-test medians at stick 30 only; forward gain at
  larger sticks is extrapolated linearly (unverified). Measure stick 60 and 100 and the camera FOV at R0, then update.
- The LC10a azimuth bins use the rank fallback (MaleCNS has no hex coordinates for LC10a); trained per-bin gains compensate.
