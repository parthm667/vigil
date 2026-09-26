# Local training results, 2026-09-26 (overnight, Parth's laptop)

All runs used the small local config (`configs/train_local.yaml`: popsize 16, 4 episodes per candidate,
30 s follow and 20 s approach episodes) on 8 CPU cores. Checkpoints are on the `rl-results` branch.

## Test-set comparison (30 held-out seeds per kind)

`python -m flyfollow.rl.evaluate --n-test 30 --config configs/train_local.yaml` at 11:16. Full numbers:
[eval_local_0926-1116.csv](eval_local_0926-1116.csv), chart: [eval_local_0926-1116.png](eval_local_0926-1116.png).

| Arm | Follow: time in band | RMS range error | Mean range error | RMS bearing error | Collisions | Follow score | Approach success |
|---|---|---|---|---|---|---|---|
| PID, hand-tuned | 33 % | 0.77 m | +0.29 m | 6.5 deg | 6.7 % | -1.00 | 87 % |
| PID, CMA-tuned (400 gen) | 35 % | 0.76 m | +0.27 m | 5.9 deg | 6.7 % | -0.88 | 87 % |
| No-brain (800 gen) | **40 %** | 0.82 m | +0.47 m | 6.2 deg | 3.3 % | **-0.83** | 80 % |
| Shuffled connectome (80 gen, still running) | 24 % | 1.15 m | +0.87 m | 7.4 deg | 3.3 % | -1.68 | 83 % |
| Fly, hand calibration only | 1 % | 5.13 m | +4.76 m | 32.1 deg | 0 % | -5.55 | 0 % |
| **Fly, trained (400 gen)** | 13 % | **1.41 m** | **+0.67 m** | 10.2 deg | 13.3 % | -2.60 | 77 % |

Score is the episode return divided by |hand-tuned PID return| on the same seed; -1.0 means as good as the hand-tuned PID.

## What this says

1. Training works: the fly went from 1 % to 13 % time in band and from -5.5 to -2.6 on follow score, and the
   no-brain control ended better than both PIDs on follow.
2. The full fly controller fails mostly on **range, not steering**. Its RMS range error is about twice the
   PID's and it hangs back 0.67 m too far on average. This is the weakness the G0 audit predicted: the
   connectome's size signal is weak and rides on DNg13 and the one-sided DNb06, while DNa02 (the clean
   bearing signal) falls with size.
3. The fly's steering is usable. As a check we started a **yaw-only fly** (the brain steers, the PID range
   loop sets forward speed) from the trained fly checkpoint: its first selection score was **-0.60**, next to
   the CMA-tuned PID (-0.52) and no-brain (-0.37), where the full fly had plateaued around -2.3.
4. As predicted in the plan (section 4.7), the controls are at least as good as the fly. The shuffled
   connectome beating the full real one is not evidence against the wiring for steering: the audit showed
   shuffled wiring loses the bearing signal (hand-calibration yaw R2 0.50 versus 0.94 for the real wiring).
   The fair test of the wiring is the yaw-only comparison below.

## Recommended Modal runs (the teammate)

In priority order. Each publishes to its own checkpoint folder on `rl-results`.

1. **Yaw-only fly**, the main fly result and the demo candidate:
   `--arm fly --config configs/train_modal_yaw_only.yaml --label fly_yaw_only --init-from ../fruitfly-training-results/checkpoints/fly_yaw_only/latest.json`
2. **Yaw-only controls** with the same config: `--arm nobrain --label nobrain_yaw_only --init-from .../nobrain_yaw_only/latest.json`
   and `--arm fly_shuf --label fly_shuf_yaw_only` (from scratch).
3. Extra seeds (`--seed 2`, `--seed 3`) of the yaw-only fly and controls, so the chart has run-to-run spread.
4. Optional: the full fly (brain forward) at full size, to see whether longer episodes and a bigger population fix range.

Then run `python -m flyfollow.rl.evaluate --checkpoints ../fruitfly-training-results/checkpoints --n-test 100`
for the final chart. Compare yaw-only arms on bearing error, loss per minute and time in band.

## Update 12:00: yaw-only comparison (best selection score, 16 selection seeds)

| Yaw-only arm | Generations | Best selection |
|---|---|---|
| No-brain yaw-only | 400 (finished) | -0.489 |
| PID, CMA-tuned (reference) | 400 | -0.523 |
| Fly yaw-only | 50 (running) | -0.537 |
| Shuffled yaw-only | just started | n/a |

With the brain only steering, the fly is level with the PID and the no-brain control, within the noise of
16 selection seeds. The test-set evaluation with 100 seeds per kind (after the Modal runs) is the real comparison.

## Final local state, 12:15 (all runs stopped, checkpoints pushed)

Fly yaw-only reached -0.515 best selection at 78 generations. See the table in HANDOFF.md on `rl-pipeline`.

## Local runs at 11:25 (now stopped)

- `fly_yaw_only` (6 cores, from the fly checkpoint), `nobrain_yaw_only` (1 core, from the no-brain checkpoint),
  `fly_shuf` full (1 core). Stop them with Ctrl+C or `runs/<run_name>/STOP` once the Modal runs start.
