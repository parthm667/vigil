# Training log

## Modal smoke test: FLY-CMA (2026-09-26 09:19)

- One generation: population 32, K = 8, 264 episodes incl. PID-HAND denominators, 17 chunks of 16, episode_s override None; 0 errors.
- Generation wall time 21.5 s; chunk wall mean 5.36 s, max 8.71 s.
- Episode wall time (s): follow mean 2.50, median 2.30, p90 4.42, max 7.02 (n=165); approach mean 1.16, median 1.02, p90 1.87, max 3.36 (n=99); brain time mean 1.96 s.
- Container startup (submit to first chunk start, cold containers): mean 9.00, median 9.73, p90 11.77, max 13.51 (n=12).
- Container-seconds per generation 91; cost per generation $0.013 at list price, $0.015 with the 1.2x startup/idle allowance.
- Projection for all planned runs:

```
runs: 12, eval container {'cpu': 8.0, 'memory_mib': 16384, 'workers': 16} at $0.506/h, max_containers 80
container-hours: 32.9; cost: eval $20.00 + drivers $0.12 = $20.12
wall: 0.54 h (one chunk wave per generation: 0.54 h; container cap: 0.41 h)
  FLY-CMA   container-s/gen 91, chunk wall 8.7 s (smoke)
  FLY-SHUF  container-s/gen 169, chunk wall 10.0 s (config guess)
  NOBRAIN   container-s/gen 2, chunk wall 0.1 s (config guess)
  PID-CMA   container-s/gen 1, chunk wall 0.1 s (config guess)
```

## Launch launch-20260926-092010 (2026-09-26 09:20): PLUMBING TEST ONLY (tag e2etest, 2 gens, no-brain arms), not the training run

- orchestrate call `fc-01M3F88NZH2KYCD34TZ7JXX5VW` (train call ids are in the Volume at `runs/_launch/launch-20260926-092010.json`)
- runs: NOBRAIN s1 (2 gens), PID-CMA s1 (2 gens)
- projected $0, 0.0 h

## Modal smoke test: FLY-YAW (2026-09-26 09:53)

- One generation: population 32, K = 32, 1056 episodes incl. PID-HAND denominators, 66 chunks of 16, episode_s override None; 0 errors.
- Generation wall time 24.1 s; chunk wall mean 7.89 s, max 12.93 s.
- Episode wall time (s): follow mean 5.89, median 4.78, p90 8.63, max 11.50 (n=660); approach mean 0.95, median 0.89, p90 1.54, max 2.26 (n=396); brain time mean 3.98 s.
- Container startup (submit to first chunk start, cold containers): mean 7.40, median 7.23, p90 10.52, max 11.44 (n=59).
- Container-seconds per generation 521; cost per generation $0.073 at list price, $0.088 with the 1.2x startup/idle allowance.
- Projection for all planned runs:

```
runs: 12, eval container {'cpu': 8.0, 'memory_mib': 16384, 'workers': 16} at $0.506/h, max_containers 80
container-hours: 112.8; cost: eval $68.57 + drivers $0.13 = $68.71
wall: 1.41 h (one chunk wave per generation: 0.66 h; container cap: 1.41 h)
  FLY-SHUF-YAW  container-s/gen 367, chunk wall 8.7 s (smoke of FLY-CMA)
  FLY-YAW       container-s/gen 524, chunk wall 12.9 s (smoke)
  NOBRAIN-YAW   container-s/gen 8, chunk wall 0.1 s (config guess)
  PID-CMA       container-s/gen 4, chunk wall 0.1 s (config guess)
```

## Launch launch-20260926-101626 (2026-09-26 10:16)

- orchestrate call `fc-01M3FBFQ9NMJ5QWPFS8XTFQMR1` (train call ids are in the Volume at `runs/_launch/launch-20260926-101626.json`)
- runs: FLY-YAW s1 (150 gens), FLY-SHUF-YAW s1 (150 gens), NOBRAIN-YAW s1 (150 gens), PID-CMA s1 (150 gens), FLY-YAW s2 (150 gens), FLY-SHUF-YAW s2 (150 gens), NOBRAIN-YAW s2 (150 gens), PID-CMA s2 (150 gens), FLY-YAW s3 (150 gens), FLY-SHUF-YAW s3 (150 gens), NOBRAIN-YAW s3 (150 gens), PID-CMA s3 (150 gens)
- projected $69, 1.4 h

## Launch launch-20260926-123524 (2026-09-26 12:35)

- orchestrate call `fc-01M3FKE5K8DZZ40621DWN8E4WR` (train call ids are in the Volume at `runs/_launch/launch-20260926-123524.json`)
- runs: FLY-YAW s3 (40 gens)
- projected $4, 0.2 h

## Launch launch-20260926-125350 (2026-09-26 12:53)

- orchestrate call `fc-01M3FMFXVFTXKBQ8PEC6WX588K` (train call ids are in the Volume at `runs/_launch/launch-20260926-125350.json`)
- runs: FLY-YAW s3 (40 gens)
- projected $4, 0.2 h
