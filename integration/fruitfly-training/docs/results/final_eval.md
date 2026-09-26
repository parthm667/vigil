# Evaluation: final

2026-09-26T18:50:27+00:00. Tag `v1`, 125 follow + 75 approach episodes on TEST_SEEDS per set, episode_s override None. Trained arms: best checkpoint (by selection fitness) of each run seed; values are the mean across run seeds with [min, max] across seeds. Fitness is each episode's return / max(|PID-HAND return on the same seed and profile|, 20), clipped to [-4, 2], weighted 0.5 follow + 0.5 approach, as in training; higher is better.

## Set: test (profile train)

| arm | fitness | follow in band | signed range err (m) | RMS bearing (deg) | losses/min | follow min dist (m) | safety/min | approach success | time to standoff (s) | overshoot (m) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLY-YAW-HAND | -0.859 | 0.415 | 0.36 | 8.9 | 2.61 | 1.41 | 6.39 | 0.947 | 6.1 | -0.02 | 0 |
| FLY-YAW | -0.634 [-0.640, -0.628] | 0.424 [0.424, 0.425] | 0.35 [0.34, 0.35] | 6.6 [6.4, 6.9] | 2.62 [2.62, 2.63] | 1.39 [1.39, 1.39] | 6.27 [6.22, 6.34] | 0.933 [0.933, 0.933] | 6.0 [6.0, 6.0] | -0.05 [-0.05, -0.05] | 0 [0, 0] |
| FLY-SHUF-YAW | -0.813 [-0.880, -0.775] | 0.410 [0.405, 0.412] | 0.39 [0.38, 0.40] | 11.0 [10.0, 12.4] | 2.72 [2.67, 2.77] | 1.40 [1.39, 1.41] | 6.61 [6.58, 6.65] | 0.933 [0.933, 0.933] | 6.0 [6.0, 6.0] | -0.05 [-0.05, -0.05] | 0 [0, 0] |
| NOBRAIN-YAW | -0.613 [-0.619, -0.600] | 0.429 [0.428, 0.430] | 0.34 [0.34, 0.35] | 5.5 [5.0, 5.8] | 2.60 [2.60, 2.60] | 1.39 [1.39, 1.40] | 6.21 [6.19, 6.24] | 0.933 [0.933, 0.933] | 6.0 [6.0, 6.0] | -0.05 [-0.05, -0.04] | 0 [0, 0] |
| FLY-HAND | -2.129 | 0.188 | 0.87 | 9.1 | 2.58 | 1.68 | 5.69 | 0.347 | 11.2 | 0.20 | 0 |
| PID-HAND | -0.683 | 0.426 | 0.34 | 6.3 | 2.60 | 1.39 | 6.30 | 0.933 | 6.0 | -0.05 | 0 |
| PID-CMA | -0.642 [-0.651, -0.627] | 0.422 [0.410, 0.443] | 0.36 [0.32, 0.37] | 5.4 [5.2, 5.7] | 2.60 [2.60, 2.60] | 1.40 [1.39, 1.41] | 6.27 [6.21, 6.38] | 0.951 [0.933, 0.960] | 6.2 [5.9, 6.4] | -0.02 [-0.08, 0.02] | 0 [0, 0] |

## Set: demo (profile demo)

| arm | fitness | follow in band | signed range err (m) | RMS bearing (deg) | losses/min | follow min dist (m) | safety/min | approach success | time to standoff (s) | overshoot (m) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLY-YAW-HAND | -0.987 | 0.437 | 0.30 | 8.2 | 2.43 | 1.42 | 5.77 | 0.773 | 6.5 | -0.09 | 0 |
| FLY-YAW | -0.697 [-0.699, -0.693] | 0.436 [0.435, 0.436] | 0.32 [0.31, 0.32] | 4.6 [4.5, 4.8] | 2.33 [2.33, 2.34] | 1.44 [1.44, 1.45] | 5.65 [5.62, 5.67] | 0.773 [0.773, 0.773] | 6.0 [6.0, 6.0] | -0.13 [-0.13, -0.13] | 0 [0, 0] |
| FLY-SHUF-YAW | -0.823 [-0.913, -0.774] | 0.425 [0.418, 0.429] | 0.33 [0.32, 0.34] | 6.5 [5.4, 8.5] | 2.43 [2.37, 2.51] | 1.45 [1.45, 1.45] | 5.79 [5.60, 6.05] | 0.778 [0.773, 0.787] | 6.0 [5.9, 6.0] | -0.12 [-0.13, -0.11] | 0 [0, 0] |
| NOBRAIN-YAW | -0.687 [-0.703, -0.673] | 0.436 [0.435, 0.437] | 0.33 [0.32, 0.34] | 4.8 [4.5, 5.0] | 2.32 [2.32, 2.33] | 1.45 [1.44, 1.45] | 5.61 [5.53, 5.71] | 0.778 [0.773, 0.787] | 6.0 [5.9, 6.0] | -0.12 [-0.12, -0.11] | 0 [0, 0] |
| FLY-HAND | -2.113 | 0.157 | 0.82 | 7.9 | 2.35 | 1.75 | 4.78 | 0.200 | 11.3 | 0.32 | 0 |
| PID-HAND | -0.708 | 0.437 | 0.32 | 4.7 | 2.33 | 1.45 | 5.67 | 0.787 | 5.9 | -0.11 | 0 |
| PID-CMA | -0.704 [-0.710, -0.700] | 0.426 [0.410, 0.455] | 0.34 [0.30, 0.36] | 4.6 [4.5, 4.8] | 2.33 [2.33, 2.33] | 1.45 [1.44, 1.46] | 5.70 [5.66, 5.73] | 0.800 [0.760, 0.827] | 6.3 [5.8, 6.6] | -0.08 [-0.15, -0.03] | 0 [0, 0] |

## Set: stress (profile stress)

| arm | fitness | follow in band | signed range err (m) | RMS bearing (deg) | losses/min | follow min dist (m) | safety/min | approach success | time to standoff (s) | overshoot (m) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FLY-YAW-HAND | -0.729 | 0.392 | 0.39 | 14.0 | 2.60 | 1.32 | 7.75 | 0.933 | 5.9 | -0.04 | 0 |
| FLY-YAW | -0.588 [-0.622, -0.552] | 0.398 [0.394, 0.402] | 0.38 [0.36, 0.39] | 12.0 [11.2, 13.5] | 2.61 [2.57, 2.67] | 1.33 [1.32, 1.35] | 8.09 [7.92, 8.20] | 0.942 [0.933, 0.947] | 5.8 [5.8, 5.8] | -0.06 [-0.07, -0.05] | 0 [0, 0] |
| FLY-SHUF-YAW | -0.630 [-0.687, -0.588] | 0.392 [0.389, 0.398] | 0.38 [0.37, 0.39] | 12.3 [11.1, 13.1] | 2.66 [2.61, 2.71] | 1.34 [1.34, 1.34] | 7.72 [7.57, 7.82] | 0.947 [0.947, 0.947] | 5.7 [5.7, 5.7] | -0.05 [-0.06, -0.05] | 0 [0, 0] |
| NOBRAIN-YAW | -0.593 [-0.611, -0.567] | 0.401 [0.399, 0.404] | 0.36 [0.36, 0.37] | 11.4 [10.8, 11.7] | 2.61 [2.58, 2.65] | 1.33 [1.33, 1.33] | 7.93 [7.78, 8.17] | 0.938 [0.933, 0.947] | 5.9 [5.8, 6.0] | -0.06 [-0.06, -0.05] | 0 [0, 0] |
| FLY-HAND | -1.174 | 0.231 | 0.73 | 13.2 | 2.56 | 1.49 | 6.69 | 0.880 | 10.6 | 0.11 | 0 |
| PID-HAND | -0.679 | 0.394 | 0.41 | 13.0 | 2.75 | 1.33 | 9.22 | 0.933 | 6.0 | -0.07 | 0 |
| PID-CMA | -0.622 [-0.625, -0.619] | 0.396 [0.386, 0.414] | 0.39 [0.35, 0.40] | 10.7 [10.6, 10.9] | 2.63 [2.58, 2.66] | 1.34 [1.33, 1.35] | 8.51 [8.42, 8.67] | 0.942 [0.933, 0.947] | 6.0 [5.9, 6.1] | -0.03 [-0.10, 0.01] | 0 [0, 0] |

## Brain-use checks: lesion and bias audit (test set subset)

| run | RMS bearing (deg) intact | lesioned | in view intact | lesioned | follow in band intact | lesioned | approach success intact | lesioned | |b_yaw| / yaw drive | |b_fwd| / fwd drive |
|---|---|---|---|---|---|---|---|---|---|---|
| FLY-YAW_s1_v1 | 8.7 | 29.5 | 0.967 | 0.797 | 0.407 | 0.271 | n/a | n/a | 0.51 | n/a (PID forward) |
| FLY-YAW_s2_v1 | 8.8 | 29.8 | 0.973 | 0.794 | 0.412 | 0.287 | n/a | n/a | 0.03 | n/a (PID forward) |
| FLY-YAW_s3_v1 | 9.3 | 29.3 | 0.964 | 0.811 | 0.407 | 0.278 | n/a | n/a | 0.38 | n/a (PID forward) |

Lesion: every DN readout channel clamped to its own episode mean (from an intact first pass). If the brain does the steering, bearing error should rise and time in view and in band collapse. Bias ratio: |b| over the mean |DN-driven term| inside the tanh (above about 1, the bias does more than the brain).

## Entries

| entry | arm | source | best gen | selection fitness |
|---|---|---|---|---|
| FLY-YAW-HAND | FLY-YAW-HAND | init |  | n/a |
| FLY-YAW_s1_v1 | FLY-YAW | best.json | 40 | -0.760 |
| FLY-YAW_s2_v1 | FLY-YAW | best.json | 60 | -0.759 |
| FLY-YAW_s3_v1 | FLY-YAW | best.json | 140 | -0.753 |
| FLY-SHUF-YAW_s1_v1 | FLY-SHUF-YAW | best.json | 120 | -0.987 |
| FLY-SHUF-YAW_s2_v1 | FLY-SHUF-YAW | best.json | 150 | -0.840 |
| FLY-SHUF-YAW_s3_v1 | FLY-SHUF-YAW | best.json | 150 | -0.839 |
| NOBRAIN-YAW_s1_v1 | NOBRAIN-YAW | best.json | 50 | -0.751 |
| NOBRAIN-YAW_s2_v1 | NOBRAIN-YAW | best.json | 10 | -0.751 |
| NOBRAIN-YAW_s3_v1 | NOBRAIN-YAW | best.json | 10 | -0.757 |
| FLY-HAND | FLY-HAND | init |  | n/a |
| PID-HAND | PID-HAND | init |  | n/a |
| PID-CMA_s1_v1 | PID-CMA | best.json | 30 | -0.768 |
| PID-CMA_s2_v1 | PID-CMA | best.json | 60 | -0.777 |
| PID-CMA_s3_v1 | PID-CMA | best.json | 20 | -0.745 |
