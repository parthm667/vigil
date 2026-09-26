# Archive: merged branches kept for reference

On 2026-09-26 every branch was merged into `main` so the repo has one branch. Two branches contained a
parallel implementation of the same plan that uses the same `flyfollow/` package paths as the live code, so
their files are preserved here as read-only snapshots instead of being mixed into the live package. Their full
git history is also in `main` (real merge commits), and each old branch tip is tagged.

| Folder | From | Tag of the old tip | What it is |
|---|---|---|---|
| `rl-pipeline/` | branch `rl-pipeline` (Parth) | `archive/rl-pipeline` | Parth's overnight pipeline: simulator, controllers, CMA-ES trainer with git publishing, Modal app (never run on Modal), audit, the **Tello lag test** (`data/lag_test/`, the source of the simulator's measured dynamics), local results (`docs/results/LOCAL_RESULTS_0926.md`: full fly fails on range, yaw-only fly matches the controls) and `HANDOFF.md`. Its vendored `third_party/FlyDrones` was dropped (same upstream as the live copy). |
| `rl-results/` | branch `rl-results` (Parth's trainer) | `archive/rl-results` | Checkpoints published by that trainer (`checkpoints/<arm>/latest.json`) and its status `HANDOFF.md`. |

The live code is `flyfollow/` at the repo root (see `docs/RUNBOOK.md`, `docs/TECH_NOTE.md`). Nothing under
`archive/` is imported or tested. Files that were identical to the root copies (`FLYGUIDE_SPEC.md`, `research/`,
`docs/DRONE_RL_PLAN.md`, `docs/PLAN.md`) were not duplicated here.
