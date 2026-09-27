# Final report: a fruit fly brain steers the drone

2026-09-26. Short version for the team. Details: `docs/TECH_NOTE.md` (how it works), `docs/results/final_eval.md`
(all numbers), `docs/FLIGHT_TEST_CHECKLIST.md` (what to do at the drone), `docs/integration/` (the two patches).

## What we built

- **A real fly brain circuit that steers.** 1,446 neurons from the male fruit fly connectome (MaleCNS), the
  circuit a male fly uses to chase a target. Its wiring is never changed. We trained only the small adapter around
  it: how the camera's target box becomes input to the brain, and how the brain's two steering neurons become a
  yaw command.
- **What the fly does on the drone:** it turns the drone toward the person (or the object it found). Distance,
  safety, search and everything else is regular code in the team's ReachGlass stack.
- **How it plugs in:** two patch files for the ReachGlass repo (`docs/integration/`). One adds `steering: fly`,
  the other adds obstacle avoidance while following. Default is the old PID steering, so nothing changes until you
  switch it on.
- **Visuals for the demo:** a 3D fly body and the brain lighting up live, driven by the brain's outputs
  (`docs/VIZ.md`).

## Does it work? (simulation)

Trained on Modal: 12 runs, 150 generations each (about $86), plus two short smoothness fine-tunes (about $14), so
about $100 in total. Tested on 200 situations it never saw in training.

| Steering | Steering error | vs hand-tuned PID score |
|---|---|---|
| **Fly, trained** | **6.6 deg** | better (-0.63 vs -0.68) |
| Fly, untrained | 8.9 deg | worse |
| Fly with its wiring shuffled | 11.0 deg | clearly worse (-0.81) |
| No brain (same adapter, brain removed) | 5.5 deg | about the same |
| PID tuned by the same training | 5.4 deg | about the same |

Plain version: the trained fly steers about as well as normal control code. Scrambling its wiring breaks it, so
the real wiring matters. Knocking out its steering neurons makes the error jump from about 9 to about 29 deg, so
the brain really is doing the steering. The no-brain version is a hair better, which we predicted up front.

**In the team's own simulator**, with the version we ship (trained, then fine-tuned for a smoother stick):
6.1 deg steering error vs 8.5 for the stack's PID, a better worst case too (13.9 vs 15.5 deg), fewer times losing
the person (0.55 vs 0.70 per run), and no collisions. Its find-the-bottle runs all arrived, about 1 s slower than
the PID because the fly turns smoothly instead of in steps. With obstacles in the way the fly held up better than the no-brain
and PID versions in our simulator (8.5 vs about 10 deg).

## Known issues

- **Stick smoothness.** The fly moves the stick more than the stack's PID. Training cut that by two thirds, a
  fine-tune that penalizes stick changes cut it another quarter, and a small deadband plus hysteresis on the output
  brought it to 0.57 per step vs 0.29 for the PID, while still steering better than the PID. A stronger fine-tune
  is a bit smoother (0.53) but starts steering worse, so we did not pick it. Filtering the stick more adds lag and
  makes the drone wobble.
- **Hiding behind tall things.** Obstacle avoidance stops crashes (a doorway crash went from every time to never),
  but if the person walks behind something tall like a bookcase, the drone usually loses them. Demo in an open
  space.
- **Simulation only so far.** The drone model uses the team's real lag test, but only at one stick level, and the
  camera field of view is not calibrated yet. Real flights may need a small re-tune (about $20, 30 minutes).

## What's left (needs people at the drone)

1. Ground checks and a dry-run sign check (10 to 15 min).
2. Follow in the open area with PID first, then the fly.
3. Record the demo: phone video of the drone next to the laptop showing the fly body and brain.

Setup on the flying laptop is one command, `python scripts/setup_flight.py --viz` (then `python scripts/preflight.py`
at the drone). To fly from this repo alone (its own YOLO person detector, no ReachGlass):
`python scripts/setup_standalone.py`, then `docs/STANDALONE.md`. Every step, command and fix is in `docs/FLIGHT_TEST_CHECKLIST.md`. Note: on the second Mac an unrelated program holds
the Tello control port (UDP 8889); stop it or fly from another laptop.

## Where things are

| What | Where |
|---|---|
| Trained fly (use this) | `data/brains/trained/FLY-YAW_smooth_best.json` with `deadband: 4`, `hysteresis: 3` (also in the repo: `FLY-YAW_best.json`, `FLY-YAW_smooth2_best.json`) |
| Patches for ReachGlass | `docs/integration/reachglass_flysteer.patch`, `reachglass_follow_avoid.patch` |
| Setup and training commands | `docs/RUNBOOK.md` |
| Flight checklist | `docs/FLIGHT_TEST_CHECKLIST.md` (ReachGlass), `docs/STANDALONE.md` (this repo alone) |
| Parth's earlier pipeline and lag test | `archive/rl-pipeline/` |
