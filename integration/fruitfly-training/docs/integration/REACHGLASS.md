# Fruit fly steering in ReachGlass (FlySteer)

For the ReachGlass team (github.com/nathanwuzhao/jerkgt13). Patch: `docs/integration/reachglass_flysteer.patch`,
made against your HEAD `2302928`. We did not push anything to your repository: you apply the patch yourselves.

## What it does

`flyfollow.steer.FlySteer` puts the fruit fly's pursuit circuit in charge of **one number: the yaw stick**.
That circuit is 1,446 spiking neurons from the MaleCNS connectome (visual LC10a inputs, then AOTU relays,
then the steering descending neurons DNa02 and DNa01). Your stack keeps everything else: perception,
target lock, range, forward, altitude, orbit, lost-person search, the approach's discrete moves, the
mission, and the `SafetyGovernor`. Every rc command still goes through `ctx.drone`.

* **FOLLOW** (`follow.steering: fly`): your `yaw_gain * deadband(bearing)` line becomes the fly's yaw.
  Between frames and during dropouts under 0.35 s the fly's latency filter predicts the person's position.
  After that, your lost-person logic (hold, then search turn) owns yaw, exactly as today.
* **APPROACH** (`approach.steering: fly`): when `|bearing| > align_deg`, the fly turns onto the target with
  continuous rc yaw instead of `Discrete("rotate", bearing)`. It stops when the target is within
  `align_deg` on 3 frames, after 0.5 s without the target, or after 5 s. Then the drone hovers and settles
  as after a rotate. If the target is still more than `align_deg` off, one discrete rotate trims the rest.
  Forward moves, descents and sidesteps are unchanged.
* **Failure**: if `flyfollow` is not installed or the brain fails to load, the behaviour logs one warning
  and uses your yaw law. If the fly raises in flight, the behaviour also falls back to your yaw law.

Inside FlySteer, each 50 ms brain tick runs: bearing, then a latency Kalman filter (predicts across
`drone.video_lag_s`), then the LC10a encoder, then the LIF network, then the DN readout, then stick smoothing
(slew cap, deadband, hysteresis), then yaw. The tick
is fixed at 50 ms (the rate it was trained at), whatever your loop rate. Cost: 2 to 3 ms per tick on the
M-series laptop CPU. A single call runs at most 3 ticks and never blocks.

## Install and apply

```bash
cd jerkgt13 && source .venv/bin/activate
git apply /path/to/fruitfly-training/docs/integration/reachglass_flysteer.patch
pip install -e /path/to/fruitfly-training/third_party/FlyDrones -e /path/to/fruitfly-training   # add [viz] for the viewer
python -m pytest tests/test_fly_steer.py
```
The brain and its calibration are read from `fruitfly-training/data/brains/` (`scripts/setup_data.sh`
there). Set `FLYFOLLOW_DATA=/path/to/data` if the package is not installed in editable mode.

## Config keys (`site.yaml`)

```yaml
follow:   {steering: fly}        # pid (default) = your yaw law
approach: {steering: fly}        # pid (default) = discrete rotate
fly:
  params_path: /path/to/fruitfly-training/data/brains/trained/FLY-YAW_smooth_best.json   # recommended; "" = hand calibration
  brain_path: ""                 # "" = the brain named in best.json (pursuit_core1)
  latency_s: null                # null = drone.video_lag_s
  max_rc_yaw: 40                 # the fly's clamp; follow.max_rc_yaw and safety.max_rc still apply
  deadband: 4                    # stick units subtracted from |yaw| (like your 4 deg bearing deadband)
  hysteresis: 3                  # the sent stick only moves when the new value is more than 3 away
  slew: 300                      # max stick change per second (safety cap; never reached in the sim)
  smoothing_ms: 0                # low-pass on the stick; off: every setting from 40 to 250 ms cost bearing error
  viz: false                     # publish the fly body + brain view
```
**Back to PID**: `steering: pid` (or delete the keys). The code path is then exactly your current one.
**Trained parameters** (in `data/brains/trained/`):
- **`FLY-YAW_smooth_best.json` (recommended)**: v1 fine-tuned with a moderate stick-jerk penalty (w_j 26.4).
  With `deadband: 4` and `hysteresis: 3` (the defaults): 6.1 deg RMS and 13.9 deg p95 bearing error (better than
  your PID's 8.5 / 15.5 on both), 0.55 losses per run (PID 0.70), 0.57 stick change per loop (PID 0.29).
- `FLY-YAW_smooth2_best.json`: a stronger fine-tune (w_j 99). Slightly smoother (0.53 per loop) but it tracks
  worse (7.3 deg RMS, 16.9 deg p95, worse than your PID on fast sideways walking), and in our own sim its
  tracking loss is statistically significant. Use it only if the stick must be as calm as possible.
- `FLY-YAW_best.json`: v1, run `FLY-YAW_s3_v1` (selection score -0.753). It tracks tightest but moves the
  stick most. With it, set `deadband: 3` and `hysteresis: 2`.
Leave `params_path` empty for the untrained hand calibration. Both the `"x"` and `"params"` forms of a
trainer `best.json` load, and the file's `"arm"` and `"brain"` are used.
**Fly visualization** next to your dashboard: set `fly.viz: true`, then in a second terminal run
`python -m flyfollow.viz.live --brain /path/to/fruitfly-training/data/brains/pursuit_core1.npz`.
This needs the `[viz]` extra and `scripts/setup_viz.sh` for the fly body. Frames are dropped, never queued.

## Signs, units, geometry (checked)

Your conventions match ours, so no sign flips were needed. Bearing is positive when the target is right
of center, rc yaw is positive clockwise, and sticks run from -100 to 100. The patch passes your
`bearing_deg` (from your calibrated camera) as radians, so your `fx` and your 480x360 sim frames never
reach the fly.
**Size.** The fly always sees the target at its trained standoff size (normalized s = 1). The patch passes
no range, and your stack owns distance. Head size and the camera's vertical position do not enter.
We measured the alternative, s = `follow.distance_m` / your range estimate. In your sim, 27 % of person
detections had no range. Ranged s had a median of 0.45 (5 to 95 %: 0.2 to 0.9), because the drone trails
a walking person at about 2.2 m, not at the 1.0 m setpoint. At s = 0.45 the fly's drive is weaker, and the
hand calibration's bias then dominates. With the hand calibration over 20 runs, the RMS bearing error was
9.4 deg with range-based s and 6.4 deg with s = 1 (PID: 8.5 deg). `FlySteer.yaw(range_m=...)` remains
available. The trained fly was only measured at s = 1.
`Frame.t` is the arrival time, and the filter subtracts `latency_s`. Your follow and approach had no
continuous latency compensation, so nothing is compensated twice.
Behind-and-above pose: yaw uses only the horizontal bearing. The head sits about 14 deg below the axis,
which changes the bearing by under 3 %. Your 1.0 m follow distance is shorter than the fly's 1.5 to 2.5 m
training standoff. At that range a walking person sweeps the bearing faster, but the fly sees the same
normalized size.

## Sim comparison (your simulator)

**FOLLOW, closed loop.** This uses your `tests/test_follow.py` harness with the default follow geometry
(2.0 m high, 1.0 m behind). Error is the ground-truth bearing from the drone's heading to the person,
measured after t = 3 s. There are 4 scenarios with 5 seeds each, the same 20 runs for every row.
Scenarios: walk and 90 deg turn (your test), sideways zigzag, standing 30 deg off axis, slow arc.
"Stick change" is the mean |change in the yaw stick| per 15 Hz loop.

| Steering | RMS bearing error (deg) | p95 bearing (deg) | Losses > 1 s per run | Stick change per loop | RMS per scenario: walk / zigzag / standing / arc |
|---|---|---|---|---|---|
| **your PID** | 8.47 | 15.5 | 0.70 | **0.29** | 6.8 / 13.9 / 5.7 / 7.5 |
| fly, hand calibration, raw | 6.35 | 9.8 | 0.60 | 4.09 | 4.7 / 7.3 / 6.8 / 6.6 |
| fly, trained (s3), raw | **4.41** | **10.1** | 0.60 | 1.36 | 2.8 / 6.8 / 5.7 / 2.4 |
| **fly, trained, deadband 3 / hysteresis 2** | **5.00** | **11.0** | **0.45** | **0.86** | 3.1 / 8.0 / 5.9 / 3.1 |
| fly, trained s1, 150 ms low-pass + deadband 3 | 5.51 | 11.8 | 0.55 | 0.67 | 3.1 / 10.1 / 5.3 / 3.6 |
| fly, trained s2, 150 ms low-pass + deadband 3 | 5.16 | 11.1 | 0.50 | 0.72 | 2.8 / 9.2 / 5.5 / 3.2 |

No collisions in any row. Minimum distance to the person was 1.62 to 1.69 m in every row.
Open loop (fixed target, s = 1), the yaw stick with the target at 0 deg is +10.8 for the hand calibration
and +0.3 / +0.6 / +0.7 for trained s3 / s1 / s2. **The left bias is gone.** The trained s3 fly gives
-10 / 0 / +9 / +20 / +34 stick at -4 / 0 / +4 / +8 / +15 deg.

**Smoothing sweep** (trained s3, same 20 runs). Most of the jitter came from the untrained readout, and
training cut it from 4.1 to 1.4. What remains is mostly real steering, which is why no filter reached
your 0.29. Low-pass filters add loop lag, and the drone then oscillates: 250 ms raises the stick change
and loses the bearing advantage.

| Output conditioning | RMS (deg) | p95 (deg) | Losses > 1 s | Stick change per loop |
|---|---|---|---|---|
| none | 4.41 | 10.1 | 0.60 | 1.36 |
| deadband 3 | 4.98 | 11.0 | 0.55 | 1.04 |
| deadband 4 | 5.34 | 11.7 | 0.50 | 0.99 |
| deadband 3 + slew 300/s | 4.98 | 11.0 | 0.55 | 1.04 |
| **deadband 3 + hysteresis 2 (+ slew 300/s), the v1 choice** | **5.00** | **11.0** | **0.45** | **0.86** |
| deadband 3 + hysteresis 3 | 5.28 | 11.8 | 0.50 | 0.82 |
| low-pass 40 ms + deadband 3 | 5.40 | 12.1 | 0.50 | 0.94 |
| low-pass 40 ms + deadband 3 + hysteresis 2 | 5.36 | 11.9 | 0.60 | 0.80 |
| low-pass 80 ms + deadband 3 | 5.87 | 13.3 | 0.65 | 0.91 |
| low-pass 150 ms + deadband 3 | 6.41 | 14.2 | 0.60 | 0.88 |
| low-pass 150 ms + deadband 3 + slew 150/s | 6.81 | 15.5 | 0.70 | 0.90 |
| low-pass 250 ms + deadband 3 | 8.53 | 17.9 | 0.70 | 1.19 |
| DN rates averaged over 3 ticks + deadband 3 | 5.74 | 12.8 | 0.55 | 0.85 |

For v1, this setting keeps most of the bearing advantage over your PID: 41 % lower RMS and 29 % lower p95, with
the fewest losses of any row. It cuts the stick change per loop by 37 % against the raw trained fly and
by 79 % against the hand calibration. The stick still moves about 3 times as much per loop as your PID.
For a calmer stick, `smoothing_ms: 40` with `hysteresis: 2` gives 0.80 at 5.4 deg RMS.

**Smooth fine-tunes** (same 20 FOLLOW runs). Stick change is per 15 Hz loop.

| Parameters + conditioning | RMS (deg) | p95 (deg) | Losses > 1 s | Stick change per loop |
|---|---|---|---|---|
| your PID | 8.47 | 15.5 | 0.70 | 0.29 |
| v1 (s3) + deadband 3 / hysteresis 2 | 5.00 | 11.0 | 0.45 | 0.86 |
| smooth + deadband 3 / hysteresis 2 | 5.80 | 13.3 | 0.55 | 0.67 |
| smooth + deadband 4 / hysteresis 3 | 6.13 | 13.9 | 0.55 | 0.57 |
| smooth + deadband 5 / hysteresis 4 | 6.99 | 16.1 | 0.60 | 0.55 |
| smooth + 40 ms low-pass + deadband 4 / hysteresis 3 | 6.57 | 15.2 | 0.60 | 0.57 |
| smooth2 + deadband 3 / hysteresis 2 | 6.88 | 15.9 | 0.60 | 0.60 |
| **smooth2 + deadband 4 / hysteresis 3 (new default)** | **7.34** | **16.9** | **0.65** | **0.53** |

The rule was the smoothest setting whose RMS bearing error stays at or below your PID's 8.5 deg; the new
default is that setting. It is 38 % less stick motion than v1 and still 13 % lower RMS than your PID, but
it gives up some tail tracking. Its p95 (16.9 deg) is above your PID's 15.5, all of it in the zigzag
scenario: p95 30 deg against 29 deg for PID. `smooth` with deadband 4 / hysteresis 3 is the balanced
alternative: 0.57 stick change per loop at 6.1 deg RMS and 13.9 deg p95. The remaining stick motion is
mostly real steering, which is why no setting reached your PID's 0.29.

**Full mission** (`python -m reachglass sim --query "can you find my water bottle" --headless`, seeds 0 to 3)
and **follow me** (60 s demo scenario, logged RMS person bearing). Every run ARRIVED.

| Steering | Query to ARRIVED (mean) | APPROACH (mean) | Collisions (4 runs) | Follow-me RMS bearing |
|---|---|---|---|---|
| your PID | 27.7 s | 20.8 s | 0 | 4.2 deg |
| v1 (s3) + deadband 3 / hysteresis 2 | 31.9 s | 25.0 s | 0 | 0.8 deg |
| smooth + deadband 3 / hysteresis 2 | 28.9 s | 26.0 s | 1 | 1.1 deg |
| smooth + deadband 4 / hysteresis 3 | 28.8 s | 25.9 s | 0 | 2.2 deg |
| smooth2 + deadband 3 / hysteresis 2 | 33.8 s | 26.9 s | 1 | 1.0 deg |
| **smooth2 + deadband 4 / hysteresis 3 (default)** | **30.0 s** | **27.1 s** | **0** | **1.7 deg** |

Loops with a person detection were 0.90 in every follow-me run. Both fly collisions, and the one from the
earlier hand calibration, happened at the end of APPROACH during your discrete `descend 25 cm` next to
the table (logged as "move refused/failed"), not during a fly turn. Continuous fly turns make the
approach 4 to 6 s slower than your discrete rotates. With `reachglass_follow_avoid.patch` applied on top
(FollowAvoid wraps FollowBehind and passes the yaw through unchanged), a 40 s follow-me run with the
trained fly gave 1.0 deg RMS and no collisions.
**Your tests** (measured with the hand calibration, before training): with the patch and the default `steering: pid`, the full suite gives the same result as
unpatched HEAD: 235 passed, 3 failed, versus 230 passed, 3 failed on HEAD. The 3 failures are identical
before and after the patch, and the 5 extra tests are the new `tests/test_fly_steer.py`. With both
steerings forced to `fly`, 10 of the 14 tests in `test_follow`, `test_mission` and `test_scenarios` pass:
* Same as PID: the walk-and-turn distance assert (1.10 m against the test's 1.3 m, while
  `follow.distance_m` is 1.0) and the floor-bottle descent.
* New: the orbit test ended 56 deg off the person's back, where the test allows 45 deg. The fly's yaw and
  your facing-based lateral orbit do not yet cooperate.
* `test_full_mission` got past the follow-distance assert that fails with PID, reached ARRIVED, and then
  missed the guidance-direction check by 30 deg, where the test allows 25 deg.
Videos (in fruitfly-training, gitignored):
- recommended setting (both patches applied): `runs/integration/reachglass_find_fly_smooth2_final.mp4`
- smooth2 with deadband 3 / hysteresis 2: `reachglass_followme_fly_smooth2.mp4`
- smooth with deadband 3 / hysteresis 2: `reachglass_followme_fly_smooth.mp4`
- v1: `reachglass_find_fly_trained.mp4` and `reachglass_followme_fly_trained.mp4`
- hand calibration: `reachglass_find_fly.mp4` and `reachglass_followme_fly.mp4`
- your PID: `reachglass_find_pid.mp4`

## Known gaps

* **Stick activity**: with smooth2 at the defaults the stick moves about 0.5 per loop, against 0.3 for
  your PID. That is mostly real steering. Check it on the real drone. Note that `follow_avoid` ignores
  looming while |yaw| > 20.
* **Tail tracking with smooth2**: fast sideways movement lags (zigzag p95 30 deg). If the demo has quick
  sideways walking, use `FLY-YAW_smooth_best.json`.
* **Yaw plant mismatch**: your sim turns at 100 deg/s at rc 100 with tau 0.35 s. The fly was trained on
  39 to 72 deg/s with tau 0.02 to 0.10 s. Measure the real Tello at rc 30 and 60.
* **Approach is slower** than your discrete rotates (about 4 s per mission). Use `approach: {steering: pid}`
  if that matters.
* **Brain pause**: in FOLLOW the brain stops ticking while the person has been lost for more than
  0.35 s, because your search owns yaw then. It resumes on the next detection.
