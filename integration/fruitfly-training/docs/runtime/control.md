# Control loop (`flyfollow.runtime.controller_runner`) and geometry (`flyfollow.runtime.geometry`)

The process that turns detections into rc sticks in FOLLOW, APPROACH and GUIDE (plan 4.1, 4.9, 5.3, 5.5), and the
shared camera geometry the mission logic and guidance also use. Message contract: `flyfollow/runtime/messages.py`.

```
python -m flyfollow.runtime.controller_runner [--controller fly|pid] [--params data/runs/<run>/best.json] [--viz] [--hz 20]
    [--auto-fallback] [--brain data/brains/pursuit_core1.npz] [--mode FOLLOW] [--viz-address tcp://127.0.0.1:5557]
    [--viz-spawn] [--no-reacquire-single] [--duration S] [-v]
```

`--mode` only sets the starting mode for bench tests; in a real session the mission's `mode` messages set it.
Bus addresses come from the environment or `messages.py` (never passed), so `launch --port` can redirect a session.

## 1. The loop

In: `det`, `tello_state`, `settings`, `lock`, `target`, `mode`, `kill`. Out: `rc` (20 Hz, only as rc owner),
`ctrl_status` (10 Hz, always), viz frames (with `--viz`).

```
det of the selected target ──> BoxFilter.update(t, t_decoded, cx, cy, h)      (on arrival)
every 50 ms:
  box      = BoxFilter.output(now)                    latency compensated: predicts from capture (t_decoded - video_latency_s) to now
  settings = interfaces.Settings for the mode         (table below)
  yaw, fb  = controller.act(box, settings, 0.05)      FOLLOW / APPROACH; GUIDE uses the PID yaw law and fb = 0
  go       = PersonGovernor.filter(yaw, fb, box, settings, dt, alt_m, brain_age_s, battery_pct)
  publish rc {lr, fb, ud, yaw, src "controller", mode, gov {safety, clamped, reasons}, brain_tick_ms, controller, law}
```

The same `BoxFilter`, `PersonGovernor`, `PIDController` and `make_controller` objects as in training, with
`configs/env.yaml` `filter` and `governor` settings. The loop waits on the subscriber between ticks (10 ms slices),
never blocks on a send (ZeroMQ NOBLOCK, VizSink drops), and skips ahead after an overrun instead of bursting.

### Per mode

| Mode | rc | Controller | Settings | Notes |
|---|---|---|---|---|
| FOLLOW | yes | fly: FLY-YAW (the fly steers, PID-HAND's range loop sets forward); pid: PID-HAND | kind follow, z_ref = `follow_distance_m`, size = `head_size_m`, side offset = `side_offset_deg`, z_min = `min_person_dist_m`, cy_ref 0.55 | target message may override z_ref_m, size_m, z_min_m, cy_ref_frac |
| APPROACH | yes | same controller | kind approach, z_ref = `target.z_ref_m`, else the plan 5.3 rule from the box (3.0 x (max(0.8, center) - base), 1.0 to 2.4 m), size = `target.size_m` else class prior, z_min = `target.z_min_m` else 0.5 m, offset 0 | governor altitude floor 0.8 m |
| GUIDE | yes | PID yaw law only (plan 5.5), fb = 0, brain idle | kind follow, offset 0, z_min = `min_person_dist_m` | the governor backs off (fb -40) inside 1.2 m. ud holds altitude while the head stays in the middle 25 to 75 % of rows |
| FIND, FACE_PERSON, OVERWATCH, RETURN, HOLD | no (mission) | brain idle | | the user (or the target message's object) is still tracked and reported in ctrl_status |
| LAND, IDLE | no | idle | | |

- Kill: after a `kill` message no rc is sent until the next `mode` message.
- Land: the loop never lands. When the governor says land (person lost > 10 s in FOLLOW or GUIDE, or battery < 25 %),
  the rc is hover (0 0 0 0) and `ctrl_status.land_request` is true with `land_reason` "lost_land" or "battery"
  (published immediately, then every status). The mission latches it and owns LAND.
- Entering an rc-owned mode from a mission mode resets the governor (the sticks ramp up from hover through the slew
  limits). Entering FOLLOW or APPROACH from a non-brain mode resets and warms the controller (1 s of rest input).
- Mode changes to a different target (person vs object, another class or track) reset the box filter; FOLLOW <-> GUIDE
  keeps it.

### Target selection (robust to an unknown detector)

Person (FOLLOW, GUIDE and the user-tracking mission modes):
1. Candidates: every `person_head` box, and every `person` box without a head inside its top 40 % turned into a head
   box (`geometry.head_box_from_person`, flagged `person_top`). A head without a track_id inherits the person's.
2. The box with the locked track_id (`lock`), or the track the loop is following, heads first, then confidence.
3. Else association by position and size against the filter's prediction (or the last box, up to 2 s): center within
   2.5 box heights (+2 per second of gap), size ratio 0.5 to 2. The followed track_id then updates (the lock stays).
4. Else, only if exactly one person is in view: take them when nothing is locked, or when the lock has been unseen for
   3 s (`--no-reacquire-single` turns the second case off). Two or more unlocked people: follow nobody.
`lock` with track_id null unlocks.

Object (APPROACH, or any mode given an object target): boxes of the target class (`geometry.canonical_class`, so
"water bottle" matches "bottle"), then the target track_id, then association as above (while tracking it never jumps to
another box of the same class), then the box nearest the expected bearing within 20 degrees, then the most confident.

Dets are scaled to 960x720 if `img_w`/`img_h` differ. Arrival time is the det message's `t` (same laptop clock),
so `latency_s` is not needed; `t_decoded` later than `t` is clamped.

## 2. Biology versus engineering in the live loop

| Biology (fixed MaleCNS connectome, LIF) | Engineering |
|---|---|
| The yaw command: LC10a encoder input spreads through the 1,446-neuron pursuit core and the steering DN rates set the yaw stick (FLY-YAW) | Forward speed: PID-HAND's range loop (Kp_f 40 per m, d0 0.15 m) from the head size, not the fly (RUNBOOK 7: the full fly failed on range) |
| | The camera-to-LC10a encoder and the DN readout (trained adapters, 34 parameters) |
| | The latency filter, target selection, up/down, lost target, minimum distance, clamps, slew limits, watchdog, battery (governor) |
| | GUIDE yaw hold (PID law, brain off) and every mission-mode motion |

## 3. Switching fly and PID, and the fallback

- `settings.controller` "fly" or "pid" (also `--controller`). Fly = `make_controller("FLY-YAW", x)` from
  `settings.params_path` / `--params` (a trainer best.json; its "arm" and "brain" are used), or FLY-YAW-HAND (hand
  calibration) when no file is given. Pid = PID-HAND.
- Hot swap on a settings change: built controllers are cached, the new one is reset and warmed up (1 s of rest input,
  about 7 ms), and the governor keeps its slew state, so the sent sticks ramp from where they were (tested: no step
  larger than the slew limits across fly -> pid -> fly).
- If the fly cannot be built (missing brain, calibration or bad params) the loop flies on PID-HAND and warns
  `fly_build_failed`.
- Plan 3.5 rule: every act() is timed. Once the fly has been active 5 s, if the p95 over the last 5 s exceeds 45 ms the
  warning `brain_over_budget` is raised; with `--auto-fallback` the loop switches to PID-HAND (`fallback: true`,
  warning `fallback_pid`). An explicit operator `settings.controller` clears the fallback.
- Watchdog: `brain_age_s` passed to the governor is the act() duration when it succeeds, and the time since the last
  successful act() when it raises. Over 0.5 s (a 500 ms act, or 10 failed ticks) the governor hovers with reason
  `watchdog`. act() exceptions never stop the loop (`act_error` warning).

## 4. ctrl_status (10 Hz)

Required: `mode`, `controller` ("fly" | "pid"), `target_valid`, `range_m`, `bearing_deg`, `in_band`. Also:
`arm`, `law` (fly_yaw+pid_fwd | pid | guide_yaw_hold | idle), `rc_owner`, `killed`, `in_band_s`, `in_band_2s`
(APPROACH success: in the +-15 % band 2 s), `range_sd_m`, `elevation_deg` and `xy_m` (drone_level, attitude corrected),
`z_ref_m`, `z_min_m`, `side_offset_deg`, `target` {kind, cls, lock_id, track_id, src, conf}, `box` {cx, cy, h},
`box_age_s`, `lost_s`, `lost_2s` (plan 5.3 rescan trigger), `det_age_s` (any det), `brain_tick_ms_p50/p95` (10 s),
`tick_work_ms_p95`, `loop_hz`, `gov` {safety, clamped, hover, reasons}, `land_request`, `land_reason`, `fallback`,
`warnings`, `alt_m`, `visible_floor_min_range_m`. Floats are rounded; missing values are null (strict JSON).

`range_m` / `bearing_deg` are from the filtered box: range = fy x size / h, bearing = atan((cx - cx0) / fx) in the
camera frame (what the controllers see). `xy_m` uses the attitude-corrected bearing.

## 5. Viz

With `--viz` every rc-owned tick builds a frame with `flyfollow.viz.frames.frame_from_controller` (spike counts,
encoder channels, input and DN rates, readout drives, raw and sent sticks, target box, bearing and range, meta source
"drone") plus the newest camera frame from the FrameRing (attached lazily, retried every 2 s, dropped if older than
1 s), and pushes it through `VizSink.publish` (non-blocking, downscaled to 480 px, drops when the viewer lags). Run
the viewer with `python -m flyfollow.viz.live --brain data/brains/pursuit_core1.npz`, or add `--viz-spawn`.
Viz errors only raise a `viz_error` warning. In GUIDE the frame has sticks and target only (brain idle).

## 6. Geometry API (`flyfollow.runtime.geometry`, pure functions)

| Function | Returns |
|---|---|
| `Intrinsics(fx, fy, cx, cy)`, `DEFAULT_K`, `load_intrinsics(path=None, overrides=None)`, `camera_defaults()` | configs/camera.json (flat keys or an OpenCV `camera_matrix`, scaled from its width/height) over SETTINGS_DEFAULTS; `.hfov_deg` 55.0, `.vfov_deg` 42.8 by default |
| `bearing_deg(u, k)` | camera bearing, no attitude (what the controllers use) |
| `bearing_elevation(u, v, k, pitch_deg, roll_deg)`, `pixel_ray(...)`, `pixel_of(bearing, elevation, k)` | drone_level direction with pitch (+ nose up) and roll (+ right down) removed |
| `tello_attitude(tello_state)`, `tilt_ok(pitch, roll, 8)` | attitude in that convention; plan 4.9 range-trust check |
| `range_from_size(h_px, size_m, fy)`, `range_from_class(h_px, cls, fy)`, `size_range_sd(z, h, sd_frac)` | pinhole range and its 1-sigma error (prior error and 3 px box noise) |
| `ground_range(u, v_base, cam_height_m, k, pitch, roll)` | range to a floor point (foot or object base) |
| `user_height_range(u, v_head_top, v_feet, user_height_m, k, pitch, roll)` | plan 5.4 (Z, camera height) |
| `height_above_floor(u, v, range_m, cam_height_m, ...)` | height of an image point, used for the APPROACH standoff |
| `floor_visible_min_range(alt_m, k, pitch_deg)` | plan 3.4 blind zone: 1.28 / 2.55 / 3.06 m at 0.5 / 1.0 / 1.2 m |
| `drone_level_xy(bearing, range)`, `xy_to_bearing_range(x, y)`, `wrap_deg(a)` | x forward, y left |
| `CLASS_SIZE_M`, `class_size(cls)`, `class_size_m(cls, default)`, `canonical_class(cls)` | (size_m, sd_frac) priors for people, FIND targets and furniture, with aliases ("water bottle", "phone", "mug", "fridge", ...) |
| `head_box_from_person(bbox, user_height_m, head_size_m)` | head box from a person box (full body: height ratio; feet cut: 0.23 / 0.45 of the width) |
| `approach_z_ref(base_h_m, center_h_m, cfg=None)` | plan 5.3 standoff with our config (wraps `flyfollow.sim.objects.approach_z_ref`) |

## 7. Measured on this Mac (Apple M5, 10 cores), 2026-09-26

| Quantity | Value |
|---|---|
| FLY-YAW-HAND act(), tight loop | 1.8 ms p50, 2.1 ms p95 |
| Build fly controller + warmup 1 s | 0.12 s + 7 ms |
| Real time through the ZeroMQ broker with sim_world, 40 s FOLLOW, fly | brain 3.1 / 6.0 ms p50 / p95, loop 20.0 Hz, rc inter-arrival 50.1 / 52.0 ms p50 / p95 (max 64) |
| Same plus a YOLO11n ONNX 640x480 load at 15 Hz (12.6 ms median per inference) | brain 2.5 / 3.9 ms, loop 20.0 Hz, rc p95 52.8 ms (max 93) |
| Same with YOLO11n 960x736 at 15 Hz (30 ms median) and `--viz` (no viewer attached) | brain 2.7 / 8.8 ms, loop 20.0 Hz, rc p95 54.4 ms (max 168) |
| Closed loop in-process (sim_world follow scenario, seeds 0 and 1, 30 s after takeoff) | user in view 100 % (PID), 97 to 100 % (fly); no collisions; min distance 1.08 to 1.35 m |

The plan's 45 ms budget holds with a wide margin here; the Windows runtime laptop (Intel Core Ultra 7 258V) is
unmeasured. `ctrl_status.brain_tick_ms_p95` and `tick_work_ms_p95` show it live; start with `--auto-fallback` there.

## 8. Tests

`tests/test_runtime_control.py` (41 tests, about 8 s): geometry against hand-computed numbers; runner on an InProcBus
with simulated time: target right -> yaw right (fly and pid), far -> forward, close -> back with the min_distance
safety flag, no rc in mission modes, kill, fly <-> pid hot swap without a jump, GUIDE fb = 0 with the brain idle and
back-off inside 1.2 m, watchdog on a failing and on a 600 ms act(), auto fallback on a faked 60 ms tick, late dets
compensated (error under 10 % of the raw lag), `validate()` and strict JSON on every rc and ctrl_status, head missing,
track_id changes and missing ids, unlocked with two people, lost target -> land request without landing, APPROACH
object selection and standoff, bad settings, unlock, the real-time loop rate, the CLI over a real broker, viz frames
with a FrameRing image, and a closed loop with sim_world (FOLLOW, 30 s, both controllers).

## 9. Open issues

- Tello pitch and roll signs are unverified (`geometry.TELLO_PITCH_SIGN`, `TELLO_ROLL_SIGN`; sim_world assumes forward
  flight gives negative pitch, which matches). Check at R0 by tilting the drone by hand. They only affect `xy_m`,
  `elevation_deg` and the APPROACH standoff estimate, never the controllers (which see raw pixels, as in training).
- `target.bearing_deg` is read as the camera bearing at the message's time and pinned to the Tello yaw then, so the
  expected bearing stays right while the drone turns. The mission must send it in that convention.
- With no `target.z_ref_m` and no altitude, APPROACH uses 1.0 m until the box gives an estimate; the mission should send
  `z_ref_m`. The 0.5 m low-scan altitude floor (plan 5.3) is not supported (governor floor fixed at 0.8 m).
- Following a lone person after 3 s without the locked track is on by default (detectors re-ID often); it can take a
  bystander if the user is out of view. Off with `--no-reacquire-single`.
- Close walk-backs: in sim_world seeds 2 and 3 the user walks back toward the drone faster than the 40-stick reverse
  (0.38 m/s), minimum distance 0.42 to 0.56 m. That is the known no-rear-sensing limit (RUNBOOK 7), not a loop bug.
- The loop runs the brain in-process. If the Windows laptop misses the budget, plan 3.5's next step (brain in its own
  process) is not built; `--auto-fallback` to PID is.
