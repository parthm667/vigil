# Person following from this repo alone (no ReachGlass)

The whole loop runs from this repo: Tello video, a YOLO person detector, the trained fly steering, a PID for
distance, a safety governor and an operator console. The detector is the ReachGlass person detector ported into
`flyfollow/runtime/detector.py`, with the same model and settings (yolo11n-pose, class person, conf 0.4, imgsz 640)
and its tracker. The ReachGlass route (`docs/FLIGHT_TEST_CHECKLIST.md`) still works and is unchanged.

| Piece | What it does |
|---|---|
| `tello_io` | the only process that talks to the Tello; `--dry` never sends anything |
| `detector` | newest frame, then YOLO people, then track ids, then `det` (about 30 frames/s on an M-series Mac, 29 ms per frame) |
| `controller_runner` | locked person, then the trained fly for yaw (`FLY-YAW_smooth_best.json`, deadband 4, hysteresis 3, the same as FlySteer), the PID range loop for forward, and the governor for up/down, clamps, lost target |
| `mission_lite` | modes: takeoff, HOLD, FOLLOW, land |
| `operator` | keys and live status |

## 1. Setup (once, with internet, before joining the Tello Wi-Fi)

```
git pull
python scripts/setup_standalone.py          # add --viz for the live fly body + brain window
```

This builds `.venv` (Python 3.12) and downloads the detector weights to `models/` (6 MB). It then runs the
preflight. The ports and Wi-Fi checks fail until you are at the drone; that is expected. On Windows, replace
`.venv/bin/python` below with `.venv\Scripts\python`.

## 2. At the drone

1. Join the `TELLO-xxxxxx` Wi-Fi, then run `.venv/bin/python scripts/preflight.py --standalone`. Everything
   except "internet" should PASS. A port FAIL means another program holds the Tello port: close it.
2. **Bench test (sends nothing).** Put the drone on a table facing an open area and stand about 2 m in front of it.
   ```
   .venv/bin/python -m flyfollow.runtime.launch --dry
   ```
   Press `t` `t` (a pretend takeoff), then `k` (lock onto you), then `f` (FOLLOW). Step left and right. On the
   console, `RC OUT ... DRY RUN` should show yaw **+** when you are on the drone's right and **-** on its left. The
   yaw is zero when you are slightly right of centre, because the drone is set to trail 8° to the side
   (`side_offset_deg`, change it with `s`). Forward (fb) goes **+** when you step back. `l` ends it, `q` `q` quits.
3. **Fly.** Use an open area, a spotter, and keep bystanders out of view: the lock can jump to another person.
   ```
   .venv/bin/python -m flyfollow.runtime.launch --send        # type "fly" to confirm
   ```
   Press `t` `t` (takeoff, then HOLD), `k` (lock), `f` (FOLLOW). Walk slowly. **Space = emergency motor stop,
   `l` = land.** `c` switches the steering between the fly and the PID mid-flight, to compare them.
   If it loses you it turns slowly to look, hovers after 5 s and lands after 10 s.
4. Every run is recorded in `recordings/<session>/` (bus log, video, process logs).
   `python -m flyfollow.runtime.launch --replay recordings/<session> --detector yolo` replays it through the
   detector and the controller.

Useful options:
- `--viz` shows the live fly body + brain window.
- `--args controller_runner="--params none"` runs the untrained fly.
- `--args controller_runner="--controller pid"` starts on the PID.
- `--args detector="--device cpu"` runs the detector on the CPU.
- Settings (`s` on the console): `follow_distance_m` (2.0), `side_offset_deg` (8), `max_fwd_stick` (60),
  `min_person_dist_m` (1.2).

## What was tested (2026-09-26)

- `tests/test_runtime_detector.py` covers the tracker ids, one `det` per new frame, the yaw shaping, the launcher
  default and YOLO on a real photo. `tests/test_runtime_tello.py` covers the dry run: a dry takeoff pretends to
  fly and never sends a packet. Full suite: 264 passed.
- **Replay, end to end:** video of a person walking left and right, then the launcher with `--detector yolo`, then
  FOLLOW. The person was found in every frame with one track id, at 29 frames/s and 27 ms per frame. The fly's yaw
  followed the person's bearing (correlation 0.96, no lag), with a brain tick of 5 ms p95.
- **Not tested:** the real Tello with this path. The sim renders people as boxes, which YOLO does not see, so
  `--sim` keeps using the synthetic detector.
