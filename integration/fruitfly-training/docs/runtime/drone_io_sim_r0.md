# Drone I/O, simulated Tello, R0 measurement tools

Owner: R2. Contract: `flyfollow/runtime/messages.py`. Bus and launcher: [bus_operator_launch.md](bus_operator_launch.md).

| File | What |
|---|---|
| `flyfollow/runtime/tello_io.py` | The only process that talks to the Tello: state, video, rc stream, command queue, safety |
| `flyfollow/runtime/sim_world.py` | Drop-in replacement for tello_io: simulated Tello, room, user, synthetic detections, rendered frames |
| `flyfollow/runtime/drone_common.py` | Shared by both: rc arbitration, safety rules, move parsing, `tello_state` fields |
| `flyfollow/tools/stick_response.py` | R0: stick steps at 30/60/100 per axis, fit gain, dead time, tau; vgx unit and frame check |
| `flyfollow/tools/latency_test.py` | R0: end-to-end video latency from a flashing screen, no human reading |
| `flyfollow/tools/calibrate_camera.py` | R0: checkerboard calibration, writes `configs/camera.json` |
| `flyfollow/tools/update_sim_from_r0.py` | R0 results into `configs/env.yaml` (demo values, train ranges +-30 %), fine-tune verdict |
| `tests/test_runtime_tello.py` | 20 tests, no hardware (fake djitellopy Tello, in-process bus), about 10 s |

Usually you do not start tello_io or sim_world yourself: `launch --dry`, `launch --send` and `launch --sim` do it.

## tello_io

```
.venv/bin/python -m flyfollow.runtime.tello_io                     # dry run (default)
.venv/bin/python -m flyfollow.runtime.tello_io --send              # real commands
  options: --no-video  --ip 192.168.10.1  --min-battery 25  --max-temp 90  --video-lost-s 5  --bus-lost-s 2
           --idle-land-s 0 (land after this long with no accepted rc or command; 0 = off)  --video-warmup-s 3
```

In: `mode`, `rc`, `tello_cmd`, `kill`. Out: `tello_state` (every state packet, about 10 Hz, at least every 0.5 s),
`frame` + FrameRing, `rc_sent`, `tello_ack`, `tello_event` (kill, safety). Messages with `"replayed": true` are ignored.

**Dry run** (no `--send`): connects (`command`), turns the video on (`streamon`), publishes state and frames, and never
sends takeoff, land, emergency, rc or moves. `rc_sent` still goes out at 20 Hz with `dry_run: true` and shows exactly
what would be sent; `tello_ack` answers `ok: true, detail: "dry run: ..."`; `tello_state.sending` is false.

**rc arbitration** (same code in sim_world):
- The current mode comes from `mode` messages (`to`); IDLE until the first one. A mode change drops the stored rc of
  controller and mission, so the new owner starts from hover.
- `rc` is accepted from `src: "operator"` in any mode, or from `RC_OWNER[mode]` when its `mode` field equals the
  current mode. Messages older than `RC_TIMEOUT_S` (0.5 s) on arrival are dropped.
- Every 50 ms: the newest operator rc if under 0.5 s old, else the owner's newest rc if under 0.5 s old, else
  `rc 0 0 0 0`. Values are rounded and clamped to -100..100 and passed to djitellopy `send_rc_control` as final
  integers (no FlyDrones 60 % scaling). rc is streamed only while flying (after a successful takeoff, until land).

**Command queue** (one sender thread does both the queue and the 20 Hz rc stream, so nothing else can interleave):
- `takeoff`: refused if battery < 25 %, temph >= 90 C or (with video) no frame in the last second. One attempt,
  20 s timeout (djitellopy would retry a takeoff 3 times); if the reply is lost but `h` >= 30 cm it counts as flying.
- `move` (`args: {"direction": forward|back|left|right|up|down|cw|ccw, "value": 20..500 cm or 1..360 deg,
  "timeout_s"?}`): rc stops, the move is sent once, the thread waits for "ok" or the timeout (5 s + value / 40),
  then sends `rc 0 0 0 0` and resumes rc. Out-of-range values are rejected, not clamped.
- `stop`: cancels queued commands (acked `ok: false, "cancelled by stop"`), forgets the stored rc, hovers.
- `land`, `emergency` (as `tello_cmd` or `kill`, from anyone, in every mode): drop everything queued (acked
  `"preempted by ..."`), stop the rc stream, and go first. If a blocking command is in flight, `land`/`emergency` is
  sent raw at once instead of waiting. Land retries up to 3 times; if never confirmed, rc stays off so the Tello's own
  15 s no-command auto-land takes over.
- Every command gets a `tello_ack` with `id`, `cmd`, `ok`, `detail`, `elapsed_s`, `dry_run`.

**Safety** (only with `--send` while flying; one land per flight, reason in `tello_state.safety` and `tello_event`):
battery < 25 %; temph >= 90 C; video enabled and no frame for 5 s; no loopback of our own `tello_state` through the
broker for 2 s (bus or broker lost); no state packet for 3 s; optional idle land. On exit (Ctrl+C, SIGTERM, exception,
sender thread crash) while flying it lands. A second Ctrl+C exits at once (node.stop_event); the Tello then auto-lands
15 s after the last rc.

**Video**: PyAV in a thread (`udp://@0.0.0.0:11111`, format h264, nobuffer/low_delay), frames discarded for the first
3 s and until the first key frame, then every 960x720 RGB frame goes into the FrameRing with `t_decoded`. The reader
reopens after stalls. Measured locally with a libx264 stream over UDP: about 34 ms from the last packet to the decoded
frame (one frame, the raw H.264 parser waits for the next frame's start code). No cv2 in this process.

**tello_state extras** beyond the contract: `vgx_mps` etc. (raw / 10), `v_fwd_mps`, `v_right_mps`, `v_up_mps` (signs
flipped: the lag test read forward, right and up as negative vgx, vgy, vgz), `templ_c`, `baro_m`, `flight_time_s`,
`state_ok`, `state_age_s`, `mode`, `busy`, `queued`, `safety`. The vg* unit is dm/s: re-fitting Parth's lag log, each
"forward 50" integrates to 0.36 to 0.40 m of -vgx / 10 (cm/s would give 0.04 m). Body frame vs takeoff-heading frame
is not verified yet (stick_response checks it).

**Troubleshooting** (tello_io prints these): "Did not receive a state packet" or no video = Windows firewall (allow
Python, open inbound UDP 8890 and 11111) or not on the TELLO Wi-Fi; `Errno 48 Address already in use` on 8889, 8890 or
11111 = a stale process: `lsof -nP -iUDP:8890` and kill it (tello_io checks the ports before connecting).
With no drone, the dry run exits with those hints after djitellopy's connect gives up.

## sim_world

```
.venv/bin/python -m flyfollow.runtime.sim_world --scenario follow|find|full [--seed N] [--det synthetic|none]
                                                [--speed 1.0] [--frames] [--auto-takeoff] [--no-script] [--duration S]
```

Same inputs and outputs as tello_io (same arbitration, queue, kill and safety code), `sending: true`, `sim: true`.
- **Drone**: `flyfollow.sim.drone_model.DroneModel` with `configs/env.yaml profiles.demo` (the measured Tello: yaw 55
  deg/s per 100, dead 0.18 s; forward 0.96 m/s per 100, dead 0.47 s, tau 0.45 s; vertical 0.32 m/s), small OU drift,
  pitch during acceleration. Takeoff "ok" after about 8 s at 0.85 to 1.05 m, `forward 50` about 2.2 s, land about 3 s
  (lag-test timings). State has the real sign conventions (forward = negative vgx and pitch, yaw grows clockwise, h
  in 10 cm steps). Battery drains 0.18 %/s flying; temph drifts up on the ground. Collisions with walls, furniture and
  the user are pushed out and counted (`tello_event kind collision`, `sim_truth.collisions`).
- **Room** 7 x 6 x 2.6 m: dining table (5.3, 4.3), desk (0.9, 5.4), counter along x = 7, couch (1.4, 0.5), chair
  by the table, tv on the wall. Objects: bottle on the table, cup on the desk, backpack on the floor.
- **User** 1.72 m (settings `user_height_m`), 0.23 m head, walks 0.5 m/s along a scripted path.
- **det** (`--det synthetic`): ground-truth boxes with the `flyfollow.sim.camera.CameraParams` model (2 px center
  noise, 5 % size noise, 3 % dropout, bursts, `(h / 20 px)^2` detection probability, 40 px for bottle and cup),
  15 Hz, capture-to-box latency 0.30 s (`t_decoded` = capture + 0.25 s). `person` + `person_head` with track_id 1
  (head box from `camera.project`, like PursuitEnv), objects (bottle 11, cup 12, backpack 13), furniture (`dining
  table`, `desk`, `counter`, `couch`, `chair`, `tv`, ids 101 to 106). Objects hidden behind furniture are not
  reported. `--det none` publishes no det (run a detector on `--frames`).
- **Frames** (`--frames`): flat-shaded 960x720 render (floor, walls, furniture, the user with a head, objects) into
  the FrameRing, delayed by the 0.25 s video latency, about 4 ms per frame. A real YOLO may or may not fire on it.
- **sim_truth** (10 Hz): drone pose (`x, y, z, psi_deg, yaw_deg, phase`), user (`x, y, heading_deg, moving`), objects
  and furniture (`x, y, z, size`), collisions, current rc, mode, camera intrinsics. World frame: x, y on the floor,
  z up, psi counter-clockwise from +x.

Scenarios:
- `follow`: drone on the floor at (0.8, 3.0) facing +x, user at (2.6, 3.0) facing away. The user waits until the drone
  has flown 3 s, then walks to (5.2, 3.0), pauses, (5.2, 1.5), pauses, (3.0, 1.7), (2.6, 3.2) and stands.
- `find`: user stands at (1.4, 3.0); drone on the floor at (3.2, 3.0) facing the user, so the bottle on the table is
  behind it. Seeds other than 0 put the bottle on the table, desk or counter and randomize the drone heading.
- `full`: the follow walk, then 4 s standing, then a scripted operator publishes `mode_cmd FIND` with target
  `{"cls": "bottle", "prompt": "water bottle", "height_m": 0.22}` (off with `--no-script`).

`--auto-takeoff` takes off 1 s after start (to run a controller without the mission). `--speed 2` runs sim time twice
as fast as wall time (message times stay wall time). Full system: `launch --sim --scenario full`.

## R0 measurements (plan 4.8 R0)

All tools write JSON into `data/r0/` (gitignored). Run them with tello_io stopped (they bind the Tello ports). After
any of them: `python -m flyfollow.tools.update_sim_from_r0` (see the last section).

### 1. Camera calibration (no flight, 10 min)

```
.venv/bin/python -m flyfollow.tools.calibrate_camera --live --board 9x6 --square-mm 25
.venv/bin/python -m flyfollow.tools.calibrate_camera --frames data/r0/calib_frames_<stamp>   # redo from saved views
.venv/bin/python -m flyfollow.tools.calibrate_camera --synthetic                             # self-test
```
1. Print a 10 x 7 squares chessboard (9x6 inner corners), tape it flat to a board or show it on a screen at 100 %
   zoom; measure one square with a ruler and pass `--square-mm`.
2. Laptop on the TELLO Wi-Fi. The tool sends `command` and `streamon` over raw UDP and reads the video with OpenCV.
3. Hold the Tello (or move the board) 0.4 to 1.2 m away; the window turns the text green when the board is found.
   SPACE captures, `a` auto-captures every second while the board moves, `c` calibrates after 12 or more views (aim
   for 20, with the board in every corner of the image and tilted up to 40 degrees), `q` quits.
4. Out: `configs/camera.json` (`fx, fy, cx, cy, dist, hfov_deg, vfov_deg, dfov_deg, rms_px, n_views`) and a copy in
   `data/r0/`. Good: rms below 0.5 px. Plan 3.4 assumed fx 921, fy 919 (HFOV 55, VFOV 43); this settles it.
5. Feed back: `fx_px` in the sim (update tool) and runtime settings `fx, fy, cx, cy`.

### 2. Video latency (no flight, 2 min)

```
.venv/bin/python -m flyfollow.tools.latency_test [--duration 30]
.venv/bin/python -m flyfollow.tools.latency_test --analyze data/r0/latency_log_<stamp>.npz
```
1. Laptop on the TELLO Wi-Fi, screen brightness at maximum, steady room light.
2. Tello on a table (or held still) 30 to 60 cm from the laptop screen, camera on the middle of the screen, so the
   flashing field fills most of its view.
3. Run it. A full-screen window flips black/white at random 0.35 to 0.9 s intervals for 30 s (with a large ms counter
   on top for a manual phone check); the tool decodes the Tello video with the same PyAV reader as tello_io.
4. Out: `video_latency_s` (median capture-to-decoded latency, half a frame of sampling removed), `p90_s`, range,
   number of edges and the sequence agreement (should be above 0.9). It includes the laptop display's refresh delay
   (up to 17 ms), so it slightly overestimates. The synthetic test recovers 0.3 s and 0.8 s within 20 ms.
5. Feed back: runtime setting `video_latency_s` (the box filter's prediction) and the sim's `video_latency_s`.
   Training randomizes 0.15 to 0.45 s; a value outside means a fine-tune.

### 3. Stick response (one short flight, about 3 min)

```
.venv/bin/python -m flyfollow.tools.stick_response --send [--sticks 30,60,100] [--axes yaw,fb,lr,ud] [--trials 1]
.venv/bin/python -m flyfollow.tools.stick_response --analyze data/r0/stick_log_<stamp>.json   # re-fit, no flight
.venv/bin/python -m flyfollow.tools.stick_response --sim                                      # rehearse on the model
```
1. Clear area of at least 4 x 4 m with a 2 m ceiling; stick 100 forward for 1 s moves 1 m or more. Battery >= 50 %.
2. Tello on the floor in the middle, camera toward the longest free direction. Spotter ready.
3. The tool shows the plan, then asks you to type `FLY`. During the flight type `l` + Enter to land, `e` + Enter for
   emergency, Ctrl+C lands.
4. It takes off, hovers 3 s, then for each axis (yaw, fb, lr, ud) and stick (30, 60, 100) sends + then - for 1.5 / 1.2
   / 1.0 s with 2.5 s settles (it skips "down" below 60 cm), while rc streams at 20 Hz and the 10 Hz state is logged.
   Then the unit and frame check: forward 50, back 50, cw 90, forward 50, back 50, ccw 90. Then it lands.
5. Out: `stick_log_<stamp>.json` (raw, same state columns as the lag test) and `stick_response_<stamp>.json`: per axis
   and stick the gain per 100 stick, dead time and tau (median over the + and - steps), a linear gain over sticks
   <= 60 (what the linear sim uses), the ratio of each stick's gain to stick 30's (`NONLINEAR` if outside 0.8 to 1.25),
   the unit check (integrated -vgx / 10 over each 50 cm move, 0.5 m = dm/s; after cw 90 a body-frame vgx still reads
   the move) and `sim_update` for profiles.demo.
6. The fit is Parth's analyze_lag method (grid over dead time and tau, least-squares gain), vectorized, with a
   closed-form yaw-angle integral. `--analyze` also reads the lag test's own log and reproduces its numbers (yaw 55.2
   deg/s dead 0.17 s; forward 0.95 m/s dead 0.46 s tau 0.45 s; vertical 0.33 m/s dead 0.41 s). Limits: 10 Hz state and
   0.1 m/s steps; vertical speed at stick 30 is about 1 dm/s, so the vertical gain is good to about 20 % only.

### 4. Back into the simulator and training

```
.venv/bin/python -m flyfollow.tools.update_sim_from_r0            # prints the env.yaml diff and the verdict
.venv/bin/python -m flyfollow.tools.update_sim_from_r0 --write    # applies it
  options: --stick FILE --latency FILE --camera FILE (defaults: newest in data/r0, configs/camera.json)
           --env configs/env.yaml  --no-train (demo profile only)
```
- Sets `profiles.demo` to the measured values (sim_world uses them at once).
- Recenters `profiles.train` ranges on them (plan Section 0): +-30 %, fx +-10 %, at least +-0.05 s on dead times and
  +-0.04 s on taus (10 Hz state), gain ranges widened to cover every per-stick gain when the stick map is nonlinear.
  Line-based edit: comments survive.
- Prints, per value, `inside`, `marginal` (last 10 % of the range) or `OUTSIDE` against the CURRENT train ranges, which
  are what the running Modal job trains on. Any OUTSIDE: `MODAL FINE-TUNE NEEDED: YES`, i.e. a 1 to 2 h fine-tune from
  the best checkpoint with the recentered ranges while R1/R2 continue with PID (plan 4.8 R0). Then commit the env.yaml
  change so the fine-tune uses it.
- Also prints the runtime settings to send (`video_latency_s`, `fx`).

## Tests

`.venv/bin/python -m pytest tests/test_runtime_tello.py -q`: arbitration (owner, operator, mode mismatch, stale,
timeout hover, mode change), dry run never sends, 20 Hz stream and clamping, operator wins, a move stops rc and ends with
`rc 0 0 0 0` before rc resumes, kill preempts the queue and an in-flight move, battery and video-loss land, no takeoff
without video, exit while flying lands, replayed commands ignored, sim takeoff height and timing, yaw and forward
response and signs, sim messages valid (`messages.validate`) with person/head boxes, sim move and kill, sim frames in
the ring, stick fit recovers known parameters, latency analysis recovers known latencies, the env.yaml update keeps
comments, synthetic checkerboard calibration, PyAV reader on a local UDP H.264 stream.
