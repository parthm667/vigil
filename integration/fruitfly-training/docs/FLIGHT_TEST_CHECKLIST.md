# Flight test checklist: the fruit fly steers the Tello

Goal: go from "code is ready" to a real video of the Tello following a person, with the fruit fly connectome
(`FLY-YAW`, trained) deciding the yaw stick inside the ReachGlass stack. Work from top to bottom. Every step
has a PASS line. Do not move on after a FAIL: use the fix given with the step, or the table in Section 6.
To fly from this repo alone (own YOLO person detector, no ReachGlass), use `docs/STANDALONE.md` instead.

Shorthand used below (set it in every terminal):
```bash
export FT=~/Documents/GitHub/fruitfly-training RG=~/Documents/GitHub/jerkgt13
```

## 0. What you need

| Item | Notes |
|---|---|
| Tello (standard) with prop guards | Takeoff from the floor: altitude limits are measured from the takeoff surface. |
| 3 or more charged batteries (4 is better) | About 8 short flights in total. Start a run at 50 % or more, land at 30 %. Charge while you work. |
| Laptop, Python 3.12 | The TELLO-xxxxxx Wi-Fi has **no internet**. Install, pull and download everything in Section 1 BEFORE you join it. |
| 3 people | **Wearer** (the person followed), **operator** (laptop), **spotter** (watches only the drone, calls "LAND"). |
| Space | At least 6 x 4 m clear, ceiling above 2.5 m (follow height 2.0 m, cap 2.3 m). Textured floor (the Tello holds position with a downward camera). Outdoors: no wind. |
| Tape measure, floor tape | Marks at 1.0 m and 1.6 m behind the wearer's start line: independent range check. |
| Phone on a tripod | Video of the drone and wearer, from the side. |
| Stack of cardboard boxes, about 2 m tall | Section 4 only. |

**the second Mac:** an unrelated process (`gs-stand --fake-arduino --port 8889`) holds UDP 8889, the Tello control
port. Before any Tello tool run `lsof -nP -iUDP:8889`. PASS: no output. If `gs-stand` shows up, stop it in its
own terminal (Ctrl+C) or `kill <PID>`. Or fly from another laptop.

## 1. Setup, once, on normal internet (about 30 min)

**1.0 One command (recommended; macOS, Windows or Linux).** With Python 3.12 and git installed:
```bash
git clone https://github.com/parthm667/fruitfly-training ~/Documents/GitHub/fruitfly-training
cd ~/Documents/GitHub/fruitfly-training
python scripts/setup_flight.py --viz          # add --rg <path> if your ReachGlass checkout is elsewhere
```
It clones ReachGlass if needed, applies both patches on a local branch `fly-demo` (never push it), builds one venv in
the ReachGlass checkout with their requirements plus the fly, downloads their models, writes the fly block into
`site.yaml` plus `site_pid.yaml`, fetches the viewer assets and runs the preflight. The trained fly, the brain and its
calibration come with the git clone (`data/brains/`). Afterwards use the ReachGlass venv's python for every command in
this checklist (`$RG/.venv/bin/python`, on Windows `$RG\.venv\Scripts\python.exe`). Then skip to 1.3 (only to set
the wearer height if you know it) and Section 2. Steps 1.1 and 1.2 below are the same thing by hand.

**Preflight, any time (at the drone too):** `python scripts/preflight.py` (from this repo, with the ReachGlass venv's
python). It checks the Tello ports, CPU load, brain speed, the steering sign, the config and the Wi-Fi, and ends with
READY TO FLY or NOT READY. Ports and Wi-Fi only pass once you are on the TELLO network.


**1.1 fruitfly-training**
```bash
cd $FT && git pull
scripts/setup_env.sh          # venv with dev + viz extras
scripts/setup_data.sh         # brains + MaleCNS annotations, ~10 min, 1.2 GB (already done on the second Mac)
scripts/setup_viz.sh          # fly body model, 140 MB
ls data/brains/trained/FLY-YAW_smooth_best.json    # the trained fly (a copy of data/runs/FLY-YAW_s3_v1/best.json)
```
On a laptop other than the teammate's, `data/` is not in git: after `setup_data.sh`, copy the teammate's whole `data/brains/`
folder (under 1 MB, it holds the calibration the trained file was made with) over yours. Never run
`flyfollow.rl.calibrate` on the demo laptop: it changes what the trained parameters mean.

**1.2 ReachGlass with both patches** (skip the clone and venv lines if the team already has them)
```bash
git clone https://github.com/nathanwuzhao/jerkgt13 $RG && cd $RG
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/download_models.py                     # YOLO, pose, World, depth: needs internet
git apply --check $FT/docs/integration/reachglass_flysteer.patch && git apply $FT/docs/integration/reachglass_flysteer.patch
git apply --check $FT/docs/integration/reachglass_follow_avoid.patch && git apply $FT/docs/integration/reachglass_follow_avoid.patch
pip install -e $FT/third_party/FlyDrones -e "$FT[viz]"
```
Order matters: flysteer first. If a `--check` fails, their main has moved past `2302928` (the commit the patches
were made on): `git stash`, `git checkout -b fly-demo 2302928`, apply again, and tell the teammate.

**1.3 Config.** In `$RG/site.yaml`, edit the existing `perception:` block (do NOT add a second `perception:` key:
YAML silently keeps only the last one). Append this at the end, with your absolute path:
```yaml
# --- fruit fly steering (docs/FLIGHT_TEST_CHECKLIST.md in fruitfly-training)
follow:
  steering: fly            # pid = their yaw law. This one key switches the fly off.
  distance_m: 1.6          # their default 1.0 sees only the head; 1.6 keeps shoulders in view (fly trained at 1.5 to 2.5 m)
  altitude_m: 2.0          # must be >= wearer height + 0.2
  avoid: {enabled: false}  # turned on only in Section 4
fly:
  params_path: /Users/user/Documents/GitHub/fruitfly-training/data/brains/trained/FLY-YAW_smooth_best.json
  smoothing_ms: 0          # no low-pass: every low-pass in the sim sweep added lag and cost accuracy
  deadband: 4              # stick units
  hysteresis: 3            # only move the sent stick when the new value is more than 3 away
  viz: true                # fly body + brain window
safety: {max_altitude_m: 2.3}   # keep below your ceiling
```
Then make the PID baseline copy: `sed 's/steering: fly/steering: pid/' site.yaml > site_pid.yaml`.

**1.4 Tests and smoke runs**
```bash
cd $FT && .venv/bin/python -m pytest -q                            # ours, 252 tests
cd $RG && python -m pytest -q tests/test_fly_steer.py tests/test_looming.py tests/test_follow_avoid.py
python -m pytest -q                                                 # their full suite, ~4 min
python -m reachglass sim --config site.yaml -v --seconds 40         # simulated follow with the trained fly
```
PASS: ours all green. Theirs: only the 3 known failures (`test_follows_walking_person_through_a_turn...`,
`test_full_mission_follow_find_approach_guide_follow_land`, `test_target_on_the_floor_needs_descent`). The sim
prints `reachglass.fly: fly steering loaded: {'arm': 'FLY-YAW', ...}` and reaches FOLLOW with 0 collisions.
If `test_steer.py::test_timing_budget_and_never_catches_up` fails with `tick_ms_mean` above 15, the laptop is
too busy for the brain's 50 ms tick: quit other heavy apps (sims, Modal, video calls, browser) and rerun. Do not
fly until it passes on the flying laptop.

**1.5 Bench sign check of the trained fly** (10 s, no drone):
```bash
cd $FT && .venv/bin/python -c "
import math; from flyfollow.steer import FlySteer
s = FlySteer(params_path='data/brains/trained/FLY-YAW_smooth_best.json')
for b in (15, 0, -15):
    s.reset(z_ref_m=1.6, kind='follow'); y = [s.yaw(0.05*k, math.radians(b), t_frame=0.05*k) for k in range(60)]
    print(f'bearing {b:+d} deg -> yaw stick {sum(y[-20:])/20:+.1f}')"
```
PASS: +15 deg gives about +30, 0 gives about 0, -15 gives about -28 (+ = person right = clockwise).

**1.6 Viz offline:** `cd $FT && .venv/bin/python -m flyfollow.viz.demo --seed 1000 --kind follow` opens a Rerun
window with the fly body, the brain and the traces. PASS: it animates. Close it.

## 2. Ground checks at the site (10 to 15 min)

**2.0 Connect.** Power the Tello, join TELLO-xxxxxx. Run `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111`. PASS: no
output. On macOS, click "Allow" if asked whether Python may accept incoming connections. Only one program may
talk to the Tello at a time: quit each tool before starting the next.

**2.1 Link and video latency** (no flight, 3 min)
```bash
cd $RG && python tools/tello_latency_test.py --ground                  # command delay, state and video rates
cd $FT && .venv/bin/python -m flyfollow.tools.latency_test --duration 30   # Tello 30 to 60 cm from the screen
```
PASS: state about 10 Hz, video about 30 fps, `video_latency_s` between 0.15 and 0.45 s with sequence agreement
above 0.9. Put the measured value plus 0.05 into `site.yaml` as `drone: {video_lag_s: ...}` (the fly's latency
filter uses it). Above 0.45 s: move the laptop closer, away from other 2.4 GHz networks, and remeasure.

**2.2 Focal length** (5 min). Their method (README section 3, step 2): the blue bottle at a taped 2.0 m, then
1.5 m and 3 m: `python -m reachglass.tools.calibrate_camera tello --height 0.19 --distance 2.0 --auto`.
Or ours, with a printed 9x6 chessboard: `.venv/bin/python -m flyfollow.tools.calibrate_camera --live --board 9x6 --square-mm 25`
(also measures the FOV, still unverified in `RUNBOOK.md` Section 9). PASS: the three medians agree within 5 %.
Paste the median into `camera: {fx: ..., fy: ...}`. The bearing the fly sees scales with fx.

**2.3 Wearer height**, with shoes, to +-2 cm: in the existing `perception:` block set `person_height_m` and
`person_height_sd_m: 0.02`. Over 1.80 m, raise `follow.altitude_m` to height + 0.2 (still below 2.3). Skip their
step 3 (bottle colour): it is only for "find my water bottle".

**2.4 DRY-RUN SIGN CHECK for the fly** (the key check, 5 min). The drone sends no motion commands.
```bash
cd $FT && .venv/bin/python -m flyfollow.viz.live --brain data/brains/pursuit_core1.npz           # terminal 1
cd $RG && source .venv/bin/activate && python -m reachglass tello --dry-run --no-takeoff --config site.yaml -v   # terminal 2
```
A teammate holds the drone about 2 m high, 1.6 m behind the wearer, camera on them. Wearer: 5 s centered,
step 1 m right for 5 s, back to center, 1 m left for 5 s. The drone does not turn in a dry run, so the stick stays
up while the wearer stays off center: that is expected.

| Look at | PASS | If not |
|---|---|---|
| Terminal 2 at start | `reachglass.fly: fly steering loaded: {'arm': 'FLY-YAW' ...` | "fly steering unavailable": wrong `params_path` or flyfollow not installed in their venv (1.2) |
| Dashboard status, wearer right | `bearing +..` (positive) | Negative: a ReachGlass camera or bearing bug. Do not fly; tell their team |
| Fly window: yaw stick trace | wearer right: positive (+20 to +40); left: negative | Opposite sign: stop, recheck 1.5 and the patch version |
| Wearer centered and still | stick within +-5, mostly 0 | Jitter: `fly.deadband: 5` + `fly.hysteresis: 4`, rerun 2.4 (avoid big `smoothing_ms`: low-pass adds lag and oscillates) |
| Brain panel, wearer right | LC10a right row lights, DNa02 R fires, DNa02 L goes quiet, body banks right (the push-pull) | Viz only, not a flight blocker |
| Range on the dashboard at the 1.6 m tape mark | 1.45 to 1.75 m | Fix height (2.3) or fx (2.2) |
| Person box | on every frame, few drops | Flaky: try `follow.distance_m: 2.0` |

If the fly window will not start, set `fly.viz: false`: the bench check (1.5) plus a positive dashboard bearing
for "wearer right" cover the sign chain. Quit with `q`.

**2.5 Optional short calibration flights** (about 2 min each, if batteries allow)
- Their lag flight: `python tools/tello_latency_test.py` (3 x 3 m, battery 30 % or more). Note whether telemetry
  yaw grows with `cw`, and the pitch sign when it speeds up forward. Negative pitch: `perception.pitch_sign: 1`.
- Our stick response (forward gain was only measured at stick 30):
  `.venv/bin/python -m flyfollow.tools.stick_response --send --axes yaw,fb --sticks 30,60,100`, type `FLY`
  (4 x 4 m, battery 50 % or more; `l` + Enter lands). Then `.venv/bin/python -m flyfollow.tools.update_sim_from_r0`
  (no `--write` at the field). PASS: `MODAL FINE-TUNE NEEDED: NO` and the unit check reads dm/s. YES: keep going with
  the flights (see the last row of Section 6) and send the teammate the output.

## 3. First flights in the clear area

**Rules for every flight.** Click the dashboard window once so it has keyboard focus. Keys: `t` = takeoff,
SPACE = hold/resume, `l` = land, `e` = EMERGENCY motor stop (the drone drops: only if a hit is imminent),
`q` = land and quit, Ctrl+C = land. The spotter calls "LAND" if the drone comes within 1 m of anyone, drifts toward
a wall or the ceiling, the battery reads 30 %, or anything looks wrong; the operator presses `l` at once. Nobody
stands under the drone. The wearer never walks backward toward it.

**Record every run:** add `--log runs/<name>.jsonl --record runs/<name>.mp4`, start a screen capture
(Cmd+Shift+5) of the dashboard and the fly window, and film the drone with the phone. Say the run name aloud at the
start of the phone video.

**Keep other people out of the camera view** when FOLLOW starts and during the run: ReachGlass locks the LARGEST
person box, and from 2 m up the wearer ahead shows little more than a head, so a closer-looking bystander (operator,
spotter) can take the lock. Operator and spotter stand behind or well to the side of the drone. If the dashboard
shows the box on the wrong person, press `l` (a fix is drafted in `docs/integration/wip/`, not applied).

**Wearer script (same every run):** stand still with your back to the drone for 30 s; cover your head with both
hands for 1 s (occlusion), then drop them; walk slowly (about 0.5 m/s) 8 to 10 m straight; turn 90 deg; walk 3 m;
stop for 5 s. Then the operator lands.

**Order:**
1. **PID baseline:** `python -m reachglass tello --config site_pid.yaml -v --log runs/r3_pid_1.jsonl --record runs/r3_pid_1.mp4`.
   Wearer stands at the 1.6 m mark with their back to the drone. Press `t`: it climbs to 2.0 m and enters FOLLOW by
   itself. Run the script. Repeat for 2 to 3 runs.
2. **Fly steering:** same, with `--config site.yaml` and names `r3_fly_N`, fly window open (terminal 1 of 2.4).
   Confirm the "fly steering loaded" line before pressing `t`. 2 to 3 runs.

**After each run**, score the log with `python scripts/score_flight.py $RG/runs/r3_fly_1.jsonl --distance 1.6` (several logs at once compare runs), or the inline version below (the range is the stack's own estimate; the tape marks and the phone video are
the independent check):
```bash
python - runs/r3_fly_1.jsonl 1.6 <<'EOF'
import json, sys
path, Z = sys.argv[1], float(sys.argv[2])            # run log, follow.distance_m you flew with
rows = [r for r in map(json.loads, open(path)) if r.get("state") == "FOLLOW" and "event" not in r]
seen = [r for r in rows if r.get("person")]
rng = [r["person"]["range"] for r in seen if r["person"].get("range") is not None]
gaps = [b["t"] - a["t"] for a, b in zip(seen, seen[1:])]
print(f"FOLLOW {rows[-1]['t'] - rows[0]['t']:.0f} s | person in view {len(seen) / len(rows):.0%} | "
      f"in band (+-15 %) {sum(abs(x - Z) <= 0.15 * Z for x in rng) / max(len(rng), 1):.0%} | "
      f"min range {min(rng, default=float('nan')):.2f} m | longest loss {max(gaps, default=0):.1f} s")
EOF
```

**Pass criteria** (plan R3 and R4, `DRONE_RL_PLAN.md` 4.8):

| Part | PASS |
|---|---|
| Standing 30 s (R3) | Range within +-15 % of 1.6 m (1.36 to 1.84 m); no governor hover or land lines in the terminal |
| Occlusion (R3) | Re-locks within about 1 s after the hands drop, no search turn |
| Walk, turn, stop (R4) | No loss longer than 2 s; in band at least 70 % of FOLLOW time |
| Every run | Never closer than 1.0 m to the wearer (spotter and tape agree) |

Fly passes: go to Section 5 (or 4 first). Fly unsafe or clearly worse than PID: see Section 6. The fallback
for the flight is `steering: pid` (one key).

## 4. Avoidance test (optional, only after Section 3 passes)

The drone follows at 2.0 m, so a chair or table is flown over and is not an obstacle (their rule: it must reach
within 0.2 m of flight height). Use a soft, tall obstacle: cardboard boxes stacked to about 2 m, with tape or
newspaper on the face (looming needs texture). Make `site_avoid.yaml` = `site.yaml` with `avoid: {enabled: true}`.

Place the stack on the wearer's walk line, 2 m ahead of the drone's start. The wearer walks past it on one side
(0.5 m from it) so the drone's straight path goes through the stack. Expected: the status line shows
`| looming patch ... braking` or an obstacle brake, forward stops about 1 m before the stack, and after 1.5 s it
sidesteps 0.6 m toward the wearer's side, then keeps following. Known gap: if the wearer vanishes behind the stack
the drone holds and says "I can't see you"; the wearer steps back into view.
PASS: no contact, stops at least 0.5 m from the stack, follows again once the wearer is visible.
If it brakes with nothing ahead, sidesteps toward a wall, or oscillates: press `l`, then set
`follow.avoid.enabled: false` (exactly the old follow). Middle ground: `follow.avoid.looming: false` (map only).

## 5. Recording the demo video

- **Screen:** ReachGlass dashboard on the left half, fly window (brain, body, traces) on the right half. Nothing
  else on screen, Do Not Disturb on, brightness at maximum. Keep the fly body's "Animated from the fly brain's live
  motor outputs; not a physics simulation" label visible.
- **Capture:** Cmd+Shift+5, "Record Entire Screen", plus `--record runs/demo_takeN.mp4 --log runs/demo_takeN.jsonl`.
- **Phone:** landscape, 1080p60 or 4K, tripod at the side 4 to 6 m away, framing the wearer's full walk; lock
  exposure and focus.
- **Sync:** at the start of each take the wearer claps once in view of both the Tello camera and the phone.
- **Takes:** 2 to 3, each on a fresh battery (90 % or more), `--config site.yaml` (fly steering). Same wearer
  script as Section 3, 60 to 90 s. Check "fly steering loaded" and live traces before each takeoff.
- **After:** put the phone video, the screen recording and `runs/demo_take*` in one folder, note the best take, and
  send it to the teammate. The edit is phone video on the left, screen capture on the right, synced on the clap.

## 6. If something goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| "Did not receive a state packet", no reply, or `Errno 48 Address already in use` | Another program holds UDP 8889 / 8890 / 11111 (`gs-stand` on the second Mac, or a stale tool); not on the TELLO Wi-Fi; firewall | `lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111`, kill the holder; rejoin TELLO-xxxxxx; macOS: allow Python in the firewall; Windows: allow inbound UDP 8890 and 11111 |
| No video or black frames | Decoder warm-up (about 3 s), low battery, hot Tello | Wait 5 s; power-cycle the Tello; let it cool between runs (it heats up on the ground with video on) |
| Dashboard trails the real wearer, drone overshoots | Video lag above the configured `drone.video_lag_s` | Laptop within 5 m, fewer nearby networks; remeasure (2.1) and update `video_lag_s` |
| Drone turns the wrong way (wearer right, drone turns left) | Sign error | Press `l` now. Fly `site_pid.yaml`: if PID is also wrong, the bug is in ReachGlass or the camera, not the fly. Redo 1.5 and 2.4 |
| Yaw jitter or twitching while the wearer is still | Spiking noise; slow brain tick | `fly.deadband: 5` + `fly.hysteresis: 4`, or `fly.smoothing_ms: 40` (larger low-pass values add lag and make it oscillate); quit other apps (tick mean must stay under 15 ms) |
| Slow side-to-side swing that grows | Latency under-compensated, or yaw too strong | Raise `drone.video_lag_s` by 0.05; or `fly.max_rc_yaw: 30` |
| Drone lands on its own | Battery under 20 % or 420 s flight (their governor, reason printed); laptop link lost, so the Tello auto-lands about 15 s after its last command (Wi-Fi drop, frozen or crashed process) | Read the terminal line; fresh battery; laptop close and awake. A lost person does not land it: it holds, searches, and after 20 s goes to REACQUIRE / HOLD |
| Fly steering worse than PID (more losses, below 70 % in band) | Sim-to-real gap | `follow: {steering: pid}` (use `site_pid.yaml`), finish the demo with PID, keep the logs for us |
| Measured dynamics far from the sim (2.5 says OUTSIDE, or latency above 0.45 s) | Trained on the wrong plant | `update_sim_from_r0` output plus the `data/r0/` JSON files to the teammate: a fine-tune on Modal (about $20, 1 to 2 h) with `--write` applied, then redo 1.5, 2.4 and Section 3 |
