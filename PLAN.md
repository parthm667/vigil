# ReachGlass — System Plan v3 (28 h): vision-only scout drone + guided reach

## 0. What changed from v2

| Fact | Consequence |
|---|---|
| **Standard Tello** (not EDU) | Laptop Wi-Fi must join the Tello's own AP; no station mode, no mission pads, SDK 2.0 |
| **Only motion commands + camera footage** from the drone; no telemetry (yaw, height, battery, speed) | All drone state is estimated from **commanded moves + vision**. The Tello executes discrete moves with its own optical-flow position hold, so *commands are our odometry*; a visual compass corrects heading |
| **No MPU6050** | Wearer heading comes from the glasses camera (target, drone beacon, visual yaw) and the drone's view of the wearer |
| **Intel Core Ultra 7 258V / Arc 140V** | Run YOLO through **OpenVINO** on the iGPU (and NPU); PyTorch-CPU is too slow for two streams |
| **Unknown demo room** | Exploration must build its own map online: **SCAN–SCORE–HOP** (§5.3) |
| **28 hours** | Tiered demo (§1), single-process architecture (§3), schedule in §9. Follow-behind and RL are off the critical path |

Rule unchanged: every step has a deterministic baseline that must work; RL is a drop-in behind the same interface and ships only if it passes a go/no-go.

---

## 1. Demo script and tiers

**Tier 0 — the scout (must work, ~12 h):** drone takes off beside the wearer, explores the unknown room with SCAN–SCORE–HOP, finds the bottle, flies to it from the wearer's side, hovers as a beacon, announces "Bottle found, about 4 m ahead, to your right". Dashboard shows the live view, the detections, and the map being built. This alone is a complete, generalizable demo.

**Tier 1 — the reach (should, ~8 h):** glasses camera + servos guide the wearer: turn cues until they face the drone/bottle, walk cues, ToF STOP, ARRIVED. Drone sidesteps and lands when the wearer is within ~2.5 m.

**Tier 2 — polish (if time):** voice command through Grok, follow-behind at the start, obstacle routing around a chair, handoff audio, web dashboard.

**Tier 3 — not this weekend:** RL in the loop, monocular SLAM, reach mode with hand tracking.

Record a video of Tier 0 as soon as it works. Every later tier is added on top of a working, recorded scout.

---

## 2. Constraints and numbers

### Tello (Standard), used as "commands in, video out"
| Item | Value | Use |
|---|---|---|
| Commands we rely on | `takeoff`, `land`, `up/down/left/right/forward/back <20–500 cm>`, `cw/ccw <1–360°>`, `go x y z speed`, `speed <10–100>`, `rc a b c d`, `stop` (hover now), `emergency` | Discrete moves are executed by the drone's own controller on its optical-flow hold: **±10 cm per metre, ±3–5° per turn on a textured floor** *(measure on day 0)* |
| Auto-land | after 15 s without any command | free watchdog; send `rc 0 0 0 0` keep-alives during dwells |
| Takeoff height | ~0.8–1.0 m | then `up 40` → ~1.3 m scan altitude |
| Over furniture | the downward ToF makes it **climb by the table height** when crossing a table | never fly over tables; treat them as obstacles (the free-space estimator does) |
| Video | 960×720 H.264 over Wi-Fi, ~25–30 fps, **100–200 ms lag** | newest-frame-only reader; pick sharp frames during dwells |
| Field of view | ≈ 70° horizontal, ≈ 55° vertical, **f ≈ 680 px** at 960 wide *(measure)* | bearing = `atan((x − 480)/680)` |
| Bottle (22 cm) | 150 px @1 m · 75 @2 m · 50 @3 m · 37 @4 m · 30 @5 m | reliable ≤ 3.5 m at `imgsz=960`; scan radius for planning: **3 m** for bottle-sized, 6 m for chair/backpack-sized |
| Battery | 13 min nominal, ~8 with video; **no reading available** → hard mission timer (5 min) and a fresh battery per run | if your wrapper can send `battery?`, do it before every takeoff |
| Position hold | needs texture + light | test the venue floor first; bring a patterned rug for the takeoff/landing zone |

If your control stack is djitellopy talking to a real Tello, the state stream (yaw, height, battery) arrives on UDP 8890 for free. Nothing below depends on it, but use `battery` for safety and `yaw` as a cross-check if you find it is there.

### Glasses (unchanged except no IMU)
ESP32-CAM 640×480 MJPEG, HFOV ≈ 60°, f ≈ 550 px *(measure)*: bottle 174 px tall @0.7 m, 61 @2 m, 41 @3 m → glasses see the bottle reliably ≤ 2.5 m. A **15 cm neon card on the drone** is 28 px @3 m, 21 px @4 m → colour-blob detection to ~6 m. 2× VL53L1X (4 m, 27° cone). 2× SG90 via the ESP32-C6, which owns hard STOP.

### Laptop: Core Ultra 7 258V, Arc 140V iGPU, NPU
| Model (OpenVINO FP16) | Where | Expected |
|---|---|---|
| YOLO11n @640 | `intel:gpu` | 8–15 ms |
| YOLO11n @960 (search) | `intel:gpu` | 20–35 ms |
| YOLO11n-pose @640 | `intel:npu` or gpu | 15–25 ms |
| Depth-Anything-V2-Small @252–378 px | gpu, or CPU (100–300 ms) | only on the 8 dwell frames per vantage point, so CPU is acceptable |
| HSV beacon blob, ORB visual compass, Farneback flow @160×120 | CPU | < 5 ms each |

Export once: `yolo export model=yolo11n.pt format=openvino half=True imgsz=640` (and 960, and the pose model); load with `YOLO("yolo11n_openvino_model/")` and `device="intel:gpu"`. If your ultralytics version lacks the `intel:` device strings, compile the IR with `openvino.Core().compile_model(..., "GPU")` directly. Confirm the OS (Windows or Linux) — it changes only the Wi-Fi setup and the serial port names.

---

## 3. Architecture (one process)

"Mission controller" just means **the main loop's state machine**: a `state` variable and one function per phase. In a 28-hour build it is `main.py`, not a service.

```
main.py
  threads:  tello_video_reader (newest frame)   glasses_video_reader (newest frame)   c6_serial_reader (ToF, STOP)
  loop @ 10 Hz:
     world = perception.update(frames, ...)      # detections, tracks, bearings, ranges, map, heading error
     state, cmd, cue = mission.step(state, world) # FSM: IDLE → EXPLORE → APPROACH → BEACON → GUIDE → HANDOFF → LAND
     drone.send(cmd)   glasses.send(cue)   dashboard.draw(world, state)   log.write(world, state)
  keys: space = stop/hover   l = land   e = emergency   n = next phase (manual override)
```

One shared `World` object (a dataclass) is the only integration surface between perception, the mission FSM, the glasses I/O and the dashboard. Whoever works on the glasses writes into `world.glasses`; whoever works on the drone reads `world.drone`. Integration "at the end" is then wiring two threads, not merging two programs.

Frames: bearings in degrees (+right, +up), metres, laptop epoch seconds, image coordinates 0–1. Map frame: origin = wearer's feet at takeoff, x = wearer's initial facing direction, y = their left.

---

## 4. Decomposition

| Step | Perception (facts) | Deterministic baseline (ships) | RL slot (optional) |
|---|---|---|---|
| Takeoff & anchoring | — | wearer = map origin; drone placed at their feet facing forward | — |
| Command | detectable-class vocabulary | intent → target class (voice team) | — |
| **Exploration** | detections + bearings/ranges, free-space profile per sector, visual compass Δθ, sharp-frame selection, grid map | **SCAN–SCORE–HOP** frontier + semantic-prior search | `choose_hop(view) → sector`, trained in the 2-D room sim |
| Approach & beacon | target bearing/range, looming, person bearing/range/facing | approach from the wearer's side, visual servo, hover-beacon, sidestep-and-land | — |
| Guide | heading error fused from 5 IMU-free sources, corridor obstacles, ToF zones | cue policy with hysteresis + obstacle offset | `GuidePolicy(obs) → cue`, 2-D sim with a human model |
| Handoff | reach detection | ARRIVED cue, drone lands off the path | — |
| Follow-behind (Tier 2) | person bearing/range/facing | visual-servo PID | — |
| Always | frame latency, confidence | timers, hop watchdog, kill keys, C6 STOP | never |

---

## 5. Step-by-step design

### 5.0 Network (Standard Tello)

- Built-in Wi-Fi → `TELLO-XXXXXX` AP (192.168.10.1; laptop gets .2). Nothing else can use that adapter.
- **ESP32-C6 → USB serial** (no Wi-Fi needed). Recommended regardless.
- **ESP32-CAM:** (a) a **USB Wi-Fi dongle** on the laptop joined to a phone hotspot (2.4 GHz "maximize compatibility" on); the ESP32-CAM joins the same hotspot; the phone also provides internet for STT/Grok. Windows/Linux dongle drivers are painless. (b) Fallback: the ESP32-CAM joins the Tello's open AP (it hands out DHCP addresses; bandwidth untested — try it on day 0). (c) Internet without a dongle: phone **USB tethering** (Android: just works; iPhone on Windows needs the Apple drivers).
- Test all three links **at the same time** in the first hour. This is the most common hackathon killer.

### 5.1 Takeoff and anchoring (and Tier-2 follow)

The drone is placed on the floor **at the wearer's feet, pointing in the direction the wearer faces**. Takeoff → `up 40` → the map origin is the wearer, and map heading 0° is the wearer's initial heading. Two things come free from this:
- The vector origin→target from the map is the wearer's **initial turn instruction** ("bottle at 60° right, 4 m") before any camera confirmation.
- The drone can always find its way back: `go` toward the origin.

The wearer is told "hold still while I look". If the drone later sees the person elsewhere (pose detector), the origin is updated to that estimate.

**Follow-behind (Tier 2):** after takeoff, find the person (yaw scan), back off to 2 m by visual servo on shoulder width, orbit until `facing == away`, hold. It is a visual-servo PID (yaw ← bearing, forward ← range, lateral ← facing error) and needs no telemetry. When the command arrives, the map origin becomes the person's estimated position and exploration starts from the hover point.

### 5.2 Command intake

`{"cmd":"FIND","target":"bottle","request_id":"r7"}` from the voice team; the class must be in `perception.vocabulary()`. Also `CANCEL`, `DESCRIBE` (report the inventory). Keyboard `f bottle` as the fallback during development.

### 5.3 Exploration: SCAN–SCORE–HOP

**Problem:** find one object class in an unknown room, with a forward camera, commanded moves, no depth sensor, no position/heading feedback, within ~120 s of flight.

**Idea:** a frontier-based exploration on a coarse 2-D grid that the drone builds itself, with three sources of "where to look next": how **open** a direction is (monocular free-space), how **unexplored** it is (grid memory), and how **likely** the target is there (semantic co-occurrence: bottles live on tables and counters, backpacks near chairs and beds). Each move is a discrete Tello command, so odometry is the commanded displacement; each scan re-measures heading with a visual compass. Generalizes to any room because the map is built online.

#### State
- Pose `(x, y, θ)` in the map frame, updated from commanded moves; `θ` corrected at every scan by the visual compass.
- Grid `G`: 0.25 m cells, 16×16 m initially (grows), values `unknown / free / occupied`, plus `seen_count` per cell.
- Semantic memory: `[(class, conf, world_xy, range_src, vantage_id, frame_id, t)]`.
- Vantage list: positions visited; hop count; mission clock.

#### One vantage point (≈ 20–25 s)
1. **SCAN.** 8 steps of `cw 45`. After each turn: keep-alive `rc 0 0 0 0`, wait 0.5 s for the drone to settle, grab 3 frames, keep the sharpest (max Laplacian variance). On that frame:
   - **Detector** at `imgsz=960`, all classes. Each detection → `bearing = atan((cx − 480)/f)`, range from the class-height prior (§A), world position from the pose. The target class is what we are looking for; everything else feeds the semantic prior and the obstacle list.
   - **Free-space profile:** split the frame into 5 sectors of 14°. Per sector, `free_dist = median metric depth of the lower-middle band` (Depth-Anything-V2-Small; scale calibrated once against a wall at a measured distance). Without the depth model, use the **floor-line heuristic**: floor = pixels similar in colour/texture to the bottom-centre patch; per sector, the highest floor row `y` → `free_dist = h / tan(atan((y − cy)/f))` with `h = 1.3 m`. Cap at 5 m.
   - **Visual compass:** ORB features matched between consecutive dwell frames (25° overlap); convert matched x-coordinates to bearings and take the median difference → measured `Δθ`. If fewer than 15 matches (blank wall), fall back to the commanded 45°. After step 8, match frame 8 to frame 0 for loop closure and spread the residual over the 8 steps. Typical result: heading known to ±2–3° per scan.
2. **CONFIRM the target.** Confirmed when (a) seen in ≥ 2 dwell frames (adjacent steps overlap, so a target near a frame edge is seen twice) with bearings consistent within ±5°, or (b) one frame with `conf ≥ 0.55` and a plausible pixel size for its range (bbox height within ×0.5–2 of the prior at that range — kills "bottle" hits on door handles). Confirmed → go to APPROACH with `(world_xy, bearing, range)`. A weaker hit (`conf ≥ 0.3`) is a **candidate**: not confirmed, but its direction gets a large score bonus so the next hop goes to verify it.
3. **UPDATE the map.** For each of the 40 sectors: cells along the ray from the pose are `free` up to `free_dist`, the cell at `free_dist` is `occupied` (if < 5 m), and `seen_count += 1` for cells inside the frustum up to `free_dist`. Insert detections into the semantic memory (merge with an existing entry of the same class within 0.7 m).
4. **SCORE the 40 directions.**
   ```
   open      = clamp(free_dist / 4 m, 0, 1)                 # room to fly and to see
   novel     = 1 − seen_fraction of cells along the ray     # unexplored area that way
             × (0 if the ray passes within 1 m of a previous vantage point else 1)
   semantic  = 1 + 2·Σ container_weight(class) for classes seen that way within 5 m
             + 4 if a target candidate lies that way
   people    = 0 if a person is within 2.5 m that way else 1
   score     = open · (0.3 + novel) · semantic · people
   ```
   Container weights by target: bottle/cup → {dining table 1.0, desk 1.0, counter 1.0, refrigerator 0.6, chair 0.3}; backpack → {chair 0.8, bed 0.8, couch 0.6, dining table 0.4}; laptop → {desk 1.0, dining table 1.0, couch 0.5}; anything else → uniform 0.3 for large furniture. This is a 10-line dictionary; extend it for the demo objects.
5. **HOP.** Pick `θ* = argmax score`. If `max score < 0.15` (dead end), backtrack: hop toward the vantage point with the most unexplored neighbours. Turn to `θ*` (`cw/ccw` by the difference), then `forward d` with `d = clamp(0.5 · free_dist(θ*), 0.5 m, 2.0 m)` at `speed 40`. Pose update: `(x, y) += d · (cos θ*, sin θ*)`.
   - **Hop watchdog** (thread): during the move, at 5 fps, if the central free distance < 0.8 m or `looming` (flydrones `Retina` expansion) spikes or a person bbox height exceeds 60% of the frame → send `stop`, mark the cell ahead occupied, re-score. Send moves with `send_command_without_return` so `stop` is never queued behind them.
6. **BUDGETS.** Search ends at 120 s or 5 vantage points or the mission clock (5 min from takeoff) → "I couldn't find the bottle" → return to the origin (`go` in ≤ 2 m legs along free cells) → land.

#### Why it is efficient
A bottle-sized target is visible within ~3 m of a vantage point, so each scan clears ~28 m²; a 6×6 m room is done in 2–3 vantage points. The semantic term sends the first hop toward the table instead of the empty corner, which is where most of the gain comes from in real rooms. The novelty term prevents re-scanning the same area; the openness term keeps the drone in flyable space and out of corners; the candidate bonus turns a weak far detection into a close confirmation instead of a false "found".

#### Worked example (6×5 m room, bottle on a desk in the far corner)
Vantage 1 at the wearer: scan sees a couch left (2 m), a desk far right (4.5 m, bottle 27 px — below the confirmation size, `conf 0.28`, not even a candidate), open floor ahead (4 m). Scores: ahead 0.9 (open, novel), desk direction 0.8×(0.3+0.9)×(1+2) = 2.9 → hop 2 m toward the desk. Vantage 2: bottle at 2.6 m, 58 px, `conf 0.71`, seen in two adjacent frames → confirmed. Total ≈ 45 s.

#### Failure modes and what catches them
| Failure | Catch |
|---|---|
| Heading drift over hops | visual compass per scan; loop closure; map only needs ±0.5 m accuracy because APPROACH is closed-loop on the image |
| Position drift on a bad floor | hop distances ≤ 2 m; re-anchoring on large landmarks (optional §5.3.1); the wearer origin is re-estimated whenever the person is seen |
| Depth model fooled by glass/mirrors/bright windows | hop = 0.5 × free distance; watchdog; treat "free distance > 5 m indoors" as suspicious → cap at 3 m |
| Target false positives (bottle vs cup vs vase) | class-locked, size-plausibility check, two-frame confirmation; **the demo bottle is tall and coloured** |
| Blank walls (no compass features) | fall back to commanded angles; the loop closure still checks the total |
| Doorways (leaving the room) | hop cap 2 m and vantage cap 5; optionally penalise sectors whose free distance is far larger than the room median (a corridor) |

#### 5.3.1 Optional refinement: landmark re-anchoring
Large static classes (couch, dining table, tv, refrigerator, bed) seen from two vantage points give a position fix by triangulation (bearings from two known headings). One least-squares solve per scan; corrects `(x, y)` by the median residual. Only if the floor turns out to be slippery for the Tello's hold; not day-one.

### 5.4 Approach and beacon positioning (no overflight)

Overflying the object is out: crossing a table makes the Tello climb by the table height (§2). Instead:

1. **Plan the approach point** `P = T − 1.3 m · u`, `u = (T − W)/|T − W|`, with `T` the target's map position and `W` the wearer (origin). Approaching **from the wearer's side** puts the drone on the wearer→bottle line, so the beacon bearing equals the bottle bearing from the glasses (δ ≈ 0).
2. **Navigate to P** through free cells (straight line if free; else BFS on the grid, ≤ 2 m legs, hop watchdog on).
3. **Reacquire and servo:** turn to face `T`; the target should be within ±20°. Visual servo with `rc`: yaw ← bearing (deadband 4°), forward ← (range_prior − 1.3 m) (deadband 0.15 m), altitude untouched. Brake on `looming`. If the target is not seen within 5 s → a ±30° local scan → else fall back to the last confirmed map position and hover there.
4. **Beacon hover:** `rc 0 0 0 0`, then yaw 180° (`cw 180`) to face the wearer. Confirm the person with the pose model; refine `W`. Publish `drone_ready` with `(distance |T − W|, initial bearing)` → audio "Bottle found, about 4 metres ahead, to your right."
5. **Sidestep-and-land:** when the wearer is within ~2.5 m (pose range) *or* the glasses confirm the bottle (`height_frac ≥ 0.15`), the drone moves `left 100` or `right 100` toward the freer side (from the last scan) and lands. Props are off before the wearer is near; the drone is off the walking line.

If the room has a low ceiling or a table between `P` and `T`, `P` slides to the nearest free cell that still sees `T` within ±30°; δ is then computed from the map and added to the beacon bearing (§5.5).

### 5.5 Guide the wearer (IMU-free)

**Heading error ε** (deg, +right = turn right), the highest-priority fresh (< 0.5 s) and confident source wins:

| Priority | Source | Works when | Accuracy |
|---|---|---|---|
| 1 | Glasses see the **target** → `e_x` | ≤ 2.5 m, roughly facing it | ±3° |
| 2 | Glasses see the **drone beacon** → `e_x + δ` (δ from the map, ≈ 0 with §5.4) | ≤ 6 m while the drone hovers | ±3° |
| 3 | **Map bootstrap:** bearing of `T` from the origin, relative to the wearer's initial heading, minus the **visual yaw** integrated from the glasses camera's horizontal optical flow since takeoff (Farneback @160×120, `Δyaw = atan(median_dx / f)`) | always; drifts ~1°/s of turning, resets whenever 1 or 2 fires | ±10° short-term |
| 4 | Drone sees the wearer: **facing** from pose keypoints (left shoulder right of right shoulder ⇒ facing the camera; shoulder-width ratio ⇒ magnitude; nose offset ⇒ sign) vs. the drone→wearer ray | drone is facing them | ±20° |
| 5 | Nothing fresh | — | turn toward the last known side until 1/2 fires |

**Corridor obstacles** (drone facing the wearer from the beacon point): any detection whose bbox overlaps the column band between the wearer's feet and the bottom-centre of the frame, with its bottom edge above the wearer's feet row, is on the walking line; its side (bbox centre vs. the corridor centreline at that row) says where to route. Only the ordering matters, so no metric depth is needed.

**ToF zones** (agree with firmware): caution 0.7–1.3 m (laptop cues allowed; steer away from the short side), hard STOP < 0.7 m (C6 overrides).

**Cue policy:**
```
ε_eff = ε + offset               offset = ±35° away from a corridor/ToF obstacle while present
|ε_eff| > 12° → TURN L/R, intensity = min(1, |ε_eff| / 60)
|ε_eff| ≤ 8°  → FORWARD if corridor clear and min(tof) > caution, else STOP
between       → keep the previous cue
cue changes held ≥ 0.4 s; FORWARD re-pulsed every 1 s; STOP whenever the C6 says so
```
Audio only on events: "Bottle found…", "Obstacle, step right", "Almost there", "It's in front of you, slightly left."

### 5.6 Arrival and handoff

Arrived when: glasses target `height_frac ≥ 0.28` with `|e_x| < 15°` (bottle ≤ 0.8 m), **or** target centred and `min(tof) ≤ 0.7 m` and the drone's wearer-range ≤ 1.3 m (drone already landed by then; use the last estimate). Then ARRIVED cue + audio with the side hint. Stretch: hand keypoint inside the target bbox.

### 5.7 Safety without telemetry

- **Timers replace the battery gauge:** mission ≤ 5 min from takeoff, search ≤ 120 s, fresh battery per run, Tello LED checked by a human before takeoff.
- **Altitude only by command:** cumulative `up/down` ≤ 1.5 m; never fly over furniture; the free-space estimator treats tables as obstacles.
- **Hop watchdog** (§5.3 step 5) and `stop` sent raw (never queued).
- **People:** sectors with a person within 2.5 m score 0; the watchdog stops on a large person bbox; the beacon point is ≥ 1.3 m from the bottle and the drone lands before the wearer is within 2.5 m.
- **Keys:** `space` → `rc 0 0 0 0` + `stop`; `l` → land; `e` → emergency; `n` → skip phase. An escort stands within reach of the wearer in every test.
- Prop guards on. Tello auto-lands after 15 s of silence — keep-alives are deliberate, not accidental.

---

## 6. Perception module spec

### Inputs
| Input | Transport | Rate |
|---|---|---|
| Tello frames | djitellopy `BackgroundFrameRead` (PyAV), newest only, arrival-stamped | process 8–15 fps |
| Commanded moves / turns | from `mission` (for pose dead-reckoning) | per command |
| ESP32-CAM frames | OpenCV `VideoCapture("http://<ip>:81/stream")` in a thread, newest only | 10–20 fps |
| C6 telemetry | pyserial: `tof_L, tof_R, stop` | 20 Hz |
| Mission state | which pipeline mode: `explore / approach / beacon / guide` | on change |
| `calib.yaml` | both `f`, glasses `e_x` offset, beacon HSV, depth scale, class-height priors, Tello cm-per-command accuracy | once |

### Outputs (fields of the shared `World`)
`world.drone` (10 Hz): `pose (x, y, θ, src)`, `latency_ms`, `stale`, `person {bbox, bearing, elev, range, range_src, facing, facing_deg, conf, fresh}`, `target {cls, bbox, bearing, elev, range, world_xy, conf, confirmed, candidate}`, `detections[]`, `free_profile[5]` (per dwell frame), `compass {dtheta_meas, n_matches}`, `looming`, `obstacles[] {cls, bearing, range, in_corridor, side}`, `corridor {clear, blocker_side}`.

`world.map`: grid (uint8), `seen_count`, vantage points, semantic memory, `scores[40]` of the last scan (drawn on the dashboard).

`world.glasses` (10 Hz): `target {bbox, e_x, height_frac, conf, fresh}`, `beacon {seen, e_x, area_frac}`, `visual_yaw_deg` (integrated), `tof {L, R, stop, zone}`.

`world.guidance` (5 Hz): `heading_err_deg, err_src, err_conf, dist_to_target_m, tof_L, tof_R, stop, obstacles[], reach`.

`world.events`: `TARGET_CANDIDATE, TARGET_CONFIRMED, TARGET_LOST, PERSON_SEEN, PERSON_LOST, HOP_BLOCKED, CORRIDOR_BLOCKED/CLEAR, REACH, SEARCH_TIMEOUT`.

Plus annotated frames for the dashboard (drone view with boxes + free-space bars + the map panel; glasses view with the target/beacon) and a JSONL log of `World` every tick.

### Code layout
```
perception/
  sources/    tello.py  esp32cam.py  webcam.py  replay.py     (frame, t_capture, t_arrival), newest-only
  detect/     yolo.py (OpenVINO, per-mode imgsz)  pose.py  beacon.py (HSV)  depth.py (DA-V2-S, optional)  sharp.py
  geometry/   intrinsics.py  bearing/range priors  ground_plane.py  pose_dr.py (commanded-move dead reckoning)
  explore/    compass.py (ORB Δθ + loop closure)  freespace.py (depth | floor-line)  grid.py  semantic.py  scorer.py
  fusion/     heading_error.py  corridor.py  reach.py  visual_yaw.py
  world.py    the shared dataclass
  calib/      fov_from_ruler.py  beacon_hsv.py  depth_scale.py  tello_move_accuracy.py
  tools/      record.py  replay_scan.py  mock_world.py  bench_openvino.py
mission/      fsm.py  explore.py  approach.py  guide.py  params.yaml
drone/        tello_io.py (raw UDP commands, keep-alive, stop)  vendor/flydrones/ (base, tello, sim, safety, command, retina)
glasses/      c6_serial.py  cues.py
dashboard.py  main.py
```

### Calibration (45 min total, day 0)
Focal lengths (1 m ruler at 2 m in both cameras); glasses `e_x` offset (look at a marker straight ahead); beacon HSV under the venue lights (+ a fallback range); depth scale (a wall at a taped 3 m); **Tello move accuracy** (`forward 100` ×3 and `cw 90` ×4 on the venue floor with tape marks — this number decides the hop cap); class-height prior of the actual demo bottle.

---

## 7. `flydrones`: what it is, what we take

Read in full (commit `3e26934`, v0.1.2, MIT). It is a **spiking simulation of a fruit-fly connectome** that pilots a drone by reflex: camera → optic flow per "eye" cell → Poisson spikes → LIF network → descending-neuron rates → fixed linear read-out → `FlightCommand` → safety governor → drone. It is steered by *illusions* (fake optic flow), as its hand-gesture demo does.

- **No reinforcement learning** anywhere; the connectome is fixed; `calibrate.py` fits the read-out by ridge regression; the FAQ says "Does it learn? Not yet". No notion of target, person, map, or path.
- Tello/Crazyflie/MAVLink adapters follow the SDKs but are "**not flight-tested by the authors yet**".
- Its `TelloDrone.telemetry()` reads the djitellopy state; with your command-only access it simply returns unknowns, which is fine.

We vendor six files (MIT notice kept): `drones/base.py` (interface), `drones/tello.py` (thin wrapper; we add raw `send_command_without_return` + `stop`), `drones/sim.py` (velocity-mode quadcopter with rooms/boxes/collisions — the plant for the RL team's sims), `safety.py` (governor: clamps, slew, watchdog, flight time; we add the person rule and remove the altitude rules we cannot check), `motor/command.py` (`FlightCommand`), `senses/retina.py` (looming expansion → the hop/approach brake; its `rotation` scalar is a rough visual-yaw signal, but we use Farneback for that).

Fly-brain options, by risk: (1) `Retina` looming brake — no training, on-story, ships; (2) MiniFly as a reflex veto layer — CPU cost and surprises, demo-optional; (3) RL over the illusion channels with the connectome fixed — the nearest thing to "RL-training the fly brain", trainable in `SimDrone` (≈ 6 ms per 50 ms step), laggy; (4) plain RL policy. **Recommendation: (1) now; nothing else touches the critical path in 28 hours.**

---

## 8. RL track (parallel, never on the critical path)

### 8.1 `ExploreRoom-v0` (2-D, the useful one)
- Random rooms: 4–10 m sides, walls, 2–6 furniture rectangles with classes, one target placed with the same co-occurrence priors (bottle on a table 70% / floor 30%…), 0–2 people.
- Agent: the drone at a vantage point with the same abstract observation SCAN produces: 40 sectors × (`free_dist` with 20% noise, `seen_fraction`, `semantic bonus`, `candidate flag`), plus hop count and clock. The target is "visible" when within the class scan radius and unoccluded.
- Action: hop sector (40) + hop length {0.5, 1, 1.5, 2 m}.
- Reward: −1 per vantage point, −0.02 per metre, +30 on confirmation, −10 on a wall/furniture hit, episode cap 8 vantage points.
- Baseline: the scorer of §5.3 step 4 (implement it in the sim first — this env also **tunes the scorer weights** even if RL is never used). PPO with a small MLP trains in minutes. Go/no-go: fewer vantage points than the scorer over 500 random rooms **and** no worse worst-case; then it replaces `choose_hop()` only.

### 8.2 `GuideWalker-v0` (2-D)
Wearer as a point with heading; human model turns 30–60°/s after a 0.3–0.8 s delay, walks 0.3–0.5 m/s on FORWARD, stops in 0.3 s, ignores 10% of cues. Obs = `world.guidance` fields with noise/dropouts; action = {L, R, FORWARD, STOP} × {0.5, 1}. Baseline = §5.5 policy. Go/no-go: beats the baseline in sim **and** a blindfolded teammate reaches the bottle from 4 m with one chair, 3 of 4 runs, no contact.

### 8.3 Follow-behind in `SimDrone` — dropped for this weekend (Tier 2 is a PID).

---

## 9. 28-hour schedule (H = hours from now; shift blocks to fit sleep, keep the order)

| Block | Hours | Who | Deliverable | Exit test |
|---|---|---|---|---|
| **A. Plumbing** | H+0–3 | drone pair + glasses pair | Tello link on the Intel laptop; raw command sender with keep-alive/stop/kill keys; newest-frame video; OpenVINO exports (11n@640, 11n@960, pose); first takeoff/`cw 45`/`forward 100`/land; **move-accuracy calibration**; ESP32-CAM stream + C6 serial visible in the same process; record 5 clips | all links up simultaneously; latency numbers on screen; `forward 100` lands within ±15 cm |
| **B. Explore** | H+3–9 | drone pair | scan (dwell, sharp frame, detector, free-space, compass), grid + semantic memory + scorer, hop with watchdog, dashboard map panel, JSONL log | in the venue: bottle within 4 m found from a "blind" start in ≤ 90 s, 3 of 4 runs; map looks like the room |
| **C. Approach + beacon** | H+9–13 | drone pair | approach point from the wearer's side, servo, beacon hover, person confirmation, "bottle found" audio, sidestep-and-land | drone ends 1.0–1.7 m from the bottle facing the wearer, 3 of 4 runs. **Record the Tier-0 video now** |
| **Sleep** | 5 h somewhere in H+13–20 | everyone in shifts | — | — |
| **D. Guide** | H+13–22 (glasses pair from H+3 in parallel) | glasses pair + firmware | glasses pipeline (target, beacon, visual yaw), heading fusion, cue policy, ToF zones, C6 cue patterns, arrival | blindfolded teammate reaches the bottle from 4 m, 3 of 4 runs, no contact |
| **E. Integrate + rehearse** | H+22–27 | all | voice → FIND, full runs ×3 on fresh batteries, dashboard polish, experiment numbers from logs (time-to-find, vantage points used, cue accuracy, ToF stop latency), pitch | three clean end-to-end runs recorded |
| Buffer | H+27–28 | — | — | — |

Parallel: RL team builds 8.1 from H+3 (it tunes the scorer weights by H+9 even if RL never ships), then 8.2; voice team builds against `vocabulary()` and the `FIND` message with `mock_world.py`.

**Cut list, in order:** obstacle routing (ToF STOP only, no chair in the demo) → visual yaw (sources 1, 2, 4, 5 only) → landmark re-anchoring → follow-behind → depth model (floor-line heuristic only) → search relocation (single vantage point, bottle placed within 3.5 m; this is the emergency demo).

---

## 10. Risks

| Risk | L | I | Mitigation |
|---|---|---|---|
| Wi-Fi: Tello AP + ESP32-CAM + internet on one laptop | High | blocks | USB dongle + hotspot; C6 on USB; test in hour 1; fallback: ESP32-CAM on the Tello AP |
| Venue floor defeats the Tello's position hold (glossy, uniform, dark) | Med | hops inaccurate, drift | patterned rug for takeoff zone, lights on, hop cap from the measured accuracy, visual compass, closed-loop approach |
| Bottle too small / confused with cups | Med | search fails | tall coloured bottle, `imgsz=960`, size-plausibility, two-frame confirmation, candidate-then-verify |
| Depth model conversion eats hours | Med | no free-space | floor-line heuristic first (1 h), depth model only if B is on time; CPU inference on dwell frames is acceptable |
| Video lag > 250 ms | Med | servo oscillation | dwell-based scanning, low `rc` gains, deadbands |
| No battery reading | High | mid-air landing | 5-min mission timer, fresh battery per run, LED check |
| Table climb | Med | ceiling hit | never overfly; tables are obstacles; no altitude commands during approach |
| Beacon invisible under venue lights | Med | guidance falls to sources 3–5 | neon card + LED ring fallback; calibrate HSV on site |
| Human response to cues | High | guidance looks broken | eyes-closed cue test before wiring; hysteresis; audio only for events |
| Drone near the wearer | Low | severe | beacon ≥ 1.3 m from the bottle, lands before the wearer is within 2.5 m, sidestep off the line, escort, kill keys |
| Integration at the end | High | night lost | one `World` dataclass, `mock_world.py` from hour 3, main loop skeleton from block A |

---

## 11. Open items (answer when known)

1. Laptop OS (Windows/Linux) — Wi-Fi dongle plan and serial names.
2. Does your drone wrapper expose `battery?` / the state stream? (Use it if so; nothing depends on it.)
3. C6 protocol: serial line format for cues and telemetry; caution/hard ToF thresholds.
4. Is the ESP32-CAM streaming today? If not, Tier 1 starts with a USB webcam taped to the glasses so the guidance code is not blocked.
5. Demo bottle: pick it now (tall, opaque, coloured) and measure it.
6. Who is the escort/safety person for flight tests.

---

## Appendix A — Geometry, priors, parameters, Tello cheat sheet

- Pixel bearing `θ = atan((x − cx)/f)`; elevation `φ = −atan((y − cy)/f)`.
- Range from a known height `H`: `d = f·H / h_px`. Priors (m): bottle 0.22 (measure yours), cup 0.10, backpack 0.45, chair 0.85, laptop 0.25, dining table 0.75, couch 0.85, person shoulder width 0.42, torso 0.50.
- World position of a detection: `(x + d·cos(θ_map), y + d·sin(θ_map))`, `θ_map = θ_pose + bearing`.
- Ground distance of an image row at height `h`, level camera: `d = h / tan(atan((y − cy)/f))`.
- Visual compass: bearings of matched features `θ_i = atan((x_i − cx)/f)`; `Δθ = median(θ_prev − θ_curr)`; accept if ≥ 15 matches and MAD < 2°.
- Visual yaw (glasses): `Δyaw ≈ atan(median_dx / f_scaled)` per frame; integrate; reset on a source-1/2 fix.
- Beacon offset `δ = ∠(D − W) − ∠(T − W)` from the map; 0 by construction when the drone sits on the wearer→target line.

| Parameter | Default |
|---|---|
| scan altitude / steps / dwell settle / frames per dwell | 1.3 m / 8 × 45° / 0.5 s / 3 (sharpest kept) |
| detector: mode imgsz, target conf (confirm / candidate) | explore 960, else 640; 0.55 / 0.30 |
| confirmation | 2 frames within ±5°, or 1 frame conf ≥ 0.55 with size plausibility ×0.5–2 |
| free-space cap / hop fraction / hop min–max / speed | 5 m / 0.5 / 0.5–2.0 m / `speed 40` |
| scorer weights | open, (0.3 + novel), semantic 1 + 2·Σw (+4 candidate), people 0/1; dead end < 0.15 |
| budgets | search 120 s or 5 vantage points; mission 5 min |
| watchdog | central free < 0.8 m, looming spike, person bbox > 60% height → `stop` |
| approach | point 1.3 m before T on the wearer line; servo deadbands 4° / 0.15 m; `rc` forward ≤ 30 |
| beacon exit | wearer range ≤ 2.5 m or glasses `height_frac ≥ 0.15` → sidestep 1 m → land |
| guidance | turn > 12°, forward ≤ 8°, hold 0.4 s, obstacle offset ±35°, ToF caution 1.3 m / hard 0.7 m |
| reach | glasses `height_frac ≥ 0.28` and `|e_x| < 15°`, or centred + ToF ≤ 0.7 m |

Tello SDK 2.0 commands we use (UDP 8889, text): `command`, `streamon`, `takeoff`, `land`, `emergency`, `stop`, `up/down/left/right/forward/back <cm>`, `cw/ccw <deg>`, `go <x> <y> <z> <speed>`, `speed <cm/s>`, `rc <lr> <fb> <ud> <yaw>`. Send moves without waiting for the reply; poll for `ok`; `stop`/`emergency` always go out immediately on their own socket write.
