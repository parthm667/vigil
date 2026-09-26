# FlyGuide: a fruit fly brain pilots a drone that guides a blind person

Build brief for Claude Code. Read fully before writing code. Self-contained.

## 1. Concept

Walk into a room the system has never seen. No markers, no pre-scan, no setup. A Tello drone takes off, and the MaleCNS fruit fly connectome pilots it. The drone flies just behind and beside a blindfolded person, builds a 3D map of the room live from its camera, and an audio guide tells the person where things are, where to turn, and when to stop.

Why this design (shepherd mode):
- **The fly does what it natively does.** A male fly chases a moving target with its courtship pursuit circuit (LC10 to steering neurons) and dodges looming objects. Here the moving target is the person. No virtual target trick needed.
- **One camera does everything.** From over the person's shoulder the drone sees the person and the space ahead of them, so it can track them and map obstacles in the same frames. No fixed camera, no extra hardware, no prep.
- **Safer.** Props stay behind the person, not in front of their face.
- **The map is what makes it work.** The camera cannot see the floor right at the person's feet, but it saw that floor a few seconds earlier from farther back. The map remembers it.

Optional **scout mode**: when the person stops, the drone flies ahead to map unexplored space, then returns.

## 2. Hard constraints

1. **No environment preparation.** No markers, tags, pre-built maps, or calibration inside the room. The system is developed and tuned in advance, but it knows nothing about the demo room until it flies.
2. **Everything runs live on a MacBook M5.** CPU, Neural Engine, and integrated GPU for rendering only. No cloud, no network, no NVIDIA GPU at runtime.
3. **Free only.** Open-source or built-in software. Hand-build the core: state estimation, mapping, tracking, planning, cue logic, audio engine, visualization glue.
4. **Real time.** Every loop has a budget (Section 4), measured continuously.
5. **Nothing safety-critical depends on the fly.** Warnings and the safety governor are conventional code and work with the drone landed.

## 3. Architecture

```
Tello: video (960x720 H.264) + state packets (attitude, velocity, ToF height, baro, accel)
   │
   ├─> decode (PyAV) ─> frame
   │      ├─> YOLO (CoreML, Neural Engine) ─> person bbox + object boxes
   │      ├─> Depth Anything V2 Small (CoreML, Neural Engine) ─> relative depth
   │      ├─> ORB features ─> visual odometry
   │      └─> FlyDrones retina (optic flow, looming)
   │
   ├─> State estimator (hand-built EKF): VO + Tello velocity/attitude + ToF height ─> drone pose
   ├─> Depth scaling: floor plane fit + ToF height ─> metric depth
   ├─> Mapper (hand-built voxel log-odds) ─> 3D voxel map + 2D person-level grid + labeled objects
   ├─> Person tracker (bbox + metric depth + drone pose, Kalman) ─> person position, heading, speed
   ├─> Planner (A* on known-free cells, frontiers) ─> route, next turn
   ├─> Cue engine ─> audio (speaker demo / AirPods product)
   │
   ├─> Fly brain (LIF, core subgraph): person spot into LC10a, P1 arousal, looming active
   │      └─> DN readout ─> safety governor ─> Tello RC commands
   │
   └─> Visualization: Rerun (map, drone, person, route, cues, brain) + MuJoCo fly body
```

Processes communicate over ZeroMQ with timestamped messages. Visualization never blocks control.

## 4. Real-time budgets (Mac only)

| Loop | Rate | Budget |
|---|---|---|
| Fly brain step + readout + governor | >= 20 Hz | < 50 ms |
| YOLO person + objects (Neural Engine) | >= 20 Hz | < 30 ms |
| Depth (Neural Engine, keyframes) | 8 to 10 Hz | < 40 ms |
| VO + EKF update | >= 30 Hz | < 15 ms |
| Voxel integration + grid projection | 8 to 10 Hz | < 30 ms |
| Planner replan + TTC + cue selection | >= 10 Hz | < 20 ms |
| Audio onset after cue | per cue | < 100 ms |
| Frame to audio warning, end to end | per event | < 250 ms |
| Fly body render | 30 fps | own thread |

Apple's CoreML Depth Anything V2 Small runs in roughly 25 to 35 ms per frame on recent Macs' Neural Engine (published M1/M3 figures); verify on the M5. Depth and YOLO share the Neural Engine: schedule them so neither starves, and measure contention. `flyguide bench` reports p50/p95/p99 for every loop and end to end, from live runs and recorded replays.

## 5. Software (all free)

| Need | Choice | Notes |
|---|---|---|
| Fly brain + drone I/O + governor | Fork [SpikeCalls/FlyDrones](https://github.com/SpikeCalls/FlyDrones) (MIT) | MaleCNS loader, Shiu LIF sim, looming/optomotor, Tello adapter, safety governor, calibrate |
| Connectome | MaleCNS v1.0 | CC-BY 4.0 |
| Depth | [apple/coreml-depth-anything-v2-small](https://huggingface.co/apple/coreml-depth-anything-v2-small) | Apache-2.0, relative depth, Neural Engine |
| Detection | Ultralytics YOLO exported to CoreML | AGPL-3.0; COCO classes cover chair, couch, table, bed, plant, TV, bags, bottles. Optional: YOLO-World with a fixed vocabulary (door, box, shelf, stairs) exported to CoreML |
| VO primitives | OpenCV (ORB, PnP, RANSAC) | Free |
| Tello | `djitellopy` via FlyDrones | Free |
| Visualization | Rerun | MIT/Apache, native on Mac |
| Fly body | [TuragaLab/flybody](https://github.com/TuragaLab/flybody) model in MuJoCo | Free; anatomically detailed MuJoCo fly. Note: modeled on a female fly |
| Speech | macOS `say` / AVSpeechSynthesizer, rendered into a phrase cache | Built in |
| Audio I/O | `sounddevice` + NumPy, hand-built mixer | Free |
| Spatial audio (product) | Hand-built HRTF convolution with a public HRTF set (e.g. MIT KEMAR) | Free |
| IPC | `pyzmq` | Free |

Hand-built by us: EKF, depth scaling, VO glue, voxel map, grid, frontier detection, person tracker, A* planner, TTC predictor, cue engine, phrase cache, audio mixer, fly target encoder, pursuit decoder, fly body animator, all Rerun logging.

## 6. Modules

### 6.1 State estimation (hand-built EKF)
- State: position, velocity, yaw (Tello attitude handles roll/pitch).
- Inputs: Tello state packets (velocity, attitude, ToF height, baro, accel) at ~10 Hz, VO pose deltas at camera rate.
- VO: ORB features on keyframes, 3D points from scaled depth, PnP + RANSAC for pose change. Reject when inliers are low; fall back to Tello velocity.
- Drift is acceptable for single-room sessions. Optional: simple loop closure on keyframe ORB descriptors if drift is visible.

### 6.2 Metric depth without markers
- Depth Anything gives relative depth. Fit the floor plane in the lower image region (RANSAC on back-projected points), then scale so the camera's height above that plane equals the Tello ToF height.
- Cross-check scale with VO translation vs Tello velocity integration; smooth the scale over time.
- Mask the person region out of depth before mapping so they are not mapped as an obstacle.

### 6.3 Mapping (hand-built)
- Sparse voxel hash, 5 cm voxels, log-odds occupancy with free-space ray casting along each depth ray (subsampled).
- Voxels need repeated observations from different poses before counting as occupied; free-space updates clear stale obstacles (moved chairs).
- Person-level 2D grid: project occupied voxels between 0.05 and 2.0 m height. Cells are **free**, **occupied**, or **unknown**.
- Objects: YOLO boxes lifted to 3D with metric depth, merged across frames by 3D overlap, labeled in the map.
- Frontiers: free cells adjacent to unknown.

### 6.4 Person tracking
- YOLO person box, feet point, metric depth, drone pose: person world position.
- Kalman filter (constant velocity) for position, heading, speed. Predict through brief occlusions.
- Tracking lost > 1 s: governor slows and yaws to reacquire; > 5 s: hover and say "Stop, I lost you"; > 10 s: land.

### 6.5 Planner and guidance logic
- **Core rule: the person is only routed through known-free cells.** Unknown space is never assumed safe.
- Modes:
  - **Walk freely:** no goal; warnings plus short descriptions ("Open space ahead. Table on your right.").
  - **Take me to X:** once an object class has been mapped (couch, chair, table, door if using the open vocabulary model), A* to it on the inflated grid (0.35 m person radius).
  - **Explore:** route toward the nearest frontier with the most open space; if the route needs unseen space, trigger scout mode or ask the person to wait.
- Replan at >= 10 Hz from the person's current pose.

### 6.6 Fly piloting
- **Target encoder:** person bbox center becomes a small moving spot at the matching azimuth in the fly's visual field, driving LC10a cells with retinotopy from their lobula positions in MaleCNS (fallback: side + position-rank bins). Encode spot velocity.
- **Arousal:** tonic P1 stimulation. If LIF transmission from LC10a to steering DNs does not increase under P1 (neuromodulation may not be captured), add a documented `arousal_gain` on LC10a input.
- **Readout:** yaw from L/R asymmetry of steering DNs: DNa02 primary; DNa01, DNb05, DNg13 ipsiversive; DNb06 contraversive. Use DNp09 only if the audit shows bilateral LC10 input. Forward speed from pursuit drive. Calibrate readout gains only; never train connectome weights.
- **Avoidance:** FlyDrones looming pathway stays active; a too-close person or obstacle triggers back-off.
- **Positioning:** governor holds the shepherd pose: 1.5 to 2.0 m behind and ~0.5 m to one side of the person, altitude ~1.4 m, so the camera sees past the shoulder. The fly supplies yaw and approach; the governor clamps distance, altitude, and lateral offset.
- **Scout mode:** when the person is stationary and a frontier is needed, the target encoder switches to a spot at the frontier bearing. Drone flies there, maps, returns to shepherd pose.
- **Real time:** FlyDrones `--core-hops` subgraph keeping LC10/LC9/LC11 to AOTu to steering DNs, P1, and looming paths.
- **Audit gate first:** `flyguide audit` reports per side the LC10 to DN pathway (direct and 2-hop). Stop and report if no bilateral pathway exists.

### 6.7 Audio (hand-built)
Priority queue, one cue at a time, higher priority interrupts lower, rate limited with hysteresis:
1. `STOP` when time-to-collision < 1.0 s, or person heads into unknown space
2. Obstacle warning when TTC < 2.5 s: "Chair ahead, one meter, slightly right"
3. Turn cue when heading deviates > 25 deg from route for > 1 s: "Turn left"
4. Progress and descriptions: "Couch three steps ahead", "Open space ahead"
5. Soft tick while on route

Real-time: pre-render every template combination (object x distance bucket x direction bucket, plus turn/stop/status phrases) to in-memory PCM at startup; playback is a buffer copy. Procedural stop tone. Small output block size.

Outputs: **demo speaker**, where every cue states direction in words since a speaker cannot convey direction. **AirPods (product path)**: hand-built HRTF places a beacon at the next waypoint and warnings at the obstacle's true direction.

### 6.8 Visualization

**Rerun (main screen):**
- 3D voxel map growing live, colored by time first observed; labeled object boxes.
- Person-level grid with free/occupied/unknown and frontiers.
- Drone trajectory and camera frustum; person marker with heading and trail; planned route; red TTC zones; cue markers placed where each cue fired.
- Timeline scrubbing to replay how the map evolved; plots of mapped area and known-free coverage over time.
- Panels: drone camera with detections and depth overlay, the fly's-eye view with the spot, audio cue log, latency gauges.
- Fly brain: MaleCNS soma positions as a point cloud, activity-colored, with the LC10 to AOTu to DN pursuit path and the looming path highlighted.

**Fly body (the viral shot), MuJoCo + flybody model:**
- Kinematic animation driven live by the brain's motor outputs, rendered at 30 fps and logged into Rerun (or shown in a MuJoCo window beside it).
  - Wing stroke amplitude, left and right, from flight DN activity (DNg02 L/R, as FlyDrones uses for throttle and yaw). Wingbeat is shown slowed with a motion-blur sweep, since the real ~200 Hz is not visible.
  - Body yaw and bank from steering DN asymmetry.
  - Head turned toward the LC10 activity centroid (where the fly "sees" the person).
  - Looming or giant fiber event: escape pose (legs extend, wings raise).
  - Courtship flourish: if the audit finds the song pathway (P1 to song DNs such as pIP10), show unilateral wing extension, the fly "singing" to the person.
- Label on screen: "Animated from the fly brain's live motor outputs; not a physics simulation."
- Stretch: flybody's physics flight with its trained low-level controller, fed our steering commands, if it runs in real time on the Mac.

## 7. Build order (each step: tests pass, commit)

1. **Setup.** Fork FlyDrones; run its demo and tests on the Mac; download MaleCNS; build the core subgraph; Tello connects; record raw sessions (video + state) for replay.
2. **Connectome audit.** Pursuit pathway report; gate on bilateral pathway; check song pathway presence.
3. **Perception on replay.** CoreML YOLO and Depth Anything integrated; scheduling on the Neural Engine measured.
4. **State estimation.** EKF with Tello state + VO; metric depth scaling from floor plane + ToF; validated on replays with a tape-measured path.
5. **Mapping.** Voxel log-odds, grid, objects, frontiers; Rerun live view from replays.
6. **Person tracking.** Validated against tape-measured positions on replays.
7. **Planner + cues + audio.** Phrase cache, cue engine, speaker output; walk-through tests on replays with staged obstacles.
8. **Fly piloting in sim.** Target encoder, arousal, readout, shepherd governor, scout mode in FlyDrones sim with a scripted walking person.
9. **Fly body visualization.** flybody in MuJoCo animated from DN outputs; logged to Rerun.
10. **Latency pass.** `flyguide bench` live and on replays; fix anything over budget.
11. **Live flights.** Props-off bench, then netted space, then shepherding a walking teammate (not blindfolded).
12. **Full demo run.** Unseen room, blindfolded teammate, walk-freely and take-me-to-X modes, zero contacts.
13. **Demo hardening.** One-command launch, recorded backup run, failure playbook (tracking loss, video drop, low battery, VO drift).

Optional offline use of Modal credits (development only, never at runtime, never room-specific): massively parallel simulation sweeps to tune bridge parameters (spot size and speed, arousal gain, readout gains, governor limits).

## 8. Hardware
- 2 Tello drones, spare batteries, prop guards
- MacBook M5 connected to the Tello's Wi-Fi (no internet needed at runtime)
- Bluetooth speaker; AirPods for the product segment
- Spotter with kill switch; netting for early tests

## 9. Safety (all outside the brain)
- Dry run by default; `--send` for hardware.
- Max speed 0.4 m/s, max yaw 60 deg/s, altitude 1.2 to 1.6 m.
- Minimum 1.2 m from the person at all times; never in front of their face.
- Geofence grows with the map: the drone may only fly into known-free space, except scout mode at reduced speed with looming avoidance active.
- Tracking loss, brain stall, low battery, and VO failure each trigger hover or land (Section 6.4 thresholds).
- Audio warnings run with the drone landed and never depend on the fly.
- Spotter on kill switch for every flight.

## 10. Honesty rules for docs and pitch
- Biology: wiring, cell types, LIF parameters, pursuit and looming pathways. Engineering: camera-to-neuron encoding, readout gains, arousal gain if used, governor, state estimation, mapping, planning, audio, body animation.
- Show that split on a slide. Prototype, not a replacement for a cane or guide dog.

## 11. Repo conventions
- Python 3.11, keep FlyDrones packaging and CLI. Commands: `flyguide audit`, `flyguide replay`, `flyguide fly`, `flyguide bench`, `flyguide demo`.
- Packages: `perception/`, `estimation/`, `mapping/`, `tracking/`, `planning/`, `audio/`, `pilot/`, `viz/`, `body/`, `bench/`.
- Record every session (video, Tello state, messages); every module must run from replay.
- Configs in `configs/flyguide_*.yaml`; calibration files versioned.
- **Git: commit and push incrementally as each step or meaningful sub-step lands. Do not add Co-Authored-By trailers or any coauthor attribution to commits.**
- No em dashes in docs, comments, or commit messages.
