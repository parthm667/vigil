# ReachGlass Drone: RL and Autonomy Plan

**Owners:** Parth and the teammate (RL pair) · **Status:** final plan for the rest of the hackathon, revised after feasibility, scope and RL-rigor reviews · **Written at:** about hour 7 of the hackathon, about 29 hours left · 2026-09-26

**Supersedes:** [docs/PLAN.md](PLAN.md) for all RL work (the mushroom-body object-search policy is dropped). **Keeps from** [FLYGUIDE_SPEC.md](../FLYGUIDE_SPEC.md): the connectome audit gate, the LC10 to steering-DN readout idea, the governor outside the brain, and the biology/engineering honesty split. **Drops from** FLYGUIDE_SPEC: SLAM, EKF, Depth Anything, voxel map, planner over a world map, and the Mac M5 runtime assumption.

Background: [research/fruitfly-brain-rl.md](../research/fruitfly-brain-rl.md). Time is given as **H+n**, hours from the start of this plan.

---

## 0. Read first: timeline update (overrides the timing in Sections 7 and 8)

Sections 7 and 8 were written assuming Parth and the teammate both start tonight and both have Modal. That is not the case:

- **Only the teammate has Modal access.** Every Modal step (item 1b, smoke test, launch) is his, and nothing can launch on Modal before he arrives.
- **the teammate joins at about T-20** (20 hours before the deadline, about H+9). The plan starts at about T-29 (H+0).
- **No drone flying is needed before training.** Training runs entirely in simulation on simulated boxes. The R0 measurements (video latency, stick response, FOV, box noise) only narrow the randomization ranges in Section 4.4, and training launches with the wide ranges if they are not measured yet. If Parth supplies measured latency and lag values before launch, we center the ranges on them with about plus or minus 30 %.
- Optional ground checks that need no flight can happen any time: Tello Wi-Fi and firewall on the runtime laptop, printing the state stream, and the checkerboard FOV calibration.

**Before the teammate arrives (T-28 to T-20), if the team approves:** coding agents build everything that needs neither Modal nor a decision: items 0, 1, 2 and 3 (the audit runs locally on CPU), 4a, 4b, 5, 6, 7 and 7b, and item 12. Item 7 also gets a `--backend local` option that fans episodes out over the laptop's CPU cores with the same code as `--backend modal`. A small local CMA-ES run then checks the whole pipeline. The MaleCNS download the research used sits in a temporary scratchpad folder, so it is re-downloaded into the repo's gitignored `data/` folder.

**Revised schedule (T-minus hours to the deadline):**

| Time | What | Gate |
|---|---|---|
| T-28 to T-20 | Parth asleep, the teammate away. Optional pre-build by coding agents (above), with the audit result and local benchmarks written to `docs/audit_result.md` and `docs/training_log.md` | G0 can pass here if the audit runs |
| T-20 to T-18 | the teammate: `modal setup`, item 1b, review the audit (G0), item 8 smoke test, item 9 launch | **G1** launch by T-18 (was H+6) |
| T-18 to T-12 | Training runs on Modal (3 to 7 h estimate). Team: R0 measurements, R1 replay, R2 props-off with PID, FIND on recorded video | |
| T-12 | Evaluate all arms; fine-tune if R0 values fall outside the training ranges | **G2** |
| T-11 to T-8 | R3 and R4 flights, PID first, then the fly controller | **G3** at T-8 |
| T-8 to T-6 | FIND, APPROACH, FACE_PERSON end to end; record the backup demo | **G4** at T-6 |
| T-6 to T-4 | Integration with glasses and audio; code freeze at T-4 | Freeze |
| T-4 to T-0 | Slides, rehearsal, charging | |

The gates and pass criteria in Section 8 are unchanged; only their times move. If training cannot launch by T-16, launch NOBRAIN and PID-CMA first and a shorter fly run (fewer generations) so results exist by T-10.

---

## 1. Summary

**What the drone does.** A stock Tello follows behind a blind user. When the user says "find my water bottle", the drone searches the room, flies to the bottle, turns to face the user, and hovers where it can see the user and as much of the path between them as the room allows. The guidance module turns that view into servo taps and spoken cues. The glasses camera finishes the last arm's reach.

**What is RL.** One controller: a "pursue a box to a standoff distance" controller that drives yaw and forward speed. It is used for FOLLOW (box = the back of the user's head) and for the flight segment of APPROACH (box = the found object). Only the adapter around a fixed fly connectome is trained: an encoder (box to input currents into the fly's LC10a visual neurons, plus tonic arousal) and a readout (steering descending-neuron rates to yaw and forward commands). The connectome weights are never trained. We train with CMA-ES in simulation on Modal, overnight, with no physical drone. The fly does not encode distance (the adapter does, through the size input and the readout weights), and it does not handle altitude, lateral motion, lost targets or safety; those live in the adapter and the governor.

**What is not RL.** Search (FIND) is a scripted scan-hop-scan algorithm with object-furniture priors. Turning to face the user, the hover pose, altitude, the safety governor, all guidance cues, obstacle pathing and the final reach are algorithms. Detection is perception (the perception owner's YOLO models).

**What the fly brain contributes.** We expect the connectome to carry "target is to the left" to "turn left" through LC10a and the steering descending neurons; this is an expectation that the G0 audit tests, not a known fact. The trained fly controller is compared against shuffled connectomes and a no-brain controller trained with the same budget, and we expect a tuned PID to be at least as good. A PID follower is always ready, so the demo works without the fly.

---

## 2. The pipeline, step by step

### 2.1 Mode state machine

```
             "find X"                confirmed 2 of 3 frames          in standoff band 2 s
 FOLLOW ─────────────────> FIND ─────────────────────────> APPROACH ─────────────────────> FACE_PERSON
   ^                         │ not found (budget spent)                                         │ person reacquired
   │                         v                                                                  v
   │                      RETURN ──────────────> FOLLOW                                    OVERWATCH (hover pose)
   │                                                                                            │
   └──────────── user at object (glasses REACH done) or "stop" ─────────── GUIDE (drone holds) <┘

 Any mode: battery < 25 %, video lost > 5 s, operator key ──> LAND
 FOLLOW, OVERWATCH, GUIDE, and FACE_PERSON after its sweep and full scan fail: person lost > 10 s ──> LAND
 FIND and APPROACH: no person-lost rule (the user is out of view by design); limits are the FIND budget and battery
```

### 2.2 Classification table

Owner roles: **RL pair** (Parth, the teammate), **perception owner**, **hardware owner** (glasses, servos, ESP32), **voice owner**. "Guidance owner" means whoever writes the cue and path logic; it is not decided yet and must be confirmed at H+0 (Section 10, item 4).

| # | Step | Class | Chosen approach | Alternatives considered | Owner | Inputs | Outputs |
|---|---|---|---|---|---|---|---|
| 1 | FOLLOW | **RL** (trained adapter around fixed connectome) | Pursuit controller: head box to LC10a encoder, fixed MaleCNS pursuit subgraph, DN readout to yaw and forward. Up/down, lost-target handling and side offset are outside the brain (Section 4.1). | Classical PID follower (research says it is enough; it is our baseline and fallback). End-to-end deep RL (no evidence it beats PID on a Tello; rejected). | RL pair | Head box of the locked user track, Tello state, settings | rc commands (lr, fb, ud, yaw) |
| 1a | Person and head detection, track lock | Perception | YOLO person detector plus head box (head detector or top of person box), tracker with ID lock on the user | Face detector (useless from behind) | Perception owner | Tello frames | `det` messages (Section 6) |
| 1b | Range and bearing estimation | Algorithm | Pinhole geometry: bearing from box center, range from known size (head, class prior) or from user height (Section 5.4) | Depth network (too heavy on the laptop) | RL pair | Boxes, camera intrinsics, Tello attitude | Range and bearing with an error estimate |
| 1c | Latency filter | Algorithm | Constant-velocity Kalman filter on (cx, cy, h), predicts forward by the measured latency (Section 4.1) | Raw boxes (oscillates under latency) | RL pair | `det`, `video_latency_s` | Latency-compensated box |
| 2 | Voice intent | Perception (speech) plus algorithm (intent rules) | Speech to text, then keyword rules to `mode_cmd` with the target class | LLM intent parsing (needs internet, which the Tello laptop lacks) | Voice owner | AirPods audio | `mode_cmd` |
| 2a | FOLLOW to FIND switch | Algorithm | State machine on the voice intent. Before leaving, store the user's bearing and range in the drone frame and tell the user to stay put. | None | RL pair | `mode_cmd` with target class | Mode change, stored person pose |
| 3 | FIND | **Algorithm** | Scan-hop-scan: 360 degree yaw scan at 1.0 m with YOLO on every frame, hops toward furniture ranked by a miss-discounted prior, optional low floor scan (Section 5) | RL search policy (PLAN.md), frontier exploration, coverage control, lawnmower, learned ObjectNav. Debate in 5.1. | RL pair (motion), perception owner (detection) | `det`, Tello yaw | Confirmed object bearing and range estimate |
| 4 | FIND to APPROACH switch | Algorithm | Target class in 2 of 3 consecutive settled frames at the same bearing (within 5 degrees) | Single-frame trigger (too many false positives) | RL pair | `det` | Object bearing, class size prior |
| 5 | APPROACH | **RL** (same trained controller) plus algorithm for altitude | Same pursuit controller with the object box as target; altitude floor 0.8 m and a standoff that grows for low objects (Section 5.3) | Separate PID visual servo (fallback) | RL pair | Object box, Tello state | rc commands |
| 6 | Turn to face the person | Algorithm | Climb to 1.2 m, yaw toward the stored user bearing, reacquire visually with a sweep (the sweep is the primary method) | None needed | RL pair | Stored person pose, Tello yaw, `det` | Person box locked |
| 7 | Hover pose | Algorithm | Default: hold at the APPROACH standoff at 1.2 m altitude, facing the user. Overwatch planner (Section 5.4) is a stretch goal after G4. | Overwatch planner as default (rejected for now: needs a large room and unmeasured distance moves) | RL pair | Person and object in drone frame | Hover pose, then `scene` messages |
| 8 | GUIDE | Algorithm | Fixed cue rules plus A* on a ground-plane grid in the drone's frame | RL cue policy (rejected: safety cues for a blind user must be predictable) | Guidance owner; if none by H+13, the teammate writes minimal rules (Section 8) | `scene`, obstacle detections, glasses IMU, AirPods head tracking | Cue decisions |
| 8a | GUIDE obstacle detection | Perception | YOLO-World or segmentation on Tello frames at 2 to 5 Hz (Section 3.5) | COCO-only YOLO (cheaper, fewer classes; the budget fallback) | Perception owner | Tello frames | Obstacle boxes |
| 8b | Speech and tap output | Algorithm plus hardware | Text to speech on AirPods; servo tap patterns (left, right, aligned, stop) on the glasses | None | Hardware owner, voice owner | Cue decisions, `say` events | Taps, speech |
| 9 | REACH | Perception plus algorithm | Glasses ESP32-CAM frames to YOLO ("more left, pick it up"), plus the VL53L1X ToF for the last meter | None | Hardware owner, perception owner | Glasses camera and ToF | Taps, speech |
| 10 | RETURN | Algorithm | Yaw toward the stored user bearing, reacquire visually, resume FOLLOW; say what was searched | None | RL pair | Stored person pose, `det` | Mode change, `say` event |
| 11 | Safety governor | Algorithm, always on | FlyDrones `SafetyGovernor` plus our person-distance, lost-target, up/down and watchdog rules | None; never learned | RL pair | Every rc command, range estimate, state | Filtered rc, intervention flags |
| 12 | Glasses tapping RL (optional) | RL | Only if time remains after H+24 | Fixed cue rules (default) | RL pair | Cue logs | Tap timing policy |

---

## 3. Hardware and compute constraints

### 3.1 How the design respects them

| Constraint | Consequence for the design |
|---|---|
| Stock Tello, SDK over Wi-Fi only, no motor access | We send `rc a b c d` at 20 Hz (plus `takeoff`, `land`, `emergency`, and single `forward` moves for FIND hops). The Tello's own flight controller does stabilization. The Tello I/O owns the only command queue: djitellopy distance moves block until "ok" (7 s timeout, 3 retries), and an rc stream sent during a move may interrupt it (unverified), so for a move it stops rc, sends the move, waits for "ok" or times out and sends `rc 0 0 0 0`, then resumes rc. |
| Fixed, level, forward camera | Every behavior is image-based servoing on a box. The floor is only visible beyond about 2.5 times the drone's altitude (Section 3.4), which shapes search altitude, the approach standoff and the hover pose. |
| No position readout | No world map, no world pose. We use person space: user and object positions relative to the drone and to each other. Dead reckoning from integrated `vgx`/`vgy` is expected to be off by meters over tens of seconds (Section 3.2), so turning back to the user is visual; yaw only picks where the sweep starts. |
| Downward ToF noisy, not usable for scale | Scale comes from known sizes: head size for range to the user, class size priors for objects, and the user's own height (measured once and entered as a setting) for camera height. |
| Weak runtime laptop (Windows, integrated graphics), maybe a Mac | CPU-only runtime: a small pursuit subgraph of the connectome, YOLO nano via ONNX or OpenVINO, no SLAM, no depth network. |
| Modal only for training | Nothing at runtime calls Modal. The laptop on the Tello's Wi-Fi has no internet anyway (PLAN.md 2.2). |
| Tello AP does not forward traffic between clients (research) | All drone-side processes (Tello I/O, YOLO, controller) run on one laptop. The glasses (ESP32-CAM video, servo commands) cannot reach the laptop through the Tello AP; that link is open question 10, confirmed at H+0. |
| Flaky Tello Wi-Fi on some machines | One designated laptop for the drone; firewall rules set tonight (Section 3.3); every session recorded for replay so development does not need live flights. |

### 3.2 What we can read from the Tello

From the Tello SDK 1.3/3.0 documents and the djitellopy code (research report). State arrives on UDP 8890 at about 10 Hz, fixed by the drone.

| Field | Units | Use in this plan |
|---|---|---|
| `yaw` | degrees, relative to power-on, no magnetometer | Relative bearings during FIND and FACE_PERSON. Drift rate not measured yet. |
| `pitch`, `roll` | degrees | Remove camera tilt when converting pixel to bearing; flag frames taken while pitched. |
| `vgx`, `vgy`, `vgz` | integers, dm/s per SDK 3.0 (unit not given in 1.3), 10 Hz | Readings move in 0.1 m/s steps, so slow drift reads 0. Body frame or takeoff-heading frame is unverified. Used for response measurements for the sim, not for navigation. FlyDrones' telemetry treats the value as cm/s (`get_speed_z() / 100.0`); if R0 confirms dm/s, we change it to `/ 10`, or the governor's vz is 10 times too small. |
| `h` | cm from takeoff, changes in 10 cm steps | Coarse altitude hold targets and the altitude floors. |
| `tof` | cm to what is below | Not used for scale (team finding: noisy, effectively binary). Optional "something is below me" flag. |
| `baro` | noisy absolute, relative use only | Not used. |
| `agx`, `agy`, `agz` | unit ambiguous (unverified) | Not used. |
| `bat`, `templ`, `temph`, `time` | %, degrees C, s | Governor battery and heat rules. |
| Video | 960x720 H.264, up to 30 fps, UDP 11111 | All perception. djitellopy frames are RGB, not BGR. |

Not available: position, a downward camera image on a standard Tello, any obstacle sensor. The drone auto-lands after 15 s without a command, so the runtime streams rc at 20 Hz, including `rc 0 0 0 0` while hovering. We do not rely on the `keepalive` command (not in the 1.3 or 3.0 PDFs).

### 3.3 Wi-Fi and platform facts to act on tonight

- Windows firewall blocks inbound UDP 8890 and 11111. Symptom: "Did not receive a state packet" while commands work. Allow Python through and open both ports.
- macOS: `Errno 48 Address already in use` on 8890 after a crash; kill the stale process.
- Video decode needs about 3 s after `get_frame_read()` before frames are valid (before that `.frame` is a 300x400 zero array).
- Video latency: about 1 s reported with OpenCV `VideoCapture`, 150 to 300 ms with PyAV in a thread (unverified). **We have not measured it on our laptop yet.** The latency that matters (camera, encode, Wi-Fi, I-frame) happens before decode, so no timestamp in our code can measure it. We measure it once with the R0 stopwatch test and store it as the config constant `video_latency_s`. The sim randomizes 100 to 700 ms.

### 3.4 Camera numbers (conflict resolved)

Three sources disagree on field of view. The team quoted about 86.7 degrees; the spec sheet says 82.6 degrees and a forum analysis says that figure is diagonal; a community calibration at 960x720 (fx 921, fy 919) gives HFOV about 55 degrees, VFOV about 43 degrees. If 82.6 were the true diagonal of the stream, HFOV would be about 70 degrees.

**Decision:** we plan with HFOV 55 degrees and VFOV 43 degrees (the only figure backed by a calibration), randomize focal length across the whole plausible range in training (Section 4.4), and run a 10-minute checkerboard calibration (`cv2.calibrateCamera`) at R0 to settle it. PLAN.md used 82.6 degrees as horizontal FOV; that is likely wrong.

Consequences with VFOV 43 degrees and a level camera:
- Floor visible beyond altitude / tan(21.5 deg), about 2.5 times the altitude: 1.3 m at 0.5 m altitude, 2.0 m at 0.8 m, 2.6 m at 1.0 m, 3.1 m at 1.2 m.
- A 0.75 m table top is visible beyond about 0.65 m when flying at 1.0 m.
- The back of a head (about 0.23 m) at 2 m is about 105 px tall at native resolution; plus or minus 3 px of box noise is about plus or minus 3 % in range.
- A full standing person fits vertically only beyond about 2.2 to 2.6 m. At a 1.5 to 2.0 m follow distance we must use the head, which matches the team's idea.

### 3.5 Runtime budget on the weak laptop

| Loop | Rate | Budget | Evidence so far |
|---|---|---|---|
| Video decode (PyAV thread) | 30 fps | own thread | Not measured on our laptop |
| YOLO11n at imgsz 640, ONNX or OpenVINO CPU (FOLLOW, and every frame during FIND) | 10 to 20 Hz | < 80 ms | Ultralytics reports 56 ms ONNX CPU (CPU model not stated); OpenVINO is typically 30 to 60 % faster on Intel. Not measured on our laptop. |
| YOLO at imgsz 960 or 2 tiles (FIND confirmation, hovering) | 3 to 5 Hz | < 300 ms | Estimate: about 2.25 times the 640 cost. Not measured. |
| Open-vocabulary FIND (YOLOE or YOLO-World-S, non-COCO targets) | 5 to 10 Hz | < 200 ms | Research estimate 7 to 11 fps for YOLO-World-S on CPU. Not measured. |
| GUIDE obstacle detection (YOLO-World or segmentation) | 2 to 5 Hz | < 400 ms | Not measured. |
| Brain tick: 50 ms of simulated time (100 LIF steps at dt 0.5 ms) | 20 Hz | < 35 ms | Brain only (`flydrones bench`) on Parth's laptop (Intel Core Ultra 7 258V, 8 cores, integrated graphics; likely the Windows runtime laptop, confirm at H+0): a 17,824-neuron subgraph at 2.68x real time (0.37 s per simulated second, about 19 ms per tick), a 20,000-neuron capped subgraph at 1.88x (about 27 ms per tick). Our pursuit subgraph has different groups, so its size is unknown until we build it. Never measured with YOLO on the same CPU. |
| Encoder, readout, governor | 20 Hz | < 2 ms | Trivial arithmetic |
| rc send | 20 Hz | non-blocking | djitellopy `send_rc_control` does not wait for a reply |

**Rules.**
- FOLLOW and APPROACH: if brain tick p95 exceeds 45 ms with YOLO running, we shrink the subgraph (`max_neurons`), move the brain to its own process, and drop YOLO to 10 Hz. If it still does not fit, these modes fly on the PID and the fly runs in sim on the screen.
- FIND: the brain is idle (the scan is scripted), so the CPU goes to detection.
- GUIDE: the drone is hovering, so the brain is off and yaw hold uses the PID. Obstacle detection runs at 2 to 5 Hz. If that is still over budget, we detect COCO classes only.

---

## 4. The RL work: the pursuit controller

### 4.1 What is trained and what is fixed

FlyDrones (SpikeCalls/FlyDrones, identified by commit `3e26934`, the commit we read; MIT license per its LICENSE file) gives us the brain simulator, the MaleCNS loader, a governor and a Tello adapter. It has no pursuit behavior, no person detection, no reward, no environment wrapper and no training loop (verified in the deep read). The control loop we reuse is `Pilot.tick` in `src/flydrones/runtime.py`:

```
FlyDrones today:  Retina.encode -> GestureIllusion.apply -> InputEncoder.encode -> Brain.tick -> MotorDecoder.update -> SafetyGovernor.filter -> send
Our loop:         box (latency-compensated) -> TargetEncoder.encode -> Brain.tick -> PursuitDecoder.update -> PersonGovernor.filter -> send
```

| Part | Status | Where it lives | Notes |
|---|---|---|---|
| Connectome weights (MaleCNS v1.0, NT-signed, pairs with at least 3 synapses) | **Fixed, never trained** | `src/flydrones/brain/connectome.py::build_malecns` | **Bug to patch first:** line 254 picks `rootSide` before `somaSide`; `rootSide` is NaN for 193,638 of 211,577 bodies, so groups match 0 neurons and `--core-hops` raises `ValueError`. Fall back to `somaSide` (the deep read's `build_mcns2.py` already does this). |
| LIF dynamics (Shiu 2024 parameters, dt 0.5 ms) | **Fixed** | `src/flydrones/brain/lif.py::LIFNetwork` | CPU numpy/scipy only. `copy()` shares wiring, so many brains per process are cheap in memory. |
| Pursuit subgraph | **Fixed after build** | `Connectome.sensorimotor_core(hops, max_neurons)` | Keeps neurons within `hops` downstream of input groups AND upstream of output groups. With inputs = LC10a (plus pC1 arousal candidates) and outputs = steering DNs, `hops=1` keeps all 2-synapse paths (for example LC10a to AOTU to DNa02), so the brain is at most a 2-synapse relay; `hops=2` keeps paths up to 4 synapses. The earlier bench sizes were for different groups; we rebuild and re-bench. |
| **TargetEncoder** (trained) | New | `flyfollow/senses/target.py` | Box to Poisson rates into LC10a cells, see below. Replaces `Retina` and `InputEncoder`. `InputEncoder` maps neurons to cells by index order (no retinotopy), so we write our own mapping. |
| **PursuitDecoder** (trained) | New | `flyfollow/pilot/pursuit_decoder.py` | Replacement of `src/flydrones/motor/decoder.py::MotorDecoder`. Reads only designated steering DNs. |
| DNg02 throttle, fixed cruise, DNp01 escape reflex | **Disabled** for this task | `MotorDecoder` | Altitude is not a brain output here. DNp01 fired at 160 to 400 Hz in the deep read's calibration (escape active on 277 of 400 ticks), so the reflex is off; the governor handles proximity. We state this in the demo. |
| Optic-flow retina and looming | **Off** | `src/flydrones/senses/retina.py` | We do not render pixels in training, so we cannot train with it; running it only at runtime would change the brain's input. The governor's distance rule replaces looming avoidance. |
| **PersonGovernor** (settings randomized, not trained) | New, wraps `src/flydrones/safety.py::SafetyGovernor` | `flyfollow/pilot/governor.py` | Shared by every arm, see below. Passes `brain_age_s` so the brain watchdog actually fires (its default of 0.0 meant it never did in FlyDrones). |
| Tello I/O | Reuse with changes | `src/flydrones/drones/tello.py::TelloDrone` | `TelloDrone.send` expects normalized values in [-1, 1] and multiplies by `stick_percent` (default 60). Our controllers output final integer sticks (-100..100), so the Tello I/O calls `send_rc_control` directly (or builds `TelloDrone(stick_percent=100)` and divides by 100); the 60 % scale is never applied twice. Owns the single command queue (Section 3.1). Dry run until `--send`. It reports no x/y, so FlyDrones' geofence never applies; our distance rules replace it. |

**PersonGovernor (all arms, all settings in final sent stick units).** Putting these in one shared place means the arms differ only in how they produce yaw and forward:
- Up/down: P control on the target's image row, `ud = clip(60 x (cy_ref - cy) / H, -30, 30)` (keep the head slightly below center so the camera looks past the user's shoulder; keep an object slightly below center during APPROACH), with the altitude floors in Section 5.3.
- Left/right: 0.
- Lost target: after the Kalman filter's 0.5 s hold, fb = 0 and yaw toward the last bearing at 20; after 5 s, hover and say "I lost you"; after 10 s, land (only in the modes listed in Section 2.1).
- Minimum distance to the user or object, max forward and reverse stick, slew limits, battery, and the brain watchdog.
- Intervention flags: safety interventions (min distance, lost-target takeover, watchdog) are reported separately from clamps and slew limits.

**Encoder (25 parameters).** Input: the latency-compensated box (center x, center y, height) of the target, the target's reference size, and the governor settings.

- Bearing error: theta = atan((cx - cx0) / fx) minus the side-offset bearing (Section 4.4). The side offset is applied here, so the fly always tries to center the target.
- Normalized apparent size: s = h_px / h_ref, where h_ref = fy x H_target / Z_ref. s = 1 at the standoff distance, for any target. This is what makes one controller work for heads and bottles, and it means distance is encoded by the adapter, not by the fly.
- Each side's LC10a neurons are split into 8 azimuth bins. Right-side LC10a gets targets in the right visual field and left-side gets the left, with a frontal overlap band where both respond. Bin assignment: by lobula synapse centroid position if the MaleCNS tables carry it (unverified; check at audit), otherwise by rank within side. The left/right mapping is therefore ours; what the fly wiring decides is whether each side's LC10a reaches the DNs that turn toward that side.
- Rate per bin: r_max x tuning(theta, bin center, width) x g(s) x (1 + k_v x |d theta / dt|), where g(s) widens the active spot as the target gets closer (bigger target, more bins).
- Tonic arousal: constant rate into P1. There is no MaleCNS type named "P1"; it is probably among the 156 pC1_* neurons (inferred, not verified). If the audit cannot identify P1, we drop it and use an `arousal_gain` on LC10a input, as FLYGUIDE_SPEC allows.
- Parameters: r_max, tuning width, overlap width, 2 size-curve parameters, velocity gain, 16 per-bin gain multipliers (bounded to 0.5 to 2 so training can correct the rank mapping but cannot silence or swap bins), P1 rate or arousal gain, LC9 gain, LC11 gain.

**Readout (22 parameters).** DN set from FLYGUIDE_SPEC: DNa02 (verified present, one per side), DNa01, DNb05, DNg13 (ipsiversive), DNb06 (contraversive); presence of the last four in MaleCNS is not verified. DNp09 only if the audit shows bilateral LC10 input. Each rate is the mean over all cells of that type on that side.

- Rates are low-pass filtered (trained time constant, bounded to at most 150 ms; we log the lag it adds) and normalized by the rate at hand calibration.
- yaw = 60 x G_yaw x tanh(sum over DN types of w_type x (rate_R - rate_L) + b_yaw)
- forward = S x G_fwd x tanh(sum over DN types of (u_type,L x rate_L + u_type,R x rate_R) + b_fwd), where S = max_fwd_stick for positive values and max_back_stick (20) for negative ones, so the governor's speed clamps never fire. b_fwd sets the no-target behavior (hover). G_yaw and G_fwd are in [0, 1].
- Parameters: 5 yaw weights, 10 forward weights (per type and side), 2 biases, 2 gains, 2 time constants, 1 deadzone.

**Readout noise (expected problem).** `Brain.tick` returns rate = mean spike count x 1000 / ms (`brain/brain.py` lines 100 to 102). With one DNa02 per side and a 50 ms tick, a single-cell rate moves in 20 Hz steps, so yaw at typical rates comes from about 0 to 5 spikes per side per tick. Pooling over DN types helps only as far as the other types exist and fire. The low-pass filter that smooths this adds lag on top of 100 to 700 ms of video latency, which is why its time constant is bounded. If the audit finds single cells for most types, this noise is a likely reason for the fly arm to lose to NOBRAIN, and we will say so.

**Bypass rule.** The encoder may only write into the designated visual input neurons; the readout may only read the designated DNs. There is no direct path from encoder features to motor output. If training finds a solution, it goes through the connectome, although the biases b_yaw and b_fwd can carry some behavior on their own; Section 4.7 checks how much.

**Total:** 47 parameters (25 + 22), in the range where CMA-ES works well.

**Shared preprocessing (same for every arm).** A constant-velocity Kalman filter on (cx, cy, h) that predicts the box forward by `video_latency_s` plus the detector time (`t - t_decoded` of the `det` message), and holds it through dropouts shorter than 0.5 s. All arms get identical inputs and the same governor, so comparisons measure the controller, not the filter or the lost-target policy.

### 4.2 Decision: one controller for FOLLOW and APPROACH

**Decision: yes, unify them.** One trained "pursue a box to a standoff" controller, trained on a mix of person-following and static-object episodes.

Reasons:
- The inputs and outputs are the same (a box in, yaw and forward out), and with the normalized size s both tasks become "center the target and hold s = 1".
- We reuse the same adapter, governor and tests, so there is one thing to train, audit and check on the drone. It halves the work, and we have one night.
- If it works, the fly flies both the follow and the approach in the demo.

Costs and how we handle them:
- A far, small object is detected intermittently; a walking head is detected almost every frame. We include long dropouts and a detection probability that falls with pixel size in the approach episodes.
- Object range from a class size prior has about 20 to 25 % error (research). With the true size between 0.75 and 1.25 of the prior (randomized in training), a 1.0 m standoff ends between 0.75 and 1.25 m from the object, still above Z_min = 0.5 m.
- APPROACH needs a stop and altitude changes. The stop is a governor rule (in band for 2 s), altitude is the governor's P loop with floors (Section 5.3), and the final hover is algorithmic.
- Approach episodes get equal weight in fitness (Section 4.5), so training actually optimizes them.
- **Exit clause:** if on the test approach set the unified controller is worse than the PID servo by more than 20 % on time-to-standoff or has any overshoot collisions, APPROACH uses the PID servo and the fly keeps FOLLOW only.

### 4.3 Go/no-go gate G0: connectome audit (first, before any training)

From FLYGUIDE_SPEC 6.6, adapted. Script: `flyfollow/audit/audit.py`. Time box: 60 minutes.

**Structural audit.** Per side, report:
- direct LC10a to DN synapse counts, and 2-hop paths (LC10a to X to DN) with summed weight (product of signed weights), for each DN in the readout set;
- whether LC10a on the left reaches the DN set that turns left and the right reaches the one that turns right (bilateral pathway), and whether the paths are ipsi- or contralateral;
- the resolved sizes of every input and output group (catching the `rootSide` bug or regex misses), including the number of cells per DN type;
- the P1 or pC1 candidates and their connection to LC10a or the steering DNs.

**Functional audit** (in the LIF sim, pursuit subgraph, 10 minutes):
- A: spot at -30, -15, 0, +15, +30 degrees. Pass if the DN asymmetry (R minus L) changes sign across 0 and is monotonic within noise over 5 repeats. Also report the yaw command SNR (mean over SD across the 5 repeats) at each bearing.
- B: spot at 0 degrees with s = 0.5, 1, 2. Pass if total DN activity changes monotonically with s.
- C: A and B repeated with P1 on and off.

**Outcomes:**

| Result | Action |
|---|---|
| Structural and A and B pass | Train the full controller (yaw and forward from the brain). |
| A passes, B fails | Train a **yaw-only fly**: brain controls yaw, forward comes from the PID's range loop. Honest and still a real closed loop. |
| Structural fails for LC10a | Expand inputs to LC10b to e, LC9, LC11; expand outputs to the DN types with the highest LC10 path weight; rerun. One retry only (30 minutes). |
| Everything fails | No fly controller tonight. Launch the no-brain and PID arms so the pipeline and the chart exist; report the audit result as the fly finding. FOLLOW flies on PID. |

### 4.4 Training environment

File: `flyfollow/rl/env.py`, class `PursuitEnv` with `reset(seed, arm, settings)` and `step()`, built from the body of `run_sim` in `src/flydrones/runtime.py`. We do not render pixels: the encoder input is a box, so we simulate kinematics and project to a box.

**Components.**
- **Drone response model** (`flyfollow/sim/drone_model.py`): first-order lag from sent stick to body velocity and yaw rate (like FlyDrones' `SimDrone`), plus command latency. Later extras: a random-walk hover drift and pitch during acceleration (which shifts the target's image row).
- **Person walker** (`flyfollow/sim/walker.py`): unicycle kinematics. A random sequence of segments: straight walk, turn, stop, speed change, occasional step toward the drone. Head height and head size per episode.
- **Object targets** (`flyfollow/sim/objects.py`): static object at a random height (floor to table), size, size-prior mismatch, start range and bearing.
- **Camera and detector model** (`flyfollow/sim/camera.py`): pinhole projection with a randomized focal length; box center and size noise; independent and burst dropouts; detection probability falling below 20 px of box height; detection rate and latency queue; no box when the target leaves the field of view. Later extra: occasional false boxes.
- **Governor in the loop:** the same `PersonGovernor` as at runtime, with randomized settings, so the controller learns under the governor it will fly with.
- **Brain:** one `LIFNetwork` per episode from the pursuit subgraph file, warm-started for 1 s before the episode (excluded from reward).

**Domain randomization (per episode).** All stick values are final sent units (-100..100). Ranges marked "R0" are placeholders until the R0 measurements; if a measured value falls outside, we widen the range and run a short fine-tune.

| Quantity | Range | Basis |
|---|---|---|
| Forward stick gain | 0.5 to 3.0 m/s at sent stick 100 | Unverified; R0 measures. Research quotes Tello slow mode at 10.8 km/h (3.0 m/s). |
| Yaw stick gain | 50 to 120 deg/s at sent stick 100 | Unverified, no source; R0 measures |
| Governor: max forward stick | 20 to 50 | Tunable after training |
| Person walking speed | 0 to min(1.4, 0.9 x v_max) m/s, where v_max = forward gain x max forward stick / 100 | A person faster than the drone makes the episode unwinnable and the fitness uninformative |
| Segment lengths | straight 2 to 8 s, stop 1 to 6 s | Guess; covers the demo |
| Turn angle, turn rate | 30 to 180 degrees, up to 90 deg/s | Guess |
| Step toward the drone | 0 to 1 per episode | Tests back-off |
| Head height, head size H | 1.50 to 1.90 m; 0.20 to 0.26 m (controller assumes 0.23) | Gives up to about plus or minus 13 % irreducible range bias at the extremes (plus or minus 0.26 m at 2 m) |
| Focal length fx | 650 to 950 px at 960 wide (HFOV about 54 to 72 degrees) | Covers both FOV claims (Section 3.4) |
| Detection latency (capture to box) | 100 to 700 ms | Reports range from 150 ms to 1 s; R0 |
| Detection rate | 8 to 30 Hz | Laptop YOLO; R0 |
| Box center noise | sigma 1 to 5 px | R0, recorded video |
| Box height noise | 3 to 10 % | Research: 5 to 10 % (unverified) |
| Dropout | iid 0 to 15 % per frame, plus bursts of 0.2 to 1.5 s at 0 to 0.2 per s | R0, recorded video |
| False boxes | 0 to 2 % of frames | Guess |
| Response lag tau | 0.2 to 0.8 s forward, 0.1 to 0.4 s yaw | Unverified; R0 |
| Command latency | 20 to 120 ms | Wi-Fi; R0 |
| Hover drift | velocity random walk, sigma 0 to 0.05 m/s | Tello drift complaints, no figures |
| Governor: standoff Z_ref | 1.5 to 2.5 m (follow); approach 1.0 m, or max(1.0, 2.5 x (altitude - object height)) m for low objects (up to 2.0 m) | Tunable after training; approach rule from Section 5.3 |
| Governor: side offset bearing | -10 to +10 degrees | Tunable after training |
| Governor: min distance Z_min | 1.0 to 1.3 m (person), 0.5 m (object) | Safety |
| Follow start | range Z_min + 0.3 m to 4 m, bearing within 25 degrees | Never start inside the too-close penalty |
| Object | size 0.08 to 0.30 m; true size / class prior 0.75 to 1.25; height 0 to 1.0 m with at least 30 % on the floor; start range Z_ref + 0.5 m to 6 m; bearing within 20 degrees | FIND hands off with the object in view; the prior mismatch matches the 20 to 25 % research figure |

The walking speed we expect from the demo user is about 0.5 to 1.0 m/s (estimate for a blind user walking with a guide, not measured); we measure it from the R0 handheld video and report metrics on that subset too.

The side offset works because while the drone trails a walking person with the target held at bearing beta, the drone ends up offset sideways by about Z x sin(beta). At beta = 10 degrees and Z = 2 m that is about 0.35 m. When the user stands still, the offset has no effect, which is harmless.

**Episodes.** 20 Hz ticks. Follow episodes last 60 s (1,200 ticks); approach episodes last up to 25 s and end on success (in band for 2 s). Each candidate is scored on K = 8 episodes: 5 follow, 3 approach. All candidates in a generation see the same 8 seeds (common random numbers), which reduces fitness noise.

**Seed sets.** Three disjoint sets: training seeds (drawn fresh per generation from 10,000 upward), a selection set of 64 fixed seeds (1,000 to 1,063) used only to pick checkpoints, and a test set of 200 fixed seeds (2,000 to 2,199) used only for the final metrics.

### 4.5 Reward

The reward uses the simulator's true state; the controller only sees the noisy, delayed box. Per tick (dt = 0.05 s):

```
r_t = -dt * ( w_r * e_Z  +  w_x * e_x^2  +  w_j * j_t  +  w_g * o_t  +  w_l * l_t  +  w_c * c_t )  +  events
```

The band half-width is b = 0.15 x Z_ref (0.3 m at 2 m), used everywhere: reward, metrics and the R3 and R4 criteria. It is wider than the worst-case head-size bias (plus or minus 13 %), so a good controller can stay in band on every episode. At runtime we can measure the demo user's head once, so the real bias should be smaller.

| Term | Definition | Start weight |
|---|---|---|
| e_Z, standoff band | with d = Z - Z_ref: max(0, abs(d) - b) + max(0, d - b)^2 / (1 m), capped at 3. The added quadratic on the far side offsets the near-side penalties so the optimum is not biased long. | w_r = 1.0 |
| e_x, image position | bearing error / (HFOV/2), relative to the planned bearing (side offset), capped at 1 | w_x = 2.0 |
| j_t, jerk | ((d yaw_stick)^2 + (d fb_stick)^2) / 100^2 per tick, times 20 | w_j = 5.0 |
| o_t, safety intervention | 1 if the governor made a safety intervention this tick (min distance, lost-target takeover, watchdog). Clamps and slew limits are logged, not penalized, since the controller cannot see them. | w_g = 1.0 |
| l_t, target lost | 1 while the target has been out of the detector for more than 1 s | w_l = 5.0 |
| c_t, too close | 1 while Z < Z_min | w_c = 10.0 |
| Event: loss | -10 each time a loss exceeds 1 s | |
| Event: too close | -20 on each entry below Z_min | |
| Event: collision | Z < 0.5 m (person) or 0.2 m (object): -200, the episode ends, and the remaining ticks are charged at the maximum non-jerk rate, w_r x 3 + w_x + w_g + w_l + w_c (21 per second at the start weights, recomputed after rescaling). Staying too close costs at most about 16 per second, so crashing never pays. | |
| Event: approach success | +20 when in band for 2 s | |

**Tuning rule for the weights (smoke test, 20 minutes):** run the hand-tuned PID on 50 training-range episodes and rescale the weights so that each of the first four terms contributes between 10 and 40 % of the PID's total cost. Freeze the weights before launch and never change them between arms.

**Fitness.** Each episode's return is divided by the absolute return of PID-HAND on the same seed (denominator floored at 1; PID-HAND needs no brain, so this costs little). Fitness = 0.5 x mean over follow episodes + 0.5 x mean over approach episodes, so approach behavior counts as much as follow. We log each reward term separately so we can see what the optimizer is buying.

### 4.6 Optimizer and Modal

**Why CMA-ES.** The spiking brain is not differentiable, the parameter count is 47, and fitness is noisy. CMA-ES handles all three and needs no gradient. OpenAI-style ES (Salimans 2017) is the alternative for thousands of parameters; we do not have that many. We use `pycma`.

**Settings.**
- Parameters normalized to [0, 1] (log scale for gains and rates), sigma0 = 0.2.
- Population 32 (the default for n = 47 is 4 + floor(3 ln 47) = 15; we double it because fitness is noisy).
- K = 8 episodes per candidate, common seeds per generation.
- Up to 150 generations, or until the morning.
- Every 10 generations: evaluate the distribution mean on the 64 selection seeds and keep the best mean seen (the "best checkpoint"). The same rule applies to every trained arm.
- CMA state pickled to the Modal Volume every generation, so a crashed run resumes.

**Runs.** 3 CMA seeds per trained arm (FLY-CMA, FLY-SHUF, NOBRAIN, PID-CMA), 12 runs. FLY-SHUF uses a different shuffled connectome for each seed. One run per arm cannot show run-to-run optimizer variance, and budget is not the constraint.

**Initialization from hand calibration.** We extend `flydrones calibrate` (`src/flydrones/calibrate.py`, which today fits a ridge regression for throttle and yaw from 7 synthetic stimuli) with target stimuli: a spot at 9 bearings times 3 sizes. We record DN rates and ridge-regress the desired yaw (proportional to bearing) and forward (proportional to 1 - s) on them. That gives the initial readout weights; encoder initial values are hand-set from FlyDrones defaults (`max_hz`). FLY-SHUF (on its shuffled DN rates) and NOBRAIN (on its pooled encoder features) get the same ridge fit on the same stimuli. PID-CMA starts from the PID-HAND gains. The hand-calibrated fly is also an evaluation arm (4.7), so we can see what training added.

**Parallelism: CPU fan-out, not GPU.** FlyDrones' `LIFNetwork` is numpy/scipy on CPU and each brain holds a single state, so batching many variants on one GPU would need a new batched backend ((B, n) state arrays, or a port of a PyTorch Shiu model). That is several hours of work for a speedup we do not need at 47 parameters. We fan out episodes over many single-core Modal containers instead. A batched GPU backend is a stretch item only if the smoke test shows CPU throughput is too low.

**Modal layout** (`flyfollow/rl/modal_app.py`):
- Image: Python 3.12, `numpy`, `scipy`, `pandas`, `pyarrow`, `pyyaml`, `cma`, plus FlyDrones installed from our fork (commit `3e26934` plus the `somaSide` patch).
- Volume `flyfollow-data`: the pursuit subgraph `.npz` (built locally from the MaleCNS data already downloaded in the scratchpad `mcns/` folder, so Modal never downloads the 1.1 GB), the three shuffled subgraphs, checkpoints, logs.
- `evaluate(params, arm, seed, episode_kind)`: 1 CPU, about 2 GB memory, runs one episode, returns the return and the per-term breakdown. Automatic retries on container failure.
- `train(arm, run_seed)`: the CMA-ES driver, itself running on Modal (so laptops can close), calling `evaluate.map` over population x K = 256 items per generation. On restart it resumes from the last pickled CMA state. Timeout set to the maximum Modal allows (24 h per the docs, unverified).
- Monitoring: a Modal failure notification or budget alert, plus a phone alarm at H+9 for one person to check `modal app list` and resume anything that died.

**Wall-clock and cost: estimates, to be replaced by the smoke test.**

| Quantity | Estimate | How we get it |
|---|---|---|
| Wall time per simulated second, pursuit subgraph plus env | about 0.4 to 1.6 s on one Modal core | Brain only: 0.37 s (`flydrones bench`, 17.8k neurons). Full FlyDrones loop with retina and physics, which we do not run: 0.51 s at 17.8k, 0.53 s at 20k (`loop_timing.py`). Both on Parth's laptop (Section 3.5). One Modal CPU may be slower single-threaded (unverified). Our subgraph size is unknown until built. |
| Per episode (60 s follow) | about 30 to 100 s | Smoke test logs it |
| Per generation, wall | about 1 to 3 min with 256 parallel containers per run (plus startup) | Smoke test |
| 150 generations, wall | about 3 to 7 h | Fits in the night if the container cap allows 6 fly runs at once (open question 9) |
| Container-hours per fly run | about 300 to 1,000 | 256 episodes x 150 generations x per-episode time |
| Cost per fly run | about $20 to $80 at CPU plus memory, roughly $0.06 to $0.07 per container-hour (unverified; Modal also bills startup and idle time in the scaledown window; check modal.com/pricing) | Smoke test measures the cost of one generation |
| All 12 runs | about $130 to $500 (6 fly runs; NOBRAIN and PID-CMA run no brain and should cost a few dollars each, unverified) | Of the $10k credits |

Launch order: seed 1 of every arm, then seeds 2 and 3. If the smoke test projects that all runs cannot reach 150 generations by H+13 (container cap) or that the measured cost exceeds $500, seeds 2 and 3 run 100 generations, and the report says the seed spread comes from shorter runs. If the projected wall time for seed 1 exceeds 8 hours, we cut the population to 24 and K to 6 before launch, not after.

### 4.7 Baselines, honesty controls and metrics

All arms use the same shared preprocessing, governor (including lost-target and up/down logic), environment, seeds, and (for trained arms) the same CMA-ES budget and initialization procedure.

| Arm | What it is | Parameters | Question it answers |
|---|---|---|---|
| FLY-HAND | Fly controller, hand-calibrated only, no CMA-ES | 47 hand-set | What does training add? |
| **FLY-CMA** | Fly controller, trained (main arm), 3 seeds | 47 trained | Does a fixed connectome plus a trained interface follow well? |
| FLY-SHUF | Same, but the pursuit subgraph is replaced by a degree-preserving shuffle (each neuron keeps its in-degree and out-degree, edge signs shuffled within excitatory and inhibitory sets), same encoder and readout neuron IDs; one shuffle per seed, 3 in total | 47 trained | Does the specific fly wiring matter beyond its degree statistics? (Dhiman 2026 reports that advantages often vanish under this control; unverified, not re-checked.) |
| NOBRAIN | Same encoder; per-side bin rates pooled into 5 features per side and mapped straight to yaw and forward with the same tanh readout form, the same bounded trained low-pass filters, and the same ridge initialization; 3 seeds | 47 trained | Does the brain help at all versus a direct controller? |
| PID-HAND | Classical follower (4.9), hand-tuned | 4 hand-set | The standard engineering answer |
| PID-CMA | Same PID with gains tuned by CMA-ES, same budget, 3 seeds | 4 trained | Fair classical baseline |

**Shuffle check.** Before launch, each shuffled graph must have at least as many input-to-output paths within the hop limit as the real subgraph; if not, we regenerate it. Otherwise FLY-SHUF could fail trivially (no LC10a to DN path) rather than informatively.

**Our prediction, stated before training:** PID-CMA and NOBRAIN will be at least as good as FLY-CMA on every metric, and the FLY-CMA versus FLY-SHUF difference will fall inside the interval across run seeds. If the fly arm wins, that is interesting; if not, we show the chart anyway.

**Brain-use checks on FLY-CMA** (best checkpoint, test set):
- Lesion: clamp each DN rate to its episode mean and rerun. If the brain is doing the steering, in-band time should collapse.
- Bias audit: report |b_yaw| and |b_fwd| relative to the typical magnitude of the DN-driven terms inside each tanh.

**Metrics** (on the 200 test seeds). Headline numbers are the mean across the 3 run seeds with the range across seeds; a 95 % bootstrap interval over test episodes within each run is secondary.
- Follow: fraction of time in band (abs(Z - Z_ref) at most 0.15 x Z_ref and target in view), mean signed range error, RMS range error, RMS bearing error, loss events per minute, minimum distance, safety interventions per minute, yaw jerk.
- Approach: success rate, time to standoff, overshoot (minimum distance minus Z_ref), with floor objects reported separately.
- Demo-conditions subset: latency, stick gains and FOV set to the R0 values, governor at demo settings. Used for G2.
- Stress set: latency 800 ms, dropout 30 %, focal length at the range edges.
- Real flights: the same metrics from logs. Range comes from head size, which is the controller's own estimator, so at least one R3 or R4 run also gets independent range ground truth from tape marks on the floor or a second camera.

### 4.8 Sim-to-real steps

| Step | When | What | Pass criterion |
|---|---|---|---|
| R0 Measure | H+14 | Handheld Tello video of a teammate walking at 1.5 to 2.5 m (recorded tonight if possible). Stationary target for 30 s. Stopwatch-on-screen test for end-to-end latency, stored as `video_latency_s`. One short flight: `rc 0 30 0 0` for 2 s, `rc 0 0 0 40` for 2 s, logging `vgx` and `yaw` (also confirms the `vgx` unit and frame). Checkerboard calibration. | Measured box noise, dropout rate, latency, stick gains and lags are inside the training ranges. If not, widen the ranges and start a 1 to 2 h fine-tune from the best checkpoint on Modal while the next steps continue with PID. |
| R1 Replay, open loop | H+14.5 | Run perception, filter, encoder, brain and readout on the recorded video, no drone. | Yaw command sign correct on 100 % of frames with abs(bearing) > 10 degrees; brain tick p95 < 45 ms with YOLO running on the same laptop. |
| R2 Props-off | H+15 | Drone on a table with **propellers removed**, `--send` on, a teammate walks in front. Check channel mapping and stick units, watchdogs, lost-target behavior, `land` and `emergency` keys. | Motors respond in the right direction for each channel; watchdog stops commands within 1 s of killing the brain process; emergency key works. |
| R3 Netted or cleared-area flight | H+16 | Spotter with kill key, user stands still then takes steps, one 1 s occlusion (hand over head). PID-HAND first, then FLY-CMA. | Holds the standoff within 0.15 x Z_ref for 30 s standing; reacquires after the occlusion; no governor hard stop; never closer than 1.0 m. |
| R4 Real follow | H+17 | 10 m walk, one 90 degree turn, one stop. Three runs each for FLY-CMA and PID (two each if we have fewer than 3 batteries). | FLY-CMA: every run with no loss longer than 2 s, minimum distance at least 1.0 m, in band at least 70 % of the time. This is gate G3. |

### 4.9 Fallback: the classical follower

This is the common control law from open-source Tello followers (a widely used Tello face-tracking course and the GitHub projects in Section 11), written against the same box input and governor so it can swap in by config. All values are final sent stick units (-100..100).

```
e_x  = (cx - cx_ref) / W                       # cx_ref includes the side-offset bearing
Z    = fy * H_head / h_px                      # range from head size (control on range, not raw area)
yaw  = clip(Kp_y * e_x + Kd_y * de_x/dt, -60, 60)
fb   = clip(Kp_f * (Z - Z_ref), -20, max_fwd_stick)   # zero inside deadband d0
ud, lr, lost-target: PersonGovernor (shared with every arm)
start: Kp_y = 100, Kd_y = 20, Kp_f = 40 per m, d0 = 0.15 m   (the 4 parameters PID-CMA tunes)
```

Known failure modes from the research and how this handles them: latency oscillation (low gains, derivative damping, Kalman prediction), box jitter surging (EMA on h, deadband), pitch coupling (ignore frames with abs(pitch) above 8 degrees for range), several people in view (track ID lock from perception), no rear sensing (reverse capped at stick 20).

**Checkpoint rule:** the demo uses FLY-CMA for FOLLOW only if it passes G2 in sim and G3 on the real drone. Otherwise PID flies and the fly result is shown from simulation and recorded runs.

---

## 5. FIND, and the handoff to APPROACH and GUIDE

### 5.1 The RL versus algorithm debate, and the decision

Arguments made by the team:
- **For RL** (the old PLAN.md): object-place priors (bottles on tables) and a sequential choice of where to look next are learnable in a 2D room sim.
- **Against RL** (one teammate): the policy has no information about an unseen room, so an exploration or coverage algorithm, or Bayesian search, is enough.
- **Concern about coverage:** a sweep can miss small objects that are too far away to resolve.

Evidence from the research:
- In one room, an in-place 360 degree scan sees almost everything; the failures are detection range, occlusion and the floor blind zone. A policy fixes none of those.
- Learned ObjectNav gains (SemExp: 54.4 % versus 40.3 % for a classical map plus frontier) come in multi-room houses with perfect depth and pose; perception errors cost about 19 points, more than the policy gained. Aerial ObjectNav is unsolved (UAV-ON best baseline 7.3 % success with four RGB-D cameras).
- Frontier exploration, coverage control, lawnmower and next-best-view all need a pose and a map we do not have.
- The coverage concern is real. Pixel height of a target is f x H / Z. At imgsz 640 (f about 614): a bottle (22 cm) is 32 px tall at 4.2 m, a cup (10 cm) at 1.9 m, a phone lying flat at about 1 m. A nano model is reliable for a bottle out to about 2.5 to 3.5 m (estimate, not measured). Running at imgsz 960 or on 2 tiles while hovering extends this to about 4 to 5 m.

**Decision: FIND is an algorithm, not RL.** A scripted scan-hop-scan that ranks where to go next by an object-furniture prior discounted where we already had a good look (a miss-discounted prior ranking, not a full Bayesian search). It answers the coverage concern with four things: YOLO on every frame with higher-resolution confirmation while hovering, a 1.0 m search altitude, hops toward furniture that likely holds the object, and an optional low floor scan. We stop working on search RL; see 5.6 for the only case we would revisit it.

### 5.2 The search algorithm

`flyfollow/runtime/find.py`. All angles are relative to the Tello yaw at FIND start. YOLO11n at imgsz 640 runs on every frame throughout, including while turning.

1. **Prepare.** Store the user's bearing and range (from the head box) in the drone frame. Say "Stay where you are, I am looking for your water bottle." Descend to search altitude 1.0 m (table tops visible beyond 0.65 m, floor beyond about 2.6 m).
2. **Scan.** 8 steps of 45 degrees (with HFOV 55 degrees that leaves about 10 degrees of overlap; if calibration shows 70 degrees, use 6 steps of 60). At each step: yaw, wait until the yaw rate is below 5 deg/s, discard frames decoded less than `video_latency_s` + 100 ms after that (they can still show the scene from before the rotation ended), then take 3 frames at imgsz 960 (or 2 tiles at 640). About 2.5 to 3 s per step, 20 to 25 s per scan. A 640 match of the target class at any moment stops the scan early: yaw to its bearing, settle, and confirm at 960.
3. **Record per step:** target detections (class match, confidence at least 0.35), furniture detections with bearing and rough range, and the reliable range for the target class in that sector: R_rel = fy x H_target / 32 px.
4. **Confirm.** Target in 2 of 3 settled frames at the same bearing (within 5 degrees): go to APPROACH with the stored bearing and range estimate.
5. **Hop.** If not found, score each furniture item seen: prior(target | furniture class) x (1 if its range is beyond R_rel else 0.3) / travel time. Priors come from the PLAN.md table (for a water bottle: table 0.4, counter 0.3, desk 0.2, floor 0.1; these are our guesses, not measured statistics). Fly to the best one: yaw to its bearing, then one `forward` move (through the command queue) or timed PID forward flight, of length range minus 1.5 m. Range comes from the ground-plane position of the furniture's foot (visible beyond 2.6 m at 1.0 m altitude, using the Tello's `h`) or, if the foot is hidden, from a class size prior. We do not use the size-based pursuit controller here: furniture has no reliable size prior and its box is cut off at the frame edge well before 1.5 m. Then scan a 120 degree sector (3 steps).
6. **Low floor scan.** If the target class can be on the floor and budget remains, descend to 0.5 m and do one 360 degree scan (floor visible beyond about 1.3 m). The floor prior of 0.1 applies only here. Without this step, floor within about 2.6 m of every scan point is a known FIND miss, and we say so.
7. **Occlusion pass.** If budget remains, one rescan from the opposite side of the most likely furniture.
8. **Budget.** At most 3 hops, 120 s, or battery below 35 %, whichever comes first.

Open vocabulary: COCO covers bottle, cup, cell phone, remote, book, backpack and laptop. For anything else the perception owner switches to YOLOE or YOLO-World with the spoken class as the prompt (CPU cost estimated at 7 to 11 fps for YOLO-World-S; not measured).

### 5.3 Handoff to APPROACH, and "not found"

- **To APPROACH:** the object's bearing psi = yaw + atan((u - cx) / fx) and range Z = fy x H_class / h_px (plus or minus about 20 to 25 % from the size prior). The drone yaws to psi, then the pursuit controller takes over with the object box as the target.
- **Altitude and standoff.** The governor's P loop keeps the object slightly below image center, but never below an altitude floor of 0.8 m (0.5 m if the object was found in the low scan). Without the floor, the drone would descend to about 0.3 m over a floor object, the edge of the Tello's vision-positioning range. With the floor, a low object leaves the lower half of the field of view (21.5 degrees) when it is closer than about 2.5 x (altitude - object height). So Z_ref = max(1.0, 2.5 x (altitude - object height)) m, with object height estimated from its image row and range: 2.0 m for a floor object at 0.8 m altitude, 1.25 m at 0.5 m, 1.0 m for a table top. Training includes floor-object episodes with this rule (Section 4.4).
- **Target lost.** If the box is lost for more than 2 s, yaw back to psi and rescan a 90 degree sector.
- **Not found:** RETURN. Yaw toward the stored user bearing, reacquire the user visually (sweep plus or minus 60 degrees, then a full scan), and say "I could not find your water bottle. I checked the table and the desk. Say 'look again' to keep searching." Resume FOLLOW.

### 5.4 FACE_PERSON and the hover pose

**FACE_PERSON.** Climb to 1.2 m, yaw toward the stored user bearing, reacquire the user's full-body box. Dead reckoning of the drone's own displacement during FIND is expected to be off by meters over tens of seconds (`vgx`/`vgy` are 0.1 m/s integers at 10 Hz, frame unverified), so the visual sweep (plus or minus 60 degrees, then a full scan) is the primary method, not a backup. Record the object's position relative to the drone at this moment.

**Default pose (in scope).** Hold at the APPROACH standoff from the object at 1.2 m altitude, facing the user. The drone then cannot see the floor within about 3.1 m of itself, which is the last stretch of the user's path to the object. That blind zone is covered by the glasses ToF (VL53L1X) and the REACH stage, and we report it to the guidance owner via `scene.visible_floor_min_range_m`.

**Overwatch planner (stretch, only after G4 passes).** Move to a hover point where the user, the object and the floor between them are in view. Room size needed: the user and the object, separated by s, must both fit inside the HFOV minus a 5 degree margin on each side (45 degrees at HFOV 55). With the drone on the perpendicular bisector, its distance from their midpoint must be at least (s/2) / tan(22.5 deg), about 1.2 x s, and it must be at least 2.6 m from each of them (floor visible at 1.0 m altitude). Worked example: s = 3 m needs about 3.6 m from the midpoint, so with clearance around the pair and behind the drone the room must be roughly 6 to 7 m across. In rooms under about 6 m the default pose is the expected case. If attempted: evaluate hover points left of the user-to-object line, right of it, and beyond the object; keep those that meet the constraints above, stay at least 1.5 m from the user, and can be reached by first yawing to face the direction of travel; pick the shortest; move with timed PID flight and confirm visually, then yaw to face the midpoint and descend to 1.0 m.

**Camera height without the ToF.** When the user is fully in view (range beyond about 2.6 m), we use the user as the scale reference. With the user's known height H_p (entered once as a setting) and the pixel angles of the feet (alpha_f, below the horizon) and the top of the head (alpha_h):

```
Z = H_p / (tan(alpha_f) - tan(alpha_h))      # range to the user
a = Z * tan(alpha_f)                          # camera height above the floor
```

(alpha_h is negative when the head is above the camera.) This gives range and camera height with no size prior error and no ToF. Accuracy not measured yet; we will check it against tape marks in H+18 to H+21.

### 5.5 GUIDE and REACH (not ours, but our output feeds them)

While in GUIDE, the drone holds the hover pose. The yaw loop keeps the user in view (PID yaw hold, forward disabled, brain off to free CPU for obstacle detection) and the governor backs off if the user comes within 1.2 m. We publish `scene` at 10 Hz (Section 6). The guidance owner projects obstacle boxes to the ground plane using `scene.cam`, runs A* on a 10 cm grid in the drone frame (unknown cells are not free, as in FLYGUIDE_SPEC), and turns the path into taps and speech. REACH runs on the glasses. On-device YOLO on an ESP32-CAM is unlikely to be practical (unverified), so its frames must reach the laptop, and the Tello AP will not carry them (open question 10).

### 5.6 Optional RL search stretch

Only if the scripted search fails in at least 2 of 5 real trials **because it picked the wrong furniture** (not because of detection range). Then: a high-level policy over {scan, hop to furniture i, rescan opposite side} on abstracted detections, trained in a 2D abstract-room sim on CPU in minutes, benchmarked against the scripted search. Otherwise we skip it. It ranks below everything in Section 8.

---

## 6. Interfaces

**Transport.** ZeroMQ PUB/SUB on `tcp://127.0.0.1`, one JSON object per message, every message has `topic` and `t` (laptop `time.time()`, seconds). All drone-side processes run on the laptop connected to the Tello. Frames are drone-centric: `drone_level` means origin at the drone, x forward, y left, z up, with pitch and roll removed using the Tello attitude. Units: meters, degrees, pixels of the 960x720 frame (never letterboxed coordinates). No world frame exists anywhere.

**Detections in** (perception owner to us). `t_decoded` is when the frame was decoded; `t - t_decoded` is detector time only. End-to-end video latency is the config constant `video_latency_s` from the R0 stopwatch test.

```json
{"topic": "det", "t": 1727380000.18, "t_decoded": 1727380000.12, "frame_id": 812, "src": "tello",
 "img_w": 960, "img_h": 720, "imgsz": 640,
 "dets": [
   {"cls": "person", "conf": 0.91, "bbox": [402, 180, 560, 715], "track_id": 3},
   {"cls": "person_head", "conf": 0.84, "bbox": [455, 180, 512, 250], "track_id": 3},
   {"cls": "bottle", "conf": 0.52, "bbox": [701, 402, 716, 441], "track_id": 17}
 ]}
```

`person_head` shares the `track_id` of its person box. Which track is the user is set by `lock` (below).

**Mode commands in** (voice or operator to us):

```json
{"topic": "mode_cmd", "t": 1727380012.0, "mode": "FIND", "source": "voice",
 "target": {"cls": "bottle", "prompt": "water bottle", "height_m": 0.22}}
{"topic": "lock", "t": 1727380001.0, "track_id": 3}
{"topic": "settings", "t": 1727380000.0, "follow_distance_m": 2.0, "side_offset_deg": 8,
 "max_fwd_stick": 35, "min_person_dist_m": 1.2, "search_alt_m": 1.0, "user_height_m": 1.72}
```

Modes: `FOLLOW`, `FIND`, `APPROACH`, `FACE_PERSON`, `OVERWATCH`, `GUIDE`, `RETURN`, `HOLD`, `LAND`.

**rc out** (controller to Tello I/O; also logged). `rc` values are the final integers sent to the Tello (-100..100). The Tello I/O calls `send_rc_control` directly, or builds `TelloDrone` with `stick_percent=100` and divides by 100; the 60 % scale is not applied twice. Every stick range in this document is in these units.

```json
{"topic": "rc", "t": 1727380000.20, "lr": 0, "fb": 22, "ud": -5, "yaw": 14, "src": "fly",
 "gov": {"safety": false, "clamped": false, "reasons": []}, "brain_tick_ms": 24.1}
```

The Tello I/O owns the only command queue to the drone. For a distance move it stops the rc stream, sends the move, waits for "ok" or a timeout (then sends `rc 0 0 0 0`), and resumes rc.

**Tello state** (Tello I/O to everyone, 10 Hz):

```json
{"topic": "tello_state", "t": 1727380000.21, "yaw_deg": 12, "pitch_deg": -2, "roll_deg": 1,
 "vgx_dms": 3, "vgy_dms": 0, "vgz_dms": 0, "h_cm": 100, "tof_cm": 98, "bat_pct": 64, "temph_c": 71}
```

**Mode changes and events out** (us to guidance and audio):

```json
{"topic": "mode", "t": 1727380031.5, "from": "FIND", "to": "APPROACH", "reason": "bottle confirmed 3/3 frames"}
{"topic": "say", "t": 1727380031.5, "text": "I found your water bottle.", "priority": 2}
```

**Scene handoff** (us to guidance, 10 Hz from FACE_PERSON onward):

```json
{"topic": "scene", "t": 1727380060.0, "frame": "drone_level", "pose": "default",
 "cam": {"fx": 921, "fy": 919, "cx": 460, "cy": 351, "alt_m": 1.21, "alt_src": "user_height"},
 "person": {"track_id": 3, "bbox": [390, 210, 470, 520], "bearing_deg": -4.2, "range_m": 3.4,
            "range_sd_m": 0.15, "xy_m": [3.39, 0.25]},
 "object": {"cls": "bottle", "bbox": [610, 330, 622, 362], "bearing_deg": 180.0, "range_m": 1.0,
            "range_sd_m": 0.25, "xy_m": [-1.0, 0.0], "z_m": 0.75, "src": "dead_reckoned", "age_s": 4.0},
 "person_to_object_m": [-4.39, -0.25],
 "visible_floor_min_range_m": 3.1}
```

`object.src` is `visual` when the object is in view and `dead_reckoned` otherwise (with `age_s`); in the default pose the object is behind the drone. The user's heading is not in this message: the drone cannot tell it reliably from a head box. The guidance owner should take heading from the glasses IMU or AirPods head tracking, or from the user's track motion (open question 4).

---

## 7. TONIGHT checklist (launch training before sleeping)

Goal: all 12 runs on Modal by **H+6** (target), **H+7** hard cutoff. Estimated times are for focused work. Items marked "agent" are drafted by a coding agent in parallel and reviewed by the named person.

| # | Who | Task | Est. | Done when |
|---|---|---|---|---|
| 0 | Both | Freeze `Controller.act(box_state, settings) -> (yaw, fb)`, the encoder feature vector and the env interface in `flyfollow/interfaces.py`, with stubs, so items 4, 5 and 6 proceed in parallel | 15 min | Stubs importable |
| 1 | Both | Create `flyfollow/`; vendor our FlyDrones fork under `third_party/FlyDrones` at commit `3e26934`; Python 3.12 venv; run its 30 tests. Copy `build_mcns2.py`, `loop_timing.py`, `readout_core1.json` from the scratchpad. | 20 min | Tests pass |
| 1b | the teammate (only he has Modal) | Modal hello world: build the image, run one function, set the budget alert at $500 and failure notifications | 15 min | Function returns from Modal |
| 2 | the teammate | Patch `build_malecns` (`connectome.py` line 254) to fall back to `somaSide`. Add input groups (LC10a per side, pC1 candidates, LC9, LC11) and output groups (DNa02, DNa01, DNb05, DNg13, DNb06 per side) to `configs/pursuit_malecns.yaml`. Build `sensorimotor_core(hops=1)` and `hops=2`; run `flydrones bench`. | 45 min | Group sizes non-zero; subgraph `.npz` files; ms per 50 ms tick |
| 3 | the teammate | Audit (Section 4.3): structural report, functional tests A (with SNR), B, C. **Gate G0.** | 60 min | Decision in `docs/audit_result.md`: full, yaw-only, expanded, or no fly |
| 4a | Parth + agent | Minimal `flyfollow/sim/`: drone model, walker, objects, camera and detector model with latency queue and dropouts; `PersonGovernor`; `PursuitEnv`; reward with per-term logging; the three seed sets | 90 min | 100 random episodes run with a dummy controller; trajectory plots look sane |
| 4b | Agent, Parth reviews | Env extras: false boxes, hover drift, pitch coupling | 30 min | Same test passes with extras on |
| 5 | Parth | PID and NOBRAIN controllers against the item 0 interface. Run PID-HAND on 50 episodes; rescale reward weights (Section 4.5) and freeze them. | 30 min | PID in band more than half the time on easy settings; weights frozen in the config |
| 6 | the teammate | `TargetEncoder`, `PursuitDecoder`, hand calibration (extended `calibrate.py`), ridge initialization for FLY, FLY-SHUF and NOBRAIN; three shuffled subgraphs with the path check | 75 min | FLY-HAND runs an episode end to end in `PursuitEnv`; shuffled files pass the degree and path checks |
| 7 | Agent, Parth reviews | `flyfollow/rl/params.py`, `cma_train.py` (pycma ask/tell, checkpoint, selection eval every 10 generations, resume), `modal_app.py` (image, Volume, `evaluate` with retries, `train`). Upload subgraph files. | 45 min | Local run of 2 generations, population 4, K = 1, 10 s episodes |
| 7b | Agent, Parth reviews | `flyfollow/rl/eval.py` (test, selection, stress and demo-conditions sets; lesion test) and `chart.py` | 30 min | Tested on FLY-HAND tonight, chart renders |
| 8 | Both | Modal smoke test: one full-size generation per arm. Record per-episode wall time, container startup and the measured cost per generation; project the night; adjust population, K and generations (Section 4.6). | 30 min | Numbers written in `docs/training_log.md` |
| 9 | Both | Launch detached, seed 1 of each arm first, then seeds 2 and 3 | 10 min | 12 app IDs noted in `docs/training_log.md` |
| 10 | Both | Watch 3 generations: fitness finite and changing, checkpoints written, no container errors. Write the resume command. Set the H+9 alarm. | 15 min | Then sleep |
| 11 | Whoever is free | Optional: Tello Wi-Fi and firewall check on the runtime laptop; 3 minutes of handheld Tello video of a teammate walking at 1.5 to 2.5 m for R0 and for the perception owner | 20 min | Video in `recordings/` |
| 12 | Coding agents, overnight | Scaffold `flyfollow/runtime/`: `tello_io.py` (command queue, stick units, telemetry fix), `bus.py` (ZeroMQ), `modes.py` (state machine of Section 2.1), `controller_runner.py` (PID and fly behind one interface, Kalman filter, `det` intake), `find.py`, and a replay-from-recording mode. Tested on recorded or synthetic `det` only, no flight. | overnight | Replay of a recording runs all modes without errors |

**Hard cutoff at H+7.** Whatever is ready launches. If the fly controller is not ready, launch NOBRAIN and PID-CMA (so the pipeline and chart exist) and FLY-HAND evaluation only; the fly arm launches first thing in the morning for a shorter run.

Commands (check flags against the Modal docs):

```
modal setup
modal volume create flyfollow-data
modal volume put flyfollow-data data/brains/pursuit_core1.npz /brains/pursuit_core1.npz
modal run flyfollow/rl/modal_app.py::smoke --arm fly
modal run --detach flyfollow/rl/modal_app.py::train --arm fly --seed 1 --gens 150
modal app list
modal app logs <app-id>
```

---

## 8. Schedule

H+0 is the start of this plan (hackathon hour about 7). The plan ends at H+28 with one hour of slack before the end.

| Time | Parth | the teammate | Gate |
|---|---|---|---|
| H+0 to H+1 | Items 0, 1, 1b. Ask organizers whether indoor flight near people is allowed. Confirm the guidance owner and the glasses links (open questions 4, 10). | Items 0, 1, start item 2 | |
| H+1 to H+3 | Items 4a and 5; agents on 4b, 7, 7b | Items 2, 3 | **G0** at H+3 (audit) |
| H+3 to H+6 | Review items 7, 7b; items 8 to 10; start item 12 agents | Item 6; items 8 to 10 | **G1** at H+6 target, H+7 cutoff (runs launched, measured cost projection under $500) |
| H+6 to H+13 | Sleep; alarm at H+9 to check and resume runs | Sleep | |
| H+13 to H+14 | Evaluate all arms on the test and stress sets; chart; lesion and bias checks | Review and fix the overnight runtime code (item 12) | |
| H+14 to H+16 | R0 measurements, camera calibration, R1 replay; G2 on the demo-conditions subset with R0 values; fine-tune on Modal if R0 is outside the ranges | FIND scan on recorded video; R2 props-off | **G2** at H+15.5 |
| H+16 to H+18 | R3 and R4 flights (PID first, then fly) | FACE_PERSON and the default pose on the drone | **G3** at H+18 |
| H+18 to H+21 | FIND, APPROACH, FACE_PERSON end to end in the demo room; record the backup demo by H+21 | `scene` messages to the guidance owner; user-height range check against tape marks. If there is still no guidance owner: minimal cue rules from `scene` (straight-line guidance with an obstacle stop, no A*). | **G4** at H+21 |
| H+21 to H+24 | Integration trials with glasses and audio; re-record the backup if the run improves | Same; failure playbook (lost user, video drop, low battery); overwatch planner only if G4 passed | |
| H+24 to H+25 | Code freeze at H+25; final charts | Optional glasses tapping RL only if G4 passed early | Freeze |
| H+25 to H+28 | Slides, rehearsal, battery charging plan | Same | |

**Gates.**
- **G0 (H+3), audit.** Outcomes and actions as in Section 4.3.
- **G1 (H+6 target, H+7 cutoff), launch.** Pass: at least FLY-CMA or NOBRAIN plus PID-CMA running, projected to finish by H+13, measured cost projection under $500. Fail: shrink population, K, generations and seeds 2 and 3 until it fits; never go to sleep without a run.
- **G2 (H+15.5), sim result.** On the demo-conditions subset of the test set: FLY-CMA in band at least 70 % of the time, mean signed range error within 0.15 m, no collision episodes, and minimum distance and loss rate no worse than the lower edge of PID-CMA's range across seeds. Fail: PID is the FOLLOW controller for all real work; the fly is shown in sim. Either way the chart is made.
- **G3 (H+18), real follow.** R4 criteria in Section 4.8, with independent range ground truth on at least one run. Fail: PID flies FOLLOW in the demo.
- **G4 (H+21), end to end.** FIND to APPROACH to FACE_PERSON to a valid `scene` works in at least 2 of 3 runs with any controller. Fail: the live demo shows FOLLOW and the find sequence comes from the recorded backup.

**Flight-minute budget (estimates; battery life 8 to 10 min with video, unverified).**

| Step | Flights | Minutes |
|---|---|---|
| R0 | 1 short flight | 3 |
| R3 | PID, then fly | 6 |
| R4 | 3 runs x 2 controllers at about 1.5 min | 9 |
| G4 | 3 end-to-end runs at about 4 min | 12 |
| Backup recording | 2 runs at about 4 min | 8 |
| Total | | about 38, or 4 to 5 battery charges |

If we have fewer than 3 batteries, R4 drops to 2 runs per controller and G4 to 2 runs.

**If flight permission is refused.** We fly R3 and R4 in whatever space is permitted (a netted area or outside), the demo becomes the recorded backup plus the sim chart, and the backup is recorded by H+21 regardless.

---

## 9. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Organizers do not allow flying indoors near people | Unknown | Blocks the live demo | Ask at H+0; fly tests in whatever space is permitted; backup recorded by H+21; demo falls back to the recording plus the sim chart |
| Tello Wi-Fi flaky, firewall blocks state and video | High | Blocks all real tests | One designated laptop; open UDP 8890 and 11111 tonight; record every session; develop on replays |
| Stick scale applied twice (controller units vs `TelloDrone` scale) | High if unpatched | Full-stick commands | Final units everywhere (Section 6); check at R2 props-off |
| Video latency larger than trained for (up to 1 s reported) | Medium | Oscillation in follow | Measure at R0; latency-compensating filter in all arms; sim trains up to 700 ms; if higher, fine-tune with wider range and lower max speed |
| Brain sim too slow for real time with YOLO on the same CPU | Medium | Fly cannot fly | Bench at R1; cap `max_neurons`; separate process; YOLO at 10 Hz; PID fallback |
| DN readout too noisy (single cells, 20 Hz rate steps) | Medium to high | Jerky or laggy yaw, fly loses to NOBRAIN | SNR in audit A; bounded low-pass; reported as a finding |
| `rootSide` bug or regex misses give empty groups | High if unpatched | Audit and training meaningless | Patch first; audit prints group sizes; training refuses to start on any empty group |
| DN types other than DNa02 missing from MaleCNS; P1 not identifiable | Medium | Weaker readout or no arousal | Audit lists what exists; readout uses what exists; `arousal_gain` fallback |
| Training does not improve on hand calibration | Medium | Fly arm weak | FLY-HAND is still a result; check per-term reward logs; PID flies |
| Sim-to-real gap (stick gains, lag, detector noise) | High | Real follow worse than sim | Wide randomization; R0 measurements; morning fine-tune; staged R1 to R4 |
| Fly adds nothing over shuffled or no-brain | High (our prediction) | "Buzzword" critique | Prediction stated in advance; show the chart; claim only what the controls support |
| Small object not detected at scan range | Medium | FIND fails | YOLO every frame, 960 or tiles while hovering, 1.0 m search altitude, hops toward furniture, low floor scan, open-vocabulary model for non-COCO classes |
| Dead reckoning drifts during FIND | High | Drone turns to the wrong place for the user | Visual sweep and full scan are the primary method; user told to stay still |
| Floor blind zone near the object during GUIDE | High | Obstacles near the object unseen | Flagged in `scene`; glasses ToF and REACH cover the last meter; overwatch planner only as a stretch |
| No guidance owner | Medium | No GUIDE in the demo | Confirm at H+0; the teammate's minimal rules in H+18 to H+21 |
| Several people in view | Medium | Follows the wrong person | Track ID lock from perception; governor holds if the locked track is lost |
| Battery (8 to 10 min with video, unverified) | High | Few test flights | Flight-minute budget (Section 8); test on replays; props-off tests use no flight time |
| Drone overheats while streaming on the ground | Medium | Shutdown during setup | Stream only when needed on the ground |
| Modal run crashes overnight | Low | No results in the morning | Automatic retries; per-generation checkpoints and resume; failure notification and H+9 alarm; runs independent |

---

## 10. Open questions

1. **Tello model.** Standard, EDU or TT? This affects station mode and SDK 3.0 video commands (`setresolution`, `setfps` probably fail on a standard Tello, unverified).
2. **Indoor flight permission** and the demo space size. The default pose needs little room. The overwatch stretch needs the drone at least max(2.6 m, 1.2 x s) from the user-object midpoint for a user-object separation s (Section 5.4), about a 6 to 7 m clear room for s = 3 m.
3. **Head box source.** Does the perception owner provide a head detector, pose keypoints, or the top of the person box? The encoder only needs a consistent box.
4. **Who writes the guidance logic** (cue rules and A*), and where does the user's heading come from (glasses IMU, AirPods head tracking, or track motion)? Confirm at H+0.
5. **Runtime machine.** Is the bench laptop (Core Ultra 7 258V) the Windows runtime laptop, or is it the Mac? It changes YOLO export (OpenVINO or CoreML) and the Wi-Fi fixes.
6. **Measured values still missing:** true HFOV and VFOV, video latency, stick-to-velocity and yaw gains and lags, box noise and dropout on real footage, yaw drift over 2 minutes, pursuit subgraph tick time with YOLO running, `vgx`/`vgy` frame and quantization (unverified), and the `vgz` unit (then fix FlyDrones' telemetry conversion to `/ 10`).
7. **Number of batteries** and the charging rotation.
8. **Voice intent pipeline** owner, and whether it runs offline (the laptop on the Tello's Wi-Fi has no internet).
9. **Modal limits:** concurrent container cap (6 fly runs need about 1,500 containers at once) and maximum function timeout on our workspace.
10. **Glasses links.** The Tello AP does not forward traffic between clients, so the ESP32-CAM cannot stream to the laptop through it, and the servo link is not specified. Options: BLE for servo commands (too slow for video), a second Wi-Fi adapter on the laptop joined to a separate network the ESP32s use, or a phone hotspot. Hardware owner to confirm at H+0.

### 10.1 Conflicts between sources and how we resolved them

| Topic | Conflict | Resolution |
|---|---|---|
| Camera FOV | Team 86.7 degrees; spec 82.6 degrees (diagonal per forum); calibration implies HFOV 55 degrees | Plan with 55/43, randomize fx widely, calibrate at R0 (Section 3.4) |
| Is RL justified for FOLLOW? | Research: no evidence RL beats tuned PID on a Tello, use PID. Team: train the fly controller. | We train the fly because the project is about the fly, and say plainly that PID is expected to be at least as good; PID is both a baseline and the demo fallback |
| Search approach | PLAN.md: RL viewpoint policy. Research and one teammate: scripted scan. Others: coverage may miss small objects. | Algorithm, with YOLO on every frame, range-aware hops and a low floor scan (Section 5.1) |
| Core subgraph | FLYGUIDE_SPEC assumes `--core-hops` works | It fails on v1.0 data until the `somaSide` patch; old bench sizes were for other groups, so we rebuild |
| FlyDrones version | `CHANGELOG.md` says 0.1.2; `__init__.py` and `pyproject.toml` say 0.1.0 | Identify it by commit `3e26934` |
| FlyDrones escape reflex and looming | FLYGUIDE_SPEC keeps looming avoidance active | DNp01 saturated in calibration and we render no pixels in training, so both are off; the governor does proximity |
| Mapping and state estimation | FLYGUIDE_SPEC: EKF, VO, voxel map, Depth Anything on an M5 | Out of scope on the weak laptop; person-space geometry instead |
| Tello keepalive | djitellopy has `send_keepalive()`, SDK PDFs do not list `keepalive` | Stream rc at 20 Hz instead |
| Velocity units | SDK 3.0 says dm/s, SDK 1.3 gives none; FlyDrones telemetry assumes cm/s | Treat as dm/s, confirm at R0, then fix the FlyDrones conversion |
| Tello stick scale | FlyDrones `TelloDrone.send` takes [-1, 1] and applies a 60 % scale | Our controllers output final -100..100 integers; the Tello I/O bypasses or neutralizes the scale |
| Altitude for follow | FLYGUIDE_SPEC: 1.4 m and 0.5 m to the side; team: behind and above, head at image center | Altitude from the image-row loop (head slightly below center), side offset as a bearing offset (Section 4.4) |

---

## 11. References

**Code and data**
- FlyDrones: https://github.com/SpikeCalls/FlyDrones (commit `3e26934`)
- DJITelloPy: https://github.com/damiafuentes/DJITelloPy (PyPI: https://pypi.org/project/djitellopy/)
- pycma: https://github.com/CMA-ES/pycma
- Modal docs: https://modal.com/docs ; pricing: https://modal.com/pricing
- MaleCNS v1.0: https://male-cns.janelia.org (publication details not re-checked, unverified)
- Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11 ; YOLOE: https://docs.ultralytics.com/models/yoloe

**Tello documentation and community measurements**
- Tello SDK 1.3: https://dl-cdn.ryzerobotics.com/downloads/tello/20180910/Tello%20SDK%20Documentation%20EN_1.3.pdf
- Tello SDK 2.0: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
- Tello SDK 3.0: https://dl.djicdn.com/downloads/RoboMaster+TT/Tello_SDK_3.0_User_Guide_en.pdf
- Tello User Manual v1.4: https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20User%20Manual%20v1.4.pdf
- FOV is diagonal: https://tellopilots.com/threads/what-the-fov-field-of-view-of-the-tello-camera-refers-to-vertical-horizontal-or-diagonal.5883/
- Camera intrinsics: https://tellopilots.com/threads/camera-intrinsic-parameter.2620/
- State stream rate: https://tellopilots.com/threads/increasing-state-sampling-frequency.5624/
- Video latency: https://github.com/damiafuentes/DJITelloPy/issues/87 ; Windows firewall: https://github.com/damiafuentes/DJITelloPy/issues/90 ; macOS port: https://github.com/damiafuentes/DJITelloPy/issues/95

**Followers and control**
- Notes from a widely used Tello face-tracking course: https://gr33nonline.wordpress.com/2021/07/14/course-notes-drone-face-tracking/
- Tello-Face-Tracker: https://github.com/BobMcDear/Tello-Face-Tracker
- Person-Tracking-Tello-Drone: https://github.com/Matthewjsiv/Person-Tracking-Tello-Drone
- tello_object_tracking: https://github.com/fvilmos/tello_object_tracking
- dji-tello-target-tracking: https://github.com/dronefreak/dji-tello-target-tracking
- tello-rl-yolo: https://github.com/rkassana/tello-rl-yolo
- AirCapRL: https://arxiv.org/abs/2007.06343 ; D-VAT: https://arxiv.org/abs/2308.16874
- N. Hansen, "The CMA Evolution Strategy: A Tutorial," https://arxiv.org/abs/1604.00772
- T. Salimans et al., "Evolution Strategies as a Scalable Alternative to Reinforcement Learning," https://arxiv.org/abs/1703.03864

**Search and perception**
- SemExp (Chaplot et al. 2020): https://arxiv.org/abs/2007.00643 ; VLFM: https://arxiv.org/abs/2312.03275 ; UAV-ON: https://arxiv.org/abs/2508.00288
- YOLOv10 (AP_small figures): https://arxiv.org/abs/2405.14458 ; YOLO-World: https://arxiv.org/abs/2401.17270
- MonoLoco (range from human height): https://arxiv.org/abs/1906.06059
- Yamauchi 1997, frontier-based exploration (IEEE CIRA)

**Fly neuroscience and connectome controls**
- P. K. Shiu et al., "A Drosophila computational brain model reveals sensorimotor processing," Nature 634, 2024.
- N. Dhiman, "Topological Sensitivity in Connectome-Constrained Neural Networks," https://arxiv.org/abs/2604.04033 (claims as cited in 4.7 unverified)
- LC10 in courtship pursuit: Ribeiro et al., Cell 2018; P1 arousal gating of LC10a: Hindmarsh Sten et al., Nature 2021 (background; citations not re-checked in this session, unverified).
- Research notes for this repo: [research/fruitfly-brain-rl.md](../research/fruitfly-brain-rl.md)
