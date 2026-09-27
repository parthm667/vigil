# ReachGlass: scout-drone perception and mission stack

A DJI Tello (standard model) hovers **behind and above the wearer's head**. When it gets a text request
("can you find my water bottle?") it drops to search height, explores the room, finds the object, flies
up to it, and works out where the object is **relative to where the person stood and which way they faced**.
That result is the input to the next stage: guiding the person with the glasses.

The target is the team's **blue water bottle** (24 cm with its cap, 9 cm wide), reported as `bottle`. It is
found by **YOLO-World** told "blue water bottle" / "hydro flask water bottle" (960 px, every 3rd frame while
searching), plus a blue check on each box so another bottle is not taken for it. On the team's photos at
Tello-like distances it finds the bottle in 95-100 % of views at 1.3-4 m, with no false detections; 13 ms/frame
on a Mac GPU, ~100 ms on a CPU. `--target color` switches back to the colour-blob detector.

## 1. Setup (once per laptop)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tools/download_models.py        # YOLO (+pose, +World), Depth-Anything; test-runs each one
python -m pytest                        # ~200 tests, ~4 min (unit tests + closed-loop simulations)
```

## 2. See the whole mission on the simulator (no drone)

```bash
python -m reachglass sim --query "can you find my water bottle" --at 25
```

The dashboard shows the drone camera with detections, a live map and the mission state. You can also type
requests in the terminal: `find my water bottle`, `follow me`, `what's around me`, `stop`, `land`.
Add `--headless --record run.mp4` to save a video instead of opening a window.

The simulator renders the room, and the **real** colour-blob detector finds the dummy in those frames. People
and furniture come from "oracle" detectors (YOLO cannot recognise rendered boxes), with realistic faults:
noise, dropouts, and left/right keypoint swaps.

## 3. On site, before the first flight (about 20 minutes)

1. **Latency / sign check.** Run `python tools/tello_latency_test.py --ground`, then the flight version.
   - Set `drone.video_lag_s` to the measured video delay plus a margin. After a turn or move, frames are only
     used once they are guaranteed to show the new view.
   - Note whether telemetry `yaw` grows with `cw`. The stack also checks this on its first clean 20-135 deg
     rotation and flips it if needed.
   - Note the sign of `pitch` when the drone speeds up forward (nose down). If forward acceleration gives
     **negative** pitch, set `perception.pitch_sign: 1`.
2. **Focal length** (every distance depends on it). Place the dummy at a taped 2.0 m and run:
   `python -m reachglass.tools.calibrate_camera tello --height 0.19 --distance 2.0 --auto` (0.19 m = the
   bottle's blue body, which is what `--auto` measures)
   Repeat at 1.5 m and 3 m, then paste the median `fx/fy` into your config.
3. **Dummy colour.** Run `python -m reachglass.tools.hsv_picker tello` and click the dummy **in every lighting
   you will fly in** (room lights on, daylight by the window, its shadow side): each click widens the range to
   cover all of them (`c` clears). Check that **nothing else** in the room lights up in the mask, then press `p`
   and paste the printed `hsv_ranges` into `site.yaml`. Room lights barely move a saturated colour's hue;
   **dim** light lowers its brightness (V, already down to 70 for the bottle's shadow side). Large blue
   surfaces (a blue couch, wall, poster, bus through a window) will show up too: move them or check that
   the mask ignores them. Blue jeans/shirts are dropped when they are inside a detected person's box.
4. **Wearer height, to +-2 cm (with shoes).** Set `perception.person_height_m` and `person_height_sd_m: 0.02`.
   1 m behind at 2 m, only the head is in frame and the distance comes from how far below the camera it is
   (0.25 m): 5 cm of height error is up to 25 % of distance, 2 cm stays under 10 %. `follow.altitude_m` must be
   at least that height + 0.2 m (the config refuses anything else), so above 1.80 m raise it (e.g. 2.05).
5. **Dry run.** `python -m reachglass tello --dry-run` (or `drone: {kind: dry_run}` in the config) connects,
   streams video and reads telemetry, but **sends no motion commands**. Type `takeoff`, then hold the drone
   1 m behind the wearer at 2 m high and check the dashboard: is the person detected from just their head,
   is the range right, and what commands it *would* send (they appear in the log). `--no-takeoff` (start straight in FOLLOW) is only
   accepted together with a dry run.
6. **Fly.** `python -m reachglass tello`. **Take off from the floor**: altitude limits are measured from
   the takeoff surface. Nothing happens until you type `takeoff` or press `t` in the dashboard, so check
   that the person is detected first. Use a clear room, prop guards, and a spotter.
   Keys: `t` = takeoff, SPACE = hold/resume, `l` = land, `e` = EMERGENCY motor stop, `q` = land and quit,
   Ctrl+C = land. A land whose reply is lost is re-sent every 2 s until the drone is down.

Example `site.yaml` (use it with `--config site.yaml`):
```yaml
camera: {fx: 918, fy: 918}
drone: {video_lag_s: 0.25}
perception:
  person_height_m: 1.78
  person_height_sd_m: 0.02
  pitch_sign: 1
  object_heights_m: {bottle: 0.19}      # blue part of the dummy bottle (0.24 with a YOLO model)
  object_widths_m: {bottle: 0.09}
  target_detector:
    params: {hsv_ranges: [[100, 100, 70, 122, 255, 255]]}
follow: {altitude_m: 2.0, distance_m: 1.0}
safety: {max_altitude_m: 2.3}           # below your ceiling
```

## 4. The voice app (AirPods Pro 2 on Windows 11)

```
.venv\Scripts\pip install -r requirements-voice.txt    # once, separate from the flight deps
python -m voice --selftest        # FIRST, with the AirPods in: devices, stem press, record, STT, TTS
python -m voice                   # the real thing, next to `python -m reachglass ...`
```
Press an AirPod **stem once** = push-to-talk (chirp -> speak -> blip), **twice** = repeat/refresh the
guidance, **three times** = "stop". In the voice terminal, Enter / `r` / `s` do the same (the fallback if
the media-session capture misbehaves). Announcements come back as TTS in the AirPods: the flight app
publishes everything it says to UDP `--announce-udp` (default 5006) and the voice app speaks it.
How it works: stem presses arrive as Bluetooth AVRCP media commands captured via a pinned Windows media
session (`voice/stem.py`); the mic capture stream is opened ONLY while recording, because an open mic
locks the AirPods into the low-quality HFP profile (`voice/audio.py`); STT is local faster-whisper;
TTS is edge-tts with offline SAPI fallback. Wiring test without any audio hardware:
`python -m voice --text "find my water bottle"` against a running sim.

Every line typed in the app terminal is still a request, and anything else can send UTF-8 UDP datagrams to
port 5005:
```python
import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"find my water bottle", ("127.0.0.1", 5005))
```
Requests are parsed to a **constrained** intent and target. The target is always a class the detectors can
actually find, so an unknown object gets "I can't look for keys yet" instead of an invented one.
`query.LLMQueryParser` plugs in any language model (Grok, etc.) behind the same interface and rejects
answers outside the vocabulary.

## 4b. Finding people by name ("find arthur")

One photo per teammate in `people/` (`arthur.jpg` -> the drone knows "arthur"), then:
```
pip install onnxruntime insightface   # community cp312 wheel if the sdist build fails
python tools/enroll_faces.py          # embeddings + pairwise-confusability check (no training)
python tools/face_id_webcam.py        # go/no-go: names overlaid on the webcam, walk back to 3-4 m
```
"Find arthur" then runs the same explore/search as the bottle, hopping TOWARD unidentified people
(stopping `person_clearance_m` short) until a face match confirms them; the drone **never approaches a
person** -- on identification it announces where they are ("Arthur is about 4 meters away, ahead to your
left") and hovers. Identity is verified on person-box crops (insightface SCRFD+ArcFace; `kind: opencv`
in `site.yaml` is the zero-install fallback) and sticks to the track between checks. Faces are readable
to roughly 3.5-4 m through the Tello camera -- have people face the drone in the demo.

## 5. Switching the bottle detector

- Check it live first: `python -m reachglass.tools.target_view tello --config site.yaml` (boxes, confidence,
  distance, ms/frame).
- One run with the colour dummy: `python -m reachglass tello --config site.yaml --target color`.
- A trained model later (naming `kind` replaces all the detector's params):
```yaml
perception:
  target_detector:
    kind: yolo
    params: {weights: models/bottle.pt, classes: [bottle], conf: 0.4, rename: {water_bottle: bottle}}
  object_heights_m: {bottle: 0.24}      # a YOLO box covers the whole bottle, cap included
  object_widths_m: {bottle: 0.09}
```

## 6. How it works (and what to swap)

| Module | What it does | Swap / tune |
|---|---|---|
| `sources/` | Newest-frame video: the team's low-latency Tello reader (auto-reconnect on Wi-Fi drops), webcam, files | `TelloVideoSource(fps=60)` |
| `detect/` | `ColorBlobDetector` (dummy), `UltralyticsDetector` (YOLO / pose / trained) | `perception.*_detector` |
| `track/` | IoU tracker + target lock: confirms (3 hits or 1 plausible confident frame), never jumps to a twin, re-locks | `tracking.*` |
| `person/` | Person range (fused: full height / head elevation / shoulders / width) and facing (0 = back to us) | priors in `perception.*` |
| `perception.py` | Runs the right detectors per mode (follow / search / approach); person guard (blue jeans are not the bottle); object range from size priors | `perception.stride` |
| `mapping/` | Odometry (yaw + commanded moves), coverage/obstacle grid with obstacle heights, semantic memory | `explore.*` |
| `behaviors/` | `FollowBehind` (rc visual servo + orbit), `Scan`, `Explore` (scan-score-hop), `Approach`, `ReacquirePerson` | `follow.*`, `explore.*`, `approach.*` |
| `mission/` | State machine, query inboxes, **guidance** (target vs the person: distance, turn, clock face) | |
| `drone/` | `TelloDrone` (non-blocking commands, dry-run), `SafetyGovernor` (clamps, ceiling/floor, battery, stale video -> hover) | `safety.*` |
| `sim/` | Renderer, Tello-like kinematics, walking person, oracle detectors, `SimRunner` | |

Exploration: at each vantage point the drone scans 8 x 45 deg. It then scores directions:
open x (0.3 + novelty) x semantic prior (bottles live on tables...) x people penalty x revisit penalty.
Next it hops at most 1.5 m. Free space comes only from evidence: an object seen at 3 m proves the line
of sight to it is clear. Unknown directions get a cautious 0.5 m step.

## 7. Known limits (be honest in the demo)

- **Walls are invisible** unless something is seen beyond them. There is no forward range sensor, and
  monocular depth (Depth-Anything) proved unreliable here: it is kept as an experimental slot, off by default.
  Hops are short and the room should be clear. Keep a spotter.
- **Following 1 m behind at 2 m** (the default) sees only the wearer's **head**: the camera cannot tilt, and
  their shoulders are 29 deg below its view. So there is no facing estimate and no orbiting behind them when
  they turn, YOLO must recognise a person from the top of a head, and they drop out of view entirely when
  closer than ~0.85 m. If they walk toward the drone it backs off, and if they vanish while close it backs
  off sideways and climbs. `follow: {distance_m: 1.6}` puts the shoulders back in view (facing, orbit, more
  margin) if the dry run shows head-only detection is unreliable.
- **Person range** depends on the configured wearer height (see step 4). The follow controller only moves
  toward the person when even the most conservative estimate agrees.
- **Facing** comes from YOLO-pose's left/right shoulder labels, cross-checked with face visibility and
  smoothed. Test it on the real wearer seen from behind (dry run) before relying on the orbit.
- Target distance comes from its size: far away with its bottom hidden, it reads long. The approach
  re-measures at every step, so it still arrives.
- The Tello's downward sensor makes it hold height above whatever is below it. The stack uses
  floor-referenced height for decisions, and treats furniture reaching within 0.2 m of flight height as an
  obstacle (it passes over chair backs at 1.2 m, not over a TV on a stand). When furniture blocks the way it sidesteps, or stops
  where the furniture allows.
- People: the approach never flies within 1 m of a person. That includes where the wearer stood when they
  asked, because the drone starts behind them and they are often between the drone and the target.

## 8. Next stage: guiding the person

`Mission.guidance` (see `mission/guidance.py`) holds the target and the person's position and heading in
the mission frame. `guidance.relative_to(x, y, heading)` gives the distance and turn from the person's
current pose as they walk. The drone can keep estimating that pose with the person estimator (it hovers
next to the target facing the room). The glasses' L/R cues then come from that turn angle, plus the
glasses camera once the target is in view.

**The glasses hardware exists and is in [`firmware/`](firmware/).** Two boards: an ESP32-CAM hosting a
WiFi access point and serving MJPEG, and an Arduino Nano ESP32 running two VL53L0X range sensors and
two servos that press haptic pads into the wearer's temples.

- **Writing perception or guidance code?** Read [`firmware/HARDWARE_INTERFACE.md`](firmware/HARDWARE_INTERFACE.md).
  It is the black-box contract: you get a `FrameSource` plus two distances in mm, and you send
  `-1` / `0` / `+1` for left / nothing / right. Nothing else to learn.
- **Wiring or flashing the boards?** Read [`firmware/README.md`](firmware/README.md).

One thing worth knowing before you write against it: **the pad sides are crossed.** `+1` (go right)
presses the LEFT pad, so the wearer is nudged from the far side toward where they should go.

## 9. Optional: fruit fly steering

The fruitfly-training team's connectome controller (`flyfollow.steer.FlySteer`, 1,446 LIF neurons from the
fruit fly's pursuit circuit) can decide the **yaw stick** in FOLLOW and APPROACH. Everything else stays ours:
perception, distance and altitude hold, orbit, lost-person search, the approach's discrete moves, and the
`SafetyGovernor`, which still clamps every command. The default is unchanged (`steering: pid`).

```bash
pip install -e /path/to/fruitfly-training/third_party/FlyDrones -e /path/to/fruitfly-training   # into this venv
```
```yaml
follow: {steering: fly}          # pid = our yaw law (the default)
approach: {steering: fly}        # fly: continuous rc yaw onto the target instead of a discrete rotate
fly:
  params_path: /path/to/fruitfly-training/data/brains/trained/FLY-YAW_smooth_best.json   # "" = hand calibration
  # deadband: 4, hysteresis: 3, slew: 300, smoothing_ms: 0   # stick smoothing (defaults shown)
  # latency_s: 0.25                  # default: drone.video_lag_s
  # viz: true                        # fly body + brain window: python -m flyfollow.viz.live --brain <npz>
```
If `flyfollow` is missing or the fly fails, the behaviour logs one warning and uses our yaw law.
Details and sim numbers: `docs/integration/REACHGLASS.md` in fruitfly-training.
