# Drone runtime: bus, recording, operator console, launcher

How to run the drone-side system (plan `docs/DRONE_RL_PLAN.md` Sections 3, 6, 8), drive it from the operator
console, record every session and replay it. Message contract: `flyfollow/runtime/messages.py`.

| File | What |
|---|---|
| `flyfollow/runtime/bus.py` | ZeroMQ broker, `Publisher`, `Subscriber`, `InProcBus` (tests), `FrameRing` (shared-memory video) |
| `flyfollow/runtime/recorder.py` | every bus message to `recordings/<session>/bus.jsonl`, optional `video.mp4` + `frames.jsonl` |
| `flyfollow/runtime/replay.py` | a recording back onto the bus (real time or N x), frames into the FrameRing |
| `flyfollow/runtime/operator.py` | operator console (curses), kill keys, modes, settings, status; `--script` mode |
| `flyfollow/runtime/launch.py` | one command for the whole system, clean land-first shutdown, health warnings |
| `flyfollow/runtime/mission_lite.py` | the harness's mode owner: FOLLOW / HOLD / GUIDE / LAND / IDLE, takeoff, land triggers |
| `flyfollow/runtime/node.py` | `stop_event()`, `Rate`, `log()`, `Liveness` helpers for process loops |

## Run

```bash
.venv/bin/python -m flyfollow.runtime.launch --sim                       # sim_world instead of the Tello, synthetic det
.venv/bin/python -m flyfollow.runtime.launch --sim --takeoff-follow      # sim: take off by itself, then FOLLOW
.venv/bin/python -m flyfollow.runtime.launch --dry                       # real Tello video + state + YOLO, never sends commands
.venv/bin/python -m flyfollow.runtime.launch --send                      # REAL FLIGHT: type "fly" to confirm
.venv/bin/python -m flyfollow.runtime.launch --replay recordings/<s>     # recording instead of the drone
```

Options:

| Option | Meaning |
|---|---|
| `--detector yolo` | the built-in person detector `flyfollow.runtime.detector` (ported ReachGlass YOLO + tracker; default with `--dry` / `--send`). Needs `pip install -e ".[perception]"` and the weights in `models/` (`python -m flyfollow.runtime.detector --selftest` once, online). See `docs/STANDALONE.md`. |
| `--detector external` | another perception process publishes `det`. We only wait for it and warn when `det` goes stale. |
| `--detector synthetic` | sim ground truth (`--sim`) or the recorded `det` (`--replay`); the default for those modes |
| `--viz` | start `flyfollow.viz.live` and pass `--viz` to the controller (fly brain + body viewer) |
| `--script FILE` | operator script mode (non-interactive; the launcher stops when the script ends) |
| `--no-operator` | no console (headless or tests); Ctrl+C stops |
| `--session NAME` | recording name (default `YYYYmmdd_HHMMSS_<mode>`) |
| `--duration S` | stop (land first) after S seconds |
| `--no-video` | do not record video in `--dry` / `--send` (bus is always recorded) |
| `--replay-speed N` | replay speed with `--replay` |
| `--port P` | bus on P / P+1 instead of 5550 / 5551, and frame ring `ff_frames_P` (for a second, parallel session) |
| `--args MODULE="..."` | extra arguments for one child, e.g. `--args tello_io="--no-video"`, `--args controller_runner="--controller pid"` |
| `--takeoff-follow` | `--sim` only: mission_lite takes off at start, auto-locks the user track and enters FOLLOW |
| `--scenario follow\|find\|full`, `--seed N` | sim_world scenario and seed |
| `--no-controller`, `--no-mission` | leave the controller or mission_lite out (`--no-guidance`: skip an optional guidance module) |
| `--plain` | line-based operator UI instead of curses (Windows without curses, or a dumb terminal) |
| `--quiet` | do not echo child output (it is always in `recordings/<session>/logs/<name>.log`) |

Processes (each `python -m <module>`, in its own process group): broker (a thread in the launcher),
`recorder`, the drone backend (`tello_io [--send]` | `sim_world` | `replay`), `controller_runner`, `mission_lite`,
optional `flyfollow.viz.live`, and the operator in the foreground. This runtime is the fly-controller test harness:
the real mission (FIND, APPROACH, RETURN, ...) and flying the demo belong to the ReachGlass stack; `mission_lite`
only owns the harness modes, and an optional `flyfollow.runtime.guidance` is started if it exists. A missing controller is reported as an ERROR line and the rest starts anyway; a
missing backend stops the launch. `det` comes from the built-in `detector` process (yolo), another perception
process (external) or sim / replay (synthetic). With `--sim --detector external|yolo`, sim_world renders frames
(`--det none --frames`) for the detector (YOLO finds few people in those box drawings).

**External detector hookup** (printed at startup with `--detector external`):

```python
from flyfollow.runtime.bus import FrameRing, Publisher, Subscriber
ring = FrameRing.attach(timeout_s=10)               # frames written by tello_io (name flyfollow_frames)
frames = Subscriber(["frame"], conflate=True)       # newest frame message only
pub = Publisher("detector")                         # det goes to tcp://127.0.0.1:5550
m = frames.recv(0.1)
img = ring.read(m["slot"], m["frame_id"])           # RGB uint8 960x720 copy, or None if already overwritten
pub.publish({"topic": "det", "frame_id": m["frame_id"], "t_decoded": m["t_decoded"], "src": "yolo",
             "img_w": 960, "img_h": 720, "dets": [...]})
```

**Shutdown.** Ctrl+C (or `q q` / Ctrl+C in the console, `--duration`, the end of a script, or any child crash
in `--send` except the viewer): the launcher publishes `kill` land and waits for touchdown before stopping
anything. Confirmed means a `tello_state` after the kill with `flying` False (then the backend is kept alive up to
1 s more so its land `tello_ack`, sent after touchdown, is still published and recorded), or a land ack with no
later state saying `flying` True. An ack such as "already landing" while the drone still reports flying is not
confirmation. The backend's immediate `tello_event` kill means the kill arrived; if neither that, an ack nor
`flying` False arrives within 1.5 s, the kill is re-sent once. The wait ends at 5 s, extended to 8 s while the kill
is confirmed and the drone is still descending. `launch.log` says which path happened: `land: CONFIRMED by ack`,
`land: CONFIRMED by tello_state flying=False`, `kill land RE-SENT`, `still descending`, or `land: NOT CONFIRMED`. Then children stop in reverse order with SIGINT, SIGTERM after 3 s, SIGKILL after 1 more
second; the recorder stops last so it captures the landing. If `tello_io` is already dead in `--send`, the launcher sends `command`,
`land` straight to 192.168.10.1:8889 over UDP. A second Ctrl+C kills everything at once.

**Health.** The launcher watches its children and the liveness of `tello_state` (1 s), `det` (1.5 s) and `rc`
(1 s, only in modes with an rc owner). A stale or crashed item is printed and published on the `health` topic
(`{"key", "ok", "level", "text"}`), which the console shows under WARN.

## mission_lite (harness mode owner)

- `mode_cmd` (console keys, typed commands through R5's `intent.parse`) becomes `mode` for FOLLOW, HOLD, GUIDE,
  LAND and IDLE. FIND, APPROACH, RETURN, FACE_PERSON and OVERWATCH answer with a `say`: not in this harness, use
  ReachGlass. "stop" in GUIDE goes to FOLLOW (the user reached the object); "stay" goes to HOLD.
- FOLLOW / HOLD / GUIDE need the drone flying: take off with `t t` (a successful takeoff goes to HOLD) or
  `--takeoff-follow` in the sim (takeoff, auto-lock of the user track, FOLLOW). A refused takeoff (`ok` false) or
  no ack within 10 s stays IDLE and says why.
- Publishes `target` kind person (with the locked track) in FOLLOW, GUIDE and HOLD, and re-publishes `mode` and
  `target` every second for late joiners. In HOLD it owns rc: 0 0 0 0 with src `mission` at 20 Hz.
- LAND on any `kill` or backend `tello_event` kill (the kill itself lands), `ctrl_status.land_request`, battery
  below 25 % or video age above 5 s while flying (these send `tello_cmd` land); IDLE once `flying` is False.

## Operator console

The console runs the terminal in raw mode: Ctrl+C is a key (land, then quit), never a signal to the system.
Kill keys act at once in every state and depend only on the broker (`tello_io` also lands by itself if the
broker dies: it watches its own `tello_state` come back through the broker).

| Key | Action | Message |
|---|---|---|
| **space** | **EMERGENCY: motors off** (no confirm) | `kill` action emergency |
| **l** | **land** | `kill` action land |
| t t | take off (press twice within 2 s) | `tello_cmd` takeoff |
| f / h / r / g | FOLLOW / HOLD / RETURN / GUIDE | `mode_cmd` source operator |
| k | lock the largest, most central person track of the latest `det` | `lock` |
| u | unlock | `lock` track_id null |
| / | type a command: `find my water bottle`, `follow`, `stop`, `look again`, `land`, `takeoff`, `lock 3`, `set key value` | R5's `intent.parse` (fallback: built-in rules) |
| s | edit a setting: `follow_distance_m 2.5` (keys of `SETTINGS_DEFAULTS`) | `settings` |
| c | toggle controller fly / pid | `settings` controller |
| q q | quit (lands first) | `kill` land |
| Ctrl+E / Ctrl+L | emergency / land, also while typing a command | `kill` |

Panel: mode (and since when), controller, brain tick ms, lock, battery, height, ToF, flying, video age, temp,
sending flag, the rc actually sent (`rc_sent`, with src and DRY RUN / LIVE), the rc requested (`rc`), target
valid / range / bearing / in band (`ctrl_status`), det summary, FIND status, last `say`, last `cue`, recent mode
transitions and reason, warnings (stale topics, battery < 25 %, video age > 1 s, no rc from the mode's owner,
governor interventions, launcher health), and an event log (acks, kills, commands).

Script mode (`--script FILE`, used by the end-to-end sim test): one command per line, `<seconds from start> <command>`:

```
0.5 wait tello_state 8      # block until a message of this topic arrives (timeout counts as a failure, exit 1)
1.0 takeoff
2.0 follow                  # also hold, return, guide, mode FIND bottle
2.5 lock auto               # or lock 3, unlock
3.0 cmd find my water bottle
4.0 set follow_distance_m 2.5
5.0 controller pid
8.0 land                    # or emergency
```

## Recording and replay

Every launch records to `recordings/<session>/` (gitignored): `bus.jsonl` (every message plus `t_rec`, the
recorder's receive time), `meta.json`, `logs/`, and in `--dry` / `--send` also `video.mp4` (H.264 via
imageio-ffmpeg, which shells out to ffmpeg, so no PyAV/OpenCV clash) with `frames.jsonl`
(`n`, `frame_id`, `t_decoded`, `t_rec`). The mp4 fps is nominal; `frames.jsonl` is the timing truth.

```bash
.venv/bin/python -m flyfollow.runtime.recorder --session test1 --video           # standalone (needs a broker)
.venv/bin/python -m flyfollow.runtime.replay recordings/<s> --speed 2 --frames    # everything, 2x, with video
.venv/bin/python -m flyfollow.runtime.replay recordings/<s> --topics det,tello_state --start 30 --duration 20 --loop
.venv/bin/python -m flyfollow.runtime.bus --echo det,mode --no-broker              # watch a running bus
```

Replay shifts `t` and `t_decoded` so messages look live (detector latency `t - t_decoded` is kept);
`--original-t` keeps the recorded times. Recorded `frame` messages are never replayed; with `--frames` the video is
decoded into the FrameRing and fresh `frame` messages are published. Command topics (`kill`, `tello_cmd`, `rc`,
`mode_cmd`, `settings`, `lock`, `det_cfg`) and `health` are skipped unless named in `--topics`, and every replayed
message carries `"replayed": true` (Tello I/O should refuse commands carrying it). Replay starts its own broker
unless one is running.
`launch --replay` replays only `tello_state` (+ `det` with `--detector synthetic`) and the frames, so the live
controller, mission and guidance react to the recording (plan R1, open loop).

## Bus API (for every process)

```python
from flyfollow.runtime.bus import Publisher, Subscriber, FrameRing, InProcBus, Broker
pub = Publisher("mission")                      # connect to PUB_ADDR; waits <= 0.5 s for the handshake
pub.publish({"topic": "say", "text": "hi", "priority": 1})   # never blocks; adds t and src_node if absent
sub = Subscriber(["det", "tello_state"])        # exact topic names; None = all topics
m = sub.recv(timeout_s=0.05)                    # dict or None
ms = sub.drain()                                # everything pending, non-blocking
latest = Subscriber(["det"], conflate=True)     # only the newest message per topic
bus = InProcBus(); Publisher("x", bus=bus); Subscriber(["y"], bus=bus)   # same API in one process, for tests
```

- Do not pass addresses or the ring name: the defaults read `FLYFOLLOW_PUB_ADDR`, `FLYFOLLOW_SUB_ADDR` and
  `FLYFOLLOW_FRAME_RING`, which the launcher sets with `--port`, and fall back to the constants in `messages.py`.
- Slow joiner: ZeroMQ PUB/SUB drops what is sent before the connection and the subscriptions are in place. The
  constructors wait for the handshake plus 30 ms, which covers loopback. A process that starts later than a
  publisher misses earlier one-shot messages (a `mode` sent before it started): republish state periodically
  or on request if a late joiner needs it (the mission could repeat the current `mode` every second).
- Ordering is per publisher only. Merge streams by `t`.
- ZeroMQ subscriptions are prefixes (`det` matches `det_cfg`); `Subscriber` re-filters by exact topic.
- Numpy scalars and arrays are converted to JSON automatically; NaN is written as `NaN` (Python reads it back).
- FrameRing: the producer calls `FrameRing.create(h=720, w=960)` (a stale same-named segment from a crash is
  replaced), writes with `ring.write(rgb, frame_id, t_decoded) -> slot`, publishes `frame`, and calls
  `close()` + `unlink()` at exit. Consumers `FrameRing.attach(timeout_s=...)`, then `read(slot, frame_id)`
  (a copy, or None if overwritten) or `latest()`. Names are shortened to the macOS limit of 30 characters, and
  attaching does not register with the resource tracker (no "leaked shared_memory" warning, no unlink on exit).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `broker cannot bind tcp://127.0.0.1:5550` | a broker from a crashed run is alive: `lsof -nP -iTCP:5550`, `kill <PID>`; or use `--port 5560` |
| macOS `Errno 48 Address already in use` on UDP 8890 (or 8889, 11111) after a crash | the launcher checks these ports before `--dry` / `--send` and prints the holder; `lsof -nP -iUDP:8890`, then `kill <PID>` |
| Windows: commands work but "Did not receive a state packet", no video | Windows Defender Firewall blocks inbound UDP 8890 and 11111: allow `.venv\Scripts\python.exe` on private and public networks, or add inbound UDP rules for 8890 and 11111 |
| No internet after joining the Tello Wi-Fi | expected, the Tello AP has none. Install, `git pull` and start everything before joining it, or use a second Wi-Fi adapter (USB) for the internet |
| Glasses / ESP32 cannot reach the laptop through the Tello AP | the AP does not forward between clients; use a second adapter or network (plan open question 10) |
| First frames black or 300x400 | PyAV decode needs about 3 s after the stream starts (tello_io's `--video-warmup-s`) |
| `det stale` warning in `--dry` / `--send` | the external detector is not running or not publishing to this bus (check `--port`) |
| Garbled terminal after a crash | `reset` (or `stty sane`) |
| Where are the logs | `recordings/<session>/logs/<name>.log`, `launch.log` for the launcher |

## Tests

```bash
.venv/bin/python -m pytest tests/test_runtime_bus.py tests/test_runtime_launch.py tests/test_runtime_mission_lite.py -q
FLYFOLLOW_E2E=1 .venv/bin/python -m pytest tests/test_runtime_launch.py -q -k e2e   # 40 s sim follow
```

Loopback only on random free ports, no hardware: pub/sub through a real broker, exact topic filter and
per-publisher order, conflate, non-blocking publish without a broker, `InProcBus`, FrameRing overwrite detection
across processes and stale-segment recovery, recorder to replay round trip (bus + video), operator script and key
handling, the curses console in a pseudo-terminal, the four land-confirmation paths at shutdown (ack, state only,
re-send, nothing, slow descent, "already landing"), mission_lite transitions and land triggers
(`tests/test_runtime_mission_lite.py`), `launch --replay` with a script, and `launch --sim` with an operator script.
The 40 s follow check (`launch --sim --no-operator --takeoff-follow --duration 40`: user in view more than 80 % of
FOLLOW, no collision, landed at shutdown) takes about 50 s, so it runs only with `FLYFOLLOW_E2E=1`.
