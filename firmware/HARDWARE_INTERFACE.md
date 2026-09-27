# ReachGlass hardware interface — handoff contract

**Audience:** agents and people working on perception, mission, and guidance. You do not
need to read, flash, or understand the firmware. This document is the black-box boundary:
what the glasses hardware gives you, and how you talk back to it.

**Status:** protocol v1. Sensors, servos and the radio link are verified on hardware; the
end-to-end UDP path is not yet run — see [Trust level](#9-trust-level). For wiring, flashing
and bring-up, see [`README.md`](README.md) in this folder.

---

## 1. What the hardware is

ReachGlass has two pieces of hardware. The drone half you already have. This document
covers the other half: **the glasses**.

The glasses are a wearable with three jobs:

| Job | Device | What perception gets |
|---|---|---|
| POV camera for fine adjustment at the end of a run | ESP32-CAM (OV2640) | an ordinary `FrameSource`, ~12 fps MJPEG |
| Obstacle / wall proximity | 2× VL53L0X time-of-flight | two distances in mm, ~15 Hz |
| Haptic direction cue to the wearer | 2× servo, one per temple | you command press depth 0–100 per side |

The haptic pads are the **output** side of guidance. The left pad presses into the left
side of the face to mean *go left*; the right pad means *go right*. That is the entire
user-facing output of the guidance stage, so its correctness matters more than anything
else in this document.

### Topology

```
┌─ GLASSES (worn) ──────────────────────────────┐
│                                               │
│   VL53L0X(L) ─┐                               │
│   VL53L0X(R) ─┴─ I2C A4/A5 ─┐                 │
│                             ▼                 │
│                      ┌───────────────┐        │
│   haptic pad L ◄─ D7 │ Nano ESP32    │        │
│   haptic pad R ◄─ D6 │ 192.168.4.50  │────┐   │
│                      └───────────────┘    │   │
│                                           │   │
│                      ┌───────────────┐    │   │
│                      │ ESP32-CAM     │    │   │
│                      │ 192.168.4.1   │    │   │
│                      │ OV2640 + AP   │◄───┘   │
│                      └───────┬───────┘        │
│                    NO WIRES BETWEEN THEM      │
└──────────────────────────────┼────────────────┘
                               │ 2.4 GHz, the CAM's AP "rover"
                               ▼
                     ┌─────────────────────┐
                     │ laptop: reachglass  │
                     │ 2nd WiFi adapter    │
                     └─────────────────────┘
```

**Why it is split this way.** The ESP32-CAM hosts the access point and serves video; the
Nano joins that AP as a station and speaks the rover protocol straight to the laptop.
**The two boards are not wired to each other at all.**

There used to be a UART bridge, where the CAM relayed the Nano's bytes. It was removed
because it bought nothing and cost latency: servo commands had to queue behind a task that
was also shipping 25 KB JPEG frames. Now the control path and the video path are fully
independent, and the protocol can change with a Nano-only reflash over USB-C — nobody has
to open the glasses to reach the CAM's FTDI header.

The consequence for you: **video and data come from two different addresses.** The CAM has
no telemetry and the Nano has no web server. Do not pass one's address to the other.

---

## 2. Quick start

Everything you need is in one module: `firmware/host/reachglass_glasses.py` (copy it into
`reachglass/sources/` or wherever fits your layout).

```python
from reachglass_glasses import GlassesVideoSource, GlassesLink

# --- video: the CAM, 192.168.4.1. An ordinary FrameSource, like TelloVideoSource ---
cam = GlassesVideoSource().start()
frame = cam.wait_first(10.0)          # Frame(image, t, seq, source="glasses")

# --- data + haptics: the NANO, 192.168.4.50. A DIFFERENT board ---
link = GlassesLink().start()          # beacon, else the Nano's static address
st = link.state()                     # state() is a METHOD; it returns an immutable snapshot
print(st.tof_left_mm, st.tof_right_mm, st.stale)

link.guide(turn_deg=+35)              # + = person turns RIGHT -> right pad presses
link.pulse("B", count=2, on_ms=120, off_ms=120, depth=80)   # "arrived" buzz
link.release()                        # stop pressing
```

Smoke-test the whole chain with no perception code at all:

```bash
python firmware/host/reachglass_glasses.py
```

That prints live ToF readings and sweeps both pads.

---

## 3. What you receive

### 3.1 Video

An MJPEG stream at `http://<cam-ip>/stream`, wrapped for you as `GlassesVideoSource`, a
`reachglass.sources.base.FrameSource` subclass. It obeys the same rules as every other
source in the repo: `read()` is non-blocking and returns the **newest** frame or `None`;
compare `Frame.seq` to tell new from repeated; frames are BGR uint8 and **read-only**
(copy before annotating).

| Property | Value |
|---|---|
| Resolution | 640×480 (VGA) |
| Rate | ~12 fps target, frame-gated in firmware |
| Codec | MJPEG over HTTP multipart |
| Latency | roughly 100–250 ms, higher when sharing the drone's radio |
| `Frame.source` | `"glasses"` |

**Four things that will bite you:**

1. **The intrinsics in `CameraCfg` are the Tello's, not the glasses'.** `site.yaml`
   currently sets `fx: 1077, fy: 1077`, measured for the Tello's 960×720 stream. Those
   numbers are meaningless for a 640×480 OV2640. If you do *any* geometry on glasses
   frames — bearing, range-from-pixel-height, `Pose2D.point_at` — calibrate this camera
   separately with `tools/calibrate_camera.py` and carry a second `CameraCfg`. Using the
   Tello's `fx` on glasses frames silently produces wrong bearings, not an error.
2. **Do not put this stream in a tight control loop.** At 12 fps with WiFi jitter it is
   for *fine adjustment and confirmation at the end of a run*, which is what it was asked
   for. The Tello stream remains the primary perception input.
3. **The stream reconnects.** `seq` stays monotonic across reconnects (the same offset
   trick `TelloVideoSource` uses), but there will be gaps. Do not assume `seq` increments
   by exactly one.
4. **`/health` is unreachable while `/stream` is open.** The CAM's HTTP task serves one
   connection at a time and `/stream` holds it for the connection's life. Query `/health`
   before you start video or after you stop it; a second request queues rather than
   dropping the stream.

### 3.2 Telemetry

One ASCII line per UDP datagram, **~15 Hz**, from UDP port 4210:

```
T,<seq>,<uptime_ms>,<tof_l_mm>,<tof_r_mm>,<press_l>,<press_r>,<link>
```

| Field | Meaning |
|---|---|
| `seq` | uint32, +1 per packet, wraps. Gaps mean packet loss. |
| `uptime_ms` | `millis()` on the Nano. Resets on reboot. |
| `tof_l_mm` | LEFT sensor, integer millimetres. **`-1` = out of range or sensor dead.** |
| `tof_r_mm` | RIGHT sensor, same. |
| `press_l` | 0–100, press depth the Nano is *actually* applying to the left pad. |
| `press_r` | 0–100, right pad. |
| `link` | `1` if the Nano has heard a command in the last 600 ms, else `0`. |

`GlassesLink` parses this into `link.state()` — a **method** returning an immutable
`GlassesState` snapshot — with those fields plus the properties `stale` (no packet for
>0.5 s) and `tof_min_mm` (nearer of the two, `None` if both are `-1`), and a cumulative
`packets_lost` counted from `seq` gaps.

**The rate is ~15 Hz, not the 20 Hz the timer asks for.** `TELEM_PERIOD_MS` is 50, but each
VL53L0X single-shot read blocks for ~33 ms at the sensor's default timing budget, so two
sequential reads floor the loop period at ~66 ms. Nothing depends on the exact number — the
600 ms failsafe and the 0.5 s staleness threshold both have room — but do not build a
15-vs-20 Hz assumption into a filter.

**`press_l`/`press_r` are the hardware's own report, not an echo of your command.** Use
them to confirm the Nano actually accepted what you sent. If you command `H,80,0` and
telemetry keeps reporting `press_l=0`, your commands are not arriving.

**`-1` is not a distance.** Guard every read. Out-of-range on a VL53L0X means "nothing
within ~2 m", which for obstacle avoidance is *good* news — but it is not the number 1200
or 0, and treating `-1` as a proximity reading will make the system warn about a wall
that is not there.

### 3.3 Which sensor is "left"

`tof_l_mm` is the sensor whose XSHUT pin is on Nano D8, re-addressed to I2C `0x30`.
`tof_r_mm` is XSHUT on D9, address `0x31`.

**That mapping is a firmware fact, not a physical guarantee.** Whether the 0x30 sensor is
actually mounted on the wearer's left depends on how the glasses were assembled. Verify
it once by hand: wave a hand in front of one sensor and watch which field moves. If they
are swapped, swap the two XSHUT wires — do not "fix" it in perception code, because the
next person to read this document will assume the contract holds.

---

## 4. What you send

All commands are ASCII, newline-terminated, one per UDP datagram to port 4210.
`GlassesLink` has a method for each; you should not need to build these by hand.

| Wire format | Method | Meaning |
|---|---|---|
| `D,<dir>[,<depth>]` | `direction(d)` | **Primary interface.** `-1` go left, `0` none, `+1` go right. Sides crossed — see 4.1. |
| `H,<left>,<right>` | `hold(left, right)` | Hold press depth, 0–100 each. Graded pressure, when you want it. |
| `P,<side>,<count>,<on_ms>,<off_ms>,<depth>` | `pulse(...)` | Pulse burst, `side` = `L` / `R` / `B`. |
| `Z` | `release()` | Release both immediately. Safety stop. |
| `C,<side>,<rel_deg>,<press_deg>` | `calibrate(...)` | Retrim one side without reflashing. |
| `S,<l_deg>,<r_deg>` | `raw_angles(l, r)` | Raw servo angles. Bench only. |

### 4.1 The method you actually want

```python
link.direction(d)      # d = -1 go left, 0 no cue, +1 go right
```

That is the whole interface. Your algorithm already emits `-1 / 0 / +1`, so there is nothing
to convert.

> **The sides are crossed on purpose.**
> `d = +1` (go right) presses the **LEFT** pad.
> `d = -1` (go left) presses the **RIGHT** pad.
>
> The pad opposite the turn presses, so the wearer feels a nudge from the far side pushing
> them the way they should go.

**This is the inverse of the convention used before 2026-09-27**, which pressed the pad on the
side you were turning toward. If you find an older note, a stale copy of this file, or a
comment that says "positive means press the right pad", it is wrong — the firmware's `D`
handler is authoritative. Getting this backwards sends a blind person the wrong way.

Only one pad is ever driven. Two pads pressing together is not a direction, it is just
pressure, and the wearer cannot read it as left or right.

Optional second argument overrides the press depth for one command:

```python
link.direction(1, depth=70)
```

`d = 0`, a non-integer, or an out-of-range value all release both pads — no cue beats a wrong
cue. (The firmware keys only on the *sign*, so a raw bearing sent by mistake still cues the
right direction rather than going silent; the host clamps to `-1/0/+1` before sending.)

**Still available if you need it:** `guide(turn_deg)` takes a continuous bearing error from
`Guidance.relative_to()` and reduces it to a direction using a 12° deadband. There is no
proportional ramp — the upstream signal carries no magnitude, so a ramp would be inventing
precision. Use `hold(left, right)` directly if you genuinely want graded pressure.

### 4.2 Why absolute setpoints, resent continuously

`hold()`/`guide()` set an **absolute** depth, and `GlassesLink` resends the current
setpoint at 20 Hz on a background thread. Both facts matter:

- **Absolute, not incremental.** A dropped UDP packet costs one stale frame instead of
  permanently desynchronising pad position.
- **Resent, not fire-and-forget.** The resend stream is what feeds the Nano's failsafe
  watchdog. If you stop calling into `GlassesLink` the thread keeps the last setpoint
  alive; if the *process* dies, the watchdog fires. Do not optimise the resend away.

### 4.3 Pulses are generated on the Nano, on purpose

`pulse()` hands the Nano a count and an on/off period, and the Nano runs the pattern
itself as a non-blocking state machine. Driving a rhythm from Python over a congested
2.4 GHz link produces audibly ragged timing; a firmware-generated burst is clean.

Use pulses for **events** ("arrived", "target lost", "turn now") and `guide()` for
**continuous** steering. A new `hold`/`guide`/`pulse` cancels any burst in flight.

---

## 5. Failsafe and safety

**If no command arrives for 600 ms, the Nano releases both servos to 0%.**

It releases rather than holding the last position, and that choice is deliberate: a servo
latched against a blind person's face after a WiFi dropout is the worst failure this
system can produce. There is also a compile-time maximum press angle per side, so a bad
command value cannot over-drive a pad into the wearer.

Consequences for your code:

- Losing the link **cannot** leave the wearer being squeezed. It leaves them with no
  guidance, which is a degraded-but-safe state you should surface by another channel
  (speech).
- `state().link == 0` means *the Nano is not hearing you*. `state().stale == True` means
  *you are not hearing the Nano*. They are different failures with different causes —
  check both, and report which one.
- Call `link.stop()` (or use the context manager) on shutdown. It releases the pads
  explicitly rather than waiting out the 600 ms.

---

## 6. Network reality

The CAM hosts an access point named `rover` (password `rover1234`). Everything lives on it:

| Board | Address | Serves |
|---|---|---|
| ESP32-CAM | `192.168.4.1` | MJPEG video over HTTP, plus `/health` |
| Nano ESP32 | `192.168.4.50` | ToF telemetry and haptic commands over UDP 4210 |
| laptop | `192.168.4.2` (DHCP) | — |

The Nano's address is **static, compiled into the firmware**, so nothing has to be
discovered for the system to work. It also broadcasts a 1 Hz beacon on UDP 4211
(`ROVER,<ip>,<port>`) until a peer registers, which `resolve_nano()` uses when present.

**The laptop needs two WiFi adapters**, because the Tello also insists on being an access
point and one radio cannot join two. Built-in adapter on one, USB dongle on the other. The
subnets differ (`192.168.4.0/24` for the glasses, `192.168.10.0/24` for the drone), so
Windows routes by interface with no default-route conflict, and djitellopy's hardcoded
`192.168.10.1` keeps working untouched.

This also keeps the drone's radio uncontended, which matters for `main.py`'s latency
measurements: glasses video sharing the Tello's embedded AP would inflate exactly the
numbers that harness exists to measure.

Quickest check that the whole radio side is up: `http://192.168.4.1/health` reports
`"clients":2` once both the laptop and the Nano have associated. `clients:1` means the
Nano did not join — check its serial output for `join failed`.

**Neither adapter has internet while you are testing.** Both APs are offline networks, so
install your pip packages first.

---

## 7. Integrating into the perception stack

Four touch points. None require firmware changes.

**1. Register the source.** `GlassesVideoSource` is a `FrameSource`, so it drops in
beside `TelloVideoSource`. Export it from `reachglass/sources/__init__.py`.

**2. Add a config section.** `reachglass/config.py` uses nested dataclasses and **raises
on unknown keys** — so you must add a `GlassesCfg` dataclass *before* putting a
`glasses:` block in `site.yaml`, or loading fails loudly. Suggested shape:

```python
@dataclass
class GlassesCfg:
    enabled: bool = False
    host: str | None = None        # None = auto-discover
    stream_fps: int = 30           # reader retrieve cap; keep well above the 12 fps stream
    deadband_deg: float = 12.0
    full_deg: float = 60.0
    max_press: int = 100
    camera: CameraCfg = field(default_factory=CameraCfg)   # NOT the Tello's intrinsics
```

**3. Drive the haptics from guidance.** The natural home is wherever the guiding stage
decides which way the person should go as they walk — the hook `guidance.py`'s docstring
already anticipates. Feed your `-1 / 0 / +1` straight to `link.direction()`; feed "arrived"
to `link.pulse()`. If that stage still works in continuous bearings, `link.guide(turn_deg)`
reduces one for you with a 12° deadband.

**4. Use ToF for the obstacle warning.** Both distances, `-1`-guarded, are a proximity
signal for the warn-the-user path. They are **not** a mapping input: two fixed forward
cones on a head that turns constantly will not build a usable occupancy grid, and
`mapping/` already gets its free-space estimate from the drone.

---

## 8. Calibration

The mechanism will need trimming, and you can do it live without reflashing:

```python
link.calibrate("L", released_deg=90, pressed_deg=0)   # 0 deg = hardest press
```

On this rig **0° is the arm fully inward (hardest press) and 90° is clear of the face**, so
`pressed_deg` is numerically *lower* than `released_deg` and the span is negative. The
firmware handles that sign; do not "fix" it by swapping them.

Find the numbers with `raw_angles()` while someone wears the glasses: decrease the pressed
angle toward 0 until the cue is unmistakable but comfortable, then back off. Persist the result by
editing the constants at the top of `firmware/nano_tof/nano_tof.ino` — `C` is deliberately not
saved across reboot, so a bad experiment cannot spoil the fit permanently.

---

## 9. Trust level

Being precise about what has actually been proven, because the rest of this document
reads as though it all works:

| Path | Status |
|---|---|
| Firmware compiles (Nano both cores, CAM) | **verified** |
| CAM AP + MJPEG stream in a browser | **verified on hardware** |
| Servo PWM actually driving the arms, full travel sweep | **verified on hardware** (see the LEDC note below) |
| Both VL53L0X initialising at 0x30/0x31 and returning mm | **verified on hardware** |
| Command parsing, press math, travel clamp, burst timing, failsafe release, T-line format | **verified in a native test harness** (26 assertions against the real `.ino` compiled under MSVC with Arduino stubs) |
| `direction()` sign mapping (-1 -> right pad, +1 -> left pad, garbage -> release) | **verified by direct execution** against a stubbed socket |
| T parsing, max inter-command gap < 600 ms, bursts surviving concurrent steering | **verified against a stand-in Nano** |
| Camera-wedge detection and recovery | implemented, not yet triggered on purpose |
| Nano joining the CAM's AP as a station, static IP `192.168.4.50` | **verified on hardware** — `/health` reports `clients:2` |
| Telemetry and commands over UDP end to end | **not yet run on hardware** |
| Haptic cue actually felt by a wearer | **not yet run on hardware** |
| Two-adapter laptop setup (glasses + drone at once) | **not yet run on hardware** |

Treat anything below the verified rows as designed-and-compiled, not proven. The fastest
way to move a row up is `python firmware/host/rover_client.py`.

**The LEDC trap, recorded because it cost an evening and compiles clean.**
`SOC_LEDC_TIMER_BIT_WIDE_NUM` is **14** on the ESP32-S3, not the 16 you would use on an
original ESP32. `ledcSetup()`/`ledcAttach()` *reject* anything wider: they return 0/false
and configure **no timer at all**, so `ledcWrite()` writes a duty into an unconfigured
channel and the pin never toggles. The servos sit perfectly still while every number in the
firmware — angles, microseconds, telemetry — looks correct. It is a **runtime** limit, so a
16-bit build compiles clean on both cores and silently does nothing. Both sketches now print
their attach result at boot; trust that line, not the build.

Related: `SERVO_FLOOR_US` is 600, not the 500 µs that 0° maps to. 500 may sit against the
servo's internal end stop, where an SG90 stalls, draws ~700 mA and destroys its own gearbox
within minutes — silently, if serial is unplugged. 600 is bench-validated on this rig.

---

## 10. Non-goals

Things the glasses hardware deliberately does **not** do, so you do not wait on them:

- No onboard compute, detection, or filtering. Raw millimetres and raw frames only.
- No IMU, no head orientation. If guidance needs the wearer's heading, it comes from the
  drone's person estimator, not from here.
- No audio. Speech output is the laptop's job.
- No persistent state or logging on either microcontroller.
- No mesh, no multiple glasses. One wearer, one laptop.

---

## 11. Protocol reference card

```
NANO -> PC, ~15 Hz
  T,<seq>,<uptime_ms>,<tof_l_mm>,<tof_r_mm>,<press_l>,<press_r>,<link>

PC -> NANO
  D,<dir>[,<depth>]                             PRIMARY: -1 left, 0 none, +1 right
  H,<left>,<right>                              hold press 0..100
  P,<side>,<count>,<on_ms>,<off_ms>,<depth>     burst; side = L|R|B
  Z                                             release both
  C,<side>,<rel_deg>,<press_deg>                calibrate; side = L|R
  S,<l_deg>,<r_deg>                             raw angles, bench only

TRANSPORT  (two boards, two addresses -- do not mix them up)
  video      http://192.168.4.1/stream     MJPEG, 640x480, ~12 fps     <- the CAM
  health     http://192.168.4.1/health     {"cam":..,"fps":..,"clients":..,"uptime_ms":..,"rssi":..}
                                           "clients":2 = laptop + Nano both joined
                                           NOTE: unreachable while /stream is open
                                           (single-threaded HTTP task). Query before or after.
  data       192.168.4.50 UDP 4210         PC sends first to register   <- the NANO
  discovery  UDP 4211 broadcast            "ROVER,<ip>,<port>" at 1 Hz, from the Nano

PINS (Nano ESP32)   A4/A5 I2C | D9/D8 XSHUT L/R | D7/D6 servo L/R | no inter-board wiring
PINS (ESP32-CAM)    GPIO1/3 reserved for FTDI | nothing else wired

FAILSAFE   600 ms of command silence -> both pads RELEASE to 0%
SIGN       D,+1 = go RIGHT = presses the LEFT pad   (sides CROSSED on purpose)
           D,-1 = go LEFT  = presses the RIGHT pad
SENTINEL   tof_*_mm == -1 means out of range or dead sensor, never a distance
GOTCHA     LEDC resolution must be <= 14 bits on the S3, or the pins never toggle
```