# ReachGlass firmware: the glasses hardware

The drone half of ReachGlass lives in `reachglass/`. This is the other half — the **wearable**.
It is the output stage of README §8 ("Next stage: guiding the person"): the thing that actually
tells the wearer which way to walk.

Three jobs, one wearable:

| Job | Device | What the laptop gets |
|---|---|---|
| Haptic direction cue | 2× SG90 servo, one per temple | you send `-1` / `0` / `+1`, it presses a pad into the face |
| Obstacle / wall proximity | 2× VL53L0X time-of-flight | two distances in mm, ~15 Hz |
| POV camera for the last metre | ESP32-CAM (OV2640) | an ordinary `FrameSource`, ~12 fps |

**If you are integrating perception, read [`HARDWARE_INTERFACE.md`](HARDWARE_INTERFACE.md)
instead of this file.** That is the contract: what you receive, what you send, what the
failure modes are. This file is for whoever flashes and wires the boards.

---

## 1. What talks to what

```
┌─ GLASSES (worn) ──────────────────────────────┐
│   VL53L0X(L) ─┐                               │
│   VL53L0X(R) ─┴─ I2C A4/A5 ─┐                 │
│                             ▼                 │
│                      ┌───────────────┐        │
│   haptic pad L ◄─ D7 │ Nano ESP32    │        │
│   haptic pad R ◄─ D6 │ 192.168.4.50  │────┐   │
│                      └───────────────┘    │   │
│                      ┌───────────────┐    │   │
│                      │ ESP32-CAM     │    │   │
│                      │ 192.168.4.1   │◄───┘   │
│                      │ OV2640 + AP   │        │
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

The ESP32-CAM hosts an access point and serves video. The Nano joins that AP as a station and
speaks the protocol straight to the laptop. **The two boards are not wired to each other.**

An earlier version had the CAM relay the Nano's bytes over UART. It was removed: servo commands
had to queue behind a task that was also shipping 25 KB JPEG frames, and it meant opening the
glasses to reflash the CAM whenever the protocol changed. Now the control path and the video
path are fully independent.

**The consequence you must not forget: video and data are two different addresses.** The CAM has
no telemetry; the Nano has no web server.

| Board | Address | Serves |
|---|---|---|
| ESP32-CAM | `192.168.4.1` | `/stream` (MJPEG), `/health` (JSON), `/` (test page) |
| Nano ESP32 | `192.168.4.50` | UDP 4210, telemetry + commands |

---

## 2. What is in here

| Path | What it is |
|---|---|
| `nano_tof/` | **Production Nano firmware.** ToF sensors, haptic servos, WiFi, the protocol. |
| `cam_bridge/` | **Production ESP32-CAM firmware.** Access point + MJPEG + `/health`. Nothing else. |
| `testitcle/` | Bench sketch. Sweeps the servos and prints ToF, with a serial console. Not flown. |
| `host/reachglass_glasses.py` | The laptop-side module: `GlassesVideoSource` + `GlassesLink`. Drop into `reachglass/sources/`. |
| `host/rover_client.py` | GUI bench viewer. Video window, ToF/press overlay, keyboard cues. |
| `HARDWARE_INTERFACE.md` | **The contract.** Read this to write perception code. |

---

## 3. Wiring

**Nano ESP32**

| Pin | To |
|---|---|
| A4 / A5 | SDA / SCL, **both** VL53L0X |
| D9 | XSHUT, LEFT sensor (becomes I2C `0x30`) |
| D8 | XSHUT, RIGHT sensor (becomes I2C `0x31`) |
| D7 | LEFT haptic servo signal |
| D6 | RIGHT haptic servo signal |

**ESP32-CAM:** GPIO1/GPIO3 are the FTDI header. Nothing else is wired.

**Power the servos from a separate 5 V supply, with its ground tied to the Nano's ground.** Two
SG90s pull well over an amp between them and will brown out the board. Do **not** connect servo
+5 V to the Nano's 5 V pin. Keeping USB-C plugged in at the same time is fine.

---

## 4. Arduino IDE setup

Install the **`Adafruit_VL53L0X`** library (it pulls in Adafruit BusIO). Nothing else.

> **The Nano ESP32 appears TWICE in the Boards menu**, under "Arduino ESP32 Boards"
> (`arduino:esp32`, core 2.0.x) and under "esp32 by Espressif Systems" (`esp32:esp32`, core 3.0.x).
> They have **incompatible LEDC APIs** and identical menu names. Both sketches compile on either
> — they route every PWM call through a `servoAttach`/`servoWrite` shim that picks the right API
> at compile time. If you see `'ledcAttach' was not declared in this scope ... suggested:
> 'ledcAttachPin'`, you are on the 2.x core and the shim is missing or has been edited out.

| Sketch | Board | Extra settings |
|---|---|---|
| `nano_tof`, `testitcle` | Arduino Nano ESP32 (either package) | — |
| `cam_bridge` | AI Thinker ESP32-CAM | Partition Scheme: **Huge APP** |

`cam_bridge` is Espressif-core only — the ESP32-CAM board does not exist in the Arduino package,
and the sketch uses 3.x-only APIs (`pin_sccb_sda`, `WiFiServer::accept()`).

Flash the ESP32-CAM through the **ESP32-CAM-MB** carrier board, not bare FTDI wires. The MB has
the DTR/RTS auto-reset circuit; bare FTDI does not, so esptool cannot enter the bootloader
without you manually timing a reset, and long jumper wires corrupt the upload
(`Invalid head of packet`).

---

## 5. Bring-up, in order

Flash the **CAM first** and leave it powered, so the Nano has an AP to join on its first boot.
Otherwise you get `join failed` and cannot tell whether it is a real fault.

Do not involve the Tello until the glasses work end to end. Debug one network at a time.

**0.** `pip install opencv-python numpy` — do it now, both APs are offline networks.

**1.** Flash `cam_bridge`. Serial at 115200 should say:
```
AP up: join "rover" then http://192.168.4.1/
```
Leave it powered from here on.

**2.** Flash `nano_tof` over USB-C. **No servo power yet.** All four lines must appear:
```
servo PWM ok (14-bit @ 50 Hz, floor 600 us)
ToF L ok  ToF R ok
joining "rover"....
joined, data on 192.168.4.50:4210
```
Each failure points somewhere different:

| Line | Meaning |
|---|---|
| `servo PWM FAILED` | LEDC rejected the resolution. See §7. |
| `ToF L FAIL` / `ToF R FAIL` | XSHUT wiring, or that sensor. Swap the D9/D8 jumpers: if the fault moves, it is the wire; if it stays with the same sensor, it is the sensor. |
| `join failed` | CAM not powered, or SSID/password mismatch between the two sketches. |

**3.** Plug in the USB WiFi dongle and join `rover` / `rover1234`. Windows will say "no internet" —
correct, ignore it.

**4.** Open `http://192.168.4.1/health`. You want:
```json
{"cam":false,"fps":0.0,"clients":2,"uptime_ms":...,"rssi":0}
```
**`clients:2` is the check that matters** — laptop plus Nano. `clients:1` means the Nano never
associated; go back to step 2. (`cam:false` is normal until something opens `/stream`.)

**5.** `http://192.168.4.1/` — live video should appear in the browser.

**6.** `python host/rover_client.py`. Video window with a ToF/press overlay.
`a` cues left, `d` cues right, `space` releases, `p` pulses, `q` quits.

**7.** Only now connect servo power.

**8.** Afterwards: join the Tello's AP on the *other* adapter. Subnets differ (`192.168.4.x` vs
`192.168.10.x`), so Windows routes both with no conflict and `main.py` needs no changes.

---

## 6. Reading the telemetry by eye

With no laptop connected, the Nano prints telemetry to USB serial instead of UDP — so serial
`T,` lines mean **no client has registered yet**, which is itself useful information.

```
T,554,30337,-1,137,0,0,0
  │   │     │   │  │ │ └─ link  1 = a command arrived in the last 600 ms
  │   │     │   │  └─┴─── press left, right (0..100), as actually applied
  │   │     │   └──────── tof_right mm
  │   │     └──────────── tof_left mm   (-1 = out of range OR dead sensor)
  │   └────────────────── uptime ms
  └────────────────────── seq, +1 per packet
```

Lines 66 ms apart is healthy (~15 Hz). **Exactly 50 ms apart means one sensor is dead** — a
failed sensor returns instantly instead of blocking ~33 ms, which speeds the loop up. A
suspiciously *fast* telemetry rate is a symptom, not a win.

---

## 7. Two traps that cost us an evening each

**LEDC resolution must be ≤ 14 bits on the ESP32-S3.** `SOC_LEDC_TIMER_BIT_WIDE_NUM` is 14 (it is
20 on the original ESP32, which is where the usual 16-bit example code comes from).
`ledcSetup()`/`ledcAttach()` *reject* anything wider: they return 0/false and configure **no timer
at all**, so `ledcWrite()` writes a duty into an unconfigured channel and the pin never toggles.
The servos sit perfectly still while every number in the firmware — angles, microseconds,
telemetry — looks correct. It is a **runtime** limit, so a 16-bit build compiles clean on both
cores and silently does nothing. That is why both sketches print their attach result at boot:
**trust that line, not the build.**

**`SERVO_FLOOR_US` is 600, not the 500 µs that 0° maps to.** 500 may sit against the servo's
internal end stop, where an SG90 stalls, draws ~700 mA and destroys its own gearbox in minutes —
silently, if serial is unplugged. 600 is bench-validated on this mechanism. Lower it only after
watching an arm hold at 600 quietly and cool.

Related, from the same family of bug: **`constrain()` is a macro that evaluates its first
argument up to three times.** `constrain(nextInt(0), 0, 100)` therefore consumes three fields of
the input line and returns the default. That made `P` ignore its depth (always full press) and
made `C` set both calibration angles to 0. Always read into a local, then clamp the local.

---

## 8. Tuning constants

All at the top of `nano_tof/nano_tof.ino`.

| Constant | Default | Why you would change it |
|---|---|---|
| `MAX_PRESS_TRAVEL_DEG` | 90 | **Lower this before anyone wears the rig for real.** At 90 it permits the full mechanical range and gives no safety margin. It is the only thing between a runaway command and the wearer's face. |
| `DIR_PRESS_PCT` | 100 | Press depth a bare `D,<dir>` uses. Turn down once someone has worn it. |
| `SERVO_FLOOR_US` | 600 | Inner pulse limit. See §7. |
| `L_/R_RELEASED_DEG` | 90 | Arm clear of the face. |
| `L_/R_PRESSED_DEG` | 0 | Arm fully inward. Lower than released on purpose — shorter pulse = inward on this linkage. |
| `LINK_TIMEOUT_MS` | 600 | Failsafe window. The host resends at 20 Hz, so there is 12× margin. |

Sides can also be trimmed at runtime without reflashing: `link.calibrate("L", 90, 0)`.

---

## 9. Safety

**If no command arrives for 600 ms, the Nano releases both pads to 0%.** It does not hold the
last position, and that is deliberate: a servo latched against a blind person's face after a WiFi
drop is the worst thing this system can do. Link loss degrades to *no guidance*, which is a safe
state you should surface by speech instead.

SG90s have **no holding torque without power** — they go limp and back-drive. So power loss also
means the pads relax. That is the failure direction you want, but it means a press only exists
while both power and signal are live.

---

## 10. State of testing — be honest in the demo

| Path | Status |
|---|---|
| Servo PWM driving the arms through full travel | verified on hardware |
| Both VL53L0X at `0x30`/`0x31` returning mm | verified on hardware |
| CAM access point + MJPEG in a browser | verified on hardware |
| Nano associating to the CAM's AP, static IP | verified on hardware (`clients:2`) |
| Command parsing, press math, travel clamp, burst timing, failsafe release | verified in a native test harness |
| `direction()` sign mapping | verified by direct execution against a stubbed socket |
| Telemetry + commands over UDP, end to end | **not yet run** |
| Haptic cue felt by a wearer | **not yet run** |
| Two-adapter laptop (glasses + drone simultaneously) | **not yet run** |

The fastest way to move a row up is `python host/rover_client.py`.
