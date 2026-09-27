# ReachGlass demo workflow — takeoff to "find arthur"

End-to-end runbook for the full system: Tello drone + fruit-fly yaw steering, AirPods voice
control, and the glasses' haptic pads. Written against the wiring as of 2026-09-27.

## The pieces and how they talk

```
AirPods (stem + mic + ears)
   │ Bluetooth
   ▼
laptop: python -m voice ──── UDP 5005 (queries) ───► python -m reachglass tello
   ▲                                                        │
   └──────────── UDP 5006 (announcements → TTS) ◄───────────┤
                                                            │ WiFi "TELLO-…" (adapter 1)
                                                            ▼
                                                      Tello drone (video + rc)

glasses (worn):  ESP32-CAM hosts WiFi "rover" (adapter 2)
                 Nano ESP32 (192.168.4.50, UDP 4210): ToF + the two haptic pads
                 reachglass drives the pads via firmware/host/reachglass_glasses.py
```

- **Voice in:** stem single-press → mic opens → Whisper → text → UDP 5005 → mission query parser.
- **Voice out:** every `mission.announce(...)` → UDP 5006 → spoken in the AirPods.
- **Haptics:** the guide stage emits cues `-1 / 0 / +1 / 2`; `reachglass/glasses.py` forwards them
  to the Nano. **Sides are crossed on purpose** (firmware contract): cue **+1 "turn right"
  presses the LEFT pad**, cue **-1 "turn left" presses the RIGHT pad** — the pad pushes the head
  the way to go. Cue 2 (arrived) = double buzz on both pads. Cue changes are also spoken
  ("Turn left." / "Walk forward." / "You're there. It's right in front of you.").
- **Fly brain:** `site.yaml` sets `follow.steering: fly` — the 1,446-neuron fruit-fly circuit
  drives the yaw stick during FOLLOW (distance/altitude stay PID; `fly.forward` must stay
  `false`, the trained weights are yaw-only).

## 0. Hardware setup (once per session)

1. **Laptop WiFi ×2:** built-in adapter → the Tello's `TELLO-XXXXXX` network; USB adapter →
   the glasses' `rover` network (password `rover1234`). Different subnets, no conflict.
   Check the glasses radio: `http://192.168.4.1/health` should report `"clients":2`.
2. **AirPods** paired to the laptop (they are both the mic and the speaker).
3. **Glasses on the wearer**, servos free to move. Bench check if unsure:
   `python firmware/host/reachglass_glasses.py` (prints ToF, sweeps both pads).
4. Wearer's height is in `site.yaml` (`person_height_m`) — it sets follow geometry.

## 1. Start the software

Two terminals, venv active (`.\.venv\Scripts\activate`):

```powershell
python -m reachglass tello --config site.yaml     # terminal 1: the drone stack
python -m voice                                   # terminal 2: AirPods voice control
```

Startup lines worth watching in terminal 1:

- `fly steering loaded: …` — the fruit-fly brain is in. If you see
  `fly steering unavailable`, it flies on the PID yaw law instead (still safe, tell the team).
- `glasses haptics up: 192.168.4.50` — pads connected. `glasses unavailable (…)` means cues
  will print + speak only; the run continues.

Terminal 2 says "Voice control ready." in the AirPods when it's up.

## 2. Takeoff and FOLLOW

- Wearer: single stem press → chirp → say **"takeoff"**.
- Drone takes off, climbs to `follow.altitude_m` (1.9 m) and enters **FOLLOW**: it sits
  1.8 m behind and above the wearer, fly brain on yaw (keeps the person centered), PID on
  distance and altitude. It follows as they walk; if it loses them it holds, then turns
  toward where they were last seen.
- Orbit-behind is currently **off** (`orbit_gain: 0` — noisy facing estimates caused lateral
  drift; see the investigation notes). While walking, the drone ends up behind them naturally.

## 3. "Find my water bottle"

Wearer: stem press → **"find my water bottle"**.

1. Mission answers **"Looking for your water bottle."** (in the AirPods) and switches
   FOLLOW → EXPLORE. It records where the wearer stood — guidance later is computed for them.
2. **EXPLORE:** the drone scans the room. YOLO-World looks for the team's blue Hydro Flask
   specifically (open-vocabulary prompts + a blue-color check so someone else's bottle is not
   taken for it); furniture detections build a semantic prior (bottles live on tables/desks).
3. On a confirmed sighting: **"I see the bottle. Going there."** → **APPROACH**: turn to the
   mapped position, climb over its top if needed, fly straight to it and 0.2 m past
   (Arthur's map-based approach). If a person is in the path, it stays put and says so.
4. **ARRIVED:** the drone hovers just past the bottle, announces distance and direction for
   the wearer (e.g. "Three steps away, slightly to your left (ten o'clock).").

## 4. Guiding the person to the bottle (GUIDE)

The mission looks back, finds the wearer, plans a walking path, and streams cues:

| Cue | Pads (crossed!)        | Voice              | Meaning              |
|-----|------------------------|--------------------|----------------------|
| -1  | RIGHT pad presses      | "Turn left."       | turn/step left       |
|  0  | both release           | "Walk forward."    | keep walking forward |
| +1  | LEFT pad presses       | "Turn right."      | turn/step right      |
|  2  | double buzz, both pads | "You're there. It's right in front of you." | arrived |

- Cues repeat at ~1 Hz to the pads (the Nano needs a live stream — 600 ms of silence
  auto-releases, so a laptop crash can never leave a pad pressed into a face).
- Speech only fires when the cue *changes* — no nagging.
- Wearer briefly out of view: last cue keeps repeating; the drone turns to reacquire.
- Double stem press any time = repeat / refresh guidance. After arrival it recomputes from
  the wearer's current position instead of restarting the search.
- After the arrived cue, the drone lands.

## 5. "Find arthur"

Same flow, different target — requires Arthur enrolled (his photo is in `people/`,
embeddings in `people/faces.npz`; add people with `tools/enroll_faces.py`).

- Wearer: stem press → **"find arthur"**. The parser recognizes enrolled names; the mission
  answers **"Looking for Arthur."**
- EXPLORE uses the person prior (chairs, couches, tables) instead of the bottle prior.
  Person detections get face-ID'd; only a face match counts as finding *Arthur*.
- APPROACH treats a person target more conservatively (stops short, never flies over).
- GUIDE then walks the wearer to Arthur with the same pads + voice cues.

## Any time / safety

| Say                | Effect                                              |
|--------------------|-----------------------------------------------------|
| "stop" / "cancel"  | abandon the search, come back to following          |
| "land"             | land now                                            |
| triple stem press  | sends "stop" (deliberately NOT mapped to "land")    |
| "where is it?" / repeat query | re-announce guidance without restarting  |

- SafetyGovernor clamps every rc command; ceiling `safety.max_altitude_m` 2.3 m.
- Fly brain failure mid-flight → automatic fallback to the PID yaw law, one warning logged.
- Glasses link loss → pads release (never latch), guidance continues by voice.
- `Ctrl+C` in terminal 1 → land.

## Known state / gotchas (2026-09-27)

- **Haptics end-to-end over UDP is not yet proven on hardware** (firmware doc §9): the pads,
  sensors and protocol are bench-verified, but "cue felt by a wearer during a run" has not
  been. Do one dry GUIDE with a hand on the pads before the real demo.
- Verify pad left/right once on the assembled glasses (sensor/servo sides are a wiring fact,
  not a software guarantee): `link.direction(+1)` must press the physical LEFT pad.
- `fly.forward` stays `false` — the trained fly weights are yaw-only; enabling it flies on an
  untrained forward readout (holds ~8-10 m and surges; measured 2026-09-27).
- Both WiFi networks are offline — pip installs must happen beforehand.
- Voice needs the AirPods as input device; if STT is slow, run `python -m voice --stt tiny.en`.
