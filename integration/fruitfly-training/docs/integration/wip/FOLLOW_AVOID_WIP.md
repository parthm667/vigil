# Obstacle avoidance while following (FollowAvoid) for ReachGlass

For the ReachGlass team (github.com/nathanwuzhao/jerkgt13). Patch: `docs/integration/reachglass_follow_avoid.patch`,
made against your HEAD `2302928`. Nothing was pushed to your repository: you apply the patch yourselves.

## What it does

FOLLOW had no obstacle handling. The patch adds three things. `FollowBehind` is unchanged: yaw (your PID or
the fly), distance, altitude, orbit and the lost-person search are all still decided there.

* **Furniture during FOLLOW.** `perception.stride.follow` becomes `{person: 1, target: 3, context: 5, looming: 1}`.
  The target detector runs only once a target class is set. Context detections go into memory and the grid
  the same way the search does (`explore.observe`, at most every 0.5 s). While following, the pose is
  dead-reckoned from the rc actually sent (`Odometry.on_velocity`, rc 100 = 1 m/s, 0.35 s lag), so the map
  stays put.
* **Looming** (`detect/looming.py`, `LOOMING` registry, `perception.looming`). This is the fly's LPLC2 channel,
  adapted from the FlyDrones retina (MIT). The frame is downsampled to 96x72 and compared with the frame
  0.12 s earlier. The global image shift is removed first (turning is not looming). Then each cell of a 3x4
  grid over the central 70 % of the frame, plus the whole region as one cell, fits a local expansion.
  Pixels in the locked person's box are ignored. The output `res.looming` holds: rate (1/s), TTC, the side of
  the looming patch, and `texture` (the fraction of cells that can measure; 0 means blind, which is not clear).
* **`FollowAvoid(FollowBehind)`** (`behaviors/follow_avoid.py`). The mission now builds it where it built
  `FollowBehind`. It catches FollowBehind's rc and then:
  * **Brakes** (forward capped at 0) when either of these holds:
    * An obstacle is in the 1.0 m x 0.6 m forward corridor and reaches flight height (your grid rule: top +
      0.2 m). The obstacle can come from recent detections or from the grid. Other people count; the followed
      person does not.
    * Looming says so on two frames in a row. While flying, that means TTC < 2.5 s and own speed x TTC <
      0.9 m. While hovering, it means TTC < 1.2 s.

    It backs away at rc 15 only from seen obstacles closer than 0.45 m. A looming patch is remembered as an
    obstacle for 3 s.
  * **Goes round** (`sidestep`).
    * **When it starts:** something seen (a detection, or a bystander) blocking the corridor starts it after
      0.3 s. Something known only from looming or the map (extent unknown, may be a wall) starts it after
      1.5 s.
    * **Side:** the one toward the user's last position, checked like `approach.py`'s sidestep (room of
      0.5 m or more; the slide keeps 1 m from every person).
    * **Slide:** it slides with rc until the corridor is clear of the obstacle's keep-out. A seen obstacle
      gets at most `detour_max_m` (1.5 m), an unknown one `sidestep_m`. The total sideways offset per blocked
      episode is capped at 1.5 m (walls are not mapped).
    * **Pass and return:** it then passes the obstacle while holding the keep-out, and drifts back toward
      the line it was on (at most 3 s).
    * **Yaw** stays with FollowBehind (fly or PID) throughout.
  * **Bystanders.** Other people get a 1.1 m keep-out disc (1 m plus range noise).
    * `perception.py` now locks the **nearest** person by range estimate, not the largest box. From 2 m up,
      a user close ahead shows only a head, so the largest box was a bystander 3 m away: it locked the
      bystander from the first FOLLOW frame.
    * FollowAvoid remembers where bystanders stood (30 s). A person at that spot and more than 1.2 m from
      where the user should be goes into `Perception.not_the_person`: the lock releases it and never
      re-selects it, and FollowBehind never steers at it.
  * **Ducks under headers** (`duck`, **opt-in**). All four must hold:
    * the corridor is blocked only by a looming patch (nothing detected within 2.5 m ahead);
    * the user's own recent path went within 0.3 m of it (they walked under it, so a person-height gap
      exists below);
    * nothing looms in the bottom-centre looming cells (below the horizon, straight ahead);
    * the drone is not already ducking.

    Then it descends to (estimated bottom - 0.35 m), never below `duck_floor_m` (1.2 m). The estimated
    bottom is the drone's height, because the obstacle blocked it there. While ducked:
    * the looming brake is off (the header is above and the posts are to the sides); seen obstacles still
      brake;
    * it drives forward itself at rc 20 while the user's box stays small (far). Below head height your person
      range cues fail, and FollowBehind would not move;
    * it climbs back to `follow.altitude_m` once the header is 0.6 m behind, or after 20 s if not right
      under it.

    Only the duck relaxes `follow.altitude_m >= wearer + 0.2`.
  * **Rear rule.** It never backs into something mapped behind it. Hovering next to a mapped obstacle makes it
    drift away.
  * **Occlusion recovery** (`recover`). This applies when the user vanishes while it brakes or sidesteps, or
    within 2 s after: an obstacle most likely hides them.
    * For up to `recover_max_s` (15 s), yaw holds on the user's last-seen spot, moved on along their walking
      direction by at most 1 m. Yaw uses your yaw law on that bearing, or the fly when `follow.steering: fly`.
    * The drone creeps there at rc 20. The brake and sidestep still apply, so it goes round the obstacle's free
      side. It stops 1 m short of that spot.
    * Meanwhile `FollowAvoid.reacquire_wait` holds off the mission's REACQUIRE (a 3-line hook in
      `mission.py`).
    * When the user is seen again, normal following resumes.
    * It does not recover when the user vanished close (walking under the drone): FollowBehind's back-off
      runs then.

    Found while building this: your "person lost for 20 s" rule fires about 4.5 s after a loss once FOLLOW
    has run 20 s. That is because the person lock releases and `person_unseen_s` becomes inf. A timer
    extension through `person_unseen_s` could not work, hence the flag.

  Status strings are appended, for example `| looming patch 0.6 m ahead: braking` or `| sidestep right (0.3/0.6 m)`.
  Events go to `ctx.note` and the log. The run log gets `loom`, the dashboard draws the looming grid and TTC,
  and the sim takes `--scenario`.

## Config (`site.yaml`), defaults shown

```yaml
follow:
  avoid: {enabled: true,          # KILL SWITCH: false = exactly the old FollowBehind (no brake, no dead reckoning)
          looming: true, sidestep: true,
          use_map: false,                 # the grid never forgets; dead-reckoned sightings smear into ghosts
          corridor_m: 1.0, half_width_m: 0.3, reverse_m: 0.45, reverse_rc: 15,
          ttc_brake_s: 1.2, ttc_move_s: 2.5, stop_range_m: 0.9, max_yaw_rc: 20, hold_s: 1.0, memory_s: 3.0,
          detour_after_s: 0.3, sidestep_after_s: 1.5, sidestep_m: 0.6, detour_max_m: 1.5, sidestep_rc: 25,
          max_sidesteps: 2, person_clearance_m: 1.0,
          duck: false, duck_below_m: 0.35, duck_floor_m: 1.2, duck_max_s: 20.0,
          recover: true, recover_window_s: 2.0, recover_max_s: 15.0, recover_rc: 20}
perception:
  stride: {follow: {person: 1, target: 3, context: 5, looming: 1}}   # all 0 except person = the old CPU budget
  looming: {kind: divergence}     # or null (then set follow.avoid.looming: false; the config refuses otherwise)
```

## Apply

```bash
cd jerkgt13
git apply /path/to/reachglass_flysteer.patch      # optional, the other patch; apply it first
git apply /path/to/reachglass_follow_avoid.patch
python -m pytest tests/test_looming.py tests/test_follow_avoid.py
python -m reachglass sim --scenario doorway --headless --record doorway.mp4 --seconds 45
```

Both patches apply cleanly together on `2302928` in either order. With both applied, the fly steers yaw
while FollowAvoid brakes, sidesteps and recovers (see the final combined check below).

## Results

**Tests.** Your suite at `2302928` has 233 tests, with 3 failing before any change:
`test_follows_walking_person_through_a_turn...`, `test_full_mission_follow_find_approach_guide_follow_land` and
`test_target_on_the_floor_needs_descent`. They fail identically on the untouched clone.
* This patch alone: 262 tests (29 new). The same 3 fail and the rest pass. Two assertions in
  `test_mode_strides_decide_which_detectors_run` were updated for the new follow strides.
* Final combined copy (fresh `2302928` + the latest `reachglass_flysteer.patch` + this patch): 267 tests,
  264 passed, and only the same 3 failed.

**Cost per frame.** On an M-series CPU: looming takes 0.9 ms at 960x720 (0.7 ms at 480x360), and yolo11n
context takes about 25 ms every 5th frame, so about 5 ms per frame on average.

**Simulator** (`sim/follow_scenarios.py`, 50 s runs, 3 seeds each).
* The drone follows at 2.0 m, so tables and chairs are flown over.
* "In view" is the fraction of frames with the user locked, in every state after following starts.
* "Lost" is the time the user was unseen for more than 1 s, in any state.

Your PID yaw:

| scenario | mode | collisions (runs) | min dist to obstacle | user in view | time lost | ends in |
|---|---|---|---|---|---|---|
| doorway (header 2.03 m) | off | **3/3** (pinned on the header) | 0.12 m | 0.86 | 0 s | FOLLOW, stuck |
| | on | 0/3 | 0.33 m | 0.39 | 23.4 s | REACQUIRE, HOLD, HOLD |
| | on + recovery | 0/3 | 0.34 m | 0.50 | 18.5 s | FOLLOW, REACQUIRE, REACQUIRE |
| tall box in path | off | 0/3 | **0.25 m** (near miss) | 0.87 | 0 s | FOLLOW x3 |
| | on | 0/3 | 0.90 m | 0.29 | 28.1 s | HOLD x3 |
| | on + recovery | 0/3 | 0.40 m | 0.44 | 19.2 s | FOLLOW, FOLLOW, REACQUIRE |
| table + tall bookcase | off | 0/3 | **0.34 m** (near miss) | 0.86 | 0 s | FOLLOW x3 |
| | on | 0/3 | 1.03 m | 0.36 | 24.7 s | HOLD, HOLD, REACQUIRE |
| | on + recovery | 0/3 | 0.49 m | 0.42 | 21.9 s | REACQUIRE, REACQUIRE, FOLLOW |

Final combined check with `follow.steering: fly` and `fly.params_path: .../data/runs/FLY-YAW_s3_v1/best.json`.
Off and on are seed 0 only; on + recovery has 3 seeds.

| scenario | mode | collisions | min dist | user in view | time lost | ends in |
|---|---|---|---|---|---|---|
| doorway | off | **yes** (826 steps on the header) | 0.13 m | 0.89 | 0 s | FOLLOW, stuck |
| | on | 0 | 1.13 m | 0.36 | 25.0 s | HOLD |
| | on + recovery | 0/3 | 0.30 m | 0.41 | 21.9 s | REACQUIRE, FOLLOW, FOLLOW |
| tall box | off | 0 | 0.25 m | 0.86 | 0 s | FOLLOW |
| | on | 0 | 1.30 m | 0.28 | 28.8 s | HOLD |
| | on + recovery | 0/3 | 0.28 m | 0.31 | 27.3 s | REACQUIRE x3 |
| table + bookcase | off | 0 | 0.28 m | 0.89 | 0 s | FOLLOW |
| | on | 0 | 1.34 m | 0.40 | 23.2 s | REACQUIRE |
| | on + recovery | 0/3 | 0.64 m | 0.39 | 23.1 s | REACQUIRE x3 |

Missions with the fly, headless, 0 collisions both:
* follow-me: `--query "follow me"` for 60 s, stays in FOLLOW.
* find: "find my water bottle" at 25 s, then DESCEND, EXPLORE, APPROACH, and ARRIVED at about 60 s.

Read this honestly:
* Avoidance removes the doorway crash (every run without it) and the 25 to 35 cm near misses.
* Occlusion recovery gets the user back in 5 of 9 PID runs (end in FOLLOW) versus 0 of 9 without it, and
  cuts lost time by 3 to 9 s. It works less well with the fly (2 of 9 runs end in FOLLOW), because with
  larger yaw commands the drone mostly turns and creeps less.
* The price is clearance. The creep goes back toward the obstacle's edge, so minimum distance drops from
  about 1 m to 0.3 to 0.5 m. There were no collisions after the fix below.
* An earlier version collided with the fly: its larger yaw switched the looming check off while creeping.
  So the creep now only moves when not turning hard, when looming has texture ahead, and for at most 1.5 m.

Videos (scratchpad `final_videos2/`) show the final combined copy with the fly, avoid and recovery. The earlier
off / on videos with your PID are in `avoid_videos/`.

## Limits and open issues

* **Forward camera only.** There is no rear or side sensing: the rear and side rules only know what was
  mapped. Walls are not mapped.
* **Weak looming.** Blank walls and dark rooms give none (`texture` goes to 0 and the status says "looming
  blind"). Occlusion edges, such as door jambs against the room behind, can over-read (a doorway braked
  about 2 m early in 2 of 3 runs). Turns faster than yaw rc 20 are ignored.
* **Real Tello unverified.** The derotation uses a plain shift. On sim renders it beat the rotation
  homography (kept as `_rotation`). Verify on real video with a dry run and watch the dashboard's looming grid.
* **Occlusion recovery** is partial. It creeps toward where the user was, never into what looming cannot see.
  A blank wall ahead, or a turn, stops the creep; it then hovers facing that spot until `recover_max_s` runs
  out. Sidesteps have no side camera, so they only avoid mapped obstacles. Its clearance to the occluder is
  0.3 to 0.5 m.
* **Tall COCO classes cut off by the frame bottom.** When the camera sees only the top of one of these, its
  range comes from the top edge and the class height prior (as for a head). A fridge at 1.70 m is not an
  obstacle at 2.0 m altitude (your 0.2 m rule).
* **Speed source.** Own speed comes from the commanded rc, as in `follow.py`. Telemetry `vgx` could replace
  it once its frame and sign are checked.
