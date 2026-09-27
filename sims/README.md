# Simulations: every example use case

Run everything from the repo root (`jerkgt13/`) with the venv active (`source .venv/bin/activate`).
The simulator renders a 7 x 6 m living room (table with the blue bottle, chairs, couch, shelf), a walking person
and a Tello-like drone. No drone needed.

## 1. Watch it live (dashboard window)

| Use case | Command |
|---|---|
| Whole pipeline: follow -> "find my water bottle" -> search -> fly over it -> look back -> guide the person (cues `-1/0/1`, then `2`) -> land | `python -m reachglass sim --query "find my water bottle" --at 25` |
| Same, another random room/noise seed | `python -m reachglass sim --query "find my water bottle" --at 25 --seed 3` |
| Same, at real-world speed | `python -m reachglass sim --query "find my water bottle" --at 25 --realtime` |
| Same, no window, saved to a video | `python -m reachglass sim --query "find my water bottle" --at 25 --headless --record runs/mission.mp4` |
| Follow the person only (take off, climb, stay 1 m behind) | `python -m reachglass sim --seconds 60` |
| Type your own requests while it runs | `python -m reachglass sim`, then type in the terminal: `find my water bottle`, `follow me`, `what's around me`, `stop`, `land` |

Keys in the dashboard: `SPACE` hold/resume, `l` land, `e` emergency stop, `q` land and quit.
In the simulator the person does not follow the guide cues (they are only printed as `guide cue: N`);
section 2's guide test has a simulated person who does.

## 2. Closed-loop scenario runs (pass/fail, no window)

Each is one pytest command. Add `-s` to see the printed cues and messages.

| Use case | Command |
|---|---|
| **Guide:** whole pipeline, a simulated person walks by the cues around the chairs to the table, cue `2`, drone lands | `python -m pytest -m slow tests/test_guide.py::test_whole_pipeline_guides_the_wearer_to_the_bottle_then_lands` |
| Guide: path planner and cue logic only (fast) | `python -m pytest tests/test_guide.py -m "not slow"` |
| Whole mission: follow, find, approach, guidance text, "follow me", land | `python -m pytest -m slow tests/test_mission.py::test_full_mission_follow_find_approach_guide_follow_land` |
| Bottle behind the drone's start, found by scanning | `python -m pytest -m slow tests/test_scenarios.py::test_target_behind_the_start_found_by_scanning` |
| Bottle on the floor | `python -m pytest -m slow tests/test_scenarios.py::test_target_on_the_floor_needs_descent` |
| Person walks away during the search, "follow me" finds them | `python -m pytest -m slow tests/test_scenarios.py::test_person_walks_away_during_search_then_follow_me_finds_them` |
| Explore + approach from several starts | `python -m pytest -m slow tests/test_explore.py::test_explore_then_approach_reaches_the_target` |
| Bottle too far to see from the start: needs hops, odometry stays accurate | `python -m pytest -m slow tests/test_review_fixes_9_12.py::test_target_out_of_detection_range_requires_hops_and_odometry_stays_accurate` |
| Whole mission with a long video lag | `python -m pytest -m slow tests/test_review_fixes_9_12.py::test_mission_with_long_video_lag` |
| Direction told to the person is right, across seeds | `python -m pytest -m slow tests/test_review_fixes_9_12.py::test_guidance_turn_is_right_across_seeds` |
| Follow: hover behind a standing person | `python -m pytest tests/test_follow.py::test_climbs_to_follow_altitude_and_holds_distance_behind_standing_person` |
| Follow: person walks and turns | `python -m pytest tests/test_follow.py::test_follows_walking_person_through_a_turn_and_ends_up_behind_them` |
| Follow: person turns to face the drone (orbit behind) | `python -m pytest tests/test_follow.py::test_person_turning_to_face_the_drone_makes_it_orbit_behind` |
| Follow: person lost, drone searches toward their side | `python -m pytest tests/test_follow.py::test_person_lost_then_search_turns_toward_last_side` |
| Follow steered by the fruit-fly controller | `python -m pytest tests/test_fly_steer.py::test_fly_steers_follow_closed_loop` |
| Scan from one spot: finds the bottle / covers 360 deg | `python -m pytest tests/test_explore.py -m "not slow"` |
| Requests: unknown object, "what's around me", nonsense | `python -m pytest tests/test_mission.py::test_unknown_target_and_describe_keep_following` |
| Waits on the ground until "takeoff" | `python -m pytest tests/test_mission.py::test_waits_on_the_ground_for_takeoff` |
| "Never mind" during the search, then "land now" | `python -m pytest tests/test_mission.py::test_land_during_search_and_cancel` |
| Low battery mid-search: safety lands | `python -m pytest tests/test_mission.py::test_safety_lands_on_low_battery_mid_search` |
| All closed-loop scenarios at once (~15 min) | `python -m pytest -m slow` |

Known failures right now (the tests are older than the latest changes; the behaviour is expected):
- `test_full_mission_follow_find_approach_guide_follow_land`: expects the drone to stop 0.7-2 m from the bottle,
  but it now flies over it.
- `test_target_on_the_floor_needs_descent`: the search stays at 2 m, where a floor bottle closer than ~4.5 m is
  below the camera's view.
- `test_follows_walking_person_through_a_turn_and_ends_up_behind_them`: expects 1.3-2.4 m behind; follow is now 1.0 m.
