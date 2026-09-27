"""Run one episode of a controller in the pursuit simulator and score it (plan sections 4.4 and 4.5).

The reward uses the true simulator state; the controller only sees the noisy,
delayed box through the shared Kalman filter.
"""

from __future__ import annotations

import math

import numpy as np

from ..pilot.governor import PersonGovernor
from ..pilot.kalman import BoxFilter, observation
from ..sim.camera import DetectorModel, project
from ..sim.drone_model import DroneModel
from ..sim.walker import StaticObject, Walker

TERMS = ["range", "image", "jerk", "safety", "lost", "close"]


def run_episode(controller, scenario: dict, cfg: dict) -> dict:
    ep = cfg["episode"]
    rw = cfg["reward"]
    cam = cfg["camera"]
    dt = ep["dt"]
    kind = scenario["kind"]

    drone = DroneModel(scenario, dt, np.random.default_rng(scenario["camera_seed"] + 1))
    if kind == "follow":
        target = Walker(scenario, cfg["person"], dt, np.random.default_rng(scenario["walker_seed"]))
    else:
        target = StaticObject(scenario)
    detector = DetectorModel(scenario, cam, np.random.default_rng(scenario["camera_seed"]))
    box_filter = BoxFilter(cfg["drone"]["measured"]["video_latency_s"], cfg["governor"]["lost_hold_s"])
    governor = PersonGovernor(cfg["governor"], scenario)
    controller.reset(scenario)

    z_ref = scenario["follow_distance_m"]
    if kind == "approach":
        band = rw["band_frac_approach"] * z_ref
    else:
        band = rw["band_frac"] * z_ref
    hfov_half = math.atan(cam["width"] / 2.0 / cam["fx_nominal"])
    collision_dist = 0.5 if kind == "follow" else 0.2
    n_ticks = int(round(scenario["duration_s"] / dt))

    weights = {"range": rw["w_range"], "image": rw["w_image"], "jerk": rw["w_jerk"],
               "safety": rw["w_safety"], "lost": rw["w_lost"], "close": rw["w_close"]}
    max_rate = weights["range"] * 3.0 + weights["image"] + weights["safety"] + weights["lost"] + weights["close"]

    totals = {}
    for term in TERMS:
        totals[term] = 0.0
    events = 0.0
    in_band_ticks = 0
    in_band_run = 0.0
    lost_active = False
    close_active = False
    loss_events = 0
    close_events = 0
    safety_ticks = 0
    min_dist = float("inf")
    range_errors = []
    bearing_errors = []
    jerk_sum = 0.0
    prev_yaw = 0.0
    prev_fb = 0.0
    outcome = "timeout"
    success_time = None
    ticks_run = 0

    for i in range(n_ticks):
        t = i * dt
        target.drone_xy = (drone.x, drone.y)
        target.step()
        size = scenario["target_size_m"]
        detector.capture(t, drone, target, size)
        for t_avail, t_cap, box in detector.available(t):
            box_filter.update(t_avail, box)
        est = box_filter.estimate(t)
        obs = observation(est, scenario, cam)

        yaw, fb = controller.act(obs, dt)
        yaw, fb, ud, flags = governor.filter(yaw, fb, obs, drone.z, dt)
        drone.step(fb, yaw, ud)
        ticks_run += 1

        # true state after the step
        dist = math.hypot(target.x - drone.x, target.y - drone.y)
        true_box = project(drone, target.x, target.y, target.z, size, scenario["fx"], scenario["fy"], cam["width"], cam["height"])
        in_view = true_box is not None
        min_dist = min(min_dist, dist)

        d = dist - z_ref
        e_range = max(0.0, abs(d) - band) + max(0.0, d - band) ** 2
        e_range = min(e_range, 3.0)
        dx = target.x - drone.x
        dy = target.y - drone.y
        forward = dx * math.cos(drone.psi) + dy * math.sin(drone.psi)
        left = -dx * math.sin(drone.psi) + dy * math.cos(drone.psi)
        rel_bearing = math.atan2(-left, forward)  # positive = target to the right
        planned = obs["side_offset"]
        e_image = min(abs(rel_bearing - planned) / hfov_half, 1.0)
        jerk = ((yaw - prev_yaw) ** 2 + (fb - prev_fb) ** 2) / 100.0 ** 2 * 20.0
        jerk_sum += abs(yaw - prev_yaw)
        prev_yaw = yaw
        prev_fb = fb
        lost = est["age"] > 1.0
        close = dist < scenario["min_dist_m"]

        totals["range"] += e_range * dt
        totals["image"] += e_image ** 2 * dt
        totals["jerk"] += jerk * dt
        totals["safety"] += (1.0 if flags["safety"] and "lost" not in flags["reasons"] else 0.0) * dt
        totals["lost"] += (1.0 if lost else 0.0) * dt
        totals["close"] += (1.0 if close else 0.0) * dt
        if flags["safety"]:
            safety_ticks += 1

        if lost and not lost_active:
            events += rw["ev_loss"]
            loss_events += 1
        lost_active = lost
        if close and not close_active:
            events += rw["ev_close"]
            close_events += 1
        close_active = close

        range_errors.append(d)
        bearing_errors.append(rel_bearing - planned)
        if abs(d) <= band and in_view:
            in_band_ticks += 1
            in_band_run += dt
        else:
            in_band_run = 0.0

        if dist < collision_dist:
            outcome = "collision"
            events += rw["ev_collision"]
            break
        if governor.landed:
            outcome = "lost_land"
            break
        if kind == "approach" and in_band_run >= rw["approach_success_hold_s"]:
            outcome = "success"
            success_time = t
            events += rw["ev_approach_success"]
            break

    cost = 0.0
    for term in TERMS:
        cost += weights[term] * totals[term]
    if outcome in ("collision", "lost_land"):
        remaining_s = (n_ticks - ticks_run) * dt
        cost += max_rate * remaining_s
    episode_return = -cost + events

    range_errors = np.array(range_errors)
    bearing_errors = np.array(bearing_errors)
    return {
        "return": float(episode_return),
        "terms": {term: float(weights[term] * totals[term]) for term in TERMS},
        "events": float(events),
        "outcome": outcome,
        "kind": kind,
        "in_band_frac": in_band_ticks / max(n_ticks, 1),
        "min_dist": float(min_dist),
        "loss_events": loss_events,
        "close_events": close_events,
        "safety_frac": safety_ticks / max(ticks_run, 1),
        "mean_range_err": float(range_errors.mean()) if range_errors.size else 0.0,
        "rms_range_err": float(np.sqrt(np.mean(range_errors ** 2))) if range_errors.size else 0.0,
        "rms_bearing_deg": float(np.degrees(np.sqrt(np.mean(bearing_errors ** 2)))) if bearing_errors.size else 0.0,
        "yaw_jerk_per_s": jerk_sum / max(ticks_run * dt, 1e-6),
        "success_time": success_time,
        "duration": ticks_run * dt,
    }
