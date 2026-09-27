"""Sample one episode's randomized conditions from a seed (domain randomization, plan section 4.4)."""

from __future__ import annotations

import math

import numpy as np


def _uniform(rng: np.random.Generator, pair) -> float:
    return float(rng.uniform(pair[0], pair[1]))


def sample_scenario(cfg: dict, seed: int, kind: str) -> dict:
    """Return a flat dict describing the drone, camera, governor settings and target for one episode."""
    rng = np.random.default_rng(seed)
    dr = cfg["drone"]["randomize"]
    cam = cfg["camera"]
    gov = cfg["governor"]
    person = cfg["person"]
    obj = cfg["objects"]
    ep = cfg["episode"]

    s = {"seed": seed, "kind": kind}

    # drone response
    s["yaw_dead_time_s"] = _uniform(rng, dr["yaw_dead_time_s"])
    s["yaw_tau_s"] = _uniform(rng, dr["yaw_tau_s"])
    s["yaw_gain_dps_per_100"] = _uniform(rng, dr["yaw_gain_dps_per_100"])
    s["fwd_dead_time_s"] = _uniform(rng, dr["fwd_dead_time_s"])
    s["fwd_tau_s"] = _uniform(rng, dr["fwd_tau_s"])
    s["fwd_gain_mps_per_100"] = _uniform(rng, dr["fwd_gain_mps_per_100"])
    s["vert_gain_mps_per_100"] = _uniform(rng, dr["vert_gain_mps_per_100"])
    s["vert_dead_time_s"] = cfg["drone"]["measured"]["vert_dead_time_s"]
    s["video_latency_s"] = _uniform(rng, dr["video_latency_s"])
    s["detector_time_s"] = _uniform(rng, dr["detector_time_s"])
    s["detection_rate_hz"] = _uniform(rng, dr["detection_rate_hz"])
    s["hover_drift_sigma_mps"] = _uniform(rng, dr["hover_drift_sigma_mps"])

    # camera and detector
    s["fx"] = _uniform(rng, cam["fx_range"])
    s["fy"] = s["fx"] * cam["fy_nominal"] / cam["fx_nominal"]
    s["box_center_sigma_px"] = _uniform(rng, cam["box_center_sigma_px"])
    s["box_height_noise"] = _uniform(rng, cam["box_height_noise"])
    s["dropout_iid"] = _uniform(rng, cam["dropout_iid"])
    s["burst_rate_per_s"] = _uniform(rng, cam["burst_rate_per_s"])
    s["burst_len_s"] = list(cam["burst_len_s"])
    s["false_box_rate"] = _uniform(rng, cam["false_box_rate"])

    # governor settings (tunable after training, so randomized in training)
    s["max_fwd_stick"] = _uniform(rng, gov["max_fwd_stick"])
    s["side_offset_deg"] = _uniform(rng, gov["side_offset_deg"])
    s["cy_ref_frac"] = _uniform(rng, gov["cy_ref_frac"])
    v_max = s["fwd_gain_mps_per_100"] * s["max_fwd_stick"] / 100.0
    s["drone_v_max"] = v_max

    if kind == "follow":
        s["follow_distance_m"] = _uniform(rng, gov["follow_distance_m"])
        s["min_dist_m"] = _uniform(rng, gov["min_person_dist_m"])
        s["person_height_m"] = _uniform(rng, person["height_m"])
        s["target_size_m"] = _uniform(rng, person["head_size_m"])
        s["target_size_assumed_m"] = person["head_size_assumed_m"]
        s["target_z_m"] = s["person_height_m"] - 0.12
        s["person_speed_max"] = min(person["speed_cap_mps"], person["speed_frac_of_drone_max"] * v_max)
        start_range = s["min_dist_m"] + _uniform(rng, ep["follow_start_range_extra_m"])
        start_bearing = math.radians(_uniform(rng, ep["follow_start_bearing_deg"]))
        s["duration_s"] = ep["follow_s"]
        s["drone_alt_m"] = s["target_z_m"] + float(rng.uniform(-0.2, 0.4))
        s["alt_floor_m"] = 0.5
    else:
        size = _uniform(rng, obj["size_m"])
        s["target_size_m"] = size
        s["target_size_assumed_m"] = size / _uniform(rng, obj["prior_mismatch"])
        if rng.random() < obj["floor_frac"]:
            s["target_z_m"] = size / 2.0
        else:
            s["target_z_m"] = _uniform(rng, [0.4, obj["height_m"][1]])
        s["alt_floor_m"] = gov["approach_alt_floor_m"]
        s["follow_distance_m"] = max(1.0, 2.5 * (s["alt_floor_m"] - s["target_z_m"]))
        s["follow_distance_m"] = min(s["follow_distance_m"], 2.0)
        s["min_dist_m"] = gov["min_object_dist_m"]
        start_range = s["follow_distance_m"] + _uniform(rng, obj["start_extra_range_m"])
        start_bearing = math.radians(_uniform(rng, obj["start_bearing_deg"]))
        s["duration_s"] = ep["approach_s"]
        s["drone_alt_m"] = max(s["alt_floor_m"], s["target_z_m"] + float(rng.uniform(0.2, 0.8)))

    # place the drone at the origin facing +x; target at the start range and bearing (positive = right)
    s["target_x0"] = start_range * math.cos(start_bearing)
    s["target_y0"] = -start_range * math.sin(start_bearing)  # y is to the drone's left
    s["walker_seed"] = int(rng.integers(0, 2**31 - 1))
    s["camera_seed"] = int(rng.integers(0, 2**31 - 1))
    s["brain_seed"] = int(rng.integers(0, 2**31 - 1))
    return s
