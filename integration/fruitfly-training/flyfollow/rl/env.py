"""PursuitEnv (plan 4.4): box-level pursuit simulator with the shared filter and governor in the loop.

    box, st = env.reset(seed, kind, profile)      # kind "follow" | "approach"; profile "train" | "demo" | "stress"
    box, r, done, info = env.step(yaw, fb)        # raw controller sticks; the PersonGovernor is applied inside
    res = env.result()                            # EpisodeResult with per-term reward sums and plan 4.7 metrics

Common random numbers: every random quantity comes from its own stream
default_rng([seed, kind_id, profile_id, stream]) and is drawn at reset (parameters, start pose,
walker path, object, drone drift, detector noise and dropout schedule), so the same
(seed, kind, profile) gives the same world whatever the controller does. The only
controller-dependent part of the world is the direction of the person's single step toward the drone.

Obstacle scenarios (follow only, off by default): profile "obstacles" (or cfg obstacles.enabled) adds
static footprints around the user's path from their own stream, on top of the same world as the
profile named by its world_profile key (train), so obstacles on and off compare the same seed. An
avoidance layer mirroring the runtime stack (forward-corridor brake, then a lateral sidestep; never
yaw) sits between the governor and the drone; tall obstacles occlude the user's head.

Truth versus what the controller sees: reward and metrics use true range and bearing; the
controller and the governor only see the filtered, noisy, delayed box and Settings. Settings.fx/fy
are the true intrinsics times a calibration error of +-3 % (the plan calibrates at R0), and
Settings.video_latency_s is the true video latency times U(0.8, 1.2).
"""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import yaml

from flyfollow.interfaces import DT, KINDS, PROFILES, BoxState, EpisodeResult, Settings, configs_dir
from flyfollow.pilot.box_filter import BoxFilter
from flyfollow.pilot.governor import PersonGovernor
from flyfollow.rl.reward import RewardConfig, RewardTracker
from flyfollow.sim.camera import Camera, CameraParams, project, visible_box
from flyfollow.sim.drone_model import DroneModel, DroneParams
from flyfollow.sim.objects import ObjectTarget, sample_object
from flyfollow.sim.obstacles import Obstacle, place_obstacles
from flyfollow.sim.walker import Walker

# One uniform per key, always in this order, so a range change never shifts the other draws.
PARAM_KEYS = (
    "fwd_gain_mps", "yaw_gain_dps", "vz_gain_mps", "tau_fwd_s", "tau_yaw_s", "tau_z_s", "fwd_dead_s",
    "drift_sigma_mps", "pitch_scale", "max_fwd_stick", "z_ref_follow_m", "side_offset_deg", "z_min_person_m",
    "head_height_m", "head_size_m", "fx_px", "fy_over_fx", "fx_calib_err", "video_latency_s", "det_time_s",
    "latency_est_err", "det_rate_hz", "box_center_sigma_px", "box_h_noise_frac", "dropout_iid", "burst_rate_hz",
    "false_box_frac", "yaw_dead_s", "vz_dead_s", "max_back_stick",
)
S_PARAMS, S_START, S_WALKER, S_OBJECT, S_CAMERA, S_DRONE, S_OBSTACLES = range(7)
HEAD_SIZE_ASSUMED_M = 0.23


def load_env_config(path: str | Path | None = None) -> dict:
    with open(path or configs_dir() / "env.yaml") as f:
        return yaml.safe_load(f)


def profile_ranges(cfg: dict, profile: str) -> dict:
    """Train ranges with the profile's overrides on top."""
    prof = cfg["profiles"]
    out = dict(prof["train"])
    if profile != "train":
        out.update(prof.get(profile) or {})
    return out


def _draw(spec, u: float) -> float:
    if isinstance(spec, dict):
        vals = spec["choice"]
        return float(vals[min(int(u * len(vals)), len(vals) - 1)])
    if isinstance(spec, (list, tuple)):
        return float(spec[0]) + u * (float(spec[1]) - float(spec[0]))
    return float(spec)


def _profile_id(profile: str) -> int:
    return PROFILES.index(profile) if profile in PROFILES else 100 + zlib.crc32(profile.encode()) % 100_000


def _rng(seed: int, kind: str, profile: str, stream: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), KINDS.index(kind), _profile_id(profile), stream])


class PursuitEnv:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else load_env_config()
        self.rcfg = RewardConfig.from_dict(self.cfg.get("reward"))
        self.gov = PersonGovernor(self.cfg.get("governor"))
        self.reward = RewardTracker(self.rcfg)
        names = [p for p in PROFILES if p == "train" or p in self.cfg["profiles"]]
        names += [p for p in self.cfg["profiles"] if p not in names]
        self._profiles = {p: profile_ranges(self.cfg, p) for p in names}
        self._obs_on = False
        self.done = True

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, kind: str, profile: str = "train") -> tuple[BoxState, Settings]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if profile not in self._profiles:
            raise ValueError(f"unknown profile {profile!r}")
        cfg = self.cfg
        rp = self._profiles[profile]
        ep = cfg["episode"]
        self.seed, self.kind, self.profile = int(seed), kind, profile
        wp = str(rp.get("world_profile", profile))  # whose random world this profile reuses
        u = _rng(seed, kind, wp, S_PARAMS).random(len(PARAM_KEYS))
        v = {k: _draw(rp[k], float(u[i])) for i, k in enumerate(PARAM_KEYS)}
        self.params = v
        follow = kind == "follow"
        T = float(ep["follow_s"] if follow else ep["approach_s"])
        self.n_max = int(round(T / DT))
        pre = float(ep.get("preroll_s", 1.0))

        dp = DroneParams(fwd_gain_mps=v["fwd_gain_mps"], yaw_gain_dps=v["yaw_gain_dps"], vz_gain_mps=v["vz_gain_mps"],
                         tau_fwd_s=v["tau_fwd_s"], tau_yaw_s=v["tau_yaw_s"], tau_z_s=v["tau_z_s"],
                         fwd_dead_s=v["fwd_dead_s"], yaw_dead_s=v["yaw_dead_s"], vz_dead_s=v["vz_dead_s"],
                         drift_sigma_mps=v["drift_sigma_mps"], pitch_scale=v["pitch_scale"])
        noise = _rng(seed, kind, wp, S_DRONE).standard_normal(2 * (self.n_max + 2)).tolist()
        self.drone = DroneModel(dp, noise)

        fx = v["fx_px"]
        fy = fx * v["fy_over_fx"]
        video_lat = v["video_latency_s"]  # capture to decoded frame
        det_time = v["det_time_s"]
        lat = video_lat + det_time  # capture to box
        cc = cfg.get("camera", {})
        cp = CameraParams(fx=fx, fy=fy, rate_hz=v["det_rate_hz"], latency_s=lat, det_time_s=det_time,
                          center_sigma_px=v["box_center_sigma_px"], h_noise_frac=v["box_h_noise_frac"],
                          dropout=v["dropout_iid"], burst_rate_hz=v["burst_rate_hz"],
                          burst_len_s=tuple(cc.get("burst_len_s", (0.2, 1.5))), false_frac=v["false_box_frac"],
                          false_h_px=tuple(cc.get("false_h_px", (15.0, 150.0))),
                          h_full_det_px=float(cc.get("h_full_det_px", 20.0)))
        self.cam = Camera(cp, _rng(seed, kind, wp, S_CAMERA), -pre, T + 0.5)
        self.camp = cp

        us = _rng(seed, kind, wp, S_START).random(4).tolist()
        max_fwd = v["max_fwd_stick"]
        self.walker: Walker | None = None
        self.obj: ObjectTarget | None = None
        if follow:
            z_ref, z_min, offset = v["z_ref_follow_m"], v["z_min_person_m"], v["side_offset_deg"]
            size_assumed = HEAD_SIZE_ASSUMED_M
            r_lo = z_min + float(rp["follow_start_margin_m"])
            r0 = r_lo + us[0] * (float(rp["follow_start_range_max_m"]) - r_lo)
            b0 = math.radians((2.0 * us[1] - 1.0) * float(rp["follow_start_bearing_deg"]))
            tx, ty = r0 * math.cos(-b0), r0 * math.sin(-b0)
            wc = cfg["walker"]
            heading = math.atan2(ty, tx) + math.radians((2.0 * us[2] - 1.0) * wc.get("start_heading_jitter_deg", 45.0))
            v_max = v["fwd_gain_mps"] * max_fwd / 100.0
            self.v_cap = min(float(wc.get("speed_cap_mps", 1.4)), float(wc.get("speed_frac_of_vmax", 0.9)) * v_max)
            self.walker = Walker(_rng(seed, kind, wp, S_WALKER), tx, ty, heading, self.v_cap, self.n_max, wc,
                                 rp.get("walk_speed_frac", (0.0, 1.0)))
            self.tz, self.tsize = v["head_height_m"], v["head_size_m"]
            lo, hi = ep.get("follow_start_alt_offset_m", (-0.15, 0.25))
            z0 = self.tz + lo + us[3] * (hi - lo)
        else:
            self.obj = sample_object(_rng(seed, kind, wp, S_OBJECT), cfg["objects"], cfg["approach"],
                                     float(rp["object_bearing_deg"]))
            tx, ty = self.obj.x, self.obj.y
            z_ref, z_min, offset = self.obj.z_ref_m, float(rp["z_min_object_m"]), 0.0
            size_assumed = self.obj.prior_size_m
            self.tz, self.tsize = self.obj.center_h_m, self.obj.size_m
            self.v_cap = 0.0
            z0 = float(ep.get("approach_start_alt_m", 1.0))
        self.tx, self.ty = tx, ty
        self.drone.reset(0.0, 0.0, z0, 0.0)
        oc = cfg.get("obstacles") or {}
        self._obs_on = follow and bool(oc.get("enabled", False) or rp.get("obstacles", False))
        self.obstacles: list[Obstacle] = []
        if self._obs_on:
            self._reset_obstacles(oc, _rng(seed, kind, wp, S_OBSTACLES))
        calib = v["fx_calib_err"]
        self.st = Settings(kind=kind, z_ref_m=z_ref, target_size_m=size_assumed, side_offset_deg=offset, z_min_m=z_min,
                           max_fwd_stick=max_fwd, max_back_stick=v["max_back_stick"], max_yaw_stick=60.0, fx=fx * calib, fy=fy * calib,
                           video_latency_s=video_lat * v["latency_est_err"])
        self.filter = BoxFilter.from_config(cfg.get("filter"), self.st.video_latency_s)
        self.gov.reset()
        # Reward/metric standoff. "nominal" (plan 4.5): the band is around Z_ref. "achievable": for objects the band
        # is around Z_ref * true size / prior, where a size-based controller settles (the prior error is irreducible).
        self.z_ref_eval = z_ref
        if not follow and cfg["reward"].get("approach_ref", "nominal") == "achievable":
            self.z_ref_eval = z_ref * self.obj.size_m / self.obj.prior_size_m
        self.reward.reset(self.z_ref_eval, z_min, cp.half_hfov)
        self._offset = math.radians(offset)
        self._success_ticks = int(round(float(ep.get("success_hold_s", 2.0)) / DT))
        self._lost_after = self.rcfg.lost_after_s
        self._collide = self.rcfg.collide_person_m if follow else self.rcfg.collide_object_m

        # pre-roll: detector and filter run on the static start scene so the track exists at t = 0
        self._t_last_true = -pre
        self._g = self._geom()
        n_pre = int(round(pre / DT))
        for k in range(n_pre):
            t0, t1 = (k - n_pre) * DT, (k + 1 - n_pre) * DT
            self.cam.step(t0, t1, self._g, self._g)
            self.cam.deliver(t1, self._on_det)
        self.i = 0
        self.t = 0.0
        self._box = self.filter.output(0.0)
        self.done = False
        self.ret = 0.0
        self._prev_yaw = self._prev_fb = 0
        self._band_run = 0
        self._prev_safety = False
        self.collided = self.landed = self.succeeded = False
        self.t_success = math.nan
        # metric accumulators
        self._n_band = self._n_view = self._n_saf_on = self._n_saf = self._n_clamp = 0
        self._sd = self._sd2 = self._sb2 = self._syj = 0.0
        self._min_dist = math.inf
        return self._box, self.st

    def _reset_obstacles(self, oc: dict, rng: np.random.Generator) -> None:
        obs, xs, ys = place_obstacles(rng, self.walker.xs, self.walker.ys, (0.0, 0.0), oc, DT)
        self.walker.xs, self.walker.ys = xs, ys
        self.obstacles = obs
        self._avoid_on = bool(oc.get("avoid", True))
        self._occl_on = bool(oc.get("occlusion", True))
        self._vmargin = float(oc.get("vertical_margin_m", 0.3))
        self._drone_r = float(oc.get("drone_radius_m", 0.12))
        self._corr_len = float(oc.get("corridor_len_m", 1.0))
        self._corr_hw = float(oc.get("corridor_half_width_m", 0.25))
        self._ss_after = float(oc.get("sidestep_after_s", 1.5))
        self._ss_m = float(oc.get("sidestep_m", 0.5))
        self._ss_stick = float(oc.get("sidestep_stick", 40.0))
        self._ss_timeout = float(oc.get("sidestep_timeout_s", 3.0))
        self._ss_cooldown = float(oc.get("sidestep_cooldown_s", 1.0))
        self._blocked_t = self._blocked_s = self._cool = 0.0
        self._ss_on = False
        self._ss_dir = 0
        self._ss_done = self._ss_t = 0.0
        self._sidesteps = self._ss_vetoed = self._n_occl = 0
        self._min_obs = math.inf
        self.obstacle_collision = False
        self.braking = self.occluded = False

    def _avoidance(self, lr: int, fb: int) -> tuple[int, int]:
        """Runtime avoidance stack (teammate's): brake forward when the corridor ahead is blocked; after
        sidestep_after_s blocked, sidestep sidestep_m away from the obstacle, unless that brings the drone
        closer than z_min to the user's box-estimated position. Never touches yaw."""
        d = self.drone
        c, s = math.cos(d.psi), math.sin(d.psi)
        blocked = False
        left_near = 0.0
        if self._avoid_on:
            half = 0.5 * self._corr_len
            cx, cy = d.x + c * half, d.y + s * half
            zc = d.z - self._vmargin
            dmin = math.inf
            for o in self.obstacles:
                if o.height > zc and o.overlaps_rect(cx, cy, c, s, half, self._corr_hw):
                    blocked = True
                    dd = o.dist(d.x, d.y)
                    if dd < dmin:
                        dmin = dd
                        left_near = -s * (o.x - d.x) + c * (o.y - d.y)
        self.braking = blocked and fb > 0
        if self.braking:
            fb = 0
            self._blocked_t += DT
            self._blocked_s += DT
        else:
            self._blocked_t = 0.0
        if self._cool > 0.0:
            self._cool -= DT
        if self._ss_on:
            self._ss_done += (s * d.vx - c * d.vy) * self._ss_dir * DT  # body-right speed (runtime: vgy)
            self._ss_t += DT
            if self._ss_done >= self._ss_m or self._ss_t >= self._ss_timeout:
                self._ss_on = False
                self._cool = self._ss_cooldown
            else:
                lr = int(self._ss_dir * self._ss_stick)
        elif self.braking and self._blocked_t >= self._ss_after and self._cool <= 0.0:
            direction = 1 if left_near > 0.0 else -1  # obstacle on the left -> move right (lr > 0)
            self._blocked_t = 0.0
            if self._sidestep_safe(direction):
                self._ss_on = True
                self._ss_dir = direction
                self._ss_done = self._ss_t = 0.0
                self._sidesteps += 1
                lr = int(direction * self._ss_stick)
            else:
                self._ss_vetoed += 1
        return lr, fb

    def _sidestep_safe(self, direction: int) -> bool:
        """False if the sidestep would end closer than z_min to the user, judged from the box as at runtime."""
        box, st = self._box, self.st
        if not box.valid or box.h <= 0.0:
            return True
        z_est = st.fy * st.target_size_m / box.h
        b = math.atan((box.cx - st.cx0) / st.fx)
        ux, uy = z_est * math.cos(b), -z_est * math.sin(b)  # drone frame, y left
        return math.hypot(ux, uy + direction * self._ss_m) >= st.z_min_m

    def _occluded_now(self) -> bool:
        """True if a footprint rises above the camera-to-head line of sight where the line crosses it."""
        d = self.drone
        z0, z1 = d.z, self.tz
        zlow = z0 if z0 < z1 else z1
        for o in self.obstacles:
            if o.height <= zlow:
                continue
            iv = o.segment_interval(d.x, d.y, self.tx, self.ty)
            if iv is None:
                continue
            h_in, h_out = z0 + iv[0] * (z1 - z0), z0 + iv[1] * (z1 - z0)
            if o.height > (h_in if h_in < h_out else h_out):
                return True
        return False

    def _obstacle_contact(self) -> bool:
        d = self.drone
        zc = d.z - self._vmargin
        hit = False
        for o in self.obstacles:
            if o.height > zc:
                dd = o.dist(d.x, d.y)
                if dd < self._min_obs:
                    self._min_obs = dd
                if dd < self._drone_r and d.z < o.height + 0.05:
                    hit = True
        return hit

    def _on_det(self, t: float, t_dec: float, cx: float, cy: float, h: float, true_target: bool) -> None:
        self.filter.update(t, t_dec, cx, cy, h)
        if true_target:
            self._t_last_true = t

    def _geom(self):
        d = self.drone
        rx, ry = self.tx - d.x, self.ty - d.y
        c, s = math.cos(d.psi), math.sin(d.psi)
        self._fwd = fwd = c * rx + s * ry
        self._left = left = -s * rx + c * ry
        self._rx, self._ry = rx, ry
        p = self.camp
        return project(fwd, left, self.tz - d.z, self.tsize, d.pitch, p.fx, p.fy, p.cx0, p.cy0)

    # ------------------------------------------------------------------ step
    def step(self, yaw_stick: float, fb_stick: float, brain_age_s: float = 0.0) -> tuple[BoxState, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() after the episode ended; call reset()")
        d = self.drone
        st = self.st
        go = self.gov.filter(yaw_stick, fb_stick, self._box, st, DT, alt_m=round(d.z * 10.0) * 0.1,
                             brain_age_s=brain_age_s)
        self.i += 1
        i = self.i
        t = i * DT
        self.t = t
        if go.land:
            # Landing ends the flight: charge the remaining ticks like a collision tail (no -200), so landing never pays.
            r = self.reward.tail(self.n_max - i + 1, DT)
            self.ret += r
            self.landed = self.done = True
            self._box = BoxState(False)
            return self._box, r, True, {"t": t, "land": True, "collision": False, "success": False}
        lr, fb_sent = go.lr, go.fb
        obs_on = self._obs_on
        if obs_on:
            lr, fb_sent = self._avoidance(lr, fb_sent)
        d.step(lr, fb_sent, go.ud, go.yaw)
        if self.walker is not None:
            self.tx, self.ty = self.walker.position(i, d.x, d.y)
        g = self._geom()
        occ = False
        if obs_on:
            occ = self._occl_on and self._occluded_now()
            self.occluded = occ
            if occ:
                self._n_occl += 1
        self.cam.step(t - DT, t, self._g, g, occ)
        self.cam.deliver(t, self._on_det)
        self._g = g
        box = self.filter.output(t)
        self._box = box

        # truth
        z = math.hypot(self._rx, self._ry)
        bearing = math.atan2(-self._left, self._fwd)
        berr = bearing - self._offset
        if berr > math.pi:
            berr -= 2.0 * math.pi
        elif berr < -math.pi:
            berr += 2.0 * math.pi
        in_view = visible_box(g) is not None and not occ
        lost = (t - self._t_last_true) > self._lost_after
        dyaw = go.yaw - self._prev_yaw
        dfb = go.fb - self._prev_fb
        self._prev_yaw, self._prev_fb = go.yaw, go.fb
        rt = self.reward
        r = rt.tick(DT, z, berr, dyaw, dfb, go.safety, lost)

        dz = z - self.z_ref_eval
        in_band = abs(dz) <= rt.band
        if in_band and in_view:
            self._n_band += 1
        if in_view:
            self._n_view += 1
        self._sd += dz
        self._sd2 += dz * dz
        bd = math.degrees(berr)
        self._sb2 += bd * bd
        self._syj += dyaw * dyaw * 0.002
        if z < self._min_dist:
            self._min_dist = z
        if go.safety:
            self._n_saf += 1
            if not self._prev_safety:
                self._n_saf_on += 1
        self._prev_safety = go.safety
        if go.clamped:
            self._n_clamp += 1

        done = False
        success = collision = False
        if self.walker is not None:
            collision = z < self._collide
            if obs_on and self._obstacle_contact():
                collision = self.obstacle_collision = True
        else:
            collision = math.sqrt(z * z + (self.tz - d.z) ** 2) < self._collide
        if collision:
            r += rt.collision(self.n_max - i, DT)
            self.collided = done = True
        elif self.walker is None:
            self._band_run = self._band_run + 1 if (in_band and in_view) else 0
            if self._band_run >= self._success_ticks:
                r += rt.success()
                self.succeeded = done = True
                self.t_success = t
        if i >= self.n_max:
            done = True
        self.done = done
        self.ret += r
        info = {"t": t, "z": z, "bearing": bearing, "berr": berr, "in_view": in_view, "lost": lost,
                "lr": lr, "fb": fb_sent, "ud": go.ud, "yaw": go.yaw, "occluded": occ, "safety": go.safety, "clamped": go.clamped,
                "reasons": go.reasons, "collision": collision, "success": success or self.succeeded, "land": False}
        return box, r, done, info

    # ------------------------------------------------------------------ results
    def world(self) -> dict:
        """True episode parameters and the pre-sampled world, for tests and analysis."""
        w = {"params": dict(self.params), "settings": self.st, "target_xy0": None, "v_cap": self.v_cap,
             "tz": self.tz, "tsize": self.tsize}
        if self.walker is not None:
            w["walker_xs"], w["walker_ys"] = list(self.walker.xs), list(self.walker.ys)
        if self.obj is not None:
            w["object"] = self.obj
        if self._obs_on:
            w["obstacles"] = [o.to_dict() for o in self.obstacles]
        w["camera_t"] = list(self.cam._t[:50])
        w["camera_drop"] = list(self.cam._drop)
        return w

    def metrics(self) -> dict[str, float]:
        n = max(1, self.i)
        mins = n * DT / 60.0
        z_ref = self.z_ref_eval
        m = {
            # follow counts the ticks a collision or landing cut off as out of band
            "frac_in_band": self._n_band / (self.n_max if self.kind == "follow" else n),
            "frac_in_view": self._n_view / n,
            "mean_signed_range_err": self._sd / n,
            "rms_range_err": math.sqrt(self._sd2 / n),
            "rms_bearing_err_deg": math.sqrt(self._sb2 / n),
            "loss_events_per_min": self.reward.n_loss / mins,
            "min_dist": self._min_dist,
            "safety_interventions_per_min": self._n_saf_on / mins,
            "frac_safety": self._n_saf / n,
            "frac_clamped": self._n_clamp / n,
            "yaw_jerk": self._syj / n,
            "too_close_events": float(self.reward.n_close),
            "collided": float(self.collided),
            "landed": float(self.landed),
            "cond_walk_cap_mps": self.v_cap,
            "cond_latency_s": self.camp.latency_s,
            "cond_fx": self.camp.fx,
            "cond_fwd_gain_mps": self.params["fwd_gain_mps"],
            "cond_z_ref_m": z_ref,
        }
        if self.kind == "approach":
            m["success"] = float(self.succeeded)
            m["time_to_standoff"] = (self.t_success - self._success_ticks * DT) if self.succeeded else math.nan
            m["overshoot"] = self._min_dist - z_ref
            m["floor_object"] = float(self.obj.floor) if self.obj is not None else 0.0
        if self._obs_on:
            m["obstacle_collision"] = float(self.obstacle_collision)
            # horizontal clearance from the drone center to the nearest footprint that reaches flight height
            m["min_obstacle_dist"] = self._min_obs if self._min_obs < math.inf else math.nan
            m["blocked_s"] = self._blocked_s
            m["sidesteps"] = float(self._sidesteps)
            m["sidesteps_vetoed"] = float(self._ss_vetoed)
            m["frac_occluded"] = self._n_occl / n
            m["n_obstacles"] = float(len(self.obstacles))
        return m

    def result(self) -> EpisodeResult:
        return EpisodeResult(seed=self.seed, kind=self.kind, profile=self.profile, ret=self.ret,
                             terms=self.reward.terms(), metrics=self.metrics(), n_ticks=self.i)
