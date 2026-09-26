"""One episode of any Controller in PursuitEnv (plan 4.4 to 4.7)."""

from __future__ import annotations

import time

from flyfollow.interfaces import DT, Controller, EpisodeResult
from flyfollow.rl.env import PursuitEnv

TRACE_KEYS = ("t", "z", "bearing_deg", "berr_deg", "in_view", "valid", "box_cx", "box_cy", "box_h", "yaw_raw", "fb_raw",
              "lr", "fb", "ud", "yaw", "safety", "clamped", "lost", "r", "x", "y", "alt", "psi_deg", "tx", "ty",
              "r_e_z", "r_e_x", "r_jerk", "r_safety", "r_lost", "r_too_close", "r_events", "occluded")


def run_episode(env: PursuitEnv, controller: Controller, seed: int, kind: str, profile: str = "train",
                record: bool = False, warmup_s: float = 1.0) -> EpisodeResult:
    """reset -> controller.reset -> controller.warmup -> act/step until done. brain_s is the time spent in act()."""
    t_start = time.perf_counter()
    box, st = env.reset(seed, kind, profile)
    controller.reset(st, seed)
    controller.warmup(warmup_s)
    act = controller.act
    step = env.step
    t_act = 0.0
    trace: dict[str, list] | None = {k: [] for k in TRACE_KEYS} if record else None
    pc = time.perf_counter
    done = False
    if trace is None:
        while not done:
            a = pc()
            yaw, fb = act(box, st, DT)
            t_act += pc() - a
            box, _, done, _ = step(yaw, fb)
    else:
        rt = env.reward
        prev = [0.0] * 7
        while not done:
            a = pc()
            yaw, fb = act(box, st, DT)
            t_act += pc() - a
            box, r, done, info = step(yaw, fb)
            if info.get("land"):
                break
            _append(trace, env, box, info, yaw, fb, r, rt, prev)
    res = env.result()
    res.wall_s = time.perf_counter() - t_start
    res.brain_s = t_act
    res.trace = trace
    return res


def _append(tr: dict, env: PursuitEnv, box, info: dict, yaw_raw: float, fb_raw: float, r: float, rt, prev: list) -> None:
    import math

    d = env.drone
    cur = [rt.s_ez, rt.s_ex, rt.s_j, rt.s_g, rt.s_l, rt.s_c, rt.s_evl + rt.s_evc + rt.s_col + rt.s_tail + rt.s_succ]
    diff = [c - p for c, p in zip(cur, prev)]
    prev[:] = cur
    vals = (info["t"], info["z"], math.degrees(info["bearing"]), math.degrees(info["berr"]), info["in_view"], box.valid,
            box.cx, box.cy, box.h, yaw_raw, fb_raw, info["lr"], info["fb"], info["ud"], info["yaw"], info["safety"],
            info["clamped"], info["lost"], r, d.x, d.y, d.z, math.degrees(d.psi), env.tx, env.ty, *diff, info.get("occluded", False))
    for k, val in zip(TRACE_KEYS, vals):
        tr[k].append(val)
