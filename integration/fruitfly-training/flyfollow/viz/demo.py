"""Demo: a simulated pursuit episode driven by the fly brain, streamed live to Rerun and/or recorded as MP4.

    python -m flyfollow.viz.demo --brain data/brains/pursuit_core1.npz --seed 1000 --kind follow \
        [--params runs/<run>/best.json | none] [--record runs/viz/demo.mp4] [--seconds 30] [--no-live]

The episode is A's PursuitEnv with C's FLY controller (FLY-HAND hand calibration by default; --params
takes a trainer best.json with "x" or "params"). Frames go to a separate viewer process through
VizSink at 20 Hz in real time, so the loop never waits on rendering. With --record the frames are kept
and, after the episode, rendered into a 1920x1080 30 fps composite (fly body, brain, drone camera, top
view, fly's-eye view, DN and stick traces). --synthetic swaps the simulator for a scripted spot driving
the subgraph LIF directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from flyfollow.interfaces import DT


def load_params(path: str | None, arm: str, brain: str) -> tuple[np.ndarray | None, str]:
    """(normalized x or None for hand calibration, arm). A best.json may override the arm."""
    if not path or path.lower() == "none":
        return None, arm
    rec = json.loads(Path(path).read_text())
    arm = rec.get("arm", arm) or arm
    if rec.get("x") is not None:
        return np.asarray(rec["x"], np.float64), arm
    if rec.get("params"):
        from flyfollow.rl.params import param_space

        return param_space(arm).encode(rec["params"]), arm
    raise ValueError(f"{path}: expected 'x' or 'params'")


def sim_frames(brain: str, seed: int, kind: str = "follow", profile: str = "demo", arm: str = "FLY-HAND",
               x: np.ndarray | None = None, seconds: float = 30.0, warmup_s: float = 1.0) -> Iterator[dict]:
    """One PursuitEnv episode with the fly controller, as viz frames (one per 50 ms tick)."""
    from flyfollow.rl.controllers import make_controller
    from flyfollow.rl.env import PursuitEnv
    from flyfollow.viz.frames import frame_from_controller

    env = PursuitEnv()
    ctrl = make_controller(arm, x=x, brain_path=brain)
    box, st = env.reset(seed, kind, profile)
    ctrl.reset(st, seed)
    ctrl.warmup(warmup_s)
    n_max = min(env.n_max, round(seconds / DT))
    hfov = 2 * math.degrees(env.camp.half_hfov)
    done, i = False, 0
    while not done and i < n_max:
        seen = box  # the box the controller acts on this tick
        yaw, fb = ctrl.act(seen, st, DT)
        box, _, done, info = env.step(yaw, fb)
        i += 1
        if info.get("land"):
            break
        target = {"valid": bool(seen.valid), "cx": float(seen.cx), "cy": float(seen.cy), "h": float(seen.h),
                  "bearing_deg": math.degrees(info["bearing"]), "range_m": float(info["z"]),
                  "in_view": bool(info["in_view"]), "z_ref_m": float(env.z_ref_eval)}
        sim = {"drone_xy": (float(env.drone.x), float(env.drone.y)), "drone_psi_deg": math.degrees(env.drone.psi),
               "person_xy": (float(env.tx), float(env.ty)), "hfov_deg": hfov}
        yield frame_from_controller(ctrl, t=info["t"], tick=i, sticks=(info["yaw"], info["fb"]), target=target, sim=sim,
                                    meta={"seed": seed, "kind": kind, "profile": profile, "source": "sim", "arm": arm})


def record_mp4(frames: list[dict], brain: str, out: str, fps: int = 30, verbose: bool = True) -> dict:
    """Render the composite video from recorded frames (offline; about 15 frames/s on an M-series Mac)."""
    import imageio.v2 as imageio

    from flyfollow.viz.brain_view import BrainGeometry
    from flyfollow.viz.composite import Compositor

    geo = BrainGeometry.load(brain)
    comp = Compositor(geo)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8)
    t_end = frames[-1]["t"] if frames else 0.0
    n_video = int(t_end * fps)
    k = 0
    t0 = time.perf_counter()
    ms = []
    stills = {}
    for j in range(n_video):
        tv = j / fps
        while k < len(frames) and frames[k]["t"] <= tv + 1e-9:
            comp.feed(frames[k], DT)
            k += 1
        a = time.perf_counter()
        img = comp.render(1.0 / fps)
        ms.append(1000 * (time.perf_counter() - a))
        w.append_data(img)
        if j in (int(0.3 * n_video), int(0.6 * n_video)):
            stills[j] = img
        if verbose and j % (fps * 5) == 0:
            print(f"  video {tv:5.1f} / {t_end:.1f} s  ({np.mean(ms[-fps:]):.0f} ms per frame)", flush=True)
    w.close()
    stats = {"frames": n_video, "seconds": time.perf_counter() - t0, "ms_per_frame": float(np.mean(ms)) if ms else 0.0,
             "body_render_ms": comp.body.render_ms, "stills": stills}
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fly brain -> drone demo: live Rerun view and/or MP4 composite")
    ap.add_argument("--brain", default="data/brains/pursuit_core1.npz")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--kind", default="follow", choices=("follow", "approach"))
    ap.add_argument("--profile", default="demo")
    ap.add_argument("--arm", default="FLY-YAW-HAND", help="FLY-YAW-HAND (fly steers, PID sets speed; the demo controller) or any arm")
    ap.add_argument("--params", default=None, help="trainer best.json (x or params); 'none' or omitted: hand calibration")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--record", default=None, help="write a 1920x1080 MP4 composite here")
    ap.add_argument("--stills", default=None, help="folder for two PNG stills of the composite (with --record)")
    ap.add_argument("--no-live", action="store_true", help="do not start the Rerun viewer")
    ap.add_argument("--save-rrd", default=None, help="viewer writes this .rrd instead of opening a window")
    ap.add_argument("--speed", type=float, default=1.0, help="live playback speed (1 = real time)")
    ap.add_argument("--address", default=None)
    ap.add_argument("--synthetic", action="store_true", help="scripted spot driving the LIF directly (no simulator)")
    ap.add_argument("--script", default="sweep", choices=("sweep", "left_right", "hold"))
    a = ap.parse_args(argv)
    from flyfollow.brain.build import ensure_flydrones

    ensure_flydrones()
    from flyfollow.viz.live import DEFAULT_ADDRESS, VizSink, spawn_viewer

    if a.synthetic:
        from flyfollow.viz.frames import synthetic_frames

        gen = synthetic_frames(a.brain, seconds=a.seconds, seed=a.seed, script=a.script)
        label = f"synthetic {a.script}"
    else:
        x, arm = load_params(a.params, a.arm, a.brain)
        gen = sim_frames(a.brain, a.seed, a.kind, a.profile, arm, x, a.seconds)
        label = f"{arm} {a.kind} seed {a.seed} ({'trained params' if x is not None else 'hand calibration'})"
    live = not a.no_live
    sink = viewer = None
    if live:
        addr = a.address or DEFAULT_ADDRESS
        viewer = spawn_viewer(a.brain, addr, save=a.save_rrd)
        sink = VizSink(addr)
    print(f"running {label} for up to {a.seconds:.0f} s" + (" (live)" if live else ""), flush=True)
    frames: list[dict] = []
    t_start = time.perf_counter()
    loop_ms = []
    for fr in gen:
        if sink is not None:
            a0 = time.perf_counter()
            sink.publish(fr)
            loop_ms.append(1000 * (time.perf_counter() - a0))
            lag = fr["t"] / a.speed - (time.perf_counter() - t_start)
            if lag > 0:
                time.sleep(lag)
        if a.record:
            frames.append(fr)
    wall = time.perf_counter() - t_start
    n = len(frames) if a.record else (sink.sent if sink else 0)
    print(f"episode done: {n} ticks in {wall:.1f} s wall", flush=True)
    if sink is not None:
        s = sink.stats()
        print(f"sink: sent {s['sent']}, publish {s['publish_ms_mean']:.3f} ms mean, {s['publish_ms_max']:.2f} ms max "
              f"(never blocks; the viewer drops what it cannot draw)", flush=True)
        sink.close()
    if a.record and frames:
        print(f"recording {a.record} ...", flush=True)
        st = record_mp4(frames, a.brain, a.record)
        print(f"wrote {a.record}: {st['frames']} frames in {st['seconds']:.0f} s ({st['ms_per_frame']:.0f} ms per composite frame)")
        if a.stills:
            import imageio.v2 as imageio

            Path(a.stills).mkdir(parents=True, exist_ok=True)
            for j, img in st["stills"].items():
                imageio.imwrite(Path(a.stills) / f"composite_{j:04d}.png", img)
    if viewer is not None:
        try:
            viewer.wait(timeout=10)
        except Exception:  # noqa: BLE001
            viewer.terminate()


if __name__ == "__main__":
    sys.exit(main())
