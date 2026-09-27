"""R0 video latency (plan 3.3, 4.8 R0 "stopwatch-on-screen test"), measured with no human reading.

    python -m flyfollow.tools.latency_test                    # Tello on a table looking at this laptop's screen
    python -m flyfollow.tools.latency_test --analyze data/r0/latency_log_<stamp>.npz
    python -m flyfollow.tools.latency_test --display-only     # just the flashing screen (for a phone-camera check)

A child process shows a full-screen field that flips black/white at random 0.35 to 0.9 s intervals (with a large
millisecond counter at the top) and prints each flip time. This process decodes the Tello video with the same PyAV
reader as tello_io and keeps a small luminance image per frame with its t_decoded. The analysis picks the pixels
that flash (largest temporal spread), finds the global lag by matching the black/white sequences, then takes every
video edge minus its display flip: median and p90 = capture-to-decode latency, stored as video_latency_s.
The drone does not fly. cv2 (display) and PyAV (decode) run in different processes on purpose (duplicate FFmpeg
dylibs on macOS). The result slightly overestimates (display refresh up to 17 ms at 60 Hz and up to one video frame,
33 ms, of sampling; half of that frame on average is removed from the reported value, see "frame_correction_s").
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from flyfollow.tools.stick_response import r0_dir

DS = 8  # luminance downsample factor (960x720 -> 120x90)


# ------------------------------------------------------------------------------------------------ display child
def display(duration_s: float, seed: int, windowed: bool = False) -> None:
    import cv2

    rng = np.random.default_rng(seed)
    win = "flyfollow latency (q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not windowed:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    H, W = 900, 1440
    level, t_next = 0, time.time() + 1.0
    t_end = time.time() + duration_s
    while time.time() < t_end:
        now = time.time()
        flipped = False
        if now >= t_next:
            level ^= 1
            flipped = True
            t_next = now + rng.uniform(0.35, 0.9)
        img = np.full((H, W, 3), 255 if level else 0, np.uint8)
        img[: H // 6] = 128
        ms = int(now * 1000) % 100000
        cv2.putText(img, f"{ms // 1000:02d}.{ms % 1000:03d}", (W // 2 - 260, H // 8), cv2.FONT_HERSHEY_SIMPLEX, 3.2,
                    (0, 0, 255), 8, cv2.LINE_AA)
        cv2.imshow(win, img)
        k = cv2.waitKey(1) & 0xFF
        if flipped:
            print(f"FLIP {time.time():.6f} {level}", flush=True)
        if k in (27, ord("q")):
            break
    cv2.destroyAllWindows()


# ------------------------------------------------------------------------------------------------ analysis
def analyze(t: np.ndarray, lum: np.ndarray, flips: np.ndarray, max_lag_s: float = 2.0) -> dict:
    """t (n,) decode times, lum (n, h, w) luminance, flips (m, 2) of (time, level 0|1). Returns latency stats."""
    t = np.asarray(t, float)
    L = np.asarray(lum, np.float32)
    flips = np.asarray(flips, float).reshape(-1, 2)
    if len(t) < 30 or len(flips) < 4:
        return {"ok": False, "error": f"too little data ({len(t)} frames, {len(flips)} flips)"}
    sd = L.std(axis=0)
    roi = sd >= 0.6 * np.percentile(sd, 99.5)
    if roi.sum() < 10 or np.percentile(sd, 99.5) < 5.0:
        return {"ok": False, "error": "no flashing region in view (point the camera at the screen)"}
    s = L[:, roi].mean(axis=1)
    lo, hi = np.percentile(s, 5), np.percentile(s, 95)
    mid, band = (lo + hi) / 2, 0.15 * (hi - lo)
    b = np.zeros(len(s), bool)
    state = s[0] > mid
    for i, v in enumerate(s):  # hysteresis
        if state and v < mid - band:
            state = False
        elif not state and v > mid + band:
            state = True
        b[i] = state
    edges = [(t[i], int(b[i])) for i in range(1, len(b)) if b[i] != b[i - 1]]
    ft, fl = flips[:, 0], flips[:, 1].astype(int)

    def disp_level(tq: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(ft, tq, side="right") - 1
        first = 1 - fl[0]  # level before the first flip
        return np.where(idx >= 0, fl[np.clip(idx, 0, None)], first)

    lags = np.arange(0.0, max_lag_s, 0.005)
    inside = (t > ft[0] + max_lag_s) & (t < ft[-1])
    if inside.sum() < 20:
        inside = t > ft[0]
    agree = np.array([np.mean(disp_level(t[inside] - lg) == b[inside]) for lg in lags])
    k = int(np.argmax(agree))
    lag = float(lags[k])
    samples = []
    for tv, lv in edges:
        cand = ft[(fl == lv) & (tv - ft >= lag - 0.2) & (tv - ft <= lag + 0.2)]
        if cand.size:
            samples.append(float(tv - cand[np.argmin(np.abs(tv - cand - lag))]))
    samples = np.array(samples)
    dt_frame = float(np.median(np.diff(t)))
    if samples.size < 3:
        return {"ok": False, "error": f"only {samples.size} matched edges", "global_lag_s": lag, "agreement": float(agree[k])}
    corr = 0.5 * dt_frame
    res = {
        "ok": bool(agree[k] > 0.9 and samples.size >= 5),
        "video_latency_s": round(float(np.median(samples)) - corr, 3),
        "p90_s": round(float(np.percentile(samples, 90)) - corr, 3),
        "mean_s": round(float(samples.mean()) - corr, 3),
        "min_s": round(float(samples.min()) - corr, 3), "max_s": round(float(samples.max()) - corr, 3),
        "raw_median_s": round(float(np.median(samples)), 3), "frame_correction_s": round(corr, 4),
        "n_edges": int(samples.size), "n_flips": int(len(flips)), "global_lag_s": round(lag, 3),
        "agreement": round(float(agree[k]), 3), "fps": round(1.0 / dt_frame, 1), "roi_px": int(roi.sum()),
        "samples_s": [round(x, 4) for x in samples.tolist()],
    }
    if not res["ok"]:
        res["warning"] = "agreement < 0.9 or < 5 edges: check that the flashing area fills a good part of the view"
    return res


def report(res: dict) -> str:
    if "video_latency_s" not in res:
        return f"FAILED: {res.get('error')}"
    return (f"video latency (capture -> decoded): median {res['video_latency_s'] * 1000:.0f} ms, p90 {res['p90_s'] * 1000:.0f} ms, "
            f"range {res['min_s'] * 1000:.0f} to {res['max_s'] * 1000:.0f} ms, n {res['n_edges']} edges, fps {res['fps']}, "
            f"sequence agreement {res['agreement']:.2f}" + (f"\nWARNING: {res['warning']}" if res.get("warning") else ""))


# ------------------------------------------------------------------------------------------------ live run
def run(a) -> Path:
    import logging

    from djitellopy import Tello

    from flyfollow.runtime.tello_io import (
        CONTROL_PORT,
        FIREWALL_HINT,
        STATE_PORT,
        VIDEO_PORT,
        TelloLink,
        VideoReader,
        check_udp_port,
        port_hint,
    )

    logging.getLogger("djitellopy").setLevel(logging.WARNING)
    for port in (CONTROL_PORT, STATE_PORT, VIDEO_PORT):
        if check_udp_port(port):
            sys.exit(port_hint(port))
    print(f"""
R0 VIDEO LATENCY ({a.duration:.0f} s, the drone does not fly):
 1. Laptop on the TELLO-XXXXXX Wi-Fi, screen brightness at maximum, no tello_io running.
 2. Tello on a table (or held still) 30 to 60 cm from the screen, camera looking at the middle of the screen; the
    flashing field should fill most of the Tello's view. Keep the room light steady.
 3. A full-screen window flashes black/white for {a.duration:.0f} s (q quits early). Do not cover the screen.
""")
    tello = Tello(host=a.ip)
    link = TelloLink(tello)
    try:
        link.connect()
    except Exception as e:
        sys.exit(f"connect failed: {e}\n{FIREWALL_HINT}")
    print("battery", link.state().get("bat"), "% | streamon:", link.cmd("streamon", 7))
    ts: list[float] = []
    lums: list[np.ndarray] = []

    def on_frame(img, t_dec):
        ts.append(t_dec)
        lums.append(img[::DS, ::DS].mean(axis=2).astype(np.uint8))

    reader = VideoReader(on_frame, warmup_s=2.0)
    reader.start()
    t0 = time.time()
    while not ts and time.time() - t0 < 15.0:
        time.sleep(0.1)
    if not ts:
        reader.stop()
        sys.exit(f"no video frames after 15 s\n{FIREWALL_HINT}\n{port_hint(VIDEO_PORT)}")
    child = subprocess.Popen([sys.executable, "-m", "flyfollow.tools.latency_test", "--display", "--duration", str(a.duration),
                              "--seed", str(a.seed)] + (["--windowed"] if a.windowed else []), stdout=subprocess.PIPE, text=True)
    flips: list[tuple[float, int]] = []

    def read_child():
        for line in child.stdout:
            p = line.split()
            if len(p) == 3 and p[0] == "FLIP":
                flips.append((float(p[1]), int(p[2])))

    th = threading.Thread(target=read_child, daemon=True)
    th.start()
    child.wait()
    th.join(timeout=1.0)
    time.sleep(1.5)  # frames still in flight
    reader.stop()
    try:
        link.cmd("streamoff", 2)
    except Exception:
        pass
    stamp = time.strftime("%Y%m%d_%H%M%S")
    p = r0_dir() / f"latency_log_{stamp}.npz"
    np.savez_compressed(p, t=np.array(ts), lum=np.stack(lums), flips=np.array(flips, float).reshape(-1, 2))
    print(f"log: {p} ({len(ts)} frames, {len(flips)} flips)")
    return p


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="R0 video latency: flashing screen seen by the Tello camera")
    ap.add_argument("--analyze", metavar="NPZ", help="re-analyze a saved latency_log_*.npz")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ip", default="192.168.10.1")
    ap.add_argument("--windowed", action="store_true", help="do not go full screen")
    ap.add_argument("--display", action="store_true", help=argparse.SUPPRESS)  # child process mode
    ap.add_argument("--display-only", action="store_true", help="only show the flashing screen")
    a = ap.parse_args(argv)
    if a.display or a.display_only:
        display(a.duration, a.seed, a.windowed)
        return
    src = Path(a.analyze) if a.analyze else run(a)
    z = np.load(src)
    res = analyze(z["t"], z["lum"], z["flips"])
    res["source"] = str(src)
    print(report(res))
    if "video_latency_s" in res:
        out = r0_dir() / f"latency_{time.strftime('%Y%m%d_%H%M%S')}.json"
        out.write_text(json.dumps(res, indent=1))
        print(f"results: {out}\nnext: python -m flyfollow.tools.update_sim_from_r0 (sets video_latency_s in profiles.demo)"
              f"\n      and send settings video_latency_s={res['video_latency_s']} to the runtime")


if __name__ == "__main__":
    main()
