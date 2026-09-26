#!/usr/bin/env python3
"""
Tello response-lag test: one flight, djitellopy, everything timestamped.

Measures (each reported as mean ± std [min … max] over repeats):
  command delay   round trip of a read command ('battery?'), 10 repeats on the ground
  takeoff / land  time until the drone answers 'ok'
  state / video   packet rate, frame rate, jitter, worst gap
  yaw, throttle, pitch, roll   (rc steps in both directions, --trials repeats each)
      lag          command -> first reaction in the telemetry (yaw/pitch/roll angle, vertical speed)
      speed-up     first reaction -> 90 % of peak speed
      stop         rc 0 -> speed back under 10 % of peak
      video        command -> first video frame that shows motion
      video delay  video - lag   (how much later the camera shows what the drone already did)
  forward/back N cm   discrete moves: time to 'ok', lag, speed-up, video

Install:  pip install djitellopy opencv-python numpy
Run:      python tools/tello_latency_test.py              # full flight (~2 min)
          python tools/tello_latency_test.py --ground     # no takeoff: command delay + stream rates
          python tools/tello_latency_test.py --analyze results/tello_latency_<stamp>.json
Safety:   3 x 3 m clear space, prop guards, battery >= 30 %. Ctrl+C lands.

The state stream is ~10 Hz, so one measurement is quantised to ~100 ms; the mean over repeats is better
than that, and the std includes the quantisation.
"""

import argparse
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

FIELDS = ["pitch", "roll", "yaw", "vgx", "vgy", "vgz", "tof", "h", "bat"]

# axis: rc vector (left_right, forward_back, up_down, yaw), reaction channel, its minimum threshold,
#       speed channels (first one that responds is used)
AXES = {
    "yaw":      ((0, 0, 0, 1), "yaw",   2.0, ["yaw_rate"]),
    "throttle": ((0, 0, 1, 0), "vgz",   1.0, ["vgz", "tof_rate"]),
    "pitch":    ((0, 1, 0, 0), "pitch", 2.0, ["vgx"]),
    "roll":     ((1, 0, 0, 0), "roll",  2.0, ["vgy"]),
}


# ------------------------------------------------------------------ recording (background threads)
def log_state(tello, rec, stop):
    """Store every new state packet with the time we first saw it.

    djitellopy (2.5) replaces the whole state dict per packet and does not timestamp it, so we poll at
    ~500 Hz and stamp on arrival: <= 2 ms of jitter, well under the ~100 ms state period.
    """
    last = None
    while not stop.is_set():
        st = tello.get_current_state()
        if st and st is not last:
            last = st
            rec["state"].append([time.time()] + [float(st.get(k, "nan")) for k in FIELDS])
        time.sleep(0.002)


def log_video(frame_read, rec, stop):
    """Store every new frame's arrival time and how much it differs from the previous frame."""
    import cv2

    last, prev = None, None
    while not stop.is_set():
        img = frame_read.frame
        if img is None or img is last:
            time.sleep(0.002)
            continue
        t = time.time()
        last = img
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            rec["video"].append([t, float(np.mean(np.abs(small - prev)))])
        prev = small


# ------------------------------------------------------------------ drone commands
def timed_cmd(tello, cmd, timeout=10.0):
    """Send a command, return (reply, t_sent, t_reply). Polls every 1 ms (djitellopy polls every 100 ms)."""
    replies = tello.get_own_udp_object()["responses"]
    time.sleep(0.1)  # the Tello drops commands sent too close together
    replies.clear()
    t0 = time.time()
    tello.send_command_without_return(cmd)
    while not replies:
        if time.time() - t0 > timeout:
            return None, t0, time.time()
        time.sleep(0.001)
    t1 = time.time()
    return replies.pop(0).decode(errors="ignore").strip(), t0, t1


def hold(tello, rc, seconds):
    """Re-send one rc command at 20 Hz for `seconds` (also keeps the 15 s auto-land from firing)."""
    t_end = time.time() + seconds
    while True:
        tello.send_rc_control(*rc)
        if time.time() >= t_end:
            return
        time.sleep(0.05)


def rc_step(tello, rec, axis, sign, mag, dur, settle):
    rc = [sign * mag * c for c in AXES[axis][0]]
    ev = {"type": "rc", "axis": axis, "sign": sign, "t_cmd": time.time()}
    hold(tello, rc, dur)
    ev["t_release"] = time.time()
    rec["events"].append(ev)
    hold(tello, [0, 0, 0, 0], settle)


def move(tello, rec, cmd, settle):
    reply, t0, t1 = timed_cmd(tello, cmd, timeout=15)
    rec["events"].append({"type": "move", "cmd": cmd, "axis": "pitch", "t_cmd": t0, "t_release": t1, "reply": reply})
    hold(tello, [0, 0, 0, 0], settle)


# ------------------------------------------------------------------ analysis
def build_series(rec):
    S = np.array(rec["state"], float)
    d = {"t": S[:, 0]}
    for i, k in enumerate(FIELDS):
        d[k] = S[:, i + 1]
    d["yaw"] = np.degrees(np.unwrap(np.radians(d["yaw"])))
    d["yaw_rate"] = np.gradient(d["yaw"], d["t"])
    d["tof_rate"] = np.gradient(d["tof"], d["t"])
    return d


def first_time(t, mask):
    i = np.flatnonzero(mask)
    return float(t[i[0]]) if i.size else None


def analyze_step(d, noise, video, v_thr, ev, t_next):
    _, react, min_thr, speeds = AXES[ev["axis"]]
    t, t_cmd, t_rel = d["t"], ev["t_cmd"], ev["t_release"]
    pre = (t >= t_cmd - 0.5) & (t < t_cmd)
    after = (t >= t_cmd) & (t < t_next)
    out = {"axis": ev["axis"], "cmd": ev.get("cmd", f"{ev['axis']} {'+' if ev.get('sign', 1) > 0 else '-'}")}

    # lag: first telemetry sample that moved away from the pre-command value
    base = np.median(d[react][pre]) if pre.any() else d[react][after][0]
    t_react = first_time(t, after & (np.abs(d[react] - base) >= max(min_thr, 3 * noise.get(react, 0))))
    out["lag"] = None if t_react is None else t_react - t_cmd

    # speed-up and stop, on the first speed channel that really moved
    for ch in speeds:
        v = d[ch]
        dev = v - (np.median(v[pre]) if pre.any() else 0.0)
        win = (t >= t_cmd) & (t <= t_rel + 0.5)
        if win.sum() < 3:
            break
        peak = float(np.median(np.sort(np.abs(dev[win]))[-3:]))
        if peak < max(3 * noise.get(ch, 0), 1e-6):
            continue
        sign = np.sign(dev[win][np.argmax(np.abs(dev[win]))])
        r = dev * sign / peak
        t90 = first_time(t, win & (r >= 0.9))
        t_stop = first_time(t, (t > t_rel) & (t < t_next) & (r <= 0.1))
        out.update(speed_channel=ch, peak=peak,
                   speedup=None if (t90 is None or t_react is None) else max(0.0, t90 - t_react),
                   stop=None if t_stop is None else t_stop - t_rel)
        break

    # video: first frame after the command whose change is well above hover level and the pre-command level
    if len(video):
        vt, ve = video[:, 0], video[:, 1]
        pre_v = ve[(vt >= t_cmd - 0.3) & (vt < t_cmd)]
        thr = max(v_thr, 1.5 * pre_v.max()) if pre_v.size else v_thr
        t_vid = first_time(vt, (vt >= t_cmd) & (vt < t_next) & (ve >= thr))
        out["video"] = None if t_vid is None else t_vid - t_cmd
        out["video_delay"] = None if (t_vid is None or t_react is None) else t_vid - t_react
    return out


def spread(vals, scale=1000.0, unit="ms"):
    v = np.array([x for x in vals if x is not None], float) * scale
    if not v.size:
        return "-"
    sd = v.std(ddof=1) if v.size > 1 else 0.0
    return f"{v.mean():6.0f} ± {sd:4.0f} {unit}  [{v.min():.0f} … {v.max():.0f}]  n={v.size}"


def stream_line(ts):
    if len(ts) < 5:
        return "not enough samples"
    dt = np.diff(ts)
    return (f"{1 / np.median(dt):5.1f} Hz   interval {1000 * np.median(dt):.0f} ms   jitter {1000 * dt.std():.0f} ms   "
            f"worst gap {1000 * dt.max():.0f} ms   n={len(ts)}")


def analyze(rec):
    print("\n" + "=" * 78 + "\nTELLO LATENCY REPORT   " + rec["meta"].get("stamp", "") + f"   battery {rec['meta'].get('battery')}%")
    print("=" * 78)
    rtt = rec["meta"].get("rtt", [])
    print(f"command delay (round trip)   {spread(rtt)}")
    for k in ("takeoff", "land"):
        if rec["meta"].get(k) is not None:
            print(f"{k:<28} 'ok' after {rec['meta'][k] * 1000:.0f} ms")
    video = np.array(rec["video"], float).reshape(-1, 2)
    if rec["state"]:
        print(f"state stream                 {stream_line(np.array(rec['state'])[:, 0])}")
    if len(video):
        print(f"video stream                 {stream_line(video[:, 0])}")
    steps = [e for e in rec["events"] if e["type"] in ("rc", "move")]
    if not steps or len(rec["state"]) < 10:
        print("no flight steps to analyse")
        return {}

    d = build_series(rec)
    h0, h1 = rec["meta"]["hover"]
    hov = (d["t"] >= h0) & (d["t"] <= h1)
    noise = {k: float(np.nanstd(d[k][hov])) for k in d if k != "t"}
    hv = video[(video[:, 0] >= h0) & (video[:, 0] <= h1), 1] if len(video) else np.array([])
    v_thr = float(hv.mean() + 5 * hv.std()) if hv.size > 5 else 5.0

    results = []
    for i, ev in enumerate(steps):
        t_next = steps[i + 1]["t_cmd"] if i + 1 < len(steps) else ev["t_release"] + 3.0
        results.append(analyze_step(d, noise, video, v_thr, ev, t_next))

    groups = [(a, [r for s, r in zip(steps, results) if s["type"] == "rc" and s["axis"] == a]) for a in AXES]
    groups.append(("forward/back move", [r for s, r in zip(steps, results) if s["type"] == "move"]))
    for name, rs in groups:
        if not rs:
            continue
        ch = next((r["speed_channel"] for r in rs if "speed_channel" in r), "?")
        print("-" * 78)
        print(f"{name.upper()}   ({sum('speed_channel' in r for r in rs)}/{len(rs)} steps responded, speed channel {ch}, "
              f"peak {spread([r.get('peak') for r in rs], 1.0, '')})")
        if name.startswith("forward"):
            oks = [s["t_release"] - s["t_cmd"] for s in steps if s["type"] == "move"]
            print(f"  time to 'ok'   {spread(oks)}")
        print(f"  lag            {spread([r.get('lag') for r in rs])}")
        print(f"  speed-up       {spread([r.get('speedup') for r in rs])}")
        if not name.startswith("forward"):
            print(f"  stop           {spread([r.get('stop') for r in rs])}")
        print(f"  video          {spread([r.get('video') for r in rs])}")
        print(f"  video delay    {spread([r.get('video_delay') for r in rs])}")
    print("=" * 78)
    print("lag = command -> first telemetry reaction | speed-up = reaction -> 90 % of peak speed | stop = rc 0 -> < 10 %")
    print("video = command -> first moving frame | video delay = video - lag")
    return {"steps": results, "noise": noise, "video_threshold": v_thr}


# ------------------------------------------------------------------ flight
def run(args):
    import logging

    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.WARNING)  # otherwise every rc packet is printed
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = {"meta": {"stamp": stamp, "args": vars(args)}, "state": [], "video": [], "events": []}
    stop = threading.Event()

    tello = Tello()
    tello.connect()
    rec["meta"]["battery"] = tello.get_battery()
    print(f"connected, battery {rec['meta']['battery']}%")
    if not args.ground and rec["meta"]["battery"] < args.min_battery:
        print(f"battery below {args.min_battery}% -> not flying")
        return
    threading.Thread(target=log_state, args=(tello, rec, stop), daemon=True).start()
    tello.streamon()
    threading.Thread(target=log_video, args=(tello.get_frame_read(), rec, stop), daemon=True).start()
    time.sleep(3.0)  # let the video decoder start

    rec["meta"]["rtt"] = []
    for _ in range(10):
        reply, t0, t1 = timed_cmd(tello, "battery?")
        if reply is not None:
            rec["meta"]["rtt"].append(t1 - t0)

    if args.ground:
        t = time.time()
        time.sleep(5.0)
        rec["meta"]["hover"] = [t, time.time()]
    else:
        print("taking off - Ctrl+C lands")
        try:
            reply, t0, t1 = timed_cmd(tello, "takeoff", timeout=20)
            tello.is_flying = True
            rec["meta"]["takeoff"] = t1 - t0
            hold(tello, [0, 0, 0, 0], 2.0)
            t = time.time()
            hold(tello, [0, 0, 0, 0], args.hover)
            rec["meta"]["hover"] = [t, time.time()]
            for axis in AXES:
                print(f"{axis}: {args.trials} x (+{args.mag} / -{args.mag}) for {args.dur} s")
                for _ in range(args.trials):
                    rc_step(tello, rec, axis, +1, args.mag, args.dur, args.settle)
                    rc_step(tello, rec, axis, -1, args.mag, args.dur, args.settle)
            if args.forward > 0:
                print(f"forward/back {args.forward} cm x {args.trials}")
                for _ in range(args.trials):
                    move(tello, rec, f"forward {args.forward}", args.settle)
                    move(tello, rec, f"back {args.forward}", args.settle)
        except KeyboardInterrupt:
            print("\nCtrl+C -> landing")
        finally:
            tello.send_rc_control(0, 0, 0, 0)
            reply, t0, t1 = timed_cmd(tello, "land")
            rec["meta"]["land"] = t1 - t0
            tello.is_flying = False
            rec["meta"].setdefault("hover", [time.time() - 1, time.time()])

    time.sleep(0.5)
    stop.set()
    tello.end()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"tello_latency_{stamp}.json")
    with open(path, "w") as f:
        json.dump(rec, f)
    summary = analyze(rec)
    with open(path.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print(f"raw data -> {path}")


def main():
    p = argparse.ArgumentParser(description="Measure Tello command, motion and video lag in one flight.")
    p.add_argument("--mag", type=int, default=30, help="rc stick value for the steps, 10-100 (default 30)")
    p.add_argument("--dur", type=float, default=1.5, help="seconds each step is held (default 1.5)")
    p.add_argument("--settle", type=float, default=2.0, help="hover seconds between steps (default 2.0)")
    p.add_argument("--trials", type=int, default=3, help="repeats per axis, each = one + and one - step (default 3)")
    p.add_argument("--forward", type=int, default=50, help="cm for the forward/back move test, 0 = skip (default 50)")
    p.add_argument("--hover", type=float, default=3.0, help="hover seconds used to measure sensor noise (default 3)")
    p.add_argument("--min-battery", type=int, default=30)
    p.add_argument("--ground", action="store_true", help="no takeoff: command delay and stream rates only")
    p.add_argument("--analyze", metavar="JSON", help="re-analyse a saved run without flying")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.analyze:
        with open(args.analyze) as f:
            analyze(json.load(f))
    else:
        run(args)


if __name__ == "__main__":
    main()
