"""Viz frames: one plain dict per brain tick (20 Hz), the only thing the viewer ever sees.

A frame is picklable (numpy arrays, floats, dicts) and every key except "t" is optional:

    t              float  seconds since the start of the run
    tick           int
    counts         int32 (n,) spikes per subgraph neuron in this tick (FlyController.last_counts)
    channels       float32 (22,) encoder rates: LC10a L0..L7, R0..R7, LC9 L/R, LC11 L/R, AROUSAL L/R (Hz)
    bin_centers_deg float (16,) encoder bin centers for L0..L7, R0..R7 (+ = right)
    inputs         {input group: mean rate Hz}
    dn             {output group: rate Hz} (FlyController.last_dn_rates)
    sticks         {"yaw", "fb"} final sent sticks, plus "yaw_raw", "fb_raw" from the controller
    readout        {"yaw_drive", "fwd_drive"} the readout's pre-tanh drives from the DN rates (FLY controllers)
    target         {"valid", "cx", "cy", "h" (px, 960x720), "bearing_deg", "range_m", "in_view", "z_ref_m"}
    sim            {"drone_xy", "drone_psi_deg", "person_xy", "hfov_deg"} (simulation only)
    image          optional uint8 (H, W, 3) RGB drone camera frame
    meta           {"arm", "brain", "seed", "kind", "source"} (source: "sim", "synthetic", "drone")

frame_from_controller() builds one from any controller that has FlyController's read-only hooks
(last_counts, last_channels, last_input_rates, last_dn_rates, last_sticks); that is the call the live
drone runtime makes before VizSink.publish(). synthetic_frames() drives the subgraph LIF directly with a
scripted spot, for tests and when the simulator is not available.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from flyfollow.interfaces import DT, IMG_H, IMG_W, N_AZIMUTH_BINS, OUTPUT_GROUPS

NB = N_AZIMUTH_BINS


def default_bin_centers_deg(theta_max_deg: float = 40.0, overlap_deg: float = 10.0) -> np.ndarray:
    """Encoder bin centers [L0..L7, R0..R7] (same formula as flyfollow.senses.target.TargetEncoder)."""
    span = theta_max_deg + overlap_deg
    c_r = -overlap_deg + (np.arange(NB) + 0.5) * span / NB
    return np.concatenate([-c_r, c_r])


def _get(obj, name: str):
    try:
        return getattr(obj, name)
    except Exception:  # noqa: BLE001 (a hook that raises is treated as missing)
        return None


def frame_from_controller(controller, t: float, tick: int = 0, box=None, sticks: tuple[float, float] | None = None,
                          target: dict | None = None, image: np.ndarray | None = None, sim: dict | None = None,
                          meta: dict | None = None) -> dict:
    """Snapshot a controller's last tick as a viz frame. Works with partial hooks (missing ones are omitted)."""
    fr: dict = {"t": float(t), "tick": int(tick)}
    counts = _get(controller, "last_counts")
    if counts is not None:
        fr["counts"] = np.asarray(counts, np.int32).copy()
    ch = _get(controller, "last_channels")
    if ch is not None:
        fr["channels"] = np.asarray(ch, np.float32).copy()
    enc = _get(controller, "encoder")
    centers = _get(enc, "centers") if enc is not None else None
    if centers is not None:
        fr["bin_centers_deg"] = np.degrees(np.asarray(centers, np.float64))
    ir = _get(controller, "last_input_rates")
    if isinstance(ir, dict):
        fr["inputs"] = {g: float(np.mean(v)) if np.size(v) else 0.0 for g, v in ir.items()}
    dn = _get(controller, "last_dn_rates")
    if isinstance(dn, dict):
        fr["dn"] = {k: float(v) for k, v in dn.items()}
    else:
        rates = _get(controller, "last_rates")
        if rates is not None and np.size(rates) == len(OUTPUT_GROUPS):
            fr["dn"] = dict(zip(OUTPUT_GROUPS, np.asarray(rates, float).tolist()))
    dec = _get(controller, "decoder")
    try:  # the readout's pre-tanh drives (read-only): weighted, per-side normalized DN asymmetry + bias
        if dec is not None and getattr(dec, "f_yaw", None) is not None:
            fr["readout"] = {"yaw_drive": float(dec.wy @ dec.f_yaw + dec.b_yaw), "fwd_drive": float(dec.wf @ dec.f_fwd + dec.b_fwd)}
    except Exception:  # noqa: BLE001, S110 (decoder internals changed: fall back to raw DNa02)
        pass
    raw = _get(controller, "last_sticks")
    st: dict = {}
    if raw is not None:
        st["yaw_raw"], st["fb_raw"] = float(raw[0]), float(raw[1])
    if sticks is not None:
        st["yaw"], st["fb"] = float(sticks[0]), float(sticks[1])
    elif raw is not None:
        st["yaw"], st["fb"] = st["yaw_raw"], st["fb_raw"]
    if st:
        fr["sticks"] = st
    tg = dict(target or {})
    if box is not None:
        tg.setdefault("valid", bool(box.valid))
        tg.setdefault("cx", float(box.cx))
        tg.setdefault("cy", float(box.cy))
        tg.setdefault("h", float(box.h))
    if tg:
        fr["target"] = tg
    if sim:
        fr["sim"] = sim
    if image is not None:
        fr["image"] = image
    m = {"brain": str(_get(controller, "brain_path") or ""), "arm": str(_get(controller, "name") or "")}
    m.update(meta or {})
    fr["meta"] = m
    return fr


# --------------------------------------------------------------------------- derived signals
def lc10a_bin_rates(frame: dict, bins: dict[str, list[np.ndarray]] | None = None, dt: float = DT) -> np.ndarray:
    """Per-bin LC10a activity [L0..L7, R0..R7] in Hz: measured spikes when counts and bins exist, else encoder rates."""
    counts = frame.get("counts")
    if counts is not None and bins:
        out = np.zeros(2 * NB)
        for s_i, g in enumerate(("LC10a_L", "LC10a_R")):
            for k, b in enumerate(bins.get(g, [])[:NB]):
                if len(b):
                    out[s_i * NB + k] = float(np.mean(counts[b])) / dt
        return out
    ch = frame.get("channels")
    if ch is not None:
        return np.asarray(ch[: 2 * NB], float)
    return np.zeros(2 * NB)


def look_bearing(bin_rates: np.ndarray, centers_deg: np.ndarray | None = None, min_total_hz: float = 20.0) -> float | None:
    """LC10a activity centroid in radians (+ = right), None when LC10a is quiet."""
    r = np.maximum(np.asarray(bin_rates, float), 0.0)
    tot = r.sum()
    if tot < min_total_hz:
        return None
    c = np.asarray(centers_deg if centers_deg is not None else default_bin_centers_deg(), float)
    return math.radians(float((r * c).sum() / tot))


def motor_state(frame: dict, bins: dict | None = None):
    """Frame -> fly_body.MotorState (DNa02 L/R, final sticks, LC10a centroid bearing)."""
    from flyfollow.viz.fly_body import MotorState

    dn = frame.get("dn") or {}
    st = frame.get("sticks") or {}
    br = lc10a_bin_rates(frame, bins)
    lb = look_bearing(br, frame.get("bin_centers_deg"))
    tot = sum(dn.values()) if dn else 0.0
    ro = frame.get("readout") or {}
    return MotorState(steer_drive=ro.get("yaw_drive"), dna02_l=float(dn.get("DNa02_L", 0.0)), dna02_r=float(dn.get("DNa02_R", 0.0)),
                      yaw_stick=float(st.get("yaw", 0.0)), fb_stick=float(st.get("fb", 0.0)), look_bearing=lb,
                      activity=float(min(1.0, tot / 1000.0)))


# --------------------------------------------------------------------------- synthetic stream
def scripted_bearing(t: float, script: str = "sweep") -> float:
    """Scripted target bearing in degrees (+ = right)."""
    if script == "left_right":
        return 25.0 if (t % 4.0) < 2.0 else -25.0
    if script == "hold":
        return 0.0
    return 30.0 * math.sin(2 * math.pi * t / 6.0)


def synthetic_frames(brain_path: str | Path, seconds: float = 10.0, seed: int = 0, script: str = "sweep",
                     r_max: float = 150.0, sigma_deg: float = 10.0, aux_hz: float = 10.0, realtime: bool = False) -> Iterator[dict]:
    """Scripted spot -> LC10a Gaussian over the npz azimuth bins -> subgraph LIF (FlyBrain) -> frames.

    No simulator and no trained readout: yaw = 60 tanh((DNa02_R - DNa02_L) / 200), fb from the spot size.
    """
    import time as _time

    from flyfollow.brain.build import ensure_flydrones

    ensure_flydrones()
    from flyfollow.pilot.fly_brain import FlyBrain

    brain = FlyBrain(brain_path, seed=seed)
    bins = brain.t.bins if hasattr(brain.t, "bins") else None
    if bins is None:
        from flyfollow.pilot.fly_brain import lc10a_bins

        bins = {g: lc10a_bins(brain.connectome, g) for g in ("LC10a_L", "LC10a_R")}
    centers = default_bin_centers_deg()
    groups = brain.groups
    # warm up without a target
    for _ in range(10):
        brain.tick({}, 50.0)
    n = round(seconds / DT)
    t_start = _time.perf_counter()
    yaw_f = 0.0
    for i in range(n):
        t = i * DT
        b = scripted_bearing(t, script)
        rng_m = 2.0 + 0.6 * math.sin(2 * math.pi * t / 9.0)
        s = 2.0 / rng_m
        tun = np.exp(-0.5 * ((b - centers) / (sigma_deg * s**0.6)) ** 2)
        ch = r_max * tun * (1.5 * s / (s + 0.5))
        rates: dict[str, np.ndarray | float] = {}
        for s_i, g in enumerate(("LC10a_L", "LC10a_R")):
            per = np.zeros(groups[g].size, np.float32)
            pos = {int(v): j for j, v in enumerate(groups[g])}
            for k, bb in enumerate(bins[g]):
                for v in bb:
                    per[pos[int(v)]] = ch[s_i * NB + k]
            rates[g] = per
        side_act = (tun[:NB].max(), tun[NB:].max())
        for stem in ("LC9", "LC11"):
            rates[f"{stem}_L"] = aux_hz * side_act[0]
            rates[f"{stem}_R"] = aux_hz * side_act[1]
        dn = brain.tick(rates, 50.0)
        yaw_f += 0.4 * (60.0 * math.tanh((dn["DNa02_R"] - dn["DNa02_L"]) / 200.0) - yaw_f)
        fb = float(np.clip(40.0 * (1.0 - s), -20, 35))
        fx = 921.0
        cx = IMG_W / 2 + fx * math.tan(math.radians(b))
        h = 919.0 * 0.23 / rng_m
        chans = np.zeros(22, np.float32)
        chans[: 2 * NB] = ch
        yield {
            "t": t, "tick": i, "counts": brain.last_counts.astype(np.int32).copy(), "channels": chans,
            "bin_centers_deg": centers, "dn": dn,
            "inputs": {g: float(np.mean(v)) for g, v in rates.items()},
            "sticks": {"yaw": yaw_f, "fb": fb, "yaw_raw": yaw_f, "fb_raw": fb},
            "target": {"valid": True, "cx": cx, "cy": IMG_H * 0.45, "h": h, "bearing_deg": b, "range_m": rng_m,
                       "in_view": abs(cx - IMG_W / 2) < IMG_W / 2, "z_ref_m": 2.0},
            "meta": {"arm": "SYNTHETIC", "brain": str(brain.path), "seed": seed, "source": "synthetic", "script": script},
        }
        if realtime:
            lag = (i + 1) * DT - (_time.perf_counter() - t_start)
            if lag > 0:
                _time.sleep(lag)


__all__ = ["frame_from_controller", "lc10a_bin_rates", "look_bearing", "motor_state", "synthetic_frames"]
