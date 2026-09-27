"""G0 connectome audit (docs/DRONE_RL_PLAN.md 4.3).

    python -m flyfollow.audit.audit --brain data/brains/pursuit_core1.npz
    python -m flyfollow.audit.audit --brain data/brains/pursuit_core1.npz data/brains/pursuit_core2.npz
    python -m flyfollow.audit.audit --structural-only

Structural part (on the FULL brain, data/brains/malecns_full.npz, per side):
  direct LC10a -> DN synapse counts; 2-hop and 3-hop signed path sums (sum over paths of the product of
  signed synapse counts); ipsi vs contra; group sizes; P1 (AROUSAL) links; top intermediates; the DN
  types with the highest LC10a path weight (for the expansion retry).
Functional part (LIF on each pursuit core):
  A: spot at -30, -15, 0, +15, +30 deg, 5 repeats; DN asymmetry (R - L) per type and pooled; SNR.
  B: spot at 0 deg with s = 0.5, 1, 2; total DN activity must change monotonically with s.
  C: A and B with arousal (P1) on and off.
Results go to stdout and to data/audit/audit_*.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from flyfollow.brain.build import ensure_flydrones, full_path, load_config, resolve
from flyfollow.interfaces import DN_TYPES, N_AZIMUTH_BINS, OUTPUT_GROUPS, azimuth_bins, data_root

ensure_flydrones()

from flydrones.brain.connectome import Connectome
from flydrones.brain.lif import LIFNetwork, LIFParams

BEARINGS_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
SIZES = (0.5, 1.0, 2.0)
REPEATS = 5
WARMUP_MS = 300.0
STIM_TICKS = 20  # 1 s of spot per repeat
SKIP_TICKS = 4  # first 200 ms of the spot are transient and excluded
TICK_MS = 50.0

# DN turning direction when the cell on side d fires (FLYGUIDE_SPEC 6.6): +1 ipsiversive, -1 contraversive.
TURN_SENSE = {"DNa02": 1, "DNa01": 1, "DNb05": 1, "DNg13": 1, "DNb06": -1, "DNp09": 1}


# ---------------------------------------------------------------- spot encoder (audit only)
class SpotEncoder:
    """Simple audit encoder: a spot at bearing theta (rad, + right) with normalized size s.

    Right LC10a covers the right visual field and left LC10a the left, with a frontal overlap band
    where both respond. Bin k of the right side is centered at -overlap + k * span / 7 deg (bin 0
    most frontal); the left side is mirrored. Tuning is Gaussian in bearing; larger s widens the
    tuning (more bins) and raises the gain. LC9 and LC11 of a side get one rate each, scaled by that
    side's peak tuning. AROUSAL (P1) gets a tonic rate when arousal is on.
    """

    def __init__(self, c: Connectome, r_max: float = 150.0, sigma_deg: float = 10.0, overlap_deg: float = 10.0,
                 span_deg: float = 100.0, aux_rate: float = 50.0, arousal_rate: float = 20.0,
                 lc10_only: bool = False):
        self.c = c
        bins = c.meta.get("azimuth_bins") or {}
        self.bins = {}
        for g in ("LC10a_L", "LC10a_R"):
            bl = bins.get(g) or [b.tolist() for b in azimuth_bins(c.group(g), N_AZIMUTH_BINS)]
            self.bins[g] = [np.asarray(b, dtype=np.int64) for b in bl]
        k = np.arange(N_AZIMUTH_BINS)
        right = -overlap_deg + k * span_deg / (N_AZIMUTH_BINS - 1)
        self.centers = {"LC10a_R": np.radians(right), "LC10a_L": -np.radians(right)}
        self.r_max, self.sigma = r_max, np.radians(sigma_deg)
        self.aux_rate, self.arousal_rate, self.lc10_only = aux_rate, arousal_rate, lc10_only

    def encode(self, theta: float | None, s: float = 1.0, arousal: bool = True) -> tuple[np.ndarray, np.ndarray]:
        idx, rates = [], []
        if theta is not None:
            s = max(float(s), 0.1)
            sigma = self.sigma * s**0.6
            gain = 1.5 * s / (s + 0.5)
            for g, side in (("LC10a_L", "L"), ("LC10a_R", "R")):
                tun = np.exp(-0.5 * ((theta - self.centers[g]) / sigma) ** 2)
                for b, t in zip(self.bins[g], tun):
                    if t > 1e-3 and b.size:
                        idx.append(b)
                        rates.append(np.full(b.size, self.r_max * gain * t, np.float32))
                if not self.lc10_only:
                    for aux in ("LC9", "LC11"):
                        a = self.c.group(f"{aux}_{side}")
                        if a.size and tun.max() > 1e-3:
                            idx.append(a)
                            rates.append(np.full(a.size, self.aux_rate * gain * tun.max(), np.float32))
        if arousal:
            for g in ("AROUSAL_L", "AROUSAL_R"):
                a = self.c.group(g)
                if a.size:
                    idx.append(a)
                    rates.append(np.full(a.size, self.arousal_rate, np.float32))
        if not idx:
            return np.zeros(0, np.int64), np.zeros(0, np.float32)
        return np.concatenate(idx), np.concatenate(rates)


# ---------------------------------------------------------------- structural audit (full brain)
def _col_sum(Wcsc, cols: np.ndarray) -> np.ndarray:
    """Sum of the given presynaptic columns: total signed input each neuron gets from the group."""
    if cols.size == 0:
        return np.zeros(Wcsc.shape[0])
    return np.asarray(Wcsc[:, cols].sum(axis=1)).ravel().astype(np.float64)


def structural(full: Connectome, top: int = 12) -> dict:
    W = full.weights.tocsc()
    Wr = W.tocsr()
    types = full.types.astype(str)
    sides = full.sides.astype(str)
    dn_names = [f"{t}_{s}" for t in (*DN_TYPES, "DNp09") for s in "LR"]
    out: dict = {"sizes": {k: int(v.size) for k, v in full.groups.items()}}

    def label(i: int) -> str:
        return f"{types[i]}_{sides[i] or '?'}"

    paths = {}
    inter = {}
    for src_name in ("LC10a_L", "LC10a_R", "AROUSAL_L", "AROUSAL_R", "LC9_L", "LC9_R", "LC11_L", "LC11_R"):
        src = full.group(src_name)
        v1 = _col_sum(W, src)  # 1 synapse: src -> X
        v2 = Wr @ v1  # 2 synapses: src -> X -> Y
        v3 = Wr @ v2
        row = {}
        for dn in dn_names:
            d = full.group(dn)
            if d.size == 0:
                continue
            row[dn] = {"direct": float(v1[d].sum()), "hop2": float(v2[d].sum()), "hop3": float(v3[d].sum())}
        paths[src_name] = row
        if src_name.startswith(("LC10a", "AROUSAL")):
            per_x = {}
            for dn in dn_names:
                d = full.group(dn)
                if d.size == 0:
                    continue
                contrib = np.asarray(Wr[d].multiply(v1[None, :]).sum(axis=0)).ravel()
                nz = np.flatnonzero(contrib)
                order = nz[np.argsort(-np.abs(contrib[nz]))][:5]
                per_x[dn] = [(label(i), float(contrib[i])) for i in order]
            # top intermediates over the whole steering DN set
            dsel = np.concatenate([full.group(dn) for dn in dn_names if full.group(dn).size])
            contrib_all = np.asarray(abs(Wr[dsel]).multiply(np.abs(v1)[None, :]).sum(axis=0)).ravel()
            order = np.argsort(-contrib_all)[:top]
            per_x["_top_all_DNs_abs"] = [(label(i), float(contrib_all[i])) for i in order if contrib_all[i] > 0]
            inter[src_name] = per_x
    out["paths"] = paths
    out["intermediates"] = inter

    # ipsi vs contra per DN type and hop, and the turn-toward check for LC10a
    ic = {}
    for t in (*DN_TYPES, "DNp09"):
        e = {}
        for hop in ("direct", "hop2", "hop3"):
            ipsi = paths["LC10a_L"].get(f"{t}_L", {}).get(hop, 0) + paths["LC10a_R"].get(f"{t}_R", {}).get(hop, 0)
            contra = paths["LC10a_L"].get(f"{t}_R", {}).get(hop, 0) + paths["LC10a_R"].get(f"{t}_L", {}).get(hop, 0)
            e[hop] = {"ipsi": ipsi, "contra": contra}
        ic[t] = e
    out["ipsi_contra"] = ic
    turn = {}
    for side in "LR":
        other = "R" if side == "L" else "L"
        for hop in ("direct", "hop2", "hop3"):
            # >0 means this side's LC10a drives turning toward its own side (toward the spot it sees)
            val = 0.0
            for t in DN_TYPES:
                p = paths[f"LC10a_{side}"]
                val += TURN_SENSE[t] * (p.get(f"{t}_{side}", {}).get(hop, 0) - p.get(f"{t}_{other}", {}).get(hop, 0))
            turn[f"LC10a_{side}_{hop}"] = val
    out["turn_toward_index"] = turn

    # DN types with the highest LC10a path weight (candidates for the expansion retry)
    sc = full.superclass.astype(str) if full.superclass is not None else np.array([""] * full.n)
    is_dn = sc == "descending_neuron"
    lc_all = np.concatenate([full.group("LC10a_L"), full.group("LC10a_R")])
    v1 = _col_sum(W, lc_all)
    v2 = Wr @ v1
    rank = {}
    for key, v in (("direct", v1), ("hop2", v2)):
        dn_idx = np.flatnonzero(is_dn & (v != 0))
        agg: dict[str, float] = {}
        for i in dn_idx:
            agg[types[i]] = agg.get(types[i], 0.0) + float(v[i])
        rank[key] = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:15]
    out["top_dn_types_by_LC10a_path"] = rank

    # LC10 subtypes (retry inputs) -> readout DNs, 2-hop
    retry = {}
    for g in ("LC10b", "LC10d", "LC10e", "LC9", "LC11"):
        for side in "LR":
            src = full.group(f"{g}_{side}")
            if src.size == 0:
                continue
            v = Wr @ _col_sum(W, src)
            retry[f"{g}_{side}"] = {dn: float(v[full.group(dn)].sum()) for dn in dn_names if full.group(dn).size}
    out["retry_inputs_hop2"] = retry

    # P1 (AROUSAL) direct links to LC10a, AOTU019 and the DNs
    p1 = np.concatenate([full.group("AROUSAL_L"), full.group("AROUSAL_R")])
    tgt = {g: full.group(g) for g in ("LC10a_L", "LC10a_R", "AOTU019_L", "AOTU019_R")}
    p1v = _col_sum(W, p1)
    p1_into = np.asarray(Wr[p1].sum(axis=0)).ravel()
    out["p1"] = {
        "n_cells": int(p1.size),
        "p1_to": {g: float(p1v[i].sum()) for g, i in tgt.items()},
        "to_p1_from": {g: float(p1_into[i].sum()) for g, i in tgt.items()},
        "p1_hop2_to": {g: float((Wr @ p1v)[i].sum()) for g, i in tgt.items()},
        "p1_out_synapses_total": float(np.abs(W[:, p1]).sum()),
    }
    return out


def print_structural(s: dict) -> None:
    print("\n=== STRUCTURAL (full brain, signed synapse counts; path sums are sums of products) ===")
    print("group sizes:", ", ".join(f"{k}={v}" for k, v in s["sizes"].items()))
    for src in ("LC10a_L", "LC10a_R"):
        print(f"\n{src} ->   " + "  ".join(f"{h:>22s}" for h in ("direct", "2-hop", "3-hop")))
        for dn, e in s["paths"][src].items():
            print(f"  {dn:9s} {e['direct']:22.0f} {e['hop2']:22.0f} {e['hop3']:22.3g}")
    print("\nipsi vs contra (LC10a_L->X_L + LC10a_R->X_R vs crossed):")
    for t, e in s["ipsi_contra"].items():
        print(f"  {t:6s} " + "  ".join(f"{h}: ipsi {v['ipsi']:.4g} contra {v['contra']:.4g}" for h, v in e.items()))
    print("\nturn-toward index (>0: that side's LC10a drives turning toward its own side):")
    for k, v in s["turn_toward_index"].items():
        print(f"  {k:18s} {v:.4g}")
    print("\ntop 2-hop intermediates LC10a -> X -> steering DNs (|contribution|):")
    for src in ("LC10a_L", "LC10a_R"):
        print(f"  {src}: " + ", ".join(f"{n} {v:.0f}" for n, v in s["intermediates"][src]["_top_all_DNs_abs"]))
    print("\nDN types (all DNs) ranked by LC10a path weight:")
    for k, v in s["top_dn_types_by_LC10a_path"].items():
        print(f"  {k}: " + ", ".join(f"{n} {w:.4g}" for n, w in v))
    print("\nP1 / AROUSAL:", json.dumps(s["p1"]))
    print("  2-hop P1 -> DNs:", {dn: round(e["hop2"]) for dn, e in s["paths"]["AROUSAL_L"].items()},
          {dn: round(e["hop2"]) for dn, e in s["paths"]["AROUSAL_R"].items()})
    print("  top P1 intermediates:", s["intermediates"]["AROUSAL_L"]["_top_all_DNs_abs"][:6])


# ---------------------------------------------------------------- functional audit (LIF on a core)
_W: dict = {}


def _init_worker(path: str, enc_kwargs: dict | None = None) -> None:
    c = Connectome.load(path)
    _W["c"] = c
    _W["net"] = LIFNetwork(c.weights, LIFParams.from_dict(c.meta.get("lif")), seed=0)
    _W["enc"] = SpotEncoder(c, **(enc_kwargs or {}))
    _W["enc_lc10"] = SpotEncoder(c, **{**(enc_kwargs or {}), "lc10_only": True})
    _W["dn"] = [c.group(g) for g in OUTPUT_GROUPS]


def _run_one(job: tuple) -> dict:
    """One repeat: warmup with arousal only, then STIM_TICKS ticks of the spot. Returns per-tick DN counts."""
    theta_deg, s, arousal, seed, enc_name = job
    c, dn = _W["c"], _W["dn"]
    enc = _W[enc_name]
    net = _W["net"].copy(seed=seed)
    idx, r = enc.encode(None, s, arousal=arousal)
    net.set_input(idx, r)
    net.run(WARMUP_MS)
    theta = None if theta_deg is None else np.radians(theta_deg)
    idx, r = enc.encode(theta, s, arousal=arousal)
    net.set_input(idx, r)
    counts = np.zeros((STIM_TICKS, len(dn)), np.int32)
    t0 = time.perf_counter()
    for k in range(STIM_TICKS):
        cnt, _ = net.run(TICK_MS)
        counts[k] = [cnt[g].sum() for g in dn]
    return {"job": job, "counts": counts.tolist(), "wall_s": time.perf_counter() - t0, "n": c.n}


def _summ(runs: list[dict]) -> dict:
    """Per-repeat mean rates over the analysed ticks and per-tick statistics."""
    C = np.asarray([r["counts"] for r in runs], dtype=np.float64)[:, SKIP_TICKS:, :]  # (rep, tick, dn)
    rates = C * (1000.0 / TICK_MS)  # Hz per tick (one cell per group)
    per_rep = rates.mean(axis=1)  # (rep, dn)
    names = list(OUTPUT_GROUPS)
    iL = {t: names.index(f"{t}_L") for t in DN_TYPES}
    iR = {t: names.index(f"{t}_R") for t in DN_TYPES}
    asym_rep = {t: per_rep[:, iR[t]] - per_rep[:, iL[t]] for t in DN_TYPES}
    asym_tick = {t: rates[:, :, iR[t]] - rates[:, :, iL[t]] for t in DN_TYPES}
    pooled_rep = sum(asym_rep.values())
    pooled_tick = sum(asym_tick.values())
    total_rep = per_rep.sum(axis=1)

    def ms(x):
        x = np.asarray(x, dtype=np.float64)
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        return {"mean": float(x.mean()), "sd": sd, "snr": float(x.mean() / sd) if sd > 0 else float("inf") if x.mean() != 0 else 0.0}

    return {
        "rates_mean": {n: float(per_rep[:, i].mean()) for i, n in enumerate(names)},
        "rates_tick_sd": {n: float(rates[:, :, i].std()) for i, n in enumerate(names)},
        "asym": {t: ms(asym_rep[t]) for t in DN_TYPES},
        "asym_tick": {t: {"mean": float(asym_tick[t].mean()), "sd": float(asym_tick[t].std())} for t in DN_TYPES},
        "pooled": ms(pooled_rep),
        "pooled_tick": {"mean": float(pooled_tick.mean()), "sd": float(pooled_tick.std())},
        "total": ms(total_rep),
        "wall_s": float(np.mean([r["wall_s"] for r in runs])),
    }


def _monotone(means: list[float], ses: list[float], direction: float | None = None) -> tuple[bool, float]:
    m = np.asarray(means)
    se = np.asarray(ses)
    if direction is None:
        direction = float(np.sign(m[-1] - m[0])) or 1.0
    ok = all((m[i + 1] - m[i]) * direction > -2.0 * np.hypot(se[i], se[i + 1]) for i in range(len(m) - 1))
    return ok, direction


def functional(path: Path, workers: int, enc_name: str = "enc", enc_kwargs: dict | None = None,
               bearings: tuple = BEARINGS_DEG) -> dict:
    jobs = []
    for arousal in (True, False):
        for rep in range(REPEATS):
            jobs.append((None, 1.0, arousal, 1000 + rep, enc_name))
            for b in bearings:
                jobs.append((b, 1.0, arousal, 20_000 + rep + round((b + 180) * 10) * 10, enc_name))
            for s in SIZES:
                jobs.append((0.0, s, arousal, 3000 + rep + int(s * 100), enc_name))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(path), enc_kwargs)) as ex:
        results = list(ex.map(_run_one, jobs))
    wall = time.perf_counter() - t0
    groups: dict = {}
    for r in results:
        theta_deg, s, arousal, _seed, _e = r["job"]
        key = (theta_deg, s, arousal)
        groups.setdefault(key, []).append(r)
    out: dict = {"brain": str(path), "encoder": enc_name, "encoder_kwargs": enc_kwargs or {}, "wall_s": wall,
                 "n_jobs": len(jobs), "conditions": {}}
    se = lambda d: d["sd"] / np.sqrt(REPEATS)
    for arousal in (True, False):
        tag = "arousal_on" if arousal else "arousal_off"
        base = _summ(groups[(None, 1.0, arousal)])
        A = {b: _summ(groups[(b, 1.0, arousal)]) for b in bearings}
        B = {s: _summ(groups[(0.0, s, arousal)]) for s in SIZES}
        pm = [A[b]["pooled"]["mean"] for b in bearings]
        ps = [se(A[b]["pooled"]) for b in bearings]
        ends_sig = abs(pm[0]) > 2 * ps[0] and abs(pm[-1]) > 2 * ps[-1]
        sign_change = bool(np.sign(pm[0]) != np.sign(pm[-1]) and ends_sig)
        mono, direction = _monotone(pm, ps)
        # effect size: the swing across the bearing range must exceed one per-tick SD of the pooled
        # asymmetry, otherwise a 50 ms readout cannot use it (guards against tiny but "significant" swings)
        tick_sd = float(np.mean([A[b]["pooled_tick"]["sd"] for b in bearings]))
        a_effect = abs(pm[-1] - pm[0]) > tick_sd
        per_type = {}
        for t in DN_TYPES:
            tm = [A[b]["asym"][t]["mean"] for b in bearings]
            ts = [se(A[b]["asym"][t]) for b in bearings]
            tsig = abs(tm[0]) > 2 * ts[0] and abs(tm[-1]) > 2 * ts[-1]
            per_type[t] = {
                "sign_change": bool(np.sign(tm[0]) != np.sign(tm[-1]) and tsig),
                "monotone": _monotone(tm, ts)[0],
                "slope_sign": float(np.sign(tm[-1] - tm[0])),
            }
        tm = [B[s]["total"]["mean"] for s in SIZES]
        ts = [se(B[s]["total"]) for s in SIZES]
        b_mono, b_dir = _monotone(tm, ts)
        b_range_sig = abs(tm[-1] - tm[0]) > 2 * np.hypot(ts[0], ts[-1])
        b_effect = abs(tm[-1] - tm[0]) > 0.1 * abs(tm[len(tm) // 2])  # at least a 10 % change
        out["conditions"][tag] = {
            "baseline": base,
            "A": {str(b): v for b, v in A.items()},
            "B": {str(s): v for s, v in B.items()},
            "A_pass": bool(sign_change and mono and a_effect),
            "A_effect_ok": bool(a_effect),
            "A_tick_sd": tick_sd,
            "A_sign_change": sign_change,
            "A_monotone": mono,
            "A_direction": direction,  # +1: pooled (R - L) rises as the spot moves right
            "A_per_type": per_type,
            "B_pass": bool(b_mono and b_range_sig and b_effect),
            "B_effect_ok": bool(b_effect),
            "B_monotone": b_mono,
            "B_range_significant": bool(b_range_sig),
            "B_direction": b_dir,
        }
    return out


def print_functional(f: dict) -> None:
    print(f"\n=== FUNCTIONAL {f['brain']} (encoder {f['encoder']}, {f['n_jobs']} runs in {f['wall_s']:.0f} s wall) ===")
    for tag, c in f["conditions"].items():
        b = c["baseline"]
        print(f"\n[{tag}] baseline (no spot) DN rates Hz: " + " ".join(f"{k}={v:.0f}" for k, v in b["rates_mean"].items()))
        print(f"A: pass={c['A_pass']} (sign change {c['A_sign_change']}, monotone {c['A_monotone']}, "
              f"swing > per-tick SD {c['A_effect_ok']}, direction {c['A_direction']:+.0f})")
        print("  bearing   pooled R-L mean  sd   SNR | per-tick sd | " + " ".join(f"{t:>12s}" for t in DN_TYPES))
        for bstr, a in c["A"].items():
            p = a["pooled"]
            print(f"  {float(bstr):+6.0f}   {p['mean']:8.1f} {p['sd']:6.1f} {p['snr']:6.2f} | {a['pooled_tick']['sd']:8.1f}    | "
                  + " ".join(f"{a['asym'][t]['mean']:6.1f}+-{a['asym'][t]['sd']:4.1f}" for t in DN_TYPES))
        print("  per-type: " + ", ".join(f"{t} sign-change={v['sign_change']} mono={v['monotone']} slope={v['slope_sign']:+.0f}" for t, v in c["A_per_type"].items()))
        print("  DN rates at +30 deg: " + " ".join(f"{k}={v:.0f}" for k, v in c["A"]["30.0"]["rates_mean"].items()))
        print("  DN rates at -30 deg: " + " ".join(f"{k}={v:.0f}" for k, v in c["A"]["-30.0"]["rates_mean"].items()))
        print("  per-tick (50 ms) rate SD at +30: " + " ".join(f"{k}={v:.0f}" for k, v in c["A"]["30.0"]["rates_tick_sd"].items()))
        print(f"B: pass={c['B_pass']} (monotone {c['B_monotone']}, range significant {c['B_range_significant']}, "
              f"change > 10 % {c['B_effect_ok']}, direction {c['B_direction']:+.0f})")
        for sstr, bb in c["B"].items():
            print(f"  s={float(sstr):.1f}: total DN {bb['total']['mean']:.1f} +- {bb['total']['sd']:.1f} Hz; pooled R-L {bb['pooled']['mean']:.1f}")


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flyfollow.audit.audit", description="G0 connectome audit (plan 4.3)")
    p.add_argument("--brain", nargs="+", default=[], help="pursuit core .npz file(s) for the functional audit")
    p.add_argument("--full", default=None, help="full brain for the structural audit (default data/brains/malecns_full.npz)")
    p.add_argument("--config", default=None)
    p.add_argument("--structural-only", action="store_true")
    p.add_argument("--functional-only", action="store_true")
    p.add_argument("--lc10-only", action="store_true", help="functional audit drives LC10a (and P1) only, no LC9/LC11")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--rmax", type=float, default=150.0, help="audit encoder peak LC10a rate (Hz)")
    p.add_argument("--sigma", type=float, default=10.0, help="audit encoder tuning width (deg) at s = 1")
    p.add_argument("--overlap", type=float, default=10.0, help="frontal overlap (deg): each side's bin 0 sits this far across the midline")
    p.add_argument("--arousal-rate", type=float, default=20.0, help="tonic P1 rate (Hz) when arousal is on")
    p.add_argument("--aux-rate", type=float, default=10.0,
                   help="LC9 / LC11 rate (Hz) on the spot side at peak tuning (50 Hz swamps LC10a, see docs/audit_result.md)")
    p.add_argument("--bearings", type=float, nargs="+", default=list(BEARINGS_DEG))
    p.add_argument("--tag", default="", help="suffix for the output JSON name")
    p.add_argument("--out-dir", default=str(data_root() / "audit"))
    args = p.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.functional_only:
        full = Connectome.load(args.full or full_path())
        resolve(full, load_config(args.config))
        s = structural(full)
        print_structural(s)
        (out_dir / "audit_structural.json").write_text(json.dumps(s, indent=1, default=_default))
        del full
    if not args.structural_only:
        for b in args.brain:
            path = Path(b)
            kw = {"r_max": args.rmax, "sigma_deg": args.sigma, "overlap_deg": args.overlap, "arousal_rate": args.arousal_rate,
                  "aux_rate": args.aux_rate}
            f = functional(path, args.workers, "enc_lc10" if args.lc10_only else "enc", kw, tuple(args.bearings))
            print_functional(f)
            suffix = ("_lc10only" if args.lc10_only else "") + (f"_{args.tag}" if args.tag else "")
            (out_dir / f"audit_{path.stem}{suffix}.json").write_text(json.dumps(f, indent=1, default=_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
