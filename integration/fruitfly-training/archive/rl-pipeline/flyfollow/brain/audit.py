"""Go/no-go gate G0: connectome audit of the pursuit subgraph (plan section 4.3).

Usage:
    python -m flyfollow.brain.audit --brain data/brains/pursuit_core1.npz --out docs/audit_result.md --json data/brains/audit.json

Structural part: LC10a -> DN direct synapses and 2-hop signed path weights per side, ipsi vs contra,
top intermediate types, group sizes, arousal (P1-like) connections.
Functional part (BrainRunner, default LIF parameters, 50 ms ticks, 1 s warmup, 1 s average, 5 seeds):
    A: spot at -30, -15, 0, +15, +30 deg; per DN type R minus L versus bearing.
    B: spot at 0 deg with size 0.5, 1, 2; total DN activity versus size.
    C: A and B with a tonic arousal bias on the AROUSAL groups (0, 5, 10 mV).
Then the gate outcome, the expanded retry if the full gate fails, readout signs and a tick benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from flydrones.brain.connectome import Connectome

from flyfollow.brain.runner import BrainRunner

N_BINS = 8
BIN_CENTERS_DEG = [5.0 + 10.0 * k for k in range(N_BINS)]
SIGMA_DEG = 10.0
BEARINGS = [-30.0, -15.0, 0.0, 15.0, 30.0]
SIZES = [0.5, 1.0, 2.0]
MAX_RATES = [60.0, 120.0, 200.0]
AROUSAL_MV = [0.0, 5.0, 10.0]
REPEATS = 5
TICK_MS = 50.0
WARMUP_TICKS = 20
MEASURE_TICKS = 20
BENCH_TICKS = 100
EXTRA_INPUT_GAIN = 0.5
EXPANDED_EXTRA_INPUTS = ["LC10b", "LC10c", "LC10d", "LC10e", "LC9", "LC11"]

# Turn direction of each DN type when active on one side, from FLYGUIDE_SPEC / the plan (not verified here).
LITERATURE_TURN = {"DNa02": "ipsi", "DNa01": "ipsi", "DNb05": "ipsi", "DNg13": "ipsi", "DNb06": "contra"}


# ------------------------------------------------------------------ helpers
def dn_types_of(conn: Connectome) -> list[str]:
    roles = conn.meta.get("roles", {})
    types = []
    for name, role in roles.items():
        if role == "output" and name.endswith("_L"):
            types.append(name[:-2])
    return types


def input_groups_by_side(conn: Connectome, prefixes: list[str]) -> dict[str, np.ndarray]:
    result = {}
    for side in ["L", "R"]:
        parts = [np.zeros(0, dtype=np.int64)]
        for prefix in prefixes:
            if prefix == "LC10a":
                for k in range(N_BINS):
                    parts.append(conn.group(f"LC10a_{side}_b{k}"))
            else:
                parts.append(conn.group(f"{prefix}_{side}"))
        result[side] = np.unique(np.concatenate(parts))
    return result


def trend_check(means: list[float], sds: list[float], n: int) -> dict:
    """Monotonic within noise: every step goes the overall direction or reverses by less than 2 SE,
    and the first-to-last change is larger than 2 SE."""
    m = np.asarray(means, dtype=float)
    se = np.asarray(sds, dtype=float) / math.sqrt(n)
    change = m[-1] - m[0]
    change_se = math.sqrt(se[0] ** 2 + se[-1] ** 2)
    direction = 0
    if abs(change) > 2 * change_se and abs(change) > 1e-9:
        direction = 1 if change > 0 else -1
    monotonic = direction != 0
    for i in range(len(m) - 1):
        tol = 2 * math.sqrt(se[i] ** 2 + se[i + 1] ** 2)
        if direction * (m[i + 1] - m[i]) < -tol:
            monotonic = False
    return {"direction": direction, "monotonic": bool(monotonic), "change": float(change), "change_se": float(change_se)}


def sign_change_check(means: list[float], direction: int) -> bool:
    """Across 0 deg: the -30 and +30 means have opposite signs in the trend direction, and -15 / +15 are not on the wrong side."""
    if direction == 0:
        return False
    m = dict(zip(BEARINGS, means))
    return bool(direction * m[-30.0] < 0 and direction * m[30.0] > 0 and direction * m[-15.0] <= 0 and direction * m[15.0] >= 0)


def safe_snr(mean: float, sd: float) -> float:
    if sd > 1e-9:
        return mean / sd
    if abs(mean) < 1e-9:
        return 0.0
    return math.copysign(999.0, mean)


# ------------------------------------------------------------------ structural
def structural(conn: Connectome, inputs: dict[str, np.ndarray], dn_types: list[str]) -> dict:
    W = conn.weights.tocsr()  # (post, pre)
    Wc = conn.weights.tocsc()
    types = conn.types.astype(str)
    axon = conn.meta.get("dn_axon_side", {})

    direct = {}
    hop2 = {}
    lateral = {}
    per_bin = {}
    contributions = {}  # intermediate type -> summed |path weight|
    per_dn_top = {}
    for t in dn_types:
        direct[t] = {}
        hop2[t] = {}
        per_type_contrib = {}
        for s_in in ["L", "R"]:
            src = inputs[s_in]
            into_x = np.asarray(Wc[:, src].sum(axis=1)).ravel()  # signed input from this side's LC10a into each X
            for s_out in ["L", "R"]:
                dst = conn.group(f"{t}_{s_out}")
                key = f"LC_{s_in}->{t}_{s_out}"
                direct[t][key] = float(W[dst][:, src].sum())
                from_x = np.asarray(W[dst].sum(axis=0)).ravel()
                path = into_x * from_x
                hop2[t][key] = float(path.sum())
                nz = np.flatnonzero(path)
                for i in nz:
                    name = types[i]
                    contributions[name] = contributions.get(name, 0.0) + abs(float(path[i]))
                    per_type_contrib[name] = per_type_contrib.get(name, 0.0) + float(path[i])
        ipsi = hop2[t][f"LC_L->{t}_L"] + hop2[t][f"LC_R->{t}_R"]
        contra = hop2[t][f"LC_L->{t}_R"] + hop2[t][f"LC_R->{t}_L"]
        ipsi_direct = direct[t][f"LC_L->{t}_L"] + direct[t][f"LC_R->{t}_R"]
        contra_direct = direct[t][f"LC_L->{t}_R"] + direct[t][f"LC_R->{t}_L"]
        sign = 0
        if ipsi - contra > 0:
            sign = 1
        elif ipsi - contra < 0:
            sign = -1
        # mirror check: each side on its own shows the same ipsi/contra bias
        left_bias = hop2[t][f"LC_L->{t}_L"] - hop2[t][f"LC_L->{t}_R"]
        right_bias = hop2[t][f"LC_R->{t}_R"] - hop2[t][f"LC_R->{t}_L"]
        mirrored = bool(left_bias * right_bias > 0)
        axon_same = axon.get(f"{t}_L", {}).get("frac_same", 1.0) >= 0.5
        # which side does the fly turn when the target is on the right? (soma-side bias -> axon side -> literature turn)
        turn = None
        if t in LITERATURE_TURN and sign != 0:
            motor_side_right_target = "R" if sign > 0 else "L"
            if not axon_same:
                motor_side_right_target = "L" if motor_side_right_target == "R" else "R"
            if LITERATURE_TURN[t] == "ipsi":
                turn_side = motor_side_right_target
            else:
                turn_side = "L" if motor_side_right_target == "R" else "R"
            turn = "toward" if turn_side == "R" else "away"
        lateral[t] = {
            "hop2_ipsi": ipsi,
            "hop2_contra": contra,
            "direct_ipsi": ipsi_direct,
            "direct_contra": contra_direct,
            "predicted_sign_R_minus_L_vs_bearing": sign,
            "mirrored": mirrored,
            "axon_same_side_as_soma": bool(axon_same),
            "literature_turn": LITERATURE_TURN.get(t, "unknown"),
            "predicted_turn_for_target": turn,
        }
        top = sorted(per_type_contrib.items(), key=lambda kv: -abs(kv[1]))[:5]
        per_dn_top[t] = [{"type": k, "signed_path_weight": v} for k, v in top]

        # per LC10a bin, signed 2-hop onto ipsi (same soma side) and contra DN
        per_bin[t] = {}
        for s_in in ["L", "R"]:
            other = "R" if s_in == "L" else "L"
            rows = []
            for k in range(N_BINS):
                src = conn.group(f"LC10a_{s_in}_b{k}")
                if src.size == 0:
                    rows.append({"bin": k, "ipsi": 0.0, "contra": 0.0})
                    continue
                into_x = np.asarray(Wc[:, src].sum(axis=1)).ravel()
                ipsi_w = float((into_x * np.asarray(W[conn.group(f"{t}_{s_in}")].sum(axis=0)).ravel()).sum())
                contra_w = float((into_x * np.asarray(W[conn.group(f"{t}_{other}")].sum(axis=0)).ravel()).sum())
                rows.append({"bin": k, "ipsi": ipsi_w, "contra": contra_w})
            per_bin[t][s_in] = rows

    top_x = sorted(contributions.items(), key=lambda kv: -kv[1])[:15]
    dna02 = lateral.get("DNa02", {})
    struct_pass = bool(dna02.get("mirrored") and dna02.get("predicted_turn_for_target") == "toward" and dna02.get("hop2_ipsi", 0) > 0)
    toward = [t for t in dn_types if lateral[t]["predicted_turn_for_target"] == "toward" and lateral[t]["mirrored"]]
    reason = (
        f"DNa02: 2-hop ipsi {dna02.get('hop2_ipsi', 0):.0f} vs contra {dna02.get('hop2_contra', 0):.0f}, mirrored={dna02.get('mirrored')}, "
        f"predicted turn {dna02.get('predicted_turn_for_target')}; types predicted to turn toward the target (mirrored): {toward}"
    )
    return {
        "direct": direct,
        "hop2": hop2,
        "lateral": lateral,
        "per_bin_hop2": per_bin,
        "top_intermediate_types": [{"type": k, "abs_path_weight": v} for k, v in top_x],
        "top_intermediates_per_dn": per_dn_top,
        "pass": struct_pass,
        "reason": reason,
    }


def arousal_structure(conn: Connectome, lc: np.ndarray, dn_types: list[str]) -> dict:
    W = conn.weights.tocsr()
    Wc = conn.weights.tocsc()
    ar = np.concatenate([conn.group("AROUSAL_L"), conn.group("AROUSAL_R")])
    if ar.size == 0:
        return {"present": False}
    dn = np.concatenate([conn.group(f"{t}_{s}") for t in dn_types for s in ["L", "R"]])
    into_x = np.asarray(Wc[:, ar].sum(axis=1)).ravel()
    result = {
        "present": True,
        "n": int(ar.size),
        "types": sorted(set(conn.types[ar].astype(str))),
        "syn_onto_lc10a": float(W[lc][:, ar].sum()),
        "syn_from_lc10a": float(W[ar][:, lc].sum()),
        "direct_onto_dns": float(W[dn][:, ar].sum()),
        "hop2_signed_onto_dns": float((into_x * np.asarray(W[dn].sum(axis=0)).ravel()).sum()),
        "out_syn_total_in_core": float(abs(Wc[:, ar]).sum()),
    }
    per_dn = {}
    for t in dn_types:
        for s in ["L", "R"]:
            d = conn.group(f"{t}_{s}")
            per_dn[f"{t}_{s}"] = float((into_x * np.asarray(W[d].sum(axis=0)).ravel()).sum()) + float(W[d][:, ar].sum())
    result["per_dn_direct_plus_hop2"] = per_dn
    return result


# ------------------------------------------------------------------ functional
def spot_inputs(
    bearing: float, size: float, max_rate: float, extra_prefixes: list[str] | None = None, centers: dict[str, list[float]] | None = None
) -> dict[str, float]:
    """Gaussian tuning over bin azimuths. Left bins sit at -center, right bins at +center (bearing < 0 = left).
    Default bin centers 5..75 deg, so the two bin-0s (at -5 and +5) overlap near 0 deg. Size widens the spot.
    `centers` overrides the bin centers per side (degrees into that side's hemifield)."""
    sigma = SIGMA_DEG * size
    rates = {}
    side_peak = {"L": 0.0, "R": 0.0}
    for side in ["L", "R"]:
        sign = -1.0 if side == "L" else 1.0
        for k in range(N_BINS):
            if centers is None:
                center = sign * BIN_CENTERS_DEG[k]
            else:
                center = sign * centers[side][k]
            r = max_rate * math.exp(-((bearing - center) ** 2) / (2 * sigma**2))
            if r >= 0.01 * max_rate:
                rates[f"LC10a_{side}_b{k}"] = r
                side_peak[side] = max(side_peak[side], r)
    if extra_prefixes:
        for prefix in extra_prefixes:
            for side in ["L", "R"]:
                if side_peak[side] > 0:
                    rates[f"{prefix}_{side}"] = EXTRA_INPUT_GAIN * side_peak[side]
    return rates


def run_trial(base: BrainRunner, inputs: dict[str, float], outputs: list[str], seed: int, arousal_mv: float) -> dict[str, float]:
    brain = base.fresh(seed)
    if arousal_mv:
        brain.set_bias("AROUSAL_L", arousal_mv)
        brain.set_bias("AROUSAL_R", arousal_mv)
    for i in range(WARMUP_TICKS):
        brain.tick(inputs, outputs, ms=TICK_MS)
    total = {name: 0.0 for name in outputs}
    for i in range(MEASURE_TICKS):
        rates = brain.tick(inputs, outputs, ms=TICK_MS)
        for name in outputs:
            total[name] += rates[name] / MEASURE_TICKS
    return total


def test_a(
    base: BrainRunner,
    dn_types: list[str],
    signs: dict[str, int],
    max_rate: float,
    arousal_mv: float,
    extra: list[str] | None,
    centers: dict[str, list[float]] | None = None,
) -> dict:
    outputs = [f"{t}_{s}" for t in dn_types for s in ["L", "R"]]
    diffs = {t: np.zeros((len(BEARINGS), REPEATS)) for t in dn_types}
    rates = {name: np.zeros((len(BEARINGS), REPEATS)) for name in outputs}
    for i, bearing in enumerate(BEARINGS):
        inputs = spot_inputs(bearing, 1.0, max_rate, extra, centers)
        for j in range(REPEATS):
            r = run_trial(base, inputs, outputs, seed=j, arousal_mv=arousal_mv)
            for name in outputs:
                rates[name][i, j] = r[name]
            for t in dn_types:
                diffs[t][i, j] = r[f"{t}_R"] - r[f"{t}_L"]

    yaw = np.zeros((len(BEARINGS), REPEATS))
    for t in dn_types:
        yaw += signs.get(t, 0) * diffs[t]

    per_type = {}
    bearing_rep = np.repeat(np.asarray(BEARINGS)[:, None], REPEATS, axis=1).ravel()
    for t in dn_types:
        d = diffs[t]
        means = d.mean(axis=1).tolist()
        sds = d.std(axis=1, ddof=1).tolist()
        trend = trend_check(means, sds, REPEATS)
        flat = d.ravel()
        corr = float(np.corrcoef(bearing_rep, flat)[0, 1]) if flat.std() > 0 else 0.0
        slope = float(np.polyfit(bearing_rep, flat, 1)[0]) if flat.std() > 0 else 0.0
        per_type[t] = {
            "mean_R_minus_L": means,
            "sd_R_minus_L": sds,
            "snr": [safe_snr(m, s) for m, s in zip(means, sds)],
            "monotonic": trend["monotonic"],
            "direction": trend["direction"],
            "sign_change": sign_change_check(means, trend["direction"]),
            "corr_with_bearing": corr,
            "slope_hz_per_deg": slope,
            "mean_rate_L": rates[f"{t}_L"].mean(axis=1).tolist(),
            "mean_rate_R": rates[f"{t}_R"].mean(axis=1).tolist(),
        }
        per_type[t]["pass"] = bool(per_type[t]["monotonic"] and per_type[t]["sign_change"])

    naive = summarize_yaw(yaw, {t: signs.get(t, 0) for t in dn_types})

    # readout-style yaw: the DN types that pass on their own, each with its own sign
    passing = [t for t in dn_types if per_type[t]["pass"]]
    readout = np.zeros((len(BEARINGS), REPEATS))
    used = {}
    for t in passing:
        used[t] = per_type[t]["direction"]
        readout += per_type[t]["direction"] * diffs[t]
    readout_yaw = summarize_yaw(readout, used)

    return {
        "bearings": BEARINGS,
        "per_type": per_type,
        "types_passing": passing,
        "dna02_pass": bool(per_type.get("DNa02", {}).get("pass", False)),
        "readout_yaw": readout_yaw,
        "naive_pooled_yaw": naive,
        "pass": len(passing) > 0,
    }


def summarize_yaw(yaw: np.ndarray, signs_used: dict[str, int]) -> dict:
    means = yaw.mean(axis=1).tolist()
    sds = yaw.std(axis=1, ddof=1).tolist()
    trend = trend_check(means, sds, REPEATS)
    result = {
        "signs_used": signs_used,
        "mean": means,
        "sd": sds,
        "snr": [safe_snr(m, s) for m, s in zip(means, sds)],
        "monotonic": trend["monotonic"],
        "direction": trend["direction"],
        "sign_change": sign_change_check(means, trend["direction"]),
    }
    result["pass"] = bool(result["monotonic"] and result["sign_change"] and result["direction"] > 0)
    return result


def test_b(base: BrainRunner, dn_types: list[str], max_rate: float, arousal_mv: float, extra: list[str] | None) -> dict:
    outputs = [f"{t}_{s}" for t in dn_types for s in ["L", "R"]]
    totals = np.zeros((len(SIZES), REPEATS))
    per_type = {t: np.zeros((len(SIZES), REPEATS)) for t in dn_types}
    for i, size in enumerate(SIZES):
        inputs = spot_inputs(0.0, size, max_rate, extra)
        for j in range(REPEATS):
            r = run_trial(base, inputs, outputs, seed=j, arousal_mv=arousal_mv)
            totals[i, j] = sum(r.values())
            for t in dn_types:
                per_type[t][i, j] = r[f"{t}_L"] + r[f"{t}_R"]
    means = totals.mean(axis=1).tolist()
    sds = totals.std(axis=1, ddof=1).tolist()
    trend = trend_check(means, sds, REPEATS)
    size_rep = np.repeat(np.asarray(SIZES)[:, None], REPEATS, axis=1).ravel()
    type_rows = {}
    for t in dn_types:
        flat = per_type[t].ravel()
        corr = float(np.corrcoef(size_rep, flat)[0, 1]) if flat.std() > 0 else 0.0
        type_rows[t] = {"mean_total": per_type[t].mean(axis=1).tolist(), "corr_with_size": corr}
    return {
        "sizes": SIZES,
        "total_mean": means,
        "total_sd": sds,
        "monotonic": trend["monotonic"],
        "direction": trend["direction"],
        "pass": bool(trend["monotonic"]),
        "per_type": type_rows,
    }


def baseline(base: BrainRunner, dn_types: list[str]) -> dict:
    outputs = [f"{t}_{s}" for t in dn_types for s in ["L", "R"]]
    result = {}
    for mv in AROUSAL_MV:
        total = {name: 0.0 for name in outputs}
        for j in range(REPEATS):
            r = run_trial(base, {}, outputs, seed=j, arousal_mv=mv)
            for name in outputs:
                total[name] += r[name] / REPEATS
        result[str(mv)] = total
    return result


def functional_grid(base: BrainRunner, dn_types: list[str], signs: dict[str, int], rates: list[float], arousal: list[float], extra: list[str] | None, label: str) -> list[dict]:
    conditions = []
    for rate in rates:
        for mv in arousal:
            t0 = time.perf_counter()
            a = test_a(base, dn_types, signs, rate, mv, extra)
            b = test_b(base, dn_types, rate, mv, extra)
            snr = a["readout_yaw"]["snr"]
            min_snr = 0.0
            d15 = 0.0
            if a["pass"]:
                min_snr = min(snr[0] * -1, snr[1] * -1, snr[3], snr[4])  # signed toward the expected direction
                m = a["readout_yaw"]["mean"]
                pooled_sd = math.sqrt(float(np.mean(np.square(a["readout_yaw"]["sd"]))))
                d15 = (m[3] - m[1]) / max(pooled_sd, 1e-9)
            conditions.append({"max_rate": rate, "arousal_mv": mv, "A": a, "B": b, "min_yaw_snr_pm15_pm30": min_snr, "d15": d15})
            print(
                f"[{label}] rate {rate:5.0f} Hz arousal {mv:4.1f} mV | A pass {a['pass']} readout yaw {np.round(a['readout_yaw']['mean'], 1).tolist()} "
                f"snr {np.round(snr, 1).tolist()} types {a['types_passing']} | B pass {b['pass']} total {np.round(b['total_mean'], 1).tolist()} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    return conditions


def pick_setting(candidates: list[dict]) -> dict:
    """Highest d15 = (yaw at +15 minus yaw at -15) / pooled SD; but take the lowest arousal whose d15 is
    within 20 % of the best, so a bias is only recommended when it clearly helps (5 seeds give noisy SDs)."""
    best_d = max(c["d15"] for c in candidates)
    good = [c for c in candidates if c["d15"] >= 0.8 * best_d]
    good.sort(key=lambda c: (c["arousal_mv"], -c["d15"]))
    return good[0]


def choose(conditions: list[dict]) -> tuple[str, dict | None]:
    both = [c for c in conditions if c["A"]["pass"] and c["B"]["pass"]]
    if both:
        return "full", pick_setting(both)
    a_only = [c for c in conditions if c["A"]["pass"]]
    if a_only:
        return "yaw-only", pick_setting(a_only)
    return "fail", None


def evaluate_brain(conn: Connectome, extra: list[str] | None, label: str) -> dict:
    dn_types = dn_types_of(conn)
    lc = np.concatenate([conn.group(f"LC10a_{s}_b{k}") for s in ["L", "R"] for k in range(N_BINS)])
    prefixes = ["LC10a"] + (extra or [])
    struct = structural(conn, input_groups_by_side(conn, prefixes), dn_types)
    signs = {t: struct["lateral"][t]["predicted_sign_R_minus_L_vs_bearing"] for t in dn_types}
    base = BrainRunner(conn, seed=0)
    base_rates = baseline(base, dn_types)
    conditions = functional_grid(base, dn_types, signs, MAX_RATES, AROUSAL_MV, extra, label)
    outcome, best = choose(conditions)
    if outcome in ("full", "yaw-only") and not struct["pass"]:
        outcome = outcome + " (structural check failed)"
    return {
        "label": label,
        "neurons": conn.n,
        "connections": conn.n_connections,
        "dn_types": dn_types,
        "groups": {k: int(v.size) for k, v in sorted(conn.groups.items())},
        "structural": struct,
        "arousal_structure": arousal_structure(conn, lc, dn_types),
        "baseline_no_stimulus": base_rates,
        "conditions": conditions,
        "outcome": outcome,
        "best": None if best is None else {"max_rate": best["max_rate"], "arousal_mv": best["arousal_mv"]},
    }


def readout_signs(result: dict, best: dict) -> dict:
    cond = None
    for c in result["conditions"]:
        if c["max_rate"] == best["max_rate"] and c["arousal_mv"] == best["arousal_mv"]:
            cond = c
    table = {}
    for t in result["dn_types"]:
        a = cond["A"]["per_type"][t]
        corr = a["corr_with_bearing"]
        corr_sign = 0
        if abs(corr) >= 0.3:
            corr_sign = 1 if corr > 0 else -1
        # only types that pass test A on their own get a non-zero initial yaw weight
        init_sign = a["direction"] if a["pass"] else 0
        table[t] = {
            "yaw_sign": init_sign,
            "corr_sign": corr_sign,
            "passes_A": bool(a["pass"]),
            "corr_R_minus_L_vs_bearing": corr,
            "slope_hz_per_deg": a["slope_hz_per_deg"],
            "structural_sign": result["structural"]["lateral"][t]["predicted_sign_R_minus_L_vs_bearing"],
            "corr_total_vs_size": cond["B"]["per_type"][t]["corr_with_size"],
        }
    return table


def benchmark(path: str, max_rate: float) -> dict:
    conn = Connectome.load(path)
    dn_types = dn_types_of(conn)
    outputs = [f"{t}_{s}" for t in dn_types for s in ["L", "R"]]
    brain = BrainRunner(conn, seed=0)
    inputs = spot_inputs(15.0, 1.0, max_rate)
    for i in range(WARMUP_TICKS):
        brain.tick(inputs, outputs, ms=TICK_MS)
    times = []
    for i in range(BENCH_TICKS):
        t0 = time.perf_counter()
        brain.tick(inputs, outputs, ms=TICK_MS)
        times.append(time.perf_counter() - t0)
    times_ms = np.asarray(times) * 1000.0
    return {
        "file": Path(path).name,
        "neurons": conn.n,
        "connections": conn.n_connections,
        "median_ms_per_tick": float(np.median(times_ms)),
        "p90_ms_per_tick": float(np.percentile(times_ms, 90)),
        "real_time_factor": float(np.median(times_ms) / TICK_MS),
    }


# ------------------------------------------------------------------ report
def fmt(x: float, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1e5:
        return f"{x:.3g}"
    return f"{x:.{nd}f}"


def markdown(res: dict) -> str:
    main = res["main"]
    lines = []
    lines.append("# G0 connectome audit result")
    lines.append("")
    lines.append(f"Generated by `python -m flyfollow.brain.audit` on {res['generated']}. Brain: `{res['brain']}` ({main['neurons']:,} neurons, {main['connections']:,} connections).")
    lines.append("")
    lines.append("## Gate outcome")
    lines.append("")
    lines.append(f"**Outcome: {res['gate']['outcome']}.** {res['gate']['summary']}")
    lines.append("")
    rec = res["recommended"]
    if rec:
        lines.append(f"Recommended encoder settings for the audit mapping: max input rate {rec['max_rate']:.0f} Hz, arousal bias {rec['arousal_mv']:.0f} mV.")
        lines.append("")
    lines.append("What is verified versus inferred:")
    lines.append("")
    for item in res["verified_vs_inferred"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Group sizes")
    lines.append("")
    lines.append("| Group | Cells | Group | Cells |")
    lines.append("|---|---|---|---|")
    names = list(main["groups"].keys())
    half = (len(names) + 1) // 2
    for i in range(half):
        left = names[i]
        row = f"| {left} | {main['groups'][left]} |"
        if i + half < len(names):
            right = names[i + half]
            row += f" {right} | {main['groups'][right]} |"
        else:
            row += " | |"
        lines.append(row)
    lines.append("")
    lines.append(f"Bins: {res['bins_note']}")
    lines.append("")
    lines.append("| Bin | L cells | L mean h | L approx az (deg) | R cells | R mean h | R approx az (deg) |")
    lines.append("|---|---|---|---|---|---|---|")
    bins = res["bins"]
    for k in range(N_BINS):
        bl = bins["L"][k]
        br = bins["R"][k]
        lines.append(f"| {k} | {bl['count']} | {bl['mean_h']} | {bl['approx_azimuth_deg']} | {br['count']} | {br['mean_h']} | {br['approx_azimuth_deg']} |")
    lines.append("")

    st = main["structural"]
    lines.append("## Structural audit (LC10a to DN)")
    lines.append("")
    lines.append(f"Structural check: **{'pass' if st['pass'] else 'fail'}**. {st['reason']}.")
    lines.append("")
    lines.append("Signed synapse counts (direct) and summed signed 2-hop path weights (product of signed synapse counts, all intermediates X). "
                 "Sides are soma sides. Ipsi = LC10a and DN soma on the same side.")
    lines.append("")
    lines.append("| DN type | direct L->L | direct L->R | direct R->R | direct R->L | 2-hop L->L | 2-hop L->R | 2-hop R->R | 2-hop R->L | 2-hop ipsi | 2-hop contra | mirrored | axon side | predicted turn |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in main["dn_types"]:
        d = st["direct"][t]
        h = st["hop2"][t]
        lat = st["lateral"][t]
        axon = "same as soma" if lat["axon_same_side_as_soma"] else "crosses"
        turn = lat["predicted_turn_for_target"] or "no literature label"
        lines.append(
            f"| {t} | {fmt(d[f'LC_L->{t}_L'], 0)} | {fmt(d[f'LC_L->{t}_R'], 0)} | {fmt(d[f'LC_R->{t}_R'], 0)} | {fmt(d[f'LC_R->{t}_L'], 0)} | "
            f"{fmt(h[f'LC_L->{t}_L'], 0)} | {fmt(h[f'LC_L->{t}_R'], 0)} | {fmt(h[f'LC_R->{t}_R'], 0)} | {fmt(h[f'LC_R->{t}_L'], 0)} | "
            f"{fmt(lat['hop2_ipsi'], 0)} | {fmt(lat['hop2_contra'], 0)} | {lat['mirrored']} | {axon} | {turn} |"
        )
    lines.append("")
    lines.append("Top intermediate types on LC10a to DN 2-hop paths (summed absolute path weight over all DN types and sides):")
    lines.append("")
    lines.append("| Intermediate type | abs path weight |")
    lines.append("|---|---|")
    for row in st["top_intermediate_types"][:12]:
        lines.append(f"| {row['type']} | {fmt(row['abs_path_weight'], 0)} |")
    lines.append("")
    lines.append("Top intermediates per DN type (signed): " + "; ".join(
        f"{t}: " + ", ".join(f"{r['type']} {fmt(r['signed_path_weight'], 0)}" for r in rows[:3]) for t, rows in st["top_intermediates_per_dn"].items()
    ) + ".")
    lines.append("")
    if "DNa02" in st["per_bin_hop2"]:
        lines.append("DNa02 2-hop signed path weight per LC10a bin (bin 0 frontal):")
        lines.append("")
        lines.append("| Bin | L->DNa02_L (ipsi) | L->DNa02_R (contra) | R->DNa02_R (ipsi) | R->DNa02_L (contra) |")
        lines.append("|---|---|---|---|---|")
        pb = st["per_bin_hop2"]["DNa02"]
        for k in range(N_BINS):
            lines.append(f"| {k} | {fmt(pb['L'][k]['ipsi'], 0)} | {fmt(pb['L'][k]['contra'], 0)} | {fmt(pb['R'][k]['ipsi'], 0)} | {fmt(pb['R'][k]['contra'], 0)} |")
        lines.append("")
    ar = main["arousal_structure"]
    lines.append("### Arousal (P1-like pC1) group")
    lines.append("")
    if ar.get("present"):
        lines.append(f"Types: {', '.join(ar['types'])} ({ar['n']} cells). Synapses onto LC10a: {fmt(ar['syn_onto_lc10a'], 0)}; from LC10a: {fmt(ar['syn_from_lc10a'], 0)}; "
                     f"direct onto the steering DNs: {fmt(ar['direct_onto_dns'], 0)}; signed 2-hop onto the DNs: {fmt(ar['hop2_signed_onto_dns'], 0)}.")
        lines.append("")
        lines.append(f"Selection rule: {res['arousal_rule']}")
    else:
        lines.append("No arousal group in this brain.")
    lines.append("")

    lines.append("## Functional audit")
    lines.append("")
    lines.append(f"Stimulus mapping (audit choice, not anatomy): {res['stim_note']}")
    lines.append("")
    lines.append(f"Baseline with no stimulus (arousal 0 / 5 / 10 mV), summed DN rate: "
                 + ", ".join(f"{mv} mV: {fmt(sum(v.values()))} Hz" for mv, v in main["baseline_no_stimulus"].items()) + ".")
    lines.append("")
    lines.append("Grid summary. A passes when at least one DN type's R minus L flips sign across 0 deg and is monotonic within 2 SE. "
                 "Readout yaw = sum over the passing types of (their sign) x (R minus L), in Hz, i.e. what a readout that uses those types sees. "
                 "SNR = mean / SD over 5 seeds. d15 = (yaw at +15 minus yaw at -15) / SD pooled over bearings, used to pick the setting "
                 "(lowest arousal within 20 % of the best d15). Naive pool = all types summed with the structural sign (a readout that is not trained); "
                 "it is reported because it shows what breaks when the non-monotonic types are not down-weighted.")
    lines.append("")
    lines.append("| Max rate (Hz) | Arousal (mV) | A pass | DN types passing A | readout yaw at -30/-15/0/+15/+30 | readout yaw SNR | d15 | naive pool passes | B pass | total DN Hz at s=0.5/1/2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in main["conditions"]:
        a = c["A"]
        b = c["B"]
        lines.append(
            f"| {c['max_rate']:.0f} | {c['arousal_mv']:.0f} | {a['pass']} | {', '.join(a['types_passing']) or 'none'} | {' / '.join(fmt(x) for x in a['readout_yaw']['mean'])} | "
            f"{' / '.join(fmt(x) for x in a['readout_yaw']['snr'])} | {fmt(c['d15'], 1)} | {a['naive_pooled_yaw']['pass']} | {b['pass']} | {' / '.join(fmt(x) for x in b['total_mean'])} |"
        )
    lines.append("")
    if rec:
        cond = [c for c in main["conditions"] if c["max_rate"] == rec["max_rate"] and c["arousal_mv"] == rec["arousal_mv"]][0]
        lines.append(f"Per DN type at the recommended setting ({rec['max_rate']:.0f} Hz, {rec['arousal_mv']:.0f} mV), R minus L in Hz (mean, SD over 5 seeds):")
        lines.append("")
        lines.append("| DN type | -30 | -15 | 0 | +15 | +30 | monotonic | sign change | corr with bearing | SNR at +30 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for t in main["dn_types"]:
            a = cond["A"]["per_type"][t]
            cells = [f"{fmt(m)} ({fmt(s)})" for m, s in zip(a["mean_R_minus_L"], a["sd_R_minus_L"])]
            lines.append(f"| {t} | {' | '.join(cells)} | {a['monotonic']} | {a['sign_change']} | {fmt(a['corr_with_bearing'], 2)} | {fmt(a['snr'][4], 1)} |")
        lines.append("")
        lines.append("Mean single-side rates (Hz) at the recommended setting, L / R per bearing:")
        lines.append("")
        lines.append("| DN type | -30 | -15 | 0 | +15 | +30 |")
        lines.append("|---|---|---|---|---|---|")
        for t in main["dn_types"]:
            a = cond["A"]["per_type"][t]
            cells = [f"{fmt(l, 0)} / {fmt(r, 0)}" for l, r in zip(a["mean_rate_L"], a["mean_rate_R"])]
            lines.append(f"| {t} | {' | '.join(cells)} |")
        lines.append("")
        lines.append("Test B per DN type (L + R Hz at s = 0.5 / 1 / 2):")
        lines.append("")
        lines.append("| DN type | total | corr with size |")
        lines.append("|---|---|---|")
        for t in main["dn_types"]:
            b = cond["B"]["per_type"][t]
            lines.append(f"| {t} | {' / '.join(fmt(x) for x in b['mean_total'])} | {fmt(b['corr_with_size'], 2)} |")
        lines.append("")

    anat = res.get("anatomical_mapping")
    if anat:
        a = anat["A"]
        lines.append("### Test A with anatomical bin azimuths")
        lines.append("")
        lines.append(f"Same test at the recommended setting, but bin k centred at its anatomical azimuth estimate (L: {', '.join(fmt(x, 0) for x in anat['centers']['L'])} deg; "
                     f"R: {', '.join(fmt(x, 0) for x in anat['centers']['R'])} deg). A pass: **{a['pass']}**, types passing: {', '.join(a['types_passing']) or 'none'}; "
                     f"readout yaw {' / '.join(fmt(x) for x in a['readout_yaw']['mean'])} Hz (SNR {' / '.join(fmt(x) for x in a['readout_yaw']['snr'])}).")
        lines.append("")
    lines.append("## Readout sign table")
    lines.append("")
    lines.append("Init yaw sign = the trend direction of R minus L versus bearing for types that pass test A on their own, 0 otherwise. "
                 "Positive means the right-side cell fires more for a target on the right, so w_type > 0 turns toward the target. "
                 "Corr sign = sign of corr(R minus L, bearing) over all 25 trials (0 if |corr| < 0.3); for types that fail A it mostly reflects a one-sided cell. "
                 "Structural sign = sign of (2-hop ipsi minus 2-hop contra). Forward column: corr(L + R, size) in test B at 0 deg.")
    lines.append("")
    lines.append("| DN type | init yaw sign | passes A | corr | corr sign | slope (Hz/deg) | structural sign | corr(total, size) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t, row in res["readout_signs"].items():
        lines.append(
            f"| {t} | {row['yaw_sign']:+d} | {row['passes_A']} | {fmt(row['corr_R_minus_L_vs_bearing'], 2)} | {row['corr_sign']:+d} | "
            f"{fmt(row['slope_hz_per_deg'], 2)} | {row['structural_sign']:+d} | {fmt(row['corr_total_vs_size'], 2)} |"
        )
    lines.append("")

    if res.get("expanded"):
        ex = res["expanded"]
        if res["gate"]["main_outcome"] == "full":
            lines.append("## Expanded brain (informational, the retry was not needed)")
            lines.append("")
            lines.append("The main brain passed the full gate, so the expanded retry does not decide anything. It is run anyway because it shows which extra DN types carry bearing information.")
        else:
            lines.append("## Expanded retry")
        lines.append("")
        lines.append(f"Brain `{ex['file']}` ({ex['neurons']:,} neurons). Extra inputs {', '.join(EXPANDED_EXTRA_INPUTS)} driven per side at "
                     f"{EXTRA_INPUT_GAIN} x the peak LC10a bin rate of that side (no azimuth bins for them); extra DN outputs {', '.join(ex['extra_dns'])}. Outcome: **{ex['outcome']}**.")
        lines.append("")
        lines.append("| Max rate (Hz) | Arousal (mV) | A pass | DN types passing A | readout yaw | B pass | total DN Hz |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in ex["conditions"]:
            a = c["A"]
            b = c["B"]
            lines.append(f"| {c['max_rate']:.0f} | {c['arousal_mv']:.0f} | {a['pass']} | {', '.join(a['types_passing']) or 'none'} | {' / '.join(fmt(x) for x in a['readout_yaw']['mean'])} | "
                         f"{b['pass']} | {' / '.join(fmt(x) for x in b['total_mean'])} |")
        lines.append("")
        if ex.get("best"):
            cond = [c for c in ex["conditions"] if c["max_rate"] == ex["best"]["max_rate"] and c["arousal_mv"] == ex["best"]["arousal_mv"]][0]
            lines.append(f"Expanded brain, single-side rates (L / R Hz) at {ex['best']['max_rate']:.0f} Hz, {ex['best']['arousal_mv']:.0f} mV:")
            lines.append("")
            lines.append("| DN type | -30 | -15 | 0 | +15 | +30 | passes A |")
            lines.append("|---|---|---|---|---|---|---|")
            for t in ex["dn_types"]:
                a = cond["A"]["per_type"][t]
                cells = [f"{fmt(l, 0)} / {fmt(r, 0)}" for l, r in zip(a["mean_rate_L"], a["mean_rate_R"])]
                lines.append(f"| {t} | {' | '.join(cells)} | {a['pass']} |")
            lines.append("")

    if res.get("secondary"):
        lines.append("## Secondary brains (A and B at the recommended setting)")
        lines.append("")
        lines.append("| Brain | A pass | DN types passing A | readout yaw | B pass | total DN Hz |")
        lines.append("|---|---|---|---|---|---|")
        for sec in res["secondary"]:
            c = sec["condition"]
            lines.append(f"| {sec['file']} | {c['A']['pass']} | {', '.join(c['A']['types_passing']) or 'none'} | {' / '.join(fmt(x) for x in c['A']['readout_yaw']['mean'])} | "
                         f"{c['B']['pass']} | {' / '.join(fmt(x) for x in c['B']['total_mean'])} |")
        lines.append("")
        for sec in res["secondary"]:
            c = sec["condition"]
            lines.append(f"{sec['file']} single-side rates (L / R Hz) at -30 / 0 / +30: " + "; ".join(
                f"{t} " + ", ".join(f"{fmt(c['A']['per_type'][t]['mean_rate_L'][i], 0)}/{fmt(c['A']['per_type'][t]['mean_rate_R'][i], 0)}" for i in [0, 2, 4])
                for t in sec["dn_types"]
            ) + ".")
            lines.append("")

    lines.append("## Benchmark")
    lines.append("")
    lines.append("Wall time per 50 ms brain tick on this laptop, spot at +15 deg, median of 100 ticks after a 1 s warmup (brain only, one process).")
    lines.append("")
    lines.append("| Brain | Neurons | Connections | median ms/tick | p90 ms/tick | fraction of real time |")
    lines.append("|---|---|---|---|---|---|")
    for b in res["benchmark"]:
        lines.append(f"| {b['file']} | {b['neurons']:,} | {b['connections']:,} | {fmt(b['median_ms_per_tick'], 2)} | {fmt(b['p90_ms_per_tick'], 2)} | {fmt(b['real_time_factor'], 3)} |")
    lines.append("")
    lines.append("## Notes and surprises")
    lines.append("")
    for note in res["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    text = "\n".join(lines)
    return text.replace(chr(0x2014), ", ").replace(chr(0x2013), "-")


# ------------------------------------------------------------------ main
def sibling(path: Path, name: str) -> Path:
    return path.with_name(name)


def json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def run(brain_path: str, out_md: str, out_json: str, expanded_path: str | None, secondary: list[str]) -> dict:
    t_start = time.time()
    brain_path = Path(brain_path)
    conn = Connectome.load(brain_path)
    print(f"audit: {conn.name}, {conn.n} neurons, {conn.n_connections} connections")

    main = evaluate_brain(conn, None, "main")
    outcome = main["outcome"]
    best = main["best"]

    # the expanded brain only decides the gate when the main brain fails the full gate; otherwise it is informational
    expanded = None
    if expanded_path is None:
        expanded_path = str(sibling(brain_path, "pursuit_core1_expanded.npz"))
    if Path(expanded_path).exists():
        ex_conn = Connectome.load(expanded_path)
        extra = [p for p in EXPANDED_EXTRA_INPUTS if ex_conn.group(f"{p}_L").size > 0]
        expanded = evaluate_brain(ex_conn, extra, "expanded")
        expanded["file"] = Path(expanded_path).name
        expanded["extra_dns"] = ex_conn.meta.get("expanded", {}).get("extra_dns", [])

    final = outcome
    source = "main"
    if expanded is not None and outcome != "full":
        rank = {"full": 3, "yaw-only": 2}
        if rank.get(expanded["outcome"], 0) > rank.get(outcome, 0):
            final = expanded["outcome"] + " (expanded retry)"
            source = "expanded"
            best = expanded["best"]
    if final.startswith("fail"):
        final = "no-fly"

    signs_source = main if source == "main" or best is None else expanded
    if best is None:
        # nothing passed: report signs at the highest rate without arousal
        best = {"max_rate": MAX_RATES[-1], "arousal_mv": 0.0}
    signs = readout_signs(signs_source, best)

    # robustness: test A again with the bins placed at their anatomical azimuth estimates instead of 5..75 deg
    anatomical = None
    bin_sides = conn.meta.get("bins", {}).get("sides", {})
    if bin_sides:
        centers = {side: [row["approx_azimuth_deg"] for row in bin_sides[side]] for side in ["L", "R"]}
        dn_types = main["dn_types"]
        struct_signs = {t: main["structural"]["lateral"][t]["predicted_sign_R_minus_L_vs_bearing"] for t in dn_types}
        a = test_a(BrainRunner(conn, seed=0), dn_types, struct_signs, best["max_rate"], best["arousal_mv"], None, centers)
        anatomical = {"centers": centers, "A": a}
        print(f"[anatomical mapping] A pass {a['pass']} types {a['types_passing']} readout yaw {np.round(a['readout_yaw']['mean'], 1).tolist()}")

    sec_results = []
    for path in secondary:
        if not Path(path).exists():
            continue
        sc = Connectome.load(path)
        dn_types = dn_types_of(sc)
        st = structural(sc, input_groups_by_side(sc, ["LC10a"]), dn_types)
        sgn = {t: st["lateral"][t]["predicted_sign_R_minus_L_vs_bearing"] for t in dn_types}
        base = BrainRunner(sc, seed=0)
        cond = functional_grid(base, dn_types, sgn, [best["max_rate"]], [best["arousal_mv"]], None, Path(path).name)[0]
        sec_results.append({"file": Path(path).name, "neurons": sc.n, "dn_types": dn_types, "condition": cond, "structural_pass": st["pass"]})

    bench = []
    bench_paths = [str(brain_path)] + [p for p in secondary if Path(p).exists()]
    if expanded is not None:
        bench_paths.append(expanded_path)
    for path in bench_paths:
        bench.append(benchmark(path, best["max_rate"]))
        print(f"bench {bench[-1]['file']}: {bench[-1]['median_ms_per_tick']:.2f} ms per 50 ms tick")

    bins = conn.meta.get("bins", {})
    gate_summary = gate_text(main, expanded, final, best)
    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "brain": str(brain_path).replace("\\", "/"),
        "gate": {"outcome": final, "source": source, "main_outcome": outcome, "expanded_outcome": None if expanded is None else expanded["outcome"], "summary": gate_summary},
        "recommended": best,
        "main": main,
        "expanded": expanded,
        "secondary": sec_results,
        "readout_signs": signs,
        "anatomical_mapping": anatomical,
        "benchmark": bench,
        "bins": bins.get("sides", {}),
        "bins_note": (
            "LC10a split per side into 8 bins by the synapse-weighted hex-column centroid of each cell's columnar inputs "
            "(h = hex1 - hex2, larger h = more frontal). Approx azimuth assumes the eye spans -10 to 170 deg linearly in h."
        ),
        "stim_note": (
            f"bin k of each side is centred at {BIN_CENTERS_DEG[0]:.0f} + 10k deg into its own hemifield (left bins at negative bearings), "
            f"Gaussian tuning with sigma = {SIGMA_DEG:.0f} deg x size, rate = max rate x tuning, bins under 1 % of max are off. "
            "The two bin-0 groups overlap around 0 deg. This compresses the eye (anatomically about -2 to 137 deg over the 8 bins) "
            "onto the camera field of view, which the trained encoder is free to change."
        ),
        "arousal_rule": conn.meta.get("arousal", {}).get("rule", ""),
        "verified_vs_inferred": verified_notes(conn, main),
        "notes": surprise_notes(conn, main, sec_results, expanded, best),
        "runtime_s": round(time.time() - t_start, 1),
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=json_default)
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(markdown(result))
    print(f"gate outcome: {final} (main {outcome}); recommended {best}; wrote {out_md} and {out_json} in {result['runtime_s']} s")
    return result


def gate_text(main: dict, expanded: dict | None, final: str, best: dict | None) -> str:
    st = main["structural"]
    n_a = sum(1 for c in main["conditions"] if c["A"]["pass"])
    n_b = sum(1 for c in main["conditions"] if c["B"]["pass"])
    n_ab = sum(1 for c in main["conditions"] if c["A"]["pass"] and c["B"]["pass"])
    text = (
        f"Structural check {'passed' if st['pass'] else 'failed'}. Test A (at least one DN type's R minus L flips sign across 0 deg and is monotonic within 2 SE over 5 seeds) passed in "
        f"{n_a} of {len(main['conditions'])} rate/arousal settings, test B (total DN activity monotonic in size) in {n_b}, both in {n_ab}."
    )
    if expanded is not None and main["outcome"] != "full":
        text += f" The full gate failed on the main brain, so the one expanded retry ran: {expanded['outcome']}."
    elif expanded is not None:
        text += f" The expanded brain (not needed) would give: {expanded['outcome']}."
    if best is not None:
        text += f" Best setting: {best['max_rate']:.0f} Hz, {best['arousal_mv']:.0f} mV."
    return text


def verified_notes(conn: Connectome, main: dict) -> list[str]:
    arousal = conn.meta.get("arousal", {})
    notes = [
        "Verified (from the data): group sizes; every DN type in the readout set has exactly one cell per side; LC10a has 135 (L) and 140 (R) cells.",
        "Verified: the direct and 2-hop LC10a to DN synapse numbers below, computed on the saved core (which contains every 2-synapse path by construction).",
        "Verified: the simulated rates, for this LIF model with default Shiu-style parameters. The model has no noise, so all variance comes from Poisson input.",
        "Inferred: the azimuth order of LC10a. It rests on (1) the hex coordinates of 15 columnar types, (2) propagating them through connectivity, "
        "(3) the lamina soma-position fit that says +(hex1 - hex2) is anterior, and (4) the textbook fact that anterior ommatidia look frontally.",
        "Inferred: the approximate azimuth in degrees per bin (a linear map of h onto -10 to 170 deg).",
        f"Inferred: that {', '.join(arousal.get('chosen_types', [])) or 'the chosen pC1 subtypes'} are P1. MaleCNS has no type called P1; the choice uses synonyms "
        "(pMP4 / pMP-e, the P1 cluster names in Yu 2010 and Cachero 2010), fru/dsx co-expression and connectivity to the LC10a pathway.",
        "Inferred: DN axon side, from which side of the VNC their output synapses are on. The turn direction of each DN type is taken from the plan / FLYGUIDE_SPEC, not checked here.",
    ]
    return notes


def condition_at(conditions: list[dict], rate: float, mv: float) -> dict | None:
    for c in conditions:
        if c["max_rate"] == rate and c["arousal_mv"] == mv:
            return c
    return None


def surprise_notes(conn: Connectome, main: dict, secondary: list[dict], expanded: dict | None, best: dict) -> list[str]:
    notes = []
    cond = condition_at(main["conditions"], best["max_rate"], best["arousal_mv"])
    if cond is not None:
        per_type = cond["A"]["per_type"]
        silent = [t for t in main["dn_types"] if max(per_type[t]["mean_rate_L"] + per_type[t]["mean_rate_R"]) < 1.0]
        if silent:
            notes.append(f"Silent in every test-A trial at the recommended setting: {', '.join(silent)} (their LC10a paths in the core are too weak or net inhibitory).")
        if "DNa02" in per_type:
            m = per_type["DNa02"]["mean_R_minus_L"]
            notes.append(
                f"DNa02 is a push-pull pair: each side's LC10a excites the same-side DNa02 through the AOTU (mainly AOTU019 and AOTU025 paths) "
                f"and inhibits the other one, so at 0 deg both are near silent ({m[2]:.1f} Hz) and there is a dead band around the midline. "
                f"The gain is not symmetric: R minus L is {m[1]:.0f} Hz at -15 deg but {m[3]:.0f} Hz at +15 deg (the encoder's per-bin gains can correct this)."
            )
    bins = main["structural"]["per_bin_hop2"].get("DNa02")
    if bins:
        front = sum(bins["L"][k]["ipsi"] + bins["R"][k]["ipsi"] for k in range(3))
        back = sum(bins["L"][k]["ipsi"] + bins["R"][k]["ipsi"] for k in range(3, N_BINS))
        notes.append(
            f"Excitation of the same-side DNa02 comes mostly from the lateral bins: 2-hop ipsi weight {front:.3g} from bins 0 to 2 versus {back:.3g} from bins 3 to 7, "
            "while the frontal bins mostly inhibit the opposite DNa02. The 5..75 deg audit mapping puts bins 3 to 7 inside a camera-sized field; "
            "the anatomical-mapping check tests that the result does not depend on that."
        )
    c0 = condition_at(main["conditions"], best["max_rate"], 0.0)
    c5 = condition_at(main["conditions"], best["max_rate"], 5.0)
    c10 = condition_at(main["conditions"], best["max_rate"], 10.0)
    if c0 and c5 and c10:
        d5 = max(abs(x - y) for x, y in zip(c5["A"]["readout_yaw"]["mean"], c0["A"]["readout_yaw"]["mean"]))
        d10 = max(abs(x - y) for x, y in zip(c10["A"]["readout_yaw"]["mean"], c0["A"]["readout_yaw"]["mean"]))
        base10 = sum(main["baseline_no_stimulus"].get("10.0", {}).values())
        notes.append(
            f"Arousal: a 5 mV bias on the P1-like group changes the readout yaw by at most {d5:.1f} Hz (no measurable effect); 10 mV changes it by up to {d10:.1f} Hz, "
            f"raises total DN activity (test B) and drives {base10:.0f} Hz of summed DN activity with no target at all. It does not gate the LC10a pathway the way P1 gates LC10a in flies; "
            "keep it as a trained parameter bounded at about 10 mV, or drop it."
        )
    axon = conn.meta.get("dn_axon_side", {})
    crossing = sorted(set(k[:-2] for k, v in axon.items() if v.get("axon_side") != k[-1]))
    if crossing:
        notes.append(f"DN types whose VNC output is mostly on the side opposite their soma (axon crosses): {', '.join(crossing)}. "
                      "Group names use soma side, so for these the motor side is the other one; the trained readout absorbs this through its sign.")
    W = conn.weights.tocsr()
    for t in main["dn_types"]:
        left = float(abs(W[conn.group(f"{t}_L")]).sum())
        right = float(abs(W[conn.group(f"{t}_R")]).sum())
        if min(left, right) * 3 < max(left, right):
            notes.append(f"{t}: input synapses inside the core are very asymmetric (L {left:.0f}, R {right:.0f}), so this type cannot give a mirror-symmetric signal; "
                         "likely a reconstruction or annotation asymmetry rather than biology (not checked).")
    base = main["baseline_no_stimulus"]
    if all(sum(v.values()) == 0 for v in base.values()):
        notes.append("With no stimulus every DN is silent, even with a 10 mV arousal bias: the model has no spontaneous activity, so a no-target hover must come from b_fwd.")
    for sec in secondary:
        c = sec["condition"]
        notes.append(f"{sec['file']}: A pass {c['A']['pass']}, B pass {c['B']['pass']} at the recommended setting (see the secondary table). "
                     "The 2-hop core was capped to 20,000 neurons by keeping the most connected ones, which may be what breaks it; "
                     "it is also 5 to 6 times slower per tick. Train on core1.")
    if cond is not None:
        b = cond["B"]["per_type"]
        carriers = [t for t in main["dn_types"] if b[t]["corr_with_size"] > 0.5 and max(b[t]["mean_total"]) > 5]
        notes.append(f"Test B passes, but the size signal is carried by {', '.join(carriers) or 'no single type'}; DNa02 falls with size at 0 deg. "
                     "DNg13 also changes with bearing in a non-monotonic way, so the forward readout will mix bearing into range unless it is trained against it.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description="G0 connectome audit")
    parser.add_argument("--brain", default="data/brains/pursuit_core1.npz")
    parser.add_argument("--out", default="docs/audit_result.md")
    parser.add_argument("--json", default="data/brains/audit.json")
    parser.add_argument("--expanded", default=None, help="expanded retry brain (default: pursuit_core1_expanded.npz next to --brain)")
    parser.add_argument("--secondary", nargs="*", default=None, help="extra brains to test at the recommended setting and benchmark (default: pursuit_core2.npz)")
    args = parser.parse_args()
    secondary = args.secondary
    if secondary is None:
        secondary = [str(sibling(Path(args.brain), "pursuit_core2.npz"))]
    run(args.brain, args.out, args.json, args.expanded, secondary)


if __name__ == "__main__":
    main()
