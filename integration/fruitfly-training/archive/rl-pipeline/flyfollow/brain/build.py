"""Build the pursuit subgraphs (LC10a azimuth bins in, steering DNs out) from the full MaleCNS connectome.

Usage:
    python -m flyfollow.brain.build --full malecns_fixed.npz --ann body-annotations-male-cns-v1.0-minconf-0.5.feather --out data/brains

Outputs pursuit_core1.npz, pursuit_core2.npz and pursuit_core1_expanded.npz (the one G0 retry,
LC10b-e added as inputs and the DN types with the highest LC10 path weight added as outputs).

How the LC10a azimuth bins are made (see meta["bins"] in the output file):
    MaleCNS gives optic lobe hex column coordinates (assignedOlHex1/2) only for 15 columnar types
    (L1-L5, Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9, Tm20, T1, C2, C3). LC10a has none, and its direct
    inputs from those types are tiny. So we propagate: every neuron gets the synapse-weighted mean
    hex of its hex-annotated inputs (Tm3, Tm5a/Y, TmY and T2/T3 come out with a spread under 2
    columns), then each LC10a gets the synapse-weighted mean hex of its columnar inputs. That is the
    centroid of its lobula dendrite in column coordinates.
    Axis: fitting lamina (L1-L5) soma positions against hex gives anterior = larger (hex1 - hex2)
    on both sides; the medulla (Mi1, Tm1, T1, Tm9) gives the opposite sign, as expected from the
    first optic chiasm. The lamina sits right under the retina, so larger h = hex1 - hex2 means a
    more anterior ommatidium, which looks more frontally. Bin 0 = the eighth of LC10a with the largest h.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from flydrones.brain.connectome import Connectome
from scipy import sparse

N_BINS = 8
DN_TYPES = ["DNa02", "DNa01", "DNa03", "DNb05", "DNb06", "DNg13", "DNp09"]
EXPANDED_INPUT_TYPES = {"LC10b": ["LC10b"], "LC10c": ["LC10c-1", "LC10c-2"], "LC10d": ["LC10d"], "LC10e": ["LC10e"]}
N_EXTRA_DNS = 5
N_AROUSAL_TYPES = 4
HEX_TYPES_MIN_SYN = 10
COLUMNAR_MAX_SPREAD = 2.0
VNC_SUPERCLASSES = ["vnc_intrinsic", "vnc_motor", "ascending_neuron", "vnc_efferent"]
AZ_FRONT_EDGE = -10.0
AZ_REAR_EDGE = 170.0

SIDE_NOTE = (
    "Each optic lobe sees the ipsilateral visual hemifield: LC10a_L_* (left optic lobe) sees the left half "
    "of the visual field (negative bearing), LC10a_R_* sees the right half (positive bearing). Bin 0 is the most "
    "frontal eighth of that side's LC10a (near the midline of the visual field), bin 7 the most lateral/rear. "
    "Neuron sides are soma sides (rootSide is NaN for most MaleCNS bodies, so the build falls back to somaSide). "
    "For DNb06 and DNg13 the VNC output is mostly on the side opposite the soma (axon crosses), see meta['dn_axon_side']."
)


def load_annotations(path: str | Path, body_ids: np.ndarray) -> pd.DataFrame:
    ann = pd.read_feather(path)
    ann = ann.drop_duplicates("bodyId").set_index("bodyId")
    return ann.reindex(body_ids)


def soma_xyz(ann: pd.DataFrame) -> np.ndarray:
    xyz = np.full((len(ann), 3), np.nan)
    values = ann["somaLocation"].to_numpy()
    for i in range(len(values)):
        if isinstance(values[i], np.ndarray | list) and len(values[i]) == 3:
            xyz[i] = np.asarray(values[i], dtype=float)
    return xyz


def fit_hex_axes(conn: Connectome, ann: pd.DataFrame) -> dict:
    """Least-squares fit of soma position (x, y, z) against (hex1, hex2), for lamina and medulla cells, per side.

    MaleCNS axes (checked against landmarks): x grows toward the fly's left, y grows ventral,
    z grows posterior (antennal lobe PNs z ~ 15k, Kenyon cells ~ 34k, VNC ~ 100k).
    Returns, per side, the (hex1, hex2) direction that points anterior in the lamina and in the medulla.
    """
    types = conn.types.astype(str)
    sides = conn.sides.astype(str)
    hex1 = ann["assignedOlHex1"].to_numpy(dtype=float)
    hex2 = ann["assignedOlHex2"].to_numpy(dtype=float)
    xyz = soma_xyz(ann)
    has = ~np.isnan(hex1) & ~np.isnan(xyz[:, 0])

    result = {}
    for layer, layer_types in [("lamina", ["L1", "L2", "L3", "L4", "L5"]), ("medulla", ["Mi1", "Tm1", "T1", "Tm9"])]:
        for side in ["L", "R"]:
            mask = has & np.isin(types, layer_types) & (sides == side)
            X = np.c_[hex1[mask], hex2[mask], np.ones(mask.sum())]
            fit = {"n": int(mask.sum())}
            for k, axis in enumerate(["x", "y", "z"]):
                target = xyz[mask, k]
                coef = np.linalg.lstsq(X, target, rcond=None)[0]
                pred = X @ coef
                r2 = 1.0 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()
                fit[axis] = {"d_hex1": round(float(coef[0]), 1), "d_hex2": round(float(coef[1]), 1), "r2": round(float(r2), 3)}
            anterior = -np.array([fit["z"]["d_hex1"], fit["z"]["d_hex2"]])
            anterior = anterior / np.linalg.norm(anterior)
            fit["anterior_dir_hex"] = [round(float(anterior[0]), 3), round(float(anterior[1]), 3)]
            fit["cos_to_hex1_minus_hex2"] = round(float(anterior @ np.array([1.0, -1.0]) / np.sqrt(2.0)), 3)
            result[f"{layer}_{side}"] = fit
    return result


def propagate_hex(A: sparse.csr_matrix, hex_xy: np.ndarray, known: np.ndarray, sides: np.ndarray, min_syn: float):
    """Synapse-weighted mean (and spread) of the hex position of each neuron's same-side inputs from `known` neurons."""
    n = A.shape[0]
    mean = np.full((n, 2), np.nan)
    spread = np.full(n, np.nan)
    total = np.zeros(n)
    hex_filled = np.nan_to_num(hex_xy)
    for side in ["L", "R"]:
        use = (known & (sides == side)).astype(float)
        A_side = (A @ sparse.diags(use)).tocsr()
        syn = np.asarray(A_side.sum(axis=1)).ravel()
        m1 = (A_side @ hex_filled) / np.maximum(syn, 1e-9)[:, None]
        m2 = (A_side @ hex_filled**2) / np.maximum(syn, 1e-9)[:, None]
        sd = np.sqrt(np.maximum(m2 - m1**2, 0.0).sum(axis=1))
        target = (sides == side) & (syn >= min_syn)
        mean[target] = m1[target]
        spread[target] = sd[target]
        total[target] = syn[target]
    return mean, spread, total


def lc_hex_centroids(conn: Connectome, ann: pd.DataFrame, cell_type: str) -> pd.DataFrame:
    """Dendritic centroid of each neuron of `cell_type` in optic lobe hex coordinates (two-step propagation)."""
    types = conn.types.astype(str)
    sides = conn.sides.astype(str)
    A = abs(conn.weights).tocsr().astype(np.float64)  # (post, pre) synapse counts
    hex_xy = np.c_[ann["assignedOlHex1"].to_numpy(dtype=float), ann["assignedOlHex2"].to_numpy(dtype=float)]
    known = ~np.isnan(hex_xy[:, 0])

    step1, spread1, total1 = propagate_hex(A, hex_xy, known, sides, HEX_TYPES_MIN_SYN)
    columnar = known | ((spread1 <= COLUMNAR_MAX_SPREAD) & (total1 >= HEX_TYPES_MIN_SYN))
    hex_all = np.where(known[:, None], hex_xy, step1)

    rows = []
    for side in ["L", "R"]:
        cells = np.flatnonzero((types == cell_type) & (sides == side))
        sub = A[cells].tocoo()
        ok = columnar[sub.col] & (sides[sub.col] == side) & (types[sub.col] != cell_type)
        w = sub.data[ok]
        r = sub.row[ok]
        pos = hex_all[sub.col[ok]]
        syn = np.bincount(r, weights=w, minlength=cells.size)
        h1 = np.bincount(r, weights=w * pos[:, 0], minlength=cells.size) / np.maximum(syn, 1e-9)
        h2 = np.bincount(r, weights=w * pos[:, 1], minlength=cells.size) / np.maximum(syn, 1e-9)
        dh = (pos[:, 0] - pos[:, 1]) - (h1 - h2)[r]
        sd_h = np.sqrt(np.bincount(r, weights=w * dh**2, minlength=cells.size) / np.maximum(syn, 1e-9))
        for i in range(cells.size):
            ok_cell = syn[i] > 0
            rows.append(
                {
                    "index": int(cells[i]),
                    "body_id": int(conn.body_ids[cells[i]]),
                    "side": side,
                    "hex1": float(h1[i]) if ok_cell else np.nan,
                    "hex2": float(h2[i]) if ok_cell else np.nan,
                    "h": float(h1[i] - h2[i]) if ok_cell else np.nan,
                    "v": float(h1[i] + h2[i]) if ok_cell else np.nan,
                    "sd_h": float(sd_h[i]) if ok_cell else np.nan,
                    "syn_used": float(syn[i]),
                }
            )
    return pd.DataFrame(rows)


def eye_h_extent(ann: pd.DataFrame, sides: np.ndarray) -> dict[str, list[float]]:
    """Range of h = hex1 - hex2 over all hex-annotated columns, per side (the whole eye, front to back)."""
    h = (ann["assignedOlHex1"] - ann["assignedOlHex2"]).to_numpy(dtype=float)
    extent = {}
    for side in ["L", "R"]:
        values = h[(sides == side) & ~np.isnan(h)]
        extent[side] = [float(values.min()), float(values.max())]
    return extent


def approx_azimuth(h: float, extent: list[float]) -> float:
    """Assumed linear map from h to azimuth: the front edge of the eye looks at -10 deg (10 deg of binocular
    overlap into the other hemifield), the rear edge at 170 deg. An assumption for the encoder, not measured."""
    h_min, h_max = extent
    return AZ_FRONT_EDGE + (h_max - h) / (h_max - h_min) * (AZ_REAR_EDGE - AZ_FRONT_EDGE)


def make_azimuth_bins(cells: pd.DataFrame, ann: pd.DataFrame, extent: dict[str, list[float]]) -> tuple[dict[str, np.ndarray], dict]:
    """Split each side's LC10a into N_BINS bins, bin 0 = largest h (most frontal)."""
    groups = {}
    info = {
        "method": "hex_centroid",
        "coordinate": "h = hex1 - hex2 of the synapse-weighted centroid of each LC10a's columnar inputs; larger h = more anterior on the eye = more frontal",
        "n_bins": N_BINS,
        "bin0": "most frontal (largest h)",
        "eye_h_extent": extent,
        "approx_azimuth_rule": f"linear in h, eye front edge {AZ_FRONT_EDGE:.0f} deg, rear edge {AZ_REAR_EDGE:.0f} deg (assumption, about 5 deg per h unit)",
        "sides": {},
    }
    fallback_count = 0
    for side in ["L", "R"]:
        part = cells[cells["side"] == side].copy()
        # neurons without a centroid (none in MaleCNS v1.0) fall back to soma y rank, which correlates +0.55 with h
        missing = part["h"].isna()
        if missing.any():
            fallback_count += int(missing.sum())
            xyz = soma_xyz(ann.iloc[part.loc[missing, "index"].to_numpy()])
            y = xyz[:, 1]
            h_known = part.loc[~missing, "h"].to_numpy()
            ranks = np.argsort(np.argsort(np.nan_to_num(y, nan=np.nanmedian(y))))
            part.loc[missing, "h"] = np.quantile(h_known, ranks / max(len(ranks) - 1, 1))
        part = part.sort_values(["h", "body_id"], ascending=[False, True])
        chunks = np.array_split(part.to_dict("records"), N_BINS)
        side_info = []
        for k in range(N_BINS):
            chunk = list(chunks[k])
            idx = np.array(sorted(row["index"] for row in chunk), dtype=np.int64)
            groups[f"LC10a_{side}_b{k}"] = idx
            h_vals = [row["h"] for row in chunk]
            side_info.append(
                {
                    "bin": k,
                    "count": len(chunk),
                    "mean_h": round(float(np.mean(h_vals)), 2),
                    "min_h": round(float(np.min(h_vals)), 2),
                    "max_h": round(float(np.max(h_vals)), 2),
                    "mean_hex1": round(float(np.mean([row["hex1"] for row in chunk])), 2),
                    "mean_hex2": round(float(np.mean([row["hex2"] for row in chunk])), 2),
                    "mean_v": round(float(np.mean([row["v"] for row in chunk])), 2),
                    "median_dendrite_sd_h": round(float(np.median([row["sd_h"] for row in chunk])), 2),
                    "approx_azimuth_deg": round(approx_azimuth(float(np.mean(h_vals)), extent[side]), 1),
                }
            )
        info["sides"][side] = side_info
    info["fallback_neurons"] = fallback_count
    return groups, info


def two_hop(W_csr: sparse.csr_matrix, W_csc: sparse.csc_matrix, src: np.ndarray, dst: np.ndarray) -> float:
    """Sum over intermediates X of W[dst, X] * W[X, src] (signed synapse-count products)."""
    into_x = np.asarray(W_csc[:, src].sum(axis=1)).ravel()
    from_x = np.asarray(W_csr[dst].sum(axis=0)).ravel()
    return float((into_x * from_x).sum())


def pick_arousal(conn: Connectome, ann: pd.DataFrame, lc_idx: np.ndarray, dn_idx: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Choose P1-like pC1 subtypes: male-specific, fru/dsx co-expressing, synonym pMP4 / pMP-e (the P1 cluster
    names of Yu 2010 and Cachero 2010). Rank them by direct synapses onto the LC10a -> X -> DN pathway
    (intermediates X plus the DNs) and keep the top N_AROUSAL_TYPES with net excitatory 2-hop drive onto the DNs."""
    types = conn.types.astype(str)
    sides = conn.sides.astype(str)
    W = conn.weights.tocsr()
    Wc = conn.weights.tocsc()
    A = abs(W)

    from_lc = np.asarray(A[:, lc_idx].sum(axis=1)).ravel()
    to_dn = np.asarray(A[dn_idx].sum(axis=0)).ravel()
    pathway = np.flatnonzero((from_lc > 0) & (to_dn > 0))
    pathway = np.union1d(pathway, dn_idx)

    synonyms = ann["synonyms"].to_numpy()
    fru_dsx = ann["fruDsx"].to_numpy()
    dimorphism = ann["dimorphism"].to_numpy()

    candidates = []
    for t in sorted(set(types[np.char.startswith(types.astype(str), "pC1")])):
        idx = np.flatnonzero(types == t)
        syn_text = str(synonyms[idx[0]])
        fd = str(fru_dsx[idx[0]])
        dim = str(dimorphism[idx[0]])
        p1_like = ("pMP4" in syn_text or "pMP-e" in syn_text) and fd.startswith("coexpress") and "male-specific" in dim
        onto_pathway = float(W[pathway][:, idx].sum())
        hop2_dn = two_hop(W, Wc, idx, dn_idx)
        from_lc10a = float(A[idx][:, lc_idx].sum())
        candidates.append(
            {
                "type": str(t),
                "n": int(idx.size),
                "fruDsx": fd,
                "dimorphism": dim,
                "p1_like": bool(p1_like),
                "syn_onto_pathway": onto_pathway,
                "hop2_signed_to_dns": hop2_dn,
                "syn_from_lc10a": from_lc10a,
            }
        )

    ranked = [c for c in candidates if c["p1_like"] and c["hop2_signed_to_dns"] > 0 and c["syn_onto_pathway"] > 0]
    ranked.sort(key=lambda c: -c["syn_onto_pathway"])
    chosen = [str(c["type"]) for c in ranked[:N_AROUSAL_TYPES]]

    groups = {}
    for side in ["L", "R"]:
        groups[f"AROUSAL_{side}"] = np.flatnonzero(np.isin(types, chosen) & (sides == side)).astype(np.int64)

    info = {
        "chosen_types": chosen,
        "rule": "pC1 subtypes with synonym pMP4/pMP-e (P1 cluster), fruDsx co-expression and male-specific; ranked by "
        "direct synapses onto LC10a->X->DN intermediates plus the steering DNs; top 4 with positive net signed 2-hop drive to the DNs",
        "pathway_neurons": int(pathway.size),
        "p1_like_types": [c["type"] for c in candidates if c["p1_like"]],
        "candidates": sorted(candidates, key=lambda c: -c["syn_onto_pathway"])[:20],
        "note": "Inferred, not verified: no MaleCNS type is named P1; LC10a receives no direct pC1 input (min 3 synapses).",
    }
    return groups, info


def dn_axon_sides(conn: Connectome, dn_groups: dict[str, np.ndarray]) -> dict:
    """Fraction of each DN's VNC output synapses onto neurons on the DN's own soma side (proxy for axon side)."""
    sides = conn.sides.astype(str)
    sc = conn.superclass.astype(str) if conn.superclass is not None else np.array([""] * conn.n)
    vnc = np.isin(sc, VNC_SUPERCLASSES)
    Wc = abs(conn.weights).tocsc()
    result = {}
    for name, idx in dn_groups.items():
        if idx.size == 0:
            continue
        own_side = name[-1]
        same = 0.0
        opposite = 0.0
        for i in idx:
            col = Wc[:, i].tocoo()
            post = col.row
            w = col.data
            same += float(w[vnc[post] & (sides[post] == own_side)].sum())
            opposite += float(w[vnc[post] & (sides[post] != own_side) & (sides[post] != "")].sum())
        frac = same / max(same + opposite, 1.0)
        if frac >= 0.5:
            axon = own_side
        else:
            axon = "R" if own_side == "L" else "L"
        result[name] = {"vnc_syn_same_side": same, "vnc_syn_opposite_side": opposite, "frac_same": round(frac, 3), "axon_side": axon}
    return result


def pick_extra_dns(conn: Connectome, lc_all: np.ndarray, exclude: list[str], n_extra: int) -> tuple[list[str], list[dict]]:
    """DN types (present on both sides) with the highest LC10 (a-e) path weight, direct plus 2-hop."""
    types = conn.types.astype(str)
    sides = conn.sides.astype(str)
    sc = conn.superclass.astype(str)
    W = conn.weights.tocsr()
    A = abs(W)
    Ac = A.tocsc()

    into_x = np.asarray(Ac[:, lc_all].sum(axis=1)).ravel()
    scores = []
    for t in sorted(set(types[sc == "descending_neuron"])):
        if t in exclude or t == "":
            continue
        idx = np.flatnonzero(types == t)
        if not (np.any(sides[idx] == "L") and np.any(sides[idx] == "R")):
            continue
        direct = float(A[idx][:, lc_all].sum())
        from_x = np.asarray(A[idx].sum(axis=0)).ravel()
        hop2 = float((into_x * from_x).sum())
        scores.append({"type": str(t), "n": int(idx.size), "direct_syn": direct, "hop2_abs": hop2})
    max_direct = max([s["direct_syn"] for s in scores] + [1.0])
    max_hop2 = max([s["hop2_abs"] for s in scores] + [1.0])
    for s in scores:
        s["score"] = s["direct_syn"] / max_direct + s["hop2_abs"] / max_hop2
    scores.sort(key=lambda s: -s["score"])
    return [s["type"] for s in scores[:n_extra]], scores[:15]


def type_side_groups(conn: Connectome, prefix: str, type_names: list[str]) -> dict[str, np.ndarray]:
    types = conn.types.astype(str)
    sides = conn.sides.astype(str)
    groups = {}
    for side in ["L", "R"]:
        groups[f"{prefix}_{side}"] = np.flatnonzero(np.isin(types, type_names) & (sides == side)).astype(np.int64)
    return groups


def file_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def report(core: Connectome, path: Path) -> dict:
    sizes = {}
    for name in sorted(core.groups):
        sizes[name] = int(core.groups[name].size)
    print(f"{path.name}: {core.n:,} neurons, {core.n_connections:,} connections, {core.n_synapses:,} synapses, {file_mb(path):.2f} MB")
    return {"file": path.name, "neurons": core.n, "connections": core.n_connections, "synapses": core.n_synapses, "mb": round(file_mb(path), 3), "groups": sizes}


def build(full_path: str, ann_path: str, out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full = Connectome.load(full_path)
    print(full.summary())
    ann = load_annotations(ann_path, full.body_ids)
    types = full.types.astype(str)
    sides = full.sides.astype(str)

    axes = fit_hex_axes(full, ann)
    for key, fit in axes.items():
        print(f"hex axis fit {key}: n={fit['n']} anterior dir {fit['anterior_dir_hex']} cos to (1,-1) {fit['cos_to_hex1_minus_hex2']} z R2 {fit['z']['r2']}")

    cells = lc_hex_centroids(full, ann, "LC10a")
    bin_groups, bin_info = make_azimuth_bins(cells, ann, eye_h_extent(ann, sides))
    bin_info["axis_fits"] = axes
    bin_info["axis_check"] = (
        "lamina: anterior is +(hex1 - hex2) on both sides; medulla: the opposite sign (first optic chiasm). "
        "The lamina lies under the retina, so +h = anterior ommatidia = frontal view. Verified from soma positions; "
        "the ommatidium-to-lamina mapping itself is textbook anatomy, not checked in this data."
    )
    xyz = soma_xyz(ann.iloc[cells["index"].to_numpy()])
    bin_info["corr_h_vs_soma_y"] = round(float(np.corrcoef(cells["h"], xyz[:, 1])[0, 1]), 3)
    bin_info["lc10a_cells"] = {
        "body_id": cells["body_id"].tolist(),
        "side": cells["side"].tolist(),
        "h": [round(float(x), 3) for x in cells["h"]],
        "v": [round(float(x), 3) for x in cells["v"]],
    }

    dn_groups = {}
    for dn in DN_TYPES:
        dn_groups.update(type_side_groups(full, dn, [dn]))

    lc10a_all = np.flatnonzero(types == "LC10a")
    dn_all = np.concatenate(list(dn_groups.values()))
    arousal_groups, arousal_info = pick_arousal(full, ann, lc10a_all, dn_all)
    print(f"arousal (P1-like) types: {arousal_info['chosen_types']}")

    groups = {}
    groups.update(bin_groups)
    groups.update(type_side_groups(full, "LC9", ["LC9"]))
    groups.update(type_side_groups(full, "LC11", ["LC11"]))
    groups.update(arousal_groups)
    groups.update(dn_groups)

    roles = {}
    for name in groups:
        if name.startswith("DN"):
            roles[name] = "output"
        else:
            roles[name] = "input"

    # convenience groups (no role, so they do not change the subgraph)
    groups.update(type_side_groups(full, "LC10a", ["LC10a"]))

    full.groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
    full.meta["roles"] = roles
    full.meta["bins"] = bin_info
    full.meta["side_note"] = SIDE_NOTE
    full.meta["arousal"] = arousal_info
    full.meta["dn_axon_side"] = dn_axon_sides(full, dn_groups)
    full.meta["built_by"] = "flyfollow.brain.build"

    summary = {"full": full.summary(), "files": {}}

    core1 = full.sensorimotor_core(hops=1)
    core1.name = "pursuit_core1"
    core1.meta["hops"] = 1
    path1 = core1.save(out / "pursuit_core1.npz")
    summary["files"]["core1"] = report(core1, path1)

    core2 = full.sensorimotor_core(hops=2)
    uncapped = core2.n
    if uncapped > 20000:
        core2 = full.sensorimotor_core(hops=2, max_neurons=20000)
    core2.name = "pursuit_core2"
    core2.meta["hops"] = 2
    core2.meta["uncapped_neurons"] = int(uncapped)
    core2.meta["capped"] = bool(uncapped > 20000)
    print(f"core2 before cap: {uncapped:,} neurons")
    path2 = core2.save(out / "pursuit_core2.npz")
    summary["files"]["core2"] = report(core2, path2)

    # expanded retry brain: add LC10b-e inputs and the DN types with the highest LC10 path weight
    lc_expanded = np.flatnonzero(np.isin(types, ["LC10a", "LC10b", "LC10c-1", "LC10c-2", "LC10d", "LC10e"]))
    extra_dns, dn_scores = pick_extra_dns(full, lc_expanded, DN_TYPES, N_EXTRA_DNS)
    print(f"expanded retry: extra DN types {extra_dns}")
    exp_groups = dict(groups)
    exp_roles = dict(roles)
    for name, type_names in EXPANDED_INPUT_TYPES.items():
        for key, idx in type_side_groups(full, name, type_names).items():
            exp_groups[key] = idx
            exp_roles[key] = "input"
    for dn in extra_dns:
        for key, idx in type_side_groups(full, dn, [dn]).items():
            exp_groups[key] = idx
            exp_roles[key] = "output"
    full.groups = {k: np.asarray(v, dtype=np.int64) for k, v in exp_groups.items()}
    full.meta["roles"] = exp_roles
    full.meta["expanded"] = {"extra_inputs": list(EXPANDED_INPUT_TYPES), "extra_dns": extra_dns, "dn_scores": dn_scores}
    full.meta["dn_axon_side"] = dn_axon_sides(full, {k: v for k, v in exp_groups.items() if k.startswith("DN")})
    core_exp = full.sensorimotor_core(hops=1)
    core_exp.name = "pursuit_core1_expanded"
    core_exp.meta["hops"] = 1
    path3 = core_exp.save(out / "pursuit_core1_expanded.npz")
    summary["files"]["core1_expanded"] = report(core_exp, path3)

    print("group sizes (core1):")
    for name in sorted(core1.groups):
        print(f"  {name:14s} {core1.groups[name].size}")
    for side in ["L", "R"]:
        for b in bin_info["sides"][side]:
            print(f"  LC10a_{side}_b{b['bin']}: n={b['count']} mean h={b['mean_h']} (range {b['min_h']}..{b['max_h']}) mean v={b['mean_v']} approx az {b['approx_azimuth_deg']} deg")
    summary["bins"] = bin_info["sides"]
    summary["arousal"] = arousal_info["chosen_types"]
    summary["extra_dns"] = extra_dns
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", required=True, help="full MaleCNS connectome .npz (FlyDrones format, sides fixed)")
    parser.add_argument("--ann", required=True, help="MaleCNS body annotations .feather")
    parser.add_argument("--out", default="data/brains")
    args = parser.parse_args()
    summary = build(args.full, args.ann, args.out)
    for key, info in summary["files"].items():
        print(f"{key}: {info['file']} {info['neurons']:,} neurons, {info['connections']:,} connections, {info['mb']:.2f} MB")


if __name__ == "__main__":
    main()
