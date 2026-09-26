"""A tiny hand-wired pursuit connectome for tests. NOT fly data.

Same group names as the real pursuit subgraph (interfaces.INPUT_GROUPS / OUTPUT_GROUPS), so
FlyBrain, the controllers and calibration run on it in seconds:

    LC10a_side (8 azimuth bins) -> AOTU_side -> DNa02/DNa01/DNb05/DNg13 same side (ipsiversive)
                                             -> DNb06 other side (contraversive)
                                             -> LALinh other side -| ipsiversive DNs there
    LC9 / LC11 side -> AOTU_side (weak);  AROUSAL -> all DNs and AOTU (tonic baseline)
    plus a random hidden background that only receives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from flydrones.brain.connectome import Connectome
from scipy import sparse

from flyfollow.interfaces import DN_TYPES, INPUT_GROUPS, N_AZIMUTH_BINS, OUTPUT_GROUPS

PER_BIN = 6


def make_synthetic_pursuit_brain(path: str | Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    types: list[str] = []
    sides: list[str] = []
    sign: list[float] = []
    idx: dict[str, np.ndarray] = {}

    def pop(name: str, typ: str, n: int, side: str, sg: float) -> None:
        start = len(types)
        idx[name] = np.arange(start, start + n, dtype=np.int64)
        types.extend([typ] * n)
        sides.extend([side] * n)
        sign.extend([sg] * n)

    for s in "LR":
        pop(f"LC10a_{s}", "LC10a", PER_BIN * N_AZIMUTH_BINS, s, 1.0)
        pop(f"LC9_{s}", "LC9", 12, s, 1.0)
        pop(f"LC11_{s}", "LC11", 12, s, 1.0)
        pop(f"AROUSAL_{s}", "pC1a", 6, s, 1.0)
        pop(f"AOTU_{s}", "AOTU019", 40, s, 1.0)
        pop(f"LALinh_{s}", "LALinh", 12, s, -1.0)
        for t in DN_TYPES:
            pop(f"{t}_{s}", t, 1, s, 1.0)
    pop("BG", "bg", 100, "", 1.0)
    n = len(types)

    rows, cols, vals = [], [], []

    def connect(pre: np.ndarray, post: np.ndarray, syn: float, p: float = 1.0) -> None:
        for q in post:
            for r in pre:
                if p >= 1.0 or rng.random() < p:
                    rows.append(q)
                    cols.append(r)
                    vals.append(syn)

    other = {"L": "R", "R": "L"}
    for s in "LR":
        o = other[s]
        connect(idx[f"LC10a_{s}"], idx[f"AOTU_{s}"], 20.0, p=0.25)
        connect(idx[f"LC9_{s}"], idx[f"AOTU_{s}"], 6.0, p=0.2)
        connect(idx[f"LC11_{s}"], idx[f"AOTU_{s}"], 6.0, p=0.2)
        connect(idx[f"AROUSAL_{s}"], idx[f"AOTU_{s}"], 1.0, p=0.5)
        for t in DN_TYPES:
            dn_ipsi = idx[f"{t}_{s}"] if t != "DNb06" else idx[f"{t}_{o}"]
            connect(idx[f"AOTU_{s}"], dn_ipsi, 6.0, p=0.6)
            for a in "LR":
                connect(idx[f"AROUSAL_{a}"], idx[f"{t}_{s}"], 10.0)
        connect(idx[f"AOTU_{s}"], idx[f"LALinh_{o}"], 8.0, p=0.3)
        for t in DN_TYPES:
            if t != "DNb06":
                connect(idx[f"LALinh_{o}"], idx[f"{t}_{o}"], 4.0, p=0.8)
        connect(idx[f"AOTU_{s}"], idx["BG"], 2.0, p=0.05)
    connect(idx["BG"], idx["BG"], 2.0, p=0.03)

    sg = np.asarray(sign)
    vals_signed = np.asarray(vals) * sg[np.asarray(cols)]
    W = sparse.csc_matrix((vals_signed.astype(np.float32), (rows, cols)), shape=(n, n))
    W.sum_duplicates()

    groups = {g: idx[g] for g in INPUT_GROUPS + OUTPUT_GROUPS}
    bins = {g: [b.tolist() for b in np.array_split(idx[g], N_AZIMUTH_BINS)] for g in ("LC10a_L", "LC10a_R")}
    roles = {**{g: "input" for g in INPUT_GROUPS}, **{g: "output" for g in OUTPUT_GROUPS}}
    conn = Connectome(
        name="synthetic_pursuit",
        weights=W,
        types=np.asarray(types),
        sides=np.asarray(sides),
        groups=groups,
        meta={"roles": roles, "azimuth_bins": bins, "synthetic": True, "seed": seed},
    )
    return conn.save(path)
