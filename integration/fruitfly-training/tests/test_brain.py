"""Fast brain tests (no full build): built cores, shuffles and azimuth bins.

Tests on built files skip when data/brains/*.npz are absent (run scripts/setup_data.sh).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from flyfollow.brain.build import ensure_flydrones
from flyfollow.interfaces import INPUT_GROUPS, N_AZIMUTH_BINS, OUTPUT_GROUPS, brains_dir

ensure_flydrones()

from flydrones.brain.connectome import Connectome

from flyfollow.brain.shuffle import degree_signature, io_indices, path_counts, shuffle_core

CORES = [brains_dir() / f"pursuit_core{h}.npz" for h in (1, 2)]
SHUFS = sorted(brains_dir().glob("pursuit_core*_shuf*.npz"))


def _load(p):
    if not p.exists():
        pytest.skip(f"{p.name} not built")
    return Connectome.load(p)


@pytest.mark.parametrize("path", CORES, ids=lambda p: p.stem)
def test_core_groups_present_and_nonempty(path):
    c = _load(path)
    for g in (*INPUT_GROUPS, *OUTPUT_GROUPS):
        assert g in c.groups, g
        assert c.group(g).size > 0, g
    roles = c.meta["roles"]
    assert all(roles[g] == "input" for g in INPUT_GROUPS)
    assert all(roles[g] == "output" for g in OUTPUT_GROUPS)
    # audit-only groups never become inputs or outputs
    assert set(roles) == set(INPUT_GROUPS) | set(OUTPUT_GROUPS)
    for t in ("DNa02", "DNa01", "DNb05", "DNg13", "DNb06"):
        assert c.group(f"{t}_L").size == 1 and c.group(f"{t}_R").size == 1


@pytest.mark.parametrize("path", CORES + SHUFS, ids=lambda p: p.stem)
def test_bins_cover_each_lc10a_side_exactly_once(path):
    c = _load(path)
    bins = c.meta["azimuth_bins"]
    for g in ("LC10a_L", "LC10a_R"):
        assert len(bins[g]) == N_AZIMUTH_BINS
        flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in bins[g]])
        assert flat.size == np.unique(flat).size, "a neuron is in two bins"
        assert np.array_equal(np.sort(flat), np.sort(c.group(g))), "bins do not cover the group"
        assert all(len(b) > 0 for b in bins[g])


@pytest.mark.parametrize("path", SHUFS, ids=lambda p: p.stem)
def test_saved_shuffle_matches_its_core(path):
    sh = _load(path)
    real = _load(path.with_name(sh.meta["shuffle"]["of"]))
    assert sh.n == real.n
    assert {k: v.tolist() for k, v in sh.groups.items()} == {k: v.tolist() for k, v in real.groups.items()}
    assert sh.meta["azimuth_bins"] == real.meta["azimuth_bins"]
    a, b = degree_signature(sh), degree_signature(real)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    # weights permuted within each sign set
    for m in (lambda x: x > 0, lambda x: x < 0):
        assert np.array_equal(np.sort(sh.weights.data[m(sh.weights.data)]), np.sort(real.weights.data[m(real.weights.data)]))
    info = sh.meta["shuffle"]
    assert info["shuf_paths"]["walks"] >= info["real_paths"]["walks"]


def _toy(seed=0, n=300, m=4000):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, m)
    post = rng.integers(0, n, m)
    ok = pre != post
    pre, post = pre[ok], post[ok]
    key = np.unique(pre * n + post)
    pre, post = key // n, key % n
    sign = np.where(rng.random(n) < 0.3, -1.0, 1.0)  # Dale's law: sign per presynaptic neuron
    w = rng.integers(3, 40, pre.size).astype(np.float32) * sign[pre]
    W = sparse.csc_matrix((w, (post, pre)), shape=(n, n), dtype=np.float32)
    groups = {g: np.array([i], dtype=np.int64) for i, g in enumerate((*INPUT_GROUPS, *OUTPUT_GROUPS))}
    groups["LC10a_L"] = np.arange(20, 36)
    groups["LC10a_R"] = np.arange(40, 56)
    return Connectome(name="toy", weights=W, types=np.array(["x"] * n), sides=np.array([""] * n), groups=groups,
                      meta={"roles": {}, "azimuth_bins": {"LC10a_L": [[20, 21]], "LC10a_R": [[40]]}})


@pytest.mark.parametrize("layered", [False, True])
def test_shuffle_preserves_degrees_per_sign_and_groups(layered):
    c = _toy()
    sh, st = shuffle_core(c, seed=3, rounds=20, layered=layered, hops=1)
    assert st["swaps_accepted"] > 0
    a, b = degree_signature(sh), degree_signature(c)
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    coo = sh.weights.tocoo()
    assert not np.any(coo.row == coo.col), "self-loop"
    assert sh.weights.nnz == c.weights.nnz, "duplicate edges merged"
    assert {k: v.tolist() for k, v in sh.groups.items()} == {k: v.tolist() for k, v in c.groups.items()}
    assert sh.meta["azimuth_bins"] == c.meta["azimuth_bins"]
    # the wiring actually changed
    assert (abs(sh.weights) != abs(c.weights)).nnz > 0
    ins, outs = io_indices(c)
    pc = path_counts(sh.weights, ins, outs, 2)
    assert pc["walks"] >= 0 and len(pc["walks_by_len"]) == 2
