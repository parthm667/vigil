"""Tests for the pursuit brain files, the degree-preserving shuffle and the runner (fast: core1 only for simulation)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flydrones.brain.connectome import Connectome  # noqa: E402

from flyfollow.brain.runner import BrainRunner  # noqa: E402
from flyfollow.brain.shuffle import degree_preserving_shuffle, path_check, role_indices  # noqa: E402

BRAINS = ROOT / "data" / "brains"
CORE1 = BRAINS / "pursuit_core1.npz"
CORE2 = BRAINS / "pursuit_core2.npz"
EXPANDED = BRAINS / "pursuit_core1_expanded.npz"
DN_TYPES = ["DNa02", "DNa01", "DNa03", "DNb05", "DNb06", "DNg13", "DNp09"]

needs_core1 = pytest.mark.skipif(not CORE1.exists(), reason="run python -m flyfollow.brain.build first")


@pytest.fixture(scope="module")
def core1() -> Connectome:
    return Connectome.load(CORE1)


def out_weights_by_pre(conn: Connectome) -> dict[int, list[float]]:
    W = conn.weights.tocsc()
    result = {}
    for j in range(conn.n):
        start = W.indptr[j]
        end = W.indptr[j + 1]
        result[j] = sorted(W.data[start:end].tolist())
    return result


@needs_core1
@pytest.mark.parametrize("path", [CORE1, CORE2, EXPANDED])
def test_build_files_load(path):
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    conn = Connectome.load(path)
    assert conn.n > 0
    assert conn.n_connections > 0
    assert conn.weights.shape == (conn.n, conn.n)
    roles = conn.meta["roles"]
    assert any(role == "input" for role in roles.values())
    assert any(role == "output" for role in roles.values())
    assert "bins" in conn.meta and "side_note" in conn.meta
    assert path.stat().st_size < 20e6


@needs_core1
def test_groups_non_empty(core1):
    for side in ["L", "R"]:
        bins = []
        for k in range(8):
            idx = core1.group(f"LC10a_{side}_b{k}")
            assert idx.size > 0, f"LC10a_{side}_b{k} is empty"
            bins.append(idx)
        all_bins = np.concatenate(bins)
        assert np.unique(all_bins).size == all_bins.size, "bins overlap"
        assert set(all_bins.tolist()) == set(core1.group(f"LC10a_{side}").tolist())
        for name in ["LC9", "LC11", "AROUSAL"]:
            assert core1.group(f"{name}_{side}").size > 0, f"{name}_{side} is empty"
        for dn in DN_TYPES:
            assert core1.group(f"{dn}_{side}").size == 1, f"{dn}_{side} should be one cell"
    for name, role in core1.meta["roles"].items():
        assert core1.group(name).size > 0, f"role group {name} is empty"


@needs_core1
def test_bins_ordered_front_to_back(core1):
    for side in ["L", "R"]:
        rows = core1.meta["bins"]["sides"][side]
        assert len(rows) == 8
        mean_h = [row["mean_h"] for row in rows]
        assert all(mean_h[i] > mean_h[i + 1] for i in range(7)), "bin 0 must be the most frontal (largest h)"


@needs_core1
@pytest.mark.parametrize("stratify", [False, "roles"])
def test_shuffle_preserves_degrees_signs_groups(core1, stratify):
    shuffled = degree_preserving_shuffle(core1, seed=3, stratify=stratify, max_tries=3)
    A = core1.weights != 0
    B = shuffled.weights != 0
    assert np.array_equal(np.asarray(A.sum(axis=0)).ravel(), np.asarray(B.sum(axis=0)).ravel()), "out-degree changed"
    assert np.array_equal(np.asarray(A.sum(axis=1)).ravel(), np.asarray(B.sum(axis=1)).ravel()), "in-degree changed"
    assert shuffled.weights.diagonal().sum() == 0, "self-loop created"

    # the weights leaving each presynaptic neuron (so its sign, Dale's law) are unchanged
    before = out_weights_by_pre(core1)
    after = out_weights_by_pre(shuffled)
    for j in range(core1.n):
        assert before[j] == after[j]

    assert set(shuffled.groups) == set(core1.groups)
    for name in core1.groups:
        assert np.array_equal(shuffled.group(name), core1.group(name))

    # the graph really changed
    overlap = A.multiply(B).sum() / A.sum()
    assert overlap < 0.5

    info = shuffled.meta["shuffle"]
    inputs = role_indices(core1, "input")
    outputs = role_indices(core1, "output")
    assert info["paths_shuffled"] == path_check(shuffled, inputs, outputs, info["path_hops"])
    if stratify == "roles":
        assert info["paths_shuffled"] == info["paths_original"]


@needs_core1
def test_runner_tick_returns_finite_rates(core1):
    runner = BrainRunner(core1, seed=0)
    outputs = [f"{dn}_{side}" for dn in DN_TYPES for side in ["L", "R"]]
    inputs = {"LC10a_R_b1": 200.0, "LC10a_R_b2": 200.0, "LC10a_R_b0": 120.0}
    total = {name: 0.0 for name in outputs}
    for i in range(20):
        rates = runner.tick(inputs, outputs, ms=50.0)
        for name in outputs:
            assert np.isfinite(rates[name]) and rates[name] >= 0.0
            total[name] += rates[name]
    # a target on the right drives the right DNa02 (ipsilateral push-pull found in the audit)
    assert total["DNa02_R"] > total["DNa02_L"]
