"""Degree-preserving shuffle of a pursuit subgraph (the FLY-SHUF control of plan section 4.7).

Every neuron keeps its in-degree and out-degree. Edges are rewired with directed double-edge swaps
(a->b, c->d) -> (a->d, c->b), rejecting self-loops and duplicate edges. The swapped edge keeps the
signed weight of its presynaptic slot, so each neuron keeps the exact multiset of its outgoing
weights: the global weight magnitude distribution, every neuron's out-strength and Dale's law
(a neuron's outgoing edges all keep that neuron's sign) are preserved. In-strength is not.

Group index lists and neuron labels are unchanged, so encoder and readout use the same neurons.
"""

from __future__ import annotations

import warnings

import numpy as np
from flydrones.brain.connectome import Connectome
from scipy import sparse

ROUNDS_PER_EDGE = 10
MAX_TRIES = 20
SEED_STRIDE = 1_000_003


def role_indices(conn: Connectome, role: str) -> np.ndarray:
    roles = conn.meta.get("roles", {})
    parts = [np.zeros(0, dtype=np.int64)]
    for name, idx in conn.groups.items():
        if roles.get(name) == role:
            parts.append(np.asarray(idx, dtype=np.int64))
    return np.unique(np.concatenate(parts))


def path_check(conn: Connectome, inputs: np.ndarray, outputs: np.ndarray, hops: int) -> float:
    """Number of directed input-to-output walks with 1..hops synapses (binary adjacency, signs ignored)."""
    A = sparse.csr_matrix((conn.weights != 0).astype(np.float64))  # (post, pre)
    walks = np.zeros(conn.n)
    walks[inputs] = 1.0
    total = 0.0
    for i in range(hops):
        walks = A @ walks
        total += float(walks[outputs].sum())
    return total


def _hop_distance(A: sparse.csr_matrix, seeds: np.ndarray, max_hops: int) -> np.ndarray:
    """Hop distance from `seeds` along A (post, pre), capped at max_hops + 1."""
    dist = np.full(A.shape[0], max_hops + 1, dtype=np.int64)
    dist[seeds] = 0
    frontier = np.zeros(A.shape[0])
    frontier[seeds] = 1.0
    for i in range(max_hops):
        frontier = (A @ frontier > 0) & (dist > i + 1)
        dist[frontier] = i + 1
        frontier = frontier.astype(np.float64)
    return dist


def _edge_strata(conn: Connectome, pre: np.ndarray, post: np.ndarray, stratify: str) -> np.ndarray:
    """Stratum id per edge = (class of pre, class of post).

    "roles":  class = input / output / other.
    "layers": class = (hop distance from inputs, hop distance to outputs), capped at the core's hops + 1.
    """
    inputs = role_indices(conn, "input")
    outputs = role_indices(conn, "output")
    if stratify == "roles":
        node_class = np.full(conn.n, 2, dtype=np.int64)
        node_class[inputs] = 0
        node_class[outputs] = 1
        n_classes = 3
    elif stratify == "layers":
        hops = int(conn.meta.get("hops", 1))
        A = sparse.csr_matrix((conn.weights != 0).astype(np.float64))
        d_in = _hop_distance(A, inputs, hops)
        d_out = _hop_distance(A.T.tocsr(), outputs, hops)
        node_class = d_in * (hops + 2) + d_out
        n_classes = (hops + 2) ** 2
    else:
        raise ValueError(f"unknown stratify mode {stratify!r} (use False, 'roles' or 'layers')")
    return node_class[pre] * n_classes + node_class[post]


def _contains(sorted_keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_keys, values)
    pos = np.minimum(pos, sorted_keys.size - 1)
    return sorted_keys[pos] == values


def _swap_rounds(pre: np.ndarray, post: np.ndarray, n: int, rng: np.random.Generator, rounds: int, strata: np.ndarray | None) -> int:
    """Run rounds of vectorized double-edge swaps in place on `post`. Returns the number of accepted swaps."""
    accepted = 0
    m = pre.size
    for r in range(rounds):
        order = rng.permutation(m)
        if strata is not None:
            order = order[np.argsort(strata[order], kind="stable")]
        e1 = order[0 : m - 1 : 2]
        e2 = order[1:m:2]

        a, b = pre[e1], post[e1]
        c, d = pre[e2], post[e2]
        ok = (b != d) & (a != d) & (c != b)
        if strata is not None:
            ok &= strata[e1] == strata[e2]

        keys = np.sort(pre * n + post)
        new1 = a * n + d
        new2 = c * n + b
        ok &= ~_contains(keys, new1) & ~_contains(keys, new2)

        # two accepted swaps in the same round must not create the same edge
        cand = np.sort(np.concatenate([new1[ok], new2[ok]]))
        dup = cand[1:][cand[1:] == cand[:-1]]
        if dup.size:
            dup = np.unique(dup)
            ok &= ~_contains(dup, new1) & ~_contains(dup, new2)

        post[e1[ok]] = d[ok]
        post[e2[ok]] = b[ok]
        accepted += int(ok.sum())
    return accepted


def _shuffle_once(conn: Connectome, seed: int, stratify: bool | str, rounds_per_edge: int) -> tuple[Connectome, int]:
    W = conn.weights.tocoo()
    post = W.row.astype(np.int64).copy()
    pre = W.col.astype(np.int64).copy()
    data = W.data.astype(np.float32).copy()

    strata = None
    if stratify:
        strata = _edge_strata(conn, pre, post, stratify)

    rng = np.random.default_rng(seed)
    accepted = _swap_rounds(pre, post, conn.n, rng, 2 * rounds_per_edge, strata)

    W_new = sparse.csc_matrix((data, (post, pre)), shape=conn.weights.shape, dtype=np.float32)
    shuffled = Connectome(
        name=f"{conn.name}-shuf{seed}",
        weights=W_new,
        types=conn.types.copy(),
        sides=conn.sides.copy(),
        superclass=None if conn.superclass is None else conn.superclass.copy(),
        body_ids=None if conn.body_ids is None else conn.body_ids.copy(),
        groups={k: np.asarray(v, dtype=np.int64).copy() for k, v in conn.groups.items()},
        meta=dict(conn.meta),
    )
    return shuffled, accepted


def degree_preserving_shuffle(
    connectome: Connectome,
    seed: int,
    hops: int | None = None,
    max_tries: int = MAX_TRIES,
    stratify: bool | str = False,
    rounds_per_edge: int = ROUNDS_PER_EDGE,
) -> Connectome:
    """Shuffle `connectome` preserving in/out-degree per neuron, weights per presynaptic neuron and groups.

    Retries with a new seed until the shuffled graph has at least as many input-to-output walks within
    `hops` synapses (default 2 x the core's hop setting) as the original, up to `max_tries`; then warns
    and returns the try with the most walks.

    stratify="roles" only swaps edges whose (pre role, post role) match, roles being input / output / other.
    That also keeps each neuron's number of edges to and from input and output neurons, and therefore the
    exact number of 2-synapse input-to-output walks, while still scrambling which neurons connect.
    stratify="layers" uses (hop distance from inputs, hop distance to outputs) as the class instead,
    which keeps more of the feed-forward layering of a deeper core.
    """
    if hops is None:
        hops = 2 * int(connectome.meta.get("hops", 1))
    inputs = role_indices(connectome, "input")
    outputs = role_indices(connectome, "output")
    original_paths = path_check(connectome, inputs, outputs, hops)

    best = None
    best_paths = -1.0
    tries = 0
    for t in range(max_tries):
        tries = t + 1
        try_seed = seed + t * SEED_STRIDE
        shuffled, accepted = _shuffle_once(connectome, try_seed, stratify, rounds_per_edge)
        paths = path_check(shuffled, inputs, outputs, hops)
        shuffled.meta["shuffle"] = {
            "seed": int(seed),
            "try_seed": int(try_seed),
            "tries": tries,
            "stratify": stratify if stratify else False,
            "accepted_swaps": accepted,
            "edges": int(connectome.n_connections),
            "path_hops": hops,
            "paths_original": original_paths,
            "paths_shuffled": paths,
            "path_ratio": paths / max(original_paths, 1.0),
        }
        if paths > best_paths:
            best = shuffled
            best_paths = paths
        if paths >= original_paths:
            shuffled.meta["shuffle"]["passed_path_check"] = True
            return shuffled

    best.meta["shuffle"]["passed_path_check"] = False
    warnings.warn(
        f"degree_preserving_shuffle: no try out of {max_tries} reached the original {original_paths:.0f} "
        f"input-to-output walks within {hops} synapses; best try has {best_paths:.0f} "
        f"({best_paths / max(original_paths, 1.0):.2f} of original). Consider stratify='roles' or 'layers'.",
        stacklevel=2,
    )
    return best
