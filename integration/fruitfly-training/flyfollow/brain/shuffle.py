"""Degree-preserving shuffles of a pursuit core for the FLY-SHUF arm (plan 4.7).

    python -m flyfollow.brain.shuffle --hops 1            # pursuit_core1_shuf{1,2,3}.npz
    python -m flyfollow.brain.shuffle --hops 1 2 --n 3

Null model: vectorized double-edge swaps (a->b, c->d) -> (a->d, c->b), done separately within the
excitatory and inhibitory edge sets, so every neuron keeps its in-degree and out-degree in each sign
set (and hence Dale's law: a neuron's outgoing edges keep their sign). No self-loops, no duplicate
(pre, post) pairs across the two sets. Synapse-count weights are then randomly permuted within each
sign set. Neuron order, groups and meta (including azimuth_bins) are unchanged.

Path check (plan 4.7): the shuffle must have at least as many input-to-output walks of length
<= 2 * hops synapses (the core's hop limit) as the real core. If a plain shuffle fails, we retry with
new seeds; if `--mode auto` and plain keeps failing, we fall back to a *layered* shuffle that also
stratifies swaps by the (distance from inputs, distance to outputs) layer of both endpoints, which
still keeps every per-sign degree but also keeps the input -> relay -> output layering. The mode,
seed, swap statistics and path counts are recorded in meta["shuffle"].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from flyfollow.brain.build import core_path, ensure_flydrones
from flyfollow.interfaces import INPUT_GROUPS, OUTPUT_GROUPS

ensure_flydrones()

from flydrones.brain.connectome import Connectome


def shuf_path(hops: int, k: int) -> Path:
    return core_path(hops).with_name(f"pursuit_core{hops}_shuf{k}.npz")


def io_indices(c: Connectome) -> tuple[np.ndarray, np.ndarray]:
    ins = np.unique(np.concatenate([c.group(g) for g in INPUT_GROUPS]))
    outs = np.unique(np.concatenate([c.group(g) for g in OUTPUT_GROUPS]))
    return ins, outs


def path_counts(W: sparse.spmatrix, ins: np.ndarray, outs: np.ndarray, max_len: int) -> dict:
    """Input -> output walks of length 1..max_len (binary adjacency) and reachable (input, output) pairs."""
    A = sparse.csr_matrix((W != 0).astype(np.float64))  # (post, pre)
    x = np.zeros(A.shape[0])
    x[ins] = 1.0
    walks = []
    for _ in range(max_len):
        x = A @ x
        walks.append(float(x[outs].sum()))
    # pairs: for each output, how many inputs reach it within max_len (backward BFS)
    AT = sparse.csr_matrix(A.T)
    pairs = 0
    per_out = {}
    for o in outs:
        seen = np.zeros(A.shape[0], dtype=bool)
        f = np.zeros(A.shape[0])
        f[o] = 1.0
        for _ in range(max_len):
            f = (AT @ f > 0).astype(np.float64)
            f[seen] = 0
            seen |= f > 0
        n = int(seen[ins].sum())
        pairs += n
        per_out[int(o)] = n
    return {"walks_by_len": walks, "walks": float(sum(walks)), "pairs": int(pairs)}


def _layers(c: Connectome, hops: int) -> np.ndarray:
    """Layer label per neuron: (distance from inputs, distance to outputs), each capped at hops + 1."""
    ins, outs = io_indices(c)
    A = sparse.csr_matrix((c.weights != 0).astype(np.float32))

    def dist(M, seeds):
        d = np.full(c.n, hops + 1, dtype=np.int64)
        d[seeds] = 0
        f = np.zeros(c.n, np.float32)
        f[seeds] = 1
        seen = f > 0
        for k in range(1, hops + 1):
            f = (M @ f > 0).astype(np.float32)
            f[seen] = 0
            new = f > 0
            d[new] = k
            seen |= new
        return d

    d_in = dist(A, ins)
    d_out = dist(sparse.csr_matrix(A.T), outs)
    return d_in * (hops + 2) + d_out


def shuffle_core(c: Connectome, seed: int, rounds: int = 30, layered: bool = False, hops: int = 1) -> tuple[Connectome, dict]:
    rng = np.random.default_rng(seed)
    W = c.weights.tocoo()
    post, pre, w = W.row.astype(np.int64), W.col.astype(np.int64), W.data.astype(np.float32)
    keep = w != 0
    post, pre, w = post[keep], pre[keep], w[keep]
    n = c.n
    sign = (w > 0).astype(np.int64)  # 1 excitatory, 0 inhibitory
    cls = sign.copy()
    if layered:
        lab = _layers(c, hops)
        nl = int(lab.max()) + 1
        cls = (sign * nl + lab[pre]) * nl + lab[post]
    E = post.size
    keys = np.sort(pre * n + post)
    accepted_total = 0
    t0 = time.perf_counter()
    for _ in range(rounds):
        order = np.lexsort((rng.random(E), cls))
        i, j = order[0 : E - 1 : 2], order[1:E:2]
        m = cls[i] == cls[j]
        i, j = i[m], j[m]
        a, b, cc, d = pre[i], post[i], pre[j], post[j]
        ok = (a != d) & (cc != b) & (a != cc) & (b != d)
        k1, k2 = a * n + d, cc * n + b
        pos1 = np.minimum(np.searchsorted(keys, k1), keys.size - 1)
        pos2 = np.minimum(np.searchsorted(keys, k2), keys.size - 1)
        ok &= keys[pos1] != k1
        ok &= keys[pos2] != k2
        # proposals must not collide with each other
        allk = np.concatenate([k1[ok], k2[ok]])
        _, inv, cnt = np.unique(allk, return_inverse=True, return_counts=True)
        dup = cnt[inv] > 1
        nok = int(ok.sum())
        bad = dup[:nok] | dup[nok:]
        idx_ok = np.flatnonzero(ok)
        ok[idx_ok[bad]] = False
        ii, jj = i[ok], j[ok]
        new_post_i, new_post_j = post[jj].copy(), post[ii].copy()
        post[ii], post[jj] = new_post_i, new_post_j
        accepted_total += int(ok.sum())
        keys = np.sort(pre * n + post)
    # sanity: no duplicates, no self-loops
    assert np.unique(keys).size == keys.size, "duplicate edges after shuffle"
    assert not np.any(pre == post), "self-loop after shuffle"
    # weights permuted within each sign set
    new_w = np.empty_like(w)
    for sgn in (0, 1):
        s = np.flatnonzero(sign == sgn)
        new_w[s] = w[s][rng.permutation(s.size)]
    Ws = sparse.csc_matrix((new_w, (post, pre)), shape=(n, n), dtype=np.float32)
    out = Connectome(
        name=f"{c.name}-shuf-s{seed}",
        weights=Ws,
        types=c.types,
        sides=c.sides,
        superclass=c.superclass,
        body_ids=c.body_ids,
        groups={k: v.copy() for k, v in c.groups.items()},
        meta=json.loads(json.dumps(c.meta)),
    )
    stats = {"edges": int(E), "rounds": rounds, "swaps_accepted": accepted_total,
             "swaps_per_edge": accepted_total * 2 / max(E, 1), "seconds": time.perf_counter() - t0}
    return out, stats


def degree_signature(c: Connectome) -> dict[str, np.ndarray]:
    """Per-neuron in/out degree within the excitatory and inhibitory sets."""
    W = c.weights.tocoo()
    out = {}
    for name, m in (("exc", W.data > 0), ("inh", W.data < 0)):
        out[f"{name}_in"] = np.bincount(W.row[m], minlength=c.n)
        out[f"{name}_out"] = np.bincount(W.col[m], minlength=c.n)
    return out


def make_shuffles(hops: int, n_shuf: int, mode: str, base_seed: int, max_tries: int, rounds: int) -> list[dict]:
    real = Connectome.load(core_path(hops))
    ins, outs = io_indices(real)
    L = 2 * hops
    real_pc = path_counts(real.weights, ins, outs, L)
    print(f"core{hops}: {real.n:,} neurons, {real.n_connections:,} edges; real walks<= {L}: {real_pc['walks']:.4g} "
          f"(by length {[f'{x:.3g}' for x in real_pc['walks_by_len']]}), io pairs {real_pc['pairs']}")
    sig_real = degree_signature(real)
    reports = []
    seed = base_seed
    for k in range(1, n_shuf + 1):
        tries = []
        done = None
        for t in range(max_tries * (2 if mode == "auto" else 1)):
            layered = mode == "layered" or (mode == "auto" and t >= max_tries)
            sh, st = shuffle_core(real, seed, rounds=rounds, layered=layered, hops=hops)
            pc = path_counts(sh.weights, ins, outs, L)
            passed = pc["walks"] >= real_pc["walks"]
            tries.append({"seed": seed, "layered": layered, "walks": pc["walks"], "pairs": pc["pairs"], "passed": passed})
            print(f"  shuf{k} seed {seed} {'layered' if layered else 'plain'}: {st['swaps_per_edge']:.1f} swaps/edge in "
                  f"{st['seconds']:.1f} s, walks {pc['walks']:.4g} (by length {[f'{x:.3g}' for x in pc['walks_by_len']]}), "
                  f"pairs {pc['pairs']} -> {'PASS' if passed else 'fail'}")
            seed += 1
            if passed:
                done = (sh, st, pc, layered)
                break
        if done is None:
            print(f"  shuf{k}: no seed passed the path check in {len(tries)} tries; not saved")
            reports.append({"k": k, "saved": False, "tries": tries})
            continue
        sh, st, pc, layered = done
        sig = degree_signature(sh)
        assert all(np.array_equal(sig[x], sig_real[x]) for x in sig), "degree signature changed"
        sh.meta["shuffle"] = {
            "of": core_path(hops).name,
            "k": k,
            "mode": "layered" if layered else "plain",
            "seed": tries[-1]["seed"],
            "tries": tries,
            "stats": st,
            "real_paths": real_pc,
            "shuf_paths": pc,
            "path_len_limit": L,
            "degree_preserved_per_sign": True,
        }
        sh.name = f"pursuit-core{hops}-shuf{k}"
        p = sh.save(shuf_path(hops, k))
        print(f"  saved -> {p}")
        reports.append({"k": k, "saved": True, "path": str(p), **sh.meta["shuffle"]})
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flyfollow.brain.shuffle", description=__doc__.splitlines()[0])
    p.add_argument("--hops", type=int, nargs="+", default=[1])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--mode", choices=["plain", "layered", "auto"], default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-tries", type=int, default=5, help="seeds per shuffle (per mode in auto)")
    p.add_argument("--rounds", type=int, default=30, help="swap rounds (each proposes about E/2 swaps)")
    args = p.parse_args(argv)
    for h in args.hops:
        make_shuffles(h, args.n, args.mode, args.seed + 1000 * h, args.max_tries, args.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
