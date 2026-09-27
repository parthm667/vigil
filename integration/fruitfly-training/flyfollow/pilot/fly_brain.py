"""FlyBrain: lean per-episode wrapper around a pursuit subgraph .npz and FlyDrones' LIFNetwork.

flydrones.brain.Brain re-resolves groups with regexes over every neuron in its constructor, too
slow per episode. Here the connectome and a template LIFNetwork are loaded once per process
(cached by path) and every episode gets template.copy(seed), which shares the wiring.

Inputs arrive either as a dict {group: rate or per-neuron rates} (tick) or, on the fast path, as
the TargetEncoder's 22-channel vector (tick_channels), expanded to neurons with one precomputed
index. Outputs are mean rates (spikes * 1000 / ms) per OUTPUT_GROUPS entry.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from flydrones.brain.connectome import Connectome
from flydrones.brain.lif import LIFNetwork, LIFParams

from flyfollow.interfaces import INPUT_GROUPS, N_AZIMUTH_BINS, OUTPUT_GROUPS, azimuth_bins, brains_dir
from flyfollow.senses.target import CH_AROUSAL, CH_LC9, CH_LC11, N_CHANNELS

OPTIONAL_GROUPS = ("AROUSAL_L", "AROUSAL_R")


def resolve_brain_path(path: str | Path) -> Path:
    """Accept an existing path, or a bare file name / stem inside brains_dir()."""
    p = Path(path)
    if p.exists():
        return p.resolve()
    for cand in (brains_dir() / p.name, brains_dir() / f"{p.name}.npz"):
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"brain file {path} not found (also looked in {brains_dir()})")


def lc10a_bins(conn: Connectome, group: str) -> list[np.ndarray]:
    """LC10a azimuth bins (global neuron indices), bin 0 frontal.

    Uses meta["azimuth_bins"][group] when present: a list of N_AZIMUTH_BINS index lists, either
    global neuron indices (members of the group) or positions within the group. Otherwise ranks.
    """
    idx = conn.groups.get(group, np.zeros(0, np.int64))
    stored = (conn.meta.get("azimuth_bins") or {}).get(group)
    if stored is not None and len(stored) == N_AZIMUTH_BINS:
        flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in stored]) if any(len(b) for b in stored) else np.zeros(0, np.int64)
        if flat.size and np.isin(flat, idx).all():
            return [np.asarray(b, dtype=np.int64) for b in stored]
        if flat.size and flat.min() >= 0 and flat.max() < idx.size:
            return [idx[np.asarray(b, dtype=np.int64)] for b in stored]
        warnings.warn(f"meta azimuth_bins for {group} do not match the group; using rank bins", stacklevel=2)
    return azimuth_bins(idx)


@dataclass
class _Template:
    path: Path
    name: str
    n: int
    net: LIFNetwork
    in_idx: np.ndarray  # neurons receiving encoder channels
    in_chan: np.ndarray  # channel index per entry of in_idx
    group_idx: dict[str, np.ndarray]
    out_order: np.ndarray  # OUTPUT_GROUPS neurons concatenated
    out_starts: np.ndarray
    out_sizes: np.ndarray
    has_arousal: bool  # AROUSAL neurons exist (the encoder may still use the LC10a-gain fallback)
    group_sizes: dict[str, int]
    bins: dict[str, list[np.ndarray]]
    group_chan: dict[str, np.ndarray]  # per input group, channel of each neuron in group order (-1: none)
    connectome: Connectome


_CACHE: dict[tuple, _Template] = {}


def load_template(path: str | Path, lif: dict | None = None) -> _Template:
    p = resolve_brain_path(path)
    key = (str(p), p.stat().st_mtime_ns, tuple(sorted((lif or {}).items())))
    t = _CACHE.get(key)
    if t is not None:
        return t
    conn = Connectome.load(p)
    sizes = {g: int(conn.groups.get(g, np.zeros(0)).size) for g in INPUT_GROUPS + OUTPUT_GROUPS}
    empty = [g for g, k in sizes.items() if k == 0 and g not in OPTIONAL_GROUPS]
    if empty:
        raise ValueError(f"{p.name}: interface groups with no neurons: {', '.join(empty)} (sizes {sizes})")

    bins = {g: lc10a_bins(conn, g) for g in ("LC10a_L", "LC10a_R")}
    idx_parts, chan_parts = [], []
    for side_i, g in enumerate(("LC10a_L", "LC10a_R")):
        for k, b in enumerate(bins[g]):
            idx_parts.append(b)
            chan_parts.append(np.full(b.size, side_i * N_AZIMUTH_BINS + k, np.int64))
    for base, stem in ((CH_LC9, "LC9"), (CH_LC11, "LC11"), (CH_AROUSAL, "AROUSAL")):
        for side_i, s in enumerate("LR"):
            g = conn.groups.get(f"{stem}_{s}", np.zeros(0, np.int64))
            idx_parts.append(g)
            chan_parts.append(np.full(g.size, base + side_i, np.int64))
    in_idx = np.concatenate(idx_parts).astype(np.int64)
    in_chan = np.concatenate(chan_parts).astype(np.int64)
    assert in_chan.max() < N_CHANNELS

    chan_of = np.full(conn.n, -1, np.int64)
    chan_of[in_idx] = in_chan
    group_chan = {g: chan_of[np.asarray(conn.groups.get(g, np.zeros(0)), np.int64)] for g in INPUT_GROUPS}

    out = [np.asarray(conn.groups[g], np.int64) for g in OUTPUT_GROUPS]
    out_sizes = np.array([o.size for o in out], np.int64)
    out_starts = np.concatenate([[0], np.cumsum(out_sizes)[:-1]]).astype(np.int64)

    net = LIFNetwork(conn.weights, LIFParams.from_dict({**(conn.meta.get("lif") or {}), **(lif or {})}), seed=0)
    t = _Template(
        path=p,
        name=conn.name,
        n=conn.n,
        net=net,
        in_idx=in_idx,
        in_chan=in_chan,
        group_idx={g: np.asarray(conn.groups.get(g, np.zeros(0)), np.int64) for g in INPUT_GROUPS + OUTPUT_GROUPS},
        out_order=np.concatenate(out),
        out_starts=out_starts,
        out_sizes=out_sizes,
        has_arousal=sizes["AROUSAL_L"] + sizes["AROUSAL_R"] > 0,
        group_sizes=sizes,
        bins=bins,
        group_chan=group_chan,
        connectome=conn,
    )
    _CACHE[key] = t
    return t


class FlyBrain:
    """One simulated fly for one episode. reset(seed) gives fresh state and noise, same wiring."""

    def __init__(self, path: str | Path, seed: int = 0, lif: dict | None = None):
        self.t = load_template(path, lif)
        self.has_arousal = self.t.has_arousal
        self.n = self.t.n
        self.path = self.t.path
        self.brain_s = 0.0
        self.last_counts = np.zeros(self.n, np.int32)
        self.reset(seed)

    @property
    def connectome(self) -> Connectome:
        """The loaded subgraph (shared by every FlyBrain on this file in the process; do not mutate)."""
        return self.t.connectome

    @property
    def groups(self) -> dict[str, np.ndarray]:
        return self.t.connectome.groups

    def input_rates_by_group(self, ch_rates: np.ndarray) -> dict[str, np.ndarray]:
        """22 channel rates -> {input group: per-neuron rate (Hz) in group index order}."""
        ext = np.append(np.asarray(ch_rates, np.float32), np.float32(0.0))  # index -1 -> 0 Hz
        return {g: ext[c] for g, c in self.t.group_chan.items()}

    def reset(self, seed: int) -> None:
        self.net = self.t.net.copy(seed=seed)
        self.brain_s = 0.0

    def _run(self, ms: float) -> np.ndarray:
        t0 = time.perf_counter()
        counts, _ = self.net.run(ms)
        self.brain_s += time.perf_counter() - t0
        self.last_counts = counts  # fresh array from run(), safe to keep without a copy
        c = counts[self.t.out_order].astype(np.float64)
        return np.add.reduceat(c, self.t.out_starts) / self.t.out_sizes * (1000.0 / ms)

    def tick_channels(self, ch_rates: np.ndarray, ms: float = 50.0) -> np.ndarray:
        """Fast path: 22 encoder channel rates -> 10 DN rates (Hz) in OUTPUT_GROUPS order."""
        self.net.set_input(self.t.in_idx, ch_rates[self.t.in_chan])
        return self._run(ms)

    def tick(self, input_rates: dict[str, np.ndarray | float], ms: float = 50.0) -> dict[str, float]:
        """{input group: rate or per-neuron rates} -> {output group: mean rate (Hz)}."""
        idx_all, rate_all = [], []
        for name, rates in input_rates.items():
            idx = self.t.group_idx.get(name)
            if idx is None:
                raise KeyError(f"unknown input group {name!r}")
            if idx.size:
                idx_all.append(idx)
                rate_all.append(np.broadcast_to(np.asarray(rates, dtype=np.float32), idx.shape))
        if idx_all:
            self.net.set_input(np.concatenate(idx_all), np.concatenate(rate_all))
        else:
            self.net.set_input(np.zeros(0, np.int64), 0.0)
        return dict(zip(OUTPUT_GROUPS, self._run(ms).tolist()))
