"""Thin wrapper around the FlyDrones LIF simulator for named neuron groups.

We bypass `flydrones.brain.Brain` because it re-resolves groups from regex
config specs, and our input groups (LC10a azimuth bins) are index lists built
by `flyfollow.brain.build`, stored in the connectome file itself.

The connectome weights are never modified here or anywhere else in training.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from flydrones.brain.connectome import Connectome
from flydrones.brain.lif import LIFNetwork, LIFParams


@lru_cache(maxsize=8)
def load_connectome(path: str) -> Connectome:
    return Connectome.load(Path(path))


class BrainRunner:
    """A connectome plus LIF state, driven by per-group Poisson rates and read out as per-group rates."""

    def __init__(self, connectome: Connectome, lif: dict | None = None, seed: int = 0, net: LIFNetwork | None = None):
        self.c = connectome
        self.params = LIFParams.from_dict(lif)
        self.net = net if net is not None else LIFNetwork(connectome.weights, self.params, seed=seed)

    @classmethod
    def from_file(cls, path: str | Path, lif: dict | None = None, seed: int = 0) -> BrainRunner:
        return cls(load_connectome(str(path)), lif=lif, seed=seed)

    def fresh(self, seed: int) -> BrainRunner:
        """Same wiring, new state and noise (cheap: shares the sparse matrix)."""
        return BrainRunner(self.c, net=self.net.copy(seed=seed))

    def group(self, name: str) -> np.ndarray:
        return self.c.group(name)

    def has(self, name: str) -> bool:
        return self.c.group(name).size > 0

    def set_bias(self, group: str, mv: float) -> None:
        idx = self.c.group(group)
        if idx.size:
            self.net.set_bias(idx, float(mv))

    def tick(self, inputs: dict[str, float | np.ndarray], outputs: list[str], ms: float = 50.0) -> dict[str, float]:
        """Drive `inputs` (Hz per neuron, scalar or per-neuron array) for `ms`, return mean rate (Hz) per output group."""
        idx_all, rate_all = [], []
        for name, rates in inputs.items():
            idx = self.c.group(name)
            if idx.size == 0:
                continue
            idx_all.append(idx)
            rate_all.append(np.broadcast_to(np.asarray(rates, dtype=np.float32), idx.shape))
        if idx_all:
            self.net.set_input(np.concatenate(idx_all), np.concatenate(rate_all))
        else:
            self.net.set_input(np.zeros(0, np.int64), 0.0)
        counts, _ = self.net.run(ms)
        out = {}
        for name in outputs:
            idx = self.c.group(name)
            out[name] = float(counts[idx].mean() * 1000.0 / ms) if idx.size else 0.0
        return out
