"""Build the pursuit brains (plan 4.1, item 2).

    python -m flyfollow.brain.build            # full brain (if missing) + pursuit_core1/2.npz
    python -m flyfollow.brain.build --force    # rebuild the full brain from the feather files too
    python -m flyfollow.brain.build bench      # ms of wall time per 50 ms brain tick for each core

Outputs (in data/brains/, or $FLYFOLLOW_DATA/brains):
- malecns_full.npz: the whole signed MaleCNS v1.0 connectome (pairs with >= 3 synapses), groups resolved.
- pursuit_core{h}.npz: `Connectome.sensorimotor_core(hops=h)` with inputs = INPUT_GROUPS and
  outputs = OUTPUT_GROUPS from flyfollow.interfaces. meta carries roles, group specs, the LC10a
  azimuth bins (meta["azimuth_bins"]) and build info.

Load a core with `flydrones.brain.connectome.Connectome.load(path)`; groups are in `.groups`,
bins in `.meta["azimuth_bins"]["LC10a_L"]` (8 lists of core indices, bin 0 most frontal).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from flyfollow.interfaces import INPUT_GROUPS, N_AZIMUTH_BINS, OUTPUT_GROUPS, REPO_ROOT, azimuth_bins, brains_dir, configs_dir


def ensure_flydrones() -> None:
    """Import flydrones, falling back to the vendored source tree.

    The venv's editable-install .pth files can carry the macOS "hidden" flag, which makes
    Python 3.12 skip them; then `import flydrones` fails although it is installed.
    """
    try:
        import flydrones
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "third_party" / "FlyDrones" / "src"))
        import flydrones  # noqa: F401


ensure_flydrones()

from flydrones.brain.connectome import Connectome, GroupSpec, build_malecns

DEFAULT_CONFIG = "pursuit_malecns.yaml"
FULL_NAME = "malecns_full.npz"
BIN_METHOD_RANK = "rank"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else configs_dir() / DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def group_specs(cfg: dict) -> dict[str, GroupSpec]:
    """All groups from the config, with role input / output / hidden (audit-only)."""
    specs: dict[str, GroupSpec] = {}
    for section, role in (("inputs", "input"), ("outputs", "output"), ("audit_groups", "hidden")):
        for name, d in (cfg.get(section) or {}).items():
            s = GroupSpec.from_dict(name, d)
            s.role = role
            specs[name] = s
    missing = [g for g in (*INPUT_GROUPS, *OUTPUT_GROUPS) if g not in specs]
    if missing:
        raise ValueError(f"config is missing frozen interface groups: {missing}")
    extra_io = [k for k, s in specs.items() if s.role != "hidden" and k not in (*INPUT_GROUPS, *OUTPUT_GROUPS)]
    if extra_io:
        raise ValueError(f"inputs/outputs must be exactly the interface groups; move these to audit_groups: {extra_io}")
    return specs


def resolve(c: Connectome, cfg: dict) -> dict[str, GroupSpec]:
    """Resolve every config group on `c` and set meta roles (audit groups stay hidden)."""
    specs = group_specs(cfg)
    c.groups = {}
    c.resolve_groups(specs)
    c.meta["roles"] = {k: s.role for k, s in specs.items() if s.role in ("input", "output")}
    c.meta["group_specs"] = {k: {"types": s.types, "side": s.side, "role": s.role} for k, s in specs.items()}
    return specs


def full_path() -> Path:
    return brains_dir() / FULL_NAME


def core_path(hops: int) -> Path:
    return brains_dir() / f"pursuit_core{hops}.npz"


def build_full(cfg: dict, data_dir: Path, force: bool = False) -> Connectome:
    out = full_path()
    if out.exists() and not force:
        t0 = time.perf_counter()
        c = Connectome.load(out)
        print(f"loaded {out} in {time.perf_counter() - t0:.1f} s: {c.summary()}")
        resolve(c, cfg)
        return c
    t0 = time.perf_counter()
    c = build_malecns(data_dir, min_synapses=int(cfg.get("brain", {}).get("min_synapses", 3)))
    print(f"built in {time.perf_counter() - t0:.1f} s: {c.summary()}")
    resolve(c, cfg)
    c.save(out)
    print(f"saved -> {out}")
    return c


def lc10a_bins(core: Connectome, method: str = BIN_METHOD_RANK) -> dict[str, list[list[int]]]:
    """8 azimuth bins per LC10a side, bin 0 most frontal.

    Only the rank fallback is implemented: no retinotopic coordinate for LC10a is available in the
    MaleCNS flat tables (assignedOlHex1/2 are empty for LC10a), see docs/audit_result.md.
    """
    if method != BIN_METHOD_RANK:
        raise ValueError(f"unknown bin method {method}")
    out = {}
    for g in ("LC10a_L", "LC10a_R"):
        out[g] = [b.tolist() for b in azimuth_bins(core.group(g), N_AZIMUTH_BINS)]
    return out


def group_sizes(c: Connectome, names) -> dict[str, int]:
    return {k: int(c.group(k).size) for k in names}


def build_core(full: Connectome, hops: int, cfg: dict) -> Connectome:
    t0 = time.perf_counter()
    core = full.sensorimotor_core(hops=hops)
    core.name = f"pursuit-core{hops}"
    core.meta = {
        **{k: v for k, v in full.meta.items() if k not in ("subgraph_neurons",)},
        "parent": full.name,
        "hops": hops,
        "subgraph_neurons": int(core.n),
        "subgraph_connections": int(core.n_connections),
        "input_groups": list(INPUT_GROUPS),
        "output_groups": list(OUTPUT_GROUPS),
        "azimuth_bins": lc10a_bins(core),
        "azimuth_bins_method": BIN_METHOD_RANK,
        "azimuth_bins_note": "rank within side (sorted core index); bin 0 treated as most frontal. No retinotopy data for LC10a.",
        "built_by": "flyfollow.brain.build",
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lif": cfg.get("brain", {}).get("lif", {}),
    }
    print(f"core hops={hops}: {core.n:,} neurons, {core.n_connections:,} connections ({time.perf_counter() - t0:.1f} s)")
    return core


def check_nonempty(c: Connectome) -> None:
    empty = [g for g in (*INPUT_GROUPS, *OUTPUT_GROUPS) if c.group(g).size == 0]
    if empty:
        raise SystemExit(f"refusing to save {c.name}: empty interface groups {empty}")


def print_sizes(c: Connectome) -> None:
    for k in sorted(c.groups, key=lambda k: (k not in INPUT_GROUPS, k not in OUTPUT_GROUPS, k)):
        role = c.meta.get("roles", {}).get(k, "audit")
        print(f"  {k:10s} {role:6s} {c.group(k).size:5d}")


def cmd_build(args) -> int:
    cfg = load_config(args.config)
    full = build_full(cfg, Path(args.data_dir), force=args.force)
    check_nonempty(full)
    print("groups in the full brain:")
    print_sizes(full)
    for h in args.hops:
        core = build_core(full, h, cfg)
        check_nonempty(core)
        out = core.save(core_path(h))
        print(f"saved -> {out}")
        bins = core.meta["azimuth_bins"]
        print("  LC10a bin sizes: " + "  ".join(f"{g}={[len(b) for b in bins[g]]}" for g in bins))
    return 0


# ------------------------------------------------------------------ bench
def bench_core(path: Path, ticks: int = 40, seed: int = 0) -> dict:
    """ms of wall time per 50 ms tick (100 LIF steps at dt 0.5 ms) with a right-side spot and arousal on."""
    from flydrones.brain.lif import LIFNetwork, LIFParams

    from flyfollow.audit.audit import SpotEncoder

    c = Connectome.load(path)
    net = LIFNetwork(c.weights, LIFParams.from_dict(c.meta.get("lif")), seed=seed)
    enc = SpotEncoder(c)
    idx, rates = enc.encode(np.radians(15.0), 1.0, arousal=True)
    net.set_input(idx, rates)
    net.run(500.0)  # settle
    walls = []
    for _ in range(ticks):
        t0 = time.perf_counter()
        net.run(50.0)
        walls.append((time.perf_counter() - t0) * 1000.0)
    w = np.asarray(walls)
    return {
        "brain": path.name,
        "neurons": int(c.n),
        "connections": int(c.n_connections),
        "ms_per_tick_mean": float(w.mean()),
        "ms_per_tick_median": float(np.median(w)),
        "ms_per_tick_p95": float(np.percentile(w, 95)),
        "spikes_per_tick": float(net.stats.spikes / net.stats.steps * 100),
    }


def cmd_bench(args) -> int:
    paths = [Path(p) for p in args.brains] if args.brains else [core_path(h) for h in (1, 2)]
    res = []
    for p in paths:
        if not p.exists():
            print(f"skip {p}: not built")
            continue
        r = bench_core(p, ticks=args.ticks)
        res.append(r)
        print(
            f"{r['brain']}: {r['neurons']:,} neurons, {r['connections']:,} connections -> "
            f"{r['ms_per_tick_mean']:.2f} ms per 50 ms tick (median {r['ms_per_tick_median']:.2f}, p95 {r['ms_per_tick_p95']:.2f}), "
            f"{r['spikes_per_tick']:.0f} spikes per tick"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flyfollow.brain.build", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--config", help=f"group config (default configs/{DEFAULT_CONFIG})")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "malecns_v1"))
    p.add_argument("--force", action="store_true", help="rebuild the full brain even if malecns_full.npz exists")
    p.add_argument("--hops", type=int, nargs="+", default=[1, 2])
    b = sub.add_parser("bench", help="ms per 50 ms brain tick for each core")
    b.add_argument("brains", nargs="*", help="core .npz files (default pursuit_core1/2)")
    b.add_argument("--ticks", type=int, default=40)
    b.add_argument("--json", help="also write results to this JSON file")
    args = p.parse_args(argv)
    if args.cmd == "bench":
        return cmd_bench(args)
    return cmd_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
