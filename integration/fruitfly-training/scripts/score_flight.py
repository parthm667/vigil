"""Score a ReachGlass run log from a real follow flight (docs/FLIGHT_TEST_CHECKLIST.md Section 3).

    .venv/bin/python scripts/score_flight.py ~/Documents/GitHub/jerkgt13/runs/r3_fly_1.jsonl [--distance 1.6]
    .venv/bin/python scripts/score_flight.py runs/r3_pid_*.jsonl runs/r3_fly_*.jsonl     # compare several

Range is the stack's own estimate from the person's size, so check it against the tape marks and the phone video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(path: Path, z: float) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("state") == "FOLLOW" and "event" not in r:
            rows.append(r)
    if not rows:
        return {"run": path.stem, "follow_s": 0.0}
    seen = [r for r in rows if r.get("person")]
    rng = [r["person"]["range"] for r in seen if r["person"].get("range") is not None]
    bearing = [abs(r["person"]["bearing"]) for r in seen if r["person"].get("bearing") is not None]
    gaps = [b["t"] - a["t"] for a, b in zip(seen, seen[1:])]
    return {
        "run": path.stem,
        "follow_s": rows[-1]["t"] - rows[0]["t"],
        "in_view": len(seen) / len(rows),
        "in_band": sum(abs(x - z) <= 0.15 * z for x in rng) / max(len(rng), 1),
        "min_range_m": min(rng, default=float("nan")),
        "mean_abs_bearing_deg": sum(bearing) / len(bearing) if bearing else float("nan"),
        "longest_loss_s": max(gaps, default=0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--distance", type=float, default=1.6, help="follow.distance_m flown")
    a = ap.parse_args()
    print(f"{'run':18s} {'FOLLOW s':>8s} {'in view':>8s} {'in band':>8s} {'min m':>6s} {'|bearing|':>9s} {'max loss s':>10s}")
    for p in a.logs:
        s = score(p, a.distance)
        if not s["follow_s"]:
            print(f"{s['run']:18s} no FOLLOW rows")
            continue
        print(f"{s['run']:18s} {s['follow_s']:8.0f} {s['in_view']:8.0%} {s['in_band']:8.0%} {s['min_range_m']:6.2f} "
              f"{s['mean_abs_bearing_deg']:8.1f}° {s['longest_loss_s']:10.1f}")
    print("PASS (plan R3/R4): in band >= 70 %, min range >= 1.0 m, no loss longer than 2 s")


if __name__ == "__main__":
    main()
