"""Comparison chart for the slides (plan 4.7).

    python -m flyfollow.rl.chart --name final                 # data/runs/eval/final.json -> final.png
    python -m flyfollow.rl.chart --name final --set demo
    python -m flyfollow.rl.chart --name final --arms FLY-YAW,FLY-CMA,NOBRAIN-YAW,PID-CMA
    python -m flyfollow.rl.chart --curves-only --tag v1       # learning curves while training runs

Top row: the four headline metrics on one evaluation set (follow in-band fraction, approach
success, loss events per minute, minimum distance). Columns are the chart.arms (default the
yaw-only comparison: FLY-YAW, FLY-SHUF-YAW, NOBRAIN-YAW, PID-CMA); a trained arm's column is the
mean across run seeds, the whisker the range across seeds and the dots each seed. The untrained
references (chart.references: FLY-YAW-HAND, PID-HAND) are labeled horizontal lines.
Bottom row: learning curves, one panel per trained column arm (population median fitness per
generation, one line style per run seed; selection fitness of the CMA mean as dots), shared y axis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from flyfollow.interfaces import TRAINED_ARMS, YAW_ONLY_ARMS, data_root
from flyfollow.rl import cma_train as ct

DEFAULT_ARMS = ("FLY-YAW", "FLY-SHUF-YAW", "NOBRAIN-YAW", "PID-CMA")
DEFAULT_REFS = ("FLY-YAW-HAND", "PID-HAND")
# Readable names for slides. Yaw-only arms: the named controller steers, the PID sets forward.
LABELS = {
    "FLY-YAW": "Fly steers (trained)",
    "FLY-SHUF-YAW": "Shuffled fly steers",
    "NOBRAIN-YAW": "No brain steers",
    "PID-CMA": "PID (tuned)",
    "FLY-YAW-HAND": "Fly steers (untrained)",
    "PID-HAND": "PID (hand)",
    "FLY-CMA": "Fly flies (trained)",
    "FLY-SHUF": "Shuffled fly flies",
    "NOBRAIN": "No brain flies",
    "FLY-HAND": "Fly flies (untrained)",
}
TICK_LABELS = {
    "FLY-YAW": "Fly steers\n(trained)",
    "FLY-SHUF-YAW": "Shuffled fly\nsteers",
    "NOBRAIN-YAW": "No brain\nsteers",
    "PID-CMA": "PID\n(tuned)",
    "FLY-YAW-HAND": "Fly steers\n(untrained)",
    "PID-HAND": "PID\n(hand)",
    "FLY-CMA": "Fly flies\n(trained)",
    "FLY-SHUF": "Shuffled fly\nflies",
    "NOBRAIN": "No brain\nflies",
    "FLY-HAND": "Fly flies\n(untrained)",
}
# Categorical slots of the validated reference palette, fixed per arm (color follows the entity).
# Slots 1 to 4 (the default columns) pass the adjacent CVD and normal-vision checks.
ARM_COLORS = {
    "FLY-YAW": "#2a78d6",
    "FLY-SHUF-YAW": "#eb6834",
    "NOBRAIN-YAW": "#1baf7a",
    "PID-CMA": "#eda100",
    "FLY-CMA": "#e87ba4",
    "FLY-SHUF": "#008300",
    "NOBRAIN": "#4a3aa7",
    "FLY-HAND": "#e34948",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#f0efec"
REF_DASH = {"FLY-YAW-HAND": (0, (5, 3)), "PID-HAND": (0, (1.5, 2)), "FLY-HAND": (0, (8, 3, 2, 3))}
# (config chart key, report key prefix, panel title, y label, value format)
HEADLINE = (
    ("follow_in_band", "follow", "Follow: time in band", "fraction of episode", "{:.2f}"),
    ("approach_success", "approach", "Approach: success rate", "fraction of episodes", "{:.2f}"),
    ("loss_per_min", "follow", "Follow: target losses", "loss events per minute", "{:.1f}"),
    ("bearing_rms", "follow", "Follow: steering error", "RMS bearing error (deg), lower is better", "{:.1f}"),
)
Z_MIN_PERSON_M = 1.0


def label(arm: str) -> str:
    return LABELS.get(arm, arm)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _metric_key(cfg: dict, chart_key: str, kind: str) -> str:
    name = (cfg.get("chart") or {}).get(chart_key, chart_key)
    if isinstance(name, list):
        name = name[0]
    return f"{kind}.{name}"


def _leaf(summary: dict, dotted: str):
    v = summary
    for part in dotted.split("."):
        if not isinstance(v, dict):
            return None
        v = v.get(part)
    return v


def _seed_values(report: dict, set_name: str, arm: str, key: str) -> list[float]:
    vals = [
        _leaf(report["summaries"][set_name][e["name"]], key)
        for e in report["entries"]
        if e["arm"] == arm and e["name"] in report["summaries"][set_name]
    ]
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def headline_panel(ax, report: dict, set_name: str, cfg: dict, spec: tuple, arms: list[str], refs: list[str]) -> None:
    chart_key, kind, title, ylabel, fmt = spec
    key = _metric_key(cfg, chart_key, kind)
    have = report["arms"][set_name]
    arms = [a for a in arms if a in have]
    tops = [0.0]
    for i, arm in enumerate(arms):
        agg = have[arm].get(key)
        if not agg:
            ax.annotate("n/a", (i, 0), xytext=(0, 3), textcoords="offset points", ha="center", color=TEXT_2, fontsize=8)
            continue
        ax.bar(i, agg["mean"], width=0.62, color=ARM_COLORS.get(arm, TEXT_2), edgecolor=SURFACE, linewidth=2, zorder=2)
        seeds = _seed_values(report, set_name, arm, key)
        if agg["n_seeds"] > 1:
            ax.vlines(i, agg["min"], agg["max"], color=TEXT, linewidth=1.5, zorder=3)
            ax.scatter([i] * len(seeds), seeds, s=18, color=SURFACE, edgecolor=TEXT, linewidth=1.2, zorder=4)
        top = max([agg["mean"], agg["max"]] + seeds)
        tops.append(top)
        ax.annotate(fmt.format(agg["mean"]), (i, top), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                    color=TEXT, fontsize=8.5, zorder=8,
                    bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"})
    ref_vals = [(r, have[r][key]["mean"]) for r in refs if r in have and have[r].get(key)]
    tops += [v for _, v in ref_vals]
    if chart_key in ("follow_in_band", "approach_success"):
        ymax = 1.12
    else:
        ymax = max(tops) * 1.18 if max(tops) > 0 else 1.0
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(arms) - 0.4)
    if chart_key == "min_distance":
        ax.axhspan(0, Z_MIN_PERSON_M, color=BAND, zorder=0)
        ylabel = f"{ylabel}; shaded: under Z_min {Z_MIN_PERSON_M:.0f} m"
    # Untrained references: full-width lines, named once in the figure legend (direct labels
    # collided with the column values whenever the numbers were close).
    for ref, v in ref_vals:
        ax.axhline(v, color=TEXT_2, linewidth=1.4, linestyle=REF_DASH.get(ref, "--"), zorder=6)
    _style(ax)
    ax.set_xticks(np.arange(len(arms)), [TICK_LABELS.get(a, a) for a in arms], fontsize=8.5, color=TEXT)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_2)


def load_curves(tag: str, seeds: list[int], arms: list[str] | None = None) -> dict:
    """{arm: {seed: {"gen", "fit_mean", "fit_median", "fit_best", "sel_gen", "sel"}}} from the run logs."""
    out: dict = {}
    for arm in arms or TRAINED_ARMS:
        for seed in seeds:
            d = ct.runs_dir() / ct.run_name(arm, seed, tag)
            log = d / "log.jsonl"
            if not log.exists():
                continue
            recs = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            recs = [r for r in recs if "fit_mean" in r]
            sel = []
            if (d / "selection.jsonl").exists():
                sel = [json.loads(line) for line in (d / "selection.jsonl").read_text().splitlines() if line.strip()]
            out.setdefault(arm, {})[seed] = {
                "gen": [r["gen"] for r in recs],
                "fit_mean": [r["fit_mean"] for r in recs],
                "fit_median": [r["fit_median"] for r in recs],
                "fit_best": [r["fit_best"] for r in recs],
                "sel_gen": [r["gen"] for r in sel],
                "sel": [r["sel_fitness"] for r in sel],
            }
    return out


def curves_panels(axes, curves: dict, arms: list[str]) -> None:
    seed_dash = {1: "-", 2: (0, (5, 2)), 3: (0, (1.5, 1.5))}
    allv = [v for a in arms for s in curves.get(a, {}).values() for v in s["fit_median"] + s["sel"] if np.isfinite(v)]
    ylim = None
    if allv:
        lo, hi = np.percentile(allv, 2), max(allv)
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ylim = (lo - pad, hi + pad)
    for ax, arm in zip(axes, arms):
        _style(ax)
        ax.set_title(label(arm), loc="left", fontsize=11, color=TEXT, fontweight="bold")
        runs = curves.get(arm, {})
        if not runs:
            ax.text(0.5, 0.5, "no runs yet", transform=ax.transAxes, ha="center", va="center", color=TEXT_2, fontsize=9)
            continue
        for seed, c in sorted(runs.items()):
            ax.plot(c["gen"], c["fit_median"], color=ARM_COLORS.get(arm, TEXT_2), linewidth=2, linestyle=seed_dash.get(seed, "-"), label=f"seed {seed}")
            ax.scatter(c["sel_gen"], c["sel"], s=30, color=ARM_COLORS.get(arm, TEXT_2), edgecolor=TEXT, linewidth=0.8, zorder=3)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("generation", fontsize=9, color=TEXT_2)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="lower right")
    axes[0].set_ylabel("training fitness (higher is better)", fontsize=9, color=TEXT_2)
    from matplotlib.ticker import MaxNLocator

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def render(report: dict | None, curves: dict, cfg: dict, set_name: str, out: Path, arms: list[str], refs: list[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trained = [a for a in arms if a in TRAINED_ARMS] or list(DEFAULT_ARMS)
    fig = plt.figure(figsize=(16, 9.4 if report else 4.8), dpi=150, facecolor=SURFACE)
    if report:
        outer = fig.add_gridspec(2, 1, hspace=0.5, left=0.05, right=0.99, top=0.8, bottom=0.07, height_ratios=[1.1, 1])
        top = outer[0].subgridspec(1, len(HEADLINE), wspace=0.28)
        for i, spec in enumerate(HEADLINE):
            headline_panel(fig.add_subplot(top[0, i]), report, set_name, cfg, spec, arms, refs)
        bottom = outer[1].subgridspec(1, len(trained), wspace=0.28)
    else:
        bottom = fig.add_gridspec(1, len(trained), wspace=0.28, left=0.05, right=0.99, top=0.8, bottom=0.14)
    cax = [fig.add_subplot(bottom[0, i]) for i in range(len(trained))]
    for a in cax[1:]:
        a.sharey(cax[0])
    curves_panels(cax, curves, trained)
    yaw_note = (
        " In the \"steers\" arms that controller sets yaw only and the PID sets forward speed."
        if any(a in YAW_ONLY_ARMS for a in list(arms) + list(refs))
        else ""
    )
    f = cfg["fitness"]
    clip = f.get("clip")
    fit_note = (
        f"Fitness: each episode's return / max(|PID (hand) return on the same episode|, {float(f['denom_floor']):g})"
        + (f", clipped to [{clip[0]:g}, {clip[1]:g}]" if clip else "")
        + f"; {f['w_follow']:g} follow + {f['w_approach']:g} approach."
    )
    if report:
        n = f"{report['n_follow']} follow + {report['n_approach']} approach test episodes per controller"
        fig.suptitle(f"Pursuit controllers on the {set_name} set", x=0.05, ha="left", fontsize=15, color=TEXT, fontweight="bold")
        fig.text(
            0.05, 0.845,
            f"{n}.{yaw_note}\nTrained: column = mean across training runs, whisker = range, dots = each run. "
            "Gray lines = untrained references (legend, top right).\n"
            f"Bottom: training fitness (population median per generation) and selection fitness of the CMA mean (dots).\n{fit_note}",
            fontsize=9.5, color=TEXT_2, linespacing=1.5, va="bottom",
        )
        present = [r for r in refs if r in report["arms"].get(set_name, {})]
        if present:
            from matplotlib.lines import Line2D

            handles = [Line2D([0], [0], color=TEXT_2, linewidth=1.4, linestyle=REF_DASH.get(r, "--")) for r in present]
            fig.legend(handles, [label(r) for r in present], loc="upper right", bbox_to_anchor=(0.99, 0.975), ncol=len(present),
                       frameon=False, fontsize=9.5, labelcolor=TEXT, handlelength=3.2)
    else:
        fig.suptitle("Learning curves: population median fitness (lines), selection fitness of the CMA mean (dots)." + yaw_note,
                     x=0.05, ha="left", fontsize=12, color=TEXT)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m flyfollow.rl.chart", description="Comparison chart (plan 4.7).")
    p.add_argument("--name", default="final", help="eval report name under data/runs/eval/")
    p.add_argument("--set", default="test", help="evaluation set for the headline row")
    p.add_argument("--arms", default=None, help="comma-separated column arms (default chart.arms)")
    p.add_argument("--refs", default=None, help="comma-separated reference arms drawn as lines (default chart.references)")
    p.add_argument("--tag", default=None, help="run tag for the learning curves (default: the report's tag or config tag)")
    p.add_argument("--curves-only", action="store_true")
    p.add_argument("--out", default=None, help="output PNG (default data/runs/eval/<name>[_<set>].png)")
    args = p.parse_args(argv)
    cfg = ct.load_config()
    ch = cfg.get("chart") or {}
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else list(ch.get("arms") or DEFAULT_ARMS)
    refs = [a.strip() for a in args.refs.split(",")] if args.refs else list(ch.get("references") or DEFAULT_REFS)
    report = None
    if not args.curves_only:
        path = data_root() / "runs" / "eval" / f"{args.name}.json"
        report = json.loads(path.read_text())
        if args.set not in report["arms"]:
            raise SystemExit(f"set {args.set!r} not in {path.name}; has {list(report['arms'])}")
    tag = args.tag or (report or {}).get("tag") or cfg["tag"]
    curves = load_curves(tag, cfg["runs"]["seeds"], [a for a in arms if a in TRAINED_ARMS])
    suffix = "" if args.set == "test" else f"_{args.set}"
    default = data_root() / "runs" / "eval" / (f"curves_{tag}.png" if args.curves_only else f"{args.name}{suffix}.png")
    out = render(report, curves, cfg, args.set, Path(args.out) if args.out else default, arms, refs)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
