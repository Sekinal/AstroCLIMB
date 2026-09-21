"""Plot matched gold-only vs public4k runs from an aggregate receipt only.

Reads only the JSON receipt written by analysis/summarize_matched_runs.py
(validation macro-F1 aggregates, hashes, chosen scores). Never reads raw
logs, predictions, or private paths.

Usage:
  python analysis/plot_matched.py --input receipt.json --out-prefix out/matched

Writes <out-prefix>.pdf and <out-prefix>.png (matplotlib Agg).

Panels (plotted only for fully paired seeds, i.e. seeds where both arms
are complete in the receipt):
  (a) DEV-selected validation CXI macro-F1 over gold steps 279/558/837
      for each paired run; color by arm, linestyle by seed.
  (b) Per-seed paired chosen scores (gold_only vs public4k) connected
      by a seed-colored line, with the paired selected delta annotated.

Validation: expected-step uniqueness, finite scores in [0, 1], chosen and
final scores matching the plotted validation points, pair entries matching
their runs, and paired differences matching chosen-score arithmetic.
Missing values are never filled; violations raise ValueError.
"""

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ("gold_only", "public4k")
ARM_COLORS = {"gold_only": "#727272", "public4k": "#0072B2"}
ARM_LABELS = {"gold_only": "Gold only", "public4k": "Public4k \u2192 gold"}
SEED_LINESTYLES = {7: "-", 19: "--", 37: ":"}
SEED_FALLBACK_STYLES = ["-", "--", ":", "-."]
SEED_COLORS = {7: "#0072B2", 19: "#D55E00", 37: "#009E73"}
SEED_FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]
TOL = 1e-9


def expected_steps(horizon):
    if horizon == 837:
        return [279, 558, 837]
    if horizon % 3 == 0:
        return [horizon // 3, 2 * horizon // 3, horizon]
    return [horizon]


def seed_linestyle(seed, order):
    if seed in SEED_LINESTYLES:
        return SEED_LINESTYLES[seed]
    return SEED_FALLBACK_STYLES[order % len(SEED_FALLBACK_STYLES)]


def seed_color(seed, order):
    if seed in SEED_COLORS:
        return SEED_COLORS[seed]
    return SEED_FALLBACK_COLORS[order % len(SEED_FALLBACK_COLORS)]


def check_score(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number")
    f = float(value)
    if not math.isfinite(f):
        raise ValueError(f"{what} must be finite")
    if f < 0.0 or f > 1.0:
        raise ValueError(f"{what} out of range [0, 1]")
    return f


def load_receipt(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("receipt must be an object")
    return obj


def validate_receipt(receipt):
    horizon = receipt.get("expected_horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("receipt expected_horizon must be a positive int")
    expected = expected_steps(horizon)
    results = receipt.get("seed_results")
    if not isinstance(results, list) or not results:
        raise ValueError("receipt seed_results must be a nonempty list")
    pairs = receipt.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("receipt pairs must be a list")
    if not pairs:
        raise ValueError("no complete pairs to plot")
    seen = set()
    by_key = {}
    for r in results:
        if not isinstance(r, dict):
            raise ValueError("each seed result must be an object")
        arm = r.get("arm")
        seed = r.get("seed")
        if arm not in ARMS:
            raise ValueError(f"run arm must be one of {list(ARMS)}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("run seed must be an int")
        key = (arm, seed)
        if key in seen:
            raise ValueError(f"duplicate run for arm {arm!r} seed {seed!r}")
        seen.add(key)
        status = r.get("status")
        if status not in ("complete", "incomplete"):
            raise ValueError("run status must be complete or incomplete")
        if status != "complete":
            continue
        scores = r.get("validation_scores")
        if not isinstance(scores, list) or len(scores) != len(expected):
            raise ValueError(f"run {arm} seed {seed}: missing validation scores")
        step_to_macro = {}
        for entry in scores:
            if not isinstance(entry, dict):
                raise ValueError(f"run {arm} seed {seed}: bad validation entry")
            s = entry.get("step")
            m = entry.get("macro_f1")
            if isinstance(s, bool) or not isinstance(s, int):
                raise ValueError(f"run {arm} seed {seed}: validation step must be an int")
            m = check_score(m, f"run {arm} seed {seed}: validation macro")
            if s in step_to_macro:
                raise ValueError(f"run {arm} seed {seed}: duplicated validation steps")
            step_to_macro[s] = m
        if sorted(step_to_macro) != sorted(expected):
            raise ValueError(
                f"run {arm} seed {seed}: validation steps "
                f"{sorted(step_to_macro)} != expected {sorted(expected)}"
            )
        chosen_step = r.get("chosen_step")
        if isinstance(chosen_step, bool) or not isinstance(chosen_step, int):
            raise ValueError(f"run {arm} seed {seed}: chosen_step must be an int")
        if chosen_step not in step_to_macro:
            raise ValueError(f"run {arm} seed {seed}: chosen_step not in validation steps")
        chosen_score = check_score(r.get("chosen_score"), f"run {arm} seed {seed}: chosen_score")
        if abs(chosen_score - step_to_macro[chosen_step]) > TOL:
            raise ValueError(f"run {arm} seed {seed}: selected score must match chosen validation")
        final_score = check_score(
            r.get("final_epoch_score"), f"run {arm} seed {seed}: final_epoch_score"
        )
        if horizon not in step_to_macro:
            raise ValueError(f"run {arm} seed {seed}: missing final epoch validation")
        if abs(final_score - step_to_macro[horizon]) > TOL:
            raise ValueError(f"run {arm} seed {seed}: final score must match final validation")
        by_key[key] = {
            "chosen_step": chosen_step,
            "chosen_score": chosen_score,
            "final_epoch_score": final_score,
            "curve": sorted(step_to_macro.items()),
        }
    validated_pairs = []
    pair_seeds = set()
    for p in pairs:
        if not isinstance(p, dict):
            raise ValueError("each pair must be an object")
        seed = p.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("pair seed must be an int")
        if seed in pair_seeds:
            raise ValueError(f"duplicate pair for seed {seed!r}")
        pair_seeds.add(seed)
        sides = {}
        for arm in ARMS:
            side = p.get(arm)
            if not isinstance(side, dict):
                raise ValueError(f"pair seed {seed}: missing {arm} side")
            cs = side.get("chosen_step")
            if isinstance(cs, bool) or not isinstance(cs, int):
                raise ValueError(f"pair seed {seed}: {arm} chosen_step must be an int")
            cscore = check_score(side.get("chosen_score"), f"pair seed {seed}: {arm} chosen_score")
            fscore = check_score(
                side.get("final_epoch_score"), f"pair seed {seed}: {arm} final_epoch_score"
            )
            run = by_key.get((arm, seed))
            if run is None:
                raise ValueError(f"pair seed {seed}: {arm} run is not complete")
            if cs != run["chosen_step"] or abs(cscore - run["chosen_score"]) > TOL:
                raise ValueError(f"pair seed {seed}: {arm} selected score must match run")
            if abs(fscore - run["final_epoch_score"]) > TOL:
                raise ValueError(f"pair seed {seed}: {arm} final score must match run")
            sides[arm] = {"chosen_step": cs, "chosen_score": cscore, "final_epoch_score": fscore}
        diff = p.get("paired_difference")
        if isinstance(diff, bool) or not isinstance(diff, (int, float)):
            raise ValueError(f"pair seed {seed}: paired_difference must be a number")
        if not math.isfinite(float(diff)):
            raise ValueError(f"pair seed {seed}: paired_difference must be finite")
        want = sides["public4k"]["chosen_score"] - sides["gold_only"]["chosen_score"]
        if abs(float(diff) - want) > TOL:
            raise ValueError(f"pair seed {seed}: paired difference must match chosen scores")
        fdiff = p.get("paired_final_epoch_difference")
        if isinstance(fdiff, bool) or not isinstance(fdiff, (int, float)):
            raise ValueError(f"pair seed {seed}: paired_final_epoch_difference must be a number")
        if not math.isfinite(float(fdiff)):
            raise ValueError(f"pair seed {seed}: paired_final_epoch_difference must be finite")
        want_f = sides["public4k"]["final_epoch_score"] - sides["gold_only"]["final_epoch_score"]
        if abs(float(fdiff) - want_f) > TOL:
            raise ValueError(f"pair seed {seed}: final epoch difference must match final scores")
        validated_pairs.append(
            {
                "seed": seed,
                "gold_only": sides["gold_only"],
                "public4k": sides["public4k"],
                "paired_difference": float(diff),
                "paired_final_epoch_difference": float(fdiff),
            }
        )
    validated_pairs.sort(key=lambda d: d["seed"])
    return horizon, expected, by_key, validated_pairs, len(results)


def render(receipt, out_prefix):
    horizon, expected, by_key, pairs, n_runs = validate_receipt(receipt)
    pair_seeds = [p["seed"] for p in pairs]
    seed_order = {s: i for i, s in enumerate(sorted(pair_seeds))}

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.subplots_adjust(left=0.07, bottom=0.24, right=0.96, top=0.88, wspace=0.22)

    plotted_scores = []
    for (arm, seed), run in by_key.items():
        if seed not in set(pair_seeds):
            continue
        plotted_scores.extend([m for _, m in run["curve"]])
        plotted_scores.append(run["chosen_score"])
    for p in pairs:
        plotted_scores.append(p["gold_only"]["chosen_score"])
        plotted_scores.append(p["public4k"]["chosen_score"])
    lo = min(plotted_scores) - 0.04
    hi = max(plotted_scores) + 0.04
    lo = max(0.0, math.floor(lo * 100) / 100.0)
    hi = min(1.0, math.ceil(hi * 100) / 100.0)
    if hi - lo < 0.08:
        mid = (lo + hi) / 2.0
        lo = max(0.0, mid - 0.04)
        hi = min(1.0, mid + 0.04)
    lo = min(lo, 0.50)
    hi = max(hi, 0.80)
    shared_ylim = (lo, hi)

    ax = axes[0]
    for (arm, seed) in sorted(by_key, key=lambda k: (k[1], k[0])):
        if seed not in set(pair_seeds):
            continue
        run = by_key[(arm, seed)]
        xs = [s for s, _ in run["curve"]]
        ys = [m for _, m in run["curve"]]
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            color=ARM_COLORS[arm],
            linestyle=seed_linestyle(seed, seed_order[seed]),
            linewidth=1.4,
            label=f"{ARM_LABELS[arm]} seed {seed}",
        )
    ax.set(
        title="(a) Gold adaptation learning curves",
        xlabel="Gold step",
        ylabel="CXI macro-F1 (DEV)",
        xticks=list(expected),
        ylim=shared_ylim,
    )
    ax.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower left", ncol=2, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    bx = axes[1]
    for p in pairs:
        seed = p["seed"]
        color = seed_color(seed, seed_order[seed])
        g = p["gold_only"]["chosen_score"]
        u = p["public4k"]["chosen_score"]
        bx.plot(
            [0, 1],
            [g, u],
            marker="o",
            markersize=4,
            color=color,
            linewidth=1.6,
            label=f"seed {seed}",
        )
    for i, p in enumerate(sorted(pairs, key=lambda d: d["seed"])):
        seed = p["seed"]
        color = seed_color(seed, seed_order[seed])
        bx.text(
            0.55,
            0.30 - 0.08 * i,
            f"seed {seed}: {p['paired_difference']:+.3f}",
            ha="left",
            va="center",
            fontsize=9,
            color=color,
            transform=bx.transAxes,
        )
    bx.set(
        title="(b) Selected checkpoints by seed",
        xlabel="Arm",
        ylabel="Chosen CXI macro-F1 (DEV)",
        xticks=[0, 1],
        xticklabels=["Gold only", "Public4k→gold"],
        xlim=(-0.08, 1.15),
        ylim=shared_ylim,
    )
    bx.grid(True, color="#DDDDDD", linewidth=0.5, alpha=0.9, axis="y")
    bx.set_axisbelow(True)
    bx.legend(fontsize=9, loc="best", framealpha=0.9)
    bx.spines[["top", "right"]].set_visible(False)

    per_seed = "; ".join(
        f"seed {p['seed']}: selected {p['paired_difference']:+.3f}, "
        f"final-epoch {p['paired_final_epoch_difference']:+.3f}"
        for p in pairs
    )
    plotted = 2 * len(pairs)
    excluded = n_runs - plotted
    footer = (
        f"Paired deltas (public4k minus gold_only): {per_seed}. "
        "Zoomed score axes. "
        "DEV-selected scores on the same gold schedule; public arm starts from "
        "the fixed seed-7 public parent with additional public compute; "
        "no independent test estimate."
    )
    if excluded > 0:
        footer += f" Excluded {excluded} incomplete/unpaired runs (not plotted)."
    footer = textwrap.fill(footer, width=135)
    fig.text(0.01, 0.015, footer, ha="left", va="bottom", fontsize=9, wrap=True)

    out_prefix = Path(out_prefix)
    if out_prefix.parent != Path(""):
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_prefix) + ".pdf")
    fig.savefig(str(out_prefix) + ".png", dpi=160)
    plt.close(fig)
    return str(out_prefix) + ".pdf", str(out_prefix) + ".png"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Aggregate receipt JSON")
    ap.add_argument("--out-prefix", type=Path, required=True, help="Output path prefix")
    args = ap.parse_args()
    receipt = load_receipt(args.input)
    render(receipt, args.out_prefix)


if __name__ == "__main__":
    main()
