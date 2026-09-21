"""Regenerate the manuscript's vector figure from its published result values."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())
MATCHED = json.loads((HERE.parents[1] / "analysis" / "matched-gold-controls.json").read_text())
gold, public = {}, {}
for row in MATCHED["seed_results"]:
    if row["arm"] == "gold_only":
        gold[row["seed"]] = row["chosen_score"]
    elif row["arm"] == "public4k":
        public[row["seed"]] = row["chosen_score"]
SEEDS = [7, 19, 37]
assert all(s in gold and s in public for s in SEEDS)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "astroclimb-results", "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.4))
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.24, top=0.71, wspace=0.45)
blue, orange, gray = "#0072B2", "#D55E00", "#7F7F7F"
p = DATA["public_adaptation"]
left.scatter(p["public_updates"], p["scores"], s=30, color=blue, zorder=3)
left.scatter([4000], [p["scores"][1]], s=49, marker="D", color=orange, zorder=4)
for x, y in zip(p["public_updates"], p["scores"]):
    left.annotate(f"{y:.5f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
left.set(xticks=p["public_updates"], xticklabels=["2,000", "4,000", "8,000"],
         xlim=(1000, 9100), ylim=(0.66, 0.78), yticks=[0.66, 0.70, 0.74, 0.78],
         xlabel="Public adaptation updates", ylabel="CXI macro-F1")
left.set_title("(a) Historical adaptation duration", loc="left", pad=30)
left.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
left.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
ys = [2, 1, 0]
gold_vals = [gold[s] for s in SEEDS]
public_vals = [public[s] for s in SEEDS]
for y, s in zip(ys, SEEDS):
    right.plot([gold[s], public[s]], [y, y], color="0.65", linewidth=1.0, zorder=1)
right.scatter(gold_vals, ys, s=30, color=gray, zorder=3)
right.scatter(public_vals, ys, s=30, color=blue, zorder=3)
for y, s in zip(ys, SEEDS):
    right.annotate(f"+{public[s] - gold[s]:.4f}", (public[s], y), xytext=(4, 0),
                   textcoords="offset points", va="center", fontsize=7.5)
right.set(yticks=ys, yticklabels=[f"Seed {s}" for s in SEEDS], ylim=(-0.7, 2.7),
          xlim=(0.66, 0.78), xticks=[0.66, 0.70, 0.74, 0.78], xlabel="CXI development macro-F1")
right.set_title("(b) Matched gold-training seeds", loc="left", pad=30)
right.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
right.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
right.tick_params(axis="y", length=0)
handles = [
    Line2D([], [], marker="o", linestyle="None", markersize=5, color=gray, label="Gold only"),
    Line2D([], [], marker="o", linestyle="None", markersize=5, color=blue, label="Public 4k + gold"),
]
right.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.24),
             ncol=2, frameon=False, fontsize=7.5, handletextpad=0.4, columnspacing=1.2)
for fmt in ("pdf", "svg", "png"):
    kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if fmt == "pdf" else ({"metadata": {"Date": None}} if fmt == "svg" else {})
    fig.savefig(HERE / f"results.{fmt}", dpi=220, **kwargs)
svg = HERE / "results.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
plt.close(fig)
