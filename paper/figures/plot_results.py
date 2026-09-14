"""Regenerate the manuscript's vector figure from its published result values."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results.json").read_text())
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.25))
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.81, wspace=1.05)
blue, orange = "#0072B2", "#D55E00"
p = DATA["public_adaptation"]
left.scatter(p["public_updates"], p["scores"], s=30, color=blue, zorder=3)
left.scatter([4000], [p["scores"][1]], s=49, marker="D", color=orange, zorder=4)
for x, y in zip(p["public_updates"], p["scores"]):
    left.annotate(f"{y:.5f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center")
left.set(xticks=p["public_updates"], xticklabels=["2,000", "4,000", "8,000"],
         xlim=(1000, 9100), ylim=(0.700, 0.763), yticks=[0.70, 0.72, 0.74, 0.76],
         xlabel="Public adaptation updates", ylabel="CXI macro-F1")
left.set_title("(a) Matched gold adaptation", loc="left", pad=12)
left.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
left.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
p = DATA["generative_alternatives"]
ys = list(range(len(p["scores"])))
right.scatter(p["scores"], ys, s=30, color=blue, zorder=3)
right.scatter([p["scores"][0]], [0], s=49, marker="D", color=orange, zorder=4)
right.axvline(p["scores"][0], color=orange, linestyle="--", linewidth=0.9, zorder=1)
for x, y in zip(p["scores"], ys):
    right.annotate(f"{x:.5f}", (x, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3})
right.set(yticks=ys, yticklabels=p["labels"], ylim=(4.65, -0.65),
          xlim=(0.741, 0.764), xticks=[0.745, 0.755], xlabel="Whole-task macro-F1")
right.set_title("(b) Generative replacements", loc="left", pad=12)
right.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
right.grid(axis="x", color="0.9", linewidth=0.6, zorder=0)
right.tick_params(axis="y", length=0)
for fmt in ("pdf", "svg", "png"):
    kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if fmt == "pdf" else {}
    fig.savefig(HERE / f"results.{fmt}", dpi=220, **kwargs)
plt.close(fig)
