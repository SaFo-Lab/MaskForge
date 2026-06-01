"""Plot mask-ratio ablation curve: ASR vs mask ratio (single panel).
Style follows the reference figure (rStar-Math style): pastel palette,
single top legend, light grid, small round markers."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.linewidth": 1.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 10.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SUMMARY = "results/mask_ratio/summary.json"
OUT_DIR = "papers"

RATIOS = [0.10, 0.30, 0.50, 0.70, 0.90]
TAGS = ["mr10", "mr30", "mr50", "mr70", "mr90"]
VICTIMS = ["dream", "llada", "llada1_5", "mmada"]
LABELS = {"dream": "Dream", "llada": "LLaDA",
          "llada1_5": "LLaDA-1.5", "mmada": "MMaDA"}

COLORS = {
    "dream":    "#7FB1C2",
    "llada":    "#E0BD55",
    "llada1_5": "#9F86C0",
    "mmada":    "#E07B68",
    "AVG":      "#3F7E68",
}
MARKERS = {"dream": "o", "llada": "s", "llada1_5": "D", "mmada": "^", "AVG": "o"}

d = json.load(open(SUMMARY))


def cell(v, tag, key):
    return d.get(f"jbb_{v}_{tag}", {}).get(key)


fig, ax = plt.subplots(figsize=(6.0, 4.2))

# Individual victims (pastel)
for v in VICTIMS:
    ys = [cell(v, t, "asr_llm") * 100 for t in TAGS]
    ax.plot(RATIOS, ys, marker=MARKERS[v], color=COLORS[v],
            linewidth=2.2, markersize=6,
            markerfacecolor="white",
            markeredgecolor=COLORS[v],
            markeredgewidth=2.0,
            label=LABELS[v])

# AVG (headline)
avg = [sum(cell(v, t, "asr_llm") for v in VICTIMS) / len(VICTIMS) * 100 for t in TAGS]
ax.plot(RATIOS, avg, marker=MARKERS["AVG"], color=COLORS["AVG"],
        linewidth=3.0, markersize=7.5,
        markerfacecolor=COLORS["AVG"],
        markeredgecolor=COLORS["AVG"],
        label="AVG")

ax.set_xlabel(r"mask ratio $r$")
ax.set_ylabel("ASR (%)")
ax.set_xticks(RATIOS)
ax.set_xlim(0.02, 0.98)
ax.set_ylim(50, 88)
ax.set_yticks([55, 65, 75, 85])
ax.grid(True, which="major", linestyle="-", alpha=0.35,
        linewidth=0.55, color="#cccccc")
ax.set_axisbelow(True)

# top legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5,
           bbox_to_anchor=(0.5, 1.02), handlelength=1.8,
           columnspacing=2.0, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(OUT_DIR, exist_ok=True)
pdf = os.path.join(OUT_DIR, "mask_ratio_ablation.pdf")
png = os.path.join(OUT_DIR, "mask_ratio_ablation.png")
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=300)
print(f"saved: {pdf}\nsaved: {png}")
