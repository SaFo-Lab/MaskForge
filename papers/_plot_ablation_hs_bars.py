"""Bar plot of Harmscore from the four defense-ablation tables (one panel per
victim). Same style as ablation_defenses_bars.py."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": True,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Harmscore (1-5) per (victim, attack, defense).
# Defense order: Base, Self-Reminder, PO, A2D.
# Attack order: PAD, DIJA, SA^3.
HS = {
    "dream":    [[4.38, 3.39, 4.12, 4.54],
                 [4.49, 4.15, 3.68, 2.84],
                 [4.84, 4.79, 4.74, 4.46]],
    "llada":    [[3.82, 3.32, 4.40, 3.71],
                 [4.43, 3.84, 4.33, 3.58],
                 [4.52, 4.55, 4.70, 4.22]],
    "llada1_5": [[3.86, 3.15, 4.32, 3.78],
                 [4.34, 3.77, 4.30, 3.60],
                 [4.48, 4.57, 4.70, 4.00]],
    "mmada":    [[4.42, 4.26, 4.41, 3.96],
                 [4.46, 4.28, 4.12, 4.31],
                 [4.60, 4.44, 4.47, 4.59]],
}
TITLES = {"dream": "Dream-Instruct", "llada": "LLaDA-Instruct",
          "llada1_5": "LLaDA-1.5", "mmada": "MMaDA-CoT"}
DEFENSES = ["Base", "Self-Reminder", "PO", "A2D"]
ATTACKS = ["PAD", "DIJA", "MaskForge (ours)"]
COLORS = ["#B6B6B6", "#C97064", "#9CC78A", "#6FA8DC"]

fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), sharey=True)
victims = ["dream", "llada", "llada1_5", "mmada"]
x = np.arange(len(ATTACKS))
width = 0.20

for ax, v in zip(axes, victims):
    rows = HS[v]
    for j, (defense, color) in enumerate(zip(DEFENSES, COLORS)):
        vals = [rows[i][j] for i in range(len(ATTACKS))]
        offset = (j - (len(DEFENSES) - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.7,
               label=defense if v == victims[0] else None)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=10)
    ax.set_title(TITLES[v])
    ax.set_ylim(2.5, 5.0)
    ax.set_yticks([2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Harmscore (1–5)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.04), handlelength=1.6,
           columnspacing=2.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_dir = "papers"
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "ablation_defenses_hs_bars.pdf")
png = os.path.join(out_dir, "ablation_defenses_hs_bars.png")
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=300)
print(f"saved: {pdf}\nsaved: {png}")
