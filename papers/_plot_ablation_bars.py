"""Bar plot of the four defense-ablation tables (one panel per victim).
Style follows the reference (huggingface ... bar.png): coloured grouped bars
with thin black edge, value labels on top, legend at top, dashed grid."""
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

# Data: ASR-LLM (%) per (victim, defense, attack).
# Defense order: Base, +System Prompt, +PO, +A2D.
# Attack order: PAD, DIJA, Search (ours).
DATA = {
    "dream":    [[66.0, 43.0, 50.0, 30.0],
                 [79.0, 56.0, 31.0, 26.0],
                 [94.0, 90.0, 87.0, 85.0]],
    "llada":    [[62.0, 43.0, 66.0, 48.0],
                 [74.0, 43.0, 71.0, 47.0],
                 [73.0, 73.0, 79.0, 61.0]],
    "llada1_5": [[55.0, 39.0, 64.0, 48.0],
                 [67.0, 38.0, 68.0, 42.0],
                 [74.0, 68.0, 82.0, 60.0]],
    "mmada":    [[77.0, 68.0, 72.0, 45.0],
                 [67.0, 65.0, 63.0, 67.0],
                 [80.0, 73.0, 79.0, 80.0]],
}
TITLES = {"dream": "Dream-Instruct", "llada": "LLaDA-Instruct",
          "llada1_5": "LLaDA-1.5", "mmada": "MMaDA-CoT"}
DEFENSES = ["Base", "Self-Reminder", "PO", "A2D"]
ATTACKS = ["PAD", "DIJA", "MaskForge (ours)"]
# Reference-style palette: 4 defenses → 4 colours.
COLORS = ["#B6B6B6", "#C97064", "#9CC78A", "#6FA8DC"]

fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.0), sharey=True)
victims = ["dream", "llada", "llada1_5", "mmada"]
x = np.arange(len(ATTACKS))
width = 0.20

for ax, v in zip(axes, victims):
    # Transpose: rows[attack][defense]  →  bars[defense] per attack group.
    rows = DATA[v]
    for j, (defense, color) in enumerate(zip(DEFENSES, COLORS)):
        vals = [rows[i][j] for i in range(len(ATTACKS))]
        offset = (j - (len(DEFENSES) - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=color,
               edgecolor="black", linewidth=0.7,
               label=defense if v == victims[0] else None)
    ax.set_xticks(x)
    ax.set_xticklabels(ATTACKS, fontsize=10)
    ax.set_title(TITLES[v])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(axis="y", linestyle="--", alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("ASR (%)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4,
           bbox_to_anchor=(0.5, 1.04), handlelength=1.6,
           columnspacing=2.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_dir = "papers"
os.makedirs(out_dir, exist_ok=True)
pdf = os.path.join(out_dir, "ablation_defenses_bars.pdf")
png = os.path.join(out_dir, "ablation_defenses_bars.png")
plt.savefig(pdf, bbox_inches="tight")
plt.savefig(png, bbox_inches="tight", dpi=300)
print(f"saved: {pdf}\nsaved: {png}")
