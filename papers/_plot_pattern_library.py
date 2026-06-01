"""Pattern-library statistics figure: goal-type pie + slot-count histogram.
Loads logs/shared_pattern_registry.json + logs/shared_pattern_set.json."""
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

reg = json.load(open("logs/shared_pattern_registry.json"))
ps  = json.load(open("logs/shared_pattern_set.json"))

# --- panel 1: goal-type bar (count per bucket) ---
gt_counts = sorted(((gt, len(ids)) for gt, ids in ps.items()), key=lambda x: -x[1])
gt_names  = [g for g, _ in gt_counts]
gt_vals   = [v for _, v in gt_counts]

# --- panel 2: structure-type bar (top 8) ---
stypes = Counter()
for p in reg.values():
    stypes[p.get("schema", {}).get("structure_type", "?")] += 1
top_stypes = stypes.most_common(8)
st_names = [s.replace("_", "\n") for s, _ in top_stypes]
st_vals  = [v for _, v in top_stypes]

# --- panel 3: slot-count histogram ---
sc = Counter()
for p in reg.values():
    sc[p.get("schema", {}).get("slot_count", 0)] += 1
keys = sorted(k for k in sc if 1 <= k <= 12)
sc_x = keys
sc_y = [sc[k] for k in keys]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 7.5),
                         gridspec_kw={"width_ratios": [1.6, 1.0, 0.9]})

palette_a = "#7FB1C2"
palette_b = "#9F86C0"
palette_c = "#3F7E68"

axes[0].barh(gt_names[::-1], gt_vals[::-1], color=palette_a, edgecolor="#34495E", linewidth=0.6)
axes[0].set_xlabel("# patterns")
axes[0].set_title("(a) Goal-type buckets (30 categories)")
axes[0].tick_params(axis="y", labelsize=8.5)
for i, v in enumerate(gt_vals[::-1]):
    axes[0].text(v + 2, i, str(v), va="center", fontsize=7.5)

st_flat = [s.replace("\n", "_") for s in st_names]
# largest at top: matplotlib barh draws first item at bottom, so reverse both labels and vals
y_pos = list(range(len(st_vals)))
axes[1].barh(y_pos, st_vals[::-1], color=palette_b, edgecolor="#34495E", linewidth=0.6)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(st_flat[::-1], fontsize=8.5)
axes[1].set_xlabel("# patterns")
axes[1].set_title("(b) Top-8 structure types")
for i, v in enumerate(st_vals[::-1]):
    axes[1].text(v + 3, i, str(v), va="center", fontsize=8.5)

axes[2].bar(sc_x, sc_y, color=palette_c, edgecolor="#34495E", linewidth=0.6)
axes[2].set_xticks(sc_x)
axes[2].set_xlabel("slot count")
axes[2].set_ylabel("# patterns")
axes[2].set_title("(c) Slot-count distribution")
for x, y in zip(sc_x, sc_y):
    axes[2].text(x, y + 5, str(y), ha="center", fontsize=8.5)

axes[0].grid(True, axis="x", linestyle="-", alpha=0.25, linewidth=0.5)
axes[1].grid(True, axis="x", linestyle="-", alpha=0.25, linewidth=0.5)
axes[2].grid(True, axis="y", linestyle="-", alpha=0.25, linewidth=0.5)
for ax in axes:
    ax.set_axisbelow(True)

plt.tight_layout()
os.makedirs("papers", exist_ok=True)
fig.savefig("papers/pattern_library_stats.pdf", bbox_inches="tight")
fig.savefig("papers/pattern_library_stats.png", bbox_inches="tight", dpi=300)
print("saved -> papers/pattern_library_stats.{pdf,png}")
print(f"  registry size: {len(reg)} patterns")
print(f"  buckets: {len(ps)} ({sum(gt_vals)} pids assigned, {sum(gt_vals)/len(reg)*100:.1f}% coverage)")
print(f"  distinct structure types: {len(stypes)}")
