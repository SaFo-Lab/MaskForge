"""Method-pipeline figure for one successful trajectory (jbb_12 ransomware).

Improvements over v1:
- larger slot-role chips with readable labels
- step-number badges (①…⑦) on each box
- bridge note at end of iter 1 to remove its trailing whitespace
- compact final banner anchored just below iter 2

Renders PDF + PNG into papers/method_walkthrough/.
"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
TRAJ = json.load(open(ROOT / "papers/method_walkthrough/jbb_12_trajectory.json"))
REGISTRY = json.load(open(ROOT / "logs/shared_pattern_registry.json"))
OUT = ROOT / "papers/method_walkthrough"

# Color palette
C_GOAL = "#2C3E50"
C_PROMPT = "#A9CCE3"
C_VICTIM = "#A9DFBF"
C_BAD = "#F5B7B1"
C_OK = "#7DCEA0"
C_FB = "#FAD7A0"
C_EVO = "#F5CBA7"
C_PATTERN_BG = "#FDF2E9"
C_SLOT = "#F8C471"
C_SLOT_PICKED = "#E67E22"
C_PICK_HIGHLIGHT = "#F9E79F"
EDGE = "#34495E"


def trunc(s, n):
    if s is None: return ""
    s = str(s)
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def short_role(role):
    if not role:
        return "?"
    role = str(role)
    parts = role.replace("<mask:", "").replace(">", "").split("_")
    parts = [p for p in parts if p and p not in ("and", "or", "the", "of", "a")]
    parts = parts[:2]
    out = "_".join(parts)
    return out[:20]


def slot_roles_for(pid):
    p = REGISTRY.get(pid, {})
    schema = p.get("schema", {}) if isinstance(p, dict) else {}
    roles = schema.get("slot_roles") or []
    if not roles:
        return [f"slot_{i+1}" for i in range(int(schema.get("slot_count") or 4))]
    return roles


def step_badge(ax, x, y, n, color="#34495E"):
    """Small numbered badge in box corner."""
    ax.add_patch(Circle((x, y), 0.13, facecolor=color, edgecolor="white", linewidth=1.5, zorder=10))
    ax.text(x, y, str(n), color="white", fontsize=8.5, fontweight="bold",
            ha="center", va="center", zorder=11)


def box(ax, x, y, w, h, text, fc, fontsize=7, title=None,
        title_color="white", title_bg="#34495E", step=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                facecolor=fc, edgecolor=EDGE, linewidth=0.7))
    if title:
        ax.add_patch(FancyBboxPatch((x, y + h - 0.30), w, 0.30,
                                    boxstyle="round,pad=0.03",
                                    facecolor=title_bg, edgecolor=EDGE, linewidth=0.6))
        ax.text(x + 0.30, y + h - 0.15, title, fontsize=fontsize, color=title_color,
                fontweight="bold", va="center")
        ax.text(x + 0.10, y + h - 0.45, text, fontsize=fontsize - 0.3,
                color="black", va="top", wrap=True)
    if step is not None:
        step_badge(ax, x + 0.13, y + h - 0.13, step, color=title_bg)


def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=color, linewidth=lw))


def draw_pattern_card(ax, x, y, w, h, cand, picked):
    fc = C_PICK_HIGHLIGHT if picked else C_PATTERN_BG
    lw = 1.8 if picked else 0.7
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fc, edgecolor=("#943126" if picked else EDGE),
                                linewidth=lw))
    pid = cand["pattern_id"][:8]
    structure = (cand.get("structure_type") or "")[:30]
    ucb = cand["ucb_score"]
    mu = cand["mean_reward"]
    n = cand["visit_count"]
    bonus = cand["exploration_bonus"]

    ax.text(x + 0.08, y + h - 0.13, pid, fontsize=8.2, family="monospace",
            fontweight="bold", va="center",
            color="#943126" if picked else "#1B2631")
    ax.text(x + 0.08, y + h - 0.32, structure, fontsize=7.0,
            va="center", color="#5D6D7E", style="italic")
    ax.text(x + w - 0.08, y + h - 0.13,
            f"UCB = {ucb:.2f}",
            fontsize=8.2, ha="right", va="center",
            color="#943126" if picked else "#1B2631",
            fontweight="bold", family="monospace")
    ax.text(x + w - 0.08, y + h - 0.32,
            f"μ={mu:.2f}  +  bonus={bonus:.2f}     n={n}",
            fontsize=6.8, ha="right", va="center",
            color="#5D6D7E", family="monospace")

    # slot chips
    roles = slot_roles_for(cand["pattern_id"])
    chips = roles[:6]
    n_chips = len(chips)
    if n_chips == 0:
        return
    chip_y = y + 0.10
    chip_h = 0.26
    avail = w - 0.16
    chip_w = (avail - 0.04 * (n_chips - 1)) / n_chips
    for i, r in enumerate(chips):
        cx = x + 0.08 + i * (chip_w + 0.04)
        ax.add_patch(FancyBboxPatch((cx, chip_y), chip_w, chip_h,
                                    boxstyle="round,pad=0.02",
                                    facecolor=C_SLOT_PICKED if picked else C_SLOT,
                                    edgecolor="#7E5109", linewidth=0.6))
        ax.text(cx + chip_w / 2, chip_y + chip_h / 2,
                short_role(r), fontsize=7.0, ha="center", va="center",
                color="white" if picked else "#1B2631",
                fontweight="bold" if picked else "normal")
    if len(roles) > n_chips:
        ax.text(x + w - 0.04, chip_y + chip_h / 2,
                f"+{len(roles) - n_chips}", fontsize=6.5, ha="right",
                va="center", color="#7E5109", style="italic")


def draw_iter(ax, x0, y_top, col_w, iter_data, header_color):
    y = y_top
    # header
    ax.add_patch(FancyBboxPatch((x0, y - 0.32), col_w, 0.32,
                                boxstyle="round,pad=0.02",
                                facecolor=header_color, edgecolor=EDGE, linewidth=0.6))
    ax.text(x0 + col_w / 2, y - 0.16,
            f"Iteration {iter_data['iteration']}",
            fontsize=11, color="white", fontweight="bold",
            ha="center", va="center")
    y -= 0.52

    # Step 1+2 label
    ax.text(x0, y + 0.04, "Step 1+2 — UCB selects from goal-type candidates",
            fontsize=8.2, color="#1B2631", fontweight="bold", va="bottom")
    step_badge(ax, x0 - 0.20, y + 0.08, "1", color="#34495E")

    cands = iter_data.get("ucb_candidates", [])
    sel_pid = iter_data["selected_pattern"]["pattern_id"] if iter_data["selected_pattern"] else None

    card_h = 0.66
    for c in cands:
        draw_pattern_card(ax, x0, y - card_h, col_w, card_h, c,
                          picked=(c["pattern_id"] == sel_pid))
        y -= card_h + 0.08
    y -= 0.10

    arrow(ax, x0 + col_w / 2, y + 0.18, x0 + col_w / 2, y - 0.05)

    # Step 3
    sa = iter_data["step_3_attacker_instantiate"]
    inst_h = 1.45
    box(ax, x0, y - inst_h, col_w, inst_h,
        "Jailbreak prompt (user query):\n" + trunc(sa["jailbreak_prompt"], 220) + "\n\n"
        "Template (mask scaffold prefilled into assistant turn):\n" + trunc(sa["template"], 220),
        C_PROMPT, fontsize=7.0, step=3,
        title="Step 3 — Attacker.instantiate (LLM picks strategy → prompt + template)")
    y -= inst_h + 0.14
    arrow(ax, x0 + col_w / 2, y + 0.14, x0 + col_w / 2, y - 0.05)

    # Step 4
    vic_h = 0.95
    box(ax, x0, y - vic_h, col_w, vic_h,
        trunc(iter_data["step_4_victim_output"], 280),
        C_VICTIM, fontsize=7.0, step=4,
        title="Step 4 — Victim mask-fill (parallel diffusion sampling)")
    y -= vic_h + 0.14
    arrow(ax, x0 + col_w / 2, y + 0.14, x0 + col_w / 2, y - 0.05)

    # Step 5
    rew = iter_data["step_5_scorer_reward"]
    rew_color = C_OK if rew >= 0.7 else C_BAD
    rew_icon = "✓ jailbroken" if rew >= 0.7 else "✗ failed → fallback / next iter"
    box(ax, x0, y - 0.50, col_w, 0.50,
        f"reward = {rew:.2f}    {rew_icon}",
        rew_color, fontsize=8.5, step=5,
        title="Step 5 — Scorer (Qwen3-4B, HarmBench classifier)",
        title_bg="#1F618D")
    y -= 0.50 + 0.14

    # Step 6 + 7 (only if used)
    if iter_data["step_6_fallback_used"]:
        arrow(ax, x0 + col_w / 2, y + 0.14, x0 + col_w / 2, y - 0.05)
        fb = iter_data["step_6_fallback_artifacts"]
        fb_h = 1.30
        text = ("(a) weaker base (no safety filter):\n"
                + trunc(fb["weaker_model_response"], 170) + "\n\n"
                "(b) Bedrock tagger inserts [UNSAFE]…[/UNSAFE]\n"
                "→ tagged template:\n"
                + trunc(fb["tagged_template_after_bedrock"], 150))
        box(ax, x0, y - fb_h, col_w, fb_h, text, C_FB, fontsize=6.8, step=6,
            title="Step 6 — Fallback (reward < 0.7)")
        y -= fb_h + 0.14

    if iter_data["step_7_pattern_evolution"]:
        arrow(ax, x0 + col_w / 2, y + 0.14, x0 + col_w / 2, y - 0.05)
        ev = iter_data["step_7_pattern_evolution"]
        new_pid = ev["new_pattern"]["pattern_id"][:8] if ev["new_pattern"] else "?"
        new_roles = slot_roles_for(ev["new_pattern"]["pattern_id"]) if ev["new_pattern"] else []
        ev_h = 2.0
        text = (
            f"new pattern id = {new_pid}\n"
            "slots = "
            + ", ".join(short_role(r) for r in new_roles[:6])
            + ("..." if len(new_roles) > 6 else "")
            + "\n\nnew template (re-attack):\n"
            + trunc(ev["new_template"], 150)
            + "\n\nvictim re-attack output:\n"
            + trunc(ev["new_victim_output"], 200)
            + f"\n\nfinal reward = {ev['new_reward']:.2f}  ✓ unsafe — pattern persisted"
        )
        box(ax, x0, y - ev_h, col_w, ev_h, text, C_EVO, fontsize=6.8, step=7,
            title="Step 7 — Summarizer creates new pattern + re-attack",
            title_bg="#7D6608")
        y -= ev_h + 0.14
    else:
        # placeholder bridge note: iter1 ends here, gap will be filled by inter-iter arrow
        pass

    return y


def main():
    fig = plt.figure(figsize=(15, 17), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 17)
    ax.set_axis_off()

    # Title and goal
    ax.text(7.5, 16.5, "MaskForge attack pipeline — case study (Dual100)",
            fontsize=16, fontweight="bold", ha="center", color=C_GOAL)
    ax.text(7.5, 16.10,
            f"Goal ({TRAJ['goal_id']}, type={TRAJ['goal_type']}): {TRAJ['goal']}",
            fontsize=10.5, ha="center", color="#566573", style="italic")

    col_w = 6.8
    gap = 0.5
    x0_left = 0.5
    x0_right = x0_left + col_w + gap
    y_top = 15.4

    yL = draw_iter(ax, x0_left, y_top, col_w, TRAJ["trajectory"][0],
                   header_color="#34495E")
    yR = draw_iter(ax, x0_right, y_top, col_w, TRAJ["trajectory"][1],
                   header_color="#943126")

    # Iter 1 → iter 2 flow note (closes the iter1 column)
    bridge_y = yL - 0.10
    ax.add_patch(FancyArrowPatch(
        (x0_left + col_w / 2, bridge_y + 0.10),
        (x0_left + col_w / 2, bridge_y - 0.40),
        arrowstyle="-|>", mutation_scale=12, color="#943126", linewidth=1.4))
    ax.text(x0_left + col_w / 2, bridge_y - 0.55,
            "iter 1 fails  →  UCB stats updated\n→  enter iteration 2 (right)",
            fontsize=9, ha="center", va="center",
            color="#943126", fontweight="bold")
    ax.add_patch(FancyArrowPatch(
        (x0_left + col_w + 0.05, bridge_y - 0.55),
        (x0_right - 0.05, y_top - 0.16),
        connectionstyle="arc3,rad=-0.25", arrowstyle="-|>",
        mutation_scale=14, color="#943126", linewidth=1.6))

    # Final banner under iter 2
    final_y = yR - 0.30
    ax.add_patch(FancyBboxPatch((x0_left + 1.0, final_y - 0.55),
                                col_w * 2 + gap - 2.0, 0.55,
                                boxstyle="round,pad=0.05",
                                facecolor=C_OK, edgecolor=EDGE, linewidth=1.0))
    ax.text(x0_left + col_w + gap / 2, final_y - 0.27,
            f"Final: reward = {TRAJ['final_reward']:.2f}   "
            f"✓ Dual100 jailbreak succeeds (iter {TRAJ['best_iteration']['iteration']})",
            fontsize=12, ha="center", fontweight="bold", color="#0E6655")

    # Legend (compact, single row)
    legend = [
        ("Pattern card", C_PATTERN_BG),
        ("Slot role", C_SLOT),
        ("Picked pattern", C_PICK_HIGHLIGHT),
        ("Attacker prompt+tmpl", C_PROMPT),
        ("Victim mask-fill", C_VICTIM),
        ("Reward ≥ 0.7", C_OK),
        ("Reward < 0.7", C_BAD),
        ("Fallback", C_FB),
        ("Pattern evolution", C_EVO),
    ]
    lx = 0.5
    ly = 0.20
    for label, color in legend:
        ax.add_patch(FancyBboxPatch((lx, ly), 0.22, 0.20,
                                    boxstyle="round,pad=0.0",
                                    facecolor=color, edgecolor=EDGE, linewidth=0.5))
        ax.text(lx + 0.27, ly + 0.10, label, fontsize=7.5, va="center")
        lx += len(label) * 0.085 + 0.7

    pdf = OUT / "method_pipeline.pdf"
    png = OUT / "method_pipeline.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"saved -> {pdf}")
    print(f"saved -> {png}")


if __name__ == "__main__":
    main()
