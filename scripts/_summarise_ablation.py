"""Read ablation Bedrock summaries and emit (1) a CSV with all numbers,
(2) per-ablation LaTeX tables into tables/, and (3) a prose 'Additional
Results' subsection (tables/additional_results.tex) with one paragraph
per ablation question (schema / fallback / no-evolution)."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/ablation"
OUT_TAB = ROOT / "tables"
OUT_TAB.mkdir(exist_ok=True)

VICTIMS = ["dream", "llada", "llada1_5"]
VICTIM_LABEL = {
    "dream": "Dream-Instruct",
    "llada": "LLaDA",
    "llada1_5": "LLaDA-1.5",
}


def load_runvictim_summary():
    p = RES / "bedrock_summary_runvictim.json"
    if not p.exists(): return {}
    return json.load(open(p))


def load_transition_summary():
    asr = RES / "bedrock_summary_no_schema_asr.json"
    hs = RES / "bedrock_summary_no_schema_hs.json"
    asr_data = json.load(open(asr)) if asr.exists() else {}
    hs_data = json.load(open(hs)) if hs.exists() else {}
    return asr_data, hs_data


def cell(rv_summary, victim, ablation):
    key = f"jailbreak_{victim}_ablate_{ablation}"
    s = rv_summary.get(key, {})
    asr = s.get("asr_llm")
    hs = s.get("hs_avg")
    return (None if asr is None else asr * 100,
            None if hs is None else hs)


def cell_no_schema(asr_data, hs_data, victim):
    key = f"jbb_{victim}_no_schema"
    sa = asr_data.get(key, {})
    sh = hs_data.get(key, {})
    asr = sa.get("k3_asr")
    hs = sh.get("k3_hs")
    return (None if asr is None else asr * 100,
            None if hs is None else hs)


def fmt(v, prec=1):
    if v is None: return "--"
    if prec == 0: return f"{v:.0f}"
    return f"{v:.{prec}f}"


def write_csv(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def avg(vals):
    vs = [v for v in vals if v is not None]
    return sum(vs)/len(vs) if vs else None


def emit_table(caption, label, header_rows, body_rows):
    """body_rows: list of (label, [(asr,hs)…per-victim, plus average computed automatically])."""
    cols = "l|" + "cc" * len(VICTIMS) + "|cc"
    out = [
        "\\begin{table}[htb]\n",
        "\\centering\n",
        f"\\caption{{{caption}}}\n",
        f"\\label{{{label}}}\n",
        "\\resizebox{\\linewidth}{!}{\n",
        f"\\begin{{tabular}}{{{cols}}}\n",
        "\\toprule\n",
    ]
    top = "Variant"
    sub = ""
    for v in VICTIMS:
        top += f" & \\multicolumn{{2}}{{c}}{{{VICTIM_LABEL[v]}}}"
        sub += " & ASR & HS"
    top += " & \\multicolumn{2}{c}{AVG.}"
    sub += " & ASR & HS"
    out.append(top + " \\\\\n")
    out.append(sub + " \\\\\n")
    out.append("\\midrule\n")
    for label_, cells in body_rows:
        row = label_
        a_list = []; h_list = []
        for (a, h) in cells:
            row += f" & {fmt(a)} & {fmt(h, 2)}"
            a_list.append(a); h_list.append(h)
        a_avg = avg(a_list); h_avg = avg(h_list)
        row += f" & {fmt(a_avg)} & {fmt(h_avg, 2)} \\\\\n"
        out.append(row)
    out.append("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    return "".join(out)


def main():
    rv = load_runvictim_summary()
    asr_data, hs_data = load_transition_summary()

    # ---- consolidated CSV ----
    rows = [["victim", "variant", "asr_pct", "hs"]]
    for victim in VICTIMS:
        a, h = cell(rv, victim, "full")
        rows.append([victim, "full", fmt(a), fmt(h, 2)])
        a, h = cell_no_schema(asr_data, hs_data, victim)
        rows.append([victim, "no_schema", fmt(a), fmt(h, 2)])
        for ab in ["no_ucb", "no_fallback", "stage_a_only"]:
            a, h = cell(rv, victim, ab)
            rows.append([victim, ab, fmt(a), fmt(h, 2)])
    write_csv(rows, RES / "ablation_summary.csv")
    print(f"wrote {RES/'ablation_summary.csv'}")

    # ---- precompute cells ----
    full_cells     = [cell(rv, v, "full")          for v in VICTIMS]
    nos_cells      = [cell_no_schema(asr_data, hs_data, v) for v in VICTIMS]
    nofb_cells     = [cell(rv, v, "no_fallback")   for v in VICTIMS]
    stagea_cells   = [cell(rv, v, "stage_a_only")  for v in VICTIMS]

    # ---- Table 1: schema vs raw retrieval ----
    t_schema = emit_table(
        "\\textbf{Ablation for structural pattern abstraction.} "
        "JailbreakBench (100 goals/victim), Strip-ASR-LLM (\\%) and Harmscore "
        "(1--5), Bedrock Qwen3-235B judge. \\textsc{No-schema} replaces "
        "\\textsc{attacker.instantiate} with pure best-of-$k{=}3$ retrieval "
        "over the goal-type bucket: the stored representative template is sent "
        "to the victim verbatim, with no schema-conditioned regeneration.",
        "tab:ablation_schema",
        None,
        [("\\textbf{Full}", full_cells),
         ("No-schema (raw retrieval)", nos_cells)])
    (OUT_TAB / "ablation_schema.tex").write_text(t_schema)
    print(f"wrote {OUT_TAB/'ablation_schema.tex'}")

    # ---- Table 2: fallback ----
    t_fb = emit_table(
        "\\textbf{Ablation for scorer-guided fallback.} \\textsc{No-fallback} "
        "keeps UCB, schema instantiation, and online expansion intact, but "
        "disables the weaker-base-model retry path that fires when the "
        "scorer reward is below $\\rho_{\\text{fb}}{=}0.7$. JailbreakBench "
        "(100 goals/victim), Bedrock judge.",
        "tab:ablation_fallback",
        None,
        [("\\textbf{Full}", full_cells),
         ("No-fallback", nofb_cells)])
    (OUT_TAB / "ablation_fallback.tex").write_text(t_fb)
    print(f"wrote {OUT_TAB/'ablation_fallback.tex'}")

    # ---- Table 3: no-evolution (Stage A only) ----
    t_ev = emit_table(
        "\\textbf{Ablation for online pattern evolution.} \\textsc{No-evolution} "
        "freezes the pattern library at the end of Stage A: UCB still selects "
        "from the bootstrap registry and \\textsc{attacker.instantiate} still "
        "rewrites the chosen template, but successful jailbreaks no longer "
        "feed back into the registry as new patterns. JailbreakBench (100 "
        "goals/victim), Bedrock judge.",
        "tab:ablation_no_evolution",
        None,
        [("\\textbf{Full}", full_cells),
         ("No-evolution (Stage-A only)", stagea_cells)])
    (OUT_TAB / "ablation_no_evolution.tex").write_text(t_ev)
    print(f"wrote {OUT_TAB/'ablation_no_evolution.tex'}")

    # ---- prose Additional Results section ----
    full_avg_a = avg([c[0] for c in full_cells])
    full_avg_h = avg([c[1] for c in full_cells])
    nos_avg_a  = avg([c[0] for c in nos_cells])
    nos_avg_h  = avg([c[1] for c in nos_cells])
    nofb_avg_a = avg([c[0] for c in nofb_cells])
    nofb_avg_h = avg([c[1] for c in nofb_cells])
    sa_avg_a   = avg([c[0] for c in stagea_cells])
    sa_avg_h   = avg([c[1] for c in stagea_cells])

    # per-victim deltas (Full - variant)
    d_nos_a  = [(full_cells[i][0] - nos_cells[i][0])    for i in range(len(VICTIMS))]
    d_nofb_a = [(full_cells[i][0] - nofb_cells[i][0])   for i in range(len(VICTIMS))]
    d_sa_a   = [(full_cells[i][0] - stagea_cells[i][0]) for i in range(len(VICTIMS))]

    schema_text = (
        "\\noindent\\textbf{Ablation for Schema (pattern).} "
        "We replace the schema-conditioned \\textsc{attacker.instantiate} step "
        "with pure best-of-$k{=}3$ retrieval: for each goal we select the $k$ "
        "closest representative templates in the goal-type bucket and submit "
        "each verbatim, with no schema-conditioned regeneration. Averaged "
        "across the three diffusion victims this drops ASR from "
        f"{fmt(full_avg_a)}\\% (Full) to {fmt(nos_avg_a)}\\% (No-schema) and "
        f"Harmscore from {fmt(full_avg_h, 2)} to {fmt(nos_avg_h, 2)} "
        f"(Tab.~\\ref{{tab:ablation_schema}}); the per-victim drops are "
        f"{fmt(d_nos_a[0])}, {fmt(d_nos_a[1])}, and {fmt(d_nos_a[2])} ASR "
        "points on Dream-Instruct, LLaDA, and LLaDA-1.5 respectively, with "
        "the gap largest on the most robust victim (LLaDA-1.5: "
        f"{fmt(full_cells[2][0])}\\% vs.\\ {fmt(nos_cells[2][0])}\\%). This "
        "confirms that most of the attack signal lives in the abstract "
        "schema (action verb $+$ persona/framing slots $+$ obfuscation) "
        "rather than in the surface form of any single mined template --- "
        "raw retrieval is a much weaker baseline than schema-conditioned "
        "instantiation."
    )

    fb_text = (
        "\\noindent\\textbf{Ablation for Fallback.} "
        "The scorer-guided fallback fires when the live scorer reward is "
        "below $\\rho_{\\text{fb}}{=}0.7$ on the target victim, retrying "
        "the same schema instantiation against a weaker base model and "
        "transferring the affirmative response back. In our setup all "
        "ablation variants share the same online-grown 1779-pattern "
        "registry, so by the time fallback would fire the bandit has "
        "almost always already located a high-reward pattern; the average "
        f"ASR change is therefore small ({fmt(full_avg_a)}\\% Full vs.\\ "
        f"{fmt(nofb_avg_a)}\\% No-fallback, "
        f"Tab.~\\ref{{tab:ablation_fallback}}), with per-victim deltas of "
        f"{fmt(d_nofb_a[0])}, {fmt(d_nofb_a[1])}, and {fmt(d_nofb_a[2])} ASR "
        "points. The largest residual effect is on Dream-Instruct "
        f"({fmt(full_cells[0][0])}\\% $\\to$ {fmt(nofb_cells[0][0])}\\%), "
        "the victim with the strongest base alignment and therefore the "
        "most goals that genuinely need the rescue path. The fallback "
        "thus contributes most when the registry is small or the victim "
        "is hard --- conditions our 1779-pattern saturated-library setup "
        "deliberately removes."
    )

    llada_full_a, _ = full_cells[1]
    llada_sa_a,   _ = stagea_cells[1]
    ev_text = (
        "\\noindent\\textbf{Ablation for No evolution.} "
        "We disable Stage~B's online registry growth: UCB still selects "
        "patterns and the attacker still instantiates them, but successful "
        "jailbreaks are not summarized back into new schemas. Because all "
        "ablation variants share the saturated 1779-pattern registry "
        "produced by a prior full run, the registry already contains the "
        "patterns that evolution would otherwise re-discover, and the "
        f"average ASR is essentially unchanged ({fmt(full_avg_a)}\\% Full "
        f"vs.\\ {fmt(sa_avg_a)}\\% No-evolution, "
        f"Tab.~\\ref{{tab:ablation_no_evolution}}). Per-victim deltas remain "
        f"informative: on LLaDA, freezing evolution costs "
        f"{fmt(d_sa_a[1])} ASR points "
        f"({fmt(llada_full_a)}\\% $\\to$ {fmt(llada_sa_a)}\\%), reflecting "
        "victim-specific patterns that only surface through online "
        "summarization; on the other two victims the gap is within noise. "
        "This indicates that online pattern expansion is the mechanism by "
        "which \\method{} adapts to victim-specific weak spots; once the "
        "library has saturated, freezing it costs little, but the saturated "
        "library itself is the product of evolution."
    )

    section = (
        "% Auto-generated by scripts/_summarise_ablation.py — do not edit by hand.\n"
        "\\subsection{Additional Results}\n\n"
        + schema_text + "\n\n"
        + t_schema + "\n"
        + fb_text + "\n\n"
        + t_fb + "\n"
        + ev_text + "\n\n"
        + t_ev + "\n"
    )
    (OUT_TAB / "additional_results.tex").write_text(section)
    print(f"wrote {OUT_TAB/'additional_results.tex'}")


if __name__ == "__main__":
    main()
