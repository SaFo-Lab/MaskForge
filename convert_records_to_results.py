"""Convert expansion_records.jsonl + log into evaluation-ready results JSON.

For each goal: pick the iteration with max reward; from the log, extract the
victim output whose scorer 'attack_score' matches the recorded reward.
"""
import json
import re
import os
import sys
from collections import defaultdict


def parse_log_blocks(log_path):
    """Parse log into {(goal_id, iter): [(victim_output, attack_score), ...]}."""
    if not os.path.exists(log_path):
        return defaultdict(list)
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    goal_blocks = re.split(r"===== Goal Type Extraction =====", text)
    out = defaultdict(list)

    for blk in goal_blocks[1:]:
        m_gid = re.search(r"goal_id:\s*(\S+)", blk)
        if not m_gid:
            continue
        goal_id = m_gid.group(1)

        iter_positions = [m.start() for m in re.finditer(r"=========================(\d+)=========================", blk)]
        iter_nums = [int(m.group(1)) for m in re.finditer(r"=========================(\d+)=========================", blk)]
        iter_positions.append(len(blk))

        for i, iter_num in enumerate(iter_nums):
            seg = blk[iter_positions[i]:iter_positions[i+1]]
            vo_matches = list(re.finditer(r"victim output:\n\n(.*?)(?=\n\nscorer output:|\Z)", seg, re.DOTALL))
            scorer_matches = list(re.finditer(r"'attack_score':\s*([0-9.]+)", seg))
            n = min(len(vo_matches), len(scorer_matches))
            for k in range(n):
                vo = vo_matches[k].group(1).strip()
                score = float(scorer_matches[k].group(1))
                out[(goal_id, iter_num)].append((vo, score))
    return out


def build_results(records_path, log_path, behavior_field="goal"):
    with open(records_path, "r", encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    best = {}
    for r in records:
        g = r["goal_id"]
        rwd = max(
            r["reward"],
            r.get("new_reward") or 0.0,
        )
        if g not in best or rwd > best[g]["reward"]:
            best[g] = {
                "record": r,
                "reward": rwd,
                "iter": r["iteration"],
                "new_response": r.get("new_response"),
                "used_new": bool(r.get("new_reward") is not None and r.get("new_reward", 0.0) >= r["reward"]),
            }

    log_blocks = parse_log_blocks(log_path)

    results = []
    missing = 0
    for g, info in best.items():
        r = info["record"]
        victim_output = None

        # Prefer the record's own victim_output fields (new records).
        if info["used_new"] and r.get("new_victim_output"):
            victim_output = r["new_victim_output"]
        elif r.get("victim_output"):
            victim_output = r["victim_output"]

        # Fall back to log parsing for older records without victim_output.
        if victim_output is None and info["used_new"] and info.get("new_response"):
            victim_output = info["new_response"]

        if victim_output is None:
            candidates = log_blocks.get((g, info["iter"]), [])
            if candidates:
                best_cand = min(candidates, key=lambda x: abs(x[1] - info["reward"]))
                victim_output = best_cand[0]

        if victim_output is None:
            candidates = []
            for k, v in log_blocks.items():
                if k[0] == g:
                    candidates.extend(v)
            if candidates:
                candidates.sort(key=lambda x: -x[1])
                victim_output = candidates[0][0]

        if victim_output is None:
            missing += 1
            victim_output = ""

        template = r.get("new_template") if info["used_new"] else r.get("template")
        if not template:
            template = r.get("template") or r.get("new_template")

        results.append({
            "Behavior": r["goal"],
            "generation": victim_output,
            "template": template,
            "goal_id": g,
            "reward": info["reward"],
            "iteration": info["iter"],
        })

    return results, missing


def main():
    configs = [
        {
            "name": "jailbreak",
            "records": "./logs/attack_jailbreakbench/expansion_records.jsonl",
            "log": "./logs/jailbreakbench_attack.log",
            "out": "./results/jailbreak_direct_llada2_attack.json",
        },
        {
            "name": "strongreject",
            "records": "./logs/attack_strongreject/expansion_records.jsonl",
            "log": "./logs/all_benchmarks.log",
            "out": "./results/strongreject_direct_llada2_attack.json",
        },
    ]

    os.makedirs("./results", exist_ok=True)

    for c in configs:
        if not os.path.exists(c["records"]):
            print(f"[SKIP] {c['name']}: records not found")
            continue
        if not os.path.exists(c["log"]):
            print(f"[SKIP] {c['name']}: log not found")
            continue
        results, missing = build_results(c["records"], c["log"])
        with open(c["out"], "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        non_empty = sum(1 for r in results if r["generation"].strip())
        print(f"[{c['name']}] saved {len(results)} rows -> {c['out']}  non_empty={non_empty}  missing_victim_output={missing}")


if __name__ == "__main__":
    main()
