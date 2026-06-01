"""Transferability with TRULY random pattern selection from the whole registry
(no goal_type filtering). Establishes a clean baseline against which to gauge
the embedding-retrieval lift.
"""
import argparse
import json
import os
import random
import re

from tqdm import tqdm

import _bootstrap  # noqa: F401


def expand_mask_tokens(text, mask_token):
    def replacer(match):
        return mask_token * int(match.group(1))
    return re.sub(r"<mask:(\d+)>", replacer, text)


def mask_token_for(victim_cfg_path):
    if "dream" in victim_cfg_path or "llada2" in victim_cfg_path:
        return "<|mask|>"
    return "<|mdm_mask|>"


def reswap_goal_in_template(template_text, new_goal):
    if not template_text:
        return new_goal
    lines = template_text.split("\n", 1)
    if len(lines) == 1:
        return new_goal
    return f"{new_goal}\n{lines[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", required=True,
                    choices=["dream", "llada", "llada1_5", "llada2", "mmada"])
    ap.add_argument("--data", default="data/advbench.json")
    ap.add_argument("--max-goals", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--registry", default="logs/shared_pattern_registry.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/transition_random_global/advbench_{args.victim}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    victim_cfg_path = f"./configs/{args.victim}_attack.yaml"
    mask_token = mask_token_for(victim_cfg_path)

    from inference import load_victim, run_victim_online

    registry = json.load(open(args.registry))
    all_pids = list(registry.keys())
    print(f"[random-global] registry={len(all_pids)}")

    goals = json.load(open(args.data))[: args.max_goals]
    print(f"[random-global] victim={args.victim}  goals={len(goals)}  k={args.k}")

    print("[random-global] loading victim model...")
    _, vm, vt = load_victim(victim_cfg_path)

    fout = open(out_path, "w")
    for idx, ex in enumerate(tqdm(goals, desc=f"random-global[{args.victim}]")):
        rng = random.Random(args.seed * 100003 + idx)
        sampled = rng.sample(all_pids, k=min(args.k, len(all_pids)))

        attempts = []
        for t, pid in enumerate(sampled, start=1):
            try:
                pattern = registry[pid]
                tmpl_raw = pattern.get("representative_template") or ""
                template = expand_mask_tokens(reswap_goal_in_template(tmpl_raw, ex["goal"]),
                                              mask_token)
                output = run_victim_online(
                    victim_model=vm, victim_tokenizer=vt,
                    prompt=ex["goal"], template=template,
                    victim_config_path=victim_cfg_path,
                )
            except Exception as e:
                output = f"[ERROR] {str(e)[:200]}"
                template = ""
            attempts.append({
                "iteration": t, "pattern_id": pid,
                "template": template, "prompt": ex["goal"], "prediction": output,
            })

        record = {"goal_id": ex["id"], "goal": ex["goal"], "k": args.k,
                  "selection": "uniform-random-over-registry", "attempts": attempts}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()
    print(f"[random-global] saved -> {out_path}")


if __name__ == "__main__":
    main()
