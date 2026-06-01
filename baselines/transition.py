"""Pure-retrieval transferability test (no attacker / scorer / vLLM needed).

For each AdvBench goal:
  1. Look up goal_type from cached classification (or fall back to "general").
  2. Randomly sample k pattern_ids from pattern_set[goal_type] (or full registry
     if the bucket is empty).
  3. For each sampled pattern: take its `representative_template`, replace its
     leading goal sentence with the current AdvBench goal, expand <mask:N> to
     the victim's mask token, send to victim. No LLM-based instantiation.

The goal counts as jailbroken if any of the k attempts is judged unsafe by
Bedrock (run separately afterwards).

Usage (no vLLM required, just victim on its own GPU):
    CUDA_VISIBLE_DEVICES=0 python transition.py --victim dream \
        [--data data/advbench.json] [--max-goals 100] [--k 5]
        [--start 0 --end 100]
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401


GOAL_TYPE_CACHE_PATH = "logs/transition/goal_types_cache.json"


def expand_mask_tokens(text, mask_token):
    def replacer(match):
        n = int(match.group(1))
        return mask_token * n
    return re.sub(r"<mask:(\d+)>", replacer, text)


def mask_token_for(victim_cfg_path):
    if "dream" in victim_cfg_path or "llada2" in victim_cfg_path:
        return "<|mask|>"
    return "<|mdm_mask|>"


def reswap_goal_in_template(template_text, new_goal):
    """The pattern's representative_template starts with a literal goal sentence
    (line 1). Swap that line for ``new_goal`` so the template targets the
    current AdvBench goal."""
    if not template_text:
        return new_goal
    lines = template_text.split("\n", 1)
    if len(lines) == 1:
        return new_goal
    return f"{new_goal}\n{lines[1]}"


def load_goal_type_cache():
    """Pull cached goal_type assignments from existing transition records (any
    victim — they're all classified the same way). Falls back to empty dict if
    nothing is cached."""
    cache = {}
    if os.path.exists(GOAL_TYPE_CACHE_PATH):
        cache = json.load(open(GOAL_TYPE_CACHE_PATH))
        if cache:
            return cache
    # Fallback: scan results/transition/advbench_*.json for the goal_type field.
    for p in sorted(Path("results/transition").glob("advbench_*.json")):
        for ln in open(p):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            gid = r.get("goal_id")
            gt = r.get("goal_type")
            if gid and gt:
                cache.setdefault(gid, gt)
    if cache:
        os.makedirs(os.path.dirname(GOAL_TYPE_CACHE_PATH), exist_ok=True)
        json.dump(cache, open(GOAL_TYPE_CACHE_PATH, "w"), indent=2)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--victim", required=True,
                    choices=["dream", "llada", "llada1_5", "llada2", "mmada"])
    ap.add_argument("--data", default="data/advbench.json")
    ap.add_argument("--max-goals", type=int, default=100)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--registry", default="logs/shared_pattern_registry.json")
    ap.add_argument("--pattern-set", default="logs/shared_pattern_set.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/transition_v2/advbench_{args.victim}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    victim_cfg_path = f"./configs/{args.victim}_attack.yaml"
    mask_token = mask_token_for(victim_cfg_path)

    from inference import load_victim, run_victim_online

    registry = json.load(open(args.registry))
    pattern_set = json.load(open(args.pattern_set))
    print(f"[retrieve] registry={len(registry)} ; pattern_set={len(pattern_set)} types")

    goal_type_cache = load_goal_type_cache()
    print(f"[retrieve] goal_type cache: {len(goal_type_cache)} entries")

    goals = json.load(open(args.data))[: args.max_goals]
    end = args.end if args.end is not None else len(goals)
    print(f"[retrieve] victim={args.victim}  range=[{args.start},{end})  k={args.k}  out={out_path}")

    print("[retrieve] loading victim model...")
    _, victim_model, victim_tokenizer = load_victim(victim_cfg_path)

    # Resume support: skip goal_ids already 100% done in output file.
    done_full = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if len(r.get("attempts", [])) >= args.k:
                done_full.add(r["goal_id"])
    if done_full:
        print(f"[retrieve] resume: {len(done_full)} goals already done")

    fout = open(out_path, "a")
    for idx, ex in enumerate(tqdm(goals, desc=f"retrieve[{args.victim}]")):
        if idx < args.start or idx >= end:
            continue
        goal_id = ex["id"]
        if goal_id in done_full:
            continue
        goal = ex["goal"]
        goal_type = goal_type_cache.get(goal_id, "general")
        candidate_ids = pattern_set.get(goal_type, []) or list(registry.keys())

        rng = random.Random(args.seed * 100003 + idx)
        sampled = rng.sample(candidate_ids, k=min(args.k, len(candidate_ids)))

        attempts = []
        for t, pid in enumerate(sampled, start=1):
            try:
                pattern = registry[pid]
                tmpl_raw = pattern.get("representative_template") or ""
                template = expand_mask_tokens(reswap_goal_in_template(tmpl_raw, goal), mask_token)
                output = run_victim_online(
                    victim_model=victim_model,
                    victim_tokenizer=victim_tokenizer,
                    prompt=goal,
                    template=template,
                    victim_config_path=victim_cfg_path,
                )
            except Exception as e:
                output = f"[ERROR] {str(e)[:200]}"
                template = ""
            attempts.append({
                "iteration": t,
                "pattern_id": pid,
                "template": template,
                "prompt": goal,
                "prediction": output,
            })

        record = {
            "goal_id": goal_id,
            "goal": goal,
            "goal_type": goal_type,
            "k": args.k,
            "candidate_pool_size": len(candidate_ids),
            "attempts": attempts,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print(f"[retrieve] saved -> {out_path}")


if __name__ == "__main__":
    main()
