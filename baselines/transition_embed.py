"""Embedding-based transferability test (no attacker / scorer / vLLM needed).

Same as transition.py but candidate selection uses BGE-cosine top-K instead of
random sampling from the goal_type bucket. Lets us isolate "is sampling the
problem" vs "is goal-pattern fit the problem".

Usage:
    CUDA_VISIBLE_DEVICES=0 python transition_embed.py --victim dream \
        --max-goals 30 --k 5
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401

EMB_PATH = "logs/transition/embeddings.npz"
DEFAULT_OUT_DIR = "results/transition_embed"


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
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--registry", default="logs/shared_pattern_registry.json")
    ap.add_argument("--emb", default=EMB_PATH)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or f"results/transition_embed/advbench_{args.victim}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    victim_cfg_path = f"./configs/{args.victim}_attack.yaml"
    mask_token = mask_token_for(victim_cfg_path)

    import numpy as np
    from inference import load_victim, run_victim_online

    registry = json.load(open(args.registry))
    print(f"[embed] registry={len(registry)}")

    emb = np.load(args.emb, allow_pickle=True)
    pat_ids = list(emb["pattern_ids"])
    pat_E = emb["pattern_emb"]                     # (N_pat, d), L2-normalized
    goal_ids = list(emb["goal_ids"])
    goal_E = emb["goal_emb"]                       # (N_goal, d), L2-normalized
    print(f"[embed] patterns={len(pat_ids)}  goals={len(goal_ids)}")

    goals = json.load(open(args.data))[: args.max_goals]
    end = args.end if args.end is not None else len(goals)
    print(f"[embed] victim={args.victim}  goals={len(goals)}  range=[{args.start},{end})  k={args.k}  out={out_path}")

    print("[embed] loading victim model...")
    _, victim_model, victim_tokenizer = load_victim(victim_cfg_path)

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
        print(f"[embed] resume: {len(done_full)} goals already done")

    # build goal_id -> embedding row index
    gid_to_idx = {gid: i for i, gid in enumerate(goal_ids)}

    fout = open(out_path, "a")
    for idx, ex in enumerate(tqdm(goals, desc=f"embed[{args.victim}]")):
        if idx < args.start or idx >= end:
            continue
        gid = ex["id"]
        if gid in done_full:
            continue
        if gid not in gid_to_idx:
            print(f"[warn] no embedding for {gid}, skipping")
            continue
        gv = goal_E[gid_to_idx[gid]]                # (d,)
        # cosine similarities (since both L2-normalized, dot product = cosine)
        sims = pat_E @ gv                           # (N_pat,)
        topk_idx = np.argsort(-sims)[: args.k]
        sampled = [pat_ids[i] for i in topk_idx]
        scores = [float(sims[i]) for i in topk_idx]

        attempts = []
        for t, (pid, sc) in enumerate(zip(sampled, scores), start=1):
            try:
                pattern = registry[pid]
                tmpl_raw = pattern.get("representative_template") or ""
                template = expand_mask_tokens(reswap_goal_in_template(tmpl_raw, ex["goal"]),
                                              mask_token)
                output = run_victim_online(
                    victim_model=victim_model,
                    victim_tokenizer=victim_tokenizer,
                    prompt=ex["goal"],
                    template=template,
                    victim_config_path=victim_cfg_path,
                )
            except Exception as e:
                output = f"[ERROR] {str(e)[:200]}"
                template = ""
            attempts.append({
                "iteration": t,
                "pattern_id": pid,
                "cosine_similarity": sc,
                "template": template,
                "prompt": ex["goal"],
                "prediction": output,
            })

        record = {
            "goal_id": gid,
            "goal": ex["goal"],
            "k": args.k,
            "selection": "bge-cosine-top-k",
            "attempts": attempts,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()
    print(f"[embed] saved -> {out_path}")


if __name__ == "__main__":
    main()
