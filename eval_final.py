"""Compute ASR-K + ASR-LLM (Bedrock 235B) on all files in final_results/.

Usage:
  AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2 python eval_final.py
Output: final_results/asr_summary.json
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
sys.path.insert(0, '/weka/home/ext-yingzima/Attack4dLLM')
from evaluation import (harmbench_prompt, jailbreakbench_prompt, strongreject_prompt,
                        parse_judge_json, is_jailbroken_keyword)

FINAL = '/weka/home/ext-yingzima/Attack4dLLM/final_results'
OUT = f'{FINAL}/asr_summary.json'
MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"
client = boto3.client("bedrock-runtime", region_name="us-west-2")

PROMPTS = {
    'harmbench':     harmbench_prompt,
    'jailbreak':     jailbreakbench_prompt,
    'strongreject':  strongreject_prompt,
}


def judge(bench, goal, gen):
    p = PROMPTS[bench].format(prompt=goal, response=gen)
    try:
        r = client.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [{"text": p}]}],
            inferenceConfig={"maxTokens": 128, "temperature": 0.0, "topP": 0.9},
        )
        return parse_judge_json(r["output"]["message"]["content"][0]["text"])["safety_label"]
    except Exception as e:
        return f"err:{str(e)[:40]}"


def eval_file(path, bench, workers=12):
    rows = json.load(open(path))
    if not rows:
        return {"n": 0, "asr_k": None, "asr_llm": None}
    # ASR-K
    asr_k = sum(is_jailbroken_keyword(r.get("prediction", "")) for r in rows) / len(rows)
    # ASR-LLM
    labels = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge, bench, r.get("goal", ""), r.get("prediction", "")): i
                for i, r in enumerate(rows)}
        for f in as_completed(futs):
            labels[futs[f]] = f.result()
    u = sum(1 for l in labels if l == "unsafe")
    errs = sum(1 for l in labels if isinstance(l, str) and l.startswith("err"))
    return {"n": len(rows), "asr_k": asr_k, "asr_llm": u / len(rows),
            "unsafe": u, "errors": errs}


def main():
    # Load existing summary if present to allow resumption.
    summary = {}
    if os.path.exists(OUT):
        summary = json.load(open(OUT))

    files = sorted(f for f in os.listdir(FINAL) if f.endswith('.json') and f != 'asr_summary.json')
    total = len(files)
    for i, f in enumerate(files, 1):
        key = f.replace('.json', '')
        if key in summary and 'asr_llm' in summary[key]:
            print(f"[{i}/{total}] skip (cached) {f}")
            continue
        # Parse filename: {benchmark}_{victim}_{attack}.json
        bench = f.split('_')[0]
        res = eval_file(f'{FINAL}/{f}', bench)
        summary[key] = res
        k_pct = res['asr_k'] * 100 if res['asr_k'] is not None else -1
        e_pct = res['asr_llm'] * 100 if res['asr_llm'] is not None else -1
        print(f"[{i}/{total}] {f}: n={res['n']}  ASR-K={k_pct:.1f}  ASR-E={e_pct:.1f}  errs={res.get('errors',0)}")
        json.dump(summary, open(OUT, 'w'), indent=2)

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
