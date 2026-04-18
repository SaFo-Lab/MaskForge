#!/bin/bash
# Compute ASR-LLM on one or more result JSON files.
#
# Usage:
#   ./eval.sh <judge> <file=benchmark> [file=benchmark ...]
#
#   judge       : "local" (Qwen3-4B via vLLM@8100) or "bedrock" (Qwen3-235B on us-west-2)
#   file        : path to a JSON/JSONL result file with {Behavior|goal, generation|prediction} fields
#   benchmark   : harmbench | jailbreak | jailbreakbench | strongreject
#
# Examples:
#   ./eval.sh local ./new_results/jailbreak_dream_ours.json=jailbreak
#   ./eval.sh bedrock ./new_results/jailbreak_dream_ours.json=jailbreak ./new_results/jailbreak_dream_dija.json=jailbreak
#
# For local judge, you need vLLM Qwen3-4B-Instruct-2507 running on port 8100.
# For bedrock judge, you need AWS_BEARER_TOKEN_BEDROCK set (passed inline below).

set -e
cd "$(dirname "$0")"

JUDGE="${1:-local}"
shift || true

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <local|bedrock> <file=benchmark> [file=benchmark ...]" >&2
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm

OUT_DIR="./eval_results"
mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)

if [ "$JUDGE" = "local" ]; then
    if ! curl -s -m 3 http://localhost:8100/v1/models > /dev/null 2>&1; then
        echo "[error] vLLM on port 8100 not reachable. Start it first:" >&2
        echo "  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \\" >&2
        echo "    --model Qwen/Qwen3-4B-Instruct-2507 --port 8100 --trust-remote-code \\" >&2
        echo "    --max-model-len 4096 --gpu-memory-utilization 0.3 &" >&2
        exit 1
    fi
    python -u asr_llm_4b.py --pairs "$@" --out "$OUT_DIR/asr_llm_local_${TS}.json"
elif [ "$JUDGE" = "bedrock" ]; then
    : "${AWS_BEARER_TOKEN_BEDROCK:?set AWS_BEARER_TOKEN_BEDROCK to your Bedrock API key}" ; \
    : "${AWS_REGION:=us-west-2}" ; \
    AWS_BEARER_TOKEN_BEDROCK="$AWS_BEARER_TOKEN_BEDROCK" \
    AWS_REGION="$AWS_REGION" \
    python - "$@" << 'PY'
import json, sys, os, re, boto3
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluation import harmbench_prompt, jailbreakbench_prompt, strongreject_prompt, parse_judge_json

MODEL = "qwen.qwen3-235b-a22b-2507-v1:0"
client = boto3.client("bedrock-runtime", region_name="us-west-2")

def bench_prompt(bench, b, g):
    tpl = {
        "harmbench": harmbench_prompt,
        "jailbreak": jailbreakbench_prompt,
        "jailbreakbench": jailbreakbench_prompt,
        "strongreject": strongreject_prompt,
    }[bench]
    return tpl.format(prompt=b, response=g)

def judge(bench, b, g):
    p = bench_prompt(bench, b, g)
    try:
        r = client.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [{"text": p}]}],
            inferenceConfig={"maxTokens": 128, "temperature": 0.0, "topP": 0.9},
        )
        text = r["output"]["message"]["content"][0]["text"]
        return parse_judge_json(text)["safety_label"]
    except Exception as e:
        return f"err:{str(e)[:60]}"

def detect_fields(r):
    b = (r.get("Behavior") or r.get("behavior") or r.get("goal")
         or r.get("question") or r.get("vanilla prompt")
         or r.get("PAIR_prompt") or r.get("ReNeLLM_prompt") or "")
    g = r.get("generation") or r.get("response") or r.get("prediction") or r.get("output") or ""
    return b, g

summary = {}
for pair in sys.argv[1:]:
    if "=" not in pair:
        print(f"[skip] bad pair '{pair}', expected file=benchmark", file=sys.stderr)
        continue
    path, bench = pair.rsplit("=", 1)
    try:
        rows = json.load(open(path))
    except Exception:
        rows = [json.loads(l) for l in open(path)]

    labels = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(judge, bench, *detect_fields(r)): i for i, r in enumerate(rows)}
        for f in as_completed(futs):
            labels[futs[f]] = f.result()

    unsafe = sum(1 for l in labels if l == "unsafe")
    errs = sum(1 for l in labels if isinstance(l, str) and l.startswith("err"))
    asr = unsafe / len(rows) if rows else 0.0
    summary[os.path.basename(path)] = {
        "benchmark": bench,
        "n": len(rows),
        "asr_llm_bedrock_235b": asr,
        "unsafe": unsafe,
        "errors": errs,
    }
    print(f"[{path}] ASR-LLM-235B = {asr:.4f}  unsafe={unsafe}/{len(rows)}  errors={errs}")

ts = os.environ.get("TS", "latest")
out_path = f"./eval_results/asr_llm_bedrock_{ts}.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(summary, open(out_path, "w"), indent=2)
print(f"saved -> {out_path}")
PY
else
    echo "[error] unknown judge '$JUDGE', expected 'local' or 'bedrock'" >&2
    exit 1
fi
