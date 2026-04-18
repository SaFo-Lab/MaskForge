"""Compute ASR-LLM using local Qwen3-4B-Instruct-2507 (via vLLM on port 8100)
with the same judge prompts as evaluation.py."""
import json, re, sys, argparse, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

from evaluation import harmbench_prompt, jailbreakbench_prompt, strongreject_prompt, parse_judge_json

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"

def bench_prompt(benchmark, behavior, response):
    if benchmark in ("harmbench",):
        tpl = harmbench_prompt
    elif benchmark in ("jailbreak","jailbreakbench"):
        tpl = jailbreakbench_prompt
    elif benchmark in ("strongreject",):
        tpl = strongreject_prompt
    else:
        raise ValueError(benchmark)
    return tpl.format(prompt=behavior, response=response)

def judge_one(benchmark, behavior, generation):
    prompt = bench_prompt(benchmark, behavior, generation)
    resp = requests.post(VLLM_URL, json={
        "model": MODEL,
        "messages": [{"role":"user","content":prompt}],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 128,
    }, timeout=120)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    return parse_judge_json(text)

def _load_rows(infile):
    with open(infile) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def process(infile, benchmark, n_workers=16):
    rows = _load_rows(infile)
    results = [None]*len(rows)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {}
        for i,r in enumerate(rows):
            b = (r.get("Behavior") or r.get("behavior") or r.get("goal")
                 or r.get("question") or r.get("vanilla prompt")
                 or r.get("PAIR_prompt") or r.get("ReNeLLM_prompt") or "")
            g = r.get("generation") or r.get("response") or r.get("prediction") or r.get("output") or ""
            futs[ex.submit(judge_one, benchmark, b, g)] = i
        for f in tqdm(as_completed(futs), total=len(futs), desc=os.path.basename(infile)):
            i = futs[f]
            try:
                results[i] = f.result()
            except Exception as e:
                results[i] = {"safety_label":"safe","explanation":f"err:{e}"}
    labels = [int(r["safety_label"]=="unsafe") for r in results]
    return sum(labels)/len(labels), labels, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="file=bench [...]")
    ap.add_argument("--out", default="./eval_results/asr_llm_4b.json")
    a = ap.parse_args()

    summary = {}
    for pair in a.pairs:
        path, bench = pair.split("=")
        asr, labels, rows = process(path, bench)
        summary[os.path.basename(path)] = {"benchmark":bench,"n":len(rows),"asr_llm_4b":asr}
        print(f"[{path}] ASR-LLM-4B = {asr:.4f}  (n={len(rows)})")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summary, open(a.out,"w"), indent=2)
    print(f"saved -> {a.out}")

if __name__=="__main__":
    main()
