"""Judge Harmscore (1..5) with Bedrock Qwen3-235B for both baseline files
(single prediction per row) and transition files (k attempts per row).

Writes ``bedrock_hs`` into each prediction/attempt and aggregates:
  - baselines: avg HS per file
  - transition: avg HS of best-of-k attempts per goal, for k = 1..K

Usage:
    python scripts/_judge_hs.py --files results/transition/baselines/*.json
    python scripts/_judge_hs.py --files 'results/transition/advbench_*.json' --transition
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import _bootstrap  # noqa: F401
from evaluation import HARMSCORE_TEMPLATE, extract_score, make_judge


def judge_hs(judge_call, behavior, generation, retries=3):
    for k in range(retries):
        try:
            txt = judge_call(HARMSCORE_TEMPLATE % (behavior, generation), 1024, 0.3)
            s = extract_score(txt)
            return s
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(1.5 * (k + 1))


def process_baseline(path, judge_call, workers=16):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    scores = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_hs, judge_call,
                          r["goal"], r.get("prediction", "")): i
                for i, r in enumerate(rows)}
        for fut in tqdm(as_completed(futs), total=n,
                        desc=Path(path).stem, mininterval=2.0):
            scores[futs[fut]] = fut.result()
    for r, s in zip(rows, scores):
        r["bedrock_hs"] = s
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    valid = [s for s in scores if s is not None]
    return {"file": path, "n": n,
            "hs_avg": sum(valid) / len(valid) if valid else 0.0,
            "hs_dist": [valid.count(s) for s in (1, 2, 3, 4, 5)],
            "hs_errors": n - len(valid)}


def process_transition(path, judge_call, workers=16):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    # flatten (ri, ai, goal, prediction) jobs over all attempts
    jobs = []
    for ri, r in enumerate(rows):
        for ai, a in enumerate(r.get("attempts", [])):
            jobs.append((ri, ai, r["goal"], a.get("prediction", "")))
    scores = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_hs, judge_call, g, p): (ri, ai)
                for (ri, ai, g, p) in jobs}
        for fut in tqdm(as_completed(futs), total=len(jobs),
                        desc=Path(path).stem, mininterval=2.0):
            scores[futs[fut]] = fut.result()
    # write back into attempts
    for ri, r in enumerate(rows):
        for ai, a in enumerate(r.get("attempts", [])):
            a["bedrock_hs"] = scores.get((ri, ai))
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # best-of-k HS for k = 1..max_k
    max_k = max(len(r.get("attempts", [])) for r in rows) if rows else 0
    best_k = {k: [] for k in range(1, max_k + 1)}
    for r in rows:
        attempts = r.get("attempts", [])
        running_max = None
        for k, a in enumerate(attempts, start=1):
            s = a.get("bedrock_hs")
            if s is None:
                pass
            elif running_max is None or s > running_max:
                running_max = s
            best_k[k].append(running_max if running_max is not None else 0)
    out = {"file": path, "n": n}
    for k in range(1, max_k + 1):
        vals = best_k[k]
        out[f"k{k}_hs"] = sum(vals) / len(vals) if vals else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--transition", action="store_true",
                    help="Treat files as transition records (rows have 'attempts').")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    out_path = a.out or (
        "results/transition/bedrock_hs_summary.json"
        if a.transition else
        "results/transition/baselines/bedrock_hs_summary.json"
    )

    judge_call = make_judge("bedrock")
    files = []
    for pat in a.files:
        m = glob.glob(pat)
        files.extend(m if m else [pat])
    files = sorted(set(files))
    print(f"judging {len(files)} files (transition={a.transition}); workers={a.workers}")

    summary = {}
    for f in files:
        if a.transition:
            s = process_transition(f, judge_call, workers=a.workers)
        else:
            s = process_baseline(f, judge_call, workers=a.workers)
        stem = Path(f).stem
        summary[stem] = s
        if a.transition:
            ks = " ".join(f"k{k}={s.get(f'k{k}_hs',0):.2f}" for k in (1, 2, 3, 5)
                          if f"k{k}_hs" in s)
            print(f"  {stem:<35} n={s['n']:3d} {ks}")
        else:
            print(f"  {stem:<40} n={s['n']:3d} hs_avg={s['hs_avg']:.2f} err={s['hs_errors']}")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump(summary, open(out_path, "w"), indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
