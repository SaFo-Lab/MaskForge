"""Judge each row of a baseline (pad/dija) or dual100 file with Bedrock and
write back ``bedrock_label`` per row. Used to pinpoint per-goal wins/losses
for paper figures.

Usage:
    python scripts/_judge_per_row.py --files 'results/final/*_pad.json' \
                                             'results/final/*_dija.json' \
                                             'results/dual100/*_dual100.json'
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
from evaluation import BENCH_PROMPT, parse_judge_json, make_judge


def judge_one(judge_call, goal, prediction, retries=3):
    prompt = BENCH_PROMPT["harmbench"].format(prompt=goal, response=prediction)
    for k in range(retries):
        try:
            txt = judge_call(prompt, 128, 0.0)
            return parse_judge_json(txt).get("safety_label", "safe")
        except Exception as e:
            if k == retries - 1:
                return f"err:{str(e)[:60]}"
            time.sleep(1.5 * (k + 1))


def process_file(path, judge_call, workers=16, force=False):
    with open(path) as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(l) for l in f if l.strip()]
    n = len(rows)
    needed = [i for i, r in enumerate(rows)
              if force or r.get("bedrock_label") is None]
    if not needed:
        unsafe = sum(1 for r in rows if r.get("bedrock_label") == "unsafe")
        return {"n": n, "unsafe": unsafe, "skipped": True}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, judge_call,
                          rows[i].get("goal", ""),
                          rows[i].get("prediction", "")): i
                for i in needed}
        for fut in tqdm(as_completed(futs), total=len(needed),
                        desc=Path(path).stem, mininterval=2.0):
            rows[futs[fut]]["bedrock_label"] = fut.result()

    with open(path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    unsafe = sum(1 for r in rows if r.get("bedrock_label") == "unsafe")
    return {"n": n, "unsafe": unsafe, "skipped": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true",
                    help="re-judge even if bedrock_label exists")
    a = ap.parse_args()

    judge_call = make_judge("bedrock")
    files = []
    for pat in a.files:
        m = glob.glob(pat)
        files.extend(m if m else [pat])
    files = sorted(set(files))
    print(f"judging {len(files)} files")
    for f in files:
        s = process_file(f, judge_call, workers=a.workers, force=a.force)
        tag = "(skipped)" if s.get("skipped") else ""
        print(f"  {Path(f).stem:<40} n={s['n']:3d} unsafe={s['unsafe']:3d} {tag}")


if __name__ == "__main__":
    main()
