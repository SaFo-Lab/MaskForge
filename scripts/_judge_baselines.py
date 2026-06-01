"""Judge transition baseline records (direct/pad/dija) with Bedrock Qwen3-235B
and produce one ASR per (method, victim) file."""
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


def judge_one(judge_call, behavior, generation, retries=3):
    prompt = BENCH_PROMPT["harmbench"].format(prompt=behavior, response=generation)
    for k in range(retries):
        try:
            txt = judge_call(prompt, 128, 0.0)
            return parse_judge_json(txt).get("safety_label", "safe")
        except Exception as e:
            if k == retries - 1:
                return f"err:{str(e)[:60]}"
            time.sleep(1.5 * (k + 1))


def process_file(path, judge_call, workers=16):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = len(rows)
    labels = [None] * n
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(judge_one, judge_call,
                          r["goal"], r.get("prediction", "")): i
                for i, r in enumerate(rows)}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=Path(path).stem, mininterval=2.0):
            i = futs[fut]
            labels[i] = fut.result()

    unsafe = sum(1 for l in labels if l == "unsafe")
    err = sum(1 for l in labels if isinstance(l, str) and l.startswith("err"))
    # write back
    for r, l in zip(rows, labels):
        r["bedrock_label"] = l
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"file": path, "n": n, "unsafe": unsafe,
            "asr": unsafe / n if n else 0, "errors": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default="results/transition/baselines/bedrock_asr_summary.json")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    judge_call = make_judge("bedrock")

    files = []
    for pat in a.files:
        m = glob.glob(pat)
        files.extend(m if m else [pat])
    files = sorted(set(files))

    print(f"judging {len(files)} files")
    summary = {}
    for f in files:
        s = process_file(f, judge_call, workers=a.workers)
        stem = Path(f).stem
        summary[stem] = s
        print(f"  {stem:40s}  n={s['n']:3d}  asr={s['asr']*100:5.1f}%  err={s['errors']}")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(summary, open(a.out, "w"), indent=2)

    print(f"\nsaved -> {a.out}")
    print()
    print("=== Bedrock ASR (single-shot) ===")
    print(f"{'File':<40} {'ASR':>7}")
    for stem, s in sorted(summary.items()):
        print(f"{stem:<40} {s['asr']*100:>6.1f}%")


if __name__ == "__main__":
    main()
