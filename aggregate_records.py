"""Aggregate expansion_records.jsonl files into evaluation-ready result JSONs.

Picks the best (max-reward) iteration per goal_id; falls back to the new_*
fields when the post-summarize re-attack improved reward.

Usage examples:
    # all dual100 victims for one bench
    python aggregate_records.py --bench harmbench --suffix _dual100 --out-dir results/dual100
    python aggregate_records.py --bench jailbreak --suffix _dual100 --out-dir results/dual100
    python aggregate_records.py --bench strongreject --suffix _dual100 --out-dir results/dual100

    # specific (records, output)
    python aggregate_records.py --records logs/hb_llada2_dual100/expansion_records.jsonl \
                                --out results/dual100/harmbench_llada2_dual100.json --bench harmbench
"""
import argparse, json, os

VICTIMS = ["dream", "llada", "llada1_5", "llada2", "mmada"]
BENCH_PREFIX = {"harmbench": "hb", "jailbreak": "jbb", "strongreject": "sr"}


def best_per_goal(records_path):
    best = {}
    for line in open(records_path):
        r = json.loads(line)
        g = r["goal_id"]
        cands = [(r.get("template"), r.get("victim_output"), r.get("reward", 0.0))]
        if r.get("new_template") is not None and r.get("new_reward") is not None:
            cands.append(
                (r.get("new_template"), r.get("new_victim_output") or r.get("new_response"),
                 r.get("new_reward", 0.0))
            )
        for tmpl, out, rwd in cands:
            if rwd is None:
                continue
            if g not in best or rwd > best[g]["reward"]:
                best[g] = {
                    "goal": r["goal"],
                    "template": tmpl or "",
                    "prediction": out or "",
                    "goal_id": g,
                    "reward": float(rwd),
                }
    return best


def write_results(records_path, out_path):
    if not os.path.exists(records_path):
        print(f"[skip] {records_path} not found")
        return False
    rows = list(best_per_goal(records_path).values())
    rows.sort(key=lambda x: int(x["goal_id"].rsplit("_", 1)[-1]))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    n_full = sum(1 for r in rows if r["prediction"].strip())
    print(f"[ok] {records_path} -> {out_path}  rows={len(rows)} non_empty={n_full}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCH_PREFIX), help="batch mode: aggregate all 5 victims for this bench")
    ap.add_argument("--suffix", default="_dual100", help="log dir suffix (used with --bench)")
    ap.add_argument("--out-dir", default="results/dual100", help="output dir (used with --bench)")
    ap.add_argument("--records", help="single records file path")
    ap.add_argument("--out", help="single output path (used with --records)")
    a = ap.parse_args()

    if a.records:
        if not a.out:
            ap.error("--out required when --records is given")
        write_results(a.records, a.out)
        return

    if not a.bench:
        ap.error("either --bench or --records is required")

    short = BENCH_PREFIX[a.bench]
    for v in VICTIMS:
        rec = f"logs/{short}_{v}{a.suffix}/expansion_records.jsonl"
        out = f"{a.out_dir}/{a.bench}_{v}{a.suffix}.json"
        write_results(rec, out)


if __name__ == "__main__":
    main()
