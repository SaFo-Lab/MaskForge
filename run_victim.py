"""Unified victim runner. Replaces run_{hb,jbb,sr}_victim.py and the
defense-specific variants (a2d, po, safety).

Usage:
    python run_victim.py --bench {hb,jbb,sr} --victim {llada,llada1_5,llada2,dream,mmada}
                         [--defense {none,a2d,po,safety}] [--max-goals 100]
                         [--suffix _dual100] [--max-iters 3] [--success-threshold 0.9]

The output goes to logs/{bench_short}_{victim}{suffix}/expansion_records.jsonl
where bench_short = hb | jbb | sr.

For defense != none, the victim config used is configs/{victim}_{defense}_attack.yaml
otherwise configs/{victim}_attack.yaml. Defense='safety' additionally sets
SAFETY_PROMPT=1 which the victim wrapper reads to prepend a safety system prompt.
"""
import argparse, json, os, sys

BENCH_DATA = {
    "hb":  ("./data/harmbench_behaviors_text_all_refined_Qwen.json", "Behavior"),
    "jbb": ("./data/jailbreakbench_data_refined_Qwen.json",          "goal"),
    "sr":  ("./data/strongreject_data_refined_Qwen.json",            "vanilla prompt"),
}
VALID_VICTIMS = {"llada", "llada1_5", "llada2", "mmada", "dream"}
VALID_DEFENSES = {"none", "a2d", "po", "safety"}


def _record_best_reward(row):
    vals = []
    for key in ("reward", "new_reward"):
        if row.get(key) is None:
            continue
        try:
            vals.append(float(row[key]))
        except (TypeError, ValueError):
            pass
    return max(vals) if vals else 0.0


def _goal_complete(rows, max_iters, success_threshold):
    if not rows:
        return False
    best = max(_record_best_reward(row) for row in rows)
    max_iter = max(int(row.get("iteration") or 0) for row in rows)
    return best >= success_threshold or max_iter >= max_iters or len(rows) >= max_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(BENCH_DATA))
    ap.add_argument("--victim", required=True, choices=sorted(VALID_VICTIMS))
    ap.add_argument("--defense", default="none", choices=sorted(VALID_DEFENSES))
    ap.add_argument("--max-goals", type=int, default=100)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--success-threshold", type=float, default=0.9)
    ap.add_argument("--max-fallback-retries", type=int, default=3)
    ap.add_argument("--log-dir", default=None,
                    help="Override output directory for expansion logs/records.")
    ap.add_argument("--mask-ratio", type=float, default=None,
                    help="Target mask ratio r ∈ (0,1) for the rescaler in framework/attacker.py.")
    # Component ablations
    ap.add_argument("--ucb-strategy", default="ucb", choices=["ucb", "random"],
                    help="Pattern-selection strategy. 'random' = uniform within candidate pool.")
    ap.add_argument("--no-evolution", action="store_true",
                    help="Disable pattern evolution (Stage A library only, no online registry growth).")
    ap.add_argument("--isolated-registry", action="store_true",
                    help="Use a per-run shared_registry path so ablations don't pollute the global library.")
    a = ap.parse_args()

    if a.mask_ratio is not None:
        os.environ["MASK_RATIO_TARGET"] = str(a.mask_ratio)

    log_dir = a.log_dir or f"./logs/{a.bench}_{a.victim}{a.suffix}"
    bench_data_path, behavior_field = BENCH_DATA[a.bench]

    if a.defense == "none":
        victim_cfg = f"./configs/{a.victim}_attack.yaml"
    else:
        victim_cfg = f"./configs/{a.victim}_{a.defense}_attack.yaml"

    if not os.path.exists(victim_cfg):
        ap.error(f"missing victim config for --victim {a.victim} --defense {a.defense}: {victim_cfg}")

    if a.defense == "safety":
        os.environ["SAFETY_PROMPT"] = "1"

    # Late imports so help and config validation work without heavy deps.
    sys.argv = ["run", "config=./configs/expansion.yaml"]
    import expansion

    _orig_run = expansion.run_expansion

    def run(cfg):
        cfg["runtime"]["max_iters"] = a.max_iters
        cfg["runtime"]["max_fallback_retries"] = a.max_fallback_retries
        cfg["runtime"]["success_threshold"] = a.success_threshold
        cfg["runtime"]["log_dir"] = log_dir
        cfg["models"]["victim_config_path"] = victim_cfg
        # ablation knobs
        cfg["runtime"]["ucb_strategy"] = a.ucb_strategy
        cfg["runtime"]["no_evolution"] = a.no_evolution
        if a.isolated_registry:
            tag = a.suffix.lstrip("_") or "isolated"
            iso_dir = "logs/ablation_registries"
            os.makedirs(iso_dir, exist_ok=True)
            cfg["data"]["shared_registry_path"] = f"{iso_dir}/{tag}_{a.victim}_registry.json"
            cfg["data"]["shared_pattern_set_path"] = f"{iso_dir}/{tag}_{a.victim}_pattern_set.json"

        with open(bench_data_path) as f:
            items = json.load(f)
        items = items[: a.max_goals if a.max_goals > 0 else len(items)]
        data = [
            {
                "question": (it.get(behavior_field) or it.get("Behavior")
                             or it.get("goal") or it.get("vanilla prompt")
                             or it.get("prompt") or ""),
                "id": f"{a.bench}_{i}",
                "category": it.get("SemanticCategory") or it.get("category") or "unknown",
            }
            for i, it in enumerate(items)
        ]

        records_path = os.path.join(log_dir, "expansion_records.jsonl")
        rows_by_goal = {}
        if os.path.exists(records_path):
            with open(records_path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        rows_by_goal.setdefault(row["goal_id"], []).append(row)
                    except Exception:
                        pass
        done_ids = {
            goal_id
            for goal_id, rows in rows_by_goal.items()
            if _goal_complete(rows, a.max_iters, a.success_threshold)
        }
        data = [d for d in data if d["id"] not in done_ids]
        print(
            f"\n=== {a.bench.upper()} x {a.victim} (defense={a.defense}): "
            f"total={len(items)} done={len(done_ids)} remaining={len(data)} ===\n"
        )
        if not data:
            print("[SKIP] all goals done")
            return
        tmp = f"/tmp/{a.bench}_{a.victim}{a.suffix}_data.json"
        with open(tmp, "w") as f:
            json.dump(data, f)
        cfg["data"]["data_path"] = tmp
        cfg["data"]["seed_size"] = 0
        _orig_run(cfg)

    expansion.run_expansion = run
    expansion.main()


if __name__ == "__main__":
    main()
