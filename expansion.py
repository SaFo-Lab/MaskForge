import os
import sys
import json
import yaml
import math
import glob
import hashlib
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from llm import HuggingFaceModel, BedrockModel
from framework import Attacker, PatternExtractor, Summarizer, Scorer, Retriever
from inference import load_victim, run_victim_online
from utils import preprocess_stage_a_artifacts, get_goal_type, parse_score_result



def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    """
    Parse command line args in key=value format.
    Example:
        python stage_b.py config=../configs/expansion.yaml
    """
    parsed = {}
    for arg in argv[1:]:
        if "=" not in arg:
            raise ValueError(
                f"Invalid argument format: {arg}. Expected key=value."
            )
        key, value = arg.split("=", 1)
        parsed[key] = value
    return parsed


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Pattern:
    pattern_id: str
    schema: Dict[str, Any]
    representative_template: Optional[str] = None
    source: str = "discovery"


_GENERIC_ROLE_NAMES = {
    "input", "slot", "placeholder", "value", "field", "item",
    "x", "y", "z", "a", "b", "c",
}
_GENERIC_ROLE_RE = __import__("re").compile(
    r"^(?:input|slot|placeholder|value|field|item|mask|fill)[_\s-]?\d*$",
    __import__("re").IGNORECASE,
)


def _is_low_quality_schema(schema: Dict[str, Any]) -> Optional[str]:
    """Return None if the schema is acceptable, else a short reason string.

    Rejects schemas whose slot_roles carry no semantic information — these were
    produced by a deprecated recovery path that hard-coded
    ``slot_roles = ["input"] * slot_count`` and ``organization_style = "labeled"``,
    and they pollute the library because attacker.instantiate has nothing
    meaningful to condition on.
    """
    if not isinstance(schema, dict):
        return "schema_not_dict"
    roles = schema.get("slot_roles")
    if not isinstance(roles, list):
        return "slot_roles_not_list"
    slot_count = schema.get("slot_count")
    if isinstance(slot_count, int) and slot_count > 0 and not roles:
        return "slot_roles_empty_but_slot_count_positive"
    if roles:
        norm = [str(r).strip().lower() for r in roles]
        if all(_GENERIC_ROLE_RE.fullmatch(r) for r in norm):
            return "all_slot_roles_generic"
        if len(set(norm)) == 1 and norm[0] in _GENERIC_ROLE_NAMES:
            return f"all_slot_roles_identical_generic({norm[0]!r})"
        if len(roles) >= 4 and len(set(norm)) == 1:
            return f"all_slot_roles_identical({norm[0]!r})"
    return None


class PatternRegistry:
    def __init__(self) -> None:
        self.patterns: Dict[str, Pattern] = {}
        self._reject_counts: Dict[str, int] = defaultdict(int)

    def add_pattern(
        self,
        schema: Dict[str, Any],
        template: Optional[str] = None,
        source: str = "discovery",
    ) -> Optional[str]:
        reason = _is_low_quality_schema(schema)
        if reason is not None:
            self._reject_counts[reason] += 1
            return None
        pattern_id = stable_hash(schema)
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = Pattern(
                pattern_id=pattern_id,
                schema=schema,
                representative_template=template,
                source=source,
            )
        return pattern_id

    def ids(self) -> List[str]:
        return list(self.patterns.keys())

    def get(self, pattern_id: str) -> Pattern:
        return self.patterns[pattern_id]

    def load_from_json(self, path: str) -> None:
        raw = load_json(path)
        self.patterns = {
            pid: Pattern(**pdict)
            for pid, pdict in raw.items()
        }

    def to_json(self) -> Dict[str, Any]:
        return {
            pattern_id: asdict(pattern)
            for pattern_id, pattern in self.patterns.items()
        }


@dataclass
class PatternStats:
    mean_reward: float = 0.0
    visit_count: int = 0


@dataclass
class ExpansionRecord:
    goal_id: str
    goal: str
    goal_type: str
    iteration: int

    selected_pattern_id: str
    selected_pattern_schema: Dict[str, Any]

    template: Optional[str]
    prompt: Optional[str]
    response: Optional[str]
    raw_template: Optional[str]
    reward: float
    victim_output: Optional[str] = None

    prev_pattern_id: Optional[str] = None
    prev_reward: Optional[float] = None

    new_pattern_id: Optional[str] = None
    new_pattern_schema: Optional[Dict[str, Any]] = None
    new_template: Optional[str] = None
    new_prompt: Optional[str] = None
    new_response: Optional[str] = None
    new_victim_output: Optional[str] = None
    new_reward: Optional[float] = None
    improvement_written: bool = False


def load_single_model(model_path: str, use_vllm: bool = False, **vllm_kwargs):
    if "/" in model_path:
        if use_vllm:
            from llm import VLLMModel
            return VLLMModel(model_path, **vllm_kwargs)
        return HuggingFaceModel(model_path)
    return BedrockModel(model_path)


def load_attacker(
    attack_model_path: str,
    weaker_model_path: str,
    use_vllm: bool = False,
    **vllm_kwargs,
):
    attack_model = load_single_model(attack_model_path, use_vllm=use_vllm, **vllm_kwargs)
    weaker_model = load_single_model(weaker_model_path, use_vllm=use_vllm, **vllm_kwargs)
    attacker = Attacker(
        attack_model,
        template_library=[],
        weaker_model=weaker_model,
    )
    return attacker, attack_model


def resolve_model(
    target_model_path: str,
    attack_model_path: str,
    attack_model,
    shared_models: Dict[str, Any],
    use_vllm: bool = False,
    **vllm_kwargs,
):
    if target_model_path == attack_model_path:
        return attack_model
    if target_model_path in shared_models:
        return shared_models[target_model_path]

    model = load_single_model(target_model_path, use_vllm=use_vllm, **vllm_kwargs)
    shared_models[target_model_path] = model
    return model


def build_pattern_set(registry: PatternRegistry) -> Dict[str, List[str]]:
    pattern_set = defaultdict(list)
    for pattern_id in registry.ids():
        schema = registry.get(pattern_id).schema
        goal_types = schema.get("goal_type", [])
        if isinstance(goal_types, str):
            goal_types = [goal_types]
        for goal_type in goal_types:
            pattern_set[goal_type].append(pattern_id)
    return dict(pattern_set)


def initialize_stats(registry: PatternRegistry) -> Dict[str, PatternStats]:
    return {
        pid: PatternStats(mean_reward=0.0, visit_count=0)
        for pid in registry.ids()
    }


def load_stats(path: str, registry: PatternRegistry) -> Dict[str, PatternStats]:
    theta = initialize_stats(registry)
    if not os.path.exists(path):
        return theta

    raw = load_json(path)
    if not isinstance(raw, dict):
        return theta

    for pid, stat in raw.items():
        if pid not in theta or not isinstance(stat, dict):
            continue
        try:
            theta[pid] = PatternStats(
                mean_reward=float(stat.get("mean_reward", 0.0)),
                visit_count=int(stat.get("visit_count", 0)),
            )
        except (TypeError, ValueError):
            continue
    return theta


def merge_stats(
    theta: Dict[str, PatternStats],
    prior: Dict[str, PatternStats],
) -> Dict[str, PatternStats]:
    for pid, stat in prior.items():
        if stat.visit_count <= 0:
            continue
        if pid not in theta:
            theta[pid] = PatternStats()
        current = theta[pid]
        total = current.visit_count + stat.visit_count
        if total <= 0:
            continue
        theta[pid] = PatternStats(
            mean_reward=(
                current.mean_reward * current.visit_count
                + stat.mean_reward * stat.visit_count
            ) / total,
            visit_count=total,
        )
    return theta


def stats_to_json(theta: Dict[str, PatternStats]) -> Dict[str, Dict[str, Any]]:
    return {
        pid: {
            "mean_reward": stat.mean_reward,
            "visit_count": stat.visit_count,
        }
        for pid, stat in theta.items()
    }


def count_jsonl_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def load_bootstrap_stats(
    record_globs: List[str],
    registry: PatternRegistry,
) -> Dict[str, PatternStats]:
    theta = initialize_stats(registry)
    seen_paths = set()
    for pattern in record_globs:
        for records_path in glob.glob(pattern, recursive=True):
            if records_path in seen_paths or not os.path.isfile(records_path):
                continue
            seen_paths.add(records_path)
            merge_stats(theta, replay_stats_from_records(records_path, registry))
    return theta


def update_stats(
    theta: Dict[str, PatternStats],
    pattern_id: str,
    reward: float,
) -> None:
    if pattern_id not in theta:
        theta[pattern_id] = PatternStats()

    theta[pattern_id].visit_count += 1
    n = theta[pattern_id].visit_count
    mu = theta[pattern_id].mean_reward
    theta[pattern_id].mean_reward = mu + (reward - mu) / n


def replay_stats_from_records(
    records_path: str,
    registry: PatternRegistry,
    skip_rows: int = 0,
) -> Dict[str, PatternStats]:
    theta = initialize_stats(registry)
    if not os.path.exists(records_path):
        return theta

    with open(records_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if row_idx < skip_rows:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            selected_pid = row.get("selected_pattern_id")
            if selected_pid in registry.patterns:
                try:
                    update_stats(theta, selected_pid, float(row.get("reward", 0.0)))
                except (TypeError, ValueError):
                    pass

            new_pid = row.get("new_pattern_id")
            if new_pid in registry.patterns and row.get("new_reward") is not None:
                try:
                    update_stats(theta, new_pid, float(row["new_reward"]))
                except (TypeError, ValueError):
                    pass
    return theta


def select_pattern_ucb(
    candidate_ids: List[str],
    theta: Dict[str, PatternStats],
    step_t: int,
    alpha: float,
    tie_break_key: Optional[str] = None,
    unvisited_prior_count: int = 0,
    unvisited_prior_reward: float = 0.0,
    bonus_cap: Optional[float] = None,
    selection_pool_size: int = 1,
) -> str:
    """
    UCB:
        mu_i + alpha * sqrt(2 ln t / n_i)

    Unvisited arms are selected first.
    """
    ordered_candidate_ids = list(candidate_ids)
    if tie_break_key is not None:
        ordered_candidate_ids = sorted(
            ordered_candidate_ids,
            key=lambda pid: stable_hash({"tie": tie_break_key, "pid": pid}),
        )

    if unvisited_prior_count <= 0:
        for pid in ordered_candidate_ids:
            if pid not in theta or theta[pid].visit_count == 0:
                return pid

    scored: List[Tuple[str, float, int]] = []

    for order, pid in enumerate(ordered_candidate_ids):
        stat = theta.get(pid, PatternStats())
        if stat.visit_count <= 0:
            mu = unvisited_prior_reward
            n = unvisited_prior_count
        else:
            mu = stat.mean_reward
            n = stat.visit_count
        bonus = alpha * math.sqrt(2.0 * math.log(max(step_t, 2)) / n)
        if bonus_cap is not None:
            bonus = min(bonus, bonus_cap)
        ucb = mu + bonus
        scored.append((pid, ucb, order))

    if not scored:
        raise ValueError("No valid pattern selected by UCB.")

    scored.sort(key=lambda item: (-item[1], item[2]))
    pool_size = max(1, int(selection_pool_size))
    pool = scored[:pool_size]
    if tie_break_key is not None and pool_size > 1:
        pool = sorted(
            pool,
            key=lambda item: stable_hash({"ucb_pool": tie_break_key, "pid": item[0]}),
        )
    return pool[0][0]


def describe_ucb_candidates(
    candidate_ids: List[str],
    theta: Dict[str, PatternStats],
    step_t: int,
    alpha: float,
    registry: PatternRegistry,
    top_k: int,
    tie_break_key: Optional[str] = None,
    unvisited_prior_count: int = 0,
    unvisited_prior_reward: float = 0.0,
    bonus_cap: Optional[float] = None,
) -> List[Dict[str, Any]]:
    ordered_candidate_ids = list(candidate_ids)
    if tie_break_key is not None:
        ordered_candidate_ids = sorted(
            ordered_candidate_ids,
            key=lambda pid: stable_hash({"tie": tie_break_key, "pid": pid}),
        )

    rows = []
    for order, pid in enumerate(ordered_candidate_ids):
        stat = theta.get(pid, PatternStats())
        score_for_sort = float("inf")
        bonus = None
        ucb = None
        effective_visit_count = stat.visit_count
        effective_mean_reward = stat.mean_reward
        if stat.visit_count <= 0 and unvisited_prior_count > 0:
            effective_visit_count = unvisited_prior_count
            effective_mean_reward = unvisited_prior_reward
        if effective_visit_count > 0:
            bonus = alpha * math.sqrt(
                2.0 * math.log(max(step_t, 2)) / effective_visit_count
            )
            if bonus_cap is not None:
                bonus = min(bonus, bonus_cap)
            ucb = effective_mean_reward + bonus
            score_for_sort = ucb
        schema = registry.get(pid).schema if pid in registry.patterns else {}
        rows.append({
            "pattern_id": pid,
            "source": registry.get(pid).source if pid in registry.patterns else None,
            "structure_type": schema.get("structure_type"),
            "visit_count": stat.visit_count,
            "mean_reward": round(stat.mean_reward, 4),
            "effective_visit_count": effective_visit_count,
            "effective_mean_reward": round(effective_mean_reward, 4),
            "exploration_bonus": None if bonus is None else round(bonus, 4),
            "ucb_score": None if ucb is None else round(ucb, 4),
            "unvisited": stat.visit_count <= 0,
            "_score_for_sort": score_for_sort,
            "_tie_order": order,
        })

    rows.sort(key=lambda row: (-row["_score_for_sort"], row["_tie_order"]))
    for row in rows:
        row.pop("_score_for_sort", None)
        row.pop("_tie_order", None)
    return rows[:top_k]


def _record_best_reward(row: Dict[str, Any]) -> float:
    vals = []
    for key in ("reward", "new_reward"):
        if row.get(key) is None:
            continue
        try:
            vals.append(float(row[key]))
        except (TypeError, ValueError):
            pass
    return max(vals) if vals else 0.0


def _goal_complete(
    rows: List[Dict[str, Any]],
    max_iters: int,
    success_threshold: float,
) -> bool:
    if not rows:
        return False
    best = max(_record_best_reward(row) for row in rows)
    max_iter = max(int(row.get("iteration") or 0) for row in rows)
    return best >= success_threshold or max_iter >= max_iters or len(rows) >= max_iters


def load_stage_a_artifacts(
    init_registry_path: str,
    init_pattern_set_path: Optional[str] = None,
) -> Tuple[PatternRegistry, Dict[str, List[str]]]:
    registry = PatternRegistry()
    registry.load_from_json(init_registry_path)

    if init_pattern_set_path and os.path.exists(init_pattern_set_path):
        pattern_set = load_json(init_pattern_set_path)
    else:
        pattern_set = build_pattern_set(registry)

    return registry, pattern_set


def run_expansion(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg["models"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    log_dir = runtime_cfg["log_dir"]
    ensure_dir(log_dir)
    max_iters = runtime_cfg.get("max_iters", 5)
    success_threshold = runtime_cfg.get("success_threshold", 1.0)

    with open(data_cfg["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    seed_size = data_cfg.get("seed_size", -1)
    if seed_size > 0:
        data = data[seed_size:]

    records_path = os.path.join(log_dir, "expansion_records.jsonl")
    rows_by_goal: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if os.path.exists(records_path):
        with open(records_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                goal_id = row.get("goal_id")
                if goal_id is not None:
                    rows_by_goal[goal_id].append(row)
        done_goal_ids = {
            goal_id
            for goal_id, rows in rows_by_goal.items()
            if _goal_complete(rows, max_iters, success_threshold)
        }
        if done_goal_ids:
            print(f"[expansion] resume: skipping {len(done_goal_ids)} goals already in {records_path}")
    else:
        done_goal_ids = set()

    pending_data = []
    for idx, example in enumerate(data):
        if "question" not in example:
            continue
        if "id" not in example:
            example = dict(example)
            example["id"] = f"goal_{idx}"
        if example["id"] not in done_goal_ids:
            pending_data.append(example)

    if not pending_data:
        print("[expansion] resume: all goals already completed; skipping model load.")
        save_json(os.path.join(log_dir, "config.json"), cfg)
        return
    data = pending_data

    use_vllm = runtime_cfg.get("use_vllm", True)

    if use_vllm:
        from llm import VLLMServerModel
        # Connect to vLLM servers
        instruct_port = runtime_cfg.get("vllm_instruct_port", 8100)
        base_port = runtime_cfg.get("vllm_base_port", 8101)
        attack_model = VLLMServerModel(
            f"http://localhost:{instruct_port}",
            model_cfg["attack_model_path"])
        weaker_model = VLLMServerModel(
            f"http://localhost:{base_port}",
            model_cfg["attack_weaker_model_path"])
    else:
        attack_model = load_single_model(model_cfg["attack_model_path"])
        weaker_model = load_single_model(model_cfg["attack_weaker_model_path"])

    attacker = Attacker(attack_model, template_library=[], weaker_model=weaker_model)

    victim_config_path = model_cfg["victim_config_path"]
    _, victim_model, victim_tokenizer = load_victim(victim_config_path)

    # Non-victim roles: attack_model (vLLM heretic) for generation tasks,
    # scorer uses a separate HF model (Qwen3-4B-Instruct) for accurate evaluation.
    # Scorer uses the same instruct model (attack_model)
    scorer_model = attack_model

    extractor = PatternExtractor(attack_model)
    summarizer = Summarizer(attack_model)
    scorer = Scorer(scorer_model)
    retriever = Retriever(attack_model)


    # Prefer a SHARED registry (persists across runs) if it exists.
    shared_registry_path = data_cfg.get("shared_registry_path",
                                        "./logs/shared_pattern_registry.json")
    shared_pattern_set_path = data_cfg.get("shared_pattern_set_path",
                                           "./logs/shared_pattern_set.json")
    if os.path.exists(shared_registry_path):
        registry_load_path = shared_registry_path
        pattern_set_load_path = shared_pattern_set_path if os.path.exists(shared_pattern_set_path) else data_cfg.get("init_pattern_set_path", None)
        print(f"[expansion] loading SHARED registry from {registry_load_path}")
    else:
        registry_load_path = data_cfg["init_pattern_registry_path"]
        pattern_set_load_path = data_cfg.get("init_pattern_set_path", None)
        print(f"[expansion] loading initial registry from {registry_load_path}")

    registry, pattern_set = load_stage_a_artifacts(
        init_registry_path=registry_load_path,
        init_pattern_set_path=pattern_set_load_path,
    )

    registry, pattern_set, data, goal_type_mapping = preprocess_stage_a_artifacts(
        data=data,
        registry=registry,
        pattern_set=pattern_set,
        retriever=retriever,
        verbose=True,
    )

    theta_path = os.path.join(log_dir, "theta.json")
    shared_theta_path = data_cfg.get("shared_theta_path", "./logs/shared_theta.json")
    shared_theta_manifest_path = data_cfg.get(
        "shared_theta_manifest_path",
        f"{shared_theta_path}.manifest.json",
    )
    shared_theta_manifest = {"record_files": {}}
    if os.path.exists(shared_theta_manifest_path):
        try:
            loaded_manifest = load_json(shared_theta_manifest_path)
            if isinstance(loaded_manifest, dict):
                shared_theta_manifest = loaded_manifest
                shared_theta_manifest.setdefault("record_files", {})
        except Exception as _e:
            print(f"[expansion] warn: failed to load shared theta manifest: {_e}")

    theta = initialize_stats(registry)

    if os.path.exists(shared_theta_path):
        merge_stats(theta, load_stats(shared_theta_path, registry))
        print(f"[expansion] loading SHARED theta from {shared_theta_path}")
    else:
        bootstrap_globs = data_cfg.get("theta_bootstrap_globs", [])
        if isinstance(bootstrap_globs, str):
            bootstrap_globs = [bootstrap_globs]
        if bootstrap_globs:
            merge_stats(theta, load_bootstrap_stats(bootstrap_globs, registry))
            print(f"[expansion] bootstrapped theta from {len(bootstrap_globs)} glob(s)")

    records_abs_path = os.path.abspath(records_path)
    if os.path.exists(records_path):
        try:
            already_shared_rows = int(
                shared_theta_manifest.get("record_files", {}).get(records_abs_path, 0)
            )
        except (TypeError, ValueError):
            already_shared_rows = 0
        local_rows = count_jsonl_rows(records_path)
        merge_stats(
            theta,
            replay_stats_from_records(
                records_path,
                registry,
                skip_rows=min(already_shared_rows, local_rows),
            ),
        )
    else:
        merge_stats(theta, load_stats(theta_path, registry))
    records: List[ExpansionRecord] = []

    max_length = runtime_cfg.get("max_length", 512)
    alpha = runtime_cfg.get("alpha", 1.0)
    unvisited_prior_count = int(runtime_cfg.get("ucb_unvisited_prior_count", 32))
    unvisited_prior_reward = float(runtime_cfg.get("ucb_unvisited_prior_reward", 0.0))
    bonus_cap_raw = runtime_cfg.get("ucb_bonus_cap", 1.0)
    ucb_bonus_cap = None if bonus_cap_raw is None else float(bonus_cap_raw)
    ucb_selection_pool_size = int(runtime_cfg.get("ucb_selection_pool_size", 3))
    verbose = runtime_cfg.get("verbose", False)
    fallback_to_all_patterns = runtime_cfg.get("fallback_to_all_patterns", True)
    # ablation knobs
    ucb_strategy = runtime_cfg.get("ucb_strategy", "ucb")  # "ucb" | "random"
    no_evolution = bool(runtime_cfg.get("no_evolution", False))

    for idx, example in tqdm(
        enumerate(data),
        total=len(data),
        desc="Stage B Expansion",
    ):
        goal = example["question"]
        goal_id = example["id"]
        print(f"[progress] {idx+1}/{len(data)} goal_id={goal_id}", flush=True)
        goal_type = get_goal_type(goal, retriever, list(pattern_set.keys()))

        print("\n===== Goal Type Extraction =====")
        print(f"goal_id: {goal_id}")
        print(f"goal: {goal}")
        print(f"goal_type: {goal_type}")

        candidate_ids = pattern_set.get(goal_type, [])
        min_candidates = runtime_cfg.get("min_candidates", 0)
        if len(candidate_ids) == 0 and fallback_to_all_patterns:
            candidate_ids = registry.ids()
        elif min_candidates and len(candidate_ids) < min_candidates:
            extras = [pid for pid in registry.ids() if pid not in candidate_ids]
            candidate_ids = list(candidate_ids) + extras

        if len(candidate_ids) == 0:
            if verbose:
                print(f"[Skip] No candidate patterns for goal {goal_id}")
            continue

        prev_selected_pattern_id: Optional[str] = None
        prev_selected_pattern_schema: Optional[Dict[str, Any]] = None
        prev_template: Optional[str] = None
        prev_reward: Optional[float] = None

        existing_attempts = min(len(rows_by_goal.get(goal_id, [])), max_iters)
        for t in range(existing_attempts + 1, max_iters + 1):
            ucb_step_t = 1 + sum(
                theta.get(pid, PatternStats()).visit_count
                for pid in candidate_ids
            )
            ucb_tie_break_key = f"{goal_id}:{t}"
            ucb_debug_top_k = int(runtime_cfg.get("ucb_debug_top_k", 5))
            if ucb_debug_top_k > 0:
                print(
                    "[UCB] top candidates: "
                    + json.dumps(
                        describe_ucb_candidates(
                            candidate_ids=candidate_ids,
                            theta=theta,
                            step_t=ucb_step_t,
                            alpha=alpha,
                            registry=registry,
                            top_k=ucb_debug_top_k,
                            tie_break_key=ucb_tie_break_key,
                            unvisited_prior_count=unvisited_prior_count,
                            unvisited_prior_reward=unvisited_prior_reward,
                            bonus_cap=ucb_bonus_cap,
                        ),
                        ensure_ascii=False,
                    )
                )
            if ucb_strategy == "random":
                # Ablation: ignore UCB, sample uniformly from the candidate pool.
                # Deterministic per-(goal, iter) via stable_hash so the experiment
                # is reproducible.
                _idx = int(stable_hash({"rand": ucb_tie_break_key}), 16) % len(candidate_ids)
                selected_pattern_id = list(candidate_ids)[_idx]
            else:
                selected_pattern_id = select_pattern_ucb(
                    candidate_ids=candidate_ids,
                    theta=theta,
                    step_t=ucb_step_t,
                    alpha=alpha,
                    tie_break_key=ucb_tie_break_key,
                    unvisited_prior_count=unvisited_prior_count,
                    unvisited_prior_reward=unvisited_prior_reward,
                    bonus_cap=ucb_bonus_cap,
                    selection_pool_size=ucb_selection_pool_size,
                )
            selected_pattern = registry.get(selected_pattern_id)

            print("="*25 + str(t) + "="*25)
            print(f"selected_pattern: {selected_pattern}")

            prompt, template, response, raw_template = attacker.instantiate(
                goal,
                pattern=selected_pattern,
                max_length=max_length,
            )


            print(f"\nattack prompt:\n{prompt}")
            print(f"\nmalicious response:\n{response}")
            print(f"\nraw template:\n{raw_template}")
            print(f"\nprocessed template:\n{template}")

            output = run_victim_online(
                victim_model=victim_model,
                victim_tokenizer=victim_tokenizer,
                prompt=prompt,
                template=template,
                victim_config_path=victim_config_path,
            )

            print(f"\n\nvictim output:\n\n{output}")


            results = scorer.score(goal, prompt, template, output)

            print(f"\n\nscorer output:\n\n{results}")

            reward, label, evidence = parse_score_result(results)
            reward = float(reward)

            # Fallback: if dija template failed (low reward), try weaker model approach with retries
            fallback_threshold = runtime_cfg.get("fallback_threshold", 0.7)
            max_fallback_retries = runtime_cfg.get("max_fallback_retries", 5)
            last_analysis = ""
            try:
                last_analysis = (results or {}).get("analysis", "") if isinstance(results, dict) else ""
            except Exception:
                last_analysis = ""
            if reward < fallback_threshold:
                for retry in range(max_fallback_retries):
                    print(f"\n[Fallback] reward={reward:.4f} < {fallback_threshold}, attempt {retry+1}/{max_fallback_retries}...")
                    fallback_template, fallback_response = attacker.generate_fallback_template(
                        goal, pattern=selected_pattern, max_length=max_length,
                        _retry_count=retry, scorer_feedback=last_analysis[:600])

                    if fallback_template is None:
                        print("[Fallback] Failed to generate template, retrying...")
                        continue

                    print(f"\nfallback template:\n{fallback_template[:300]}...")

                    fallback_output = run_victim_online(
                        victim_model=victim_model,
                        victim_tokenizer=victim_tokenizer,
                        prompt=prompt,
                        template=fallback_template,
                        victim_config_path=victim_config_path,
                    )

                    print(f"\nfallback victim output:\n{fallback_output}")

                    fallback_results = scorer.score(goal, prompt, fallback_template, fallback_output)
                    fallback_reward, fallback_label, fallback_evidence = parse_score_result(fallback_results)
                    fallback_reward = float(fallback_reward)
                    try:
                        last_analysis = (fallback_results or {}).get("analysis", "") if isinstance(fallback_results, dict) else ""
                    except Exception:
                        last_analysis = ""

                    print(f"[Fallback] attempt {retry+1} fallback_reward={fallback_reward:.4f} vs current best={reward:.4f}")

                    if fallback_reward > reward:
                        print(f"[Fallback] Using fallback template (reward {reward:.4f} -> {fallback_reward:.4f})")
                        reward = fallback_reward
                        template = fallback_template
                        output = fallback_output
                        response = fallback_response
                        raw_template = fallback_template

                    if reward >= fallback_threshold:
                        print(f"[Fallback] Reward {reward:.4f} >= threshold, stopping retries.")
                        break

            update_stats(theta, selected_pattern_id, reward)

            record = ExpansionRecord(
                goal_id=goal_id,
                goal=goal,
                goal_type=goal_type,
                iteration=t,
                selected_pattern_id=selected_pattern_id,
                selected_pattern_schema=selected_pattern.schema,
                template=template,
                prompt=prompt,
                response=response,
                raw_template=raw_template,
                reward=reward,
                victim_output=output,
                prev_pattern_id=prev_selected_pattern_id,
                prev_reward=prev_reward,
            )

            if (
                not no_evolution
                and t >= 2
                and prev_reward is not None
                and reward >= prev_reward
                and prev_selected_pattern_schema is not None
                and prev_template is not None
            ):
                summarizer_result = summarizer.summarize(
                    goal=goal,
                    prev_pattern_schema=prev_selected_pattern_schema,
                    prev_template=prev_template,
                    prev_reward=prev_reward,
                    curr_pattern_schema=selected_pattern.schema,
                    curr_template=template,
                    curr_reward=reward,
                )

                new_template = summarizer_result.get("new_template", "").strip()
                if not new_template:
                    new_template = template

                new_schema = summarizer_result.get("new_pattern", {})
                if not new_schema or not new_schema.get("structure_type"):
                    new_schema = extractor.extract(new_template)

                # Ensure new patterns carry their goal_type into the schema, so
                # that when preprocess_stage_a_artifacts later rebuilds
                # pattern_set from the persisted registry, the pattern is not
                # orphaned. Without this, summarizer/extractor outputs typically
                # lack a goal_type field and end up in no bucket.
                existing_gt = new_schema.get("goal_type", [])
                if isinstance(existing_gt, str):
                    existing_gt = [existing_gt]
                if goal_type and goal_type not in existing_gt:
                    existing_gt.append(goal_type)
                new_schema["goal_type"] = existing_gt

                new_pattern_id = stable_hash(new_schema)

                is_new = new_pattern_id not in registry.patterns
                if is_new:
                    new_pattern_id = registry.add_pattern(
                        schema=new_schema,
                        template=new_template,
                        source="expansion",
                    )

                if new_pattern_id is not None and is_new:
                    if goal_type not in pattern_set:
                        pattern_set[goal_type] = []

                    if new_pattern_id not in pattern_set[goal_type]:
                        pattern_set[goal_type].append(new_pattern_id)

                    if new_pattern_id not in theta:
                        theta[new_pattern_id] = PatternStats()

                    _base = list(pattern_set[goal_type])
                    if min_candidates and len(_base) < min_candidates:
                        _extras = [pid for pid in registry.ids() if pid not in _base]
                        candidate_ids = _base + _extras
                    else:
                        candidate_ids = _base

                    # Persist growing registry + pattern_set across runs.
                    try:
                        os.makedirs(os.path.dirname(shared_registry_path) or ".", exist_ok=True)
                        save_json(shared_registry_path, registry.to_json())
                        save_json(shared_pattern_set_path, pattern_set)
                    except Exception as _e:
                        print(f"[expansion] warn: failed to persist shared registry: {_e}")


                output = run_victim_online(
                    victim_model=victim_model,
                    victim_tokenizer=victim_tokenizer,
                    prompt=prompt,
                    template=new_template,
                    victim_config_path=victim_config_path,
                )

                results = scorer.score(goal, prompt, new_template, output)
                new_reward, label, evidence = parse_score_result(results)

                new_reward = float(new_reward)
                update_stats(theta, new_pattern_id, new_reward)

                record.new_pattern_id = new_pattern_id
                record.new_pattern_schema = new_schema
                record.new_template = new_template
                record.new_prompt = prompt
                record.new_response = output
                record.new_victim_output = output
                record.new_reward = new_reward
                record.improvement_written = True

            records.append(record)

            with open(records_path, "a", encoding="utf-8") as _rf:
                _rf.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

            prev_selected_pattern_id = selected_pattern_id
            prev_selected_pattern_schema = selected_pattern.schema
            prev_template = template
            prev_reward = reward

            best_reward = max(reward, record.new_reward if record.improvement_written else reward)
            if best_reward >= success_threshold:
                print(f"[EarlyStop] goal={goal_id} iter={t} reward={best_reward:.4f} >= {success_threshold}, stopping exploration.")
                break

            if verbose:
                print(
                    f"[Expansion] goal={goal_id} "
                    f"type={goal_type} "
                    f"iter={t} "
                    f"pid={selected_pattern_id} "
                    f"reward={reward:.4f}"
                )
                if record.improvement_written:
                    print(
                        f"  -> new_pattern={record.new_pattern_id} "
                        f"new_reward={record.new_reward:.4f}"
                    )

    theta_json = stats_to_json(theta)

    save_json(os.path.join(log_dir, "config.json"), cfg)
    save_json(os.path.join(log_dir, "pattern_registry.json"), registry.to_json())
    save_json(os.path.join(log_dir, "pattern_set.json"), pattern_set)
    save_json(os.path.join(log_dir, "theta.json"), theta_json)
    if runtime_cfg.get("save_shared_theta", True):
        try:
            os.makedirs(os.path.dirname(shared_theta_path) or ".", exist_ok=True)
            save_json(shared_theta_path, theta_json)
            shared_theta_manifest.setdefault("record_files", {})
            if os.path.exists(records_path):
                shared_theta_manifest["record_files"][records_abs_path] = count_jsonl_rows(records_path)
            save_json(shared_theta_manifest_path, shared_theta_manifest)
        except Exception as _e:
            print(f"[expansion] warn: failed to persist shared theta: {_e}")
    # NOTE: skip the "w"-mode final save; incremental append inside the loop
    # is the authoritative records file. Rewriting with only this session's
    # in-memory `records` would clobber earlier resume-state records.

    print("Stage expansion finished")
    print(f"Total records: {len(records)}")
    print(f"Total patterns: {len(registry.patterns)}")
    print(f"Total tracked arms: {len(theta)}")


def run_stage(cfg: Dict[str, Any]) -> None:
    stage = cfg.get("stage", "expansion")

    if stage == "expansion":
        run_expansion(cfg)
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> None:
    cli_args = parse_kv_args(sys.argv)

    if "config" not in cli_args:
        raise ValueError("Missing required argument: config=path/to/config.yaml")

    cfg = load_yaml_config(cli_args["config"])
    run_stage(cfg)


if __name__ == "__main__":
    main()
