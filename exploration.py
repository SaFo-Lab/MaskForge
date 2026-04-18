import os
import sys
import json
import yaml
import hashlib
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

from llm import HuggingFaceModel, BedrockModel
from framework import Attacker, PatternExtractor, PatternMerger


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


def parse_kv_args(argv: List[str]) -> Dict[str, str]:
    """
    Parse command line args in key=value format.
    Example:
        python stage_a.py config=../configs/stage_a.yaml
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


class PatternRegistry:
    def __init__(self) -> None:
        self.patterns: Dict[str, Pattern] = {}

    def add_pattern(
        self,
        schema: Dict[str, Any],
        template: Optional[str] = None,
    ) -> str:
        pattern_id = stable_hash(schema)
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = Pattern(
                pattern_id=pattern_id,
                schema=schema,
                representative_template=template,
                source="discovery",
            )
        return pattern_id

    def ids(self) -> List[str]:
        return list(self.patterns.keys())

    def get(self, pattern_id: str) -> Pattern:
        return self.patterns[pattern_id]


@dataclass
class CandidateRecord:
    goal_id: str
    goal: str
    template: str
    prompt: Optional[str]
    malicious_response: Optional[str]
    raw_template: Optional[str]
    extracted_pattern: Optional[Dict[str, Any]] = None
    merged_pattern_id: Optional[str] = None
    latency: float = 0.0


def load_single_model(model_path: str):
    if "/" in model_path:
        return HuggingFaceModel(model_path)
    return BedrockModel(model_path)


def load_attacker(
    attack_model_path: str,
    weaker_model_path: str,
):
    attack_model = load_single_model(attack_model_path)
    weaker_model = load_single_model(weaker_model_path)
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
):
    if target_model_path == attack_model_path:
        return attack_model
    if target_model_path in shared_models:
        return shared_models[target_model_path]

    model = load_single_model(target_model_path)
    shared_models[target_model_path] = model
    return model


def build_pattern_set(registry: PatternRegistry) -> Dict[str, List[str]]:
    pattern_set = defaultdict(list)
    for pattern_id in registry.ids():
        schema = registry.get(pattern_id).schema
        goal_types = schema.get("goal_type", [])
        for goal_type in goal_types:
            pattern_set[goal_type].append(pattern_id)
    return dict(pattern_set)


def run_exploration(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg["models"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    log_dir = runtime_cfg["log_dir"]
    ensure_dir(log_dir)

    with open(data_cfg["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    seed_size = data_cfg.get("seed_size", -1)
    if seed_size > 0:
        data = data[:seed_size]

    attacker, attack_model = load_attacker(
        attack_model_path=model_cfg["attack_model_path"],
        weaker_model_path=model_cfg["attack_weaker_model_path"],
    )

    shared_models: Dict[str, Any] = {}

    extractor_model = resolve_model(
        target_model_path=model_cfg["extractor_model_path"],
        attack_model_path=model_cfg["attack_model_path"],
        attack_model=attack_model,
        shared_models=shared_models,
    )
    merger_model = resolve_model(
        target_model_path=model_cfg["merger_model_path"],
        attack_model_path=model_cfg["attack_model_path"],
        attack_model=attack_model,
        shared_models=shared_models,
    )

    extractor = PatternExtractor(extractor_model)
    merger = PatternMerger(merger_model)

    candidates: List[CandidateRecord] = []
    k = runtime_cfg.get("num_samples_per_goal", 1)
    max_length = runtime_cfg.get("max_length", 1024)
    verbose = runtime_cfg.get("verbose", False)

    for idx, example in tqdm(
        enumerate(data),
        total=len(data),
        desc="Generating candidates",
    ):
        if "question" not in example:
            continue

        goal = example["question"]
        goal_id = example.get("id", f"goal_{idx}")

        for _ in range(k):
            prompt, template, malicious_response, raw_template = attacker.instantiate(
                goal,
                pattern=None,
                max_length=max_length,
            )
            schema = extractor.extract(template)

            if verbose:
                print(schema)

            candidates.append(
                CandidateRecord(
                    goal_id=goal_id,
                    goal=goal,
                    template=template,
                    prompt=prompt,
                    malicious_response=malicious_response,
                    raw_template=raw_template,
                    extracted_pattern=schema,
                )
            )

        if verbose:
            print(f"[Discovery] goal {idx + 1}/{len(data)}")

    raw_schemas = []
    schema_to_template = {}
    for candidate in candidates:
        if candidate.extracted_pattern is not None:
            raw_schemas.append(candidate.extracted_pattern)
            schema_key = stable_hash(candidate.extracted_pattern)
            if schema_key not in schema_to_template:
                schema_to_template[schema_key] = candidate.template

    if raw_schemas:
        merged_schemas = merger.merge(raw_schemas)
        if verbose:
            print(f"[Discovery] Merged {len(raw_schemas)} raw schemas into {len(merged_schemas)} canonical patterns.")
    else:
        merged_schemas = []

    registry = PatternRegistry()
    for schema in merged_schemas:
        schema_key = stable_hash(schema)
        template = schema_to_template.get(schema_key, None)
        registry.add_pattern(schema=schema, template=template)

    # also add any raw schemas that weren't covered by merging
    for candidate in candidates:
        if candidate.extracted_pattern is not None:
            pid = registry.add_pattern(
                schema=candidate.extracted_pattern,
                template=candidate.template,
            )
            candidate.merged_pattern_id = pid

    pattern_set = build_pattern_set(registry)

    save_json(os.path.join(log_dir, "config.json"), cfg)
    save_json(os.path.join(log_dir, "pattern_set.json"), pattern_set)
    save_jsonl(
        os.path.join(log_dir, "candidates.jsonl"),
        [asdict(candidate) for candidate in candidates],
    )
    save_json(
        os.path.join(log_dir, "pattern_registry.json"),
        {
            pattern_id: asdict(pattern)
            for pattern_id, pattern in registry.patterns.items()
        },
    )

    print("Stage exploration finished")
    print(f"Total candidates: {len(candidates)}")
    print(f"Total patterns: {len(registry.patterns)}")


def run_stage(cfg: Dict[str, Any]) -> None:
    stage = cfg.get("stage", "exploration")

    if stage == "exploration":
        run_exploration(cfg)
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