import os
import sys
import json
import yaml
import math
import torch
import hashlib
from tqdm import tqdm
from omegaconf import OmegaConf
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from llm import HuggingFaceModel, BedrockModel
from framework import Attacker, PatternExtractor, PatternMerger, Summarizer, Scorer, Retriever
from transformers import AutoTokenizer
from utils import get_config, preprocess_stage_a_artifacts, get_goal_type, parse_score_result



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


class PatternRegistry:
    def __init__(self) -> None:
        self.patterns: Dict[str, Pattern] = {}

    def add_pattern(
        self,
        schema: Dict[str, Any],
        template: Optional[str] = None,
        source: str = "discovery",
    ) -> str:
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


def select_pattern_ucb(
    candidate_ids: List[str],
    theta: Dict[str, PatternStats],
    step_t: int,
    alpha: float,
) -> str:
    """
    UCB:
        mu_i + alpha * sqrt(2 ln t / n_i)

    Unvisited arms are selected first.
    """
    for pid in candidate_ids:
        if pid not in theta or theta[pid].visit_count == 0:
            return pid

    best_pid = None
    best_score = -1e18

    for pid in candidate_ids:
        mu = theta[pid].mean_reward
        n = theta[pid].visit_count
        ucb = mu + alpha * math.sqrt(2.0 * math.log(max(step_t, 2)) / n)
        if ucb > best_score:
            best_score = ucb
            best_pid = pid

    if best_pid is None:
        raise ValueError("No valid pattern selected by UCB.")
    return best_pid


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


def load_victim(victim_config_path: str):
    config = get_config(victim_config_path)

    victim_model = None
    victim_tokenizer = None

    pretrained_model = config.model
    rank = 0
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if "llada2" in victim_config_path:
        from sample.llada2.modeling_llada2_moe import LLaDA2MoeModelLM
        rank = 0

        victim_model = (
            LLaDA2MoeModelLM.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(device)
            .eval()
        )

        victim_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
            use_fast=False,
        )

    elif "dream" in victim_config_path:
        from sample.dream import DreamTokenizer
        from sample.dream.modeling_dream import DreamModel
        from sample.dream.generation_utils_block import DreamGenerationMixin
        from sample.dream.generation_utils_block import DreamGenerationConfig
        import types

        victim_model = (DreamModel.from_pretrained(pretrained_model,
                                                trust_remote_code=True,
                                                torch_dtype=torch.bfloat16)
                     .to(device)
                     .eval())
        victim_model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, victim_model)
        victim_model._sample = types.MethodType(DreamGenerationMixin._sample, victim_model)
        victim_tokenizer = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)


    elif "llada" in victim_config_path or "mmada" in victim_config_path:
        from sample.llada.modeling_llada import LLaDAModelLM

        victim_model = (LLaDAModelLM
                     .from_pretrained(pretrained_model,
                                      trust_remote_code=True,
                                      torch_dtype=torch.bfloat16)
                     .to(device)
                     .eval())

        victim_tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)





    return config, victim_model, victim_tokenizer


def expand_mask_tokens(text, mask_token="<mask>"):
    """Convert <mask:N> to N repetitions of the model's native mask token."""
    import re
    def replacer(match):
        n = int(match.group(1))
        return mask_token * n
    return re.sub(r"<mask:(\d+)>", replacer, text)


def run_victim_online(
    victim_model,
    victim_tokenizer,
    prompt: str,
    template: str,
    victim_config_path: str,
):
    config = get_config(victim_config_path)
    rank = 0

    # Select the victim-family-specific native mask token (matches inference.py).
    # DREAM and LLaDA2 use <|mask|>; LLaDA, LLaDA1.5, MMADA use <|mdm_mask|>.
    if "dream" in victim_config_path or "llada2" in victim_config_path:
        native_mask = "<|mask|>"
    else:
        native_mask = "<|mdm_mask|>"

    # Expand <mask:N> directly to N copies of native mask token
    # Also handle plain <mask> (convert to 20 copies of native mask)
    template = expand_mask_tokens(template, native_mask)
    template = template.replace("<mask>", native_mask * 20)

    if "llada2" in victim_config_path:
        from sample.llada2_inference import LLaDA2_Inference, get_new_prompt
        rank = 0
        system_prompts = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{{problem}} <|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        wrapped_prompt = get_new_prompt(system_prompts, prompt)
        wrapped_template = template  # already expanded above

        print("="*50)
        print(wrapped_template)

        seq_dict = {}
        step_dict = {}


        outputs = LLaDA2_Inference(
            model_gpu=victim_model,
            tokenizer_gpu=victim_tokenizer,
            rank=rank,
            prompts=wrapped_prompt,
            templates=wrapped_template,
            orig_idx=None,
            seq_dict=seq_dict,
            step_dict=step_dict,
            batch_size=1,
            config=config,
        )

        if not outputs:
            raise RuntimeError("Victim model returned empty outputs.")

        return outputs[0]

    elif "dream" in victim_config_path:
        from sample.dream_inference import Dream_Inference, get_new_prompt
        system_prompts = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{{problem}} <|im_end|>\n<|im_start|>assistant\n'''

        wrapped_prompt = get_new_prompt(system_prompts, prompt)
        wrapped_template = template  # already expanded above

        seq_dict = {}
        step_dict = {}

        outputs, _ = Dream_Inference(
            model_gpu=victim_model,
            tokenizer_gpu=victim_tokenizer,
            rank=rank,
            prompts=wrapped_prompt,
            templates=wrapped_template,
            orig_idx=None,
            seq_dict=seq_dict,
            step_dict=step_dict,
            batch_size=1,
            config=config,
        )

        return outputs[0]


    elif "llada" in victim_config_path or "mmada" in victim_config_path:
        from sample.llada_inference import LLaDA_Inference, get_new_prompt
        system_prompts = '''<|startoftext|><|start_header_id|>user<|end_header_id|>{{problem}}<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n'''

        wrapped_prompt = get_new_prompt(system_prompts, prompt)
        wrapped_template = template  # already expanded above

        outputs, _ = LLaDA_Inference(
            model_gpu=victim_model,
            tokenizer_gpu=victim_tokenizer,
            rank=rank,
            prompts=wrapped_prompt,
            templates=wrapped_template,
            batch_size=1,
            config=config,
        )

        return outputs[0]


    raise NotImplementedError(f"Unsupported victim config: {victim_config_path}")


def run_expansion(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg["models"]
    data_cfg = cfg["data"]
    runtime_cfg = cfg["runtime"]

    log_dir = runtime_cfg["log_dir"]
    ensure_dir(log_dir)

    with open(data_cfg["data_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    seed_size = data_cfg.get("seed_size", -1)
    if seed_size > 0:
        data = data[seed_size:]

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

    detection_model_path = model_cfg.get("detection_model_path", None)

    # Non-victim roles: attack_model (vLLM heretic) for generation tasks,
    # scorer uses a separate HF model (Qwen3-4B-Instruct) for accurate evaluation.
    # Scorer uses the same instruct model (attack_model)
    scorer_model = attack_model

    extractor = PatternExtractor(attack_model)
    merger = PatternMerger(attack_model)
    summarizer = Summarizer(attack_model)
    scorer = Scorer(scorer_model)
    retriever = Retriever(attack_model)


    registry, pattern_set = load_stage_a_artifacts(
        init_registry_path=data_cfg["init_pattern_registry_path"],
        init_pattern_set_path=data_cfg.get("init_pattern_set_path", None),
    )

    registry, pattern_set, data, goal_type_mapping = preprocess_stage_a_artifacts(
        data=data,
        registry=registry,
        pattern_set=pattern_set,
        retriever=retriever,
        verbose=True,
    )

    theta = initialize_stats(registry)
    records: List[ExpansionRecord] = []

    max_iters = runtime_cfg.get("max_iters", 5)
    max_length = runtime_cfg.get("max_length", 512)
    alpha = runtime_cfg.get("alpha", 1.0)
    verbose = runtime_cfg.get("verbose", False)
    fallback_to_all_patterns = runtime_cfg.get("fallback_to_all_patterns", True)

    for idx, example in tqdm(
        enumerate(data),
        total=len(data),
        desc="Stage B Expansion",
    ):
        if "question" not in example:
            continue

        goal = example["question"]
        goal_id = example.get("id", f"goal_{idx}")
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

        for t in range(1, max_iters + 1):
            selected_pattern_id = select_pattern_ucb(
                candidate_ids=candidate_ids,
                theta=theta,
                step_t=t,
                alpha=alpha,
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
                t >= 2
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

                new_pattern_id = stable_hash(new_schema)

                is_new = new_pattern_id not in registry.patterns
                if is_new:
                    new_pattern_id = registry.add_pattern(
                        schema=new_schema,
                        template=new_template,
                        source="expansion",
                    )

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

            with open(os.path.join(log_dir, "expansion_records.jsonl"), "a", encoding="utf-8") as _rf:
                _rf.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

            prev_selected_pattern_id = selected_pattern_id
            prev_selected_pattern_schema = selected_pattern.schema
            prev_template = template
            prev_reward = reward

            success_threshold = runtime_cfg.get("success_threshold", 1.0)
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

    theta_json = {
        pid: {
            "mean_reward": stat.mean_reward,
            "visit_count": stat.visit_count,
        }
        for pid, stat in theta.items()
    }

    save_json(os.path.join(log_dir, "config.json"), cfg)
    save_json(os.path.join(log_dir, "pattern_registry.json"), registry.to_json())
    save_json(os.path.join(log_dir, "pattern_set.json"), pattern_set)
    save_json(os.path.join(log_dir, "theta.json"), theta_json)
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