# MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models 

[**Paper (arXiv)**](https://arxiv.org/abs/2606.04027)

Search-based red-team attack on diffusion LLMs (dLLMs). The attacker generates
a `<mask:N>` template; the victim dLLM fills the masks in parallel and emits
harmful content. Evaluated on LLaDA, LLaDA-1.5, LLaDA2.0-mini, Dream-Instruct,
and MMaDA-CoT across HarmBench, JailbreakBench, StrongReject, and AdvBench.

The pipeline has three stages — bootstrap a pattern library (A), grow it via
UCB-guided online attack (B), then transfer it by pure retrieval (C). The
*Quick start* below runs Stage B against one victim using the library checked
into `logs/`.

## Setup

Hardware: ≥ 2 GPUs (one hosts the attacker / scorer LLMs via vLLM, the other
runs the victim dLLM). Each vLLM server fits in ~45% of a 40 GB card with the
default 4 k context.

```bash
# 1. Clone + enter
git clone <repo> Attack4dLLM && cd Attack4dLLM

# 2. Conda env (Python 3.10)
conda create -n dllm python=3.10 -y
conda activate dllm

# 3. Core dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate safetensors einops regex xxhash
pip install numpy pandas tqdm omegaconf pyyaml jinja2 termcolor matplotlib
pip install requests boto3      # local + Bedrock judges

# 4. (optional) Per-victim extras — install only what you plan to run
pip install decord                            # MMaDA visual tokens
pip install flash-attn --no-build-isolation   # speedup on Hopper / Ampere
```

Bedrock judge — set credentials in the environment before any `--judge
bedrock` invocation:

```bash
export AWS_BEARER_TOKEN_BEDROCK=<your-token>
export AWS_REGION=<your-region>
```

Victim model weights are pulled from HuggingFace on first use; the path is
read from `configs/<victim>_attack.yaml`. Set `HF_HOME` if you want the cache
to live somewhere specific. The benchmark data files
(`data/{jailbreakbench,strongreject,harmbench,advbench}*.json`) ship with
the repo.

A copy of the pattern library learned in Stage A + B is checked into
`logs/shared_pattern_registry.json` (≈ 1 300 patterns) and
`logs/shared_pattern_set.json`. Both stages B and C read from these — you
can reproduce Stage B and Stage C without re-running Stage A.

## Static evaluation with `benchmark/`

For a quick static probe you do **not** need to run any of the stages above.
`benchmark/{jailbreak,harmbench,strongreject}.json` ships 100 goals per
benchmark, each with up to 3 high-transferability templates that successfully
broke ≥ 4 of the 5 victims. Each record is

```json
{
  "goal_id": "jbb_0",
  "goal": "...",
  "templates": ["<template with <mask:N> placeholders>", "...", "..."]
}
```

Expand `<mask:N>` to the victim's mask token (`<|mask|>` for Dream / LLaDA2,
`<|mdm_mask|>` for LLaDA / LLaDA-1.5 / MMaDA), submit, and judge — no
attacker LLM, no scorer, no UCB:

```python
import json, re
from inference import load_victim, run_victim_online

MASK_TOKEN = "<|mask|>"   # pick per victim
victim_cfg = "configs/dream_attack.yaml"
_, model, tok = load_victim(victim_cfg)

def expand(tmpl):
    return re.sub(r"<mask:(\d+)>", lambda m: MASK_TOKEN * int(m.group(1)), tmpl)

records = json.load(open("benchmark/jailbreak.json"))
for r in records:
    for tmpl in r["templates"]:
        out = run_victim_online(
            victim_model=model, victim_tokenizer=tok,
            prompt=r["goal"], template=expand(tmpl),
            victim_config_path=victim_cfg,
        )
        # judge `out` with evaluation.py / your own scorer
```

This is the simplest reproduction path for any new victim and the cheapest
way to compare different decoding settings (mask ratio, step count, block
size, …) against a fixed set of known-good attacks.

## Quick start

```bash
conda activate dllm

# 1. Start the attacker + weaker base LLMs (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8100 \
    --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.45 &
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Base --port 8101 \
    --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.45 &

# 2. Attack: Dream on JailbreakBench, 100 goals (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run_victim.py \
    --bench jbb --victim dream --max-goals 100

# 3. Aggregate the per-iteration records into one row per goal
python aggregate_records.py --bench jailbreak --out-dir results/main

# 4. Judge ASR + Harmscore (Bedrock Qwen3-235B)
python evaluation.py --files 'results/main/jailbreak_dream.json' \
    --out results/main/summary.json --metrics asr,hs --judge bedrock
```

Records land in `logs/jbb_dream/expansion_records.jsonl`; re-running resumes
(goals already in the file are skipped). Swap `--bench jbb` for `hb` / `sr`,
and `--victim dream` for one of `{llada, llada1_5, llada2, mmada}` to cover
the other (benchmark, victim) cells.

## Stage A — Bootstrap pattern library

`exploration.py` mines an initial library of mask-fill schemas from a small
seed of harmful goals. Each goal goes through the attacker LLM, which emits a
strategy + jailbreak prompt + `<mask:N>` template; templates that succeed
against the victim are abstracted into schemas via `framework/pattern.py` and
inserted into the registry.

```bash
python exploration.py config=configs/exploration.yaml
```

Output: `logs/shared_pattern_registry.json` (the canonical library) and
`logs/shared_pattern_set.json` (per-goal-type buckets).

You usually do **not** need to re-run this — the persisted library in `logs/`
is what every later stage reads. Re-run only when changing the seed set or
the schema-extraction prompt.

## Stage B — Online expansion + attack

`expansion.py` is the core attack loop, invoked through `run_victim.py`. For
each goal it runs up to `max_iters=3` iterations of:

1. **UCB pattern selection** from the goal-type bucket of the registry.
2. **`attacker.instantiate`** — one LLM call that produces a strategy tag, a
   jailbreak prompt, and a `<mask:N>` template aligned to the chosen schema.
3. **Victim inference** — the dLLM fills the masks in parallel.
4. **Scorer** (Qwen3-4B-Instruct via vLLM) — HarmBench-style classifier,
   reward ∈ [0, 1].
5. **Fallback** (≤ 3 retries) when reward < 0.7: the weaker base model on
   port 8101 produces a raw response → Bedrock Qwen3-235B inserts
   `[UNSAFE]…[/UNSAFE]` tags → tagged spans become `<mask:N>` → re-attack.
6. **Pattern evolution** — when a later iteration improves on the previous,
   the summarizer distills a new schema + template and writes both to
   `logs/shared_pattern_registry.json`.

Stage B is what the *Quick start* drives. Suffix tags (`--suffix _foo`)
isolate per-experiment registries; without a suffix everything updates the
shared library.

## Stage C — Transferability

`baselines/transition.py` reuses the persisted library as a frozen black box:
no attacker LLM at attack time, the victim is the only LLM in the loop.

For each AdvBench goal:

1. Look up the goal-type bucket.
2. Sample `k` patterns from the bucket (or use embedding cosine, via
   `baselines/transition_embed.py`).
3. Swap line 1 of each `representative_template` for the new goal, expand
   `<mask:N>` to the victim's mask token, submit to the victim.
4. Best-of-`k` ASR — the goal counts as jailbroken if any of the `k` attempts
   is judged unsafe.

```bash
# Refresh goal-type buckets if the registry changed (needs vLLM 8100 + Bedrock)
python scripts/_expand_taxonomy_bedrock.py

# Per-GPU runners
./scripts/_run_transition_per_gpu.sh 0 "dream llada llada1_5 mmada llada2"
./scripts/_run_baseline_per_gpu.sh   0 "direct/dream pad/dream dija/dream ..."

# Judge
python scripts/_judge_transition.py \
    --files 'results/transition_v2/advbench_*.json' \
    --out results/transition_v2/bedrock_asr_summary.json --workers 16
python scripts/_judge_hs.py --transition \
    --files 'results/transition_v2/advbench_*.json' \
    --out results/transition_v2/bedrock_hs_summary.json --workers 16
```

Outputs land in `results/transition_v2/advbench_{victim}.json`. The
`baselines/` directory also contains `transition_baselines.py` (direct / pad
/ dija), `transition_embed.py` (BGE cosine retrieval), and
`transition_random_global.py` (uniform-random over the whole registry).


## Repository layout

```
exploration.py            Stage A — bootstrap pattern library
expansion.py              Stage B core loop (UCB + fallback + evolution)
run_victim.py             Stage B CLI entry — main + defense ablations
inference.py              Victim model load + run
aggregate_records.py      expansion_records.jsonl  ->  per-goal result JSON
evaluation.py             ASR-LLM + Harmscore judging (Bedrock / local)
utils.py                  Config / prompt / parsing helpers

baselines/                Stage C transferability
  transition.py             best-of-k retrieval from the registry
  transition_baselines.py   direct / pad / dija
  transition_embed.py       BGE cosine retrieval
  transition_random_global.py

framework/                Attacker, Scorer, PatternRegistry, Summarizer, Retriever
llm/                      HuggingFace, vLLM, Bedrock backends
sample/                   Per-victim dLLM mask-fill (LLaDA, LLaDA2, Dream, MMaDA)
configs/                  YAML configs (one per victim, plus expansion / exploration)
data/                     JBB, HB, StrongReject, AdvBench goal files
scripts/                  Per-GPU runners, judges, taxonomy / embedding helpers
benchmark/                Curated (goal, template) pairs for static evaluation
```

## Citation

If you use this code or the released pattern library / benchmark in your
research, please cite:

```bibtex
@misc{ma2026maskforgestructureawareadaptiveattacks,
      title={MaskForge: Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models}, 
      author={Yingzi Ma and Zhengyue Zhao and Xiaogeng Liu and Minhui Xue and Yue Zhao and Chaowei Xiao},
      year={2026},
      eprint={2606.04027},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2606.04027}, 
}
```
