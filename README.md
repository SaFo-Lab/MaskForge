# Attack4dLLMs

Search-based red-team attack on diffusion large language models (dLLMs).
Given an unsafe goal, the attacker generates a `<mask:N>` *template* that, when
filled in parallel by the victim dLLM, produces harmful content. Unlike
fixed-template baselines (DIJA, PAD), we explore a pattern library with a UCB
bandit and extend it on the fly using scorer-guided refinement.

Evaluated on five dLLM victims (LLaDA, LLaDA-1.5, LLaDA2.0-mini, Dream-Instruct,
MMaDA-CoT) across three red-team benchmarks (JailbreakBench, StrongReject,
HarmBench). Judge models: local Qwen3-4B-Instruct via vLLM or Qwen3-235B via
AWS Bedrock.

## Headline results (ASR-E = LLM-judge, Qwen3-235B, average over 5 victims)

| Attack   | JailbreakBench | StrongReject | HarmBench |
|----------|---------------:|-------------:|----------:|
| DIRECT   |           12.0 |         15.8 |      28.3 |
| PAD      |           58.2 |         72.6 |      52.4 |
| DIJA     |           64.0 |         73.7 |      64.7 |
| PAIR     |            0.2 |          4.1 |       0.5 |
| ReNeLLM  |            3.6 |         10.2 |       4.3 |
| **Ours** |     **89.0** |     **91.3** |  **75.3** |

Per-victim numbers live in `final_results/asr_summary.json`.

## Repository layout

```
Attack4dLLMs/
├── main.py, pipeline.py             # top-level orchestration
├── expansion.py                     # Stage-B: UCB over patterns, records attacks
├── exploration.py                   # Stage-A: seed pattern library discovery
├── inference.py                     # victim inference entry (DIJA / PAD / direct)
├── evaluation.py                    # HarmBench-style LLM judge (Bedrock)
├── utils.py
│
├── framework/                       # Attacker, Scorer, PatternExtractor, Retriever
├── llm/                             # HuggingFace / vLLM / Bedrock client wrappers
├── sample/                          # dLLM-specific inference (llada, llada2, dream, mmada)
├── configs/                         # per-victim YAML (e.g. llada2_attack.yaml)
├── data/                            # benchmark goal files (JBB / SR / HB)
│
├── run_jbb_victim.py                # attack one victim on JailbreakBench (100 goals)
├── run_sr_victim.py                 # attack on StrongReject (313 goals)
├── run_hb_victim.py                 # attack on HarmBench (400 goals)
│
├── convert_records_to_results.py    # expansion_records.jsonl → result JSON
├── eval.sh                          # ASR-LLM via local 4B or Bedrock 235B
├── eval_final.py                    # batch ASR-K + ASR-LLM over final_results/
├── asr_llm_4b.py                    # local Qwen3-4B judge (vLLM on :8100)
│
├── final_results/                   # 5 victims × 6 attacks × 3 benchmarks = 90 files
│                                    #   + asr_summary.json (ASR-K, ASR-E per file)
├── tables/                          # LaTeX tables for paper
└── logs/                            # runtime logs (created by runners)
```

Each `final_results/<benchmark>_<victim>_<attack>.json` is a list of
`{"goal": ..., "template": ..., "prediction": ...}` records. `ours` rows
additionally carry `reward`, `iteration`, `goal_id`.

## Running attacks

### 1. Start vLLM (attacker + weaker model)

One 4-GPU node. GPU 0 hosts both vLLM servers; GPUs 1-3 run victims.

```bash
conda activate dllm

# attacker model
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Instruct-2507 --port 8100 --trust-remote-code \
  --max-model-len 4096 --gpu-memory-utilization 0.45 &

# weaker (base) model used in fallback
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B-Base --port 8101 --trust-remote-code \
  --max-model-len 4096 --gpu-memory-utilization 0.45 &
```

### 2. Attack a victim

```bash
# JailbreakBench (100 goals)
CUDA_VISIBLE_DEVICES=1 python -u run_jbb_victim.py llada1_5

# StrongReject (313 goals)
CUDA_VISIBLE_DEVICES=2 python -u run_sr_victim.py dream

# HarmBench (up to 400 goals)
CUDA_VISIBLE_DEVICES=3 python -u run_hb_victim.py mmada
```

Valid victim names: `llada`, `llada1_5`, `llada2`, `dream`, `mmada`
(must have a matching `configs/<name>_attack.yaml`).

Each runner:
* writes records incrementally to `logs/{jbb,sr,hb}_<victim>/expansion_records.jsonl`;
* **resumes** automatically on restart — goals with an existing record are skipped;
* uses `max_iters=3`, `max_fallback_retries=3`, `success_threshold=0.9` by
  default (UCB stops on the first iteration that clears the threshold).

### 3. Convert records to result JSON

```bash
python convert_records_to_results.py       # (edit paths inside if needed)
```

Record schema (`expansion_records.jsonl`):

```json
{
  "goal_id": "jbb_0", "goal": "...", "goal_type": "disinformation",
  "iteration": 1, "selected_pattern_id": "96de351a...",
  "template": "Goal line.\nStep 1: <mask:20>. Hint.\n...",
  "victim_output": "Sure, here is ...",
  "reward": 1.0,
  ...
}
```

`template` is always the exact template that produced `victim_output` (the
fallback winner if any), so pairs stay consistent.

## Evaluating attacks

### Local judge (Qwen3-4B via vLLM)

```bash
./eval.sh local ./final_results/jailbreak_dream_ours.json=jailbreak
```

### Bedrock 235B judge

```bash
AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2 \
./eval.sh bedrock ./final_results/strongreject_llada_ours.json=strongreject
```

### Batch ASR-K + ASR-LLM over all 90 files

```bash
AWS_BEARER_TOKEN_BEDROCK=... AWS_REGION=us-west-2 \
python eval_final.py
# → final_results/asr_summary.json
```

Results are cached; re-running skips entries that already have `asr_llm`.

Benchmark labels recognised by judges:
`harmbench`, `jailbreak`/`jailbreakbench`, `strongreject`.

## Method

For each goal the attacker maintains a pattern library. On every iteration it

1. picks a pattern with UCB over empirical reward and visit count,
2. instantiates the pattern as a `<mask:N>` template **starting with the goal
   sentence verbatim**, with short structural hints (“Title:”, “Step 1:”,
   “Claim 1:”) between masks,
3. sends the template to the victim dLLM, which fills the masks in parallel,
4. scores the victim output with a red-team classifier;
5. if reward < threshold, falls back to a weaker-model-driven rewrite (goal
   rephrasing + `[UNSAFE]` tagging → masked template), re-tries, and feeds the
   scorer’s analysis back to the attacker;
6. on success the filled template’s scaffold is distilled and added to the
   pattern library (summariser + pattern extractor in `framework/`).

Implementation details: `expansion.py` (main loop, records), `framework/attacker.py`
(template generation + fallback), `framework/scorer.py` (HarmBench-style classifier
prompt), `sample/*.py` (per-victim mask-fill inference; LLaDA / LLaDA-1.5 /
MMaDA use `<|mdm_mask|>`, Dream / LLaDA2 use `<|mask|>`).

## Dependencies

Python 3.10, conda env `dllm`:

```
vllm transformers accelerate
boto3                # Bedrock judge
requests             # local judge
numpy pandas tqdm
omegaconf jinja2 termcolor
```

Benchmark data files go in `data/`:

* `jailbreakbench_data_refined_Qwen.json` (100 goals)
* `strongreject_data_refined_Qwen.json` (313 goals)
* `harmbench_behaviors_text_all_refined_Qwen.json` (400 behaviours)

Victim checkpoints are fetched at first use through the HuggingFace paths in
`configs/<victim>_attack.yaml` (e.g. `GSAI-ML/LLaDA-1.5`).
