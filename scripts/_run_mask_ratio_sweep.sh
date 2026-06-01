#!/bin/bash
# Mask-ratio ablation: 3 victims × 5 ratios × 100 JBB goals.
# Layout: GPU 0 hosts vLLM (8100/8101) — already running.
#         GPU 1: dream, GPU 2: llada, GPU 3: llada1_5; each runs the 5 ratios serially.
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm
mkdir -p logs/mask_ratio_runs results/mask_ratio

RATIOS=(0.10 0.30 0.50 0.70 0.90)

run_victim_chain() {
  local gpu=$1
  local victim=$2
  local chain_log="logs/mask_ratio_runs/${victim}_chain.log"
  echo "[$(date +%H:%M:%S)] start victim=$victim on GPU $gpu" | tee -a "$chain_log"
  for r in "${RATIOS[@]}"; do
    local tag=$(printf "mr%02d" "$(python -c "print(int($r*100))")")
    local rec_dir="logs/jbb_${victim}_${tag}"
    local run_log="logs/mask_ratio_runs/${victim}_${tag}_run.log"
    echo "[$(date +%H:%M:%S)]   r=$r tag=$tag" | tee -a "$chain_log"
    CUDA_VISIBLE_DEVICES=$gpu python -u run_victim.py \
      --bench jbb --victim "$victim" --suffix "_${tag}" \
      --max-goals 100 --mask-ratio "$r" \
      > "$run_log" 2>&1
    if [ ! -f "${rec_dir}/expansion_records.jsonl" ] || \
       [ "$(wc -l < "${rec_dir}/expansion_records.jsonl")" -lt 50 ]; then
      echo "[$(date +%H:%M:%S)]   WARN: ${victim} ${tag} produced too few records; aborting chain" \
        | tee -a "$chain_log"
      exit 1
    fi
    python aggregate_records.py \
      --records "${rec_dir}/expansion_records.jsonl" \
      --out "results/mask_ratio/jbb_${victim}_${tag}.json" \
      --bench jailbreak >> "$chain_log" 2>&1
  done
  echo "[$(date +%H:%M:%S)] done victim=$victim" | tee -a "$chain_log"
}

run_victim_chain "$1" "$2"
