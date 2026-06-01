#!/bin/bash
# Per-GPU baseline runner: takes GPU id and a list of "method/victim" jobs.
# Usage: ./scripts/_run_baseline_per_gpu.sh <GPU> "method1/victim1 method2/victim2 ..."
set -u
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm

GPU="$1"
JOBS="$2"
LOG_FILE="logs/transition_baselines/gpu${GPU}_chain.log"
mkdir -p logs/transition_baselines results/transition/baselines

for job in $JOBS; do
  m="${job%%/*}"
  v="${job##*/}"
  OUT="results/transition/baselines/${m}_advbench_${v}.json"
  if [[ -f "$OUT" ]] && [[ $(wc -l < "$OUT") -ge 100 ]]; then
    echo "[$(date +%H:%M:%S)] GPU$GPU skip $m/$v (already 100/100)" | tee -a "$LOG_FILE"
    continue
  fi
  echo "[$(date +%H:%M:%S)] GPU$GPU start $m/$v" | tee -a "$LOG_FILE"
  CUDA_VISIBLE_DEVICES=$GPU python -u baselines/transition_baselines.py \
      --victim "$v" --method "$m" \
      --data data/advbench.json --max-goals 100 \
      --out "$OUT" \
      > "logs/transition_baselines/gpu${GPU}_${m}_${v}.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU$GPU done  $m/$v ($(wc -l < $OUT)/100)" | tee -a "$LOG_FILE"
done
echo "[$(date +%H:%M:%S)] GPU$GPU CHAIN DONE" | tee -a "$LOG_FILE"
