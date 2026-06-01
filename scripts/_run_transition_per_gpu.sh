#!/bin/bash
# Per-GPU retrieve runner. Takes GPU id and a list of "victim[:start:end]" jobs.
# Example: ./scripts/_run_transition_per_gpu.sh 0 "llada2:0:50 dream llada"
set -u
cd /weka/home/ext-yingzima/Attack4dLLM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dllm

GPU="$1"
JOBS="$2"
LOG="logs/transition_v2/gpu${GPU}_chain.log"
mkdir -p logs/transition_v2 results/transition_v2

ts(){ date +%H:%M:%S; }

for job in $JOBS; do
  IFS=':' read -r victim start end <<< "$job"
  start=${start:-0}
  end=${end:-100}
  if [[ "$start" -eq 0 && "$end" -eq 100 ]]; then
    OUT="results/transition_v2/advbench_${victim}.json"
  else
    OUT="results/transition_v2/advbench_${victim}_${start}_${end}.json"
  fi
  echo "[$(ts)] GPU$GPU start $victim [$start,$end) -> $OUT" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU python -u baselines/transition.py \
      --victim "$victim" \
      --max-goals 100 --k 5 --seed 0 \
      --start "$start" --end "$end" \
      --out "$OUT" \
      > "logs/transition_v2/gpu${GPU}_${victim}_${start}_${end}.log" 2>&1
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  echo "[$(ts)] GPU$GPU done  $victim [$start,$end) ($n records)" | tee -a "$LOG"
done
echo "[$(ts)] GPU$GPU CHAIN DONE" | tee -a "$LOG"
