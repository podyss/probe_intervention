#!/usr/bin/env bash
set -euo pipefail

R1_MODEL=${R1_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
JUDGE_MODEL=${JUDGE_MODEL:-Qwen/Qwen3.8-27B}
INPUT=${INPUT:-data/deepmath-103k/train.jsonl}
OUT=${OUT:-outputs/r1-1.5b-qwen3.8-27b-correct-incorrect-pairs}
PAIRS=${PAIRS:-400}
COLLECT_LIMIT=${COLLECT_LIMIT:-2000}
SEED=${SEED:-42}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
R1_TENSOR_PARALLEL_SIZE=${R1_TENSOR_PARALLEL_SIZE:-1}
R1_DATA_PARALLEL_SIZE=${R1_DATA_PARALLEL_SIZE:-2}
JUDGE_TENSOR_PARALLEL_SIZE=${JUDGE_TENSOR_PARALLEL_SIZE:-1}
JUDGE_DATA_PARALLEL_SIZE=${JUDGE_DATA_PARALLEL_SIZE:-2}
R1_MAX_MODEL_LEN=${R1_MAX_MODEL_LEN:-32768}
JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-32768}
JUDGE_MAX_NUM_SEQS=${JUDGE_MAX_NUM_SEQS:-128}
MAX_TRAJECTORY_TOKENS=${MAX_TRAJECTORY_TOKENS:-16384}
R1_TEMPERATURE=${R1_TEMPERATURE:-1}

TRAJECTORIES="$OUT/r1-trajectories.jsonl"
DATASET="$OUT/correct-incorrect-pairs.jsonl"

if [[ ! -s "$INPUT" ]]; then
  echo "Missing or empty input dataset: $INPUT" >&2
  exit 1
fi
if [[ "$PAIRS" -le 0 || "$COLLECT_LIMIT" -le 0 ]]; then
  echo "PAIRS and COLLECT_LIMIT must be positive" >&2
  exit 1
fi
if [[ "$R1_TENSOR_PARALLEL_SIZE" -le 0 || "$R1_DATA_PARALLEL_SIZE" -le 0 \
   || "$JUDGE_TENSOR_PARALLEL_SIZE" -le 0 || "$JUDGE_DATA_PARALLEL_SIZE" -le 0 ]]; then
  echo "Tensor and data parallel sizes must be positive" >&2
  exit 1
fi
IFS=',' read -r -a VISIBLE_GPU_LIST <<< "$CUDA_DEVICES"
VISIBLE_GPU_COUNT=${#VISIBLE_GPU_LIST[@]}
if (( R1_TENSOR_PARALLEL_SIZE * R1_DATA_PARALLEL_SIZE > VISIBLE_GPU_COUNT \
   || JUDGE_TENSOR_PARALLEL_SIZE * JUDGE_DATA_PARALLEL_SIZE > VISIBLE_GPU_COUNT )); then
  echo "The selected TP x DP sizes exceed the visible GPU count ($VISIBLE_GPU_COUNT)" >&2
  exit 1
fi
mkdir -p "$OUT"
for path in "$TRAJECTORIES" "$TRAJECTORIES.manifest.json" "$DATASET"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing output: $path (choose a new OUT)" >&2
    exit 1
  fi
done

# Sample complete attempts with R1; the judge starts only after collection exits.
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run python -m probe_int.pipeline collect \
  --model "$R1_MODEL" --input "$INPUT" --limit "$COLLECT_LIMIT" --repeat-input \
  --output "$TRAJECTORIES" --tensor-parallel-size "$R1_TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$R1_DATA_PARALLEL_SIZE" \
  --boundary step --skip-probes --temperature "$R1_TEMPERATURE" \
  --max-tokens "$MAX_TRAJECTORY_TOKENS" \
  --max-model-len "$R1_MAX_MODEL_LEN" --dtype bfloat16 --seed "$SEED"

# Qwen finds each trajectory's first unresolved error; export one clean prefix
# and one prefix containing that error for every selected source trajectory.
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run python -m probe_int.label_trajectory_steps \
  --input "$TRAJECTORIES" --output "$DATASET" --model "$JUDGE_MODEL" \
  --limit "$((PAIRS * 2))" --max-model-len "$JUDGE_MAX_MODEL_LEN" \
  --max-num-seqs "$JUDGE_MAX_NUM_SEQS" \
  --tensor-parallel-size "$JUDGE_TENSOR_PARALLEL_SIZE" \
  --data-parallel-size "$JUDGE_DATA_PARALLEL_SIZE" \
  --gpu-memory-utilization 0.8 --seed "$SEED"

echo "Paired Correct/Incorrect dataset written to $DATASET"
