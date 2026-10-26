#!/usr/bin/env bash
set -euo pipefail

# Fresh DeepSeek-R1-Distill-Qwen-1.5B collection, three SFT methods, and evaluation.
MODEL=${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
SEED=${SEED:-42}
EVAL_SEED=${EVAL_SEED:-42}
OUT=${OUT:-outputs/deepmath-103k-r1-1.5b-seed${SEED}}
COUNT=${COUNT:-200}
COLLECT_LIMIT=${COLLECT_LIMIT:-1000}
CUDA_DEVICES=${CUDA_DEVICES:-0,1}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}

TRAIN_INPUT=${TRAIN_INPUT:-data/deepmath-103k/train.jsonl}
EVAL_INPUT=${EVAL_INPUT:-data/deepmath-103k/test.jsonl}
PROBES="$OUT/probes-${COLLECT_LIMIT}-judge.jsonl"
JUDGE_DATA="$OUT/sft-wrong-to-correct-judge05-${COUNT}.jsonl"
STABLE_DATA="$OUT/sft-wrong-to-correct-stablewc-${COUNT}.jsonl"
CORRECT_DATA="$OUT/sft-correct-trajectory-${COUNT}.jsonl"

for input in "$TRAIN_INPUT" "$EVAL_INPUT"; do
  if [[ ! -s "$input" ]]; then
    echo "Missing or empty dataset: $input" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
for path in "$PROBES" "$JUDGE_DATA" "$STABLE_DATA" "$CORRECT_DATA"; do
  if [[ -e "$path" || -e "$path.manifest.json" ]]; then
    echo "Refusing to overwrite existing output: $path (choose a new OUT)" >&2
    exit 1
  fi
done
for method in judge05 stablewc correct; do
  model_dir="$OUT/deepseek-r1-1.5b-sft-${method}-up-to${COUNT}-train-seed${SEED}"
  if [[ -e "$model_dir" ]]; then
    echo "Refusing to overwrite existing model directory: $model_dir (choose a new OUT)" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Collect fresh judge probes from DeepSeek rollouts.
uv run python -m probe_int.pipeline collect \
  --model "$MODEL" --input "$TRAIN_INPUT" --limit "$COLLECT_LIMIT" \
  --output "$PROBES" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --trajectory-batch-size 1 --boundary step --temperature 1 \
  --max-trajectory-tokens 16384 --probe-mode judge --probe-batch-size 16 \
  --max-model-len 32768 --dtype bfloat16 --seed "$SEED"

# Build up to COUNT examples for each method: judge-threshold W->C, stable
# step-only W->C, and generated correct trajectories as the SFT control.
uv run python -m probe_int.pipeline build-transition-sft \
  --input "$PROBES" --output "$JUDGE_DATA" --limit "$COUNT" \
  --correct-threshold 0.5 --wrong-threshold 0.5

uv run python -m probe_int.pipeline build-transition-sft \
  --input "$PROBES" --output "$STABLE_DATA" --limit "$COUNT" \
  --correct-threshold 0.7 --wrong-threshold 0.3 --target step_only \
  --skip-uncertain-probes --min-probe-gap 2 --min-target-tokens 64 \
  --max-target-tokens 2048 --require-stable-correct --require-correct-final-grade

uv run python -m probe_int.pipeline build-correct-trajectory-sft \
  --input "$PROBES" --output "$CORRECT_DATA" --model "$MODEL" --limit "$COUNT"

eval_model() {
  local model_dir="$1"
  uv run python -m probe_int.evaluate \
    --model "$model_dir" --input "$EVAL_INPUT" \
    --output "$model_dir/deepmath-103k/eval-seed${EVAL_SEED}.jsonl" \
    --max-model-len 32768 --max-tokens 16384 --temperature 1 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization 0.8 --dtype bfloat16 --seed "$EVAL_SEED"
}

for METHOD in judge05 stablewc correct; do
  case "$METHOD" in
    judge05)
      DATA="$JUDGE_DATA"
      NAME="deepseek-r1-1.5b-sft-judge05-up-to${COUNT}"
      ;;
    stablewc)
      DATA="$STABLE_DATA"
      NAME="deepseek-r1-1.5b-sft-stablewc-up-to${COUNT}"
      ;;
    correct)
      DATA="$CORRECT_DATA"
      NAME="deepseek-r1-1.5b-sft-correct-up-to${COUNT}"
      ;;
  esac
  MODEL_DIR="$OUT/${NAME}-train-seed${SEED}"
  ROWS=$(wc -l < "$DATA")
  if [[ "$ROWS" -eq 0 ]]; then
    echo "No training examples were produced for $METHOD from $COLLECT_LIMIT collected problems." >&2
    exit 1
  fi
  echo "Training $METHOD with $ROWS examples (target: up to $COUNT)."
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" uv run torchrun --standalone \
    --nproc_per_node="$NPROC_PER_NODE" -m probe_int.train \
    --model "$MODEL" --data "$DATA" --output "$MODEL_DIR" \
    --epochs 3 --learning-rate 1e-5 --batch-size 1 \
    --gradient-accumulation-steps 4 --max-length 32768 --lora-r 0 \
    --seed "$SEED"

  python - "$MODEL_DIR/experiment.json" "$SEED" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as stream:
    actual = json.load(stream)["seed"]
if actual != expected:
    raise SystemExit(f"Training seed mismatch in {path}: {actual} != {expected}")
PY

  eval_model "$MODEL_DIR"
done
