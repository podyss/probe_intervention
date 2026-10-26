#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing Judge probe collection; this script does not collect rollouts.
OUT=${OUT:-outputs/math-qwen3-4b}
MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
PROBES="$OUT/probes-2000-judge.jsonl"
COUNT=${COUNT:-489}
SEEDS=(42 123 2026)
EVAL_SEED=${EVAL_SEED:-123}

if [[ ! -s "$PROBES" || ! -s "$PROBES.manifest.json" ]]; then
  echo "Missing existing probe data: $PROBES" >&2
  exit 1
fi

build_transition_if_needed() {
  local output="$1"
  shift
  if [[ -s "$output" && -s "$output.manifest.json" ]]; then
    echo "Reusing $output"
    return
  fi
  uv run python -m probe_int.pipeline build-transition-sft \
    --input "$PROBES" --output "$output" --limit "$COUNT" "$@"
}

JUDGE_DATA="$OUT/sft-wrong-to-correct-2000-judge-05-${COUNT}.jsonl"
STABLE_DATA="$OUT/sft-wrong-to-correct-2000-judge-07-step-stable-finalcorrect-${COUNT}.jsonl"
CORRECT_DATA="$OUT/sft-correct-trajectory-2000-judge-${COUNT}.jsonl"

# Equal row counts, with method-specific targets and filtering.
build_transition_if_needed "$JUDGE_DATA" \
  --correct-threshold 0.5 --wrong-threshold 0.5
build_transition_if_needed "$STABLE_DATA" \
  --correct-threshold 0.7 --wrong-threshold 0.3 --target step_only \
  --skip-uncertain-probes --min-probe-gap 2 --min-target-tokens 64 \
  --max-target-tokens 2048 --require-stable-correct --require-correct-final-grade

if [[ ! -s "$CORRECT_DATA" || ! -s "$CORRECT_DATA.manifest.json" ]]; then
  uv run python -m probe_int.pipeline build-correct-trajectory-sft \
    --input "$PROBES" --output "$CORRECT_DATA" \
    --model "$MODEL" --limit "$COUNT"
else
  echo "Reusing $CORRECT_DATA"
fi

for data in "$JUDGE_DATA" "$STABLE_DATA" "$CORRECT_DATA"; do
  rows=$(wc -l < "$data")
  [[ "$rows" -eq "$COUNT" ]] || {
    echo "Expected $COUNT rows in $data, found $rows" >&2
    exit 1
  }
done

eval_model() {
  local model_dir="$1"
  local eval_dir="$model_dir/eval-seed${EVAL_SEED}"
  CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
    --model "$model_dir" --input data/deepmath-103k/test.jsonl \
    --output "$eval_dir/deepmath-103k/eval.jsonl" --max-model-len 32768 \
    --max-tokens 16384 --temperature 1 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 --seed "$EVAL_SEED"
  CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
    --model "$model_dir" --input data/aime-2024/test.jsonl \
    --output "$eval_dir/AIME/eval.jsonl" --max-model-len 32768 \
    --max-tokens 16384 --temperature 1 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 --seed "$EVAL_SEED"
  CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
    --model "$model_dir" --input data/amc23/test.jsonl \
    --output "$eval_dir/amc23/eval.jsonl" --max-model-len 32768 \
    --max-tokens 16384 --temperature 1 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 --seed "$EVAL_SEED"
}

for SEED in "${SEEDS[@]}"; do
  for METHOD in judge05 stablewc correct; do
    case "$METHOD" in
      judge05)
        DATA="$JUDGE_DATA"
        NAME="qwen3-4b-sft-judge05-${COUNT}"
        ;;
      stablewc)
        DATA="$STABLE_DATA"
        NAME="qwen3-4b-sft-stablewc-${COUNT}"
        ;;
      correct)
        DATA="$CORRECT_DATA"
        NAME="qwen3-4b-sft-correct-${COUNT}"
        ;;
    esac
    MODEL_DIR="$OUT/${NAME}-train-seed${SEED}"
    CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
      -m probe_int.train --model "$MODEL" --data "$DATA" --output "$MODEL_DIR" \
      --epochs 3 --learning-rate 1e-5 --batch-size 1 \
      --gradient-accumulation-steps 4 --max-length 32768 --lora-r 0 \
      --seed "$SEED"

    # Confirm the seed persisted in the training manifest.
    python - "$MODEL_DIR/experiment.json" "$SEED" <<'PY'
import json
import sys
path, expected = sys.argv[1], int(sys.argv[2])
actual = json.load(open(path, encoding="utf-8"))["seed"]
if actual != expected:
    raise SystemExit(f"Training seed mismatch in {path}: {actual} != {expected}")
PY
    eval_model "$MODEL_DIR"
  done
done
