#!/usr/bin/env bash
set -euo pipefail

seeds=(42 43 44)
gpu_devices="${CUDA_VISIBLE_DEVICES:-2,3}"

mixed_model="outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160"
mixed_prefix="outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160-deepmath-test"

generated_model="outputs/deepmath-self-generated-correct-160/r1-1.5b-full-finetune"
generated_prefix="outputs/deepmath-self-generated-correct-160/deepmath-test"

run_eval() {
  local model="$1"
  local output_prefix="$2"
  local seed="$3"
  local output="${output_prefix}-seed${seed}.jsonl"
  local manifest="${output}.manifest.json"

  if [[ -e "$output" || -e "$manifest" ]]; then
    if [[ -f "$output" && -f "$manifest" ]] && uv run python - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

try:
    complete = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status") == "complete"
except (OSError, json.JSONDecodeError):
    complete = False
raise SystemExit(0 if complete else 1)
PY
    then
      echo "Skipping completed evaluation: ${output}"
      return
    fi
    echo "Found an existing incomplete or partial evaluation: ${output}" >&2
    return 1
  fi

  echo "Evaluating ${model} with seed ${seed} -> ${output}"
  CUDA_VISIBLE_DEVICES="$gpu_devices" uv run probe-int-eval \
    --model "$model" \
    --input data/deepmath-103k/test.jsonl \
    --output "$output" \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --max-tokens 16384 \
    --gpu-memory-utilization 0.8 \
    --temperature 1 \
    --seed "$seed"
}

for seed in "${seeds[@]}"; do
  run_eval "$mixed_model" "$mixed_prefix" "$seed"
done

for seed in "${seeds[@]}"; do
  run_eval "$generated_model" "$generated_prefix" "$seed"
done

uv run python - "${mixed_prefix}" "${generated_prefix}" "${seeds[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

mixed_prefix, generated_prefix, *seeds = sys.argv[1:]
experiments = {
    "mixed_200_plus_160_generated_correct": mixed_prefix,
    "self_generated_correct_160_only": generated_prefix,
}
summary = {}
for name, prefix in experiments.items():
    runs = {}
    for seed in seeds:
        manifest_path = Path(f"{prefix}-seed{seed}.jsonl.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise SystemExit(f"Incomplete evaluation manifest: {manifest_path}")
        metrics = manifest["summary"]
        runs[seed] = {
            "accuracy": metrics["accuracy"],
            "accuracy_on_graded": metrics["accuracy_on_graded"],
            "correct": metrics["correct"],
            "graded": metrics["graded"],
            "rows": metrics["rows"],
        }
    summary[name] = {
        "runs": runs,
        "mean_accuracy": statistics.mean(run["accuracy"] for run in runs.values()),
        "mean_accuracy_on_graded": statistics.mean(
            run["accuracy_on_graded"] for run in runs.values()
        ),
    }

summary_path = Path("outputs/deepmath-three-seed-summary.json")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
print(f"Saved summary to {summary_path}")
PY
