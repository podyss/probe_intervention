#!/usr/bin/env bash
# Standard full-response SFT ablation against the 890-row judge experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MODEL=Qwen/Qwen3-4B-Instruct-2507
OUT=outputs/math-qwen3-4b
RUN="$OUT/ablation-correct-890"
DATA="$OUT/sft-correct-890.jsonl"
CHECKPOINT="$OUT/qwen3-4b-sft-correct-890"
mkdir -p "$RUN"

stage=starting
trap 'code=$?; if (( code != 0 )); then printf "failed: %s (exit %s)\n" "$stage" "$code" > "$RUN/status.txt"; fi' EXIT
set_stage() {
  stage="$1"
  printf '%s\n' "$stage" > "$RUN/status.txt"
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  printf '%s\n' "$stage"
}

# Prefer trajectories from the judge run. Fill from the previous run only for
# question IDs without a correct judge trajectory; the exporter deduplicates IDs.
set_stage build-existing-correct
uv run python -m probe_int.pipeline build-correct-trajectory-sft \
  --input "$OUT/probes-2000-judge.jsonl" "$OUT/probes-2000.jsonl" \
  --output "$RUN/sft-correct-existing-865.jsonl" \
  --model "$MODEL" --limit 865 > "$RUN/build-existing.log" 2>&1

# Both existing files cover the same 1000 training questions. Resample only the
# remaining 135 questions, without probes or reference answers in the prompt.
set_stage collect-supplement
uv run python -m probe_int.pipeline collect \
  --model "$MODEL" --input data/deepmath-103k/train.jsonl \
  --output "$RUN/trajectories-supplement-seed43.jsonl" \
  --exclude-input "$RUN/sft-correct-existing-865.jsonl" \
  --skip-probes --trajectory-batch-size 16 \
  --tensor-parallel-size 2 --boundary step --temperature 1 \
  --max-trajectory-tokens 16384 --max-model-len 32768 \
  --gpu-memory-utilization 0.8 --dtype bfloat16 --seed 43 \
  > "$RUN/collect-supplement.log" 2>&1

set_stage build-correct-890
uv run python -m probe_int.pipeline build-correct-trajectory-sft \
  --input "$OUT/probes-2000-judge.jsonl" "$OUT/probes-2000.jsonl" \
    "$RUN/trajectories-supplement-seed43.jsonl" \
  --output "$DATA" --model "$MODEL" --limit 890 \
  > "$RUN/build-correct-890.log" 2>&1

set_stage validate-data
uv run python - <<'PY' > "$RUN/data-audit.json"
import json
from collections import Counter
from pathlib import Path

root = Path("outputs/math-qwen3-4b")
run = root / "ablation-correct-890"
read = lambda p: [json.loads(line) for line in Path(p).open()]
data = read(root / "sft-correct-890.jsonl")
train = {r["id"]: r for r in read("data/deepmath-103k/train.jsonl")}
test_questions = {r["question"].strip() for p in [
    "data/deepmath-103k/test.jsonl", "data/aime-2024/test.jsonl",
    "data/amc23/test.jsonl",
] for r in read(p)}
sources = {}
for path in [root / "probes-2000-judge.jsonl", root / "probes-2000.jsonl",
             run / "trajectories-supplement-seed43.jsonl"]:
    for r in read(path):
        if r["final_grade"]["correct"] is True and r["trajectory_finish_reason"] == "stop":
            sources.setdefault(r["id"], (r, path.name))
assert len(data) == len({r["source_id"] for r in data}) == 890
assert len({train[r["source_id"]]["question"].strip() for r in data}) == 890
counts = Counter()
lengths, target_lengths = [], []
for r in data:
    source, name = sources[r["source_id"]]
    assert r["split"] == source["split"] == "train"
    assert source["model"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert r["target"] == "generated_correct_trajectory"
    assert source["question"] == train[r["source_id"]]["question"]
    assert source["question"].strip() not in test_questions
    assert r["input_ids"] == source["prompt_ids"] + source["trajectory_ids"]
    assert r["labels"] == [-100] * len(source["prompt_ids"]) + source["trajectory_ids"]
    assert len(r["input_ids"]) <= 32768
    counts[name] += 1
    lengths.append(len(r["input_ids"]))
    target_lengths.append(len(source["trajectory_ids"]))
print(json.dumps({"rows": len(data), "unique_questions": 890,
                  "source_counts": dict(counts), "evaluation_question_overlap": 0,
                  "max_length": max(lengths), "total_tokens": sum(lengths),
                  "supervised_tokens": sum(target_lengths),
                  "mean_length": sum(lengths) / len(lengths)}, indent=2))
PY

set_stage training
uv run torchrun --standalone --nproc_per_node=2 -m probe_int.train \
  --model "$MODEL" --data "$DATA" --output "$CHECKPOINT" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 32768 --lora-r 0 --seed 42 > "$RUN/train.log" 2>&1

for spec in 'deepmath-103k deepmath-103k 123' 'aime-2024 AIME 42' 'amc23 amc23 42'; do
  read -r dataset folder seed <<< "$spec"
  set_stage "evaluate-$dataset"
  filename=eval-qwen3-4b.jsonl
  if [[ "$seed" == 123 ]]; then filename=eval-qwen3-4b-seed123.jsonl; fi
  uv run python -m probe_int.evaluate \
    --model "$CHECKPOINT" --input "data/$dataset/test.jsonl" \
    --output "$CHECKPOINT/$folder/$filename" \
    --max-model-len 32768 --max-tokens 16384 --temperature 1 \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.8 --seed "$seed" \
    > "$RUN/eval-$dataset.log" 2>&1
done
set_stage summarize
uv run python - <<'PY' > "$RUN/summary.json"
import json
from pathlib import Path

root = Path("outputs/math-qwen3-4b")
results = {}
for dataset, filename in [
    ("deepmath-103k", "eval-qwen3-4b-seed123.jsonl"),
    ("AIME", "eval-qwen3-4b.jsonl"),
    ("amc23", "eval-qwen3-4b.jsonl"),
]:
    scores = {}
    for label, model in [
        ("probe_890", "qwen3-4b-sft-wrong-to-correct-2000-judge"),
        ("standard_sft_890", "qwen3-4b-sft-correct-890"),
    ]:
        manifest = json.loads((root / model / dataset / (filename + ".manifest.json")).read_text())
        assert manifest["status"] == "complete"
        scores[label] = manifest["summary"]
    scores["accuracy_delta"] = scores["standard_sft_890"]["accuracy"] - scores["probe_890"]["accuracy"]
    results[dataset] = scores
print(json.dumps(results, indent=2))
PY
set_stage complete
