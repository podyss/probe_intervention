export OUT=outputs/math-qwen3-4b
export MODEL=Qwen/Qwen3-4B-Instruct-2507


# 从完整 JSONL 中采集前 200 道题的解题轨迹，并在每个推理步骤边界生成答案探针。
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.pipeline collect \
  --model "$MODEL" --input data/math/train.jsonl --limit 2000 \
  --output "$OUT/probes-2000.jsonl" \
  --tensor-parallel-size 2 \
  --boundary step --temperature 1 --max-trajectory-tokens 16384 \
  --probe-temperature 0 --probe-samples 1 --max-answer-tokens 512 \
  --probe-batch-size 4 --max-model-len 32768 --dtype bfloat16 --seed 42

# 从采集结果中筛选错误到正确的探针转移，并导出为 SFT 训练数据。
uv run python -m probe_int.pipeline build-transition-sft \
  --input "$OUT/probes-2000.jsonl" \
  --output "$OUT/sft-wrong-to-correct-2000.jsonl" \
  --limit 1000 --correct-threshold 0.8 --wrong-threshold 0.2 \
  --allow-multiple-transitions-per-trajectory

# 修复
uv run python -m probe_int.pipeline localize \
  --input outputs/math-qwen3-4b/probes-2000.jsonl \
  --output outputs/math-qwen3-4b/correct-to-wrong-candidates-2000.jsonl \
  --correct-threshold 0.8 \
  --wrong-threshold 0.2

CUDA_VISIBLE_DEVICES=0,1 \
uv run python -m probe_int.pipeline repair \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input outputs/math-qwen3-4b/correct-to-wrong-candidates-2000.jsonl \
  --output outputs/math-qwen3-4b/repaired-int-style-2000.jsonl \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --success-criterion continuation \
  --max-step-tokens 4096 \
  --max-continuation-tokens 8192 \
  --continuation-samples 4 \
  --continuation-temperature 0.7 \
  --max-retries 1

# outputs/math-qwen3-4b/repaired-int-style-200.jsonl
# 将修复结果转换为带 input_ids/labels 的 SFT 数据，并过滤未成功的修复。
uv run python -m probe_int.pipeline build-sft \
  --input "$OUT/repaired-int-style-200.jsonl" \
  --output "$OUT/sft-repaired-int-style-200.jsonl" \
  --target prefix_step

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-repaired-int-style-200.jsonl" \
  --output "$OUT/qwen3-4b-sft-repaired-200" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 4 --gradient-accumulation-steps 1 \
  --max-length 32768 --lora-r 0 --seed 42

# 训练
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-wrong-to-correct-2000.jsonl" \
  --output "$OUT/qwen3-4b-sft-wrong-to-correct-2000" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 4 --gradient-accumulation-steps 1 \
  --max-length 32768 --lora-r 0 --seed 42

# 评估
export MODEL=outputs/math-qwen3-4b/qwen3-4b-full-sft-wrong-to-correct-97/checkpoint-25

# math
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/math-500/test.jsonl \
  --output outputs/math-qwen3-4b/w2c-2000/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/aime-2024/test.jsonl \
  --output outputs/math-qwen3-4b/w2c-2000/aime-2024/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
