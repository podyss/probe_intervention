export OUT=outputs/math-qwen3-4b
export MODEL=Qwen/Qwen3-4B-Instruct-2507

CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/deepmath-103k/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# 从完整 JSONL 中采集前 200 道题的解题轨迹，并在每个推理步骤边界生成答案探针。
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.pipeline collect \
  --model "$MODEL" --input data/deepmath-103k/train.jsonl --limit 2000 \
  --output "$OUT/probes-2000.jsonl" \
  --tensor-parallel-size 2 \
  --boundary step --temperature 1 --max-trajectory-tokens 16384 \
  --probe-temperature 0 --probe-samples 1 --max-answer-tokens 512 \
  --probe-batch-size 4 --max-model-len 32768 --dtype bfloat16 --seed 42

# 从采集结果中筛选错误到正确的探针转移，并导出为 SFT 训练数据。
uv run python -m probe_int.pipeline build-transition-sft \
  --input "$OUT/probes-2000.jsonl" \
  --output "$OUT/sft-wrong-to-correct-2000-135.jsonl" \
  --limit 135 --correct-threshold 0.8 --wrong-threshold 0.2


# 训练
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-wrong-to-correct-2000-135.jsonl" \
  --output "$OUT/qwen3-4b-sft-wrong-to-correct-2000-135" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 32768 --lora-r 0 --seed 42

# 评估

# deepmath
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135/deepmath-103k/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135" \
  --input data/aime-2024/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135/AIME/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# amc23
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135" \
  --input data/amc23/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-135/amc23/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# 修复
# -------------------------------

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
  --input "$OUT/repaired-int-style-2000.jsonl" \
  --output "$OUT/sft-repaired-int-style-2000.jsonl" \
  --target prefix_step

# 训练
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-repaired-int-style-2000.jsonl" \
  --output "$OUT/qwen3-4b-sft-repaired-2000" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 32768 --lora-r 0 --seed 42

# 评估
# deepmath
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000/deepmath-103k/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000" \
  --input data/aime-2024/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000/AIME/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 

# amc23
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000" \
  --input data/amc23/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-repaired-2000/amc23/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# 对照SFT
# -------------------------------

uv run python -m probe_int.pipeline build-correct-trajectory-sft \
  --input "outputs/math-qwen3-4b/probes-2000.jsonl" \
  --output "$OUT/sft-correct-135.jsonl" \
  --model "$MODEL" \
  --limit 135

CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-correct-135.jsonl" \
  --output "$OUT/qwen3-4b-sft-correct-135" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 32768 --lora-r 0 --seed 42

# 评估
# deepmath
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-correct-135" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-correct-135/deepmath-103k/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-correct-135" \
  --input data/aime-2024/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-correct-135/AIME/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 

# amc23
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-correct-135" \
  --input data/amc23/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-correct-135/amc23/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# --------------------

# 收集数据
CUDA_VISIBLE_DEVICES=2,3 uv run python -m probe_int.pipeline collect \
  --model "$MODEL" --input data/deepmath-103k/train.jsonl --limit 2000 \
  --output "$OUT/probes-2000-judge.jsonl" \
  --tensor-parallel-size 2 --trajectory-batch-size 2 \
  --boundary step --temperature 1 --max-trajectory-tokens 16384 \
  --probe-batch-size 16 --probe-mode judge \
  --max-model-len 32768 --dtype bfloat16 --seed 42


# 导出为 SFT 训练数据。
uv run python -m probe_int.pipeline build-transition-sft \
  --input "$OUT/probes-2000-judge.jsonl" \
  --output "$OUT/sft-wrong-to-correct-2000-judge.jsonl" \
  --limit 890 \
  --correct-threshold 0.5 --wrong-threshold 0.5


# 训练
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-wrong-to-correct-2000-judge.jsonl" \
  --output "$OUT/qwen3-4b-sft-wrong-to-correct-2000-judge" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 32768 --lora-r 0 --seed 42

# 评估

# deepmath
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge/deepmath-103k/eval-qwen3-4b-seed123.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 --seed 123

# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge" \
  --input data/aime-2024/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge/AIME/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# amc23
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge" \
  --input data/amc23/test.jsonl \
  --output outputs/math-qwen3-4b/qwen3-4b-sft-wrong-to-correct-2000-judge/amc23/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8