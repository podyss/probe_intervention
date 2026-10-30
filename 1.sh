export CUDA_VISIBLE_DEVICES=0,1,2,3

BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
FINETUNED_MODEL=outputs/teacher-first-error-short-200/r1-1.5b-full-finetune
EVAL_SEED=42

# Base 模型
uv run python -m probe_int.evaluate \
  --model "$BASE_MODEL" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/deepmath-103k/r1-1.5b-base-eval-seed${EVAL_SEED}.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --seed "$EVAL_SEED"

# 全参微调模型
uv run python -m probe_int.evaluate \
  --model "$FINETUNED_MODEL" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/deepmath-103k/r1-1.5b-full-finetune-eval-seed${EVAL_SEED}.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --seed "$EVAL_SEED"




# 训练
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train_error_explanations \
  --model "$BASE_MODEL" \
  --data data/teacher-first-error-reasoning-200-mixed-160/train.jsonl \
  --output outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-length 9216 \
  --gpu-memory-fraction 0.8 \
  --seed 42


# 评估

CUDA_VISIBLE_DEVICES=2,3 uv run python -m probe_int.evaluate_error_explanations \
  --model outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160 \
  --input data/teacher-first-error-short-200/test.jsonl \
  --output outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160-test-yes-no.jsonl \
  --tensor-parallel-size 2 \
  --max-model-len 9216 \
  --gpu-memory-utilization 0.8 \
  --batch-size 4 \
  --max-new-tokens 128

DeepMath 测试集，评估解题答案准确率：

CUDA_VISIBLE_DEVICES=2,3 uv run probe-int-eval \
  --model outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160 \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/teacher-first-error-reasoning-200/r1-1.5b-full-finetune-mixed-160-deepmath-test.jsonl \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --gpu-memory-utilization 0.8 \
  --temperature 1


# 消融
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

CUDA_VISIBLE_DEVICES=2,3 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train_error_explanations \
  --model "$BASE_MODEL" \
  --data data/deepmath-self-generated-correct-160/train.jsonl \
  --output "outputs/deepmath-self-generated-correct-160/r1-1.5b-full-finetune" \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-length 9216 \
  --gpu-memory-fraction 0.8 \
  --seed 42

在 DeepMath test 上评估解题准确率：

CUDA_VISIBLE_DEVICES=2,3 uv run probe-int-eval \
  --model "outputs/deepmath-self-generated-correct-160/r1-1.5b-full-finetune" \
  --input data/deepmath-103k/test.jsonl \
  --output outputs/deepmath-self-generated-correct-160/deepmath-test.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --seed 42