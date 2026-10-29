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
uv run python -m probe_int.prepare_error_explanations \
  --input-dir data/teacher-first-error-short-200 \
  --output-dir data/teacher-first-error-explanations-200

生成的数据保留原有切分：训练集 168 条、测试集 40 条。Yes 回答使用已有的 teacher_rationale 解释错误；No 回答为 No.。提示词只提供题目和当前推理前缀，不提供参考答案或错误标注。注意这是重组现有标注，不是新增独立标注。

随后用四张卡全参训练：

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

BASE_MODEL=/root/tmp/users/wsc/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run torchrun --standalone --nproc_per_node=4 \
  -m probe_int.train_error_explanations \
  --model "$BASE_MODEL" \
  --data data/teacher-first-error-explanations-200/train.jsonl \
  --output outputs/teacher-first-error-explanations-200/r1-1.5b-full-finetune \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-length 4096 \
  --gpu-memory-fraction 0.9 \
  --seed 42

改动位于 probe_int/prepare_error_explanations.py:8、probe_int/train_error_explanations.py:1 和 data/README.md:93。验证过数据生成计数与 Python 编译；实际训练尚未运行。训练脚本会拒绝非空输出目录。