
CUDA_VISIBLE_DEVICES=2 uv run python -m probe_int.evaluate_step_labels \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --input data/teacher-first-error-short-200/test.jsonl \
  --output outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-eval.jsonl \
  --max-model-len 8192 \
  --batch-size 4 \
  --seed 42




BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
ADAPTER=outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-lora
MERGED=outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-merged

# 训练
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  -m probe_int.train_step_labels \
  --model "$BASE_MODEL" \
  --data data/teacher-first-error-short-200/train.jsonl \
  --output "$ADAPTER" \
  --epochs 3 \
  --learning-rate 2e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-length 4096 \
  --lora-r 8 \
  --gpu-memory-fraction 0.9 \
  --seed 42

# 合并 LoRA adapter，生成可供评测脚本加载的完整模型
uv run python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "Qwen/Qwen3-4B-Instruct-2507"
adapter = "outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-lora"
merged = "outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-merged"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype="bfloat16",
    device_map="cpu",
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(model, adapter)
model.merge_and_unload().save_pretrained(merged, safe_serialization=True)
AutoTokenizer.from_pretrained(base_model, trust_remote_code=False).save_pretrained(merged)
PY

# 评测测试集
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.evaluate_step_labels \
  --model "$MERGED" \
  --input data/teacher-first-error-short-200/test.jsonl \
  --output outputs/teacher-first-error-short-200/qwen3-4b-instruct-2507-lora-eval.jsonl \
  --max-model-len 8192 \
  --batch-size 4 \
  --seed 42