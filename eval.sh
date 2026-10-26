export MODEL=Qwen/Qwen3-4B-Instruct-2507
export MODEL=Qwen/Qwen3-1.7B
export MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# math
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/math-500/test.jsonl \
  --output outputs/math500/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# gsm8k
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input outputs/gsm8k-full-test.jsonl \
  --output outputs/gsm8k/eval-full-deepseek-r1-distill-qwen-1.5b.jsonl \
  --max-model-len 16384 \
  --max-tokens 8192 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


# AIME 2024
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/aime-2024/test.jsonl \
  --output outputs/aime-2024/eval-qwen3-4b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8


export MODEL=Qwen/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$MODEL" \
  --input data/math-500/convert/test-full.jsonl \
  --output outputs/math500/eval-full-16k-qwen3-1.7b.jsonl \
  --max-model-len 32768 \
  --max-tokens 16384 \
  --temperature 1 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8