训练命令
```
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=2,3 \
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  scripts/train_jev_answer_slot_distill.py \
  --model Qwen/Qwen3-4B \
  --train data/jev-mathqa/train.jsonl \
  --output outputs/jev-ce-logits-kl-3epochs \
  --epochs 4 \
  --stop-after-epoch 1 \
  --learning-rate 5e-6 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-length 512 \
  --max-train-examples 2048 \
  --temperature 2.0 \
  --logit-loss-weight 1.0 \
  --hidden-loss-weight 0.0 \
  --seed 42
```

评估命令
```
CUDA_VISIBLE_DEVICES=2,3 \
.venv/bin/python scripts/evaluate_jev_zero_shot.py \
  --model outputs/jev-ce-logits-kl-3epochs \
  --input data/jev-mathqa/test.jsonl \
  --output outputs/jev-ce-logits-kl-3epochs-mathqa-test.jsonl \
  --batch-size 8 \
  --data-parallel-size 2 \
  --dtype bfloat16
```