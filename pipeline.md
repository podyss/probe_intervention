# GSM8K Probe Intervention：最新实验

以下保留两组近期实验：Qwen2.5-3B 的 400 条训练数据对照，以及 2026-10-22 的 DeepSeek-R1-Distill-Qwen-1.5B W→R 实验。详细产物、参数和审计记录见各输出目录中的 `README.md`、`summary.json` 及 manifest。

## Qwen2.5-3B：400 条数据对照

三种模型均在完整 GSM8K test split（1319 题）上评估，使用 temperature 1、最多生成 8192 tokens、max model length 32768；准确率以全部测试题为分母。

| 方法 | 数据与训练目标 | Test 准确率 |
| --- | --- | ---: |
| Base | Qwen2.5-3B 基础模型 | 54.36% (717/1319) |
| 普通 SFT | 基础模型生成的正确轨迹，取 400 道不同训练题；监督完整生成轨迹 | 69.14% (912/1319) |
| Wrong→correct intervention SFT | 400 条相邻探针由错转对的数据 | 77.48% (1022/1319) |

Wrong→correct 样本取自基础模型的 step 边界探针：前一个探针答案错误、相邻的后一个探针答案正确。训练目标为“题目 + 原轨迹前缀（截至错误探针及之前步骤）+ 导致探针转为正确的步骤”，在该正确步骤处截断，不包含后续完整解答；题目 prompt 的 loss 屏蔽，只监督轨迹目标。400 条样本覆盖 362 道题。两种 SFT 均为全参数训练，3 epochs、学习率 `1e-5`、batch size 1、梯度累积 4、seed 42。

相关产物位于 `outputs/pgi-gsm8k-qwen2.5-3b-20261021/`：`sft-generated-correct-trajectories-400.jsonl`、`sft-wrong-to-correct-400.jsonl` 及对应训练模型和评估结果。

### 数据导出

```bash
export OUT=outputs/pgi-gsm8k-qwen2.5-3b-20261021
export MODEL=/root/tmp/users/wsc/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b

uv run python -m probe_int.pipeline build-transition-sft \
  --input "$OUT/probes-gpu0.jsonl" "$OUT/probes-gpu1.jsonl" \
  --output "$OUT/sft-wrong-to-correct-400.jsonl" --limit 400 \
  --allow-multiple-transitions-per-trajectory

uv run python -m probe_int.pipeline build-correct-trajectory-sft \
  --input "$OUT/trajectories-fresh-gpu0.jsonl" "$OUT/trajectories-fresh-gpu1.jsonl" \
  --exclude-input "$OUT/sft-wrong-to-correct-400.jsonl" "$OUT/sft-standard-400.jsonl" \
  --output "$OUT/sft-generated-correct-trajectories-400.jsonl" \
  --model "$MODEL" --limit 400
```

`probes-gpu{0,1}.jsonl` 是已采集的探针轨迹；`trajectories-fresh-gpu{0,1}.jsonl` 是排除已有训练题后生成的 base 模型轨迹。普通 SFT 数据导出时从不同来源题中筛选正常结束且最终答案正确的 400 条轨迹。以上输入文件保存在 `$OUT`。

### 训练与评估

```bash
uv run python -m probe_int.pipeline prepare \
  --input data/gsm8k-full/test.parquet \
  --output "$OUT/test-full.jsonl" --split test

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" --data "$OUT/sft-generated-correct-trajectories-400.jsonl" \
  --output "$OUT/qwen2.5-3b-full-sft-generated-correct-trajectories-400" \
  --epochs 3 --learning-rate 1e-5 --batch-size 1 \
  --gradient-accumulation-steps 4 --max-length 16384 --lora-r 0 --seed 42

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" --data "$OUT/sft-wrong-to-correct-400.jsonl" \
  --output "$OUT/qwen2.5-3b-full-sft-wrong-to-correct-400" \
  --epochs 3 --learning-rate 1e-5 --batch-size 1 \
  --gradient-accumulation-steps 4 --max-length 16384 --lora-r 0 --seed 42

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.evaluate \
  --model "$MODEL" --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-base-temp1-8k.jsonl" \
  --max-model-len 32768 --max-tokens 8192 --temperature 1 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 --seed 42

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.evaluate \
  --model "$OUT/qwen2.5-3b-full-sft-generated-correct-trajectories-400" \
  --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-generated-correct-trajectories-temp1-8k.jsonl" \
  --max-model-len 32768 --max-tokens 8192 --temperature 1 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 --seed 42

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.evaluate \
  --model "$OUT/qwen2.5-3b-full-sft-wrong-to-correct-400" \
  --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-sft-temp1-8k.jsonl" \
  --max-model-len 32768 --max-tokens 8192 --temperature 1 \
  --tensor-parallel-size 1 --data-parallel-size 1 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 --seed 42
```

## DeepSeek-R1-Distill-Qwen-1.5B：W→R

以下两次训练均使用本地 base checkpoint、全参数微调（LoRA r=0），并保持训练集与测试集隔离。

## 数据与设置

从训练集前 200 题采集模型自身生成的轨迹（temperature=1、top_p=1、最多 8192 tokens），在 step 边界以 temperature=0、最多 128 tokens 的探针标记答案状态。得到 52 个相邻 wrong→correct 转移，覆盖 47 题。使用 GSM8K 完整测试集 1319 题。

两次训练均为 3 epochs、学习率 `1e-5`、batch size 1、梯度累积 4、BF16、seed 42；评估均使用显式思考模板、贪心解码（temperature=0、top_p=1）、最多 8192 tokens、max model length 16384。

## 实验结果

| 训练目标 | Test 正确数 | 准确率 | 相对 Base |
| --- | ---: | ---: | ---: |
| Base | 991/1319 | 75.13% | - |
| W→R 前缀监督 | 986/1319 | 74.75% | -0.38 pp |
| 仅监督纠错段 | 972/1319 | 73.69% | -1.44 pp |

两种 SFT 都未提升总体准确率。前缀监督模型位于 `outputs/pgi-gsm8k-deepseek-r1-1.5b-wr-200-trajectories/`；仅纠错段模型位于 `outputs/pgi-gsm8k-deepseek-r1-1.5b-wr-200-step-only/`。两项评估均为 complete，评估明细及训练审计见对应输出目录。达到 token 上限或无法解析的输出均计错。

## 复现

前缀监督版本使用 52 个 W→R 样本，对问题 prompt 之后的完整轨迹前缀计算 loss：

```bash
PYTHONPATH=. .venv/bin/python outputs/pgi-gsm8k-deepseek-r1-1.5b-wr-200-trajectories/train-evaluate.py
```

仅纠错段版本复用完全相同的 52 个样本，只对 W→R 区间 `[start:end]` 计算 loss，其余 prompt 和推理 token 的 labels 设为 `-100`：

```bash
PYTHONPATH=. .venv/bin/python outputs/pgi-gsm8k-deepseek-r1-1.5b-wr-200-step-only/train-evaluate.py
```

结果核验与汇总脚本为第二个目录中的 `finalize.py`。重跑时使用新的输出目录，避免覆盖已有产物。
