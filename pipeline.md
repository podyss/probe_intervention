# Probe-Guided Intervention: GSM8K

本次运行日期：2026-10-21。以下命令在仓库根目录执行，并通过 `uv run` 使用项目 `.venv` 中由 `uv.lock` 锁定依赖的 Python 环境。模型使用本地完整 checkpoint，采集和修复均启用 vLLM prefix caching（未传 `--disable-prefix-caching`）。

```bash
export MODEL=/root/tmp/users/wsc/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
export OUT=outputs/pgi-gsm8k-100-20261021
```

## 数据准备

数据源为仓库自带的 `data/gsm8k-full/train.parquet`。本次准备过 128、1600 和 3200 条的训练集；最终并行分片使用 1600 条数据。完整训练集准备命令如下：

```bash
uv run python -m probe_int.pipeline prepare \
  --input data/gsm8k-full/train.parquet \
  --output "$OUT/train-1600.jsonl" \
  --split train --limit 1600
```

初始性能检查另准备过 128 条：

```bash
uv run python -m probe_int.pipeline prepare \
  --input data/gsm8k-full/train.parquet \
  --output "$OUT/train.jsonl" \
  --split train --limit 128
```

## 候选轨迹采集

采集设置：Qwen3-1.7B，step 边界，最多 8192 个轨迹 token，轨迹温度 0，答案探针每个边界采样 1 次、温度 0；每批 4 个 probe 请求。prefix caching 默认启用。最终两个续跑分片分别对应 GPU 0 和 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.pipeline collect \
  --input "$OUT/train-gpu0-remaining.jsonl" \
  --output "$OUT/probes-gpu0-cache.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 42 --boundary step --interval 256 \
  --max-trajectory-tokens 8192 --temperature 0 \
  --max-answer-tokens 128 --probe-samples 1 \
  --probe-temperature 0 --probe-batch-size 4

CUDA_VISIBLE_DEVICES=1 uv run python -m probe_int.pipeline collect \
  --input "$OUT/train-gpu1-remaining.jsonl" \
  --output "$OUT/probes-gpu1-cache.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 42 --boundary step --interval 256 \
  --max-trajectory-tokens 8192 --temperature 0 \
  --max-answer-tokens 128 --probe-samples 1 \
  --probe-temperature 0 --probe-batch-size 4
```

以上两个采集分片在中断时没有跑完。此前各轮采集的结果合并为 `$OUT/probes-merged.jsonl`（1101 条轨迹）；使用该文件进行候选定位：

```bash
uv run python -m probe_int.pipeline localize \
  --input "$OUT/probes-merged.jsonl" \
  --output "$OUT/candidates-merged.jsonl" \
  --correct-threshold 0.8 --wrong-threshold 0.2
```

`localize` 按探针在轨迹内部的表现定位由正确转为错误的区间，不以整条原始轨迹的最终正确性作为过滤条件。定位结果中共有 98 条候选；其中 90 条原始轨迹的 `final_grade.correct` 为 `true`。这 90 条候选单独筛出为 `$OUT/candidates-final-correct.jsonl`，仅作为后续修复输入。

## 修复全部 98 条候选

修复输入保留所有有候选区间的轨迹，不按原始最终答题正误过滤。GPU 0、1 各处理 49 条；每个进程使用单卡（`--tensor-parallel-size 1`），避免双卡 tensor parallel 初始化时遇到的 CUDA symmetric-memory 错误。prefix caching 保持启用：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.pipeline repair \
  --input "$OUT/candidates-all98-gpu0.jsonl" \
  --output "$OUT/repaired-all98-gpu0.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 42 \
  --max-oracle-tokens 4096 --max-step-tokens 256 \
  --repair-temperature 0 --max-continuation-tokens 8192 \
  --continuation-samples 4 --continuation-temperature 0.7

CUDA_VISIBLE_DEVICES=1 uv run python -m probe_int.pipeline repair \
  --input "$OUT/candidates-all98-gpu1.jsonl" \
  --output "$OUT/repaired-all98-gpu1.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 42 \
  --max-oracle-tokens 4096 --max-step-tokens 256 \
  --repair-temperature 0 --max-continuation-tokens 8192 \
  --continuation-samples 4 --continuation-temperature 0.7
```

两份分片输出合并为 `$OUT/repaired-all98-merged.jsonl`，共 98 行；其中 65 条 repair step 被接受，62 条至少有一个正确 continuation。SFT 使用 `prefix_step` 目标生成：

```bash
uv run python -m probe_int.pipeline build-sft \
  --input "$OUT/repaired-all98-merged.jsonl" \
  --output "$OUT/sft-all98.jsonl" \
  --target prefix_step
```

最终 `$OUT/sft-all98.jsonl` 有 62 条训练样本。早先只针对 90 条最终答对轨迹启动的修复，以及 TP=2 启动失败的尝试均保留在独立文件中，不是上述训练集的来源。早期采集还试过不同温度、probe batch size 和 prefix caching 设置；最终汇总来自多轮采集文件，不能仅用上面的缓存续跑命令重现全部 1101 条轨迹。所有实际参数和状态以输出旁的 `.manifest.json` 为准。

## 温度 1 多轮修复与追加 SFT

应用户要求，对原 98 条候选进行温度 1 的多轮修复。每轮只重试此前还没有至少一个正确 continuation 的候选，GPU 0、1 单卡并行；每条修复仍采样 4 个 temperature 0.7 的 continuation，并沿用原有拒绝规则。不同轮次使用不同 seed。6 轮后共有 83 个不同题目至少有一个有效修复且至少一个 continuation 答对；按题目选择正确 continuation 数最多的修复记录，生成 `$OUT/repaired-all98-temp1-best.jsonl`，再导出 `$OUT/sft-temp1-retries-83.jsonl`。另有 15 条候选在这些尝试中未满足训练条件。

```bash
uv run python -m probe_int.pipeline build-sft \
  --input "$OUT/repaired-all98-temp1-best.jsonl" \
  --output "$OUT/sft-temp1-retries-83.jsonl" \
  --target prefix_step

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-temp1-retries-83.jsonl" \
  --output "$OUT/qwen3-1.7b-full-sft-temp1-83" \
  --epochs 3 --learning-rate 1e-5 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 16384 --lora-r 0 --seed 42
```

追加全量 SFT 已完成：83 条样本、3 个 epoch、63 个优化步，训练 loss 约 0.321。模型保存在 `$OUT/qwen3-1.7b-full-sft-temp1-83`。该 checkpoint 尚未重新评估 GSM8K test split。

## 全量 SFT

使用 `--lora-r 0` 在完整 Qwen3-1.7B 模型上更新全部参数，单卡 GPU 0 训练：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" \
  --data "$OUT/sft-all98.jsonl" \
  --output "$OUT/qwen3-1.7b-full-sft" \
  --epochs 3 --learning-rate 1e-6 \
  --batch-size 1 --gradient-accumulation-steps 4 \
  --max-length 16384 --lora-r 0 --seed 42
```

训练完成：62 条样本、3 个 epoch、48 个优化步，最终训练 loss 约 0.312。完整模型权重保存在 `$OUT/qwen3-1.7b-full-sft/model.safetensors`；参数记录在该目录的 `experiment.json`，其中 `lora_r` 为 0。

## GSM8K 测试集评估

完整 test split 有 1319 题。使用 GPU 0、1 做两个 vLLM data-parallel worker，temperature 0、最多生成 16384 tokens；模型上下文上限设为 32768，以容纳 prompt 和生成内容：

```bash
uv run python -m probe_int.pipeline prepare \
  --input data/gsm8k-full/test.parquet \
  --output "$OUT/test-full.jsonl" --split test

CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$OUT/qwen3-1.7b-full-sft" \
  --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-full-sft-16k.jsonl" \
  --max-model-len 32768 --max-tokens 16384 \
  --temperature 0 --tensor-parallel-size 1 \
  --data-parallel-size 2 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 42
```

Base 和全量 SFT checkpoint 均在相同的 1319 题测试集上重新评估，设置为 temperature 0、max-tokens 16384、max-model-len 32768。Base 答对 1175 题，总准确率 89.08%，达到生成上限 14 题；SFT 答对 1104 题，总准确率 83.70%，达到生成上限 133 题。总准确率以全部 1319 题为分母，达到上限且未能判定正确的题计为错误。结果分别保存在 `$OUT/eval-base-16k.jsonl` 和 `$OUT/eval-full-sft-16k.jsonl`，manifest 状态均为 `complete`。

追加训练的 83 条样本 checkpoint 使用 temperature 1、max-tokens 8192、max-model-len 32768，在相同 1319 题 test split 上评估。答对 1174 题，总准确率 89.01%；17 题达到生成上限。结果保存在 `$OUT/eval-sft-temp1-8k.jsonl`，manifest 状态为 `complete`。

原始 Base checkpoint 也使用相同设置重新评估：temperature 1、max-tokens 8192、max-model-len 32768。答对 1190/1319 题，总准确率 90.22%；32 题达到生成上限。结果保存在 `$OUT/eval-base-temp1-8k.jsonl`，manifest 状态为 `complete`。

## 以修复步探针判定成功并重试

后续修复改用修复步后的答案探针作为成功条件：上下文为原始问题 prompt、候选前缀、修复步和原有 probe suffix；探针温度 0，最多生成 128 tokens。只要该探针答案正确就保留修复步，不要求后续 continuation 正确，也不生成 continuation 作筛选。为避免答案泄漏，修复步仍须通过原有格式和答案泄漏检查。`repair` 的 `--max-retries` 设置额外重试次数，初次尝试之外最多再试 16 次，探针通过后立即停止。

第一轮处理全部 98 条候选（seed 42），第二轮只重试第一轮未通过的候选（seed 43）；第三阶段只处理第二轮仍未通过的候选，并启用最多 16 次重试（seed 44）。各轮输入分片由上一轮未通过的结果交替分配给 GPU 0、1，prefix caching 默认启用：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.pipeline repair \
  --input "$OUT/candidates-probe-r3-gpu0.jsonl" \
  --output "$OUT/repaired-probe-r3-gpu0.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 44 --max-oracle-tokens 4096 \
  --max-step-tokens 256 --repair-temperature 1 \
  --success-criterion probe --max-repair-probe-tokens 128 \
  --repair-probe-temperature 0 --max-retries 16

CUDA_VISIBLE_DEVICES=1 uv run python -m probe_int.pipeline repair \
  --input "$OUT/candidates-probe-r3-gpu1.jsonl" \
  --output "$OUT/repaired-probe-r3-gpu1.jsonl" \
  --model "$MODEL" --max-model-len 16384 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.75 \
  --dtype bfloat16 --seed 44 --max-oracle-tokens 4096 \
  --max-step-tokens 256 --repair-temperature 1 \
  --success-criterion probe --max-repair-probe-tokens 128 \
  --repair-probe-temperature 0 --max-retries 16
```

三轮分别处理 98、80、69 条记录，修复步探针通过数分别为 18、11、38；最终 67 条候选通过。第三阶段一共执行 737 次尝试，单题最多 17 次（首次加 16 次重试）。按每题优先选择第一次通过探针的记录汇总为 `$OUT/repaired-probe-temp1-best.jsonl`，用 `prefix_step` 构建出 `$OUT/sft-probe-temp1.jsonl`，共 67 条训练样本。

按全量微调（非 LoRA）设置训练 3 个 epoch、51 个优化步，学习率 `1e-5`，batch size 1、梯度累积 4、最大训练长度 16384，训练 loss 为 0.3825。模型位于 `$OUT/qwen3-1.7b-full-sft-probe-temp1-67`，`lora_r=0`。在完整 GSM8K test split（1319 题）上使用 temperature 1、max-tokens 8192、max-model-len 32768 评估，答对 1183 题，总准确率 89.69%；结果位于 `$OUT/eval-sft-probe-temp1-8k.jsonl`，manifest 状态为 `complete`。

```bash
uv run python -m probe_int.pipeline build-sft \
  --input "$OUT/repaired-probe-temp1-best.jsonl" \
  --output "$OUT/sft-probe-temp1.jsonl" \
  --target prefix_step

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" --data "$OUT/sft-probe-temp1.jsonl" \
  --output "$OUT/qwen3-1.7b-full-sft-probe-temp1-67" \
  --epochs 3 --learning-rate 1e-5 --batch-size 1 \
  --gradient-accumulation-steps 4 --max-length 16384 \
  --lora-r 0 --seed 42

CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$OUT/qwen3-1.7b-full-sft-probe-temp1-67" \
  --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-sft-probe-temp1-8k.jsonl" \
  --max-model-len 32768 --max-tokens 8192 --temperature 1 \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 --seed 42
```

## 原轨迹错误探针到正确探针的恢复 SFT

无需调用修复模型：从已缓存的原轨迹中筛选相邻探针状态为 `wrong -> correct` 的边界。每个样本的上下文为题目 prompt，目标为原轨迹从开头到该正确探针位置的内容；prompt token 的 label 屏蔽为 `-100`，只对轨迹 token 计算 loss。同一道题有多个合格边界时可分别形成样本。缓存的两个输入文件合计包含 425 个合格边界，稳定排序后取 400 条，覆盖 366 道不同题目。

```bash
uv run python -m probe_int.pipeline build-transition-sft \
  --input "$OUT/probes-merged.jsonl" "$OUT/probes-gpu01-final.jsonl" \
  --output "$OUT/sft-wrong-to-correct-400.jsonl" --limit 400

CUDA_VISIBLE_DEVICES=0 uv run python -m probe_int.train \
  --model "$MODEL" --data "$OUT/sft-wrong-to-correct-400.jsonl" \
  --output "$OUT/qwen3-1.7b-full-sft-wrong-to-correct-400" \
  --epochs 3 --learning-rate 1e-5 --batch-size 1 \
  --gradient-accumulation-steps 4 --max-length 16384 \
  --lora-r 0 --seed 42

CUDA_VISIBLE_DEVICES=0,1 uv run python -m probe_int.evaluate \
  --model "$OUT/qwen3-1.7b-full-sft-wrong-to-correct-400" \
  --input "$OUT/test-full.jsonl" \
  --output "$OUT/eval-sft-wrong-to-correct-400-temp1-8k.jsonl" \
  --max-model-len 32768 --max-tokens 8192 --temperature 1 \
  --tensor-parallel-size 1 --data-parallel-size 2 \
  --gpu-memory-utilization 0.75 --dtype bfloat16 --seed 42
```

本次导出 400 条样本，覆盖 366 道不同题目。单卡全量 SFT 完成 300 步，训练 loss 为 0.04118，模型位于 `$OUT/qwen3-1.7b-full-sft-wrong-to-correct-400`。在完整 GSM8K test split 评估，答对 1178/1319 题，总准确率 89.31%；输出 `$OUT/eval-sft-wrong-to-correct-400-temp1-8k.jsonl` 的 manifest 状态为 `complete`。
