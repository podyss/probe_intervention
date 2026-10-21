# Probe-Guided Intervention Training（暂名）

用**推理前缀上的强制答案分支**寻找回答能力退化的区间，替代 InT 中由模型阅读整条错误轨迹、定位首错的环节，再用特权信息生成局部替换步骤。

这是可独立复制、后续单独上传 GitHub 的初始研究实现。它不导入父目录的 OPSD 模块，不修改已有训练脚本，不包含模型权重和实验结果；目前没有宣称该定位方法等价于逻辑首错定位，也没有实验效果结论。

## 1. 算法边界

```text
问题 x ──模型自由采样──> 原始轨迹 y（原轨迹不被探针修改）
                         │
                   在前缀边界 b_i 复制上下文
                         └──关闭 think，强制直接回答──> a_i / 正误标签

任意轨迹的相邻边界出现 correct → wrong（不要求原轨迹最终答错）
                         │
               将 [b_(i-1), b_i) 标记为可疑区间
                         │
    问题 + 参考解 + 保留前缀 + 可疑区间 ──> 一个替换步骤
                         │
    原始问题 + 原始前缀 + 替换步骤 ──> 无特权信息的自由续写
                         │
               至少一个续写答案正确才保留
                         │
    SFT 默认训练：原始前缀 + 替换步骤（问题 prompt 不计 loss）
```

- 这里的“探针”是 **generation-based answer probe**，不是隐状态分类器，也不需要另训探针。
- 对所有轨迹定位；原轨迹最终正确也可有中途 correct → wrong 翻转并生成候选区间。截断、无可解析答案不当作错误。
- 默认选择**第一次观察到的相邻正确→错误翻转**，不是任意“曾经答对，后来答错”。不跨过 uncertain 边界。
- 找到的是**答案可恢复性下降的区间**。答案提前猜对、采样噪声、探针模板效应，都可能造成伪翻转；不能据此声称找到了第一个逻辑错误。
- 没有翻转的轨迹直接跳过，不回退到 LLM 找错，以便隔离这个 idea 的贡献。
- 参考解只送进生成替换步骤的独立 teacher 请求。续写只含原问题、保留前缀和替换步骤；teacher 的消息、参考解和旧错误区间不会拼进续写上下文。
- 生成数据时仍然需要模型具备局部纠错能力；此方案替换的是**找错位置**，没有消除对**纠错**的依赖。

### 与 InT 的关系

原论文 §4.1 的推荐设置是：用成功续写过滤干预，SFT 克隆 **prefix + intervention，不克隆 suffix**。官方代码的 SFT 数据列为 `intervention_guided_attempt`，训练模板使用不自动追加 EOS 的中间续写设置。

本实现默认对齐这个训练目标，同时提供 `step_only` 和 `full` 两个显式消融。`full` 就是“把修正后的完整正确轨迹拿去做 SFT”的版本。上游流程对应关系与来源见 `docs/upstream.md`。

## 2. 文件组织

```text
probe_intervention/
├── pyproject.toml             # 独立安装、依赖和 CLI
├── probe_int/
│   ├── core.py                # 答案判定、token 边界、翻转定位、SFT 标签
│   ├── prompts.py             # 问题、强制答案与局部纠正模板
│   ├── backend.py             # vLLM 推理适配层
│   ├── pipeline.py            # 分阶段数据流水线与统计
│   └── train.py               # Transformers SFT，默认 LoRA
├── data/
│   ├── example.jsonl          # 手写格式示例，不代表实际实验集
│   └── gsm8k-full/             # 完整 GSM8K train/test parquet
└── docs/upstream.md           # 与原论文和官方实现的对应关系
```

无需初始化嵌套 Git 仓库。以后将整个目录复制到新仓库即可；公开发布前自行确定项目名称和许可证。

## 3. 安装

Python >= 3.10。

在此目录运行以下命令即可创建环境并安装完整流程所需依赖：

```bash
uv sync
```

`uv.lock` 固定解析版本；vLLM/PyTorch wheel 仍受平台 CUDA 兼容性影响。无需 FlashAttention。初版后端支持具有 Qwen3 风格 `<think> ... </think>` 模板的**完整模型 checkpoint**；不直接加载未合并的 LoRA adapter。推理拒绝启用 `trust_remote_code`。

## 4. 最小数据流水线

所有命令在此目录运行。用 `python -m probe_int.pipeline` 可替代安装后的 `probe-int`。

### 4.1 整理训练数据

```bash
probe-int prepare --input data/gsm8k-full/train.parquet --split train \
  --limit 8 --output outputs/train.jsonl
probe-int prepare --input data/gsm8k-full/test.parquet --split test \
  --limit 8 --output outputs/test.jsonl
```

以上命令通过 `--limit 8` 从完整数据集中各取前 8 条，用于快速验证流程；移除 `--limit` 即可转换完整 split。也可直接转换其他本地 GSM8K parquet：

```bash
probe-int prepare --input /path/to/train.parquet --split train \
  --limit 100 --output outputs/train.jsonl
```

支持 `question/answer`、`problem/solution`，以及 GSM8K 的 `extra_info.question/answer` 结构。标准格式：

```json
{"id":"train:0","split":"train","question":"...","solution":"...\n#### 19"}
```

必须正确标注数据来源。裸 parquet 无法自动判断 train/test；`--split` 是调用者对来源的声明，不是数据泄漏检测器。已有 split 与参数冲突会报错，SFT 导出和训练拒绝非 train 数据。实验中还应独立核对训练集与评估集的问题交集。

### 4.2 生成完整轨迹并分支探测

```bash
export MODEL=/path/to/Qwen3-1.7B
probe-int collect --model "$MODEL" --input outputs/train.jsonl \
  --output outputs/probes.jsonl --boundary step \
  --max-trajectory-tokens 8192 --max-model-len 16384
```

- `--boundary step`：按空行近似划分步骤，边界对齐**原始生成 token 的末尾**。空行若在一个 token 内，可能包含该 token 的少量后继字符；不是语义步骤解析器。
- `--boundary token --interval 256`：每 256 个原始生成 token 探测一次。区间可能跨步或截断一句话，只能解释为 token 区间。
- 两种模式都额外探测零前缀和 `</think>` 前的末尾；不会在答案区间继续探测。
- 默认每个边界一次贪心答案生成；提高抗采样噪声能力可用 `--probe-samples 5 --probe-temperature 0.7`。
- 所有分支只读同一条原始轨迹，不以探针生成的答案替换原轨迹。保存原始 `prompt_ids`、`trajectory_ids`、边界 offset 和每次探测结果，不从保存文本重新 tokenize 原始前缀。
- 如果 Qwen3 模板只输出 assistant header、未预填 `<think>`，本实现显式预填 `<think>\n`；这样零前缀分支也能合法关闭 thinking。这个预填属于采样设置，做基线对照时必须保持一致。
- 默认每条题目串行、题内探针小批处理，启用 vLLM prefix cache。大量长轨迹的成本可能很高；先从小样本确认翻转率。
- 输出保留所有轨迹；定位阶段扫描全部轨迹的相邻探针状态，不以最终答案正误筛选。

### 4.3 独立 vLLM 评估

```bash
probe-int-eval --model "$MODEL" --input outputs/test.jsonl \
  --output outputs/qwen3-1.7b-gsm8k-test.jsonl \
  --max-tokens 8192 --temperature 0
```

评估只接受 `validation`/`test` split，使用 vLLM 采样，默认生成上限 8192 tokens；结果保存原始输出、答案判定与 manifest。后续章节编号依次顺延。

### 4.4 定位可疑区间

```bash
probe-int localize --input outputs/probes.jsonl --output outputs/candidates.jsonl \
  --correct-threshold 0.8 --wrong-threshold 0.2
```

边界所有采样答案均可解析且未截断，才按采样正确率判断：`>= 0.8` 为 correct，`<= 0.2` 为 wrong，其余 uncertain。任一采样缺失答案或被截断，该边界整体为 uncertain。K=1 时即单次答案正误。`candidate.start/end` 是相对于 `trajectory_ids` 的 **0-based、左闭右开**索引，替换 `[start, end)`，保留 `[:start]`。

### 4.5 局部纠正与无特权信息续写

```bash
probe-int repair --model "$MODEL" --input outputs/candidates.jsonl \
  --output outputs/repaired.jsonl --continuation-samples 4
```

使用同一模型在独立请求中读参考解，只输出一个替换步骤，再回到原始学生 prompt 续写。纠正步骤在首个空行处停止；空输出、异常标签、超预算截断会被拒绝。默认还拒绝包含 boxed 最终答案或原样出现 gold answer 的步骤。

**答案泄漏过滤只是启发式**：会误杀中间值恰好等于最终答案的合法步骤，也不能识别所有同义/等价答案泄漏。可以人工审查被拒绝样本；`--allow-answer-in-step` 仅用于明确标注的消融，不应把它当安全的默认训练设置。这里的“无特权续写”只表示不再传入参考解，而不是纠正步骤不含参考解导出的知识。

保留失败的纠正和全部续写，避免只记录成功导致统计失真。初版每个候选只尝试一个替换步骤；默认四次续写，不声称复现论文的 32 次预算。

### 4.6 导出 SFT

```bash
probe-int build-sft --input outputs/repaired.jsonl --output outputs/sft.jsonl \
  --target prefix_step
probe-int summarize --input outputs/repaired.jsonl
```

只输出至少一次**完整、答案正确**续写的训练样本。每条包含 `input_ids` 和 `labels`，上下文及 padding 使用 `-100`，没有 privileged teacher prompt：

| target | 输入上下文（不计 loss） | 计 loss 部分 |
| --- | --- | --- |
| `prefix_step`（默认） | 问题 chat prompt | 原始前缀 + 替换步骤 |
| `step_only` | 问题 chat prompt + 原始前缀 | 替换步骤 |
| `full` | 问题 chat prompt | 原始前缀 + 替换步骤 + 首条成功续写 |

前两个目标是未完成的 assistant 片段，**不人为追加 EOS 或关闭 think**，避免把“纠正一步后停止”训练进去。完整续写使用模型原始生成 token，包括存在于输出中的终止 token。

## 5. SFT 入口

```bash
probe-int-train --model "$MODEL" --data outputs/sft.jsonl \
  --output outputs/sft_model --lora-r 16 --epochs 1
```

多 GPU 通过标准 torchrun 启动，实际执行前检查显存与预算：

```bash
torchrun --nproc_per_node=4 -m probe_int.train --model "$MODEL" \
  --data outputs/sft.jsonl --output outputs/sft_model \
  --batch-size 1 --gradient-accumulation-steps 4 --lora-r 16
```

`--lora-r 0` 为 full SFT。训练验证 tokenizer 指纹；超长样本报错而非静默截断。LoRA 输出是 adapter，后续使用本流水线采样前需要自行合并到基座。

## 6. 产物与复现

- 每个数据命令创建 `<output>.manifest.json`，保存参数、输入 SHA256、依赖版本、完成状态和输出行数。输出存在则拒绝覆盖；下游拒绝 manifest 未完成的输入。中断留下的文件不是完整运行，初版不支持自动断点续跑。
- 采样种子只能帮助复现，不能保证不同并行设置、驱动或 kernel 下 bitwise 一致。
- 中间 JSONL 含参考解与轨迹，仅供构造/分析数据；`sft.jsonl` 不导出参考解字段。
- 数值答案使用精确有理数比较；其他表达式使用可选 `math-verify`。输出必须提供完整 `\boxed{...}`；解析失败不是 wrong。
- `summarize` 同时累加采样、探针、纠正、续写的输入/输出 token 数。输入统计为提交的请求长度（每个 n-sampling 请求只记一次），不等于扣除 prefix cache 后的实际计算量。
- 仓库忽略 `outputs/`、`eval_results/`、`wandb/`。不得上传模型、私有数据、token 或机器专有路径。
