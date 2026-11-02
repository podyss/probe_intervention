# Jev 的 CoT 前向删减课程

## 核心想法

让基础模型为数学选择题生成分步 CoT，按语义步骤切成 `C1…C5`，不在步骤内部截断。训练时从完整轨迹开始，逐阶段删除最前面的步骤，每次约删总步骤数的 1/5：

`C1…C5 → C2…C5 → C3…C5 → C4,C5 → C5 → 无 CoT`

每个阶段都要求 Jev-style 学生只输出一个选择 token。主要监督目标是完整 CoT 条件下教师对选项的概率分布；若拿不到教师 logits，则用正确选项作 one-hot 标签。最终阶段与“只监督最终正确 token”基线相同。

## 与已有 Jev 方法的重叠

**核心终点和渐进式 CoT 都已有强先例，不能把简单组合直接主张为新方法。** *General Decision Models: Benchmarking and Insights Beyond Jev* 的 Reasoning-to-Readout Self-Distillation 用模型自身生成的两条 CoT 得到推理后的候选概率分布，再训练学生不看 CoT、直接输出首个答案 token；这是“CoT → Jev 决策分布”的直接先例。更近的 *Large Decision Models* 提出 SPRD：冻结教师自生成选择任务、每题做四次推理、按投票分布蒸馏到单次读出，并过滤答案不一致样本。两者都不把 CoT 片段交给学生。

渐进式方面，ICML 2024 的 KPOD 已提出 keypoint-aware 的 rationale masking 和由易到难的渐进蒸馏；其目标仍是生成推理文本，而非 Jev 单 token 读出。当前方案与这些工作的差异只是：**学生在训练中依次看到删去前部步骤后的 CoT 后缀，并最终转为无 CoT 的选择分布预测**。这可能是有用的 Jev 专项消融，但目前不足以单独视作强 novelty；尤其要与 SII/R2R 和 SPRD 的“直接蒸馏到无 CoT”做预算匹配对照。

另外，*Releasing the Inner Jev* 的主要基线是多任务选择题 SFT；*LLM2Jev* 使用 listwise 选择损失及 KL 锚定。它们不是 CoT 删减方法，但应作为无需 CoT 课程的决策模型训练基线。

## 数据与切分

- **训练：MathQA train。** 约 29.8K 道五选一数学题，含正确选项、rationale 和运算程序；用题目及选项生成学生输入，用模型生成的 CoT 作为删减材料，并按正确答案过滤。原始 rationale 可留作分步切分的对照。只用 train，先去重，再按题目来源分组切分。
- **开发：** 从去重后的训练题中划出固定开发集，用于选择删减阶段和训练轮数；不要反复用最终测试集调参。
- **外部测试：CompMath-MCQ。** 1,500 道专家编写的高阶数学选择题，原生符合 Jev 的候选选项输入/单 token 输出形式；适合作为独立迁移测试。训练前做文本及语义去重，测试题不参与课程/超参选择。
- **同分布测试：MathQA 官方 validation/test。** 可报告 MathQA 内的效果，但不作为唯一主结论；已有数据审计指出 MathQA 官方验证/测试集可能与训练集存在近重复风险。另可把 MATH-500 转成选择题作竞赛数学补充测试，但须固定干扰项生成与验真流程。

## 最小对照

同一底模、同一训练题和总更新预算，比较：

1. 仅题目与选项 → 正确 token 监督（主基线）；
2. 完整 CoT → 正确 token 监督；
3. 按上面顺序逐步删减 CoT → 相同目标；
4. 逐步删减 CoT → 完整 CoT 教师选项分布蒸馏（若可取得 logits）。

所有方法测试时都只输入题目与选项。报告准确率、选项概率校准和不同题目难度的结果；至少运行多个随机种子。

## 主要风险

前向删减会留下推理后缀，可能缺少已删除步骤所提供的前提；尾部步骤也可能直接泄露答案。生成轨迹必须先验证答案，删除显式答案句，并抽查各删减视图是否仍可理解。若课程法不优于只监督最终 token，或只在带 CoT 的条件下有效，就不能说明它把多步能力内化进了 Jev。

## 参考

- MathQA: Amini et al., *MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms* (2019)。
- CompMath-MCQ: *The CompMath-MCQ Dataset: Are LLMs Ready for Higher-Level Math?* (2026, arXiv:2603.03334)。
- 直接重合的 Jev 方法：Duan et al., *General Decision Models: Benchmarking and Insights Beyond Jev* (2026, arXiv:2610.03935)，SII-Decision / Reasoning-to-Readout Self-Distillation；Chen et al., *Large Decision Models: Repurposing Large Language Models as Single-Pass Decision Makers* (2026, arXiv:2610.15501)，SPRD。
- 渐进 CoT 蒸馏：Feng et al., *Keypoint-based Progressive Chain-of-Thought Distillation for LLMs* (ICML 2024)。
- Jev-style SFT 与 listwise 基线：*Releasing the Inner Jev: A Simple Recipe for Open Decision Models* (2026, arXiv:2610.17205)；Li & Wagle, *LLM2Jev: LLMs Are Already Jev-Style Decision Models* (2026, arXiv:2610.02076)。
- 数据污染提醒：Calc-X 的 MathQA 数据卡建议谨慎使用 MathQA 官方 validation/test 作独立最终评估。
