# InT 对应关系与实现来源

查阅日期：2026-10-21。

- 官方代码：<https://github.com/intervention-training/int>
- 用户指定的论文版本：<https://arxiv.org/html/2601.14209v1>
- 论文标题：*InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning*。

## 官方开源实现核对

实现参照官方 `propose_interventions.py`, `sample_guided_rollouts.py`,
`filter_guided_rollouts.ipynb` 和 `sft.sh`，不把论文流程仅凭摘要重写。
官方 intervention 阶段用 oracle solution 建立上下文，要求模型逐条核验
学生轨迹、定位第一个不可恢复的 critical error，并在错误之前提出一条
不直接泄漏答案的 intervention；随后独立采样 guided rollouts，并可筛出
成功 continuation。官方 README 对 SFT 的描述是训练
`intervention_guided_attempt`，即 problem → correct prefix + intervention。

本仓库保留这些机制作为对照基线的设计约束，但**不声称逐字或逐参数复现
官方 pipeline**：探针替换“模型读完整轨迹并找 first critical error”的选点；
它定位到相邻探针退化之间的 token span，并让 teacher 基于该 span 生成替换
步骤。因此它是 probe-guided variant，不是 InT 官方方法实现。

## 官方流水线

| 官方入口 | 职责 | 本项目对应 |
| --- | --- | --- |
| `sample_rollouts.py` + `filter_rollouts.ipynb` | 采样并筛选错误轨迹 | `collect` 保存全部；`localize` 要求最终答案明确错误 |
| `propose_interventions.py` + `parse_interventions.ipynb` | 根据参考解找错并生成一步干预 | `localize` 用答案探针翻转选区间；`repair` 只在指定区间提出替换 |
| `sample_guided_rollouts.py` + `filter_guided_rollouts.ipynb` | 在 prefix + intervention 后无参考解续写，筛正确结果 | `repair` 的独立学生请求 + `build-sft` 的成功过滤 |
| `sft.sh` / LlamaFactory | 对 `intervention_guided_attempt` 做 SFT | 本项目直接导出 token 标签并用 Transformers Trainer 训练 |
| `int.yaml` | 后续 RL | 不在初版范围内 |

论文 §3.1 描述 first-error 定位及一步 intervention，并说明剔除直接包含最终答案的干预。官方 guided-rollout 脚本在 problem + 原正确前缀 + intervention 后继续生成；其参考解不是 student continuation 的上下文。§4.1/Table 2 的推荐 SFT 配置克隆 prefix + intervention、排除 suffix；成功续写过滤对官方 README 来说是可选阶段，本项目默认启用，并将 `full` 留作显式消融。

官方 README 的步骤 3 将成功过滤标为 optional；本项目默认启用，以减少无效纠正进入训练。

## 查阅快照

通过 GitHub Contents API 读取 main 分支对应文件；以下是文件 blob SHA，不是完整仓库 commit，也不声称之后的 main 不会变化。原始脚本包含完整 prompt 和数据处理细节，发布前请再次对照上游版本：

```text
README.md                  b266e67aa7e2c5852da07c4fba7248cb7d8df611
propose_interventions.py    cb0230473c00b0d24b0a1561f8610c5843a35dfb
sample_guided_rollouts.py   c76260e193597ebbca745cc98cb09e016287f827
sft.sh                     499e68ea18603f415b88fd7674805a72e7263c58
int.yaml                   fcb1e7ba0175380eddd2164c291df89ea4abcdb2
```

本目录没有 vendoring 上游代码，没有把上游研究实现或父目录中的未提交实验打包进来。探针思路参考父项目 `scripts/probe_full_trajectory_steps.py` 的“保存原轨迹、在前缀分支关闭 think 并回答”流程，但这里单独组织 I/O、原始 token 边界与训练标签，不跨目录 import。

## 有意保留的差异

- 初版仅实现 Qwen3 风格 thinking 模板；官方实验使用 Qwen3-4B-Instruct-2507，不保证模板和推理行为相同。
- 这里把空行当作步骤近似，或按固定 token 切片；没有重现官方 numbered-step 自验证日志。
- 这里的候选是 first observed adjacent answer regression，不是 first incorrect reasoning step 的人工或形式化标签。
- 默认一个替换步骤、四次续写，且启发式拒绝疑似答案泄漏；应分别报告候选覆盖、纠正有效率与过滤损失。
- 不包含上游 hard-set 构建、RL、评估集分发或论文数值的复现。

正式公开发布时，应补齐项目作者、许可证、论文引用格式和实际验证结果。不要在未确认许可时把上游代码整体复制到新仓库。
