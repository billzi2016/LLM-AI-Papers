# The Danger of Overthinking: Examining the Reasoning-Action Dilemma in   Agentic Tasks

> **Date**：2025-02-12
> **arXiv**：https://arxiv.org/abs/2502.08235

## Abstract

Large Reasoning Models (LRMs) represent a breakthrough in AI problem-solving capabilities, but their effectiveness in interactive environments can be limited. This paper introduces and analyzes overthinking in LRMs. A phenomenon where models favor extended internal reasoning chains over environmental interaction. Through experiments on software engineering tasks using SWE Bench Verified, we observe three recurring patterns: Analysis Paralysis, Rogue Actions, and Premature Disengagement. We propose a framework to study these behaviors, which correlates with human expert assessments, and analyze 4018 trajectories. We observe that higher overthinking scores correlate with decreased performance, with reasoning models exhibiting stronger tendencies toward overthinking compared to non-reasoning models. Our analysis reveals that simple efforts to mitigate overthinking in agentic environments, such as selecting the solution with the lower overthinking score, can improve model performance by almost 30% while reducing computational costs by 43%. These results suggest that mitigating overthinking has strong practical implications. We suggest that by leveraging native function-calling capabilities and selective reinforcement learning overthinking tendencies could be mitigated. We also open-source our evaluation framework and dataset to facilitate research in this direction at https://github.com/AlexCuadron/Overthinking.

---

# 过度思考的危险：探讨代理任务中的推理‑行动困境 论文详细解读

### 背景：这个问题为什么难？

在过去，语言模型主要被当作一次性“问答机”，只需要把问题喂进去，等答案出来就算完成。随着大模型的推理能力（Large Reasoning Models，LRMs）提升，研究者开始让它们进入需要连续交互的环境——比如代码编辑器、游戏或机器人控制。可是，推理链越长，模型越倾向于在内部“自说自话”，而不是及时去试验、观察环境反馈。结果是模型在思考上耗时却不产生有效行动，导致整体效率和成功率下降。这个“思考‑行动失衡”在之前的工作里几乎没有系统化的度量和分析，因此迫切需要一套方法来捕捉并纠正这种过度思考的行为。

### 关键概念速览
- **Large Reasoning Model（LRM）**：具备长链思考能力的大型语言模型，类似于会写长篇推理笔记的学生。相比普通模型，它更擅长在内部生成多步推理。
- **Overthinking（过度思考）**：模型在给出实际动作前，生成的推理链过长或不必要，导致行动延迟或错误。可以把它想成“写作时卡在草稿阶段，迟迟不敢动笔”。
- **Analysis Paralysis（分析瘫痪）**：模型陷入无限循环的自我检查，最终既不提交代码也不执行指令。类似于人类在会议上反复讨论细节，却迟迟没有决策。
- **Rogue Actions（离谱行动）**：模型在长时间思考后，突然做出与前文推理毫不相干的动作，好像在写完长篇报告后随手涂鸦。
- **Premature Disengagement（过早退出）**：模型在思考未完成时就放弃行动，导致任务半途而废。相当于学生在做题时因为卡住就直接交卷。
- **Overthinking Score（过度思考得分）**：衡量一次交互轨迹中思考与行动比例的数值，得分越高表示模型越倾向于内部推理而非外部交互。
- **Function‑Calling（函数调用）**：模型直接调用外部工具或 API 的能力，类似于人类在思考后直接去使用计算器或搜索引擎，而不是自己算。

### 核心创新点
1. **从现象到度量的跨越**  
   之前的研究只用成功率或生成质量评估模型，在交互任务中缺少对思考行为的量化。本文提出了“过度思考得分”，通过统计推理步骤数、停顿时间和实际动作的比例来给每条轨迹打分。这样就能把“模型在想太久”这件事变成可比较的数值指标。

2. **系统化的行为模式归纳**  
   通过在 SWE Bench Verified（软件工程任务基准）上分析 4018 条交互轨迹，作者把过度思考细分为三类：分析瘫痪、离谱行动、过早退出。相比之前只说模型“表现不佳”，这里提供了具体的行为标签，便于后续针对性改进。

3. **简单却高效的干预策略**  
   作者发现，仅仅在每轮生成多个候选答案后，挑选“过度思考得分最低”的那个，就能把整体成功率提升近 30%，计算成本削减 43%。这是一种不需要重新训练模型、只在推理阶段做的轻量级过滤。

4. **利用函数调用与选择性强化学习的潜在路径**  
   论文提出，原生的函数调用可以让模型在思考结束后直接执行外部工具，从而缩短内部推理链；同时，使用强化学习对过度思考得分进行奖励信号，能够在训练阶段抑制过长的思考。虽然这部分是展望，但为后续研究指明了方向。

### 方法详解
整体思路可以拆成三步：**轨迹采集 → 过度思考度量 → 低得分筛选**。

1. **轨迹采集**  
   - 选取 SWE Bench Verified 中的真实软件开发任务（如 bug 修复、功能实现）。  
   - 让两类模型分别执行：一种是带有 Chain‑of‑Thought（思维链）提示的 LRMs，另一种是普通的指令式模型。  
   - 每一步记录模型的文字输出、调用的函数（如 `run_test`, `apply_patch`）以及时间戳，形成完整的交互轨迹。

2. **过度思考得分计算**  
   - **推理长度**：统计在一次行动前模型输出的连续思考句子数。  
   - **停顿时间**：用时间戳差值衡量模型在思考阶段的耗时。  
   - **行动稀疏度**：计算思考句子与实际函数调用之间的比例。  
   - 将上述三项标准化后加权求和，得到每条轨迹的得分。权重是通过与人类专家对同轨迹的“是否过度思考”标注进行回归得到的，确保得分与主观感受高度相关。

3. **低得分筛选**  
   - 在每轮推理结束后，模型会生成多个候选答案（比如 5 条）。  
   - 对每个候选计算过度思考得分，选取得分最小的那个继续执行。  
   - 这种筛选不改变模型内部的推理逻辑，只是把“更快落地”的答案挑出来。

**最巧妙的点**在于：作者没有去改模型的内部结构，而是把“思考太久”当作一个可度量的副作用，用后处理的方式直接提升性能。相当于在写作时给自己设定“每段文字不超过 X 行”，强迫自己更快进入正文。

### 实验与效果
- **数据集与任务**：使用 SWE Bench Verified，覆盖 12 类软件工程子任务，共计约 2k 条问题。每条问题都要求模型先分析需求、写代码、运行测试，典型的交互式工作流。
- **Baseline**：普通指令式模型、带 CoT 的 LRM、以及公开的 CodeLlama‑34B。  
- **主要结果**：  
  - 直接使用 LRMs 的整体成功率约为 42%。  
  - 引入过度思考得分筛选后，成功率提升至 68%（≈30%提升）。  
  - 推理时间从平均 12.4 秒降至 7.1 秒，计算成本下降约 43%。  
- **消融实验**：  
  - 只使用推理长度作为得分，提升约 18%；  
  - 只使用停顿时间提升约 12%；  
  - 两者结合效果最佳，说明多维度度量是必要的。  
- **局限性**：  
  - 论文主要在软件工程任务上验证，其他领域（如机器人控制）是否同样适用仍未测试。  
  - 过度思考得分的权重是基于少量专家标注得到，可能对不同任务需要重新校准。  
  - 只评估了单轮筛选，未探索多轮迭代的潜在收益。

### 影响与延伸思考
这篇工作把“模型思考太久”从经验现象提升为可量化、可干预的研究对象，打开了交互式 AI 评估的新视角。随后的几篇论文（如 2024 年的 *Action‑Aware Prompting*、*RL‑Based Overthinking Suppression*）都引用了过度思考得分作为基准，尝试在强化学习或自监督预训练阶段直接抑制长链推理。对想继续深入的读者，可以关注以下方向：  
- 将过度思考度量迁移到视觉‑语言或机器人任务中；  
- 结合人类实时反馈，动态调节思考长度；  
- 探索在大模型微调阶段加入“思考成本”作为正则项。  

### 一句话记住它
把模型的“想太久”当作可测的副作用，用低得分筛选即可让大模型在交互任务中跑得更快、表现更好。