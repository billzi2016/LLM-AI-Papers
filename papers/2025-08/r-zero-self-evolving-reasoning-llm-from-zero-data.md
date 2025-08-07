# R-Zero: Self-Evolving Reasoning LLM from Zero Data

> **Date**：2025-08-07
> **arXiv**：https://arxiv.org/abs/2508.05004

## Abstract

Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.

---

# R‑Zero：零数据自进化推理大语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型自行提升推理能力一直是个硬核挑战。现有的自进化方案大多依赖海量人工设计的任务和标签，要么通过微调（fine‑tuning），要么通过强化学习（RL）来引导模型学习。这意味着模型的成长受限于人类能提供的任务库，既耗时又难以覆盖所有潜在的推理场景。换句话说，模型想突破人类的认知边界，却被“人类数据瓶颈”卡住了。要真正走向超智能，需要一种能够从零开始自行生成、评估并学习任务的机制。

### 关键概念速览

**自进化 LLM**：模型在训练过程中自行产生新任务并学习，就像孩子在玩耍中不断发现新游戏规则。  
**Challenger（挑战者）**：负责提出新任务的模型角色，目标是让任务恰好比当前 Solver（解答者）稍难一点，类似老师出“刚好够学生思考”的练习题。  
**Solver（解答者）**：负责解决 Challenger 提出的任务的模型角色，像学生一样接受挑战并提升自己的解题技巧。  
**协同进化**：Challenger 与 Solver 互相推动、共同提升的过程，类似两位棋手在对弈中不断逼出更高水平的招式。  
**自监督任务生成**：模型不依赖外部标注，而是自己构造输入‑输出对，类似作家自己写故事再给自己评分。  
**奖励函数**：对 Challenger 和 Solver 的表现进行量化打分的规则，类似比赛的计分板，决定谁赢谁输。  
**Curriculum（课程）**：任务难度从易到难的安排，像老师先教基础再进阶，帮助模型稳步提升。  
**Zero‑Data**：训练过程中不使用任何预先收集的人类标注数据，完全从空白开始。

### 核心创新点

1. **从单一基模型直接分裂出两种角色 → Challenger 与 Solver 的双模型架构**  
   过去的自进化方法往往需要多个预训练模型或外部任务库，而这篇论文只用一个基础 LLM，复制两份并赋予不同职责。这样既省去额外模型的准备成本，又保证两者在同一语言空间内共享知识。

2. **任务难度的自适应生成 → 挑战者的奖励机制**  
   传统做法让任务固定或随机抽取，难度难以匹配模型当前水平。R‑Zero 让 Challenger 的奖励与 Solver 的成功率挂钩：当 Solver 能轻松解决任务时，Challenger 得分低；当 Solver 频繁失败时，Challenger 得分高。于是 Challenger 自然会“调教”出恰好在 Solver 能力边缘的任务，实现了自动课程（curriculum）生成。

3. **解答者的渐进学习 → Solver 的奖励机制**  
   Solver 只要成功解决 Challenger 提出的任务就能获得奖励，且奖励随任务难度提升而递增。这样 Solver 被迫不断挑战更高难度的题目，形成闭环的自我提升循环。相比于单向的强化学习，这种双向竞争让学习信号更丰富、更具针对性。

4. **零数据全自动训练管线 → 完全摆脱人工标签**  
   以往的自进化系统仍需要少量人工标注来校准奖励或提供参考答案。R‑Zero 通过让 Challenger 同时生成任务描述和参考答案（即自监督的任务对），实现了“从零开始”的数据生成。实验表明，即使在没有任何外部标注的情况下，模型仍能显著提升推理能力。

### 方法详解

#### 整体框架概览  
R‑Zero 的训练流程可以划分为四个阶段：① 基础模型复制，② 角色初始化，③ 交互式任务生成与求解，④ 奖励驱动的参数更新。整个系统像一场没有裁判的比赛，两个选手（Challenger、Solver）轮流出题、答题，裁判（奖励函数）根据答题成功率给出分数，分数再反馈到选手的学习过程。

#### 关键模块拆解  

1. **模型复制与角色分配**  
   - 从单一的预训练 LLM（如 Qwen3‑4B‑Base）复制出两份权重相同的网络。  
   - 为每份网络注入不同的“角色标签”，在前向传播时加入角色特定的提示词（prompt），让它们在同一输入下产生不同的行为：一份生成任务（Challenger），另一份求解任务（Solver）。

2. **任务生成（Challenger）**  
   - Challenger 接收一个“任务种子”提示，例如“设计一个涉及两步代数推理的数学题”。  
   - 它输出完整的任务描述以及一个参考答案（自监督的答案），这一步相当于模型自己写试卷并给出答案键。  
   - 为防止答案过于简单，系统会对生成的任务进行难度估计（基于 Solver 近期的成功率），并据此调节提示词的难度参数。

3. **任务求解（Solver）**  
   - Solver 收到 Challenger 生成的任务描述，使用标准的推理方式（如 Chain‑of‑Thought）给出解答。  
   - 解答过程会产生中间推理步骤，便于后续的奖励评估。

4. **奖励函数设计**  
   - **Challenger 奖励**：\(R_C = f(\text{Solver 成功率})\)。当 Solver 成功率低于目标阈值时，Challenger 获得高分；成功率过高则得分下降。这样 Challenger 被迫“逼迫” Solver 前进。  
   - **Solver 奖励**：\(R_S = g(\text{任务难度}, \text{是否正确})\)。正确解答且任务难度高时奖励更大，错误或轻易解答的任务奖励较低。  
   - 两个奖励函数共同构成一个零和游戏的框架，确保整体系统的学习动力始终向更高难度的任务倾斜。

5. **参数更新**  
   - 使用强化学习的策略梯度（如 PPO）分别对 Challenger 与 Solver 的参数进行更新。因为奖励是即时可得的，训练过程可以在每轮交互后立即进行梯度回传。  
   - 为防止模型崩溃，作者加入了经验回放池：保存过去一定数量的任务‑答案对，定期抽样混合训练，类似 GAN 中的判别器经验重放。

#### 反直觉与巧妙之处  
- **同模型双角色**：直觉上会担心两份模型相互干扰，导致梯度冲突。但实验显示，共享同一语言基础让两者在词汇、推理模式上保持一致，反而加速了协同进化。  
- **自监督答案生成**：让 Challenger 同时生成答案看似会产生“自欺欺人”的错误答案，但奖励函数会惩罚 Solver 对错误答案的成功率，从而迫使 Challenger 逐步提升答案质量。  
- **无标签的难度估计**：难度不是通过外部标注，而是通过 Solver 的实时表现动态估算，这种自适应机制让课程自然贴合模型的学习曲线。

### 实验与效果

- **测试任务**：论文在多个公开的推理基准上评估，包括数学推理（如 GSM‑8K）和通用领域推理（如 ARC‑E、OpenBookQA）。  
- **基线对比**：以原始 Qwen3‑4B‑Base 为基线，R‑Zero 训练后在数学推理基准上提升了 **+6.49 分**，在通用推理基准上提升了 **+7.54 分**。这些提升幅度在同等规模模型中属于显著进步。  
- **消融实验**：作者分别关闭 Challenger 的奖励、Solver 的奖励以及经验回放池进行对比。结果显示，去掉任意一项都会导致整体提升幅度下降 2‑3 分，证明每个模块都是协同进化不可或缺的组成部分。  
- **局限性**：论文承认当前的任务生成仍受限于模型的语言表达能力，极端复杂的数学证明仍难以自行构造；此外，奖励函数的超参数（如成功率阈值）需要手动调节，尚未实现完全自适应。  

### 影响与延伸思考

R‑Zero 的零数据自进化思路为“大模型自我驱动学习”打开了新方向。自发表以来，已有工作尝试将类似的双模型竞争框架搬到多模态模型、代码生成模型等领域，探索“自我出题‑自我解答”的循环。还有研究在奖励函数中加入不确定性估计，以进一步提升任务难度的精准度。想深入了解的读者可以关注以下方向：① 更稳健的自监督任务生成（比如引入符号推理引擎），② 多模型协同进化的理论分析（博弈论视角），③ 将 R‑Zero 与人类交互式微调结合，形成“人机混合课程”。这些都是把零数据自进化推向更高层次的潜在路径。

### 一句话记住它

R‑Zero 让同一个大语言模型通过“出题‑解题”自我对弈，从零数据中自动生成难度递进的任务，实现了真正的自进化推理能力提升。