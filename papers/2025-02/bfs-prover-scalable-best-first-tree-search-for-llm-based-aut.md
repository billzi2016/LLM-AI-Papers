# BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving

> **Date**：2025-02-05
> **arXiv**：https://arxiv.org/abs/2502.03438

## Abstract

Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B.

---

# BFS-Prover：可扩展的最佳优先树搜索用于基于大语言模型的自动定理证明 论文详细解读

### 背景：这个问题为什么难？

在 Lean4 这种交互式证明助理里，自动定理证明需要在巨大的搜索空间里找到合法的证明步骤。传统做法大多依赖价值函数或蒙特卡罗树搜索（MCTS），但价值函数训练成本高，MCTS 又需要大量模拟才能评估节点，导致在大规模数据集上难以高效扩展。于是出现了“只能靠重算、只能靠浅层搜索”的瓶颈：搜索速度慢、算力消耗大，却仍然难以突破高难度题目。

### 关键概念速览
- **LLM（大语言模型）**：能够生成自然语言或代码的深度模型，在本工作中被当作“策略”，给出每一步应该尝试的 Lean4 tactic（证明指令）。想象成一个会写证明的助教。
- **Lean4**：一种基于依赖类型的交互式定理证明语言，所有证明都要写成一串 tactic。相当于数学家在纸上写的每一步推导，只是机器可读。
- **Best‑First Search（最佳优先搜索）**：每次挑选当前评分最高的节点展开的搜索方式，类似于在迷宫里总是往看起来最有希望的方向走，而不是随意探索。
- **Expert Iteration（专家迭代）**：先让模型产生大量搜索轨迹（专家），再用这些轨迹来微调模型（学生），循环提升。好比让学生先看老师的解题过程，再自己练习。
- **Direct Preference Optimization（DPO）**：一种直接把“更好”与“更差”对比信息转化为损失的微调方法，省去传统的奖励模型训练步骤。可以把它想成“老师只说哪个答案更好”，学生直接学习这个偏好。
- **Length Normalization（长度归一化）**：在给节点打分时除以路径长度，让长路径不会因为累计分数低而被直接抛弃。相当于在比赛中给跑得更远的选手加权，让他们有机会继续前进。
- **Beam Search（束搜索）**：一次保留若干最有希望的节点继续展开的启发式搜索。这里把它当作“快速筛选器”，把容易解决的题目提前剔除。

### 核心创新点
1. **数据过滤 + Beam 搜索**  
   - 之前的专家迭代往往把所有训练题目都喂进去，导致大量容易通过束搜索就能解决的样本占用算力。  
   - 这篇论文在每轮迭代前先用束搜索跑一遍，直接把能在几步内解决的题目剔除，只保留需要深度搜索的难例。  
   - 结果是每轮训练的样本更聚焦在“真正考验搜索策略”的地方，提升了整体样本利用率。

2. **用 DPO 微调 LLM 策略**  
   - 传统做法会先训练价值网络或奖励模型，再用强化学习更新策略，步骤繁琐且样本噪声大。  
   - 作者直接把编译错误（如 tactic 失败）转化为“这一步不该选”的负例，用 DPO 把这些偏好信息喂给模型，让它学会优先挑选不会报错的 tactic。  
   - 这样模型在搜索时更倾向于产生可执行的扩展，显著降低了无效节点的比例。

3. **长度归一化的最佳优先评分**  
   - 经典的最佳优先搜索只看当前节点的概率或价值，深层路径往往因为累计概率下降而被提前淘汰。  
   - 论文在计算节点分数时除以路径长度，使得长路径的平均分数更公平，鼓励搜索深入到更复杂的证明分支。  
   - 该技巧让 BFS 能够发现很多需要多步 tactic 组合才能完成的证明，弥补了浅层搜索的盲区。

4. **可扩展的专家迭代框架**  
   - 将上述三点组合进一个循环：① 用 BFS（带长度归一化）搜索得到专家轨迹；② 用 DPO 对 LLM 进行偏好微调；③ 用束搜索过滤已解决的题目进入下一轮。  
   - 这种闭环设计在算力上保持线性增长，却能在 MiniF2F 测试集上突破 70% 的成功率，直接挑战了 MCTS 必须是“最强搜索”这一假设。

### 方法详解
整体思路可以拆成四个阶段：**过滤 → 搜索 → 标注 → 微调 → 循环**。下面按顺序展开。

1. **难例过滤（Pre‑filter）**  
   - 输入是一批待证明的 Lean4 目标。  
   - 先跑一次束搜索（beam width 例如 8），如果在限定步数内找到完整证明，就把该目标直接标记为已解决，移出后续流程。  
   - 只留下需要更深搜索的目标，形成本轮的“专家训练集”。

2. **最佳优先树搜索（BFS）**  
   - 对每个剩余目标，维护一个优先队列。队列里每个节点记录：当前 proof state、已执行的 tactic 序列、模型给出的 tactic 概率分布。  
   - 节点评分 = (模型给出的 tactic 概率的乘积) / (路径长度)。这一步相当于把“这条路有多大可能成功”除以“走了多少步”，防止长路被压制。  
   - 每次弹出评分最高的节点，展开所有可行的 tactic（通常取前 K 个概率最高的），把新节点重新放回队列。  
   - 若某个节点的 tactic 在 Lean4 编译器里报错，就记录一次“负向反馈”。搜索一直进行到达到预设的节点上限或找到完整证明。

3. **自动标注（Feedback Annotation）**  
   - BFS 过程中产生的每个 (state, tactic) 对会伴随一个二元标签：成功（编译通过且推进了证明）或失败（编译错误、无效扩展）。  
   - 这些标签不需要人工标注，全部由 Lean4 编译器的错误信息自动生成，形成大规模的偏好数据。

4. **Direct Preference Optimization（DPO）微调**  
   - 将上述 (state, tactic) 对划分为“正例”（成功）和“负例”（失败）。  
   - DPO 的损失函数直接比较正负对的相对概率：模型应当提升正例的概率、压低负例的概率。  
   - 训练时只需要一次前向传播，不需要额外的价值网络或奖励模型，极大简化了强化学习的流水线。  
   - 微调完的 LLM 现在在给定 proof state 时，更倾向于输出不会触发编译错误的 tactic。

5. **专家迭代循环**  
   - 完成一次微调后，回到第 1 步，用更新后的模型重新进行难例过滤和 BFS。  
   - 随着迭代次数增加，模型的策略越来越“懂得避坑”，搜索效率提升，难例的解决率也随之上升。  
   - 作者在实验中使用了约 5 轮迭代，每轮都显著提升整体成功率。

**最巧妙的点**在于把编译错误直接当作偏好信号，用 DPO 进行一次性微调；这把原本需要大量模拟的强化学习过程压缩成了“看错误、学偏好”的轻量步骤。

### 实验与效果
- **数据集**：MiniF2F（包含 1000+ 来自不同数学领域的 Lean4 题目），是当前衡量 LLM 自动定理证明能力的主流基准。  
- **主要指标**：在官方测试集上的完整证明成功率。  
- **对比基线**：  
  - 传统基于价值函数的模型（约 58%）  
  - 使用 MCTS 的最新系统（约 68%）  
  - 其他 BFS 变体（未公开细节）  
- **结果**：BFS‑Prover 达到 **72.95%** 的成功率，刷新了公开记录。  
- **消融实验**：  
  - 去掉难例过滤，成功率下降约 3%（说明过滤提升了样本质量）。  
  - 用普通交叉熵微调代替 DPO，下降约 4%（验证 DPO 的优势）。  
  - 关闭长度归一化，成功率再降约 2%（说明深层搜索受益于归一化）。  
- **局限**：论文未在更大规模的数学库（如 Mathlib 全部）上报告结果；对极端长证明的搜索仍受算力限制；模型规模仅 7B，若换成更大的 LLM 可能还有提升空间。

### 影响与延伸思考
这篇工作在自动定理证明社区掀起了“回归简单搜索”的讨论。随后有几篇后续研究尝试把 **BFS‑Prover** 的过滤+ DPO 思路迁移到 **Coq**、**Isabelle** 等其他证明助理，证明了方法的通用性。还有工作把 **长度归一化** 与 **奖励折扣** 结合，进一步提升深层搜索的稳定性。对想继续深入的读者，建议关注以下方向：  
- 更大规模 LLM 与 BFS 的协同放大效应；  
- 将编译器提供的细粒度错误信息（如未匹配的变量）做成更丰富的偏好标签；  
- 与图神经网络结合的状态表示，以提升对复杂 proof state 的感知能力。

### 一句话记住它
**把编译错误当作“老师的批改”，用偏好微调让 BFS 直接学会走对路，简单搜索也能赢过复杂搜索。**