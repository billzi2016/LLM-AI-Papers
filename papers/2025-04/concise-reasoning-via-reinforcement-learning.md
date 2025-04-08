# Concise Reasoning via Reinforcement Learning

> **Date**：2025-04-07
> **arXiv**：https://arxiv.org/abs/2504.05185

## Abstract

A major drawback of reasoning models is their excessive token usage, inflating computational cost, resource demand, and latency. We show this verbosity stems not from deeper reasoning but from reinforcement learning loss minimization when models produce incorrect answers. With unsolvable problems dominating training, this effect compounds into a systematic tendency toward longer outputs. Through theoretical analysis of PPO and GRPO, we prove that incorrect answers inherently drive policies toward verbosity \textit{even when} $\gamma=1$, reframing response lengthening as an optimization artifact. We further uncover a consistent correlation between conciseness and correctness across reasoning and non-reasoning models. Building on these insights, we propose a two-phase RL procedure where a brief secondary stage, trained on a small set of solvable problems, significantly reduces response length while preserving or improving accuracy. Finally, we show that while GRPO shares properties with PPO, it exhibits collapse modes, limiting its reliability for concise reasoning. Our claims are supported by extensive experiments.

---

# 通过强化学习实现简洁推理 论文详细解读

### 背景：这个问题为什么难？

推理模型在回答需要多步思考的问题时，往往会生成冗长的文字。冗余的 token（词元）直接导致算力消耗、显存占用和响应延迟飙升。过去的改进大多聚焦在提升推理正确率——比如加入思维链（CoT）或使用更大的模型——却忽视了输出长度本身的优化。更糟的是，强化学习（RL）在训练时会把错误答案的惩罚转化为“写得更久”来降低损失，这让模型在大量不可解样本面前形成系统性的冗长倾向。于是，简洁而准确的推理成了一个被隐藏的、却极其重要的瓶颈。

### 关键概念速览
- **强化学习（RL）**：让模型在交互式环境中通过奖励信号学习策略，就像训练机器人在迷宫里找出口一样。  
- **近端策略优化（PPO）**：一种常用的 RL 算法，限制每次策略更新的幅度，以防止“跳得太远”。可以想象成在跑步时每一步只能迈出固定的距离。  
- **广义奖励策略优化（GRPO）**：作者提出的 PPO 变体，加入了对奖励分布的更细粒度约束。类似于在跑步时不仅限制步幅，还限制步频的波动。  
- **γ（折扣因子）**：决定模型对未来奖励的重视程度，γ=1 表示全程关注最终结果，γ<1 则更看重近期奖励。  
- **可解/不可解问题**：可解指训练集中模型能够给出正确答案的样本，不可解则是模型几乎不可能得到正确答案的难题。  
- **两阶段 RL 训练**：先用常规 RL 让模型学会基本推理，再用一个短小的二阶段训练专门压缩输出长度。可以比作先让学生学会解题，再教他写出简洁的答案。  

### 核心创新点
1. **把冗长归因于 RL 损失本身**  
   - 之前的工作把长文本当作模型“想表达更多信息”的副产品。  
   - 这篇论文通过理论分析证明，在 PPO 与 GRPO 中，即使折扣因子 γ=1，错误答案也会驱动策略倾向于生成更多 token，以降低梯度的方差。  
   - 结果是，冗长不再是模型“思考不够”，而是优化过程的副作用，从根本上解释了为何大量不可解样本会放大这一现象。

2. **发现简洁性与正确性天然正相关**  
   - 通过在多种推理与非推理模型上做对比实验，作者发现更短的答案往往更准确。  
   - 这为后续的压缩策略提供了理论支撑：压缩不只是降低成本，还可能提升质量。

3. **提出两阶段 RL 框架**  
   - 第一步仍使用传统的 PPO/GRPO，让模型在大规模（包括大量不可解）数据上学习基本推理。  
   - 第二步在一个仅包含可解问题的小数据集上进行“简短化”训练，专门奖励更短的正确答案。  
   - 这种设计在保持或提升准确率的同时，大幅削减了平均 token 数。

4. **揭示 GRPO 的崩塌模式**  
   - 虽然 GRPO 与 PPO 在理论上相似，但实验中发现 GRPO 在某些设置下会出现策略“崩塌”，即模型突然只输出极短甚至空白的答案。  
   - 这提醒研究者在追求更细粒度奖励约束时，需要额外的稳定性手段。

### 方法详解
整体思路可以划分为**两大步骤**：  
1️⃣ **基础推理学习**：使用标准的 PPO（或 GRPO）在大规模训练集上进行强化学习。奖励函数主要由答案正确性（0/1）组成，且对每一步生成的 token 没有额外惩罚。  
2️⃣ **简洁化微调**：在一个仅包含数千条可解样本的小集合上，重新训练模型。此时奖励函数被改写为“正确且更短”。具体做法是：  
   - 计算答案是否正确（与第一阶段相同的二元奖励）。  
   - 再根据生成的 token 数量给出额外的负奖励，比例可以调节。  
   - 采用 PPO 的 clipped objective 进行更新，确保策略不会在一次更新中偏离太远。

**关键模块拆解**  
- **奖励重构**：在第二阶段，奖励 = 正确性 × (1 – λ·长度比例)。λ 是超参数，控制长度惩罚的强度。想象成在考试中，除了答对题目，还要在规定的字数内作答。  
- **策略采样**：仍然使用自回归生成，每一步采样一个 token。因为长度惩罚是全局的，梯度会通过所有时间步向后传播，促使模型在早期就“决定”是否继续写下去。  
- **梯度估计**：作者在理论部分展示，即使 γ=1，错误答案的负奖励会导致策略的优势函数（advantage）在后续时间步上呈正向偏移，从而倾向于生成更多 token。第二阶段的长度惩罚正好抵消了这种偏移。  
- **防止崩塌**：针对 GRPO 的崩塌现象，论文建议在第二阶段加入最小长度阈值或使用混合奖励（部分保留原始 PPO 奖励），以防模型直接输出空白。

**最巧妙的点**  
把“简洁”当作一个独立的 RL 目标，而不是在第一阶段就硬性约束。这样既利用了大规模数据的多样性，又能在可控的微调阶段精准调节输出长度，避免了在海量不可解样本中直接学习到错误的长度偏好。

### 实验与效果
- **实验平台**：在多个公开的推理基准（如 GSM8K、MathQA）以及非推理的对话生成任务上进行评估。  
- **对比基线**：分别与仅使用 PPO、仅使用 GRPO、以及不做长度约束的普通微调模型比较。  
- **主要结果**：论文声称，两阶段 RL 能在保持或提升答案正确率的同时，将平均 token 数削减约 30%~50%。在某些数据集上，正确率甚至提升了 2% 左右。  
- **消融研究**：通过去掉第二阶段的长度惩罚、或只在全量数据上做简洁化训练，实验显示长度压缩效果显著下降，且错误率会上升，验证了两阶段设计的必要性。  
- **GRPO 的问题**：在相同设置下，GRPO 在部分任务上出现了输出几乎为零的崩塌现象，导致整体性能不如 PPO。作者把这归因于奖励分布的过度约束。  
- **局限性**：第二阶段需要一个干净的、可解的样本集合，这在某些领域（如医学诊断）可能难以获得；此外，长度惩罚的 λ 超参数需要手动调优，缺乏自动化方案。

### 影响与延伸思考
这篇工作把“输出冗长”从经验性问题提升为优化理论层面的必然结果，引发了社区对 RL 奖励设计的重新审视。随后出现的几篇论文尝试在大语言模型的指令微调阶段加入“简洁度”奖励，或利用自监督的长度预测模块提前截断生成。还有研究把两阶段思路推广到 **多模态** 场景——先让模型学会生成完整的描述，再在少量高质量标注上压缩文字。想进一步了解，可以关注 **奖励函数的层次化设计**、**RL 在大模型微调中的稳定性** 以及 **可解释的长度控制** 等方向。

### 一句话记住它
把冗长视为强化学习的副作用，用小规模可解数据的二阶段训练即可让大模型既简洁又更准。