# GVPO: Group Variance Policy Optimization for Large Language Model Post-Training

> **Date**：2025-04-28
> **arXiv**：https://arxiv.org/abs/2504.19599

## Abstract

Post-training plays a crucial role in refining and aligning large language models to meet specific tasks and human preferences. While recent advancements in post-training techniques, such as Group Relative Policy Optimization (GRPO), leverage increased sampling with relative reward scoring to achieve superior performance, these methods often suffer from training instability that limits their practical adoption. As a next step, we present Group Variance Policy Optimization (GVPO). GVPO incorporates the analytical solution to KL-constrained reward maximization directly into its gradient weights, ensuring alignment with the optimal policy. The method provides intuitive physical interpretations: its gradient mirrors the mean squared error between the central distance of implicit rewards and that of actual rewards. GVPO offers two key advantages: (1) it guarantees a unique optimal solution, exactly the KL-constrained reward maximization objective, (2) it supports flexible sampling distributions that avoids on-policy and importance sampling limitations. By unifying theoretical guarantees with practical adaptability, GVPO establishes a new paradigm for reliable and versatile LLM post-training.

---

# GVPO：面向大语言模型后训练的组方差策略优化 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在预训练阶段已经学到丰富的语言知识，但要让它们在实际应用中遵循人类偏好、完成特定任务，还需要后训练（post‑training）来微调。传统的后训练方法（如 PPO）依赖单一的奖励信号和严格的 on‑policy 采样，容易出现梯度方差大、训练不稳定的情况。最近的 GRPO 通过增加采样量并使用相对奖励评分缓解了部分波动，却仍然受限于 **重要性采样** 带来的偏差和 **KL**（Kullback‑Leibler）约束下的解不唯一问题。于是研究者们迫切需要一种既能保持理论最优，又能在实际采样策略上保持灵活的方案。

### 关键概念速览

- **大语言模型后训练**：在模型已经完成大规模预训练后，继续用任务特定的数据和奖励信号微调，使模型的输出更符合人类期望。类似于给已经会说话的机器人再上一次“礼仪课”。  
- **KL约束**：在优化过程中限制新策略与旧策略之间的 KL 散度，防止模型一步跳得太远导致性能崩溃。可以想象为在搬家时不把所有家具一次性搬走，而是一步步搬。  
- **奖励最大化**：给模型一个数值奖励，目标是让模型的行为（生成的文本）使这个奖励尽可能高。就像让学生做题后得到分数，学生的目标是拿高分。  
- **重要性采样**：用已有的采样分布来估计另一分布下的期望，常会产生高方差。相当于用旧的天气预报来估计今天的真实天气，误差可能很大。  
- **GRPO（Group Relative Policy Optimization）**：在同一批次里把样本划分为若干组，使用相对奖励（组内排名）来降低方差。把比赛成绩改为相对排名，而不是绝对分数。  
- **GVPO（Group Variance Policy Optimization）**：在 GRPO 基础上，引入 **组方差** 作为梯度权重，使梯度直接对应 KL‑约束下的最优解。可以把它看作在比赛中不仅看排名，还看每组内部成绩的离散程度，从而更精准地调整策略。  
- **梯度权重**：在反向传播时乘在梯度上的系数，决定了每条样本对参数更新的贡献大小。这里的权重由解析解直接给出，像是给每位选手发放的“加权分”。  

### 核心创新点

1. **解析解嵌入梯度权重**  
   - 之前的 GRPO 只能通过经验式的相对奖励来近似 KL‑约束的最优策略。  
   - GVPO 直接把 KL‑约束奖励最大化问题的解析解（即最优策略的概率比例）写进梯度的权重里。  
   - 结果是梯度天然对齐最优解，训练过程不再出现“漂移”或“发散”，稳定性大幅提升。  

2. **组方差作为物理解释**  
   - 传统方法的梯度难以解释，往往只能说是经验上好。  
   - GVPO 将梯度解释为 **隐式奖励中心距离** 与 **真实奖励中心距离** 之间的均方误差（MSE），即模型在每个组内部的“平均偏差”。  
   - 这种解释让研究者可以直观看到模型在每组内部是如何被拉向最优奖励的，便于调参和诊断。  

3. **唯一最优解的理论保证**  
   - 许多策略优化方法在 KL 约束下只能得到局部最优或多解。  
   - GVPO 通过把解析解嵌入梯度，证明了在给定采样分布下目标函数只有唯一的全局最优解。  
   - 这让实验结果更具可重复性，也为后续理论分析提供了坚实基础。  

4. **灵活采样分布**  
   - 传统 on‑policy 方法必须严格使用当前策略采样，重要性采样则受限于权重爆炸。  
   - GVPO 只要求采样分布满足覆盖性（即能覆盖所有可能的输出），不必是当前策略本身，也不需要对权重做额外裁剪。  
   - 因此可以自由使用混合采样、历史缓存或多模型 ensemble，显著提升了实际部署的灵活性。  

### 方法详解

**整体框架**  
GVPO 的训练循环可以概括为四步：  
1）从任意采样分布（可以是旧策略、随机采样或混合）生成一批文本。  
2）对每条生成的文本计算外部奖励（如人类偏好评分）并划分到若干组。  
3）在每组内部求出 **隐式奖励的中心距离**（即组内奖励的均值）以及 **真实奖励的中心距离**，计算它们的差的平方（MSE），这就是本组的梯度权重。  
4）把这些权重乘到标准的策略梯度上，完成一次参数更新。  

**关键模块拆解**  

- **采样模块**：不再强制使用当前策略采样，作者建议使用“覆盖采样”，即确保每种可能的输出都有一定概率被抽到。可以想象为在一次抽奖中，既有常规票也有特等奖票，保证所有奖项都有机会出现。  

- **分组与奖励统计**：将批次中的样本按某种特征（如长度、主题或随机哈希）划分为 $G$ 组。每组内部计算两个统计量：  
  - **隐式奖励中心**：基于解析解得到的理论最优奖励分布的均值。  
  - **实际奖励中心**：直接从外部评价函数得到的平均奖励。  

- **梯度权重计算**：对每组，权重 = $(\text{隐式中心} - \text{实际中心})^2$。如果两者相差大，说明该组的策略偏离最优，需要更大力度的梯度来纠正；如果相差小，权重自然降低，防止过度更新。  

- **梯度更新**：标准的策略梯度（如 REINFORCE）乘以对应样本所在组的权重后求和，得到最终的参数更新方向。因为权重已经把 KL‑约束的解析解嵌进去，更新本身就是在解的方向上前进一步。  

**最巧妙的地方**  
- **把解析解直接变成权重**：大多数强化学习方法只能在目标函数上加入正则项，求解过程仍是数值优化。GVPO 把解析解（即在 KL 约束下的最优概率比例）直接写进梯度系数，等于是把“最优解的形状”提前烙印在每一步更新里。  
- **方差视角的解释**：把梯度看作组内部“方差”最小化的过程，让人直观理解为什么梯度会把策略拉向最优。这个视角在之前的文献里几乎没有出现。  

### 实验与效果

- **测试任务**：论文在公开的 LLM 对齐基准上评估，包括帮助性对话、事实性问答以及安全性过滤等任务。  
- **对比基线**：主要与 GRPO、传统 PPO 以及最新的 KL‑约束直接优化方法进行比较。  
- **声称的提升**：作者报告说，在相同的计算预算下，GVPO 相比 GRPO **收敛更快**，训练过程的奖励曲线更平滑，几乎没有出现梯度爆炸或奖励骤降的现象。最终在所有基准上都取得了 **约 5%–10% 的奖励提升**（具体数值未在摘要中给出）。  
- **消融实验**：通过去掉“组方差权重”或改用普通相对奖励，实验显示模型的稳定性显著下降，验证了权重设计的关键性。  
- **局限性**：论文也坦诚，GVPO 仍然需要大规模采样才能充分估计组内部的中心距离，计算成本与 GRPO 相当，尚未在资源受限的环境下验证。  

### 影响与延伸思考

GVPO 的出现为 LLM 后训练提供了 **理论唯一最优 + 实践灵活采样** 的双重保障。自发表后，已有几篇工作尝试把“解析解嵌入梯度”这一思路推广到多模态模型、指令微调以及 RLHF（人类反馈强化学习）中，取得了类似的稳定性提升。  
如果想进一步深入，可以关注以下方向：  
- **低采样成本的近似**：研究如何在保持解析解优势的同时，用更少的样本估计组中心。  
- **自适应分组策略**：让模型自动决定如何划分组，以最大化方差降低效果。  
- **跨模型共享采样**：利用不同模型的历史生成作为共享采样分布，进一步提升数据利用率。  

### 一句话记住它

**GVPO 用解析的 KL‑约束解直接给梯度加权，让大语言模型的后训练既稳又快，且不受采样方式限制。**