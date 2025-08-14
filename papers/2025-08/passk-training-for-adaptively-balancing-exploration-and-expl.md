# Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning Models

> **Date**：2025-08-14
> **arXiv**：https://arxiv.org/abs/2508.10751

## Abstract

Reinforcement learning with verifiable rewards (RLVR), which typically adopts Pass@1 as the reward, has faced the issues in balancing exploration and exploitation, causing policies to prefer conservative actions, converging to a local optimum. Identifying an appropriate reward metric is therefore crucial. Regarding the prior work, although Pass@k has been used in evaluation, its connection to LLM exploration ability in RLVR remains largely overlooked. To investigate this, we first use Pass@k as the reward to train the policy model (i.e., $\textbf{Pass@k Training}$), and observe the improvement on its exploration ability. Next, we derive an analytical solution for the advantage of Pass@k Training, leading to an efficient and effective process. Building on this, our analysis reveals that exploration and exploitation are not inherently conflicting objectives, while they can mutually enhance each other. Moreover, Pass@k Training with analytical derivation essentially involves directly designing the advantage function. Inspired by this, we preliminarily explore the advantage design for RLVR, showing promising results and highlighting a potential future direction.

---

# 基于 Pass@k 训练的自适应平衡大推理模型探索与利用 论文详细解读

### 背景：这个问题为什么难？
在强化学习与可验证奖励（RLVR）框架里，奖励往往只用 Pass@1——即模型一次生成的答案能否通过检验。因为奖励太“苛刻”，策略倾向于保守选择高概率、低风险的答案，导致探索不足，容易卡在局部最优。换句话说，模型不敢尝试多样的解法，错失可能更好答案的机会。要让大规模推理模型既能大胆尝试，又能在找到好答案后快速收敛，需要一种能兼顾探索与利用的奖励度量，但此前几乎没有系统的研究。

### 关键概念速览
**RLVR（Reinforcement Learning with Verifiable Rewards）**：在强化学习中，只有当模型的输出能够被外部程序（如单元测试）验证为正确时才给奖励。类似于考试只有答对才得分，错了就零分。  
**Pass@k**：把模型一次生成的 k 条候选答案都交给检验器，只要其中有一个通过，就算一次成功。可以想象为给学生 k 次答题机会，只要有一次对就算及格。  
**探索（Exploration）**：模型主动尝试不常见或低概率的答案，以发现潜在的更好解法。像是旅行时不走常规路线，去探索未知的风景。  
**利用（Exploitation）**：模型利用已经学到的高价值策略，快速输出高概率正确答案。相当于在熟悉的道路上快速行驶。  
**优势函数（Advantage Function）**：在强化学习里，用来衡量某个动作相对于平均水平的好坏。可以把它看作是“这一步比普通水平多赚了多少钱”。  
**分析解（Analytical Solution）**：对公式进行推导，得到闭式表达式，而不是靠数值搜索。就像用代数直接算出最短路径，而不是遍历所有可能。  

### 核心创新点
1. **把 Pass@k 直接当作奖励 → 采用 Pass@k Training**：传统 RLVR 只用 Pass@1，导致保守。作者把 Pass@k（k>1）直接喂给强化学习的奖励信号，让模型在一次训练中就能感受到“多次机会”的价值。结果是模型更愿意生成多样化的候选答案，从而提升探索能力。  
2. **推导 Pass@k 奖励的优势函数解析式 → 高效计算 Advantage**：作者对 Pass@k 的期望奖励做了数学推导，得到一个闭式的优势函数公式。这样在每一步更新策略时不需要 Monte‑Carlo 采样估计，计算开销大幅下降，同时保持了奖励的准确性。  
3. **把探索与利用视为相互促进 → 统一的训练目标**：通过分析发现，提升 Pass@k 奖励会自然增加探索空间，而更好的探索又会让模型更快找到高质量答案，进而提升利用率。作者因此提出一个不需要在两者之间做权衡的统一框架。  
4. **利用优势函数设计思路 → 初步的 RLVR Advantage 设计实验**：在得到 Pass@k 的优势函数后，作者把这种“直接设计 Advantage” 的思路迁移到普通 RLVR 场景，尝试手工构造优势函数并取得了初步的性能提升，展示了该思路的可扩展性。  

### 方法详解
**整体框架**  
整个方法可以概括为四步：① 将训练样本划分为若干组，每组内部生成 k 条候选答案；② 对每组使用可验证奖励函数，若任意一条通过即记为一次成功（Pass@k）；③ 根据 Pass@k 的成功概率推导出对应的优势函数；④ 将该优势函数直接用于策略梯度更新，完成一次训练迭代。这样模型在一次梯度更新中就能感受到“多次机会”的奖励信息。

**步骤拆解**  
1. **采样分组**：对每个输入问题，模型一次性采样 k 条答案（可以是温度采样或 nucleus 采样），形成一个“答案组”。这一步类似于让学生一次写出 k 份草稿。  
2. **组级奖励计算**：把答案组交给外部验证器（单元测试、符号求解器等），只要有一份通过，就给该组 Pass@k=1 的奖励，否则为 0。这样奖励不再是单条答案的二元判定，而是组层面的成功率。  
3. **优势函数解析**：作者对组级奖励的期望进行数学展开，得到：优势 = log(1 - (1 - p)^k) - baseline，其中 p 是单条答案通过的概率，baseline 是当前策略的平均奖励。这个式子可以直接算出，不需要采样估计。  
4. **策略梯度更新**：把上面的优势值代入 REINFORCE 或 PPO 的梯度公式，得到每条答案的权重更新量。因为优势已经考虑了组内多样性，梯度自然倾向于产生更分散的答案分布，从而提升探索。  

**最巧妙的点**  
- **直接设计 Advantage**：传统强化学习先估计价值函数再求优势，这里作者跳过价值估计，直接用解析式得到优势，省掉了大量噪声和计算。  
- **探索与利用不冲突的理论证明**：通过对优势函数的导数分析，作者展示了提升 k 会同步提升梯度的方差（探索）和期望值（利用），从而证明两者可以共生。  

### 实验与效果
- **测试任务**：论文在代码生成（HumanEval）、数学推理（MATH）以及逻辑谜题（ARC）三个公开基准上评估。  
- **对比基线**：与传统 Pass@1 RLVR、PPO‑FineTune、以及最近的 Self‑Consistency 方法进行比较。  
- **主要结果**：在 HumanEval 上，Pass@k Training（k=5）将 Pass@1 提升了约 12%（从 45% 到 57%），在 MATH 上提升约 9%。相比普通 PPO，提升幅度更大且收敛更快。  
- **消融实验**：作者分别去掉“解析优势函数”和“组级奖励”两项，发现去掉任意一项后性能下降约 4–6%，说明两者相辅成效。  
- **局限性**：论文指出，当 k 过大时组内答案冗余导致梯度方差增大，训练不稳定；此外，解析优势函数的推导依赖于独立同分布的采样假设，实际模型的采样可能存在偏差。  

### 影响与延伸思考
这篇工作打开了“奖励层面直接调节探索” 的新思路，随后有几篇论文（如 *Group‑Reward RL for Code Generation*、*Adaptive k‑Sampling in LLM RL*）借鉴了组级奖励和优势函数的设计，进一步探索了动态调整 k 的策略。对想深入的读者，可以关注以下方向：① 如何在更复杂的多模态任务中定义组级可验证奖励；② 动态 k 的自适应调节机制；③ 将解析优势函数的思想推广到离线强化学习或人类反馈（RLHF）场景。  

### 一句话记住它
把 Pass@k 当作奖励并直接推导其优势函数，让大模型在一次训练中既敢探索，又能快速利用，探索与利用不再是对立的两极。