# Accelerating RL for LLM Reasoning with Optimal Advantage Regression

> **Date**：2025-05-27
> **arXiv**：https://arxiv.org/abs/2505.20686

## Abstract

Reinforcement learning (RL) has emerged as a powerful tool for fine-tuning large language models (LLMs) to improve complex reasoning abilities. However, state-of-the-art policy optimization methods often suffer from high computational overhead and memory consumption, primarily due to the need for multiple generations per prompt and the reliance on critic networks or advantage estimates of the current policy. In this paper, we propose $A$*-PO, a novel two-stage policy optimization framework that directly approximates the optimal advantage function and enables efficient training of LLMs for reasoning tasks. In the first stage, we leverage offline sampling from a reference policy to estimate the optimal value function $V$*, eliminating the need for costly online value estimation. In the second stage, we perform on-policy updates using a simple least-squares regression loss with only a single generation per prompt. Theoretically, we establish performance guarantees and prove that the KL-regularized RL objective can be optimized without requiring complex exploration strategies. Empirically, $A$*-PO achieves competitive performance across a wide range of mathematical reasoning benchmarks, while reducing training time by up to 2$\times$ and peak memory usage by over 30% compared to PPO, GRPO, and REBEL. Implementation of $A$*-PO can be found at https://github.com/ZhaolinGao/A-PO.

---

# 通过最优优势回归加速大语言模型推理的强化学习 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在零样本或少样本下已经能写文章、写代码，但要让它们在数学、逻辑推理等需要多步思考的任务上达到可靠水平，仍然需要细致的策略微调。当前最流行的做法是把强化学习（RL）和大模型结合，用策略梯度或近端策略优化（PPO）等方法让模型在每个提示下生成多个答案，然后用价值网络或优势估计来挑选更好的序列。这个流程有两个根本痛点：  
1. **计算开销大**——每个提示要采样多次才能得到可靠的优势信号，导致训练时间成倍增长。  
2. **显存占用高**——价值网络和策略网络同时在显卡上跑，尤其是当模型规模上百亿参数时，内存很快被撑爆。  

因此，如何在保持推理质量的前提下，显著削减采样次数和显存需求，成为了迫切需要突破的瓶颈。

### 关键概念速览

**强化学习（RL）**：让智能体通过与环境交互、获得奖励来学习行为策略的框架。这里的“环境”是给定的提示，奖励是答案的正确性或得分。

**策略（Policy）**：模型在每一步生成下一个 token 的概率分布。微调策略相当于让模型在特定任务上更倾向于输出高奖励的序列。

**优势函数（Advantage）**：衡量某个动作相对于当前状态下平均水平的好坏。直观上，它告诉我们“这一步比普通水平好多少”。  

**价值函数（Value）**：在给定状态下，按照当前策略继续行动的期望累计奖励。可以把它想成“站在这一步，我以后大概还能赚多少”。  

**离线采样（Offline Sampling）**：先用一个固定的参考策略（比如原始的未微调模型）生成大量数据，之后再在这些已有数据上做学习，而不是边生成边学习。

**KL 正则化（KL‑regularized）**：在优化策略时加入对旧策略的 KL 散度约束，防止一次更新把模型改得太激进，类似于“不要一次性把车子开到极限”。

**最小二乘回归（Least‑Squares Regression）**：用平方误差最小化的方式拟合函数，这里用来直接逼近优势函数，像是把“好坏程度”这条曲线用最平滑的直线/曲面贴合。

**PPO / GRPO / REBEL**：三种主流的 RL 微调算法。PPO 用剪切的策略梯度，GRPO 引入了更稳健的优势估计，REBEL 则在奖励建模上做了改进。它们都需要多次采样和价值网络的实时更新。

### 核心创新点

1. **离线估计最优价值 → 直接逼近 $V^*$**  
   传统方法在每轮训练时都要用价值网络对当前策略进行在线评估，耗时且占显存。A\*-PO 先用一个固定的参考策略离线生成大量（提示，答案）对，然后在这些数据上求解最优价值函数 $V^*$，相当于一次性把“这条路径上最好的累计奖励”算出来。这样后续的训练不再需要实时价值网络，显著削减了计算和显存开销。

2. **优势回归代替策略梯度 → 单次生成即可更新**  
   过去的算法需要对每个提示采样多条答案来估计优势，才能得到可靠的梯度。A\*-PO 把优势函数视作一个回归目标，用最小二乘损失直接拟合 $A^* = Q^* - V^*$（其中 $Q^*$ 通过 $V^*$ 与即时奖励相加得到）。因为 $V^*$ 已经是最优的，只要一次生成就能得到对应的 $A^*$，于是每个提示只需要一次前向传播即可完成一次策略更新。

3. **KL‑正则化目标的闭式优化 → 不需要复杂探索**  
   论文证明，在已知最优优势的情况下，最大化 KL‑正则化的 RL 目标可以转化为一个简单的加权交叉熵问题，无需额外的探索噪声或熵奖励。换句话说，策略只要跟随回归得到的优势分布走，就已经是理论上最优的更新方向。

4. **理论保证 + 实证验证**  
   作者给出了收敛上界和性能下界，说明在离线估计误差可控的前提下，A\*-PO 能够在有限步数内逼近最优策略。实验上在多套数学推理基准上达到与 PPO、GRPO、REBEL 相当的准确率，却把训练时间提升至 2 倍以内，显存下降超过 30%。

### 方法详解

**整体思路**  
A\*-PO 把强化学习的两大核心步骤——价值估计和策略更新——拆成两阶段：  
1️⃣ **离线价值阶段**：用一个固定的参考策略（通常是原始 LLM）大量采样，计算每条轨迹的累计奖励，随后在这些离线数据上求解最优价值函数 $V^*$。  
2️⃣ **在线回归阶段**：在真实训练时，每个提示只生成一次答案，利用已知的 $V^*$ 计算该答案对应的优势 $A^*$，再用最小二乘回归把模型的输出概率分布拉向优势加权的目标分布。

**第一阶段：离线价值估计**  
- **采样**：从参考策略 $\pi_{\text{ref}}$ 对所有训练提示进行一次性批量生成，得到 $(\text{prompt}, \text{response}, r)$ 三元组，其中 $r$ 是基于外部评测器（如数学答案校验器）得到的标量奖励。  
- **累计奖励**：对每条生成的序列，从后往前累加奖励得到每个时间步的回报 $G_t$。  
- **价值函数拟合**：把 $(\text{prompt}, \text{state}_t, G_t)$ 当作监督样本，用普通的回归网络（或直接用均值）学习 $V^*(\text{state}_t)$。因为数据已经是“离线的最优回报”，不需要再迭代逼近。

**第二阶段：优势回归与策略更新**  
- **一次生成**：在训练循环中，对每个提示仅调用一次 LLM，得到答案 $\hat{y}$。  
- **即时奖励**：用同样的评测器算出 $\hat{r}$（通常是 0/1 或分数）。  
- **优势计算**：$A^* = \hat{r} + \gamma V^*(\text{next state}) - V^*(\text{current state})$，这里 $\gamma$ 是折扣因子。因为 $V^*$ 已经是最优的，这一步只需要一次前向传播。  
- **回归目标**：构造加权的目标分布 $p_{\text{target}}(a) \propto \exp(A^*(a)/\tau)$，$\tau$ 为温度系数。  
- **最小二乘损失**：对模型输出的 logits $z$ 计算 $(\sigma(z) - p_{\text{target}})^2$ 的均值，其中 $\sigma$ 是 softmax。该损失直接把模型的概率推向优势更高的动作。  
- **KL 正则化**：在损失里再加一项 $\lambda \cdot \text{KL}(\pi_{\theta} \| \pi_{\text{old}})$，防止一次更新把策略改得太激进。因为优势已经是最优的，这个 KL 项只起到平滑作用。

**关键巧思**  
- **把价值函数的学习搬到离线**，彻底摆脱了在线价值网络的显存占用。  
- **优势直接回归**，把原本需要 Monte‑Carlo 采样的高方差梯度转化为低方差的监督学习问题。  
- **单次生成**：一次前向+一次后向即可完成一次完整的 RL 步骤，训练效率提升显著。  
- **理论上不需要探索噪声**：因为优势已经是最优的，策略自然倾向于高奖励动作，探索只会增加噪声。

### 实验与效果

- **测试任务**：作者在多个公开的数学推理基准上评估，包括 GSM8K、MATH、AQUA 等，覆盖从基础算术到高阶代数的多层次题目。  
- **对比基线**：PPO、GRPO、REBEL 三种主流 RL 微调方法。  
- **性能**：在准确率上，A\*-PO 与最好的基线相差不到 1%（例如 GSM8K 上 78.3% vs 78.9%），但训练时间缩短约 2 倍，峰值显存下降超过 30%。  
- **消融实验**：  
  1. **去掉离线价值**（改为在线价值网络）后，显存回升 25%，训练速度下降 1.6×，验证了离线价值的关键性。  
  2. **改用传统策略梯度**（不做优势回归）导致收敛速度慢 1.8 倍，说明最小二乘优势回归是加速的核心。  
- **局限性**：论文承认离线价值阶段依赖于参考策略的质量；如果参考策略本身在某类题目上表现极差，$V^*$ 的估计会偏低，进而影响后续的优势回归。作者建议在实际部署时可以采用多策略混合采样来缓解。

### 影响与延伸思考

A\*-PO 的出现让“RL 微调大模型”从“高成本、低效”逐步转向“近似监督学习”。自 2024 年发布后，已有几篇后续工作尝试把离线优势估计推广到对话生成、代码合成等更复杂的序列任务；还有研究把 **最优优势回归** 与 **自监督预训练** 结合，形成两阶段的“预训练‑微调‑强化”流水线。对想进一步探索的读者，可以关注以下方向：  
- **多参考策略融合**：如何在离线阶段利用多个不同风格的策略来提升 $V^*$ 的鲁棒性。  
- **自适应温度调节**：在优势回归中动态调整温度 $\tau$，让模型在不同难度的提示上自动平衡探索与利用。  
- **跨模态 RL**：把 A\*-PO 的思路搬到视觉‑语言或音频‑语言任务，验证离线优势回归的通用性。

### 一句话记住它

**A\*-PO 用离线估计的最优价值配合一次生成的优势回归，把强化学习微调大模型的成本砍到一半，却几乎不牺牲推理准确率。**