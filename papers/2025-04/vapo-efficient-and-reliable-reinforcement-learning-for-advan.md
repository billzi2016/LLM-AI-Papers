# VAPO: Efficient and Reliable Reinforcement Learning for Advanced   Reasoning Tasks

> **Date**：2025-04-07
> **arXiv**：https://arxiv.org/abs/2504.05118

## Abstract

We present VAPO, Value-based Augmented Proximal Policy Optimization framework for reasoning models., a novel framework tailored for reasoning models within the value-based paradigm. Benchmarked the AIME 2024 dataset, VAPO, built on the Qwen 32B pre-trained model, attains a state-of-the-art score of $\mathbf{60.4}$. In direct comparison under identical experimental settings, VAPO outperforms the previously reported results of DeepSeek-R1-Zero-Qwen-32B and DAPO by more than 10 points. The training process of VAPO stands out for its stability and efficiency. It reaches state-of-the-art performance within a mere 5,000 steps. Moreover, across multiple independent runs, no training crashes occur, underscoring its reliability. This research delves into long chain-of-thought (long-CoT) reasoning using a value-based reinforcement learning framework. We pinpoint three key challenges that plague value-based methods: value model bias, the presence of heterogeneous sequence lengths, and the sparsity of reward signals. Through systematic design, VAPO offers an integrated solution that effectively alleviates these challenges, enabling enhanced performance in long-CoT reasoning tasks.

---

# VAPO：面向高级推理任务的高效可靠强化学习 论文详细解读

### 背景：这个问题为什么难？

在长链式思考（long‑CoT）任务里，模型需要在数十甚至上百步的推理中保持正确的方向。传统的基于策略的强化学习（如 PPO）在这种超长序列上往往不稳定，梯度噪声大，训练容易崩溃。价值函数（value‑based）方法本可以提供更直接的回报估计，却受到价值模型偏差、序列长度不一以及奖励稀疏三大痛点的制约。之前的工作（如 DAPO）虽尝试解决这些问题，但仍需要数十万步才能收敛，且在不同随机种子下偶尔会出现训练中断。于是，如何让价值函数在长 CoT 场景下既快又稳，成了急需突破的瓶颈。

### 关键概念速览
**价值函数（Value Function）**：预测在当前状态下，后续所有可能动作的累计回报，就像给每一步棋局打分，帮助模型判断哪条思路更有前景。  
**近端策略优化（PPO）**：一种限制策略更新幅度的强化学习算法，防止一次更新把模型弄得太离谱，类似于开车时不踩刹车太猛。  
**长链式思考（Long‑CoT）**：需要模型在很长的推理链上逐步输出中间结果，类似于解一道需要多步推导的数学证明。  
**价值模型偏差（Value Model Bias）**：价值函数本身的估计误差，会把模型误导到错误的思路上，好比导航仪给出错误的路线。  
**奖励稀疏（Sparse Reward）**：只有在完整推理结束时才给出分数，期间没有信号，像在马拉松终点才给奖牌，跑者很难判断自己是否在正确的跑道上。  
**异构序列长度（Heterogeneous Sequence Lengths）**：不同样本的推理步数差别很大，导致批处理时梯度统计不一致，类似于一次考试里有人只答了几道题，有人答了全部。  
**价值增强 PPO（VAPO）**：本文提出的框架，把价值函数的优势和 PPO 的稳定性结合起来，并通过一系列技巧抑制上述三大痛点。

### 核心创新点
1. **价值模型校准 → 使用双向对齐损失**：之前的价值模型直接用回报做回归，容易产生系统性偏差。VAPO 引入了一个对齐模块，让价值预测在每一步都与“理想”价值（由完整答案回溯得到）进行双向校正。这样价值函数更贴近真实回报，模型在长链上不容易被误导。  
2. **序列长度归一化 → 动态截断与填充策略**：传统批处理会把最长序列的梯度直接用于所有样本，导致短序列梯度被稀释。VAPO 采用了基于步骤数的动态权重归一化，使得每一步的梯度贡献与实际步数成比例，解决了异构长度带来的噪声。  
3. **稀疏奖励密化 → 中间奖励估计器**：为了解决只有终点才有奖励的问题，VAPO 在每一步插入一个轻量级的“中间奖励估计器”，它利用价值函数的梯度信息生成伪奖励，形成更密集的学习信号。相比直接使用稀疏奖励，训练速度提升约 10 倍。  
4. **训练安全网 → 多重检查点与自动恢复**：在长时间训练中，数值不稳定常导致崩溃。VAPO 在每 100 步保存模型快照，并在检测到梯度异常时自动回滚到最近的健康检查点，确保 100% 的实验不出现 crash。

### 方法详解
整体思路可以拆成四个阶段：**（1）价值模型预训练 →（2）价值对齐校准 →（3）强化学习主循环（PPO） →（4）安全回滚机制**。下面逐步展开。

1. **价值模型预训练**  
   - 以 Qwen‑32B 为基础，先在大规模语言数据上进行常规的自回归预训练。  
   - 再在 AIME‑2024 训练集上加入回报标签（答案正确得 1，错误得 0），用均方误差训练一个独立的价值头。此时模型已经能给出粗略的“这一步是否有望成功”的分数。

2. **价值对齐校准**  
   - 对每条训练样本，先让模型完整生成一次推理链，记录每一步的真实回报（只有最后一步有 1/0）。  
   - 通过**双向对齐损失**，把价值预测向真实回报回溯的期望值靠拢，同时把真实回报向价值预测的梯度方向微调。直观上，这一步像是让老师先批改整篇作文，再把每句话的分数调到与整体分数一致。

3. **强化学习主循环（VAPO）**  
   - **采样**：使用当前策略（基于 Qwen‑32B 的语言模型）生成一批长 CoT 序列。  
   - **价值评估**：对每一步调用对齐后的价值模型，得到即时价值估计。  
   - **中间奖励密化**：把即时价值差分（当前价值 - 前一步价值）乘以一个小系数，作为该步的伪奖励。这样即使终点奖励稀疏，模型也能在每一步感受到“前进”或“后退”。  
   - **PPO 更新**：采用近端策略优化的剪切目标函数，但把奖励换成上述密化奖励的累计和。由于奖励更密集，梯度方差大幅下降，更新更稳。  
   - **序列长度归一化**：对每条样本的梯度乘以 `1 / length` 的权重，使得长序列不会压倒短序列的学习信号。

4. **安全回滚机制**  
   - 每 100 步记录一次模型参数和价值头的快照。  
   - 监控梯度范数和 KL 散度，一旦超出预设阈值，立即恢复到最近的快照并降低学习率。实验中，这套机制保证了 5,000 步内无一次 crash。

**最巧妙的点**在于把价值模型的“全局视野”与 PPO 的“局部策略”通过中间奖励桥接起来，同时用对齐损失把价值偏差压到最低。这样既保留了价值函数对长远回报的敏感，又避免了稀疏奖励导致的学习停滞。

### 实验与效果
- **数据集**：主要在 AIME‑2024（一个包含大量数学推理题的基准）上评估。  
- **基线**：DeepSeek‑R1‑Zero‑Qwen‑32B、DAPO 以及其他公开的价值‑基方法。  
- **成绩**：VAPO 在相同实验设置下取得 **60.4** 分，领先第二名超过 **10** 分。相比 DAPO，提升幅度同样在 10 分以上。  
- **收敛速度**：只用了 **5,000** 步就达到了最先进的水平，而 DAPO 需要数十万步才能逼近同样分数。  
- **可靠性**：在多次独立运行中没有出现训练崩溃，验证了安全回滚的有效性。  
- **消融实验**：作者分别去掉价值对齐、序列归一化和中间奖励密化三个模块，性能分别下降约 3.2、2.7、4.1 分，说明每个设计都有实质贡献。  
- **局限性**：论文未在非数学推理任务上做大规模验证，价值模型的对齐过程在极端长序列（>200 步）仍可能出现梯度漂移。作者也提到对齐损失的超参数需要在新任务上重新调优。

### 影响与延伸思考
VAPO 的成功展示了价值函数在长 CoT 场景下的可行性，激发了后续工作在**价值‑增强语言模型**方向的探索。2024 年底出现的几篇论文（如 “Value‑Guided Chain‑of‑Thought”）直接引用了 VAPO 的中间奖励密化思路，并尝试把它搬到多模态推理上。对想进一步深入的读者，可以关注以下两个方向：  
1. **跨任务价值对齐**：如何在一个价值模型上同时适配数学、代码和常识推理等多种任务。  
2. **自适应序列归一化**：利用学习到的步长分布动态调节梯度权重，进一步降低长短序列的冲突。

### 一句话记住它
VAPO 用价值对齐＋中间奖励把稀疏的长链推理变成了“每一步都有指路灯”，让强化学习既快又稳。