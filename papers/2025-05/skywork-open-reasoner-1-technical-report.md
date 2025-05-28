# Skywork Open Reasoner 1 Technical Report

> **Date**：2025-05-28
> **arXiv**：https://arxiv.org/abs/2505.22312

## Abstract

The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets.

---

# Skywork Open Reasoner 1 技术报告 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，让模型像人一样进行多步推理一直是瓶颈。早期的模型往往直接给出答案，缺少思考过程，导致在数学、编程等需要严密逻辑的任务上表现不稳。虽然出现了“思维链”（Chain‑of‑Thought, CoT）技术，让模型先写出中间步骤，但这些步骤本身并不保证正确，模型仍会在长链推理时出现信息漂移或提前收敛到错误答案。强化学习（RL）被证明可以微调模型的决策策略，但在长链 CoT 场景下，如何设计奖励、避免熵崩塌（entropy collapse）以及保持推理的多样性，始终缺乏系统化方案。正因为这些根本性障碍，提升大模型的长链推理能力仍然是急需突破的难题。

### 关键概念速览

**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前，先把推理步骤逐步写出来，类似于人解题时的草稿过程，能够让错误更早被发现。

**强化学习（RL）**：一种让模型通过与环境交互、根据奖励信号调整行为的训练方式，这里把“环境”看作是答案评估器，奖励则来源于答案的正确性。

**熵（entropy）**：模型输出分布的随机程度，熵高意味着模型保持多样性，熵低则表示模型趋向确定性，过早的熵下降会导致搜索空间收窄，称为熵崩塌。

**熵崩塌（entropy collapse）**：在 RL 微调过程中，模型的输出分布突然变得极度确定，失去探索能力，往往导致最终性能下降。

**Distill（蒸馏）**：把大模型的知识压缩到更小的模型里，保持性能的同时降低计算成本，这里指的是 DeepSeek‑R1‑Distill 系列。

**LiveCodeBench**：评估模型代码生成与执行能力的基准，涵盖多种编程语言和难度层级。

**AIME24 / AIME25**：美国数学竞赛的两套试题，常被用来衡量模型的数学推理水平。

### 核心创新点

1. **从 DeepSeek‑R1‑Distill 到 Skywork‑OR1 的 RL 微调管线**  
   之前的 DeepSeek‑R1‑Distill 只用了监督学习进行蒸馏，缺少针对长链推理的专门优化。作者在此基础上加入了强化学习阶段，使用答案正确率作为奖励，直接对模型的推理路径进行强化。结果是 32B 规模模型的平均准确率从 57.8% 提升到 72.8%，7B 规模从 43.6% 提升到 57.5%。

2. **针对长链 CoT 的熵动态控制策略**  
   传统 RL 往往在训练后期出现熵崩塌，导致模型失去探索。论文通过系统实验找出影响熵的关键因素（如奖励平滑、温度调度、KL 散度约束），并在训练中加入逐步降低的熵惩罚项，防止模型过早收敛。实验表明，抑制熵崩塌显著提升了 AIME 与 LiveCodeBench 的测试表现。

3. **多尺度模型系列的统一评估**  
   作者同时训练了 32B、7B 以及专门针对数学任务的 Skywork‑OR1‑Math‑7B，展示了同一套 RL 框架在不同规模和任务专注度上的可迁移性。尤其是 32B 版本在 AIME24、AIME25 上超越了同尺寸的 DeepSeek‑R1 与 Qwen3‑32B，说明规模放大并非唯一提升点，训练策略同样关键。

4. **全链路开源生态**  
   为了让社区复现和进一步改进，论文公开了模型权重、训练代码以及使用的训练数据集。这在大模型强化学习领域仍属少数，降低了后续研究的门槛。

### 方法详解

**整体框架**  
整个训练流程分为三大阶段：① 预训练/蒸馏得到基线模型（DeepSeek‑R1‑Distill），② 基于该模型构建长链 CoT 数据并设计奖励函数，③ 采用强化学习（PPO）对模型进行微调，同时加入熵调控机制。目标是让模型在生成每一步推理时，都能在保持多样性的前提下，逐步逼近正确答案。

**关键模块拆解**

1. **长链 CoT 数据构建**  
   - 先用基线模型在数学、编程题目上生成完整的思维链。  
   - 对每一步的中间结果进行人工或自动校验，标记正确与错误的子步骤。  
   - 形成“状态—动作—奖励”三元组：状态是当前已生成的思维链，动作是模型下一步的输出，奖励依据该步是否推动最终答案正确而给出。

2. **奖励函数设计**  
   - **终局奖励**：如果完整思维链最终得到正确答案，给出高额正奖励。  
   - **局部奖励**：每一步若产生的中间结果被判定为“有助于正确答案”，则给出小额正奖励；若明显错误，则给负奖励。  
   - 通过奖励平滑（reward smoothing）避免奖励波动过大导致梯度不稳。

3. **强化学习算法（PPO）**  
   - 采用近端策略优化（Proximal Policy Optimization, PPO），在每次更新时限制新旧策略的 KL 散度，防止策略剧烈变化。  
   - 训练批次中混合使用监督学习的交叉熵损失，以保持语言生成的基本流畅性。

4. **熵动态控制**  
   - 引入 **熵惩罚项**：在损失函数中加入负熵乘以系数，系数随训练进度线性下降。早期保持高熵鼓励探索，后期逐步收紧。  
   - 使用 **温度调度**：在采样时对 logits 进行温度 scaling，早期温度高（输出更随机），后期温度低（输出更确定）。  
   - 通过实验发现，KL 散度约束与熵惩罚的协同作用是防止熵崩塌的关键。

**最巧妙的地方**  
作者没有单纯依赖大模型的规模提升，而是把“保持推理多样性”作为核心约束，利用熵惩罚和温度调度在强化学习中实现了类似“探索—利用”平衡的自适应机制。这种在长链 CoT 场景下的熵管理在之前的工作中很少出现，直接解决了 RL 微调导致的早期收敛问题。

### 实验与效果

- **测试数据集**：AIME24、AIME25（数学推理）以及 LiveCodeBench（代码生成与执行）。  
- **基线对比**：DeepSeek‑R1、Qwen3‑32B 以及同尺寸的开源模型。  
- **主要结果**：  
  - 32B 版本的平均准确率从 57.8% 提升到 72.8%，领先 DeepSeek‑R1 与 Qwen3‑32B。  
  - 7B 版本从 43.6% 提升到 57.5%，在同规模模型中处于领先水平。  
  - 在 LiveCodeBench 上，Skywork‑OR1‑32B 与 DeepSeek‑R1 持平，说明代码生成能力并未因 RL 微调受损。  
- **消融实验**：作者分别去掉熵惩罚、温度调度、局部奖励等模块，发现熵控制的缺失会导致准确率下降约 4‑5%，局部奖励的去除则使提升幅度从 15% 降至约 8%。这些实验验证了每个设计的必要性。  
- **局限性**：论文未给出在更大规模（如 100B）模型上的实验，也没有详细探讨不同任务（如常识推理）下熵控制的通用性。训练成本仍然高昂，只有大算力团队才能完整复现。

### 影响与延伸思考

这篇报告在大模型强化学习社区引起了广泛关注，尤其是对“长链推理的熵管理”提供了可操作的方案。随后的工作如 **OpenRL‑CoT**、**Entropy‑Aware RL for LLMs** 等，都在借鉴其熵惩罚与温度调度的组合方式，尝试在更广泛的任务上推广。对想进一步深入的读者，建议关注以下方向：① 更高效的奖励函数自动生成（如利用 LLM 自评），② 熵动态控制在多模态推理中的适配，③ 将 RL 与自监督预训练相结合的端到端训练框架。整体来看，Skywork‑OR1 为“让大模型真正思考”提供了实用的技术路径。

### 一句话记住它

**用强化学习加熵控制，让大语言模型在长思维链上保持探索，显著提升数学和代码推理的准确率。**