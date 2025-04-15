# A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce

> **Date**：2025-04-15
> **arXiv**：https://arxiv.org/abs/2504.11343

## Abstract

Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training.

---

# 一种极简主义的 LLM 推理方法：从拒绝采样到强化学习 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）已经能生成流畅文本的时代，让它们在需要多步推理的任务上保持高准确率仍是瓶颈。传统的微调往往只靠监督学习，缺少对答案质量的直接反馈，导致模型在复杂逻辑题上容易走偏。近年来，强化学习（RL）被引入作为“奖励驱动”的微调手段，但主流算法（如 PPO）实现复杂、调参成本高，而且并不总能带来显著提升。于是研究者迫切需要一种既简单又有效的方式，来利用奖励信息改进 LLM 的推理能力。

### 关键概念速览
**大语言模型（LLM）**：能够在海量文本上预训练，具备生成自然语言的能力。把它想象成一个会说话的“百科全书”，但在推理时可能会“胡说八道”。  
**强化学习（RL）**：让模型通过与环境交互获得奖励，从而学习更好的行为策略。类似于训练小狗坐下，给对的动作零食。  
**策略梯度（Policy Gradient）**：RL 中直接对策略（模型输出分布）求梯度的方法，目标是让高奖励的行为概率变大。可以把它看作“把好答案的投票权加重”。  
**PPO（Proximal Policy Optimization）**：一种对策略梯度做了安全约束的算法，防止一次更新改动太大。像是给模型的“刹车”，保证学习过程平稳。  
**奖励归一化（Reward Normalization）**：把原始奖励做均值方差标准化，防止数值差异导致梯度不稳定。相当于把不同分数的考试成绩统一到同一尺度。  
**拒绝采样（Rejection Sampling）**：先生成若干候选答案，只保留满足某个条件的样本继续使用。好比在抽奖时把不合格的票直接丢掉。  
**RAFT**：本文提出的“只用正奖励样本进行梯度更新”的拒绝采样基线。它只看“好答案”，不管“坏答案”。  
**GRPO**：一种在 LLM 上流行的 RL 方法，内部实现上融合了奖励归一化和对全错样本的剔除。作者把它当作“高级版的拒绝采样”。  
**KL 效率（KL Efficiency）**：指在保持模型与原始分布 KL 散度（即信息损失）不变的情况下，获得的奖励提升幅度。高 KL 效率意味着“用更少的改动换来更大的收益”。  

### 核心创新点
1. **重新审视 GRPO → 将其视作带有过滤机制的拒绝采样**：原先的 GRPO 被描述为一种复杂的 RL 变体，作者把它拆解后发现核心操作只是在训练时把“全错”响应剔除。这样一来，GRPO 与一个极简的拒绝采样基线（RAFT）在本质上是同一类方法。  
2. **实验验证过滤全错样本是主要收益来源 → 奖励归一化的贡献微乎其微**：通过对比不做任何过滤、只过滤全错、只做奖励归一化的三种设置，作者发现仅去掉全错样本就能把性能拉到和 GRPO、PPO 相当的水平，说明复杂的归一化步骤并非关键。  
3. **提出 Reinforce‑Rej → 在过滤全错的同时也剔除全对样本**：作者进一步思考，如果全对样本已经完美，继续用它来更新策略会浪费 KL 预算。于是设计了 Reinforce‑Rej：在每轮采样后，既丢掉完全错误的答案，也丢掉完全正确的答案，只在“有争议”的样本上做梯度更新。这样既提升了 KL 效率，又保持了训练的稳定性。  
4. **把 RAFT 定位为强而易懂的基线 → 为后续研究提供明确的对照**：作者强调，RA​FT 只需要正奖励过滤，代码实现几行即可复现，且解释性强。以后想评估新算法的增益时，直接和 RAFT 对比即可判断是否真的突破了“只用好答案”的上限。  

### 方法详解
整体思路可以概括为三步：**采样 → 过滤 → 梯度更新**。  
1. **采样阶段**：给定一个推理提示，模型使用当前策略（即原始 LLM 参数）生成多个候选答案。每个答案都会通过一个外部奖励模型（或人工评分）得到一个标量奖励。  
2. **过滤阶段**：根据奖励值进行两类判定。  
   - **全错过滤**：如果奖励为零或低于预设阈值，说明答案完全不符合要求，直接丢弃，不参与后续梯度计算。  
   - **全对过滤（Reinforce‑Rej 专属）**：如果奖励达到最高分（即模型已经给出完美答案），同样丢弃，因为继续强化这类样本只会让 KL 散度增大，却不提升实际性能。  
   剩下的样本大多是“部分对、部分错”，即模型在某些推理步骤上仍有改进空间。  
3. **梯度更新阶段**：对保留下来的样本，使用最原始的策略梯度公式计算梯度：梯度等于奖励乘以对数概率的负梯度（即 ∇θ log πθ(a|x) · r）。这里没有额外的 KL 惩罚或奖励归一化，保持了最简洁的形式。更新后模型的概率分布会倾向于产生更高奖励的答案。  

**关键细节**  
- **采样数量**：作者在实验中使用 4–8 条候选答案，足以覆盖多样的推理路径。  
- **奖励函数**：可以是基于答案正确性的硬标签，也可以是软评分模型（如 GPT‑4 评审）。只要能区分全错、全对和中间状态即可。  
- **过滤阈值**：全错阈值设为 0，意味着任何非正奖励都被剔除；全对阈值设为最高可能分数。  
- **KL 效率提升**：因为只在“有争议”样本上更新，模型的 KL 散度相对原始策略的增长被显著压缩，等价于在同样的 KL 预算下获得更多的奖励提升。  

最反直觉的地方在于**主动丢掉全对样本**。直觉上会觉得“好答案越多越好”，但作者指出，这会浪费 KL 预算，使模型在已经完美的区域继续“过度学习”，导致后期的 KL 惩罚变得更严苛，反而削弱整体收益。  

### 实验与效果
- **测试任务**：论文在多个公开的复杂推理基准上评估，包括数学推理（GSM8K、MATH）、逻辑推理（ARC‑E）以及代码生成（HumanEval）等。  
- **对比基线**：包括原始的 PPO、GRPO、以及不做任何过滤的普通策略梯度。  
- **主要结果**：在所有任务上，RAFT 的表现与 GRPO、PPO 相差不到 1% 的准确率，且训练时间约缩短 30%。Reinforce‑Rej 在同等 KL 预算下进一步提升约 0.5%–1% 的准确率，尤其在高难度的 MATH 数据集上表现最为突出。  
- **消融实验**：作者分别去掉全错过滤、全对过滤以及奖励归一化，结果显示：去掉全错过滤会导致性能骤降 3%–5%；去掉全对过滤（即回到 RAFT）对整体效果影响不大，但 KL 效率下降约 15%；奖励归一化的有无差别在 0.2% 以内。  
- **局限性**：实验主要依赖于外部奖励模型的质量，若奖励噪声大，过滤机制可能误删有价值的样本。作者也提到在极端稀疏奖励场景下，仅靠正样本过滤可能不足以驱动学习。  

### 影响与延伸思考
这篇工作在社区里引发了对“负样本到底该怎么用”的重新审视。随后出现的几篇论文（如 *Negative‑Aware RL for LLM*、*Selective Sampling for Reward‑Based Fine‑Tuning*）都在尝试设计更细粒度的样本筛选策略，而不是像 PPO 那样把所有样本都强行喂进去。对想进一步探索的读者，可以关注以下方向：  
- **奖励噪声鲁棒的过滤机制**：如何在奖励不可靠时仍保持有效的样本筛选。  
- **自适应阈值学习**：让模型自行学习何时该视为“全对”或“全错”。  
- **多任务统一过滤**：在跨任务微调时，如何统一不同任务的过滤标准。  

### 一句话记住它
只用“好答案”和“有争议的答案”来训练 LLM，既简单又高效——复杂的奖励归一化和全量梯度更新并非必要。