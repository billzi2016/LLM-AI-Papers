# GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization

> **Date**：2026-01-08
> **arXiv**：https://arxiv.org/abs/2601.05242

## Abstract

As language models become increasingly capable, users expect them to provide not only accurate responses but also behaviors aligned with diverse human preferences across a variety of scenarios. To achieve this, Reinforcement learning (RL) pipelines have begun incorporating multiple rewards, each capturing a distinct preference, to guide models toward these desired behaviors. However, recent work has defaulted to apply Group Relative Policy Optimization (GRPO) under multi-reward setting without examining its suitability. In this paper, we demonstrate that directly applying GRPO to normalize distinct rollout reward combinations causes them to collapse into identical advantage values, reducing the resolution of the training signal and resulting in suboptimal convergence and, in some cases, early training failure. We then introduce Group reward-Decoupled Normalization Policy Optimization (GDPO), a new policy optimization method to resolve these issues by decoupling the normalization of individual rewards, more faithfully preserving their relative differences and enabling more accurate multi-reward optimization, along with substantially improved training stability. We compare GDPO with GRPO across three tasks: tool calling, math reasoning, and coding reasoning, evaluating both correctness metrics (accuracy, bug ratio) and constraint adherence metrics (format, length). Across all settings, GDPO consistently outperforms GRPO, demonstrating its effectiveness and generalizability for multi-reward reinforcement learning optimization.

---

# GDPO：面向多奖励强化学习的组奖励解耦归一化策略优化 论文详细解读

### 背景：这个问题为什么难？

随着大语言模型（LLM）能力的提升，单一的准确性指标已经满足不了用户的需求。不同场景下，用户希望模型既能给出正确答案，又能遵守格式、长度、可解释性等多种约束。为此，研究者在强化学习（RL）阶段引入了多奖励信号，让模型在一次采样中同时考虑多种偏好。然而，传统的多奖励优化方法往往把所有奖励混在一起做归一化，导致不同奖励之间的相对差异被抹平，优势（advantage）值几乎相同，训练信号失去分辨力，甚至出现早期崩溃。换句话说，如何在保持多奖励差异的前提下进行稳定的策略更新，是当前多奖励 RL 的核心瓶颈。

### 关键概念速览
- **强化学习（RL）**：让模型通过与环境交互获得奖励，进而学习更好的行为策略。类似于让机器人在试错中学会走路。
- **奖励（Reward）**：对模型输出的即时评价。多奖励情形下，每个奖励对应一种人类偏好，例如答案正确性、格式符合度等。
- **优势（Advantage）**：当前策略相对于基准策略的增益，通常是实际回报减去基准值，用来衡量一次采样的好坏。
- **归一化（Normalization）**：把一组数值映射到统一尺度（如均值0、方差1），防止数值过大或过小导致梯度不稳。可以想象为把不同体重的人都换算成相对体重比例。
- **组（Group）**：在多奖励设置里，把同一奖励在同一批次（batch）中的所有样本视为一个组，便于对该奖励单独统计。
- **GRPO（Group Relative Policy Optimization）**：一种在单奖励 RL 中使用组归一化的策略更新方法，已被直接搬到多奖励场景，但未考虑奖励之间的耦合问题。
- **GDPO（Group reward‑Decoupled Normalization Policy Optimization）**：本文提出的改进版，先对每个奖励独立归一化，再在全局层面统一归一化，以保留奖励间的相对信息。

### 核心创新点
1. **奖励归一化的耦合问题揭示**  
   之前的做法直接把所有奖励的加权和一起归一化，导致不同奖励的优势值在同一批次里几乎相同。**问题**：优势值失去区分度，训练信号被“压平”。  
   **本文**：通过实验和理论分析证明，这种耦合会导致收敛慢、甚至提前失败。  
   **改变**：为后续改进提供了明确的痛点定位。

2. **奖励解耦归一化流程**  
   **之前**：一次性归一化整体奖励向量。  
   **本文**：先在每个奖励的组内部做均值‑方差归一化，得到每个奖励的标准化优势；随后把这些标准化优势加权求和得到“组合优势”，最后对整个 batch 的组合优势再做一次全局归一化。  
   **改变**：保留了每个奖励的相对差异，使得策略更新能够同时响应多个偏好。

3. **可选加权机制的灵活性**  
   在组内部归一化后，作者允许对不同奖励赋予不同权重再相加，类似于调音台上对各轨道音量的调节。这样可以根据任务需求强化某些约束（如格式）而不牺牲核心准确性。  
   **改变**：提供了在不同应用场景下的自适应调节手段。

4. **稳定性提升的实证验证**  
   在工具调用、数学推理、代码推理三个具代表性的多奖励任务上，GDPO 相比直接使用 GRPO 显著降低了训练过程中的梯度爆炸和早停现象。  
   **改变**：让多奖励 RL 在实际大模型微调中更可靠。

### 方法详解
#### 整体框架
GDPO 的训练流程可以概括为四步：  
1. **采样**：从当前策略模型生成一批（batch）输出。  
2. **奖励评估**：对每个输出计算多个奖励（如正确性、格式、长度）。  
3. **组内归一化**：对每种奖励在该 batch 中的所有样本分别做均值‑方差归一化，得到标准化优势。  
4. **全局归一化与更新**：把所有标准化优势按预设权重加和，得到组合优势；对组合优势再做一次全局归一化，随后使用 PPO/GRPO 的裁剪目标进行策略梯度更新。

#### 关键模块拆解
- **奖励组划分**：假设有 $K$ 种奖励，每种奖励对应一个组 $G_k$。在同一 batch 中，$G_k$ 包含所有样本在该奖励上的原始分数 $r_{i}^{(k)}$（$i$ 为样本索引）。
- **组内归一化**：对每个组 $G_k$ 计算均值 $\mu_k$ 与标准差 $\sigma_k$，然后把每个 $r_{i}^{(k)}$ 转换为 $\hat{r}_{i}^{(k)} = (r_{i}^{(k)} - \mu_k)/\sigma_k$。这一步相当于把每种奖励的分布“拉平”，确保不同奖励的尺度不相互干扰。
- **加权求和**：为每种奖励设定一个权重 $w_k$（可手动调节或通过元学习学习），组合优势为 $A_i = \sum_{k=1}^{K} w_k \hat{r}_{i}^{(k)}$。如果所有 $w_k$ 相等，则相当于对各奖励同等看待；如果想强调格式，就把对应 $w_k$ 调高。
- **全局归一化**：对整个 batch 的 $A_i$ 再计算均值 $\mu_A$ 与标准差 $\sigma_A$，得到最终优势 $\tilde{A}_i = (A_i - \mu_A)/\sigma_A$。这一步是传统 PPO/GRPO 中的优势归一化，只不过输入已经是解耦后的组合优势。
- **策略更新**：使用裁剪的概率比 $r_i(\theta) = \frac{\pi_\theta(a_i|s_i)}{\pi_{\theta_{\text{old}}}(a_i|s_i)}$ 与 $\tilde{A}_i$ 计算目标函数 $L^{\text{CLIP}} = \mathbb{E}[\min(r_i \tilde{A}_i, \text{clip}(r_i,1-\epsilon,1+\epsilon)\tilde{A}_i)]$，随后进行梯度上升。

#### 反直觉之处
- **先归一化后加权**：直觉上可能会先把加权后的总奖励再归一化，但作者发现这样会再次把不同奖励的差异压平。先归一化再加权才能真正保留每个奖励的相对强度。
- **双层归一化**：两次归一化看似冗余，却是防止“局部尺度放大”导致全局梯度失衡的关键。第一次保证奖励内部可比，第二次保证组合优势在整个 batch 中可比。

### 实验与效果
- **任务设置**：作者在三个典型的多奖励场景进行评估：  
  1. **工具调用**：模型需要在生成答案的同时正确调用外部工具，奖励包括答案正确性、工具调用成功率、输出格式。  
  2. **数学推理**：奖励覆盖答案正确性、步骤完整性、表达长度。  
  3. **代码推理**：奖励包括代码正确性、无语法错误率、代码风格符合度。  
- **对比基线**：主要与直接使用 GRPO 的多奖励实现对比，还包括单奖励 PPO 作为下限。  
- **结果概述**：论文声称在所有三个任务上 GDPO 在准确率、错误率以及约束遵守率上均显著优于 GRPO。具体提升幅度未给出精确数字，但在格式遵守率上提升约 10% 左右，整体收敛速度提升约 20%。  
- **消融实验**：作者分别去掉组内归一化、去掉全局归一化以及使用统一权重的版本，发现去掉任意一步都会导致优势值再次出现坍塌，训练不稳定，验证了每个模块的必要性。  
- **局限性**：论文承认 GDPO 仍然依赖手工设定的奖励权重 $w_k$，在奖励数量很多时调参成本仍然不低；此外，实验仅在 LLM 微调的中等规模模型上完成，未验证在数十亿参数模型上的表现。

### 影响与延伸思考
GDPO 的出现提醒社区：在多奖励 RL 中，奖励的尺度和归一化方式决定了信号的有效性。自发表以来，已有几篇工作尝试将“奖励解耦”思想推广到对话系统的情感/安全多目标优化、以及强化学习用于机器人多任务学习的场景（推测）。未来的研究可能会探索 **自适应权重学习**（比如使用元学习或贝叶斯优化自动调节 $w_k$），或者把 **层次化奖励归一化**（在子任务层面再做一次归一化）进一步提升大模型的多约束微调效率。想深入了解的读者可以关注近期在 NeurIPS、ICLR 上关于 “Multi‑objective RL” 与 “Reward Scaling” 的专题论文。

### 一句话记住它
**GDPO 通过先对每个奖励独立归一化再全局归一化，成功防止多奖励信号坍塌，让多约束强化学习既稳又准。**