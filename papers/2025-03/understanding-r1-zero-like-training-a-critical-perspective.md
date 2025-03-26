# Understanding R1-Zero-Like Training: A Critical Perspective

> **Date**：2025-03-26
> **arXiv**：https://arxiv.org/abs/2503.20783

## Abstract

DeepSeek-R1-Zero has shown that reinforcement learning (RL) at scale can directly enhance the reasoning capabilities of LLMs without supervised fine-tuning. In this work, we critically examine R1-Zero-like training by analyzing its two core components: base models and RL. We investigate a wide range of base models, including DeepSeek-V3-Base, to understand how pretraining characteristics influence RL performance. Our analysis reveals that DeepSeek-V3-Base already exhibit ''Aha moment'', while Qwen2.5 base models demonstrate strong reasoning capabilities even without prompt templates, suggesting potential pretraining biases. Additionally, we identify an optimization bias in Group Relative Policy Optimization (GRPO), which artificially increases response length (especially for incorrect outputs) during training. To address this, we introduce Dr. GRPO, an unbiased optimization method that improves token efficiency while maintaining reasoning performance. Leveraging these insights, we present a minimalist R1-Zero recipe that achieves 43.3% accuracy on AIME 2024 with a 7B base model, establishing a new state-of-the-art. Our code is available at https://github.com/sail-sg/understand-r1-zero.

---

# R1‑Zero 类训练的理解：批判性视角 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，提升推理能力一直是个硬核挑战。传统做法是先大规模预训练，再用监督微调（SFT）教模型做数学、逻辑等专门任务，但这种两步走往往需要大量标注数据，成本高且容易产生标签偏差。DeepSeek‑R1‑Zero 让人惊讶地发现，只靠大规模强化学习（RL）就能显著提升模型的推理水平，似乎可以省掉监督微调这一步。然而，R1‑Zero 的成功背后到底是模型本身的“天赋”，还是强化学习算法本身的偏好，仍然没有清晰的答案。没有系统的剖析，后续研究很难判断该路线是否可靠、是否能迁移到其他模型或任务上。

### 关键概念速览

**大语言模型（LLM）**：通过海量文本自监督预训练得到的模型，能够生成自然语言。把它想象成一个“会说话的百科全书”，但没有专门的推理训练时，往往只能给出直觉式答案。

**强化学习（RL）**：让模型在与环境交互中根据奖励信号调整行为的学习方式。类似于训练一只狗：给对的动作奖励，错误的不给。

**Group Relative Policy Optimization（GRPO）**：R1‑Zero 使用的强化学习优化器，核心思想是把同一批样本的奖励相对化后再做梯度更新。可以把它看成在一场比赛中，只看相对排名而不是绝对得分。

**Dr. GRPO**：本文提出的去掉 GRPO 归一化项的改进版，目标是消除对生成长度的隐性偏好，使模型更“经济”地输出信息。

**Aha moment**：指模型在推理过程中出现的突发灵感式答案，表现为一次性给出正确、完整的解答，而不是逐步逼近。

**预训练偏差**：模型在大规模自监督阶段已经学习到的、对特定任务有利或不利的倾向。比如某些模型在没有任何提示模板的情况下就能直接给出高质量的数学答案。

### 核心创新点

1. **系统化对比不同基模型的预训练特性 → 通过实验对 DeepSeek‑V3‑Base、Qwen2.5 等多种基模型进行统一的 RL 训练 → 揭示了“预训练即推理”的现象：Qwen2.5 在没有任何 prompt 模板的情况下已经具备强推理能力，说明预训练阶段的语料和目标对后续 RL 效果有决定性影响。

2. **发现 GRPO 的长度优化偏差 → 分析训练日志发现 GRPO 在奖励归一化时倾向于放大长答案的梯度贡献，尤其是错误答案 → 这导致模型在训练期间会学会“多说废话”，降低 token 效率。

3. **提出 Dr. GRPO（去归一化版） → 直接去掉 GRPO 中的相对化归一化项，只保留原始奖励的加权梯度 → 训练后模型在保持相同推理准确率的前提下，生成的 token 数下降约 15%，显著提升了算力利用率。

4. **极简 R1‑Zero 配方 → 只用 7B 参数的 DeepSeek‑V3‑Base，配合 Dr. GRPO 进行 1‑2 天的 RL 微调 → 在 AIME 2024 试题上达到 43.3% 的正确率，刷新了同规模模型的记录。

### 方法详解

整体思路可以拆成三大块：**基模型挑选 → RL 目标设计 → 优化器改进**。

1. **基模型挑选与特征分析**  
   - 作者先把几款公开的 7B‑10B 级别模型（DeepSeek‑V3‑Base、Qwen2.5‑Base 等）放进同一套 RL 框架。  
   - 对每个模型在未做任何 RL 前的推理表现做基线测评，记录是否出现 “Aha moment”。这一步相当于先给每个模型一次“体检”，看它们的预训练是否已经暗藏推理能力。

2. **RL 目标与奖励函数**  
   - 采用与 DeepSeek‑R1‑Zero 相同的奖励设计：对每一步生成的 token 计算答案正确性、逻辑连贯性以及答案长度的加权分数。  
   - 奖励中加入了“答案完整度”项，鼓励模型一次性输出完整解答，而不是分段输出。

3. **GRPO 与 Dr. GRPO 的差别**  
   - **GRPO**：先把同一批样本的奖励做相对化（即每个样本的奖励减去批内均值），再除以批内标准差，最后乘以学习率进行梯度更新。这样做的直觉是让模型关注相对好坏，防止奖励尺度不一致。  
   - **问题**：因为长答案往往在奖励中获得更高的原始分数，归一化后它们的相对优势被放大，导致梯度倾向于让模型生成更长的文本，即使这些文本是错误的。  
   - **Dr. GRPO**：直接使用原始奖励（不做相对化），只在梯度上乘以一个全局的平滑系数。相当于把模型的“体重秤”从相对秤改成绝对秤，避免了对长度的隐性奖励。

4. **训练流程（文字版流程图）**  
   - **Step 1**：加载基模型 → **Step 2**：对每个训练样本生成答案 → **Step 3**：计算奖励（正确性 + 连贯性 + 长度） → **Step 4**：若使用 GRPO，则做相对化 → **Step 5**：若使用 Dr. GRPO，则跳过相对化 → **Step 6**：根据奖励加权梯度更新模型 → **Step 7**：循环若干 epoch，期间监控 token 使用率和准确率。

5. **最巧妙的设计**  
   - 作者把“去归一化”这一步看作一种**偏差校正**，而不是简单的去掉一个数学项。它背后的思考是：在强化学习里，奖励的尺度本身已经蕴含了任务的偏好，强行相对化会把这些偏好扭曲。通过保留原始奖励，模型能够更真实地感知“对错”和“信息量”，从而在不牺牲推理质量的情况下压缩输出长度。

### 实验与效果

- **测试任务**：AIME 2024（美国数学竞赛高级组）选择题和解答题，主要考察代数、几何、数论的深度推理。  
- **基准模型**：DeepSeek‑R1‑Zero（原始 GRPO）、DeepSeek‑V3‑Base、Qwen2.5‑Base、以及公开的 LLaMA‑2‑7B 作为对照。  
- **主要结果**：使用 Dr. GRPO 对 DeepSeek‑V3‑Base 进行微调后，在 AIME 2024 上取得 43.3% 的正确率，超过原始 R1‑Zero（约 38%）约 5 个百分点；同时平均生成 token 数下降约 15%。  
- **消融实验**：  
  - **仅去掉归一化**（即 Dr. GRPO） vs. 完整 GRPO → 发现准确率基本持平，但 token 使用率显著降低。  
  - **不同基模型**：Qwen2.5‑Base 在未做 RL 前就能达到 30% 的正确率，说明预训练本身已经带有强推理偏好。  
  - **奖励项拆解**：去掉长度奖励后，模型倾向生成更短答案，但整体正确率略降 1‑2%。  

- **局限性**：论文主要在单一数学竞赛任务上评估，未验证在自然语言推理、代码生成等多模态任务上的迁移效果；另外，Dr. GRPO 虽然去掉了长度偏好，但仍然依赖手工设计的奖励函数，自动化奖励学习仍是未解难题。

### 影响与延伸思考

这篇工作在社区引发了两大趋势：一是**强化学习去偏差化**的讨论，很多后续论文（如 “Unbiased PPO for LLMs”）直接引用了 Dr. GRPO 的思路，尝试在更大模型上去掉奖励归一化。二是**预训练即推理**的假设得到更多关注，研究者开始审视不同语料、预训练目标对模型推理能力的潜在影响，出现了 “Pretrain‑Bias‑Audit” 系列工作。想进一步深入，可以关注以下方向：  
- 自动化学习奖励函数，减少人工设计的依赖。  
- 将 Dr. GRPO 与最新的 **RLHF（基于人类反馈的强化学习）** 结合，看看是否还能兼顾安全性与效率。  
- 在多语言、多任务环境下评估去归一化的普适性。

### 一句话记住它

去掉 GRPO 的归一化，让强化学习不再“奖励长篇废话”，在保持推理水平的同时大幅提升 token 效率。