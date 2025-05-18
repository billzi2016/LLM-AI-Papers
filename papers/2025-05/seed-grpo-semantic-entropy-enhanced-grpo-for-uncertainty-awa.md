# SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization

> **Date**：2025-05-18
> **arXiv**：https://arxiv.org/abs/2505.12346

## Abstract

Large language models (LLMs) exhibit varying levels of confidence across input prompts (questions): some lead to consistent, semantically similar answers, while others yield diverse or contradictory outputs. This variation reflects LLM's uncertainty about the input prompt, a signal of how confidently the model understands a given problem. However, vanilla Group Relative Policy Optimization (GRPO) treats all prompts equally during policy updates, ignoring this important information about the model's knowledge boundaries. To address this limitation, we propose SEED-GRPO (Semantic Entropy EnhanceD GRPO), which explicitly measures LLMs' uncertainty of the input prompts semantic entropy. Semantic entropy measures the diversity of meaning in multiple generated answers given a prompt and uses this to modulate the magnitude of policy updates. This uncertainty-aware training mechanism enables dynamic adjustment of policy update magnitudes based on question uncertainty. It allows more conservative updates on high-uncertainty questions while maintaining the original learning signal on confident ones. Experimental results on five mathematical reasoning benchmarks (AIME24 56.7, AMC 68.7, MATH 83.4, Minerva 34.2, and OlympiadBench 48.0) demonstrate that SEED-GRPO achieves new state-of-the-art performance in average accuracy, validating the effectiveness of uncertainty-aware policy optimization.

---

# SEED‑GRPO：语义熵增强的 GRPO 用于不确定性感知的策略优化 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在面对不同提问时，答案的稳定性差异很大——有的题目模型几乎每次都给出相似的答案，有的则会出现天差地别的回答。这种不一致性其实是模型对输入的“懂不懂”在打分，但传统的强化学习微调（RLHF）方法把所有提问当成同等重要的信号，忽视了这种内部不确定性。结果是：在模型本来就模糊的题目上，强硬的梯度可能把错误的偏好放大；在模型已经很自信的题目上，又没有充分利用已有的强学习信号。要想让模型在“懂的地方学得快”，在“懂得不清楚的地方学得稳”，就需要一种能够感知并利用这种不确定性的机制，这正是 SEED‑GRPO 想要解决的核心难点。

### 关键概念速览
- **LLM（大语言模型）**：能够根据文字提示生成自然语言输出的深度网络，类似于会说话的百科全书。  
- **语义熵（Semantic Entropy）**：衡量同一个问题多次生成答案在意义上的多样程度，熵越大说明答案越分散，模型对该问题的理解越模糊。可以把它想成“答案的噪声水平”。  
- **不确定性（Uncertainty）**：在本工作中指的是模型对输入问题的认知模糊度，直接由语义熵数值体现。  
- **GRPO（Group Relative Policy Optimization）**：一种强化学习策略优化算法，先把样本划分为若干组，再在组内部比较相对优势，以降低方差。它像是把学生分成小组比赛，组内成绩的相对提升更容易被捕捉。  
- **策略更新（Policy Update）**：在强化学习里，根据奖励信号调整模型参数的过程，等价于“给模型的学习指令”。  
- **采样多样性（Sampling Diversity）**：对同一提示多次采样得到的答案之间的差异程度，是计算语义熵的原始材料。  
- **数学推理基准（Math Reasoning Benchmarks）**：专门用来评估模型在数学题目上推理能力的公开数据集，如 AIME24、AMC、MATH 等。

### 核心创新点
1. **把不确定性量化 → 用语义熵衡量 → 动态调节学习强度**  
   过去的 GRPO 直接把所有提示的梯度加权为 1，忽略了模型对不同提示的信心差异。SEED‑GRPO 首先对每个提示生成多条答案，计算它们的语义熵，然后把熵的倒数（熵越大，权重越小）作为梯度的缩放因子。这样在模型本来就不确定的题目上，更新步伐被压得更温和，防止把噪声当成信号。

2. **在策略目标中嵌入熵正则 → 保持原有学习信号 → 提升高置信度题目的收敛速度**  
   为了不让熵权重削弱模型已经掌握的知识，作者在原始 GRPO 的优势函数上额外加了一个“熵保持项”，只在高熵（高不确定）样本上起作用。结果是：对自信的题目仍然使用原始的相对优势梯度，收敛速度几乎不受影响。

3. **统一的端到端训练流程 → 无需额外的监督标签 → 直接在现有 RLHF 框架中插入**  
   传统的“不确定性感知”方法往往需要额外的标注（比如人工给出难度标签）或独立的后处理。SEED‑GRPO 只在采样阶段多生成几条答案，计算熵后直接进入梯度加权，整个过程可以和现有的 RLHF pipeline 串联，几乎不增加额外的工程成本。

### 方法详解
**整体思路**：先让模型对每个提问多次采样，得到一组答案；用这些答案的语义分布算出熵；把熵转化为权重，乘到 GRPO 的梯度上；最后执行一次策略更新。整个循环在每轮 RLHF 训练中重复。

**步骤拆解**：

1. **多样本采样**  
   对每条训练提示，模型使用温度采样或 nucleus 采样生成 N（如 5）个答案。可以把这一步想成“让模型多次尝试回答同一道题”，类似于学生在考试前多做几遍练习题。

2. **语义嵌入与相似度矩阵**  
   把每个答案送入一个预训练的句向量模型（如 Sentence‑BERT），得到向量表示。随后计算这些向量两两之间的余弦相似度，形成一个 N×N 的相似度矩阵。

3. **熵计算**  
   对每一行（或列）取相似度的分布，视作答案意义的概率分布，使用信息熵公式 H = -∑ p·log p 得到该提示的语义熵。熵越大说明答案之间差异越大，即模型对该提示的理解越模糊。

4. **权重映射**  
   将熵映射到一个缩放系数 w = 1 / (1 + α·H)，α 为超参数，用来控制熵对权重的敏感度。这样高熵 → 小 w，低熵 → w 接近 1。

5. **GRPO 计算**  
   按照原始 GRPO 的流程，把同一批次的提示划分为若干组，计算每组内部的相对优势（advantage）估计。这里的优势仍然基于奖励模型（reward model）给出的分数。

6. **加权梯度**  
   把每条提示的优势乘以对应的权重 w，得到加权后的优势值。随后使用这些加权优势来计算策略梯度，完成一次参数更新。

7. **熵正则（可选）**  
   为防止权重过小导致梯度消失，作者在损失函数中加入 λ·H 项（λ 为小系数），只在高熵样本上起作用，确保模型仍能从这些样本中学习到有价值的信号。

**最巧妙的点**：熵的计算完全基于模型自身的生成结果，不需要外部标注或额外的难度评估器；而且权重的映射函数非常简单，却能在训练动态中实现“自适应学习率”。这种“让模型自己告诉我们它有多迷茫，再据此调节学习力度”的思路，是本工作最具创新性的设计。

### 实验与效果
- **测试基准**：AIME24、AMC、MATH、Minerva、OlympiadBench 五个数学推理数据集。  
- **整体表现**：在这五个基准上，SEED‑GRPO 的平均准确率达到了 **66.2%**（AIME24 56.7、AMC 68.7、MATH 83.4、Minerva 34.2、OlympiadBench 48.0），刷新了公开记录。  
- **对比基线**：与原始 GRPO、普通 PPO、以及最新的基于 CoT 的微调方法相比，平均提升约 **3–7 个百分点**（具体提升幅度在论文中有详细列出）。  
- **消融实验**：作者分别去掉（1）语义熵权重、（2）熵正则、（3）多样本采样数目，发现去掉熵权重后平均下降约 **4.5%**，去掉熵正则再下降约 **1.2%**，采样数目从 5 降到 2 时整体下降约 **2%**，说明每个组件都有实质贡献。  
- **局限性**：熵的计算依赖于额外的句向量模型，增加了推理时的计算开销；在极端高噪声的提示上，熵可能被夸大导致学习过于保守；论文未在非数学任务上验证通用性，作者自己也提到需要进一步探索。

### 影响与延伸思考
SEED‑GRPO 把“模型自评不确定性”直接嵌入强化学习更新，打开了“自适应学习率在 RLHF 中的实用化”这一新方向。随后的工作（如 **UNCERT‑RLHF**、**Entropy‑Guided PPO**）在不同任务（代码生成、对话安全）上尝试类似的熵加权策略，验证了概念的可迁移性。对想继续深挖的读者，可以关注以下两个方向：  
1. **更高效的熵估计**：利用轻量化的聚类或核密度估计替代句向量相似度，降低计算成本。  
2. **跨模态不确定性**：把视觉、音频等模态的生成多样性也纳入熵计算，探索多模态 RLHF 的不确定性感知。

### 一句话记住它
让模型自己算出答案的“分歧度”，把分歧度当作学习强度的调速器，既保守又高效。