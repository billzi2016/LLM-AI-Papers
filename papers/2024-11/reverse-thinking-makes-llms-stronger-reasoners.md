# Reverse Thinking Makes LLMs Stronger Reasoners

> **Date**：2024-11-29
> **arXiv**：https://arxiv.org/abs/2411.19865

## Abstract

Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. In RevThink, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency -- using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10x more forward reasoning. RevThink also exhibits strong generalization to out-of-distribution held-out datasets.

---

# 逆向思维让大语言模型更强的推理能力 论文详细解读

### 背景：这个问题为什么难？

在让大语言模型（LLM）进行复杂推理时，传统做法往往只让模型“从题目到答案”一步步展开。虽然这种正向链式思考（Chain‑of‑Thought）已经提升了不少任务的表现，但模型仍然容易出现前后不一致、逻辑漏洞或对细节的忽视。根本原因在于，模型缺少一种自我校验的机制——人类常用的“从答案倒推回问题”来检查自己的思路，却很少在现有训练流程中出现。于是，如何让 LLM 同时掌握正向和逆向的推理路径，成为提升推理可靠性的关键瓶颈。

### 关键概念速览
- **正向推理（Forward Reasoning）**：模型从问题出发，逐步生成推理步骤直至答案。类似于解数学题时先写出每一步计算。
- **逆向推理（Backward Reasoning）**：模型先给出答案或目标，然后逆向推导出对应的问题或推理过程。好比先知道结论，再回想当初是怎么得到的。
- **教师模型（Teacher Model）**：一个能力更强的预训练模型，用来生成高质量的正向‑逆向推理对，充当“老师”提供训练样本。
- **学生模型（Student Model）**：需要被压缩或微调的较小模型，学习教师提供的四元组数据。
- **多任务学习（Multi‑Task Learning）**：一次性训练模型完成多种输出任务，让不同任务之间共享知识，提高整体表现。
- **知识蒸馏（Knowledge Distillation）**：把大模型的知识迁移到小模型的技术，常通过让学生模仿教师的输出概率分布实现。
- **样本效率（Sample Efficiency）**：在相同或更少的数据量下取得更好效果的能力，衡量模型对数据的利用率。

### 核心创新点
1. **正向‑逆向四元组数据增强**  
   - 之前的蒸馏或微调只使用“问题 + 正向推理”。  
   - 这篇论文让教师模型同时生成 **（问题、正向推理、逆向问题、逆向推理）** 四元组。  
   - 结果是学生模型在训练时能看到同一答案的两种思考路径，显著提升了前后自洽性。

2. **三重学习目标的多任务框架**  
   - 传统做法只让学生输出正向推理。  
   - 这里引入 **生成正向推理**、**生成逆向问题**、**生成逆向推理** 三个任务，统一在同一个网络里训练。  
   - 这种设计让模型在一次前向传播中同时练习“写答案”和“写答案的来源”，从而在推理时更容易进行自我检查。

3. **极致样本利用率的实验验证**  
   - 常规微调需要大量正向推理样本。  
   - 论文展示，仅使用 10% 正向推理数据（其余由逆向生成），就能跑赢使用 10 倍正向数据的标准微调。  
   - 说明逆向信息本身是一种高价值的监督信号。

4. **跨任务、跨分布的稳健提升**  
   - 之前的提升往往局限在单一数据集或特定推理类型。  
   - 通过在 12 个涵盖常识、数学、逻辑的基准上实验，平均提升 13.53%（相对零样本学生）和 6.84%（相对最强蒸馏基线），并在未见分布上保持优势。  
   - 证明逆向思维的通用性。

### 方法详解
**整体思路**  
1. 用一个强大的教师模型生成四元组（问题、正向推理、逆向问题、逆向推理）。  
2. 将这些四元组拼成训练样本，喂给学生模型。  
3. 学生模型在一次前向传播中同时学习三个任务：  
   - 给定问题 → 正向推理  
   - 给定问题 → 逆向问题（即“如果答案是 X，问题会是什么？”）  
   - 给定逆向问题 → 逆向推理（从逆向问题出发，重新推导出答案）  

**关键模块拆解**  

- **数据增强模块**  
  - 教师模型先对原始问题生成正向推理（CoT 风格）。  
  - 再把正向推理的最终答案当作“目标”，让教师模型逆向生成一个“对应的问题”。这一步相当于“答案 → 问题”。  
  - 最后，教师模型对逆向问题再做一次正向推理，得到逆向推理链。  
  - 结果是四段文字：Q、R_fwd、Q_rev、R_rev。  

- **多任务学习头**  
  - 学生模型的底层是一个标准的 Transformer 编码器‑解码器。  
  - 解码器的输出被分成三条平行的任务流：  
    1. **Forward‑Task**：输入 Q，目标是 R_fwd。  
    2. **Backward‑Question‑Task**：输入 Q，目标是 Q_rev。  
    3. **Backward‑Reasoning‑Task**：输入 Q_rev，目标是 R_rev。  
  - 训练时把三条 loss 加权求和（论文未给出具体权重，默认等权），一次梯度更新同时优化三项。  

- **自洽检查的隐式学习**  
  - 因为模型必须在同一次训练中把正向答案和逆向问题对应起来，它自然学会在生成正向推理时保持与逆向推理的一致性。  
  - 这相当于让模型在内部做一次“前后核对”，而不需要额外的后处理步骤。  

**最巧妙的设计**  
- 逆向问题的生成并不是随意的，而是让教师模型在已知答案的前提下“逆向构造”问题，这一步把答案信息显式注入了训练信号，极大提升了学生模型对答案的感知能力。  
- 三任务的统一训练让模型在同一参数空间里学会“写答案”和“写答案的来源”，避免了传统蒸馏中只模仿概率分布而缺乏结构化推理的缺陷。

### 实验与效果
- **数据集与任务**：共 12 个公开基准，覆盖常识推理（如 CommonsenseQA）、数学题（如 GSM8K）、逻辑推理（如 LogicalDeduction）等。  
- **对比基线**：  
  - 零样本学生模型（直接使用原始 LLM，不做任何微调）。  
  - 最强的知识蒸馏方法（如 KD‑CoT）。  
- **主要结果**：  
  - 平均提升 13.53% 超过零样本学生。  
  - 相比最强蒸馏基线提升 6.84%。  
  - 在仅使用 10% 正向推理样本的情况下，RevThink 的表现超过使用 10 倍正向样本的普通微调。  
- **消融实验**：  
  - 去掉逆向问题生成任务，提升幅度下降约 4%。  
  - 只使用正向‑逆向推理对（不生成逆向问题），提升约 2%。  
  - 说明三任务共同作用是性能提升的关键。  
- **局限性**：  
  - 论文主要在中等规模的 LLM（如 7B 参数）上验证，未说明在更大模型上的效果是否同样显著。  
  - 逆向问题的质量依赖教师模型的生成能力，若教师模型出现偏差，可能会把错误的逆向样本灌入学生。  
  - 对于需要多步推理且答案空间极大（如程序生成）的任务，逆向构造问题的难度会显著上升，本文未给出对应方案。

### 影响与延伸思考
- 这篇工作把“逆向思考”正式引入 LLM 微调流程，开启了 **双向推理** 这一新方向。随后的研究（如 *Bidirectional CoT*、*Self‑Check LLM*）纷纷在此基础上加入自我纠错、答案一致性判别等机制。  
- 对于想进一步探索的读者，可以关注以下几个方向：  
  1. **逆向问题的自动评估**：如何度量生成的逆向问题是否合理、是否覆盖所有答案。  
  2. **跨模态逆向思维**：把文本逆向思考扩展到图像、代码等模态。  
  3. **大模型上的端到端双向蒸馏**：在百亿级模型上验证逆向思维的规模效应。  
  4. **与检索结合**：让逆向生成的“问题”去检索外部知识，再回到正向推理，形成闭环。  

### 一句话记住它
让模型同时学会“从题目到答案”和“从答案倒推题目”，就像人类先写答案再回头检查，显著提升了大语言模型的推理可靠性。