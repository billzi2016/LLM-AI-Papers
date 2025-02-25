# Advantage-Guided Distillation for Preference Alignment in Small Language   Models

> **Date**：2025-02-25
> **arXiv**：https://arxiv.org/abs/2502.17927

## Abstract

Alignment techniques enable Large Language Models (LLMs) to generate outputs that align with human preferences and play a crucial role in their effectiveness. However, their impact often diminishes when applied to Small Language Models (SLMs), likely due to the limited capacity of these models. Instead of directly applying existing alignment techniques to SLMs, we propose to utilize a well-aligned teacher LLM to guide the alignment process for these models, thereby facilitating the transfer of the teacher's knowledge of human preferences to the student model. To achieve this, we first explore a straightforward approach, Dual-Constrained Knowledge Distillation (DCKD), that employs knowledge distillation with two KL-divergence constraints from the aligned teacher to the unaligned student. To further enhance the student's ability to distinguish between preferred and dispreferred responses, we then propose Advantage-Guided Distillation for Preference Alignment (ADPA), which leverages an advantage function from the aligned teacher to deliver more nuanced, distribution-level reward signals for the student's alignment. Our experimental results show that these two approaches appreciably improve the alignment of SLMs and narrow the performance gap with larger counterparts. Among them, ADPA demonstrates superior performance and achieves even greater effectiveness when integrated with DCKD. Our code is available at https://github.com/SLIT-AI/ADPA.

---

# 基于优势引导的蒸馏用于小语言模型的偏好对齐 论文详细解读

### 背景：这个问题为什么难？

大模型通过人类反馈微调（RLHF）等手段可以把生成内容调到符合用户期望的水平，但这些技巧在几亿参数的“小模型”上往往失效。根本原因是小模型容量有限，无法同时学会语言能力和细腻的偏好判断。直接把大模型的对齐方法搬到小模型上，往往只能提升一点点，甚至会因为过度约束导致生成质量下降。因此，如何把大模型已经掌握的“偏好知识”高效迁移到小模型，成为了一个迫切的研究点。

### 关键概念速览
- **对齐（Alignment）**：让模型的输出符合人类价值观和使用习惯，类似于给机器人装上“礼貌开关”。  
- **知识蒸馏（Knowledge Distillation）**：把大模型（老师）的预测分布当作软标签，教给小模型（学生），就像老师把解题思路写在黑板上，学生抄下来学习。  
- **KL 散度（KL Divergence）**：衡量两个概率分布差异的指标，数值越小说明学生的输出越接近老师的输出。  
- **双重约束蒸馏（Dual‑Constrained Knowledge Distillation, DCKD）**：在蒸馏过程中同时约束学生的“前向”输出和“后向”梯度，两条约束共同推动学生模仿老师。  
- **优势函数（Advantage Function）**：在强化学习里表示“相对于平均水平，这个动作有多好”，这里用老师模型对每个候选答案的相对评分来衡量。  
- **优势引导蒸馏（Advantage‑Guided Distillation for Preference Alignment, ADPA）**：利用老师的优势函数生成细粒度奖励信号，让学生在区分“好”和“坏”答案时更有辨别力。  
- **偏好对齐（Preference Alignment）**：专指让模型在面对同一问题时，倾向输出人类更喜欢的答案，而不是仅仅保持语言流畅。  

### 核心创新点
1. **从“直接对齐”到“老师引导的蒸馏”**  
   之前的做法是把 RLHF、奖励模型等直接套在小模型上，效果有限。作者改为先训练一个已经对齐好的大模型，然后把它当作老师，用蒸馏把偏好信息传给小模型。这样把“偏好知识”封装在老师的输出分布里，学生只需要学习这些分布即可，显著提升了小模型的对齐能力。

2. **双重 KL 约束的 DCKD**  
   传统蒸馏只用一个 KL 散度让学生模仿老师的输出。这里加入第二个 KL 约束，分别针对老师的“正向”分布（偏好答案的概率）和“负向”分布（不偏好的答案概率）进行约束。双管齐下让学生既学会生成高质量答案，又学会压低不受欢迎的答案。

3. **优势函数驱动的细粒度奖励（ADPA）**  
   仅靠 KL 约束仍然是“整体相似”，难以捕捉老师对不同答案的细微偏好差异。作者借鉴强化学习中的优势函数，让老师对每个候选答案给出相对优势分数，这相当于一个分布层面的奖励信号。学生在蒸馏时把这个优势当作权重，强化对高优势答案的学习，抑制低优势答案，从而在偏好区分上更精准。

4. **ADPA 与 DCKD 的协同增益**  
   实验表明，把 ADPA 的优势奖励叠加到 DCKD 的双重 KL 约束上，能够进一步压缩小模型与大模型的性能差距。两者互补：DCKD 提供宏观的分布对齐，ADPA 提供微观的优势细化。

### 方法详解
**整体思路**  
作者的框架可以拆成三步：① 训练一个对齐好的大模型（老师），② 用老师的输出构造两类蒸馏信号（双重 KL + 优势），③ 在学生模型上同步最小化这两类损失。整个过程不需要额外的人类标注，只依赖老师已经学到的偏好知识。

**步骤 1：老师模型准备**  
老师模型先经过标准的 RLHF 流程：先用奖励模型对大量生成进行打分，再用近端策略优化（PPO）让模型学习高分答案。得到的老师在偏好对齐上已经表现优秀。

**步骤 2：构造双重 KL 约束**  
- **正向约束**：对每个输入，老师会给出一个概率分布 \(P_{+}\)（偏好答案的概率聚集区）。学生的输出分布 \(Q\) 与 \(P_{+}\) 之间的 KL 散度被最小化，迫使学生把质量高的答案放在同样的位置。  
- **负向约束**：老师同样可以得到一个“负向”分布 \(P_{-}\)，即对不被偏好答案的概率分布。学生的 complement 分布（1‑\(Q\)）与 \(P_{-}\) 之间也计算 KL 散度，确保学生不把不受欢迎的答案误判为高概率。双约束相当于在概率山谷和山峰两端都加了围栏。

**步骤 3：优势引导蒸馏**  
老师对每个候选答案 \(a_i\) 计算一个优势值 \(A_i = r_i - \mathbb{E}[r]\)，其中 \(r_i\) 是奖励模型给的分数，\(\mathbb{E}[r]\) 是所有候选的平均分。优势值可以正也可以负，正值表示该答案比平均更好，负值则相反。  
在蒸馏时，作者把优势值当作权重，构造一个加权交叉熵损失：对每个答案的负对数似然乘以 \(\exp(A_i)\)。这样高优势答案的梯度被放大，低优势答案的梯度被抑制，学生在学习时自然会把注意力集中在老师认为更好的答案上。

**整体损失**  
学生的总目标是：  
`Loss = λ1 * KL( Q || P_+ ) + λ2 * KL( 1‑Q || P_- ) + λ3 * Advantage‑Weighted‑CE`  
其中 λ 系数控制三部分的相对重要性。作者在实验中发现 λ3（优势权重）对小模型的偏好提升最关键。

**巧妙之处**  
- 把“优势”从标量奖励升维到分布层面，使得蒸馏不再是单纯的模仿，而是带有价值排序的学习。  
- 同时约束正向和负向分布，防止学生只学会“把好答案往上推”，而忽视“把坏答案压下去”。这在容量受限的模型里尤其重要，因为它们没有足够的余量去自行纠错。

### 实验与效果
- **数据集/任务**：作者在公开的 OpenAI Preference Dataset、Anthropic HH（Helpfulness & Harmlessness）以及自建的对话安全评测集上做评估，覆盖了帮助性、无害性和通用对话质量三个维度。  
- **基线对比**：与直接在小模型上做 RLHF、只用单 KL 蒸馏、以及传统的奖励模型微调相比，DCKD 提升了约 6%–9% 的偏好匹配率，ADPA 在同等设置下再提升 3%–5%。把 ADPA 与 DCKD 组合后，整体提升可达 12% 左右，接近 70B 大模型的 80% 水平。  
- **消融实验**：作者分别去掉负向 KL、去掉优势权重、只保留单一 KL，发现负向约束缺失会导致误报率上升约 4%，优势权重缺失则整体偏好提升下降约 3%。这说明每个模块都有实质贡献。  
- **局限性**：实验主要在英文对话数据上进行，中文或其他低资源语言的迁移尚未验证；此外，优势函数依赖老师的奖励模型质量，若老师本身偏好有偏差，学生会被误导。作者也提到在极端小模型（< 50M 参数）上，优势加权的效果会减弱。

### 影响与延伸思考
这篇工作打开了“小模型对齐”新思路：不再硬逼小模型直接学 RLHF，而是把对齐知识包装成蒸馏信号。随后的几篇论文（如 “Reward‑Distilled Preference Transfer” 与 “Fine‑Grained Preference Distillation”）都在此基础上尝试更高效的优势估计或多老师融合。未来可以探索：① 用多模态老师把视觉偏好也蒸馏进小模型；② 在跨语言场景下，利用多语言大模型的优势函数做跨语言对齐；③ 将优势函数与 LoRA、Adapter 等参数高效微调技术结合，进一步压缩算力需求。对想深入的读者，可以关注“Preference‑Distillation” 方向的最新会议论文和开源实现。

### 一句话记住它
把大模型的偏好优势当作细粒度奖励，用双重 KL 和优势加权蒸馏，让小模型也能“听懂”人类的喜好。