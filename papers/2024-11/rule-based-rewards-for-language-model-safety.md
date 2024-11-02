# Rule Based Rewards for Language Model Safety

> **Date**：2024-11-02
> **arXiv**：https://arxiv.org/abs/2411.01111

## Abstract

Reinforcement learning based fine-tuning of large language models (LLMs) on human preferences has been shown to enhance both their capabilities and safety behavior. However, in cases related to safety, without precise instructions to human annotators, the data collected may cause the model to become overly cautious, or to respond in an undesirable style, such as being judgmental. Additionally, as model capabilities and usage patterns evolve, there may be a costly need to add or relabel data to modify safety behavior. We propose a novel preference modeling approach that utilizes AI feedback and only requires a small amount of human data. Our method, Rule Based Rewards (RBR), uses a collection of rules for desired or undesired behaviors (e.g. refusals should not be judgmental) along with a LLM grader. In contrast to prior methods using AI feedback, our method uses fine-grained, composable, LLM-graded few-shot prompts as reward directly in RL training, resulting in greater control, accuracy and ease of updating. We show that RBRs are an effective training method, achieving an F1 score of 97.1, compared to a human-feedback baseline of 91.7, resulting in much higher safety-behavior accuracy through better balancing usefulness and safety.

---

# 基于规则的奖励用于语言模型安全 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做强化学习（RL）微调时，常用人类偏好数据来引导模型既能提供有用答案，又不冒险违规。然而，安全相关的对话往往缺少明确的标注指引，标注者的主观判断会让模型变得过度保守，甚至出现“评判式”回复。随着模型能力提升和使用场景多样化，原有的安全数据集需要频繁增补或重新标注，成本高得吓人。换句话说，传统的“人类反馈 + RL”管线在保持安全性的同时，难以快速、低成本地做细粒度的行为调控。

### 关键概念速览
- **强化学习微调（RLHF）**：先让模型生成答案，再用一个奖励模型评估答案好坏，最后通过强化学习让模型倾向高奖励的输出。类似于让学生先写作文，再让老师打分，学生根据分数改进写作。
- **人类偏好数据**：标注者对两段模型回复的喜好比较，用来训练奖励模型。相当于让人类挑选“更好”的答案，机器学习模仿这种挑选规则。
- **规则（Rule）**：对期望或禁止的行为写成可机器检查的条件，例如“拒绝时不能使用评判性语言”。把它想成交通规则，车子必须遵守，否则会被罚分。
- **LLM 评分器（LLM grader）**：利用另一个大语言模型对给定的回复进行打分或判断是否符合规则。相当于让一位“智能裁判”来检查答案是否遵守交通规则。
- **Few‑shot Prompt**：在提示中给出少量示例，让模型学习如何在新情境下执行相同的任务。像是老师给学生几道例题，学生据此解答新题。
- **可组合（Composable）奖励**：把多个规则的判断结果按一定权重合并成最终奖励，类似于把不同交通标志的罚分累加得到总罚分。

### 核心创新点
1. **从人类反馈转向规则驱动的奖励**  
   之前的做法是大量收集人类对比数据，再训练奖励模型；本论文直接用一套可编辑的规则集合，让 LLM 评分器对每条规则进行细粒度评估。这样做把“人工标注”压缩到几条规则上，显著降低了标注成本。

2. **把 LLM 评分器的输出当作即时奖励**  
   传统方法先训练一个固定的奖励模型，然后在 RL 里使用；这里把 LLM 评分器的 few‑shot 推理过程直接嵌入 RL 环境，每一步都实时计算规则匹配度。结果是模型在训练时能更精准地感知哪些行为违反了哪条规则。

3. **奖励的可组合与可更新特性**  
   规则之间可以自由加权或增删，更新只需要改动规则列表或权重，而不必重新收集大量对比数据。相当于在交通系统里增设新标志，只要更新地图即可，无需重新训练所有司机。

4. **在安全性与有用性之间实现更好的平衡**  
   通过实验发现，基于规则的奖励在保持模型有用性的同时，显著提升了安全行为的准确率（F1 从 91.7 提升到 97.1），说明规则约束并没有牺牲模型的实用性。

### 方法详解
**整体框架**  
整个流程可以划分为三步：① 编写安全规则库；② 用 LLM 评分器对每条规则进行 few‑shot 推理，得到规则匹配分数；③ 将这些分数加权合成最终奖励，喂给强化学习算法（如 PPO）进行微调。核心思想是把“规则检查”变成 RL 环境中的即时奖励信号。

**步骤拆解**  

1. **规则库构建**  
   - 研究团队先列出一系列安全行为的正负例，例如“拒绝时不应使用‘你太蠢了’”或“对敏感话题应给出中立解释”。  
   - 每条规则用自然语言描述，便于后续 LLM 理解。

2. **LLM 评分器设计**  
   - 选用一个与目标模型规模相近的语言模型作为评分器。  
   - 为每条规则准备 few‑shot 示例：给出几段符合规则的回复和几段违规的回复，让评分器学习判断的模式。  
   - 在实际训练时，模型生成的回复会被送入评分器，评分器输出一个二元或连续的匹配分数（如 0 表示违规，1 表示完全符合）。

3. **奖励合成**  
   - 对所有规则的分数进行加权求和，权重可以手动调节或通过小规模验证集自动学习。  
   - 为了防止单条规则过度主导，作者加入了归一化和上限限制，使得总奖励保持在合理区间。

4. **RL 微调**  
   - 使用常见的近端策略优化（PPO）算法，将合成奖励作为优化目标。  
   - 训练过程中，模型每生成一次回复，就会立即得到基于规则的奖励，指导策略更新。  

**巧妙之处**  
- **即时规则评估**：把 LLM 评分器的推理过程直接嵌入 RL 环境，而不是先离线训练一个固定奖励模型，这让奖励更贴合当前上下文。  
- **规则可组合**：通过加权方式，团队可以灵活平衡不同安全需求，例如在某些场景下更强调“非评判性”，在另一些场景下更看重“信息完整”。  
- **低标注需求**：只需要少量 few‑shot 示例即可让评分器学会规则判断，省去了大规模人类对比数据的收集。

### 实验与效果
- **测试任务**：作者在公开的安全对话基准上评估模型，包括拒绝不当请求、避免评判性语言以及提供中立信息等多项安全指标。  
- **对比基线**：主要与传统的人类反馈（Human‑Feedback）RLHF 方法比较。  
- **核心结果**：基于规则的奖励在安全行为的 F1 分数上达到 97.1%，而人类反馈基线为 91.7%；在保持有用性（如回答准确率）方面，两者差距不大，说明规则并未削弱模型的实用能力。  
- **消融实验**：作者分别去掉规则加权、去除 few‑shot 示例、以及使用固定奖励模型进行对比，发现：  
  1) 去掉加权会导致某些规则过度惩罚，整体 F1 下降约 3%。  
  2) 只用单一示例的评分器准确率下降约 2%。  
  3) 替换为离线训练的奖励模型后，实时性丧失导致安全行为提升幅度减半。  
- **局限性**：论文承认规则的覆盖范围仍受限于人工编写的质量；对于极其细微或跨领域的安全需求，仍可能需要大量人类示例来补足。

### 影响与延伸思考
这篇工作向社区展示了“规则 + LLM 评分器”可以在安全微调中提供高效、可维护的解决方案。随后出现的几篇论文尝试把规则形式化为可微分的逻辑层（如 Neuro‑Symbolic RL），或把规则库与自动化规则生成（通过大模型自我审查）结合，进一步降低人工成本。对想深入的读者，可以关注以下方向：  
- **规则自动发现**：利用大模型在海量对话中挖掘潜在安全风险，自动生成规则。  
- **多模态安全规则**：把文本规则扩展到图像、音频等跨模态场景。  
- **可解释奖励**：将规则的权重和评分过程可视化，帮助开发者快速定位模型的安全失误。  

### 一句话记住它
用一套可编辑的安全规则，让大模型自己即时打分，既省标注又能把安全行为的准确率推到 97% 以上。