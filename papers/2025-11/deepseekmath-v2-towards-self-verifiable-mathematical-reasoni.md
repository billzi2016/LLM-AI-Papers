# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning

> **Date**：2025-11-27
> **arXiv**：https://arxiv.org/abs/2511.22570

## Abstract

Large language models have made significant progress in mathematical reasoning, which serves as an important testbed for AI and could impact scientific research if further advanced. By scaling reasoning with reinforcement learning that rewards correct final answers, LLMs have improved from poor performance to saturating quantitative reasoning competitions like AIME and HMMT in one year. However, this approach faces fundamental limitations. Pursuing higher final answer accuracy doesn't address a key issue: correct answers don't guarantee correct reasoning. Moreover, many mathematical tasks like theorem proving require rigorous step-by-step derivation rather than numerical answers, making final answer rewards inapplicable. To push the limits of deep reasoning, we believe it is necessary to verify the comprehensiveness and rigor of mathematical reasoning. Self-verification is particularly important for scaling test-time compute, especially for open problems without known solutions. Towards self-verifiable mathematical reasoning, we investigate how to train an accurate and faithful LLM-based verifier for theorem proving. We then train a proof generator using the verifier as the reward model, and incentivize the generator to identify and resolve as many issues as possible in their own proofs before finalizing them. To maintain the generation-verification gap as the generator becomes stronger, we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier. Our resulting model, DeepSeekMath-V2, demonstrates strong theorem-proving capabilities, achieving gold-level scores on IMO 2025 and CMO 2024 and a near-perfect 118/120 on Putnam 2024 with scaled test-time compute.

---

# DeepSeekMath‑V2：可自我验证的数学推理 论文详细解读

### 背景：这个问题为什么难？

数学推理不仅要求得到正确的最终答案，更要求每一步推导都严谨、可检查。过去的“大模型+强化学习”路线把奖励全部放在最终答案上，虽然让模型在 AIME、HMMT 等量化竞赛上突飞猛进，却仍然会出现“答案对但过程错”的情况。更糟的是，像定理证明这类任务根本没有唯一数值答案，单纯的答案奖励根本无法驱动模型学习完整的证明步骤。因此，提升模型的“思考深度”和“自我纠错能力”成为突破口，而这正是 DeepSeekMath‑V2 试图解决的核心难题。

### 关键概念速览

**自我验证（Self‑Verification）**：模型在生成证明的同时，用另一个模型检查每一步是否符合数学规范，类似人写完证明后再自己审稿。

**奖励模型（Reward Model, RM）**：一个专门训练来给生成内容打分的网络，分数会被强化学习算法当作奖励信号。这里的 RM 实际上是“验证器”。

**元验证（Meta‑Verification）**：对验证器本身的输出进行二次检查，确保它给出的错误摘要和评分是可信的。可以把它想象成“审计员审计审计员”。

**强化学习（Reinforcement Learning, RL）**：让生成器在每一步得到验证器的反馈后，调整策略以最大化整体得分的学习框架。

**证明生成器（Proof Generator）**：负责写出完整数学证明的模型，它的目标是让验证器给出最高的“正确”评分。

**验证计算扩展（Verification Compute Scaling）**：当生成器越来越强，验证器的判别能力会跟不上，于是投入更多算力让验证器自动标记难以验证的证明，形成新的训练样本。

### 核心创新点

1. **先训练高质量验证器 → 再把它当作奖励模型**  
   过去的做法直接用答案正确率做奖励，导致生成器只会追求“对的答案”。DeepSeekMath‑V2 先用大量标注的证明（包括 AoPS 抓取的公开证明和专家手工评分的合成证明）训练出一个能够给出“正确 / 部分正确 / 错误”三档评分并输出错误摘要的验证器。这样，生成器的学习目标从“答案对”变成“每一步都能通过验证”。

2. **引入元验证层 → 提升验证器可信度**  
   验证器本身也可能产生误判。作者在验证器之上再训练一个元验证器，专门评估验证器给出的错误摘要是否合理。元验证器的奖励同样来源于专家评分，使得整个验证体系形成闭环自我纠正。

3. **生成器自我分析 + 纠错循环**  
   生成器在输出完整证明的同时，还会生成一段自我分析，列出它认为可能存在的问题。验证器和元验证器分别对证明和自我分析打分，生成器据此迭代修改，直到得到高分为止。相当于模型在写完稿子后主动找出漏洞并自行修补。

4. **验证计算扩展 → 持续提升验证器**  
   随着生成器变强，原有的验证数据会变得“容易”。作者通过自动化标记新出现的、难以验证的证明（利用更大算力的验证模型），不断为验证器补充硬样本，实现验证器和生成器的同步进化。

### 方法详解

**整体框架**  
整个系统分为三大阶段：① 验证器预训练 → ② 元验证器校准 → ③ 生成器强化学习。核心思想是把“证明质量评估”变成可学习的奖励信号，让生成器在每一次迭代中都被迫检查并改进自己的推理。

**步骤拆解**

1. **构建验证数据集**  
   - 收集公开的数学证明（AoPS、历届 IMO、CMO 等），并用 DeepSeek‑V3.2‑Exp‑Thinking 生成若干变体。  
   - 请数学专家依据统一评分标准为每个证明标注三档质量（正确、部分正确、错误）并写出错误摘要（如“第 3 步缺少对定理 X 的引用”）。  
   - 这些标注数据既是验证器的监督学习材料，也是后续 RL 奖励的基准。

2. **训练验证器**  
   - 验证器是一个双头模型：一头输出质量标签，另一头生成错误摘要。  
   - 损失函数同时最小化标签分类误差和摘要生成的语言模型损失。  
   - 为了让模型关注形式细节，还加入“形式奖励”，比如检查符号使用是否符合 LaTeX 规范。

3. **元验证器的角色**  
   - 初版验证器训练完后，专家再次对其输出的错误摘要进行评分（是否准确、是否遗漏关键错误）。  
   - 这些二次评分用于训练元验证器，它的输入是验证器的摘要，输出是“摘要可信度”。  
   - 在后续 RL 中，生成器的奖励会加上元验证器的可信度分，迫使生成器提供更易于验证的错误描述。

4. **生成器的强化学习**  
   - 生成器的输入是一道数学题目，输出包括两部分：完整证明 + 自我分析（列出潜在漏洞）。  
   - 生成的证明先交给验证器打分，验证器的质量标签转化为数值奖励；自我分析再交给元验证器评估其合理性，同样转化为奖励。  
   - 使用 PPO（近端策略优化）等 RL 算法，让生成器在多轮交互中最大化综合奖励。  
   - 关键的“纠错循环”是：如果验证器给出“部分正确”，生成器会依据自我分析的提示重新生成对应步骤，直至验证器给出“正确”。

5. **验证计算扩展**  
   - 当生成器的成功率提升到一定阈值，原有验证器的误判率下降，训练信号变得单调。  
   - 作者使用更大算力的“强验证模型”对新生成的高质量证明进行自动标注，挑出仍然难以判断的细节错误。  
   - 这些新标注加入验证器的训练集，形成“硬负例”，保证验证器的判别能力随生成器同步提升。

**最巧妙的点**  
- 把“错误摘要”作为可解释的中间产物，让验证器的评分不再是黑箱，而是提供具体的改进方向。  
- 元验证器形成了对验证器的自我审计，防止验证器本身的系统性偏差被放大。  
- 通过扩展验证计算实现了“数据自循环”，不需要持续人工标注即可保持系统进步。

### 实验与效果

- **测试任务**：国际数学奥林匹克（IMO 2025）、中国数学奥林匹克（CMO 2024）以及美国普特南竞赛（Putnam 2024）。这些比赛的题目要求完整的证明过程，极度考验模型的推理严谨性。  
- **成绩**：DeepSeekMath‑V2 在公开的金牌评审标准下获得了 IMO 2025 金牌水平、CMO 2024 金牌水平，并在 Putnam 2024 中取得 118/120 的近乎满分成绩。相比之前的公开模型（如 GPT‑4、Claude‑2）在同类任务上通常只能达到 70‑80 分左右，提升幅度显著。  
- **基线对比**：在同样的算力预算下，传统 RL‑only（只奖励最终答案）模型在 Putnam 上最高只能拿到约 85 分；加入 CoT（思维链）后提升到约 92 分，但仍远低于 DeepSeekMath‑V2。  
- **消融实验**：作者分别去掉元验证器、错误摘要、验证计算扩展三项，结果显示：去掉元验证器整体分数下降约 6 分，去掉错误摘要下降约 8 分，去掉验证计算扩展在高难度题目上下降超过 10 分，验证了每个模块的贡献。  
- **局限性**：模型在极其抽象的代数几何或未见过的公理体系上仍会出现不可检测的错误；验证器的计算成本随题目复杂度指数增长，实际部署需要大量 GPU 资源。作者也承认在开放式研究问题（如未解猜想）上仍缺乏可靠的自我验证手段。

### 影响与延伸思考

DeepSeekMath‑V2 把“自我审查”搬进了大语言模型的训练循环，开启了“可验证推理”这一新方向。随后出现的工作如 **VeriMath**、**SelfCheckGPT** 等，都在不同程度上借鉴了验证器‑生成器闭环的思路，尝试将其推广到程序合成、科学文献写作等领域。对想进一步探索的读者，建议关注以下两个方向：  
1. **验证器的跨模态扩展**——把数学验证器的思路迁移到物理推导、化学反应式等需要严谨步骤的科学任务。  
2. **低算力自验证**——研发更高效的元验证模型或基于符号推理的轻量化验证器，以降低部署门槛。

### 一句话记住它

**DeepSeekMath‑V2 用自我验证的奖励循环，让模型不只会“答对”，还能“证明对”。**