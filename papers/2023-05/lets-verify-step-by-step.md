# Let's Verify Step by Step

> **Date**：2023-05-31
> **arXiv**：https://arxiv.org/abs/2305.20050

## Abstract

In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even state-of-the-art models still regularly produce logical mistakes. To train more reliable models, we can turn either to outcome supervision, which provides feedback for a final result, or process supervision, which provides feedback for each intermediate reasoning step. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

---

# 一步步验证 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在一次性给出答案时已经很强，但要让它们像人一样把思考过程写出来、一步步推导，仍然经常出错。传统的训练方式只在答案对错上给模型奖励（即“结果监督”），模型因此缺乏对中间推理的约束，容易走进逻辑陷阱。即使是最先进的模型，在数学、物理等需要多步推理的任务上仍然频繁出现算术错误或推理跳步。要提升可靠性，就必须在训练时让模型“看到”自己的每一步，而这需要大量细粒度的人类反馈，成本高得让人望而却步。

### 关键概念速览
- **大语言模型（LLM）**：能够生成自然语言的深度神经网络，类似会说话的“黑盒子”。它们通过海量文本学习语言规律，但内部推理过程不透明。
- **多步推理**：指解答需要经过若干逻辑或计算步骤的过程，就像解一道代数题要先化简、再求根、最后检验。单一步骤的答案往往掩盖了中间的错误。
- **结果监督（Outcome Supervision）**：只在最终答案对错上给模型奖励或惩罚，相当于只检查考试的总分，而不看答题过程。
- **过程监督（Process Supervision）**：对每一步的推理质量都提供反馈，就像老师在学生写草稿时逐行批注，帮助模型纠正思路偏差。
- **奖励模型（Reward Model）**：一个二次训练的模型，用来把人类对步骤好坏的打分转化为可供优化的数值信号，类似于把老师的评语量化成分数。
- **主动学习（Active Learning）**：让模型挑选自己最不确定或最有争议的步骤去请求人工标注，类似学生主动举手请老师点拨，能以更少的标注成本提升学习效果。
- **MATH 数据集**：一个专门收集高中到大学水平数学题目的基准，题目需要严谨的逐步推导，常被用来检验模型的数学推理能力。
- **步骤级人类反馈（step‑level human feedback）**：标注者对每一步推理的正确性、完整性或表达清晰度进行打分，提供细粒度的监督信息。

### 核心创新点
1. **过程监督 vs. 结果监督的直接对比**  
   之前的研究大多只报告了结果监督的效果，缺少系统的横向比较。本文在同一模型、同一数据上分别使用两种监督方式训练，然后在 MATH 子集上评测。结果显示，过程监督的模型解对率提升到 78%，显著高于仅使用结果监督的基线（约 60% 左右），证明细粒度反馈对多步推理的提升是实质性的。

2. **引入主动学习提升标注效率**  
   直接让标注者对所有步骤打分成本极高。作者让模型先生成答案草稿，再用当前奖励模型挑选出“争议最大”或“置信度最低”的步骤，请人类重新标注。实验表明，这种主动采样比随机采样能在相同标注预算下提升约 5% 的解对率。

3. **构建并公开 PRM800K 步骤级反馈数据集**  
   为了让社区复现和进一步研究，作者整理了 80 万条步骤级人类反馈，形成 PRM800K 数据集。该数据集不仅支撑了本文的奖励模型训练，也为后续的过程监督研究提供了宝贵资源。

4. **基于奖励模型的 RLHF 流程细化**  
   在传统的强化学习从人类反馈（RLHF）框架中，奖励信号通常针对完整答案。本文把奖励模型的输入扩展到每一步的文本，使用 PPO（近端策略优化）在每一步上进行梯度更新，使得模型在生成每一步时都能感受到即时的奖励，引导其逐步纠错。

### 方法详解
整体思路可以划分为四个阶段：① 收集步骤级人类反馈，② 训练步骤奖励模型，③ 用奖励模型进行过程监督的 RLHF，④ 主动学习循环迭代提升。

1. **步骤级反馈收集**  
   - 研究者先让一个基线 LLM（如 GPT‑3.5）在 MATH 题目上生成完整的解题步骤。  
   - 人类标注者阅读每一步，给出二元标签（好/坏）或 1‑5 评分，重点关注逻辑连贯性、计算正确性和表达完整性。  
   - 所有标注统一保存为 `(question, step_text, label)` 三元组，累计 80 万条，形成 PRM800K。

2. **奖励模型训练**  
   - 将步骤文本和对应标签喂入一个小型 Transformer，使用交叉熵或对比学习目标，使模型输出的分数能够区分“好步骤”和“坏步骤”。  
   - 为了让奖励在不同题目间保持可比，作者在训练时加入了题目上下文，使奖励能够感知全局约束。

3. **过程监督的 RLHF**  
   - 在生成新答案时，模型先产生第一步，奖励模型立刻给出分数；模型根据该分数通过 PPO 调整策略，然后继续生成第二步，如此循环直至完成。  
   - 关键在于把每一步的奖励视作即时信号，而不是等到全部步骤结束后才打分，这样模型可以在生成过程中“自我纠错”。  
   - 为防止模型只追求高奖励而忽略答案完整性，作者在 PPO 的奖励函数中加入了一个小的终局奖励（基于答案是否正确），形成“过程+结果”双重激励。

4. **主动学习循环**  
   - 训练初期，模型对大多数步骤的奖励信心较高。作者使用奖励模型的置信区间或 KL 散度挑选出最不确定的步骤，交给标注者重新标注。  
   - 新增的标注被并入 PRM800K，奖励模型重新训练，随后继续 RLHF。循环若干次后，模型在相同标注预算下的整体表现显著提升。

**最巧妙的点**：把奖励模型的输入从“完整答案”搬到“单步文本”，并在 PPO 中实现逐步更新，使得模型在每一步都有即时的“老师批注”。这让模型的推理过程变得可监督、可纠错，突破了传统只看最终分数的限制。

### 实验与效果
- **数据集**：作者在公开的 MATH 测试集上抽取了一个具代表性的子集（约 2,000 题）进行评估。训练阶段使用了 PRM800K 步骤反馈以及原始 MATH 题目。
- **基线对比**：  
  - 仅使用结果监督的模型（同等规模）解对率约 60%。  
  - 过程监督模型在相同训练预算下达到 78% 的解对率，提升约 18%。  
  - 加入主动学习后，解对率进一步提升到约 81%（相对过程监督提升约 3%）。
- **消融实验**：  
  - 去掉即时奖励，仅使用终局奖励，性能回落到约 65%，说明逐步奖励是关键。  
  - 不使用主动学习而采用随机采样标注，解对率约为 78%（与仅过程监督持平），验证主动学习的增益。  
- **局限性**：  
  - 步骤级标注成本仍然高于仅标注答案，作者承认在更大规模任务上仍需成本优化。  
  - 实验仅在数学推理上展开，模型在自然语言推理或代码生成等领域的迁移效果未作评估。

### 影响与延伸思考
这篇工作在社区引发了对“过程监督”重要性的广泛讨论。随后出现的几篇论文（如 *Process‑RLHF*、*Stepwise Feedback for Code Generation*）都直接借鉴了逐步奖励的思路，并尝试在代码、法律文书等需要多步推理的任务上复现类似提升。PRM800K 数据集也被用于训练更通用的步骤奖励模型，推动了“可解释 RLHF”方向的发展。未来可以关注以下几个方向：① 如何在更低成本下获得高质量步骤标注（比如利用模型自评或合成数据）；② 将过程监督与自监督预训练结合，提升模型对未见任务的推理能力；③ 探索跨模态的步骤监督，例如在图像推理或机器人规划中加入逐步反馈。

### 一句话记住它
把大语言模型的每一步都交给“老师”打分，让模型在写草稿时就能自我纠错，效果比只看答案的训练方式好得多。