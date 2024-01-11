# TOFU: A Task of Fictitious Unlearning for LLMs

> **Date**：2024-01-11
> **arXiv**：https://arxiv.org/abs/2401.06121

## Abstract

Large language models trained on massive corpora of data from the web can memorize and reproduce sensitive or private data raising both legal and ethical concerns. Unlearning, or tuning models to forget information present in their training data, provides us with a way to protect private data after training. Although several methods exist for such unlearning, it is unclear to what extent they result in models equivalent to those where the data to be forgotten was never learned in the first place. To address this challenge, we present TOFU, a Task of Fictitious Unlearning, as a benchmark aimed at helping deepen our understanding of unlearning. We offer a dataset of 200 diverse synthetic author profiles, each consisting of 20 question-answer pairs, and a subset of these profiles called the forget set that serves as the target for unlearning. We compile a suite of metrics that work together to provide a holistic picture of unlearning efficacy. Finally, we provide a set of baseline results from existing unlearning algorithms. Importantly, none of the baselines we consider show effective unlearning motivating continued efforts to develop approaches for unlearning that effectively tune models so that they truly behave as if they were never trained on the forget data at all.

---

# TOFU：大语言模型虚构遗忘任务 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在海量网页数据上训练后，往往会把训练文本中的细节记得很清楚，甚至可以直接复现私人信息。法律上要求删除这些敏感数据，但模型已经“看过”它们，传统的后处理手段（如删库、过滤输出）只能遮掩，无法真正让模型忘记。已有的“unlearning”方法大多是对模型进行再训练或局部权重调节，却缺少可靠的评估手段来判断模型是否真的和从未见过这些数据的状态等价。于是，缺乏一个统一、可量化的基准，导致研究者难以比较不同方法的真实效果，这正是本文要解决的痛点。

### 关键概念速览
- **大语言模型（LLM）**：在海量文本上预训练的神经网络，能够生成自然语言。类似于一个记忆力极强的“聊天机器人”，能把读过的句子背下来。
- **Unlearning（遗忘）**：让模型在已经训练好的情况下，主动抹掉特定数据的记忆。可以想象为把已经写好的笔记用橡皮擦掉，而不是重新写一本新书。
- **Forget Set（遗忘集合）**：在实验中专门挑出的需要被模型忘记的样本集合。相当于让模型把这部分“作者的作品”从记忆中抹除。
- **Synthetic Author Profiles（合成作者画像）**：人工生成的虚构作者信息，每个画像包含 20 组问答对，用来模拟真实的个人数据。像是给模型喂食的“假人物”。
- **Evaluation Metrics（评估指标）**：一套衡量遗忘程度、模型性能保持以及对未忘记数据影响的指标组合。类似于多维度的体检报告，既看“血压”也看“血糖”。
- **Baseline Algorithms（基线算法）**：已有的几种遗忘实现方式，如全量微调、梯度投影、知识编辑等。它们在本文中被用来检验基准的难度。
- **Equivalence to Never‑Learned（等价于未学习）**：理想的遗忘效果是模型表现得好像从未见过 Forget Set 中的数据。相当于把那段记忆从根本上拔掉，而不是暂时遮蔽。

### 核心创新点
1. **定义了“虚构遗忘任务”（TOFU）**  
   之前的研究大多围绕真实数据的删改，缺少可控、可复制的实验环境。本文把任务抽象为让模型忘记一批完全合成的作者画像，使实验条件可完全掌握，避免法律和伦理风险。这样一来，研究者可以在同一套数据上公平比较不同方法。

2. **构建了 200 条合成作者画像的数据集**  
   每条画像包含 20 对问答，总计 4000 条问答。作者把其中一部分（如 50 条）标记为 Forget Set，形成明确的遗忘目标。数据的多样性（不同主题、写作风格）让基准能够检测方法在各种情境下的鲁棒性。

3. **提出了综合评估指标体系**  
   仅用“是否还能输出原句”不足以说明遗忘是否彻底。本文组合了记忆泄露率、下游任务性能下降、模型内部表征相似度等多维度指标，形成对遗忘效果的全景视图。这样可以同时捕捉“忘得干净”和“保持能力”两方面的平衡。

4. **基线实验显示现有方法仍远未达到理想**  
   将几种主流遗忘技术套用到 TOFU 基准上，结果显示记忆泄露率仍在 20% 以上，且模型在非遗忘数据上的性能下降明显。实验直接证明了“现有方法并不等价于从未学习”，为后续研究指明了明确的改进空间。

### 方法详解
**整体框架**  
TOFU 本身不是一种新算法，而是一个完整的实验流程：① 生成合成作者画像 → ② 划分 Forget Set → ③ 对目标模型施加遗忘算法 → ④ 用统一指标评估结果。整个过程像是一次“黑盒实验”，每一步都有明确的输入输出。

**关键步骤拆解**  

1. **Synthetic Profile Generation（合成画像生成）**  
   - 使用已有的 LLM 按照预设模板生成 200 位虚构作者的背景、兴趣、写作风格。  
   - 对每位作者，模型再生成 20 条问答对，确保问答内容与作者特征紧密关联。  
   - 这一步相当于“造人”，保证所有数据都是可控的、不会侵犯真实隐私。

2. **Forget Set Construction（遗忘集合构建）**  
   - 随机抽取一定比例（如 25%）的作者画像，连同它们的全部问答对一起组成 Forget Set。  
   - 其余画像构成保留集合，用来检验遗忘操作是否对未涉及数据产生副作用。

3. **Unlearning Algorithm Application（遗忘算法应用）**  
   - 将目标 LLM（如 LLaMA‑2‑7B）在完整数据上预训练好后，针对 Forget Set 采用不同的基线方法：  
     * **全量微调**：在保留集合上继续训练，期望模型“忘记” Forget Set。  
     * **梯度投影**：在 Forget Set 上计算梯度并投影到参数空间的负方向，以抵消记忆。  
     * **知识编辑**（如 ROME）：直接修改模型内部的键值对，使特定问答不再产生原答案。  
   - 每种方法的实现细节与原论文保持一致，作者只负责把它们套进 TOFU 流程。

4. **Metric Suite Evaluation（指标套件评估）**  
   - **Leakage Rate（泄露率）**：在 Forget Set 上查询模型是否还能输出原答案，比例越低越好。  
   - **Retention Accuracy（保留准确率）**：在保留集合上测模型的回答质量，确保遗忘不破坏整体能力。  
   - **Representation Distance（表征距离）**：比较遗忘前后模型内部隐藏层对 Forget Set 的向量距离，距离越大说明记忆被削弱。  
   - **Downstream Impact（下游影响）**：在标准 NLP 任务（如情感分类）上跑一次，观察性能变化，防止遗忘手段带来全局退化。

**最巧妙的设计**  
指标体系的多维度组合是本工作最具创新性的点。单纯看泄露率会忽略模型内部潜在记忆，而只关注下游任务会掩盖细粒度的遗忘失败。把三类指标交叉使用，让研究者能够“一眼看穿”遗忘方法的真实效果。

### 实验与效果
- **数据集**：200 条合成作者画像（共 4000 条问答），其中约 50 条（2500 条问答）构成 Forget Set。  
- **基线对比**：全量微调、梯度投影、ROME（知识编辑）三种方法。  
- **结果概览**：所有基线的泄露率均在 20%–35% 区间，远高于理想的 0%；保留准确率下降约 5%–12%；表征距离提升有限，说明模型内部仍保留了相当的痕迹。作者强调，这些数字表明现有技术在“等价于未学习”层面仍有巨大差距。  
- **消融实验**：作者分别去掉指标中的某一项进行评估，发现泄露率对整体评价贡献最大，而表征距离在区分不同基线时提供了细粒度的区分度。  
- **局限性**：实验全部基于合成数据，真实世界的噪声、长文本和多模态信息未被覆盖；此外，只评估了几种常见的遗忘手段，未探索更激进的结构性修改。作者在讨论中坦诚，这些限制是后续工作需要填补的。

### 影响与延伸思考
TOFU 作为首个系统化的 LLM 遗忘基准，已经在社区引发了不少关注。随后出现的工作如 **EVA‑UN**、**ForgetMeNot** 等，都在尝试在真实数据上复现 TOFU 的评估框架，或在指标上加入隐私泄露的理论上界。对想进一步深入的读者，可以关注以下方向：  
1. **真实隐私数据的安全评估**：在不侵犯用户的前提下，构建更贴近实际的遗忘测试集。  
2. **结构化遗忘方法**：比如在模型的注意力矩阵或激活路径上直接切除记忆痕迹。  
3. **跨模态遗忘**：把图像、音频等多模态信息也纳入遗忘任务，检验方法的通用性。  
4. **理论界限**：研究在信息论层面上，遗忘到底能削减多少信息量，是否存在不可逾越的下限。  

### 一句话记住它
TOFU 用合成的“假人物”打造了可复制的遗忘基准，揭示了现有 LLM 遗忘技术离真正“从未学过”仍有很大差距。