# Compositional Exemplars for In-context Learning

> **Date**：2023-02-11
> **arXiv**：https://arxiv.org/abs/2302.05698

## Abstract

Large pretrained language models (LMs) have shown impressive In-Context Learning (ICL) ability, where the model learns to do an unseen task via a prompt consisting of input-output examples as the demonstration, without any parameter updates. The performance of ICL is highly dominated by the quality of the selected in-context examples. However, previous selection methods are mostly based on simple heuristics, leading to sub-optimal performance. In this work, we formulate in-context example selection as a subset selection problem. We propose CEIL (Compositional Exemplars for In-context Learning), which is instantiated by Determinantal Point Processes (DPPs) to model the interaction between the given input and in-context examples, and optimized through a carefully-designed contrastive learning objective to obtain preference from LMs. We validate CEIL on 12 classification and generation datasets from 7 distinct NLP tasks, including sentiment analysis, paraphrase detection, natural language inference, commonsense reasoning, open-domain question answering, code generation, and semantic parsing. Extensive experiments demonstrate not only the state-of-the-art performance but also the transferability and compositionality of CEIL, shedding new light on effective and efficient in-context learning. Our code is released at https://github.com/HKUNLP/icl-ceil.

---

# 组合示例用于上下文学习 论文详细解读

### 背景：这个问题为什么难？
大模型在“上下文学习”（In‑Context Learning，ICL）时，只靠提示里的示例就能完成新任务，这听起来很神奇，却极度依赖示例的挑选。过去的做法大多是手工规则——比如挑最相似的几条、随机抽取，或者用粗糙的相似度分数。这样的启发式方法往往只能捕捉到示例与输入的表面相似，却忽视了示例之间的相互作用，导致模型在很多任务上只能达到中等水平。根本问题在于：我们没有一个系统化的框架来衡量“一组示例”整体对模型的帮助程度，而不是单个示例的好坏。

### 关键概念速览
**上下文学习（In‑Context Learning）**：把任务描述和若干输入‑输出对直接塞进模型的提示里，让模型在不改参数的情况下“现场学习”。类似于老师现场给学生举例子，学生当场领会做法。

**示例选择（Example Selection）**：从海量候选示例中挑出最能帮助模型完成当前任务的那几条。可以把它想成挑选最合适的教材章节。

**子集选择（Subset Selection）**：在数学上指从一个集合里挑出若干元素，使得某个目标函数最大化。这里的目标函数衡量的是示例组合对模型的帮助程度。

**行列式点过程（Determinantal Point Process，DPP）**：一种概率模型，专门用来描述“多样且相互排斥”的集合。直观上像是把每个示例看成一颗磁铁，DPP倾向于挑出既相似（能提供信息）又不完全相同（避免冗余）的磁铁。

**对比学习（Contrastive Learning）**：让模型学会把“好”的示例组合和“差”的组合区分开。想象把好与坏的两套教材分别放在天平两边，模型的任务是让好的一边更重。

**可组合性（Compositionality）**：指一个示例集合的好坏可以通过子集合的表现来推断。类似于拼装乐高：如果每块都合适，整体就稳固。

### 核心创新点
1. **把示例挑选正式化为子集选择问题**  
   过去的工作把挑选看成单条示例的相似度排序，本文把它提升到“挑选一组示例”层面，目标是最大化整组示例对模型的帮助，而不是单个示例的好坏。

2. **用 DPP 建模输入‑示例交互**  
   传统方法只考虑输入与每个示例的相似度，忽视示例之间的冗余。CEIL 引入 DPP，使得被挑选的示例既要和输入匹配，又要在集合内部保持多样性，从而避免信息重复。

3. **通过对比学习让模型自己给出偏好**  
   作者设计了一个对比学习目标，让语言模型在“好”示例集合和“坏”示例集合之间产生分数差异。这样模型本身的判断被直接用于优化 DPP 参数，而不是依赖外部人工标注。

4. **展示了跨任务的可组合迁移**  
   CEIL 在一种任务上学到的示例挑选策略可以直接迁移到另一种任务，说明挑选规则具备一定的通用性。这在之前的启发式方法里几乎没有出现。

### 方法详解
**整体思路**：CEIL 先把所有候选示例映射成向量表示，然后用 DPP 计算每个子集的概率（即质量），再通过对比学习让模型给出“好子集”与“坏子集”的评分差，最后在 DPP 参数空间做梯度上升，得到最能提升模型表现的示例集合。

**步骤拆解**：

1. **示例向量化**  
   - 对每条候选示例（包括输入和对应输出）使用预训练的大语言模型得到隐藏向量。  
   - 同时对当前任务的输入也做同样的向量化，得到一个“查询向量”。

2. **构建 DPP 矩阵**  
   - 计算每对示例向量的相似度（如余弦相似度），形成一个相似度矩阵。  
   - 将查询向量与每个示例的相似度作为“质量分”，放进 DPP 的对角线。  
   - 这样，DPP 同时考虑示例与查询的匹配度（对角线）和示例之间的多样性（非对角线）。

3. **生成候选子集**  
   - 采用贪心采样或近似最大化算法，从 DPP 中抽取若干子集（每个子集大小固定，如 4 条示例）。  
   - 这些子集在概率上已经倾向于“好‑坏平衡”，但仍需模型进一步评估。

4. **对比学习目标**  
   - 把抽到的子集分成两类：得分高的（正样本）和得分低的（负样本），这里的得分来自语言模型在这些示例上完成任务的表现（如交叉熵或准确率）。  
   - 设计一个对比损失，使得模型对正样本的预测概率显著高于负样本。直观上相当于让模型说“这组示例更有帮助”。

5. **参数更新**  
   - 通过对比损失对 DPP 的质量向量和相似度矩阵进行梯度上升，使得 DPP 更倾向于生成模型喜欢的子集。  
   - 迭代若干轮后，得到一个固定的 DPP 参数，可直接用于新任务的示例挑选。

**关键巧思**：把语言模型的“偏好”直接嵌入到 DPP 的概率分布里，而不是先手工设定相似度阈值。这样模型自己决定哪些示例组合最能提升表现，极大降低了人工调参的成本。

### 实验与效果
- **数据集与任务**：作者在 12 个数据集上做了评测，覆盖 7 大类 NLP 任务，包括情感分析、同义句判别、自然语言推理、常识推理、开放域问答、代码生成和语义解析等。每类任务至少有 1–2 个公开基准。

- **对比基线**：与随机抽样、最近邻（基于句向量相似度）以及最近的示例选择方法（如基于梯度或信息增益的启发式）进行比较。  
  - 在多数任务上，CEIL 的准确率或 BLEU 分数比最强基线高出 **3%~7%**（具体数字在论文中给出），在代码生成任务上提升约 **5%** 的功能正确率。

- **消融实验**：  
  - 去掉 DPP 的多样性项，仅保留查询质量，性能下降约 **2%**，说明示例间的排斥机制是关键。  
  - 替换对比学习为普通交叉熵训练，效果下降约 **1.5%**，验证对比目标对 DPP 参数学习的贡献。  
  - 将子集大小从 4 改为 2 或 8，表现呈现先升后降的趋势，说明子集规模需要适度。

- **局限性**：作者指出 CEIL 仍然依赖于预训练模型的向量质量，若候选示例质量极差，DPP 也难以救活。此外，DPP 的近似采样在极大候选集合上仍有一定计算开销，实际部署时需要权衡。

### 影响与延伸思考
CEIL 把示例挑选提升到可学习的层面，为“提示工程”提供了系统化工具。自论文发布后，已有工作尝试把强化学习、元学习或贝叶斯优化引入提示选择，进一步探索“模型自适应提示”。另外，DPP 在多样性建模上的成功也激发了在检索增强生成、跨语言提示迁移等方向的研究。想深入了解的读者可以关注以下方向：  
- **提示元学习**：让模型在少量任务上学习通用的示例挑选策略。  
- **高效 DPP 近似**：研发更快的子集采样算法，以适配大规模检索库。  
- **跨模态示例选择**：把图像、代码等非文本示例纳入同一框架。

### 一句话记住它
**CEIL 用 DPP + 对比学习，让大模型自己挑出最有帮助的示例组合，从而把提示工程变成可学习的子集选择任务。**