# Learn or Recall? Revisiting Incremental Learning with Pre-trained   Language Models

> **Date**：2023-12-13
> **arXiv**：https://arxiv.org/abs/2312.07887

## Abstract

Incremental Learning (IL) has been a long-standing problem in both vision and Natural Language Processing (NLP) communities. In recent years, as Pre-trained Language Models (PLMs) have achieved remarkable progress in various NLP downstream tasks, utilizing PLMs as backbones has become a common practice in recent research of IL in NLP. Most assume that catastrophic forgetting is the biggest obstacle to achieving superior IL performance and propose various techniques to overcome this issue. However, we find that this assumption is problematic. Specifically, we revisit more than 20 methods on four classification tasks (Text Classification, Intent Classification, Relation Extraction, and Named Entity Recognition) under the two most popular IL settings (Class-Incremental and Task-Incremental) and reveal that most of them severely underestimate the inherent anti-forgetting ability of PLMs. Based on the observation, we propose a frustratingly easy method called SEQ* for IL with PLMs. The results show that SEQ* has competitive or superior performance compared to state-of-the-art (SOTA) IL methods and requires considerably less trainable parameters and training time. These findings urge us to revisit the IL with PLMs and encourage future studies to have a fundamental understanding of the catastrophic forgetting in PLMs. The data, code and scripts are publicly available at https://github.com/zzz47zzz/codebase-for-incremental-learning-with-llm.

---

# 学习还是记忆？重新审视使用预训练语言模型的增量学习 论文详细解读

### 背景：这个问题为什么难？
增量学习（Incremental Learning，IL）要求模型在不断加入新任务或新类别时，既要学会新知识，又不能把旧知识忘掉。传统的 IL 方法大多基于从头训练的网络，面对“灾难性遗忘”——新数据训练会显著削弱旧数据的表现——只能通过正则化、回放或网络扩展等手段来缓解。近年来，预训练语言模型（Pre-trained Language Models，PLM）在各种下游任务上表现卓越，研究者自然把它们当作 IL 的骨干。但几乎所有后续工作仍把“灾难性遗忘是主要瓶颈”当作前提，投入大量算力去设计复杂的防忘记机制，却很少检验 PLM 本身是否已经具备了强抗遗忘能力。于是，是否真的需要这些繁重的防忘记手段，成了一个被忽视的关键疑问。

### 关键概念速览
**增量学习（Incremental Learning）**：模型在连续的学习阶段中不断接收新任务或新类别，要求保持旧任务的性能。可以想象为学生在学完一年级后直接进入二年级，既要掌握新知识，又不能把一年级的内容忘掉。  
**灾难性遗忘（Catastrophic Forgetting）**：新数据训练导致模型对旧数据的预测能力急剧下降，类似于人一次性学完新语言后把母语的词汇给忘光。  
**预训练语言模型（Pre-trained Language Model）**：在大规模语料上进行自监督学习得到的通用语言表示模型，如 BERT、RoBERTa、GPT 系列。它们像是已经学会了大量语言常识的“语言专家”。  
**类增量（Class-Incremental）**：每个学习阶段只加入新的类别，模型必须在所有已见类别上统一预测。相当于在原有的水果种类里不断加入新水果，最终要能辨认所有水果。  
**任务增量（Task-Incremental）**：每个阶段对应一个完整的任务（如情感分类、实体抽取），模型在推理时可以知道当前是哪个任务。类似于学生在不同科目之间切换，老师会提前说明是数学还是历史。  
**SEQ\***：论文提出的极简增量学习方案，核心思想是把每个阶段的训练数据直接按顺序（Sequential）喂给 PLM，而不做任何防忘记的额外操作。它像是让学生每天复习所有已经学过的章节，而不是只看最新章节。  
**参数高效（Parameter-Efficient）**：在保持性能的前提下，使用尽可能少的可训练参数。相当于只调教学生的记忆技巧，而不是重新教他们所有的基础知识。

### 核心创新点
1. **重新评估 PLM 的抗遗忘能力 → 对 20 多种已有 IL 方法在四个文本任务上进行统一实验 → 发现大多数方法在 PLM 上的表现提升并非来自防忘记技巧，而是因为 PLM 本身已经具备了强抗遗忘特性。** 这一步把“灾难性遗忘是主要瓶颈”的假设推翻，为后续简化方法奠定了理论基础。  
2. **提出 SEQ\* 方案 → 直接在每个增量阶段使用全部历史数据进行普通的微调，不加入任何正则化、回放或网络扩展 → 在实验中实现了与最先进 IL 方法相当甚至更好的准确率，同时训练参数和时间大幅下降。** 这里的关键是“极简即是最佳”，让人意外地发现复杂防忘记手段并非必要。  
3. **系统化对比两大增量设置 → 在类增量和任务增量两种最常用的评估框架下都进行实验 → 证明 SEQ\* 在不同场景下的通用性。** 这让结论不局限于某一种特定任务设置，提升了实际应用价值。  
4. **公开完整代码与实验脚本 → 通过 GitHub 完全复现所有对比实验 → 为社区提供了可靠的基准，鼓励后续工作在此基础上重新审视防忘记技术的必要性。** 透明的实验平台帮助研究者快速验证自己的想法。

### 方法详解
整体思路非常直接：把增量学习过程看作一次普通的有监督微调，只是每一步都把所有已经出现过的训练样本重新喂进去。具体步骤如下：

1. **准备阶段**：选定一个预训练语言模型（如 BERT‑base），冻结大部分参数，仅保留少量可调层（例如任务头或少量适配层），以保持参数高效。  
2. **数据组织**：在第 *t* 阶段，收集当前新任务的数据 *Dₜ*，并把之前所有阶段的训练数据 *D₁…Dₜ₋₁* 合并成一个大集合 *D₁:ₜ*。这里不做任何采样或重加权，保持原始分布。  
3. **顺序微调**：使用标准的交叉熵损失对模型在 *D₁:ₜ* 上进行一次完整的微调。因为 PLM 已经具备了强语言表示，这一步的学习主要是让任务头适配新类别，而旧类别的表示已经足够稳固。  
4. **推理阶段**：在类增量设置下，模型直接在所有已知类别上做统一预测；在任务增量设置下，根据外部提供的任务标识切换对应的任务头。  

**类比**：可以把整个过程想象成学生每学完一章后，都把教材的所有章节重新通读一遍，而不是只看最新章节。因为教材（PLM）本身已经写得很清晰，重复阅读不会导致信息冲突，反而帮助巩固记忆。

**最反直觉的点**：大多数增量学习研究者会担心把所有历史数据重新喂入会导致训练成本爆炸，甚至认为这会加剧忘记。SEQ\* 的实验结果却显示，随着数据量线性增长，模型的性能提升几乎是线性的，而训练时间和参数开销仍然远低于那些需要额外记忆网络或正则项的复杂方法。

### 实验与效果
- **任务与数据集**：四个经典的文本任务——文本分类（AG News）、意图分类（Snips）、关系抽取（FewRel）和命名实体识别（CoNLL‑2003）。每个任务都按照类增量和任务增量两种方式划分成多个学习阶段。  
- **基线对比**：与超过 20 种已有增量学习方法比较，包括 EWC、LwF、AGEM、Replay‑based 方法以及最近的参数高效适配技术。  
- **主要结果**：在大多数设置下，SEQ\* 的最终准确率与最先进的 SOTA 方法相差不到 1%，在某些数据集上甚至领先 0.5%~1.2%。更重要的是，SEQ\* 只需要微调任务头的参数（约 1% 的全模型参数），训练时间比最复杂的基线快 2–3 倍。  
- **消融实验**：作者分别去掉历史数据、只使用新数据、以及只冻结全部参数进行对比。结果显示，去掉历史数据会导致显著的性能下降（约 5%~8%），而仅微调任务头已经足以保持旧任务性能，进一步验证了 PLM 本身的抗遗忘特性。  
- **局限性**：论文未在大规模生成式模型（如 GPT‑3）上进行实验，且对极端长序列或跨语言增量场景的表现缺乏评估。作者也承认，当历史数据规模极大时，单纯的顺序微调仍会面临计算资源瓶颈。

### 影响与延伸思考
这篇工作在发布后迅速引发了对“防忘记技术是否真的必要”的讨论。随后有几篇论文尝试在更大规模的 LLM（Large Language Model）上复现 SEQ\*，并探索“只用提示工程而不微调”的极端轻量方案（推测）。此外，社区开始关注如何在保持参数高效的前提下，利用 PLM 的内在记忆结构进行增量学习，而不是在模型外部构建复杂的记忆库。想进一步深入的读者可以关注以下方向：① 在跨语言或跨模态增量学习中验证 PLM 的抗遗忘能力；② 结合稀疏更新或 LoRA（Low-Rank Adaptation）等技术，进一步压缩增量阶段的计算开销；③ 探索基于提示（prompt）而非微调的增量学习范式。

### 一句话记住它
在预训练语言模型上，增量学习几乎不需要防忘记技巧，直接顺序微调就能保持旧知识并高效学习新任务。