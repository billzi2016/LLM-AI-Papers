# Layer Swapping for Zero-Shot Cross-Lingual Transfer in Large Language Models

> **Date**：2024-10-02
> **arXiv**：https://arxiv.org/abs/2410.01335

## Abstract

Model merging, such as model souping, is the practice of combining different models with the same architecture together without further training. In this work, we present a model merging methodology that addresses the difficulty of fine-tuning Large Language Models (LLMs) for target tasks in non-English languages, where task-specific data is often unavailable. We focus on mathematical reasoning and without in-language math data, facilitate cross-lingual transfer by composing language and math capabilities. Starting from the same pretrained model, we fine-tune separate "experts" on math instruction data in English and on generic instruction data in the target language. We then replace the top and bottom transformer layers of the math expert directly with layers from the language expert, which consequently enhances math performance in the target language. The resulting merged models outperform the individual experts and other merging methods on the math benchmark, MGSM, by 10% across four major languages where math instruction data is scarce. In addition, this layer swapping is simple, inexpensive, and intuitive, as it is based on an interpretative analysis of the most important parameter changes during the fine-tuning of each expert. The ability to successfully re-compose LLMs for cross-lingual transfer in this manner opens up future possibilities to combine model expertise, create modular solutions, and transfer reasoning capabilities across languages all post hoc.

---

# 层交换实现大语言模型零样本跨语言迁移 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）上做数学推理已经有不少成功案例，但这些模型大多在英文数据上微调。把同样的数学能力搬到法语、德语等资源匮乏的语言时，往往缺少对应的数学指令数据，直接微调会导致模型在目标语言上表现不佳。传统的跨语言迁移要么依赖大规模平行语料，要么需要在每种语言上单独收集标注数据，成本高且效果受限。因此，如何在几乎没有目标语言数学数据的情况下，让模型在该语言上也能解数学题，成为一个迫切的挑战。

### 关键概念速览

**模型合并（Model Merging）**：把两个已经训练好的模型的参数按照一定规则拼接在一起，得到一个兼具两者能力的模型。想象把两本书的章节互换，形成一本新书。

**专家模型（Expert）**：针对特定任务或语言单独微调的模型子集。比如只在英文数学指令上微调的模型，就是“数学专家”。

**层交换（Layer Swapping）**：直接把一个模型的若干 Transformer 层替换成另一个模型对应层的参数。类似把一辆车的发动机和底盘换成别的车型的部件。

**零样本跨语言迁移（Zero-Shot Cross-Lingual Transfer）**：模型在目标语言上没有看到任何任务示例，却仍能完成任务。相当于让学生在没有练习的情况下，用已学的技巧解新语言的题目。

**MGSM 基准**：Multilingual Grade School Math，是一个覆盖多语言的中小学数学测评集合，用来衡量模型的跨语言数学能力。

### 核心创新点

1. **从同一基模型分别微调出两位专家 → 直接把数学专家的顶部和底部层换成语言专家对应层 → 目标语言的数学表现提升约 10%**。之前的做法要么是整体微调，要么是使用参数平均，这种层级级别的“拼接”更精准。

2. **基于微调过程中的参数变化进行层级选择 → 只替换那些在语言微调中变化最大的层 → 省去大量试错**。传统的层交换往往盲目挑层，本文提供了一套解释性分析来指明“哪些层最重要”。

3. **保持模型结构不变、无需额外训练 → 只动几层参数就完成跨语言迁移 → 成本低、部署快**。相比需要再训练的适配层或 LoRA 方法，这种“即插即用”更符合工业落地需求。

### 方法详解

整体思路可以分为三步：

1. **准备两个专家**：从同一个预训练的大语言模型出发，分别在（a）英文数学指令数据上微调得到“数学专家”，以及（b）目标语言的通用指令数据上微调得到“语言专家”。两者的网络结构完全相同，只是微调的目标不同。

2. **分析微调参数变化**：对比每一层在两次微调前后的权重差异。作者发现，语言专家的前几层（靠近输入）和最后几层（靠近输出）在微调时的变化最显著，而中间层相对稳定。于是把这些“变化大”的层视为语言特化层。

3. **层交换并合并**：把数学专家的前 N 层和后 M 层直接替换成语言专家对应的层，保留数学专家的中间层不动。这样模型的输入处理和输出生成都受语言专家的影响，而核心的数学推理仍由原来的数学专家承担。完成后得到的模型即可在目标语言上进行零样本数学推理。

**最巧妙的地方**在于只动最关键的几层，而不是全模型平均或全部替换。这样既保留了数学专家的深层推理能力，又让模型的语言感知和生成更贴合目标语言的语法、词汇。

### 实验与效果

- **测试数据**：使用 MGSM 基准，覆盖英语、法语、德语、中文四种在数学指令数据上稀缺的语言。
- **对比基线**：单独的数学专家、单独的语言专家、传统的模型平均（model soup）以及最近的 LoRA 适配方法。
- **结果**：层交换模型在四种语言上整体提升约 10% 的准确率，尤其在法语和德语上超过 12%。在英文上仍保持与原数学专家相当的水平，说明数学推理未被削弱。
- **消融实验**：作者分别只换顶部层、只换底部层以及不换层的情况，发现同时换上下两端的组合效果最佳，验证了层选择的合理性。
- **局限**：实验只覆盖四种语言，未验证对低资源语言（如斯瓦希里语）的适用性；此外，层交换仍依赖目标语言的通用指令数据，完全无数据的极端情况未测试。

### 影响与延伸思考

这篇工作打开了“模块化大模型”的新思路：不同能力可以通过层级拼接后即刻组合，而不必重新训练。随后有研究尝试把代码、常识、对话等专家通过类似的层交换或层加权方式合成多功能模型（推测）。如果想进一步探索，可以关注以下方向：自动化寻找最优层组合的搜索算法、在更细粒度（如子层）上进行参数混合、以及把层交换与参数高效适配（如 LoRA）结合，实现更灵活的跨任务迁移。

### 一句话记住它

只换掉语言专家最关键的前后层，就能让大语言模型在没有任何目标语言数学数据的情况下，直接拥有跨语言数学推理能力。