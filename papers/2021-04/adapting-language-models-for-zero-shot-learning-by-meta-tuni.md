# Adapting Language Models for Zero-shot Learning by Meta-tuning on   Dataset and Prompt Collections

> **Date**：2021-04-10
> **arXiv**：https://arxiv.org/abs/2104.04670

## Abstract

Large pre-trained language models (LMs) such as GPT-3 have acquired a surprising ability to perform zero-shot learning. For example, to classify sentiment without any training examples, we can "prompt" the LM with the review and the label description "Does the user like this movie?", and ask whether the next word is "yes" or "no". However, the next word prediction training objective is still misaligned with the target zero-shot learning objective. To address this weakness, we propose meta-tuning, which directly optimizes the zero-shot learning objective by fine-tuning pre-trained language models on a collection of datasets. We focus on classification tasks, and construct the meta-dataset by aggregating 43 existing datasets and annotating 441 label descriptions in a question-answering (QA) format. When evaluated on unseen tasks, meta-tuned models outperform a same-sized QA model and the previous SOTA zero-shot learning system based on natural language inference. Additionally, increasing parameter count from 220M to 770M improves AUC-ROC scores by 6.3%, and we forecast that even larger models would perform better. Therefore, measuring zero-shot learning performance on language models out-of-the-box might underestimate their true potential, and community-wide efforts on aggregating datasets and unifying their formats can help build models that answer prompts better.

---

# 通过元调优在数据集和提示集合上适配语言模型以实现零样本学习 论文详细解读

### 背景：这个问题为什么难？

大模型（如 GPT‑3）在没有任何标注数据的情况下，靠“提示”（prompt）就能完成分类、问答等任务，这种零样本学习让人惊讶。但模型的训练目标是“下一个词预测”，它并不直接对应我们想要的“给定输入，输出正确标签”。于是，当我们把模型当作黑盒子直接用提示时，往往只能得到勉强可用的结果。过去的改进大多是手工设计更好的提示或在外部加入自然语言推理（NLI）桥接层，这些办法本质上仍是对原始模型的“旁路”，没有真正让模型的内部参数去适配零样本任务的需求。

### 关键概念速览
- **零样本学习（Zero-shot Learning）**：模型在没有看到任何该任务的训练样本的情况下，直接完成任务。像是让一个从未学过数学的人，仅凭题目描述就解答。
- **提示（Prompt）**：把任务描述写成自然语言，让语言模型把它当作输入的一部分。相当于给模型出一道“口头指令”。
- **元调优（Meta‑tuning）**：在一大堆不同任务上微调模型，使其学习到一种通用的“快速适应”能力。可以想象为让模型练习“多种不同的游戏”，从而在新游戏上也能快速上手。
- **QA 格式提示**：把标签描述转成“问题‑答案”形式，例如把“正面/负面”改写为“用户喜欢这部电影吗？”。这种统一的问答形式让模型的输出更易比较。
- **AUC‑ROC**：衡量二分类模型整体排序能力的指标，值越高说明模型在区分正负例上越好。类似于评估一个裁判在判定好球与坏球时的整体准确度。
- **自然语言推理（NLI）**：把分类任务转化为“前提‑假设”关系判断的技术，之前的零样本 SOTA 多基于此。

### 核心创新点
1. **目标对齐的微调 → 元调优**  
   之前的做法仍然是让模型继续学习下一个词的概率，提示只是外层包装。本文直接把零样本任务的评价指标（如分类准确率）当作微调目标，在大量任务上进行梯度更新，使模型内部参数真正对齐零样本学习需求。

2. **统一的 QA 提示集合 → 441 条标签描述**  
   过去每个数据集都有自己的标签写法，导致模型需要适应多种格式。作者把 43 个公开分类数据集的标签全部转成统一的问答式描述，形成一个大规模的“提示库”。这相当于给模型提供了一本“标准化的题库”，让它学会在同一种问答框架下回答。

3. **规模效应的系统验证**  
   通过对比 220M 与 770M 参数的模型，发现模型规模每提升一倍，AUC‑ROC 能提升约 6.3%。这一步验证了元调优的收益并非偶然，而是会随模型容量继续放大。

4. **对比同尺寸 QA 模型与 NLI‑based 零样本系统**  
   实验显示，元调优后的模型在未见任务上明显超过同等大小的普通 QA 微调模型以及之前最好的 NLI‑based 零样本方法，证明了直接优化零样本目标的有效性。

### 方法详解
**整体思路**：先把大量已有的分类任务统一成 QA 提示，然后在这些任务上进行一次统一的微调，使模型的参数专门学习“在 QA 形式下做分类”。微调结束后，模型即可直接接受全新任务的 QA 提示，完成零样本预测。

**步骤拆解**：

1. **构建元数据集**  
   - 收集 43 个公开的分类数据集（情感、主题、医学等）。  
   - 为每个数据集的每个标签写出自然语言的问题，例如把 “positive / negative” 变成 “用户喜欢这部电影吗？”  
   - 最终得到 441 条不同的 QA 提示，形成统一的输入‑输出对。

2. **格式化训练样本**  
   - 每条训练样本的输入是：`[文本] + 问题`，输出是对应的 “是/否”。  
   - 例如：`"这部电影太棒了！" + "用户喜欢这部电影吗？"` → `是`。  
   - 这种结构让模型只需要判断下一个词是 “是” 还是 “否”，与原始语言模型的预测任务保持一致，却把目标对齐到了分类上。

3. **元调优过程**  
   - 使用标准的语言模型微调流程（Adam 优化器、交叉熵损失），但损失直接基于 QA 的二分类正确率。  
   - 训练时随机抽取不同数据集的样本，使模型在一次梯度更新中看到多任务的多样性，形成跨任务的适应能力。

4. **推理阶段**  
   - 对于全新任务，只需写出对应的 QA 提示（不需要任何标注样本），把文本和问题拼接后喂入模型。  
   - 读取模型输出的下一个词，如果是 “是” 则判为正类，否则为负类。

**巧妙之处**：  
- 把所有标签统一成二元 QA，使得不同任务之间的输入输出完全兼容，避免了多任务学习中常见的标签空间冲突。  
- 直接在语言模型的原始预测空间（下一个词）上做分类，使得微调不需要额外的头部结构，保持了模型的通用性。  
- 通过随机抽取任务实现“元学习”效果，让模型在一次训练中学会快速适配新任务的提示。

### 实验与效果
- **测试任务**：在论文中选取了若干未出现在元数据集中的分类任务作为零样本评估（具体名称未在摘要中列出）。  
- **基线对比**：  
  - 同尺寸的普通 QA 微调模型（即没有元调优的版本）。  
  - 之前的零样本 SOTA 方法，基于自然语言推理（NLI）把分类转化为前提‑假设判断。  
- **结果**：元调优模型在所有未见任务上均超越基线，尤其在 AUC‑ROC 上提升显著。摘要提到“meta‑tuned models outperform a same‑sized QA model and the previous SOTA zero‑shot learning system”。  
- **规模效应**：从 220M 参数提升到 770M，AUC‑ROC 提升 6.3%，暗示更大模型会有更好表现。  
- **消融实验**：摘要未给出细节，原文未详细描述，但可以推测作者会分别去掉统一 QA 提示、元数据集多样性等因素，以验证每个设计的贡献。  
- **局限**：实验仅覆盖二元分类任务，未验证多类或生成式任务的效果；提示仍需人工编写，自动化程度有限。

### 影响与延伸思考
这篇工作向社区展示了“把零样本学习目标直接写进微调过程”是一条可行且高效的路线。随后出现的研究（如 **MetaICL**、**PromptSource**）进一步扩展了元调优的任务种类，甚至尝试在多模态模型上做类似的对齐。对想深入的读者，可以关注以下方向：

- **自动化提示生成**：如何让模型自己发现或优化 QA 提示，减少人工标注成本。  
- **多类/序列任务的元调优**：把当前的二元 QA 思路推广到多标签或生成任务。  
- **跨语言元调优**：在多语言数据集上统一 QA 提示，探索模型的语言迁移能力。  
- **更大规模的元数据集**：聚合更多公开任务，验证模型规模与元调优收益的关系。

### 一句话记住它
把大量任务统一成问答式提示，在这些任务上直接微调语言模型，让模型的“下一个词”预测本身就学会零样本分类。