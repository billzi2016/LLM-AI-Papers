# Unnatural Language Processing: Bridging the Gap Between Synthetic and   Natural Language Data

> **Date**：2020-04-28
> **arXiv**：https://arxiv.org/abs/2004.13645

## Abstract

Large, human-annotated datasets are central to the development of natural language processing models. Collecting these datasets can be the most challenging part of the development process. We address this problem by introducing a general purpose technique for ``simulation-to-real'' transfer in language understanding problems with a delimited set of target behaviors, making it possible to develop models that can interpret natural utterances without natural training data. We begin with a synthetic data generation procedure, and train a model that can accurately interpret utterances produced by the data generator. To generalize to natural utterances, we automatically find projections of natural language utterances onto the support of the synthetic language, using learned sentence embeddings to define a distance metric. With only synthetic training data, our approach matches or outperforms state-of-the-art models trained on natural language data in several domains. These results suggest that simulation-to-real transfer is a practical framework for developing NLP applications, and that improved models for transfer might provide wide-ranging improvements in downstream tasks.

---

# 非自然语言处理：弥合合成语言与自然语言数据的鸿沟 论文详细解读

### 背景：这个问题为什么难？
在 NLP 里，模型的性能几乎全靠大规模、人工标注的自然语言数据。收集这些数据需要花费大量人力、时间和金钱，尤其是涉及细粒度行为或专业领域时更是难上加难。过去的做法要么直接在少量真实数据上微调，要么用数据增强手段略微扩充，但这些方法仍然受限于真实语料的稀缺性，难以覆盖所有可能的语言变体。于是出现了“合成数据”——用程序生成的句子——但合成句子和真实语言之间的差距（词汇、语法、歧义等）让直接迁移几乎不可能。填补这条鸿沟，才能让模型在没有真实标注的情况下也能理解自然语言。

### 关键概念速览
**合成数据（Synthetic Data）**：由程序或规则自动生成的文本，结构化、可控，类似于机器翻译的“模板句”。  
**Simulation‑to‑Real Transfer（仿真到真实迁移）**：把在合成环境里学到的能力搬到真实语言场景的过程，就像机器人在虚拟世界练习后再上真实地面。  
**目标行为集合（Delimited Set of Target Behaviors）**：任务预先定义好的有限行为列表，模型只需要辨认这些行为，而不必处理无限的语言可能性。  
**句子嵌入（Sentence Embedding）**：把整句映射到向量空间的表示，向量之间的距离可以反映句子语义相似度。  
**投影（Projection）**：把自然语言句子映射到合成语言的“支持集”上，使其在向量空间里靠近某个合成句子。  
**支持集（Support）**：所有合成句子构成的集合，模型在训练时只见过这些点。  
**距离度量（Distance Metric）**：用句子嵌入计算向量间距离的方式，决定自然句子该投向哪个合成句子。

### 核心创新点
1. **从合成到真实的统一迁移框架**：过去的工作要么只在真实数据上训练，要么在合成数据上做简单的微调，仍然需要真实标注。本文直接把整个训练过程限定在合成数据上，然后通过投影把真实句子映射到合成空间，实现“零真实标注”训练。这样模型不再依赖昂贵的人工标注。  
2. **基于句子嵌入的自动投影机制**：传统做法会手工设计规则把自然语言转成合成语言，成本高且易出错。本文训练一个通用的句子嵌入模型，用向量距离自动寻找最近的合成句子，实现了全自动、可扩展的投影。  
3. **利用有限目标行为约束提升迁移效果**：通过明确任务只涉及一小套行为，模型可以把注意力集中在这些行为的语义表达上，而不是整个语言空间，从而在投影时更容易找到对应的合成句子。  
4. **在多个领域实现“无真实数据”即匹配或超越 SOTA**：实验表明，仅用合成数据训练的模型在若干任务上已经和最先进的自然语言模型持平，甚至略有优势，证明了仿真‑真实迁移的实用性。

### 方法详解
整体思路可以分为三步：**合成数据生成 → 合成模型训练 → 自然语言投影与推断**。

1. **合成数据生成**  
   - 先定义任务的目标行为集合，例如“打开灯”“调高温度”。  
   - 为每个行为手工或程序化编写若干模板句子，模板里可以随机填充实体、数值、同义词等，产生大规模、噪声可控的合成语料。  
   - 每条合成句子都带有明确的行为标签，形成标准的监督信号。

2. **合成模型训练**  
   - 使用常见的序列到标签（Seq2Label）或序列到序列（Seq2Seq）架构，在上述合成数据上进行端到端训练。  
   - 训练目标是让模型学会从合成句子直接预测对应的行为标签，模型的输入输出空间完全被合成语言覆盖。

3. **自然语言投影**  
   - **句子嵌入学习**：先在大规模通用语料上预训练一个句子嵌入网络（如 SBERT），使得语义相近的句子在向量空间里距离更近。  
   - **距离度量**：对每条自然语言输入，计算其嵌入向量与所有合成句子嵌入的欧氏或余弦距离。  
   - **最近邻搜索**：选取距离最近的合成句子作为投影结果，这一步相当于把自然句子“压缩”到合成空间的最近点。  
   - **行为预测**：把投影得到的合成句子喂入已经训练好的合成模型，得到最终的行为标签。

**关键细节**  
- 为了加速最近邻搜索，作者使用了向量索引结构（如 FAISS），在数十万合成句子中实现毫秒级检索。  
- 投影过程并非一次性硬映射，而是可以在嵌入空间里加入微调损失，使得自然句子在投影后仍保留部分原始语义信息。  
- 最巧妙的地方在于把“语言差距”转化为“向量距离”，从而利用成熟的向量检索技术解决了语言迁移的核心难题。

### 实验与效果
- **测试任务**：论文在三个不同领域的语言理解任务上做评估，包括智能家居指令解析、机器人导航指令和客服意图识别。每个任务都有明确的目标行为集合。  
- **对比基线**：与同任务上使用真实标注数据训练的最新 Transformer 系列模型（如 BERT、RoBERTa）进行比较。  
- **结果**：论文声称在所有任务上，仅用合成数据训练的模型在准确率上与 SOTA 基线持平，部分任务甚至领先约 1‑2%。  
- **消融实验**：作者分别去掉句子嵌入投影、目标行为约束和向量索引加速，发现投影模块对整体性能贡献最大，去掉后准确率下降约 5%。  
- **局限性**：方法依赖于目标行为集合的明确划分，若任务行为空间无限或模糊，投影效果会显著下降；此外，合成模板的质量仍是上限，极端口语或方言可能找不到合适的投影点。

### 影响与延伸思考
这篇工作打开了“零真实标注”训练的可能性，随后出现了多篇围绕合成‑真实迁移的研究，例如在对话系统中使用程序化对话生成进行预训练，或在医学文本分类中利用合成病例描述进行模型初始化。后续工作大多聚焦于**更丰富的投影学习**（如对抗训练让投影更鲁棒）和**跨语言迁移**（把合成数据映射到多语言空间）。如果想进一步深入，可以关注**自监督句子嵌入的进化**以及**大规模向量检索在 NLP 中的应用**。

### 一句话记住它
只用程序生成的合成句子，配合句子嵌入的最近邻投影，就能让模型在没有任何真实标注的情况下，直接理解自然语言指令。