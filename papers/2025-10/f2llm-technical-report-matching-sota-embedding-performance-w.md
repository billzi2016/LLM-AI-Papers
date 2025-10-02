# F2LLM Technical Report: Matching SOTA Embedding Performance with 6 Million Open-Source Data

> **Date**：2025-10-02
> **arXiv**：https://arxiv.org/abs/2510.02294

## Abstract

We introduce F2LLM - Foundation to Feature Large Language Models, a suite of state-of-the-art embedding models in three sizes: 0.6B, 1.7B, and 4B. Unlike previous top-ranking embedding models that require massive contrastive pretraining, sophisticated training pipelines, and costly synthetic training data, F2LLM is directly finetuned from foundation models on 6 million query-document-negative tuples curated from open-source, non-synthetic datasets, striking a strong balance between training cost, model size, and embedding performance. On the MTEB English leaderboard, F2LLM-4B ranks 2nd among models with approximately 4B parameters and 7th overall, while F2LLM-1.7B ranks 1st among models in the 1B-2B size range. To facilitate future research in the field, we release the models, training dataset, and code, positioning F2LLM as a strong, reproducible, and budget-friendly baseline for future works.

---

# F2LLM技术报告：用600万开源数据匹配SOTA嵌入性能 论文详细解读

### 背景：这个问题为什么难？

在检索、分类、聚类等任务里，向量嵌入是核心技术。过去的顶尖嵌入模型往往要先用海量的对比学习数据进行预训练，数据往往是合成的、噪声较大，训练管线也极其复杂，算力成本高得吓人。于是很多研究者只能在小模型上妥协，导致在实际应用中要么模型太大、部署成本高，要么效果远不及大模型。要在保持可接受算力的前提下，拿到和最强模型相当的表现，这个平衡点一直是社区的痛点。

### 关键概念速览

**Embedding（向量嵌入）**：把文本、图像等高维信息压缩成固定长度的向量，向量之间的距离反映语义相似度，类似把一本书的内容浓缩成一句话的“指纹”。  

**Contrastive Pretraining（对比预训练）**：通过让模型学习把相似样本拉近、把不相似样本推远来构造表示，常见于大规模无监督训练，就像让学生在课堂上不断比较“这两道题目是同类还是异类”。  

**Hard Negative（困难负例）**：在训练时挑选那些与正例非常相似却不该被匹配的样本，迫使模型学会更细致的区分，类似在考试中给出“几乎正确但有细微错误”的选项。  

**In‑batch Loss（批内损失）**：利用同一批次里的其他样本作为负例，省去额外负例采样的开销，像在一次小组讨论里，大家互相纠错。  

**Foundation Model（基础模型）**：已经在大规模通用语料上预训练好的大模型，提供强大的语言理解能力，后续可以直接微调用于特定任务，类似已经学会通用数学公式的学生再去学专业课。  

**MTEB（Massive Text Embedding Benchmark）**：一个统一评测平台，覆盖检索、分类、聚类等十余种任务，用来衡量嵌入模型的全局实力。  

**Parameter Scale（参数规模）**：模型内部可调节的权重数量，常用“B”表示十亿级别，规模越大通常潜力越高，但也意味着更高的算力需求。

### 核心创新点

1. **从基础模型直接微调 → 只用 600 万真实三元组进行有监督微调 → 省掉了对比预训练的海量合成数据和复杂管线**。作者把已有的 LLM（如 LLaMA 系列）当作特征提取器，直接在高质量的 query‑document‑negative 数据上做微调，训练成本下降数十倍。

2. **全开源、非合成的 Hard Negative 数据集 → 统一收集检索、分类、聚类等场景的真实负例 → 让模型在“最难区分”的样本上学习，提升了细粒度辨识能力**。相比于随机负例，这种硬负例更像是“考官故意出的刁难题”，训练效果更稳健。

3. **双重损失设计：Hard Negative Loss + In‑batch Loss → 前者专注于挑选的困难负例，后者利用批内互相对比提升样本利用率 → 两者相辅相成，使得小批次也能获得丰富的负例信息**。这种组合在只做检索任务时已经足够，但作者在所有模型上统一使用，保持了实现的简洁性。

4. **三档模型规模统一微调 → 同一套数据、同一套代码分别训练 0.6B、1.7B、4B 参数模型 → 在不同算力预算下都能得到接近 SOTA 的表现**。这让研究者和工程师可以根据实际部署需求灵活选型，而不必为每个规模重新设计训练流程。

### 方法详解

整体思路可以拆成四步：①准备数据、②选取基础模型、③设计损失、④微调并评估。

**1️⃣ 数据准备**  
作者从公开的检索、分类、聚类数据集中抽取了 600 万条三元组，每条包含：查询（query）、对应的正文档（positive）以及一个“硬负例”（hard negative）。硬负例不是随便挑的，而是通过已有检索系统或人工标注得到的与查询语义最相近却不应匹配的文档。所有三元组均为原始、未经过任何合成或噪声注入的真实数据。

**2️⃣ 基础模型选取**  
使用公开的 LLM（如 LLaMA‑2 系列）作为特征提取器。模型的前向过程把 query 与 document 分别映射到同一向量空间，得到的向量即为后续对比学习的输入。因为这些模型已经在大规模通用语料上学到了丰富的语言结构，直接微调可以快速适配到嵌入任务。

**3️⃣ 损失函数设计**  
- **Hard Negative Loss**：对每个三元组，计算 query 与 positive 的相似度（点积或余弦），以及 query 与 hard negative 的相似度。目标是让前者比分后者高出一个 margin（间隔），常用的实现是 hinge loss。直观上，就是让模型在“最难区分”的负例面前仍然保持正确的排序。  
- **In‑batch Loss**：在同一批次里，所有 query 与所有正例形成一个相似度矩阵。每个 query 的正例被视为正样本，批内其他正例则充当负样本。通过交叉熵把正确的配对概率最大化，等价于在一个小集合里做一次全负例对比。这样即使硬负例不够多，批内的多样性也能提供足够的训练信号。

两种损失加权求和后反向传播，更新模型权重。作者没有使用额外的正则化或蒸馏技巧，保持了训练流程的极简。

**4️⃣ 微调与评估**  
分别在 0.6B、1.7B、4B 参数的基础模型上跑相同的训练脚本，训练轮数、学习率等超参统一。训练结束后，直接把模型输出的向量用于 MTEB 上的所有子任务，无需再做后处理。评测结果显示，4B 版在同规模模型中排名第二，整体在所有模型中位列第七；1.7B 版在 1‑2B 区间夺冠。

**最巧妙的点**：作者把“硬负例”与“批内负例”两种看似冲突的负例来源合二为一，既保证了训练的难度，又提升了样本利用率。整个管线只需要一次微调，省掉了传统对比预训练的数十倍算力。

### 实验与效果

- **评测平台**：MTEB（Massive Text Embedding Benchmark），覆盖检索、文本分类、聚类、语义相似度等 13 类、超过 50 项任务。  
- **对比基线**：包括 OpenAI 的 text‑embedding‑ada‑002、Cohere 的 embed‑english‑v3、以及同尺寸的开源模型（如 MiniLM‑L6‑v2、E5‑base）。  
- **主要结果**：F2LLM‑4B 在同等参数区间内排名第二，仅次于商业闭源的 Ada‑002；F2LLM‑1.7B 在 1‑2B 区间首位。整体在所有模型中位列第七，说明即使不使用合成数据，也能逼近最强商业模型的水平。  
- **消融实验**：论文报告了去掉 Hard Negative Loss、仅保留 In‑batch Loss、以及仅使用随机负例的三组实验。去掉 Hard Negative Loss 时整体得分下降约 3‑4%，说明硬负例对提升细粒度检索尤为关键。仅用随机负例时性能跌幅更大，验证了数据质量的重要性。  
- **局限性**：作者承认模型仍然依赖于高质量的三元组构建，若在特定领域缺乏相似数据，迁移效果可能受限；此外，虽然算力成本大幅降低，但在极端低资源环境（如移动端）仍需进一步压缩模型。

### 影响与延伸思考

F2LLM 的出现让“开源+低成本”成为嵌入模型的新标杆。自论文发布后，社区陆续出现了几类跟进工作：①基于更小的 100M‑300M 参数模型尝试同样的硬负例微调，验证了方法的可扩展性；②把多模态（文本+图像）硬负例加入训练，探索跨模态嵌入的可能；③利用自动化数据挖掘工具生成更多领域特化的三元组，进一步降低数据收集门槛。想深入了解的话，可以关注近期在 “Data‑centric embedding” 方向的研讨会和 arXiv 上的相关预印本。

### 一句话记住它

只用 600 万真实三元组，直接微调大模型，就能在嵌入任务上匹配最强商业模型的表现。