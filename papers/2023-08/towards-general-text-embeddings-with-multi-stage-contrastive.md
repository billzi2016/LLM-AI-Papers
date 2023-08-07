# Towards General Text Embeddings with Multi-stage Contrastive Learning

> **Date**：2023-08-07
> **arXiv**：https://arxiv.org/abs/2308.03281

## Abstract

We present GTE, a general-purpose text embedding model trained with multi-stage contrastive learning. In line with recent advancements in unifying various NLP tasks into a single format, we train a unified text embedding model by employing contrastive learning over a diverse mixture of datasets from multiple sources. By significantly increasing the number of training data during both unsupervised pre-training and supervised fine-tuning stages, we achieve substantial performance gains over existing embedding models. Notably, even with a relatively modest parameter count of 110M, GTE$_\text{base}$ outperforms the black-box embedding API provided by OpenAI and even surpasses 10x larger text embedding models on the massive text embedding benchmark. Furthermore, without additional fine-tuning on each programming language individually, our model outperforms previous best code retrievers of similar size by treating code as text. In summary, our model achieves impressive results by effectively harnessing multi-stage contrastive learning, offering a powerful and efficient text embedding model with broad applicability across various NLP and code-related tasks.

---

# 面向通用文本嵌入的多阶段对比学习 论文详细解读

### 背景：这个问题为什么难？

文本嵌入是把一段文字压缩成固定维度向量的技术，几乎是所有自然语言处理任务的底层构件。过去的模型要么专注于单一任务（比如检索或情感分类），要么只在少量公开数据上做对比学习，导致向量在跨任务、跨领域的表现差强人意。更糟的是，代码检索等特殊场景往往需要单独训练或微调专用模型，成本高且难以统一。于是，如何用一种模型、一次训练，就能在海量文本、代码、以及各种下游任务上都拿出高质量的向量，成为了迫切的需求。

### 关键概念速览
**对比学习**：让模型在同一批数据里把“相似”样本拉近、把“不同”样本推远，类似于把一堆相似的照片放进同一个相册。  
**双编码器（Dual‑Encoder）**：两个结构相同的网络分别处理查询和候选文本，最后用点积衡量相似度，就像两个人各自写下自己的描述，再比对相似程度。  
**预训练**：在大规模未标注语料上先让模型学会基本的语言规律，类似于先学会拼音再学写字。  
**监督微调**：在标注好的相似/不相似对上继续训练，让模型更精准地捕捉任务需求，像在基础练习后专门练习拔河。  
**硬负样本（Hard Negative）**：挑选那些看起来很像正样本却实际上不相似的负例，像在考试中故意放几个“陷阱选项”，可以逼模型学得更细致。  
**文本嵌入**：把任意长度的文字映射到固定维度向量的过程，向量就像文字的指纹。  
**多任务统一**：把不同的下游任务（检索、分类、代码搜索等）都转成“给定两段文字，判断相似度”的统一格式，省去为每个任务单独设计模型的麻烦。  
**代码视作文本**：把程序代码直接当普通自然语言来处理，而不做语言特化的预处理，类似于把数学公式直接当普通句子阅读。

### 核心创新点
1. **从单阶段到多阶段对比学习**  
   *之前的做法*：大多数嵌入模型只在一次对比学习阶段完成训练，数据量和负样本质量受限。  
   *本文的做法*：先用海量未标注语料做无监督预训练，再用标注的相似对并加入硬负样本进行监督微调，形成两段式学习。  
   *带来的改变*：模型在两次“洗礼”后能够同时捕捉通用语言特征和任务特定的细粒度相似度，显著提升了跨任务表现。

2. **极度放大训练数据规模**  
   *之前的做法*：对比学习往往受限于公开数据集的规模，几百万对左右。  
   *本文的做法*：在预训练阶段使用数十亿对无标注句子，在微调阶段再加入来自检索、问答、代码等多源任务的数千万标注对。  
   *带来的改变*：即使模型只有 110 M 参数，也能在大规模数据的帮助下超越参数量十倍以上的竞争模型。

3. **统一数据混合与任务格式**  
   *之前的做法*：不同任务需要不同的输入格式或专门的头部结构。  
   *本文的做法*：把所有任务都转成“文本‑文本相似度”对，直接喂给同一个双编码器，不做任务特化的改动。  
   *带来的改变*：模型一次训练后即可直接用于检索、分类、代码搜索等多种场景，省去繁琐的任务适配工作。

4. **把代码直接当文本处理**  
   *之前的做法*：代码检索模型往往要专门的语法解析或额外的代码‑语言预训练。  
   *本文的做法*：在统一的相似度学习中加入代码‑代码、代码‑自然语言对，让模型自然学会代码的语义表示。  
   *带来的改变*：在不做任何语言特化的前提下，模型在代码检索基准上超过了同等规模的专用检索器。

### 方法详解
整体思路可以拆成三大块：**数据准备 → 双编码器对比学习 → 两阶段训练**。

1. **数据准备**  
   - **无监督阶段**：从公开的网页、书籍、维基等语料中随机抽取句子对，使用随机遮盖或轻微扰动生成“正样本”。负样本则直接取同批次的其他句子。  
   - **监督阶段**：收集已有的相似度标注数据（如检索点击、问答匹配、代码‑描述对），并在每个正对旁边挑选**硬负样本**——这些负样本是模型在预训练阶段误判为相似的句子，或者在向量空间里距离最近的非匹配项。

2. **双编码器结构**  
   - 两个完全相同的 Transformer 编码器（参数共享），分别接受查询和候选文本。每段文本经过分词、位置编码后进入编码器，最终取 **[CLS]**（或池化）向量作为句子表示。  
   - 相似度通过向量点积计算，点积越大表示越相似。

3. **对比学习目标**  
   - 使用 **InfoNCE** 损失：对每个查询，模型需要在一个 batch 中把对应的正候选的点积最大化，同时把所有负候选的点积最小化。  
   - 在监督阶段加入 **硬负样本权重**，让这些负样本在损失中占更大比重，迫使模型在细粒度上区分相似度。

4. **两阶段训练流程**  
   - **阶段一（预训练）**：在数十亿对无监督样本上跑数十个 epoch，主要让模型学会语言的基本结构和通用语义。  
   - **阶段二（微调）**：切换到标注数据，加入硬负样本并继续训练。这里的 batch 大小、学习率等超参会稍作调低，以防过拟合。  
   - 两阶段共享同一套模型参数，微调阶段直接在预训练得到的权重上继续优化。

**最巧妙的地方**在于：作者没有为每个任务单独设计头部或额外的适配层，而是靠**数据的多样性和负样本的硬度**让同一个双编码器自行分化出不同任务的能力。这样既保持了模型的简洁，又让规模扩展更容易。

### 实验与效果
- **评测数据**：作者在“Massive Text Embedding Benchmark”（包含检索、语义相似、聚类等 30+ 子任务）上进行评测，还在公开的代码检索基准（如 CodeSearchNet）上测试。  
- **对比基线**：包括 OpenAI 的官方嵌入 API、Facebook/Meta 的大型文本嵌入模型（参数量 1B 以上）以及专门的代码检索模型。  
- **结果**：论文声称 GTE_base（110 M 参数）在整体基准分数上超过 OpenAI API，并且在多数子任务上跑赢参数量十倍的竞争模型。代码检索实验中，GTE 直接把代码当文本使用，表现也超过了同等规模的专用检索器。  
- **消融实验**：作者分别去掉硬负样本、只用单阶段训练、或只使用单一数据源进行实验，发现硬负样本和多阶段训练对最终分数提升贡献最大，约提升 5%–10% 的相对增益。  
- **局限性**：模型仍然受限于 110 M 参数，在处理极长文档或需要跨语言迁移时可能表现不佳；论文未给出多语言或低资源语言的实验结果，暗示在这些场景仍需进一步验证。

### 影响与延伸思考
这篇工作展示了“规模+多阶段对比学习”可以在不增大模型体量的前提下显著提升通用嵌入质量，随后不少团队开始尝试在自己的检索或推荐系统里加入硬负样本的微调环节。还有研究把 **多模态（文本+图像）** 对比学习搬到类似的两阶段框架，直接受益于 GTE 的训练思路。想进一步深入的读者可以关注 **硬负样本采样策略**、**跨语言对比学习** 以及 **更高效的双编码器变体（如轻量化 Transformer）** 等方向。

### 一句话记住它
只要把海量数据分两阶段、加上硬负样本，同一个 110 M 的双编码器就能跑赢十倍大模型的通用文本嵌入。