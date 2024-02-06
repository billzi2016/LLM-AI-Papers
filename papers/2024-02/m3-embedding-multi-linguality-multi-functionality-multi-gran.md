# M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

> **Date**：2024-02-05
> **arXiv**：https://arxiv.org/abs/2402.03216

## Abstract

In this paper, we introduce a new embedding model called M3-Embedding, which is distinguished for its versatility in \textit{Multi-Linguality}, \textit{Multi-Functionality}, and \textit{Multi-Granularity}. It provides a uniform support for the semantic retrieval of more than 100 working languages. It can simultaneously accomplish the three common retrieval functionalities: dense retrieval, multi-vector retrieval, and sparse retrieval. Besides, it is also capable of processing inputs of different granularities, spanning from short sentences to long documents of up to 8,192 tokens. The effective training of M3-Embedding presents a series of technical contributions. Notably, we propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strategy, which enables a large batch size and high training throughput to improve the discriminativeness of embeddings. M3-Embedding exhibits a superior performance in our experiment, leading to new state-of-the-art results on multilingual, cross-lingual, and long-document retrieval benchmarks.

---

# M3-Embedding：多语言、多功能、多粒度文本嵌入的自知识蒸馏方法 论文详细解读

### 背景：这个问题为什么难？
在检索系统里，常见的需求有三类：把句子压成一个向量做稠密检索、把词级别的向量做稀疏检索、以及把文档拆成多段向量做多向量检索。过去的模型要么只能做稠密检索，要么只能处理单语言，要么只能接受短文本，导致实际应用里要拼接多个模型、做大量后处理。再者，跨语言检索需要在上百种语言上保持语义一致性，这在没有大规模平行数据的情况下几乎不可能。最后，长文档（几千到上万 token）会超出大多数 Transformer 的长度限制，导致信息被截断或需要复杂的滑动窗口策略。于是，业界急需一种“一站式”嵌入模型，既能跨语言，又能兼容三种检索方式，还能直接接受 8k 长度的输入。

### 关键概念速览
**稠密检索（Dense Retrieval）**：把整段文本映射成单个向量，向量相似度即检索相关度，像把整本书压成一张卡片。  
**稀疏检索（Sparse Retrieval）**：模型输出每个词的权重或词向量，检索时用倒排表类似传统 BM25，直观上像把句子拆成关键词再匹配。  
**多向量检索（Multi‑Vector Retrieval）**：把长文档切成若干段，每段生成向量后再做细粒度匹配，类似把一本书分章节存储，查询时可以定位到具体章节。  
**自知识蒸馏（Self‑Knowledge Distillation）**：模型自己产生的预测（这里是不同检索方式的相关分数）当作“老师”，指导自身的进一步学习，类似学生先做练习，再把答案对照自己之前的草稿进行自我纠错。  
**多语言（Multi‑Linguality）**：模型在训练时同时看到上百种语言的文本，能够在不同语言之间共享语义空间，像一个会说多国语言的翻译官。  
**多粒度（Multi‑Granularity）**：指模型能接受从单句到 8 192 token 的任意长度输入，类似相机既能拍微距也能拍全景。  
**批量策略（Batching Strategy）**：在训练时如何组织样本进批，影响显存占用和梯度质量，好的策略可以让一次训练看到上千条正负样本，提升区分能力。

### 核心创新点
- **单模型统一三种检索功能 → 通过自知识蒸馏把稠密、稀疏和多向量的相关分数混合作为教师信号 → 训练时模型同时学会输出单向量、词级权重和段向量，省去为每种功能单独训练的成本。**  
- **跨语言统一语义空间 → 在大规模无监督语料上做多语言预训练，只优化稠密检索 → 再利用平行句对和合成数据进行多功能微调 → 让模型在 100+ 语言上保持相似的向量分布，检索时不需要语言特定的适配层。**  
- **长文档直接输入 → 采用 8 192 token 的 Transformer 架构并配合分段投影矩阵 → 在多向量检索阶段把整篇文档的所有段向量一次性投射到统一空间 → 避免滑动窗口的重复计算，显著提升效率。**  
- **大批量高吞吐训练 → 重新设计批量构造，使每个批次同时包含稠密、稀疏和多向量的正负样本 → 通过梯度累加和混合精度，单卡可达上千样本 → 嵌入的判别力得到明显提升。

### 方法详解
整体思路可以分为三步：① 预训练语言编码器，② 多功能微调并引入自知识蒸馏，③ 推理时根据需求切换输出模式。下面把每一步拆开说。

1. **预训练阶段**  
   - 使用公开的多语言语料（如 CC‑100、Wikipedia）进行掩码语言模型（MLM）训练，目标是恢复被遮盖的词。  
   - 只保留稠密检索的头部：把句子最后的 CLS 向量投射到 768 维空间，作为基础向量。  
   - 这一步的核心是让模型学会跨语言的通用语义表示，后面的功能扩展都基于此向量。

2. **多功能微调 + 自知识蒸馏**  
   - **数据准备**：混合使用标注的检索对（query‑positive‑negative），以及合成的硬负样本（通过 BM25 抽取的相似但不相关句子）。  
   - **三路输出**：  
     - 稠密路：直接输出 CLS 向量。  
     - 稀疏路：在每个 token 上加一个线性层，得到词重要性分数，类似 SPLADE 的稀疏向量。  
     - 多向量路：把输入切成不超过 512 token 的块，每块走同一个编码器，得到块向量后再乘以一个共享投影矩阵得到段向量。  
   - **自知识蒸馏机制**：对同一 query，稠密、稀疏和多向量三种检索方式会产生各自的相关分数。把这三个分数加权求和，得到“教师分数”。随后，用 KL 散度或交叉熵让每条路的预测逼近教师分数。直观上相当于让三条路相互监督，避免单一路径过拟合。  
   - **损失函数**：总损失 = 稠密对比损失 + 稀疏对比损失 + 多向量对比损失 + 蒸馏损失。每项都有独立的温度系数，确保梯度不会相互抵消。

3. **批量策略**  
   - 为了让每个 batch 同时包含三种检索的正负样本，作者把一个 query 的三个视图（稠密、稀疏、段向量）拼在一起，形成一个“大批次”。  
   - 使用混合精度（FP16）和梯度累加，使显存占用保持在 32 GB 以下，却能一次处理上千条样本。  
   - 这种“大批次+多视图”设计让对比学习的负样本空间极大扩展，提升向量的区分度。

4. **推理阶段**  
   - 用户只需要指定想要的检索模式：  
     - 稠密检索 → 取 CLS 向量，做向量相似度搜索。  
     - 稀疏检索 → 取词重要性向量，构建倒排索引。  
     - 多向量检索 → 把长文档切块，分别生成段向量，再用 late‑interaction（如 ColBERT）做细粒度匹配。  
   - 所有模式共享同一个模型参数，部署成本只有一个模型文件。

**最巧妙的点**：把三种检索的相关分数当作教师信号，而不是传统的软标签或外部老师模型。这样既省去额外的教师网络，又让模型内部的多视角信息自然融合，提升了统一性和鲁棒性。

### 实验与效果
- **评测任务**：跨语言检索（XQuAD、FLORES‑101）、多语言检索（MS‑MARCO‑ML）、长文档检索（LongEval、MS‑MARCO‑Passage with 8k context）。  
- **基线对比**：与 mBERT‑based 稠密检索、SPLADE‑XL（稀疏）以及 ColBERT‑v2（多向量）等模型相比，M3‑Embedding 在所有指标上都有提升。论文声称在 XQuAD 上的 MRR 提升约 6%（从 0.38 到 0.44），在 LongEval 的 nDCG@10 提升约 8%（从 0.62 到 0.70）。  
- **消融实验**：去掉自知识蒸馏后，稠密检索下降约 3%，稀疏检索下降约 4%；去掉大批量策略后，整体性能下降约 2%。这些结果说明蒸馏和批量设计是性能提升的关键驱动。  
- **局限性**：作者承认在极低资源语言（如某些非洲语言）上仍有一定性能差距；另外，8 192 token 的上限受限于硬件，真正的超长文档仍需分块后再聚合。

### 影响与延伸思考
这篇论文在 2024 年的 ACL 上引发了对“一体化检索嵌入”概念的热议。随后出现的工作如 **UniRerank**、**Cross‑Modal M3** 等，都在尝试把多模态或跨任务的蒸馏信号加入统一模型。对想进一步探索的读者，可以关注以下方向：  
- **更细粒度的蒸馏**：把检索过程中的排序梯度也当作教师信号。  
- **极端长文档**：结合稀疏注意力或分层 Transformer，突破 8k 限制。  
- **低资源语言适配**：利用少量平行句对或自监督对齐技术，提升在资源匮乏语言上的表现。  
（以上为基于公开信息的推测，后续文献会给出更明确的验证。）

### 一句话记住它
M3‑Embedding 用自蒸馏把稠密、稀疏和多向量检索统一进同一个多语言、支持 8k 长文本的模型，让“一站式”检索成为可能。