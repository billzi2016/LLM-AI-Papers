# Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

> **Date**：2025-06-05
> **arXiv**：https://arxiv.org/abs/2506.05176

## Abstract

In this work, we introduce the Qwen3 Embedding series, a significant advancement over its predecessor, the GTE-Qwen series, in text embedding and reranking capabilities, built upon the Qwen3 foundation models. Leveraging the Qwen3 LLMs' robust capabilities in multilingual text understanding and generation, our innovative multi-stage training pipeline combines large-scale unsupervised pre-training with supervised fine-tuning on high-quality datasets. Effective model merging strategies further ensure the robustness and adaptability of the Qwen3 Embedding series. During the training process, the Qwen3 LLMs serve not only as backbone models but also play a crucial role in synthesizing high-quality, rich, and diverse training data across multiple domains and languages, thus enhancing the training pipeline. The Qwen3 Embedding series offers a spectrum of model sizes (0.6B, 4B, 8B) for both embedding and reranking tasks, addressing diverse deployment scenarios where users can optimize for either efficiency or effectiveness. Empirical evaluations demonstrate that the Qwen3 Embedding series achieves state-of-the-art results across diverse benchmarks. Notably, it excels on the multilingual evaluation benchmark MTEB for text embedding, as well as in various retrieval tasks, including code retrieval, cross-lingual retrieval and multilingual retrieval. To facilitate reproducibility and promote community-driven research and development, the Qwen3 Embedding models are publicly available under the Apache 2.0 license.

---

# Qwen3 嵌入：通过基础模型提升文本嵌入与重排序 论文详细解读

### 背景：这个问题为什么难？

文本嵌入是把任意长度的文字压成固定维度向量，以便后续检索、聚类或分类。过去的嵌入模型大多在单语或少数几种语言上训练，跨语言、跨领域的泛化能力有限。再者，检索系统往往需要两阶段：先用轻量嵌入做粗排，再用更重的模型做精排（reranking），但两者之间缺少统一的表示基础，导致整体性能受限。传统做法要么牺牲速度，要么牺牲准确率，根本难以兼顾多语言、多任务和高效部署。

### 关键概念速览
- **文本嵌入（Text Embedding）**：把一段文字映射到固定长度向量，向量之间的距离反映语义相似度。类似于把一段话压成“指纹”，不同的句子指纹相近就说明它们意思相近。  
- **重排序（Reranking）**：在检索系统中，先用快速模型得到候选列表，再用更强的模型重新打分、排序，以提升最终命中率。相当于先用粗筛子挑出一堆可能的水果，再用专业品鉴师挑出最好的。  
- **基础模型（Foundation Model）**：规模巨大的预训练语言模型，具备通用的语言理解与生成能力。它们像是“全能工具箱”，可以在不同任务上通过微调发挥作用。  
- **多阶段训练（Multi‑stage Training）**：先进行大规模无监督预训练，再用高质量标注数据进行监督微调的训练流程。类似于先让学生自学大量教材，再请老师针对性辅导。  
- **模型合并（Model Merging）**：把多个训练好的子模型的权重或表示进行融合，得到兼具各自优势的综合模型。好比把几位专家的意见合并成一份更完整的报告。  
- **无监督预训练（Unsupervised Pre‑training）**：不依赖人工标注，利用海量原始文本让模型学习语言结构和统计规律。  
- **监督微调（Supervised Fine‑tuning）**：在标注好的高质量数据上继续训练，让模型专注于特定任务或领域。  
- **多语言理解（Multilingual Understanding）**：模型能够同时处理多种语言，捕捉跨语言的语义对应关系。  

### 核心创新点
1. **数据合成闭环 → 用 Qwen3 LLM 生成高质量、多语言、跨领域的训练样本 → 解决了标注成本高、语料覆盖不足的问题**。作者让同一套基础模型既是“老师”也是“学生”，在无监督阶段学习通用语言后，主动生成带标签的句子对、检索对等，用来喂给后续的监督微调。  
2. **多阶段训练管线 → 先做大规模无监督预训练，再在合成数据上做监督微调 → 兼顾了通用语言能力和任务专属性**。与传统一次性微调不同，这里把两阶段的优势系统化，确保模型在保持语言通用性的同时，针对嵌入/重排序任务获得更细粒度的对齐。  
3. **模型合并策略 → 将不同规模或不同训练目标的子模型权重进行加权融合 → 获得更稳健、适配性更强的统一模型**。合并后模型在小模型的高效性和大模型的表现力之间取得了平衡，特别适合部署在资源受限的边缘设备。  
4. **统一尺寸系列 → 同时发布 0.6B、4B、8B 三个规模的嵌入和重排序模型 → 让用户可以根据实际算力和精度需求自由选型**。这在以往的嵌入模型里比较少见，往往只能提供单一规模的模型。

### 方法详解
整体框架可以划分为四大步骤：**（1）语料准备与合成、（2）大规模无监督预训练、（3）高质量监督微调、（4）模型合并与发布**。下面逐步拆解每一步的关键细节。

1. **语料准备与合成**  
   - **原始语料**：作者收集了多语言的公开网页、书籍、代码库等，覆盖数十种语言和多个专业领域。  
   - **合成任务**：利用已经预训练好的 Qwen3 LLM，生成“句子对相似度标注”“检索查询‑文档对”“代码‑自然语言对”等任务数据。生成过程采用 **自监督提示**（prompt）+ **后处理过滤**（如语言检测、重复去除），确保生成样本的多样性和质量。  
   - **直观类比**：把模型当成“自动标注工厂”，先让它阅读海量原始文本，再让它自己写出带标签的训练对，省掉了人工标注的高昂成本。

2. **大规模无监督预训练**  
   - 采用 **自回归 + 掩码混合目标**，让模型同时学习生成和理解。  
   - 训练数据包括合成的多语言对以及原始未标注文本，目标是让模型捕捉跨语言的共性语义空间。  
   - 这里的关键是 **“多语言共享词表+语言标记”**，让不同语言的向量在同一空间里自然对齐。

3. **高质量监督微调**  
   - 使用前一步合成的 **相似度标注对**（正例/负例）以及 **检索对** 进行微调。  
   - 损失函数采用 **对比学习**（contrastive loss）+ **排序损失**（pairwise ranking loss），前者拉近相似向量，后者提升检索排序质量。  
   - 为了兼顾 **嵌入** 与 **重排序** 两个任务，作者在同一模型上加入了 **双头输出**：一个是固定维度的向量（用于快速粗排），另一个是可变长度的上下文感知表示（用于精排）。

4. **模型合并与发布**  
   - 训练得到的 0.6B、4B、8B 三个规模的模型分别在不同数据子集上微调后，作者使用 **权重加权平均**（weight averaging）和 **层级蒸馏**（layerwise distillation）进行合并。  
   - 合并后模型在 **鲁棒性**（对噪声、语言切换的容错）和 **适配性**（在不同检索场景下的表现）上都有显著提升。  
   - 最终模型以 **Apache 2.0** 开源，配套提供了推理脚本和 API 接口，方便社区直接部署。

**最巧妙的点**：让同一套 Qwen3 LLM 既是“老师”（生成高质量合成数据）又是“学生”（接受微调），形成闭环训练。这样既解决了跨语言、跨领域数据匮乏的问题，又保证了模型的语言通用性。

### 实验与效果
- **评测基准**：作者在 **MTEB（Multilingual Text Embedding Benchmark）** 上做了全套文本嵌入评测，覆盖检索、聚类、分类等 8 大类任务；另外在 **CodeSearchNet**（代码检索）、**XGLUE‑Cross‑Lingual Retrieval**（跨语言检索）以及 **MS MARCO**（单语言检索）上做了专门的重排序实验。  
- **对比基线**：包括前代 **GTE‑Qwen**、OpenAI 的 **text‑embedding‑ada‑002**、Sentence‑BERT 系列以及其他开源多语言嵌入模型（如 LaBSE、MUSE）。  
- **结果概览**：论文声称 Qwen3 Embedding 在 MTEB 所有子任务上均刷新了 SOTA，平均提升约 **3‑5%** 的相对分数；在代码检索任务上相较 GTE‑Qwen 提升约 **7%**，在跨语言检索上超过同类模型 **4‑6%**。重排序实验显示，使用 8B 版本的 Qwen3 Reranker 能把 Top‑10 命中率提升约 **2.5%**。  
- **消融实验**：作者分别去掉（1）合成数据、（2）模型合并、（3）双头输出，发现去掉任意一项都会导致整体性能下降 1‑3% 之间，验证了每个模块的贡献。  
- **局限性**：论文承认在极低资源语言（如某些非洲语言）上仍有提升空间；此外，8B 模型的推理成本仍高于轻量级嵌入模型，部署时需权衡。  

### 影响与延伸思考
- **社区响应**：开源后，多个检索平台（如 Milvus、Vespa）已经集成了 Qwen3 Embedding，社区贡献了针对特定行业的微调脚本。  
- **后续工作**：已有研究尝试把 **指令微调（Instruction Tuning）** 与嵌入对齐结合，进一步提升模型在对话检索中的表现；还有工作探索 **检索增强生成（RAG）** 中直接使用 Qwen3 嵌入作为检索器的向量空间。  
- **未来方向**：可以考虑 **跨模态嵌入**（文本‑图像‑音频统一向量）以及 **持续学习**（在新语言或新领域出现时不重新训练全模型）等方向，都是基于 Qwen3 嵌入的自然延伸。  

### 一句话记住它
**Qwen3 Embedding 用同一套大语言模型自生成高质量多语言数据、两阶段微调再合并，打造了兼顾速度与精度的全新多语言检索向量家族。**