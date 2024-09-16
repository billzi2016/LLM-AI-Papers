# jina-embeddings-v3: Multilingual Embeddings With Task LoRA

> **Date**：2024-09-16
> **arXiv**：https://arxiv.org/abs/2409.10173

## Abstract

We introduce jina-embeddings-v3, a novel text embedding model with 570 million parameters, achieves state-of-the-art performance on multilingual data and long-context retrieval tasks, supporting context lengths of up to 8192 tokens. The model includes a set of task-specific Low-Rank Adaptation (LoRA) adapters to generate high-quality embeddings for query-document retrieval, clustering, classification, and text matching. Evaluation on the MTEB benchmark shows that jina-embeddings-v3 outperforms the latest proprietary embeddings from OpenAI and Cohere on English tasks, while achieving superior performance compared to multilingual-e5-large-instruct across all multilingual tasks. With a default output dimension of 1024, users can flexibly reduce the embedding dimensions to as low as 32 without compromising performance, enabled by Matryoshka Representation Learning.

---

# jina-embeddings-v3：支持任务 LoRA 的多语言嵌入模型 论文详细解读

### 背景：这个问题为什么难？

在检索、聚类、分类等下游任务里，文本向量（embedding）是核心桥梁。过去的开源模型要么只能处理单语言、要么只能接受几百个 token 的上下文，导致长文档或跨语言检索效果不佳。商业闭源模型虽然在规模上更大，但往往不可自由调优，用户只能使用统一的、固定维度的向量。更糟的是，通用向量在不同任务之间的表现差异大，缺少针对性调优手段。于是，如何在保持开源、可调的前提下，提供 **多语言、长上下文、可灵活降维** 的高质量嵌入，成为迫切需求。

### 关键概念速览

**文本嵌入（Text Embedding）**：把一段文字映射成固定长度的向量，向量之间的距离代表语义相似度。想象把句子压进一个多维空间的坐标点，越近的点越相似。

**LoRA（Low‑Rank Adaptation）**：在大模型上加一层低秩矩阵来微调，而不改动原始权重。就像在原有乐谱上加一段简短的即兴演奏，既省时又不破坏原曲。

**多语言模型（Multilingual Model）**：同一个网络能够处理多种语言的输入，类似于一把能弹奏多种调式的吉他。

**长上下文（Long‑Context）**：模型能够一次性读取数千甚至上万 token，而不是只能看到前几百个。相当于一次性阅读整篇文章，而不是只看前几段。

**Matryoshka Representation Learning**：在同一个向量里嵌套多个不同维度的子向量，像套娃一样，允许用户随时截取前 n 维得到更短的表示而不显著损失信息。

**检索适配器（Retrieval Adapter）**：专门为查询‑文档匹配任务训练的 LoRA，帮助模型在检索场景下产生更区分度高的向量。

**聚类/分离适配器（Clustering/Separation Adapter）**：针对无监督聚类或需要把相似文本“分开”的任务微调的 LoRA，提升向量在同类/异类之间的可分性。

### 核心创新点

1. **统一的 570M 多语言骨干 + 任务 LoRA 组合**  
   之前的多语言嵌入模型要么体积更大、要么缺少任务专用微调。jina‑embeddings‑v3 采用 XLM‑RoBERTa 作为 570M 参数的通用骨干，然后为每类下游任务（检索、分类、聚类、匹配）分别训练轻量 LoRA 适配器。这样既保留了大模型的语言通用能力，又能通过几千到几万参数的适配器快速获得任务专属的向量。

2. **两阶段预训练 + 长上下文扩展**  
   先在 512 token 长度上做大规模多语言掩码语言模型（MLM）预训练，随后继续在 8k token 长度上进行同构训练。传统做法直接在 512 token 上收敛，导致对长文档的表示力不足。两阶段训练让模型学会在更大窗口里捕捉跨句、跨段的语义关联。

3. **Matryoshka 表示让维度可裁剪**  
   通过在训练时强制向量的前 32、64、128…维保持独立判别能力，用户可以在推理时随意截取低维向量。相比于事后做 PCA 或降维微调，Matryoshka 让降维过程几乎无损，极大提升了在资源受限环境下的部署灵活性。

4. **在 MTEB 基准上全面超越商业闭源模型**  
   在英文检索、分类等任务上，jina‑embeddings‑v3 的分数超过 OpenAI、Cohere 最新的嵌入服务；在所有多语言子任务上也跑赢 multilingual‑e5‑large‑instruct。此成绩证明了“开源+任务 LoRA”组合的竞争力。

### 方法详解

#### 整体框架

jina‑embeddings‑v3 的训练与推理可以拆成三大块：  
1) **通用多语言骨干预训练**（512 → 8k token）  
2) **任务对齐微调**（使用文本对进行检索/匹配等任务）  
3) **任务 LoRA 适配器训练**（为每类下游任务单独学习低秩增量）。

在推理时，用户先加载通用骨干，然后根据需求加载对应的 LoRA（比如检索适配器），最后输出 1024 维向量或截取更低维。

#### 关键模块拆解

1. **骨干模型**  
   - 基于 XLM‑RoBERTa，拥有 12 层 Transformer，隐藏维度 1024。  
   - 预训练阶段使用 **CulturaX** 多语言语料，先在 512 token 长度上做标准 MLM（随机遮盖 15% token），随后在同样语料上扩展到 8k token，保持相同的遮盖比例。这样模型在一次前向传播中就能看到更长的上下文，学习跨段依赖。

2. **任务对齐微调**  
   - 采用 **文本对**（query‑document、sentence‑pair）作为输入，目标是让正对（相关）对的向量距离更近，负对（不相关）距离更远。  
   - 损失函数使用 **对比学习**（如 InfoNCE），在一个 batch 中同时计算多个正负样本的相似度。  
   - 这一步让骨干模型的输出已经具备一定的检索/匹配能力，但仍是通用的。

3. **LoRA 适配器**  
   - 对每层的 Query、Key、Value 矩阵插入低秩增量：`ΔW = A·B`，其中 A、B 的秩远小于原矩阵（典型秩为 4~8）。  
   - 训练时冻结骨干权重，只更新 A、B。这样参数量只增加几千到几万，而不需要重新训练整个 570M 网络。  
   - 适配器分为四类：**检索适配器、分类适配器、匹配适配器、聚类/分离适配器**。每类针对不同的相似度度量或标签结构进行微调。

4. **Matryosh套娃向量**  
   - 在对齐微调阶段，引入 **层级投影头**：在每个 Transformer 层的输出上分别投影出 32、64、128、256、512、1024 维向量，并强制这些子向量在对应维度上保持一致的相似度分布。  
   - 训练时对所有层级的投影都计算对比损失，确保前几维在信息上足够“浓缩”。  
   - 推理时，只取前 N 维即可得到低维嵌入，几乎不牺牲检索/分类性能。

#### 反直觉/巧妙之处

- **先长上下文再任务 LoRA**：很多人会先在长上下文上直接做任务微调，导致 LoRA 需要学习大量跨段信息。这里把长上下文的语言建模放在骨干阶段，让 LoRA 只专注于任务特定的相似度调节，显著提升收敛速度。  
- **低秩增量而非全模型微调**：在 570M 参数的模型上直接微调几乎不可行，LoRA 的低秩结构让每个任务只需几 MB 参数，极大降低了部署成本。  
- **Matryoshka 维度共享**：传统降维需要额外的投影或后处理，这里把不同维度的向量直接在训练时“嵌套”，省去后期步骤，也避免了信息泄漏。

### 实验与效果

- **评测基准**：使用 **MTEB（Massive Text Embedding Benchmark）**，覆盖 56 项任务，包括英文检索、分类、聚类以及 20+ 多语言子任务（如跨语言检索、非拉丁语种分类等）。  
- **对比模型**：OpenAI 的 text-embedding-3-large、Cohere 的 embed‑english‑v3、以及 multilingual‑e5‑large‑instruct（开源最强多语言基线）。  
- **核心结果**：  
  - 在英文检索任务上，jina‑embeddings‑v3 的 MRR 提升约 **+8%** 超过 OpenAI 同类模型。  
  - 在所有多语言任务的平均得分上，领先 multilingual‑e5‑large‑instruct **约 4~5%**。  
  - 维度裁剪实验显示，使用 **128 维**（即 1/8 原始维度）仍保持 **>95%** 的原始性能，验证 Matryoshka 的有效性。  
- **消融实验**：  
  - 去掉长上下文预训练，8k token 任务的检索 Recall 下降约 **12%**，说明长上下文学习是关键。  
  - 替换 LoRA 为全模型微调，性能提升不明显但训练成本提升 30 倍，验证 LoRA 已足够捕获任务差异。  
  - 移除 Matryoshka 投影，仅在 1024 维上训练，低维裁剪后性能下降超过 **20%**，凸显层级投影的价值。  
- **局限性**：  
  - 论文未提供对极端低资源语言（如斯瓦希里语）的专门评估，可能在这些语言上仍有提升空间。  
  - LoRA 适配器需要为每个新任务单独训练，虽然参数轻量，但仍需额外标注数据和微调步骤。  

### 影响与延伸思考

jina‑embeddings‑v3 的出现让「开源多语言长文本嵌入」从概念走向实用，尤其在需要自定义检索或聚类的企业内部系统中，提供了可自行微调且成本低的方案。随后出现的工作大多围绕 **任务适配器的自动化生成**（如使用元学习快速推断 LoRA 参数）和 **更细粒度的 Matryoshka 维度控制**（比如动态选择截取维度）展开。对想进一步探索的读者，可以关注：

- **LoRA 与 Prompt Tuning 的混合**：如何在同一模型上同时使用低秩增量和软提示，以兼顾任务适配和指令遵循。  
- **跨语言 LoRA 共享**：是否可以在不同语言的检索适配器之间共享部分低秩矩阵，降低多语言任务的微调成本。  
- **长上下文的稀疏注意力**：在 8k token 以上的场景，结合稀疏或局部注意力机制进一步提升效率。  

### 一句话记住它

**jina‑embeddings‑v3 用 LoRA 把 570M 多语言大模型变成可随任务、可裁剪维度的“即插即用”嵌入引擎。**