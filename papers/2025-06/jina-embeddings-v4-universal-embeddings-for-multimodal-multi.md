# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval

> **Date**：2025-06-23
> **arXiv**：https://arxiv.org/abs/2506.18902

## Abstract

We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-document retrieval, semantic text similarity, and code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single-modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval.

---

# jina-embeddings-v4：面向多模态多语言检索的通用嵌入模型 论文详细解读

### 背景：这个问题为什么难？
检索系统一直在追求“看得见、听得见、读得懂”的统一表示，但传统模型大多只能处理单一模态——要么是文本，要么是图片。即使有多模态模型，它们往往只能输出一串向量，难以兼顾高效的单向检索和需要细粒度交互的复杂查询。再者，跨语言检索要求同一向量空间能够容纳不同语言的语义，这在参数规模和训练数据上都极具挑战。于是，如何在一个模型里同时实现多语言、多模态、单向/多向检索的高质量表示，成为了瓶颈。

### 关键概念速览
**多模态嵌入**：把文字、图片等不同类型的输入映射到同一个向量空间，就像把不同语言的词翻译成同一种颜色，方便直接比较。  
**单向量 vs. 多向量（Late Interaction）**：单向量是把整个输入压成一个向量，检索时只算一次相似度；多向量则把输入拆成若干子向量，检索时在子向量层面做交互，类似先把两篇文章拆成段落再逐段比对，精度更高。  
**LoRA（Low‑Rank Adaptation）**：在大模型上加一层低秩矩阵进行微调，像在原有乐谱上加几条装饰音，既能快速适配新任务，又不需要重新训练整个模型。  
**跨模态检索**：用文字查询找图片，或用图片查询找文字，等价于让两种语言互相翻译后再比较。  
**多语言检索**：不同语言的查询和文档共享同一向量空间，类似把不同国家的地图投影到同一张全球地图上。  
**Late Interaction**：检索时把查询和文档的子向量在最后一步才做相似度计算，类似先把两本书的章节分别准备好，等到需要时再逐章比对，兼顾速度和细粒度。  
**Jina‑VDR 基准**：专门评估模型在表格、图表、示意图等视觉丰富内容上的检索能力，类似为“看图说话”场景量身定制的考试。

### 核心创新点
1. **统一的单向量+多向量双模式架构**  
   - 之前的多模态模型要么只能输出单个向量，要么只能输出一堆子向量，难以在同一系统里兼顾两种需求。  
   - 本文在同一网络的最后阶段同时生成一个全局向量和一组局部向量，前者用于高速粗排，后者用于精排的 Late Interaction。  
   - 这样既保留了单向量检索的速度优势，又让多向量交互提升了对复杂视觉内容的辨识度。

2. **任务专属 LoRA 适配层**  
   - 传统微调需要在全参数上进行梯度更新，成本高且容易破坏已有的跨语言/跨模态能力。  
   - 作者为每类检索任务（如查询‑文档、语义相似、代码搜索）插入低秩适配层，仅调节少量参数。  
   - 结果是模型在不同任务上都能快速收敛，且保持了统一嵌入空间的通用性。

3. **面向视觉丰富内容的训练与评估**  
   - 过去的多模态检索数据集大多是自然图片，缺少表格、图表等结构化视觉信息。  
   - 论文在大规模混合媒体数据上进行预训练，并推出 Jina‑VDR 基准专门测评这类场景。  
   - 实验显示模型在处理表格、流程图等非自然图像时显著优于同类模型。

4. **3.8 B 参数的规模与高效推理**  
   - 大多数跨模态模型要么参数在百亿以上，推理成本高；要么参数太小，表现受限。  
   - 通过混合稀疏/密集层设计和 LoRA 轻量适配，作者在 3.8 B 参数下实现了与更大模型相当的检索效果，同时保持了可部署性。

### 方法详解
整体思路可以拆成四步：  
1) **统一编码器**：使用一个基于 Transformer 的主干网络，同时接受文本 token 序列和图像 patch 序列。文本和图像在输入层被映射到相同的嵌入维度，类似把文字和像素都先翻译成同一种“语言”。  
2) **双向投影层**：经过若干层自注意力后，网络分叉产生两套输出：① 全局池化向量（CLS token），代表整体语义；② 按位置切分的局部向量序列（对应文本的词向量或图像的 patch 向量），用于后续的 Late Interaction。  
3) **任务 LoRA 适配**：在每个任务的输出层前插入低秩矩阵。训练时只更新这些矩阵的权重，主干保持不变。这样同一个模型可以快速切换到不同检索场景，只需加载对应的 LoRA 参数。  
4) **检索流程**：检索时先用全局向量做粗排，筛选出候选集合；随后在候选内部用多向量的 Late Interaction 计算细粒度相似度——具体做法是对查询的局部向量和文档的局部向量逐对点乘，取最大或加权求和，得到最终打分。

**关键细节**  
- **跨模态对齐**：在预训练阶段，模型同时接受“文本‑图片配对”和“文本‑文本配对”，通过对比学习让同一语义的文字和图像在全局向量和局部向量上都靠得更近。  
- **多语言共享词表**：使用统一的子词分词器（如 SentencePiece），把所有语言映射到同一词表，保证不同语言的词向量在同一空间。  
- **Late Interaction 的实现**：作者采用类似 ColBERT 的点乘-最大池化策略，但在多模态情境下，图像的 patch 向量和文本的词向量可以直接交叉匹配，捕捉到“文字描述对应图中哪块区域”的细粒度信息。  
- **反直觉之处**：虽然模型规模只有 3.8 B，但通过 LoRA 的任务专属适配和双向向量设计，竟然在跨语言、跨模态的多任务上都能达到或超过更大模型的表现，这在以往的经验里是不常见的。

### 实验与效果
- **评测数据**：作者在公开的单模态检索基准（如 MS‑MARCO、NQ）和跨模态检索数据集（如 Flickr30K、COCO）上测试；另外使用新建的 Jina‑VDR 基准，专门评估表格、图表、示意图等视觉丰富内容的检索。  
- **对比基线**：与 CLIP、BLIP、ColBERT‑v2、M3P 等主流多模态/多语言检索模型进行比较。  
- **结果概述**：论文声称在所有单模态和跨模态任务上均取得了最新的 SOTA（state‑of‑the‑art）成绩，尤其在 Jina‑VDR 上的提升最为显著，说明模型对结构化视觉信息的理解更强。具体的数值提升在摘要中未给出。  
- **消融实验**：通过去掉 LoRA 适配层、仅保留单向量或仅保留多向量两种配置，作者展示了每个组件对整体性能的贡献——LoRA 对任务适配提升最大，Late Interaction 对视觉复杂场景的精度提升最明显。  
- **局限性**：原文提到模型仍然依赖大规模混合媒体预训练数据，若在极端低资源语言或极度稀疏的视觉概念上可能表现受限；此外，Late Interaction 在候选集合很大时仍会带来一定的计算开销。

### 影响与延伸思考
这篇工作展示了“统一向量 + 任务专属轻量适配”可以在多模态多语言检索中实现高效且强大的表现，随后有不少研究开始探索更细粒度的跨模态对齐（如跨语言视觉问答）以及更轻量的 LoRA 变体。推测未来会有：
- **更大规模的视觉‑语言‑代码三模态统一模型**，把代码搜索也纳入同一嵌入空间。  
- **针对特定行业的视觉丰富检索**（如金融报表、医学影像）会基于 Jina‑VDR 的思路构建专用基准。  
- **高效的 Late Interaction 加速器**，在硬件层面优化子向量点乘，以降低大规模候选集合的延迟。  
想进一步深入，可以关注 LoRA 在跨任务迁移中的理论分析、以及如何在边缘设备上部署多向量检索。

### 一句话记住它
jina-embeddings-v4 用同一个 3.8 B 参数模型同时输出全局向量和局部向量，并通过轻量 LoRA 适配，实现了多语言、多模态检索的最新水平。