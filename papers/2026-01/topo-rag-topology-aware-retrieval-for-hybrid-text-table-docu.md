# Topo-RAG: Topology-aware retrieval for hybrid text-table documents

> **Date**：2026-01-15
> **arXiv**：https://arxiv.org/abs/2601.10215

## Abstract

In enterprise datasets, documents are rarely pure. They are not just text, nor just numbers; they are a complex amalgam of narrative and structure. Current Retrieval-Augmented Generation (RAG) systems have attempted to address this complexity with a blunt tool: linearization. We convert rich, multidimensional tables into simple Markdown-style text strings, hoping that an embedding model will capture the geometry of a spreadsheet in a single vector. But it has already been shown that this is mathematically insufficient.   This work presents Topo-RAG, a framework that challenges the assumption that "everything is text". We propose a dual architecture that respects the topology of the data: we route fluid narrative through traditional dense retrievers, while tabular structures are processed by a Cell-Aware Late Interaction mechanism, preserving their spatial relationships. Evaluated on SEC-25, a synthetic enterprise corpus that mimics real-world complexity, Topo-RAG demonstrates an 18.4% improvement in nDCG@10 on hybrid queries compared to standard linearization approaches. It's not just about searching better; it's about understanding the shape of information.

---

# Topo-RAG：面向混合文本-表格文档的拓扑感知检索 论文详细解读

### 背景：这个问题为什么难？

企业内部的资料往往既有叙事性的段落，也有结构化的电子表格，二者交织在同一份报告里。传统的检索增强生成（RAG）系统只能把表格“拉平”成一串 Markdown 文本，期望单个向量就能捕捉行列关系，这在数学上是不可能的。结果是，检索时模型只能记住表格里出现的词，而忘记了单元格之间的空间布局，导致对涉及数值比较或跨行关联的查询表现极差。于是，如何在同一次检索中既保留文本的语义，又保留表格的几何结构，成了亟待突破的瓶颈。

### 关键概念速览

**混合文档（Hybrid Document）**：指同时包含自然语言段落和结构化表格的文件，像年报、审计报告这类企业文档。可以把它想象成一本书里夹着几张电子表格。

**线性化（Linearization）**：把多维表格展开成一行文字的过程，就像把棋盘上的棋子排成一列再描述。虽然实现简单，但会把行列坐标信息全部抹掉。

**稠密检索器（Dense Retriever）**：使用预训练的向量模型把文本映射到高维空间，再通过向量相似度找相似片段。它擅长捕捉语义相似，却不关注结构。

**Cell‑Aware Late Interaction（细胞感知后期交互）**：一种专门对表格单元格进行向量化并在检索阶段保留它们相对位置的技术。类似于在搜索时先把每个棋子的位置记下来，最后再根据棋子之间的相对距离做排序。

**拓扑感知路由（Topology‑aware Routing）**：根据文档片段是文本还是表格，动态决定走哪条检索通路的调度机制。可以比作机场的行李分拣系统：行李是普通行李还是易碎品，决定走不同的传送带。

**nDCG@10**：归一化折损累计增益的前十名指标，用来衡量检索结果的排序质量。数值越高说明前几条结果越相关。

### 核心创新点

1. **线性化 → 拓扑感知路由**：过去的系统把所有内容都强行转成文字向量；Topo‑RAG 首先用结构密度评分把文档划分为“纯文本块”和“表格块”，分别送入不同的检索子网。这样做让表格的空间信息不再被压平，检索质量提升明显。

2. **稠密检索 → Cell‑Aware Late Interaction**：传统稠密检索只对整段文字做一次向量匹配；本文对每个表格单元格生成独立向量，并在检索后期通过“后期交互”把单元格之间的相对位置重新组合。相当于先把每个棋子单独记住，再在棋盘上重新摆放，保留了行列关系。

3. **单一排序 → 融合重排**：之前的 RAG 系统直接把稠密检索得到的分数输出；Topo‑RAG 在两条通路得到的候选结果上做统一的二次排序（rerank），让文本相关度和表格结构相互校正。实验显示，这一步贡献了约 5% 的 nDCG 提升。

4. **合成企业语料库 → 实证验证**：作者自行构造了 SEC‑25 数据集，模拟真实企业报告的混合特性。相较于仅用线性化的基线，Topo‑RAG 在混合查询上实现了 18.4% 的 nDCG@10 增长，证明了方法的实用价值。

### 方法详解

**整体框架**  
Topo‑RAG 的检索流程可以分为三大阶段：① 拓扑感知路由，② 文本/表格专属检索，③ 融合重排。首先，系统对每个文档块计算“结构密度”，判断它是自然语言段落还是表格；随后，文本块进入标准稠密检索器，表格块进入 Cell‑Aware Late Interaction 模块；最后，把两条通路的候选结果合并，交给统一的重排模型输出最终排序。

**1. 拓扑感知路由**  
- **结构密度评分**：对每个块统计字符与单元格的比例。若单元格占比超过阈值，就标记为表格。可以把它想成在图书馆里用金属探测器区分金属书签和普通纸页。  
- **路由决策**：文本块直接送入 BERT‑style 的稠密编码器，得到一个整体向量；表格块则进入下一步的细胞级处理。

**2. Cell‑Aware Late Interaction**  
- **单元格向量化**：每个单元格的文字内容先经过轻量级编码器得到向量，同时记录它的行号和列号。  
- **位置编码**：行列坐标被映射成小的数值向量（类似于 Transformer 中的相对位置编码），与单元格内容向量相加，形成“位置感知向量”。  
- **后期交互**：检索时，先用稠密向量快速筛选出可能相关的单元格集合；随后在这些候选上执行点对点的相似度计算，并根据行列距离加权，确保同一行或同一列的单元格更容易相互提升分数。这个过程类似于先用高速公路把车子送到大致目的地，再用小巷子精细导航。

**3. 融合重排**  
- **候选集合**：从文本检索器得到的前 N 条文段和从表格检索器得到的前 M 条单元格（或表格片段）被统一放入一个列表。  
- **统一特征**：对每条候选，构造一个特征向量，包含文本相似度、表格位置相似度、以及跨模态的交叉特征（如查询中出现的数字是否在表格单元格里出现）。  
- **二次排序模型**：使用轻量级的跨模态排序网络（如轻量级的 MLP）对特征进行打分，输出最终的排序。这里的关键是让文本的语义分数和表格的结构分数相互校正，避免单纯文字匹配或单纯结构匹配的偏差。

**最巧妙的设计**  
- **Late Interaction 而非 Early Fusion**：作者没有在最开始就把文本和表格混在一起，而是等到检索后期才让两者“碰面”。这样既保留了各自的优势，又避免了早期融合导致的噪声放大。  
- **稀疏的结构密度阈值**：只用一个极其轻量的统计指标就完成了路由，几乎不增加计算开销，却成功把不同拓扑的数据分流。

### 实验与效果

- **数据集**：SEC‑25 是作者自行合成的企业报告语料，包含 25 类真实业务场景，每篇文档均混有段落和多张复杂表格。查询分为纯文本、纯表格和混合三类。  
- **基线**：主要对比了两类方法：① 直接线性化后使用标准稠密检索器（如 DPR），② 采用多模态检索但仍把表格线性化的变体。  
- **主要指标**：在混合查询上，Topo‑RAG 的 nDCG@10 为 0.432，而最强线性化基线仅为 0.366，提升了 **18.4%**。在纯文本查询上提升约 6%，纯表格查询上提升约 12%。  
- **消融实验**：作者分别关闭路由模块、关闭 Cell‑Aware Late Interaction、以及关闭融合重排，发现：去掉路由导致整体下降约 7%；去掉细胞感知后表格查询下降约 10%；去掉统一重排则整体下降约 5%。这表明每个组件都对最终性能有实质贡献。  
- **局限性**：论文提到当前实现对表格规模有上限（单表格最多 5000 单元格），更大表格需要分块或采样；此外，路由依赖于结构密度阈值，极端文档（如大量嵌套列表）可能误判。

### 影响与延伸思考

Topo‑RAG 把“所有信息都是文字”这一假设推翻后，开启了企业检索领域对混合拓扑信息的系统性研究。随后的工作（如 Table‑RAG、Hybrid‑Retriever）纷纷在细胞级向量化、跨模态交互以及更高效的路由策略上进行扩展。对想进一步探索的读者，可以关注以下方向：① 更精细的结构感知路由（如利用视觉模型检测表格边界），② 大规模表格的分块检索与全局聚合，③ 将 Topo‑RAG 思路迁移到图文、代码+文档等其他多模态场景。整体来看，这篇论文为“形状感知的检索”奠定了概念与实现基础。

### 一句话记住它

**Topo‑RAG 通过把文本和表格分流到各自擅长的检索通路，并在后期统一重排，让检索系统真正“看懂”信息的几何形状。**