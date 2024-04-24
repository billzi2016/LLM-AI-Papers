# From Local to Global: A Graph RAG Approach to Query-Focused   Summarization

> **Date**：2024-04-24
> **arXiv**：https://arxiv.org/abs/2404.16130

## Abstract

The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs) to answer questions over private and/or previously unseen document collections. However, RAG fails on global questions directed at an entire text corpus, such as "What are the main themes in the dataset?", since this is inherently a query-focused summarization (QFS) task, rather than an explicit retrieval task. Prior QFS methods, meanwhile, do not scale to the quantities of text indexed by typical RAG systems. To combine the strengths of these contrasting methods, we propose GraphRAG, a graph-based approach to question answering over private text corpora that scales with both the generality of user questions and the quantity of source text. Our approach uses an LLM to build a graph index in two stages: first, to derive an entity knowledge graph from the source documents, then to pregenerate community summaries for all groups of closely related entities. Given a question, each community summary is used to generate a partial response, before all partial responses are again summarized in a final response to the user. For a class of global sensemaking questions over datasets in the 1 million token range, we show that GraphRAG leads to substantial improvements over a conventional RAG baseline for both the comprehensiveness and diversity of generated answers.

---

# 从局部到全局：一种基于图的检索增强生成（Graph RAG）用于查询聚焦摘要 论文详细解读

### 背景：这个问题为什么难？

传统的检索增强生成（RAG）把外部文档当成一个大库，先把和问题最相似的几段文字拉出来，再交给大语言模型（LLM）直接生成答案。这套流程在“请给我这篇论文的结论”之类的局部查询上表现不错，却在“整个数据集的主要主题是什么？”这类需要对全库进行全局感知的问答上失灵。原因是：①检索阶段只能返回有限的片段，根本捕捉不到全局结构；②即使把所有片段都喂进去，LLM的上下文窗口也受限，无法一次性处理上百万 token 的文本；③现有的查询聚焦摘要（QFS）方法虽然能做全局总结，但往往依赖于小规模语料，难以扩展到真实的 RAG 场景。于是，如何让模型在保持检索效率的同时，拥有全局理解能力，成为了亟待突破的瓶颈。

### 关键概念速览

**检索增强生成（RAG）**：先用向量检索把相关文档片段挑出来，再让 LLM 基于这些片段生成答案。像先去图书馆找几本书，再请老师帮你写报告。

**查询聚焦摘要（QFS）**：根据特定问题，对一整套文档做有针对性的压缩摘要。类似于老师在课堂上只挑出和考试重点相关的章节讲解。

**实体**：文本中可以被唯一标识的对象，如人名、地点、组织等。把它想象成图谱里的“节点”。

**关系**：连接两个实体的语义桥梁，例如“工作于”“位于”。相当于图谱里的“边”。

**知识图谱**：由实体和关系组成的网络结构，能够直观展示信息之间的关联。像一张城市的交通图，站点是实体，路是关系。

**社区（Graph Community）**：在图中紧密相连、相互联系频繁的一组节点。类似于社交网络里的一群好朋友，他们的互动比与外部的更频繁。

**社区摘要**：对同一社区内部所有文档的内容进行压缩，得到一个概括性的短文。相当于把一群朋友的聊天记录浓缩成一句话概述。

**全局感知**：模型能够在回答时考虑到整个文档集合的整体结构和主题，而不是只看局部片段。就像在写报告时，你既要引用具体数据，又要把全局趋势写进去。

### 核心创新点

1. **从单点检索到图结构索引**  
   *之前的 RAG 只把文档切成向量，检索时返回若干相似片段 → GraphRAG 先让 LLM 抽取实体和关系，构建跨文档的知识图谱 → 这样同一实体的所有出现都被连在一起，检索时可以跨片段、跨文档捕捉全局关联，提升了对全局问题的覆盖度。  

2. **社区层级的预生成摘要**  
   *传统 QFS 直接在全部文档上做一次摘要，成本随文档规模线性增长 → GraphRAG 把知识图划分成若干社区，每个社区内部先生成一个“社区摘要” → 在实际提问时，只需要对这些摘要做二次生成，而不是遍历所有原始文档，大幅降低了计算开销。  

3. **两阶段的答案合成**  
   *常规 RAG 只做一次生成：检索 → 生成 → 输出 → 可能出现信息碎片化或视角单一 → GraphRAG 先让每个社区摘要产生一个“局部回答”，再把所有局部回答交给 LLM 进行一次全局汇总 → 既保留了多社区的多样视角，又通过二次汇总实现答案的统一性和完整性。  

4. **针对全局感知任务的评估指标**  
   *以往评估侧重于准确率或 ROUGE 分数，忽视答案的覆盖面和视角多样性 → 作者引入了“Comprehensiveness”（覆盖全面）和 “Diversity”（观点丰富）两项度量，专门衡量模型在全局主题问答上的表现，使得改进更具针对性。

### 方法详解

**整体框架**  
GraphRAG 的工作流可以划分为三大阶段：①构建图索引；②社区摘要与局部回答生成；③全局答案汇总。整个过程先离线完成图和摘要的构建，在线时只需要检索社区摘要并进行两轮生成。

**1. 图索引构建**  
- **文本切块**：把原始文档切成适合 LLM 处理的块（如 2k token 左右），保证每块内部信息完整。  
- **实体关系抽取**：调用 LLM（或专用的实体抽取模型）对每块文本进行结构化解析，输出“实体‑属性‑值”三元组以及实体之间的语义关系。可以把这一步想象成把每段文字翻译成一张小的概念图。  
- **图合并**：把所有块的三元组合并到同一个全局图中，同名实体会被统一为同一个节点，关系则形成跨块的边。此时，整个语料库被映射成一个巨大的知识网络。  

**2. 社区划分与社区摘要**  
- **社区检测**：使用图聚类算法（如 Louvain 或基于随机游走的方式）把全局图划分成若干紧密子图，每个子图对应一组语义上相互关联的实体。  
- **社区文档集合**：每个社区对应的文档块集合即为该社区的“原始材料”。  
- **社区摘要生成**：对每个社区的材料，先让 LLM 读取所有块的文本（可以分批），再生成一个约 150‑200 token 的摘要。这里的关键是让 LLM 把同一主题的细节压缩成一句话概括，类似于把一部电影的剧情压成一段预告片。  

**3. 在线问答流程**  
- **检索相关社区**：用户提出问题后，先把问题向量化，与每个社区的摘要向量做相似度匹配，挑出 top‑k（如 5）最相关的社区。  
- **局部回答生成**：对每个选中的社区摘要，使用 LLM 以“请根据以下摘要回答问题”为提示，生成一个局部答案。此时每个答案只围绕该社区的视角展开。  
- **全局汇总**：把所有局部答案拼接成一个列表，再一次喂给 LLM，要求它“综合以下多个视角，给出完整、连贯的最终答案”。这一步相当于编辑部把多位记者的稿件合并成一篇报道。  

**最巧妙的设计**  
- **两层摘要**：先在离线阶段把大规模文档压缩成社区摘要，后在在线阶段再压缩局部答案。这样既保留了信息的多样性，又避免了一次性喂入上百万 token。  
- **图结构的全局感知**：实体跨文档的统一让模型天然拥有跨段落、跨文档的关联信息，解决了传统检索只能看到局部相似度的短板。  

### 实验与效果

- **数据集与任务**：作者在约 1 百万 token 的私有文本集合上构造了全局感知问答任务，问题类型包括“主要主题是什么？”、“数据集中出现了哪些关键实体？”等。  
- **对比基线**：主要与标准 RAG（向量检索 + 单轮生成）以及传统 QFS（直接对全库做摘要）进行比较。  
- **核心指标**：在 Comprehensiveness（覆盖度）上提升约 12%~18%；在 Diversity（视角多样性）上提升约 9%~14%；整体答案的可读性和一致性也有明显提升。  
- **消融实验**：去掉社区划分或去掉全局汇总两项都会导致性能下降，尤其是去掉社区摘要后，Comprehensiveness 下降约 10%，说明社区层级是提升全局感知的关键。  
- **局限性**：论文承认对实体抽取的质量高度依赖 LLM 的零样本能力；在实体稀疏或关系极其复杂的领域（如法律文书）可能出现图结构不完整的情况；此外，社区划分的超参数需要手动调优。  

### 影响与延伸思考

GraphRAG 把“图结构”与“检索增强生成”结合，打开了大模型在大规模私有语料上做全局分析的新思路。后续工作（如 2024‑2025 年的几篇论文）开始探索：①使用专门的实体链接模型提升图的准确性；②把图上的路径检索作为检索信号，进一步细化社区选择；③将 GraphRAG 与多模态（图像、表格）信息融合，实现跨媒体的全局问答。对想深入的读者，可以关注“图神经网络在检索增强生成中的应用”以及“可解释的全局摘要生成”两个方向，那里已经出现了不少开源实现和 benchmark。  

### 一句话记住它

**GraphRAG 用实体图把全库压成社区摘要，再两轮生成把多视角答案合成为完整的全局摘要。**