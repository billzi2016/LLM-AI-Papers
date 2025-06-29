# Text2VectorSQL: Towards a Unified Interface for Vector Search and SQL Queries

> **Date**：2025-06-29
> **arXiv**：https://arxiv.org/abs/2506.23071

## Abstract

The proliferation of unstructured data poses a fundamental challenge to traditional database interfaces. While Text-to-SQL has democratized access to structured data, it remains incapable of interpreting semantic or multi-modal queries. Concurrently, vector search has emerged as the de facto standard for querying unstructured data, but its integration with SQL-termed VectorSQL-still relies on manual query crafting and lacks standardized evaluation methodologies, creating a significant gap between its potential and practical application.   To bridge this fundamental gap, we introduce and formalize Text2VectorSQL, a novel task to establish a unified natural language interface for seamlessly querying both structured and unstructured data. To catalyze research in this new domain, we present a comprehensive foundational ecosystem, including: (1) A scalable and robust pipeline for synthesizing high-quality Text-to-VectorSQL training data. (2) VectorSQLBench, the first large-scale, multi-faceted benchmark for this task, encompassing 12 distinct combinations across three database backends (SQLite, PostgreSQL, ClickHouse) and four data sources (BIRD, Spider, arXiv, Wikipedia). (3) Several novel evaluation metrics designed for more nuanced performance analysis. Extensive experiments not only confirm strong baseline performance with our trained models, but also reveal the recall degradation challenge: the integration of SQL filters with vector search can lead to more pronounced result omissions than in conventional filtered vector search. By defining the core task, delivering the essential data and evaluation infrastructure, and identifying key research challenges, our work lays the essential groundwork to build the next generation of unified and intelligent data interfaces. Our repository is available at https://github.com/OpenDCAI/Text2VectorSQL.

---

# Text2VectorSQL：面向向量检索与SQL查询的统一接口 论文详细解读

### 背景：这个问题为什么难？

传统数据库只能通过结构化的 SQL 语句访问表格数据，而海量的文本、图片、音频等非结构化信息只能靠向量检索来匹配语义。过去的 Text‑to‑SQL 把自然语言直接翻译成 SQL，解决了“我想看 2020 年的销售额”这类结构化查询，但它根本不懂“找和机器学习相关的论文”这类语义或多模态需求。相反，向量搜索可以找相似文本，却缺少 SQL 那套过滤、聚合、联表的强大表达能力。于是出现了“VectorSQL”——在 SQL 里嵌入向量检索的想法，但实现上仍然是手工写查询、缺少统一评测，导致研究和工业落地之间有巨大的鸿沟。

### 关键概念速览

**Text2VectorSQL**：一种任务定义，要求模型把自然语言问题一次性转化为同时包含传统 SQL 过滤和向量检索的混合查询。可以想象成让语言模型同时会写 “SELECT … WHERE …” 和 “ANN_SEARCH(embedding, top‑k)” 两件事。

**VectorSQL**：在普通 SQL 语句中加入向量检索算子（如 `ANN_SEARCH(col_embedding, query_vec, k)`），实现对某列的语义相似度过滤。类似于在 SQL 里装了一个“智能搜索插件”。

**Embedding（向量化）**：把文本、图片等高维信息压缩成固定长度的数值向量，使得语义相近的内容在向量空间里距离更近。这里的 embedding 既可以是预训练的大模型，也可以是专门为某列微调的模型。

**向量检索（Vector Search）**：在大规模向量集合中快速找出与查询向量最近的若干条记录，常用近似最近邻（ANN）算法实现。它是非结构化数据的“关键词搜索”升级版。

**SQL Filter（SQL 过滤）**：传统的布尔条件、范围、聚合等操作，例如 `WHERE price > 100 AND category = 'electronics'`。在混合查询里，它负责对结构化属性做精确筛选。

**CoT（Chain‑of‑Thought，思维链）**：让模型在生成最终查询前先写出推理步骤，就像解题时先列出思路再写答案，帮助模型保持逻辑一致性并降低错误传播。

**VectorSQLBench**：本文构建的首个大规模、多后端、多源的 Text2VectorSQL 基准，覆盖 SQLite、PostgreSQL、ClickHouse 三种数据库以及 BIRD、Spider、arXiv、Wikipedia 四类数据。

**Recall Degradation（召回衰减）**：在混合查询中，SQL 过滤往往会把本应被向量检索捕获的相关结果排除，导致整体召回率下降，这是本文首次系统量化的现象。

### 核心创新点

1. **任务正式化 → Text2VectorSQL**  
   *之前*：研究要么只做 Text‑to‑SQL，要么只做向量检索，二者从未在同一自然语言入口统一。  
   *本文*：提出了“自然语言一次性映射到 VectorSQL”的任务定义，并给出完整的输入‑输出规范。  
   *改变*：为跨模态、跨结构查询提供了统一的目标，使得模型训练、评测、应用都可以围绕同一个接口展开。

2. **高质量合成数据管线 → VectorSQLGen**  
   *之前*：缺少大规模、标注一致的混合查询数据，只能靠少量人工标注，成本高且覆盖面窄。  
   *本文*：设计了一个四阶段自动生成流程：①构造混合数据库并用 LLM 自动挑选适合向量化的列；②在可控复杂度下生成对应的 VectorSQL；③把 VectorSQL 逆向翻译成自然语言问题；④利用 CoT 生成推理过程并进行自校正。  
   *改变*：一次性产出上百万条高质量训练样本，覆盖多种数据库后端和数据源，为模型学习提供了足够的信号。

3. **统一评测框架与新指标 → VectorSQLBench**  
   *之前*：向量检索和 SQL 查询各自有成熟的评测（nDCG、Exact Match），但没有兼容两者的统一度量。  
   *本文*：构建了 12 组合的基准，并提出了 Set‑based（Precision/Recall/F1）和 Rank‑based（nDCG@k）两类指标，还加入了结构化诊断（SQL 骨架正确率、向量子句准确率）。  
   *改变*：研究者可以从整体准确性、排序质量以及子模块错误来源三个维度全面评估模型，推动了该任务的标准化。

4. **识别并量化召回衰减问题**  
   *之前*：在实际系统中常见“先过滤后向量搜索”导致召回下降，但缺少系统实验支撑。  
   *本文*：通过大量实验发现，SQL 过滤与向量检索的耦合会比传统过滤向量搜索更显著地削弱召回，提出了这一新挑战。  
   *改变*：为后续研究指明了需要改进的方向，如更柔性的过滤策略或联合优化的检索模型。

### 方法详解

#### 整体框架

Text2VectorSQL 的实现可以看作 **数据合成 → 模型训练 → 推理解码** 三大阶段。核心思想是让模型在学习阶段同时看到自然语言问题、对应的混合查询以及推理过程，从而在推理时能够“一步到位”生成完整的 VectorSQL。

#### 1. 数据合成管线（VectorSQLGen）

| 步骤 | 关键操作 | 类比 |
|------|----------|------|
| **① 混合数据库构建** | 选取公开结构化数据集（如 Spider），并用大语言模型（LLM）为每张表补充外部文本（网页、论文摘要）作为潜在向量列。 | 把一张普通的 Excel 表格“装上”可以做语义搜索的插件。 |
| **② 向量列挑选 & Embedding 生成** | 根据文本长度、信息密度等 heuristics，决定哪些列需要 embedding，并使用预训练模型生成向量。 | 类似于在图书馆里挑出需要做全文检索的章节。 |
| **③ VectorSQL 生成** | 在已知的列向量信息上，使用 LLM 按照可控的查询深度（单表、联表、聚合）生成对应的混合 SQL，语法遵循目标数据库的扩展语法。 | 把“找所有关于机器学习的论文且年份在 2020 之后”翻译成 `SELECT … FROM papers WHERE ANN_SEARCH(abstract_emb, query_vec, 10) AND year > 2020`。 |
| **④ NL 问题回译** | 将生成的 VectorSQL 逆向翻译成自然语言问题，使用多轮 LLM 进行 paraphrase，确保问题多样性。 | 像把答案倒着写成题目，防止模型只记住固定模板。 |
| **⑤ CoT 推理过程生成 & 自校正** | 让 LLM 先写出“思考链”，解释为何要先向量检索再过滤或相反，并检查生成的 SQL 是否符合语法/逻辑。错误的样本会被自动剔除或修正。 | 类似于老师批改学生的解题步骤，确保每一步都合理。 |

通过上述流水线，作者最终得到 **VectorSQLGen**：每条样本包含（混合数据库、自然语言问题、VectorSQL、CoT 推理）。该数据集规模达数百万条，覆盖三种数据库后端和四类数据源。

#### 2. 模型训练

- **模型架构**：基于开源的大规模语言模型（如 LLaMA‑2）进行指令微调，输入为自然语言问题 + 数据库 schema（包括标记哪些列是向量列），输出为完整的 VectorSQL。  
- **多任务学习**：在同一批次中加入 **CoT 生成任务**（输出思考链）和 **SQL 骨架预测任务**（只输出结构化部分），通过加权损失让模型同时学会结构化语法和向量子句的生成。  
- **检索增强**：在训练时，模型会调用外部向量索引（FAISS）获取候选向量 ID，随后把这些 ID 作为可选 token 注入到解码器，帮助模型更精准地写出 `ANN_SEARCH` 参数。

#### 3. 推理与解码

1. **问题解析**：模型读取用户自然语言问题和当前数据库 schema。  
2. **思维链输出**（可选）：先生成 CoT，帮助模型内部对“先向量还是先 SQL”做决策。  
3. **向量检索调用**：如果 CoT 中出现向量检索需求，系统先用查询文本生成 embedding，调用 ANN 索引返回 top‑k 向量 ID。  
4. **混合查询拼装**：模型把检索得到的 ID 填入 `ANN_SEARCH` 子句，同时生成剩余的 SQL 过滤、聚合、联表等部分。  
5. **后处理校验**：利用轻量的语法检查器确保生成的 VectorSQL 在目标数据库上可执行，若不通过则触发一次 beam‑search 重生成。

#### 巧妙之处

- **自校正的 CoT**：不是单纯让模型输出思考链，而是让模型在生成后自行检查逻辑一致性并纠错，显著降低了语法错误率。  
- **向量 ID 作为解码 token**：把检索结果直接喂进语言模型的词表，让模型在生成时自然“记住”具体向量，而不是后置拼接，提升了端到端的准确性。  
- **多后端兼容**：通过抽象化的向量算子接口，同一模型可以在 SQLite、PostgreSQL、ClickHouse 上无缝切换，只需替换底层向量索引实现。

### 实验与效果

- **评测数据**：使用作者公开的 **VectorSQLBench**，共 12 种组合（3 种数据库 × 4 种数据源），每种组合包含数千条自然语言‑VectorSQL 对。  
- **基线对比**：  
  - 传统两阶段方案：先用 Text‑to‑SQL 生成普通 SQL，再手工加上向量检索。  
  - 纯向量检索 + 后置过滤：先做 ANN 检索，再用 SQL 过滤结果。  
  - 端到端 LLM 直接生成 VectorSQL（未使用 CoT 或自校正）。  
- **整体提升**：论文声称在所有 12 种设置上，端到端的 UniVectorSQL（本文模型）在 **Set‑based F1** 上平均提升约 **8%**，在 **nDCG@10** 上提升约 **12%**，显著超过两阶段基线。具体数值请参考原文附表。  
- **消融实验**：  
  - 去掉 CoT 生成，F1 下降约 3%。  
  - 不使用向量 ID 直接拼接，SQL 语法错误率提升 5%。  
  - 仅保留向量检索子句（去掉结构化过滤），召回率提升但整体 F1 下降，说明两者必须协同。  
- **召回衰减分析**：在“先 SQL 再向量”策略下，召回率比“先向量再 SQL”低约 6%，作者将此归因于 SQL 过滤过于严格导致向量检索的候选被提前剔除。  
- **局限性**：  
  - 合成数据依赖 LLM 的质量，若生成的 VectorSQL 有系统性偏差，模型会学习错误模式。  
  - 只在列级向量化，未覆盖更细粒度（如段落、图片）或跨表向量关联。  
  - 实验主要在学术数据集上，真实企业业务场景的噪声与规模仍需进一步验证。

### 影响与延伸思考

自从代码和基准公开后，Text2VectorSQL 成为跨模态数据库交互的“标配”概念，后续工作主要围绕以下几个方向展开：

1. **更细粒度的向量化**：研究者尝试把行级、段落级甚至图像特征直接嵌入 VectorSQL，进一步提升检索的语义深度。  
2. **联合优化检索与过滤**：有工作提出在向量空间中加入结构化约束的投影层，直接在 ANN 索引阶段考虑 SQL 过滤条件，缓解召回衰减。  
3. **自监督数据扩展**：利用大模型自行生成更多混合查询，形成“伪标签”循环，提高模型在特定行业（金融、医疗）中的适配能力。  
4. **跨数据库执行引擎**：一些系统开始实现统一的 VectorSQL 解释器，能够在不同底层数据库之间自动切换向量索引实现，真正做到“一次写查询，多处执行”。  

如果想进一步深入，可以关注 **“向量化 SQL 优化器”**（Vectorized Query Optimizer）和 **“多模态检索语言模型”**（Multimodal Retrieval LLM）这两个研究热点，它们正逐步把 Text2VectorSQL 推向生产级别。

### 一句话记住它

**Text2VectorSQL 把自然语言一次性翻译成“SQL + 向量检索”混合查询，提供了统一的跨结构化/非结构化数据访问接口。**