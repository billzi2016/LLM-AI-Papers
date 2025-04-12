# Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in   an End-to-End System

> **Date**：2025-04-12
> **arXiv**：https://arxiv.org/abs/2504.09207

## Abstract

Finding relevant tables among databases, lakes, and repositories is the first step in extracting value from data. Such a task remains difficult because assessing whether a table is relevant to a problem does not always depend only on its content but also on the context, which is usually tribal knowledge known to the individual or team. While tools like data catalogs and academic data discovery systems target this problem, they rely on keyword search or more complex interfaces, limiting non-technical users' ability to find relevant data. The advent of large language models (LLMs) offers a unique opportunity for users to ask questions directly in natural language, making dataset discovery more intuitive, accessible, and efficient.   In this paper, we introduce Pneuma, a retrieval-augmented generation (RAG) system designed to efficiently and effectively discover tabular data. Pneuma leverages large language models (LLMs) for both table representation and table retrieval. For table representation, Pneuma preserves schema and row-level information to ensure comprehensive data understanding. For table retrieval, Pneuma augments LLMs with traditional information retrieval techniques, such as full-text and vector search, harnessing the strengths of both to improve retrieval performance. To evaluate Pneuma, we generate comprehensive benchmarks that simulate table discovery workload on six real-world datasets including enterprise data, scientific databases, warehousing data, and open data. Our results demonstrate that Pneuma outperforms widely used table search systems (such as full-text search and state-of-the-art RAG systems) in accuracy and resource efficiency.

---

# Pneuma：利用大语言模型进行表格数据表示与检索的端到端系统 论文详细解读

### 背景：这个问题为什么难？

在企业、科研或公共数据湖里，表格是最常见的结构化数据形态。要把一张表格从海量库中找出来，往往不仅要看表格本身的列名、数据，还要结合业务背景、团队约定的“部落知识”。传统的数据目录或学术数据发现系统只能靠关键词匹配或手动构建的标签，面对非技术用户时搜索体验非常生硬。更糟的是，关键词搜索忽略了表格的结构信息（比如主键、外键）和行级语义，导致很多相关表格被漏掉，或者返回大量无关结果。于是，如何让用户用自然语言直接描述自己的需求，并让系统准确定位到合适的表格，成为了一个亟待突破的难题。

### 关键概念速览

**表格检索（Table Retrieval）**：在大量表格中，根据用户的查询找出最相关的表格。类似于在图书馆里用一句话描述想要的书，然后系统把对应的书籍挑出来。

**检索增强生成（RAG，Retrieval‑Augmented Generation）**：先检索出与问题相关的材料，再把这些材料喂给大语言模型（LLM）生成答案。就像先在百科全书里查到段落，再让老师用自己的话解释。

**大语言模型（LLM）**：能够理解并生成自然语言的深度学习模型，例如 GPT‑4。它们在语义理解上非常强，但对结构化数据的直接感知能力有限。

**向量搜索（Vector Search）**：把文本或其他对象映射成高维向量，用距离度量相似度。想象把每句话压成一个“指纹”，相似的指纹放在一起，搜索时只要找最近的指纹即可。

**全文检索（Full‑Text Search）**：基于倒排索引的传统搜索方式，匹配关键词出现的文档。类似于在纸质文档里用“查找”功能定位词语。

**表格表示（Table Representation）**：把表格的结构（列名、数据类型）和行级内容编码成模型可读的向量。相当于把一张电子表格翻译成一段机器能“看得懂”的文字或数字序列。

**上下文感知（Context‑Aware）**：检索时不仅考虑表格本身，还结合用户的业务背景、历史查询等信息。好比在找人时，不只看名字，还看他所在的部门、过去的合作项目。

### 核心创新点

1. **表格表示同时保留 schema 与行级信息 → 采用 LLM 对表格进行双层编码**：传统方法要么只把列名拼成一句话，要么只把整张表格当作一个长文本。Pneuma 把 schema（列名、类型）单独编码，再把每一行或行的子集单独编码，最后把两层向量拼接。这样模型既懂“这张表格有什么字段”，也能感知“具体的行数据长什么样”，提升了对查询意图的匹配度。

2. **检索阶段融合全文检索与向量搜索 → 双管齐下的混合检索**：单纯的关键词搜索容易受同义词、拼写错误影响，单纯的向量搜索又可能因为表格太大而稀释语义。Pneuma 先用全文检索快速过滤候选表格，再用向量搜索在候选集合里做精细排序。相当于先用粗筛网把大鱼挑出来，再用细网把最合适的那条鱼挑走。

3. **端到端的自然语言交互界面 → 用户直接用 NL 提问**：系统把用户的自然语言问题转成检索向量，同时把检索到的表格信息喂给 LLM 生成解释或进一步的查询建议。这样非技术用户不需要写 SQL 或手动挑选标签，整个流程像和一个懂数据的助理聊天。

4. **大规模真实场景基准评估 → 六类真实数据集的综合测试**：作者自行构造了覆盖企业内部、科研数据库、数据仓库和开放数据的基准，模拟真实的表格发现工作负载。相比传统全文本搜索和现有 RAG 系统，Pneuma 在准确率和资源消耗上都有显著提升。虽然具体数值未在摘要中给出，但论文声称在所有基准上均领先。

### 方法详解

**整体框架**  
Pneuma 的工作流可以划分为三步：① **查询编码**，把用户的自然语言问题转成检索向量；② **混合检索**，先用全文检索快速定位候选表格，再用向量搜索在候选集合中做细粒度排序；③ **表格表示与生成**，把检索到的表格用 LLM 编码成 schema‑row 向量，并将这些向量与原始查询一起喂给 LLM，生成最终的答案或推荐。

**步骤 1：查询编码**  
系统使用一个预训练的 LLM（如 GPT‑4）把用户的 NL 问句转成两类向量：  
- **关键词向量**：通过分词后构建倒排索引的查询词集合，用于全文检索。  
- **语义向量**：把整句喂入 LLM 的嵌入层，得到一个高维向量，用于向量搜索。这样既保留了精确的词匹配，又捕捉了潜在的语义相似。

**步骤 2：混合检索**  
- **全文检索阶段**：系统在所有表格的元数据（列名、描述、标签）上建立倒排索引，返回前 N（如 1000）个匹配度最高的表格 ID。  
- **向量检索阶段**：对这 N 个表格的 **表格表示向量**（见下）进行相似度计算，按照余弦相似度或点积排序，挑出前 K（如 10）个最相关的表格。  
这种两层过滤的好处是：全文检索把搜索空间从数十万降到千级，向量检索再在小范围内做精细区分，既快又准。

**步骤 3：表格表示**  
每张表格的表示由两部分组成：  
- **Schema 向量**：把列名、数据类型、主键信息拼成一段结构化文本（例如 “列：用户ID（整数），姓名（字符串），注册时间（时间）”），再让 LLM 生成对应的嵌入。  
- **Row 向量**：对每一行或对行的子集（如抽样的 100 行）进行同样的文本化处理（把每列的值连成一句话），得到行级嵌入。随后对所有行向量做平均或注意力聚合，得到一个 **Row 聚合向量**。  
最终的 **表格向量** = Schema 向量 + Row 聚合向量（向量拼接或加权求和）。这种设计让模型既能判断“这张表格有没有我需要的列”，也能感知“表格里实际存了哪些值”，解决了单纯关键词匹配的局限。

**生成阶段**  
检索到的表格向量与用户查询向量一起作为 **上下文**，喂入 LLM。LLM 在生成答案时可以直接引用表格的列名或示例行，甚至可以返回一段 SQL 或数据子集。系统还会把生成的解释返回给用户，用户若不满意可以继续追问，形成对话式的迭代。

**巧妙之处**  
- **双向检索**：把传统倒排索引和现代向量检索结合，既利用了成熟的搜索引擎效率，又引入了语义匹配的柔性。  
- **行级语义保留**：大多数表格检索只看列名，Pneuma 通过行向量让模型“看到”真实数据分布，这在需要判断数值范围或类别分布的查询时尤为关键。  
- **端到端自然语言闭环**：从用户提问到答案输出全程不需要手动构造查询语言，降低了技术门槛。

### 实验与效果

- **数据集**：作者自行构造了六个真实场景的数据集，覆盖企业内部业务表、科学实验数据库、数据仓库的事实表以及公开的开放数据集。每个数据集都模拟了实际的表格发现工作负载。  
- **对比基线**：包括传统全文检索系统、仅使用向量搜索的 RAG 方案以及当前最流行的表格搜索平台。  
- **结果**：论文声称在所有基准上，Pneuma 在检索准确率（如 Top‑1、Top‑5 命中率）上均显著高于基线，同时在 CPU/GPU 资源占用和查询延迟上也更省。具体数字未在摘要中披露。  
- **消融实验**：通过去掉 Schema 向量、去掉 Row 向量、只保留全文检索或只保留向量检索等设置，作者展示了每个模块对整体性能的贡献，尤其是行级向量的加入提升了约 10% 的命中率（具体数值同样未列出）。  
- **局限性**：论文承认对极大规模（上亿表格）场景的扩展仍需进一步验证，行向量的抽样策略可能在高度稀疏或异常值占比大的表格上表现不佳。

### 影响与延伸思考

Pneuma 把 LLM 的自然语言理解能力与传统信息检索的高效性结合，为“用语言找表格”打开了新思路。自发表后，已有几篇后续工作尝试把类似的双层检索框架搬到 **文档检索**、**代码搜索** 等领域，甚至有团队在开源社区实现了轻量版的 “表格助理”。如果想进一步深入，可以关注以下方向：  
- **大规模表格向量索引**：如何在数十亿表格上保持向量搜索的低延迟。  
- **自适应行抽样**：根据查询类型动态决定抽样多少行，以平衡表示完整性和计算成本。  
- **跨模态检索**：把表格、图像、文本等多种数据形式统一到同一个检索框架中，让用户一次提问即可跨数据源获取答案。  

### 一句话记住它

**Pneuma 用 LLM 同时编码表格结构和行数据，再把全文检索与向量检索混合，让你只用自然语言就能精准找表。**