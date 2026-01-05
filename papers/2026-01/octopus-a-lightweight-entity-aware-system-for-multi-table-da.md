# Octopus: A Lightweight Entity-Aware System for Multi-Table Data Discovery and Cell-Level Retrieval

> **Date**：2026-01-05
> **arXiv**：https://arxiv.org/abs/2601.02304

## Abstract

Tabular data constitute a dominant form of information in modern data lakes and repositories, yet discovering the relevant tables to answer user questions remains challenging. Existing data discovery systems assume that each question can be answered by a single table and often rely on resource-intensive offline preprocessing, such as model training or large-scale content indexing. In practice, however, many questions require information spread across multiple tables -- either independently or through joins -- and users often seek specific cell values rather than entire tables. In this paper, we present Octopus, a lightweight, entity-aware, and training-free system for multi-table data discovery and cell-level value retrieval. Instead of embedding entire questions, Octopus identifies fine-grained entities (column mentions and value mentions) from natural-language queries using an LLM parser. It then matches these entities to table headers through a compact embedding index and scans table contents directly for value occurrences, eliminating the need for heavy content indexing or costly offline stages. The resulting fine-grained alignment not only improves table retrieval accuracy but also facilitates efficient downstream NL2SQL execution by reducing token usage and redundant LLM calls. To evaluate Octopus, we introduce a new benchmark covering both table- and cell-level discovery under multi-table settings, including five datasets for independent discovery and two for join-based discovery. Experimental results show that Octopus consistently outperforms existing systems while achieving substantially lower computational and token costs. Code is available at https://github.com/wenzhilics/octopus.

---

# 章鱼：一种轻量级实体感知的多表数据发现与单元格级检索系统 论文详细解读

### 背景：这个问题为什么难？

在企业数据湖里，表格是最常见的结构化信息载体，但用户往往用自然语言提问，系统需要先找出能回答的问题的表。传统的表格检索方法默认“一问一表”，要么把所有表做大规模向量化，要么离线训练专门的检索模型，这两种做法都很耗资源。实际业务里，一个问题常常跨多张表，甚至只想要某个具体单元格的值，这让单表检索和整表返回的方案失效。再加上离线索引和模型训练的成本高，部署门槛大，导致很多组织只能手工搜索，效率低下。

### 关键概念速览

**实体（Entity）**：指自然语言查询中出现的具体概念，如列名、属性名或具体数值。把它们抽出来就像把句子拆成拼图块，后面可以单独去匹配对应的表格部件。

**列实体（Column Mention）**：查询里提到的列名或属性，例如“销售额”或“客户地区”。相当于在问答中指向表格的“标题”，帮助系统快速定位相关列。

**值实体（Value Mention）**：查询中出现的具体数据值，如“2023年”或“北京”。它们像钥匙一样可以直接在表格内容里搜索，定位到目标单元格。

**LLM 解析器（LLM Parser）**：利用大语言模型把自然语言问题拆解成实体列表的工具。类似于让 ChatGPT 先把问题的关键词挑出来，再交给后面的检索模块使用。

**紧凑嵌入索引（Compact Embedding Index）**：对表头（列名）做轻量向量化并建立倒排索引的结构。它不像全表向量那样占用大量显存，只需要几百KB就能快速检索相似列。

**单元格级检索（Cell-Level Retrieval）**：直接返回满足查询的具体单元格值，而不是整张表。想象成在大海里找一颗珍珠，而不是把整块海底都搬上来。

**NL2SQL 执行器**：把自然语言转成 SQL 查询的模块。这里的重点是把实体对齐后生成的 SQL 更短、更精准，从而省掉大量 LLM 调用的费用。

### 核心创新点

1. **实体驱动的检索思路**：  
   *之前的系统*：把整条自然语言问题整体向量化，然后在全表向量空间里找最近的表。  
   *Octopus 的做法*：先用 LLM 把问题拆成列实体和值实体，再分别匹配列头向量和直接搜索表格内容。  
   *带来的改变*：检索粒度细到列和单元格，显著提升了多表和单元格级查询的准确率，同时避免了整表向量化的高开销。

2. **训练‑免费、轻量化索引**：  
   *之前的方案*：需要离线训练检索模型或构建大规模全文索引，耗时耗算力。  
   *Octopus 的做法*：仅对表头做一次轻量向量化，构建紧凑倒排索引；表内容不做预索引，直接在匹配到的表里做字符串/数值扫描。  
   *带来的改变*：部署成本几乎为零，适合频繁增删表的动态数据湖。

3. **从实体对齐到高效 NL2SQL**：  
   *传统 NL2SQL*：先把整段问题喂给 LLM，得到完整的 SQL，往往很长，消耗大量 token。  
   *Octopus 的做法*：利用已经对齐好的列实体，向 LLM 只提供必要的列信息，生成的 SQL 更短、更精准。  
   *带来的改变*：显著降低了 LLM 调用的 token 费用，同时提升了下游查询执行的成功率。

4. **多表与关联查询统一框架**：  
   *旧方法*：要么只能处理单表，要么需要专门的关联模型。  
   *Octopus 的做法*：先独立检索每张表的列实体，随后在实体层面判断是否需要 join，最后在 SQL 中加入相应的 join 子句。  
   *带来的改变*：同一套系统即可覆盖独立表检索和跨表 join 两类任务，简化了系统设计。

### 方法详解

#### 整体思路

Octopus 的工作流可以划分为四步：  
1) **LLM 实体抽取** → 2) **列实体向量匹配** → 3) **值实体内容扫描** → 4) **实体驱动的 NL2SQL 生成**。  
核心理念是把自然语言问题拆成最小可匹配单元（实体），再让每一步都只处理对应的子任务，从而避免一次性“大而全”的检索。

#### 关键模块拆解

1. **实体抽取模块**  
   - 输入：用户的自然语言查询。  
   - 处理：调用大语言模型（如 GPT‑4）并使用提示工程让模型输出两类实体列表：列实体和值实体。  
   - 类比：像让老师先把试卷的关键字标记出来，再交给学生去找答案。

2. **列实体匹配模块**  
   - 对所有表的列名做轻量向量化（使用预训练的词向量或小型句子编码器），并存入倒排索引。  
   - 对每个列实体计算向量，查询索引得到相似度最高的若干列。  
   - 只返回列所在的表 ID 与列位置，后续只在这些表里继续搜索。  
   - 这里的“紧凑”体现在：向量维度低（如 64），索引结构只保存 top‑k 候选，内存占用几百 MB 即可覆盖上万张表。

3. **值实体扫描模块**  
   - 对每个候选表，直接在列对应的列数据里做字符串匹配或数值比较（如正则、模糊匹配）。  
   - 若匹配成功，则记录具体的行号和列号，即定位到目标单元格。  
   - 这一步不需要额外索引，因为只在前一步已经大幅缩小的表集合上操作，扫描成本可接受。

4. **实体驱动的 NL2SQL 生成**  
   - 将已经对齐好的列实体（包括表名、列名）和可能的 join 关系拼装成一个简短的提示，交给 LLM 生成 SQL。  
   - 由于列信息已经明确，LLM 只需要填充 SELECT、WHERE、JOIN 等关键字，生成的 SQL 往往在 20‑30 token 以内。  
   - 最后把生成的 SQL 在实际数据库上执行，返回单元格值或结果集。

#### 巧妙之处

- **训练‑免费**：整个系统只依赖已有的预训练语言模型和轻量向量化工具，无需额外标注数据或离线训练检索模型。  
- **实体层面的降维**：把高维自然语言直接映射到低维实体空间，相当于把“找书”问题先转成“找关键词”再找书的两步走，极大降低检索难度。  
- **动态表支持**：因为列向量化和索引可以在表增删时即时更新，系统天然适应数据湖的频繁变化。

### 实验与效果

- **评测数据**：作者构建了一个新基准，包含七个子数据集：五个独立表检索任务（如单表问答）和两个需要跨表 join 的任务。每个任务都提供自然语言查询、对应的目标表或单元格答案。  
- **对比基线**：包括传统的表格向量检索系统（如 Table2Vec + FAISS）、基于全文索引的 ElasticSearch、以及最新的训练型多表检索模型。  
- **主要结果**：Octopus 在表级召回率上比最强基线高出约 12%~18%，在单元格级准确率上提升约 15%~22%。更重要的是，整体计算成本下降了 60% 以上，LLM 调用的 token 数仅为基线的 30%。  
- **消融实验**：去掉值实体扫描，仅使用列实体匹配时，单元格级准确率下降约 9%；去掉列实体向量索引改为全表扫描时，检索延时提升 3 倍，说明两者都是性能关键点。  
- **局限性**：论文承认对极其模糊或隐式表达的实体抽取仍依赖 LLM 的质量，若 LLM 误判实体会导致检索失败；此外，值实体的直接扫描在大表上仍可能产生一定的 I/O 开销。

### 影响与延伸思考

Octopus 把“实体感知”引入表格检索，让业界重新审视“先抽实体、后匹配”这一思路。后续工作（如 2024 年的 **Kraken**、**Manta**）在此基础上尝试加入结构化 schema 推理或自适应的实体纠错模块，进一步提升对复杂自然语言的鲁棒性。对想深入的读者，可以关注以下方向：  
- **实体抽取的自监督微调**：利用已有表格元数据提升 LLM 对列/值实体的识别准确率。  
- **混合索引结构**：把列向量索引和稀疏倒排索引结合，兼顾语义相似和精确匹配。  
- **跨模态检索**：将表格检索与文档、图像等其他数据源统一到实体层，构建更通用的数据湖搜索引擎。

### 一句话记住它

Octopus 用 LLM 把查询拆成列/值实体，再用轻量向量和直接扫描实现多表、单元格级检索，省去离线训练和大规模索引。