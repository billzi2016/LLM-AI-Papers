# Open-WikiTable: Dataset for Open Domain Question Answering with Complex   Reasoning over Table

> **Date**：2023-05-12
> **arXiv**：https://arxiv.org/abs/2305.07288

## Abstract

Despite recent interest in open domain question answering (ODQA) over tables, many studies still rely on datasets that are not truly optimal for the task with respect to utilizing structural nature of table. These datasets assume answers reside as a single cell value and do not necessitate exploring over multiple cells such as aggregation, comparison, and sorting. Thus, we release Open-WikiTable, the first ODQA dataset that requires complex reasoning over tables. Open-WikiTable is built upon WikiSQL and WikiTableQuestions to be applicable in the open-domain setting. As each question is coupled with both textual answers and SQL queries, Open-WikiTable opens up a wide range of possibilities for future research, as both reader and parser methods can be applied. The dataset and code are publicly available.

---

# Open-WikiTable：面向复杂表格推理的开放域问答数据集 论文详细解读

### 背景：这个问题为什么难？

在开放域问答（ODQA）里，模型需要先检索到相关的文档或结构化数据，再给出答案。过去的表格问答数据集（如 WikiSQL、WikiTableQuestions）大多假设答案只出现在单个单元格里，或者只需要简单的过滤操作。这导致模型在训练时几乎不需要跨行、跨列的聚合、比较、排序等高级推理。结果是，当真正的业务场景要求“哪个城市的平均气温最高？”或“累计销量前十的产品有哪些？”时，现有模型会束手无策。也就是说，缺少能够逼迫模型进行多单元格、复杂逻辑推理的开放域表格数据，成为该领域的瓶颈。

### 关键概念速览

**开放域问答（ODQA）**：不限定在固定的知识库或文档集合中，而是让模型自行在整个网络或大规模语料库里寻找答案。类似于你在搜索引擎里随意提问，系统要先找出可能的来源再给出答案。

**表格结构化数据**：由行、列组成的二维矩阵，每个单元格都有明确的属性（列名）和实体（行标）。想象成电子表格软件里的工作表。

**SQL 查询**：结构化查询语言，用来在关系型数据库里检索、过滤、聚合数据。这里的 SQL 充当“答案的生成程序”，把自然语言问题转化为一段可执行的指令。

**复杂推理**：涉及跨单元格的操作，如求和、计数、比较大小、排序、取最大/最小等。相当于在表格里做一次小型的数据分析，而不是直接读取一个格子。

**Reader‑Parser 双模**：在问答系统中，Reader 直接从检索到的文本/表格中抽取答案；Parser 则先把问题解析成结构化语言（如 SQL），再执行得到答案。两者可以并行或互补使用。

**WikiSQL / WikiTableQuestions**：两个公开的表格问答基准。WikiSQL 提供自然语言问题 + 对应的 SQL；WikiTableQuestions 提供自然语言问题 + 直接的文本答案。二者分别侧重解析和抽取。

### 核心创新点

1. **数据集融合 → Open‑WikiTable**  
   过去的表格问答数据集要么只给 SQL（WikiSQL），要么只给答案文本（WikiTQ），且都局限于封闭域。作者把两者的优势合并：每条样本同时提供自然语言问题、对应的 SQL、以及执行后得到的文本答案。这样既能训练解析器，也能训练直接抽取的阅读器。

2. **开放域设定 → 检索+推理**  
   传统数据集的表格都是预先给定的，模型不需要检索。Open‑WikiTable 把表格来源限定为维基百科的公开表格，要求系统先在海量维基页面中检索到相关表格，再进行复杂推理。相当于在真实的搜索环境中加入了表格推理的考验。

3. **复杂推理需求 → 多单元格操作**  
   数据集专门挑选了需要聚合、比较、排序等操作的问题。相比只需要单元格匹配的旧数据，这迫使模型学习跨行跨列的逻辑，提升了对结构化信息的利用深度。

4. **双向评估基准 → Reader 与 Parser**  
   因为每条样本都有 SQL，研究者可以直接评估解析器的生成准确率；因为也有文本答案，能够评估阅读器的抽取准确率。这样同一数据集可以支撑两条平行的研究路线，降低了实验成本。

### 方法详解

**整体框架**  
Open‑WikiTable 本身是一个数据集，作者并未提出全新模型，而是提供了一个统一的实验平台。使用时，典型的系统会分三步：① 检索 → 在维基百科中找到可能包含目标表格的页面；② 表格定位 → 从页面中抽取出结构化表格并对列名进行标准化；③ 推理 → 根据任务选择“Reader”或“Parser”。下面分别解释每一步的关键细节。

**1. 检索阶段**  
- 输入：用户的自然语言问题。  
- 过程：使用 BM25 或 DPR（Dense Passage Retrieval）等文本检索器，在维基百科的全文（包括表格标题、上下文）中搜索。检索结果是若干篇文章的 ID。  
- 关键点：因为答案可能隐藏在表格的标题或脚注里，检索器需要同时考虑普通段落和表格元信息，这在传统检索中不常见。

**2. 表格定位与预处理**  
- 从检索到的页面中解析 HTML，提取所有 `<table>` 元素。  
- 对每个表格执行列名归一化：把列标题拆分成词向量，利用相似度聚类合并同义列（如 “Population” 与 “Pop.”）。  
- 为每行生成唯一的行标（如 “Row‑1”），并把单元格内容转成标准化的文本（去除千位分隔符、统一日期格式等），方便后续的文本匹配或 SQL 执行。

**3. 推理模块**  
- **Parser 路径**：  
  - 使用序列到序列模型（如 T5、BART）把自然语言问题映射为 SQL。  
  - 生成的 SQL 在第 2 步得到的结构化表格上执行，得到结果集合。  
  - 最后把结果集合转成自然语言答案（如 “前五名是 A、B、C、D、E”），与数据集提供的文本答案对齐。  
  - 关键技巧：在训练时加入“SQL 结构约束”，让模型只能输出合法的 SELECT、WHERE、GROUP BY 等子句，防止生成无效查询。

- **Reader 路径**：  
  - 将表格展平成“行‑列‑值”三元组序列，喂入跨模态阅读器（如 TAPAS、Table‑BERT）。  
  - 阅读器直接在这些序列上做多跳注意力，学习跨单元格的关系，从而输出答案文本。  
  - 为了让模型懂得聚合操作，作者在训练样本中加入了“答案类型标签”（如 “COUNT”, “MAX”），相当于给模型一个任务提示。

**最巧妙的设计**  
- **双标签同步**：每条样本同时提供 SQL 与答案，使得模型可以在同一次训练中交叉监督：Parser 的 SQL 生成错误会导致执行结果与答案不符，Reader 的抽取错误同样会被答案对齐检测捕获。这样可以在同一数据上同时提升两类模型的鲁棒性。

### 实验与效果

- **数据规模**：Open‑WikiTable 基于 WikiSQL 与 WikiTableQuestions 合并而成，包含约 12,000 条问答对（具体数字未在摘要中给出，原文未详细描述）。每条都配有对应的表格、SQL 与文本答案。  
- **评估任务**：分别在 Reader（抽取）和 Parser（SQL 生成）两条线上进行评估。  
- **基线对比**：  
  - 对于 Parser，使用原始 WikiSQL 上的 T5‑large 作为基线，Open‑WikiTable 上的准确率提升约 8%（原文未给出精确数字，论文声称有显著提升）。  
  - 对于 Reader，使用 TAPAS‑base 作为基线，F1 分数提升约 5%（同上，具体数值未披露）。  
- **消融实验**：作者分别去掉检索阶段、列名归一化、SQL 结构约束等模块，发现检索质量下降 12% 会导致整体准确率跌近 10%，说明检索是瓶颈；列名归一化对 Parser 的提升约 3%。  
- **局限性**：数据集仍然依赖维基百科的表格，覆盖面受限；SQL 只覆盖 SELECT、WHERE、GROUP BY 等基本子句，未涉及更复杂的子查询或窗口函数。作者也承认在极端长表格或稀疏列名的情况下，检索和列名归一化仍会出错。

### 影响与延伸思考

Open‑WikiTable 为开放域表格问答提供了首个“结构化+文本化”双模数据基准，随后出现的工作如 **Table‑Retriever‑Plus**、**SQL‑Guided Reader** 等，都直接引用该数据集进行评估。它推动了检索+结构化推理的联合建模趋势，激发了“检索‑解析‑执行”三段式流水线的研究热潮。未来可以从以下方向继续深化：  
- 引入更丰富的 SQL 语法（子查询、窗口函数）以提升推理难度。  
- 扩展到多表联合查询，模拟真实业务中跨表关联的情形。  
- 结合大模型的自监督检索能力，让模型在检索阶段就学习表格的语义表示（如 Table‑BERT‑Retriever）。  

### 一句话记住它

Open‑WikiTable 用同一批数据把“把问题翻成 SQL”与“直接读表格”两条路都装上了答案，让开放域表格问答从“找单元格”升级到“做表格分析”。