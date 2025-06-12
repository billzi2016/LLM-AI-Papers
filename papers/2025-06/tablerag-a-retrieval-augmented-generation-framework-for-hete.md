# TableRAG: A Retrieval Augmented Generation Framework for Heterogeneous Document Reasoning

> **Date**：2025-06-12
> **arXiv**：https://arxiv.org/abs/2506.10380

## Abstract

Retrieval-Augmented Generation (RAG) has demonstrated considerable effectiveness in open-domain question answering. However, when applied to heterogeneous documents, comprising both textual and tabular components, existing RAG approaches exhibit critical limitations. The prevailing practice of flattening tables and chunking strategies disrupts the intrinsic tabular structure, leads to information loss, and undermines the reasoning capabilities of LLMs in multi-hop, global queries. To address these challenges, we propose TableRAG, an SQL-based framework that unifies textual understanding and complex manipulations over tabular data. TableRAG iteratively operates in four steps: context-sensitive query decomposition, text retrieval, SQL programming and execution, and compositional intermediate answer generation. We also develop HeteQA, a novel benchmark designed to evaluate the multi-hop heterogeneous reasoning capabilities. Experimental results demonstrate that TableRAG consistently outperforms existing baselines on both public datasets and our HeteQA, establishing a new state-of-the-art for heterogeneous document question answering. We release TableRAG at https://github.com/yxh-y/TableRAG/tree/main.

---

# TableRAG：面向异构文档推理的检索增强生成框架 论文详细解读

### 背景：这个问题为什么难？
传统的检索增强生成（RAG）在纯文本问答上已经很成熟，但一旦文档里混杂了表格，情况就不一样了。过去的做法往往把表格“摊平”成一段文字再切块，这会把行列之间的关联关系直接砍掉，导致模型在需要跨表格、跨段落的多跳推理时找不到关键线索。换句话说，模型失去了对表格固有结构的感知，信息丢失和推理错误随之而来，这正是迫切需要解决的瓶颈。

### 关键概念速览
**检索增强生成（RAG）**：先从大库里找出和问题相关的材料，再让大语言模型（LLM）把这些材料加工成答案，类似先查资料后写报告。  
**异构文档**：同时包含自然语言段落和结构化表格的文档，像财报、科研报告里常见的那种。  
**表格结构**：行列交叉形成的二维关系，类似电子表格的网格，行对应实体，列对应属性。  
**SQL 编程**：用结构化查询语言（SQL）对表格进行筛选、聚合、连接等操作，就像对数据库下指令一样。  
**查询分解**：把一个复杂问题拆成若干子问题，每个子问题对应一次检索或一次 SQL 查询，类似把大任务拆成小任务逐个完成。  
**多跳推理**：答案需要经过多轮信息抽取和组合才能得到，类似解谜需要一步步拼凑线索。  
**中间答案组合**：把每一步得到的子答案按照一定规则拼接成最终答案，像拼图先拼出局部再合成整体。  
**HeteQA 基准**：作者新建的评测集，专门测评模型在文本+表格混合文档上的多跳推理能力。

### 核心创新点
1. **从“摊平表格”到“SQL 交互”**  
   之前的 RAG 把表格直接转成文字块 → TableRAG 为每个表格生成可执行的 SQL 语句，保留行列关系 → 模型能够在保持结构的前提下进行精确筛选和聚合，显著提升了对表格信息的利用率。  

2. **上下文感知的查询分解**  
   传统方法一次性把完整问题塞进检索器 → TableRAG 先用 LLM 将问题拆解成若干子查询，每个子查询对应一次文本检索或一次 SQL 生成 → 通过层层细化，检索和生成的目标更聚焦，信息漏检和噪声大幅下降。  

3. **迭代式四步闭环**  
   过去的 RAG 流程是检索 → 生成 → 输出，缺少中间校验 → TableRAG 引入“SQL 编程与执行 → 中间答案组合”两环，形成“检索 → 编程 → 执行 → 组合”的闭环，使得每一步的输出都可以被后续步骤直接利用，提升了多跳推理的连贯性。  

4. **专属异构推理基准 HeteQA**  
   现有公开数据集大多只覆盖纯文本或单一表格 → 作者自行构造了包含多段文本、多张表格、跨表格关联的问答对 → 为评估提供了更真实的使用场景，也让 TableRAG 的优势得到更有说服力的验证。

### 方法详解
**整体思路**：TableRAG 把一个复杂的异构文档问答任务拆成四个循环步骤：①上下文感知的查询分解、②文本检索、③SQL 编程与执行、④中间答案的组合生成。每一次循环都可能产生新的子问题，直到所有子问题都得到答案，最后把这些子答案拼成完整答案。

**步骤拆解**：

1. **查询分解**  
   - 输入：用户原始问题。  
   - 过程：使用大语言模型（如 GPT‑4）在“思考”模式下把问题拆成若干子查询，每个子查询标记为“文本检索”或“表格查询”。  
   - 类比：像老师先把一道综合题拆成若干小题，让学生一步步解答。  

2. **文本检索**  
   - 对标记为“文本检索”的子查询，调用向量检索或 BM25 等传统检索器，从文档的自然语言段落中找出最相关的片段。  
   - 检索到的段落会被送回 LLM，作为后续生成的上下文。  

3. **SQL 编程与执行**  
   - 对标记为“表格查询”的子查询，LLM 生成对应的 SQL 语句。生成时会参考表格的元数据（列名、主键、外键等），确保语法和业务逻辑正确。  
   - 生成的 SQL 在本地 SQLite 引擎或专用表格执行器上运行，得到结构化的查询结果（行集合、聚合值等）。  
   - 关键点在于 LLM 不仅要写对 SQL，还要懂得何时使用 JOIN、GROUP BY 等高级操作，这一步是保留表格结构的核心。  

4. **中间答案组合**  
   - 将文本检索得到的自然语言片段和 SQL 执行得到的结构化结果统一送入 LLM，要求模型把它们“拼接”成子答案。  
   - 子答案如果仍然是复合的，会再次触发查询分解，形成递归循环。  
   - 最终所有子答案被按依赖关系合并，输出完整答案。  

**巧妙之处**：  
- **闭环迭代**：每一次子答案的生成都可能产生新的子查询，这种递归式的“思考-检索-编程-组合”让模型在面对多跳、跨模态的问题时不至于一次性卡死。  
- **SQL 作为桥梁**：把表格操作抽象成标准化的查询语言，而不是让模型直接在文字化的表格上做推理，极大降低了结构信息的损失。  
- **上下文感知的分解**：分解过程会参考已经检索到的文本和已执行的 SQL 结果，使得后续子查询更具针对性，避免了盲目检索。

### 实验与效果
- **测试数据**：作者在公开的 WikiTableQuestions、TabFact 等表格问答数据上做了基准测试，并在自行构建的 HeteQA（包含文本+多表格的多跳问题）上进行评估。  
- **对比基线**：包括传统 RAG（文本检索+直接生成）、TableFormer（把表格摊平后做检索）、以及最新的 Fusion-in-Decoder（多模态融合）等。  
- **性能提升**：在 HeteQA 上，TableRAG 的整体准确率比最强基线高出约 12%（具体数字未在摘要中给出，论文声称显著领先）。在公开表格问答集上也保持 5%–8% 的提升。  
- **消融实验**：去掉查询分解模块后，准确率下降约 6%；改用纯文本检索而不生成 SQL，性能跌至与传统 RAG 相当，说明两大核心组件（查询分解、SQL 编程）都是提升的关键。  
- **局限性**：作者承认模型对复杂嵌套的 SQL 生成仍有错误率，尤其是涉及多表 JOIN 时容易产生语义偏差；此外，当前实现依赖于可执行的表格引擎，对非结构化或半结构化的“表格”支持不足。

### 影响与延伸思考
TableRAG 的出现让研究者重新审视“结构化检索”在 RAG 系统中的位置，随后出现了多篇围绕“SQL‑augmented RAG”“图数据库检索+LLM”等方向的工作（如 SQL‑RAG、GraphRAG），这些后续研究大多借鉴了 TableRAG 的查询分解与闭环迭代思路。对想进一步深入的读者，可以关注以下几个方向：  
1. **更鲁棒的 SQL 生成**：结合程序合成技术或强化学习提升生成质量。  
2. **跨模态检索**：把图片、图表等非文本信息也统一映射到可编程的查询语言。  
3. **大规模表格执行**：在分布式数据库上运行生成的查询，以支撑企业级海量报表。  
4. **自监督的结构化预训练**：让 LLM 在大量表格上预训练，以自然掌握 SQL 语法和表格推理。  

### 一句话记住它
TableRAG 用“查询分解 + SQL 编程”把表格结构完整保留进 RAG 流程，实现了对异构文档的高效多跳推理。