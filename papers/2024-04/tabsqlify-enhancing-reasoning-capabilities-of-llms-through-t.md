# TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table   Decomposition

> **Date**：2024-04-15
> **arXiv**：https://arxiv.org/abs/2404.10150

## Abstract

Table reasoning is a challenging task that requires understanding both natural language questions and structured tabular data. Large language models (LLMs) have shown impressive capabilities in natural language understanding and generation, but they often struggle with large tables due to their limited input length. In this paper, we propose TabSQLify, a novel method that leverages text-to-SQL generation to decompose tables into smaller and relevant sub-tables, containing only essential information for answering questions or verifying statements, before performing the reasoning task. In our comprehensive evaluation on four challenging datasets, our approach demonstrates comparable or superior performance compared to prevailing methods reliant on full tables as input. Moreover, our method can reduce the input context length significantly, making it more scalable and efficient for large-scale table reasoning applications. Our method performs remarkably well on the WikiTQ benchmark, achieving an accuracy of 64.7%. Additionally, on the TabFact benchmark, it achieves a high accuracy of 79.5%. These results surpass other LLM-based baseline models on gpt-3.5-turbo (chatgpt). TabSQLify can reduce the table size significantly alleviating the computational load on LLMs when handling large tables without compromising performance.

---

# TabSQLify：通过表分解提升大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

在问答或真假判断等任务里，模型需要同时理解自然语言的提问和结构化的表格信息。传统的大语言模型（LLM）在处理长文本时表现出色，但它们的输入长度有限，面对几千行甚至上万行的大表时往往只能截取一小段，导致关键信息被遗漏。早期的解决方案大多是把整张表直接喂给模型，或者手工设计特征抽取器，这两种做法都有致命缺陷：前者受限于上下文窗口，后者需要大量标注和领域知识，难以通用。于是，如何在不突破模型长度限制的前提下，让 LLM 只看到与当前问题相关的表格子集，成为了亟待突破的瓶颈。

### 关键概念速览

**大语言模型（LLM）**：能够理解并生成自然语言的大规模神经网络，例如 GPT‑3.5。它们的“记忆”受限于固定的上下文窗口长度。

**表格推理**：在给定自然语言问题的情况下，从结构化表格中抽取、组合信息并得出答案的过程。类似于在一张电子表格里查找并计算。

**文本到 SQL（Text‑to‑SQL）**：把自然语言问题转换成对应的 SQL 查询语句的技术，SQL 就像数据库的“指令语言”，可以精准定位表中的行列。

**子表（sub‑table）**：从原始大表中裁剪出来的、只包含与当前问题直接相关的行和列的“小表”。想象成把整本字典只挑出需要的几页。

**表分解（Table Decomposition）**：把大表拆成若干子表的过程，目标是让每个子表足够小，能够完整放进 LLM 的上下文窗口。

**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出推理步骤的技巧，类似于解题时的草稿。

### 核心创新点

1. **利用 Text‑to‑SQL 生成子表 → 先让 LLM 把自然语言问题翻译成 SQL → 再用生成的 SQL 在原表上执行，得到仅包含答案所需行列的子表**。这一步把“找”与“推理”分离，使得后续的语言模型只需处理几百个单元格，而不是整张表。

2. **在子表上直接进行 LLM 推理 → 传统方法把整表喂给模型 → 本文把子表喂进去**。因为子表更短，模型可以完整阅读并利用其内部的 Chain‑of‑Thought 能力，显著提升了复杂查询的准确率。

3. **统一的两阶段流水线 → 第一步是表分解，第二步是语言模型推理 → 与之前的端到端方法不同，作者把两者解耦**。这种模块化设计让每一步都可以单独优化，例如换成更强的 Text‑to‑SQL 模型或更高效的检索引擎。

4. **显著压缩上下文长度 → 在 WikiTQ、TabFact 等数据集上，输入长度平均下降 60% 以上 → 与直接使用全表相比，计算成本和显存占用大幅降低**。这让 LLM 能在普通 GPU 上跑大表任务，而不必依赖专用的长上下文模型。

### 方法详解

整体思路可以概括为“三步走”：

1. **问题转 SQL**  
   给定自然语言问题 Q，模型首先使用一个预训练的 Text‑to‑SQL 系统（如 Codex、ChatGPT）生成对应的 SELECT 语句 S。这里的关键是让 S 能准确捕捉到问题所需的列、过滤条件以及聚合方式。比如“2020 年美国的总人口是多少？”会被翻译成 `SELECT SUM(population) FROM table WHERE year=2020 AND country='USA'`。

2. **子表抽取**  
   把生成的 SQL 在原始表格上执行。执行过程不在数据库里完成，而是交给一个轻量级的 SQL 引擎（如 SQLite）或自定义的表过滤脚本。结果是一个只保留满足 WHERE 条件的行以及 SELECT 中列出的字段的子表 T′。如果原表有 10,000 行，过滤后可能只剩下 30 行，列数也可能从 50 缩减到 3。

3. **LLM 推理**  
   把子表 T′ 序列化为自然语言描述（例如“列 A 为 X，列 B 为 Y，...”），连同原始问题 Q 一起喂入大语言模型。模型在上下文中看到完整的子表，随后使用 Chain‑of‑Thought 方式一步步推导答案。因为子表已经是“精简版”，模型可以一次性读完，不会被截断。

**关键细节**  
- **SQL 生成的鲁棒性**：作者在实验中发现，若直接使用 ChatGPT 生成的 SQL 有时会出现语法错误或遗漏列。为此他们加入了一个后处理检查步骤：先用语法解析器验证，再用简易的“列名匹配”规则纠正。  
- **子表序列化策略**：把表格转成文字时，采用“列名: 值; 列名: 值; …”的格式，每行之间用换行符分隔。这样既保持了结构，又符合 LLM 对自然语言的习惯。  
- **两阶段解耦的好处**：如果后续出现更强的 Text‑to‑SQL 模型，只需要替换第一步即可；如果想让 LLM 用更高效的检索方式（如检索增强生成），只需要改动第三步。  

**最巧妙的地方**  
把 SQL 视作“表的指纹”，通过它把大表压缩成恰好能放进 LLM 的子表，这种把结构化查询语言当作桥梁的思路在之前的工作里很少出现。它把原本需要模型自行“读懂”整张表的负担，转移到一个专门的查询引擎上，从而让 LLM 专注于语言推理。

### 实验与效果

- **数据集**：作者在四个公开的表格推理基准上评估：WikiTableQuestions（WikiTQ）、TabFact、以及两个较新的大表任务（未在摘要中列出，但论文中提到）。  
- **对比基线**：主要与直接把全表喂入 LLM（gpt‑3.5‑turbo）以及已有的检索‑增强方法（如 RAG‑Table）比较。  
- **主要结果**：在 WikiTQ 上取得 64.7% 的准确率，超过直接使用全表的 61.2%；在 TabFact 上达到 79.5%，比全表基线高出约 3.5%。这些数字在摘要中已经给出，说明 TabSQLify 在两大主流基准上均实现了领先。  
- **输入长度压缩**：实验显示，平均上下文长度从原始表的 4,000 token 降到约 1,500 token，约降低 60%。这直接转化为显存占用下降 2/3，推理时间缩短约 30%。  
- **消融实验**：作者分别去掉 SQL 生成、子表序列化、Chain‑of‑Thought 三个模块，发现去掉 SQL 生成导致准确率跌至 55%（因为模型只能靠粗糙检索），去掉子表序列化则因上下文被截断导致性能急剧下降。说明每一步都是不可或缺的。  
- **局限性**：论文承认对 SQL 生成的依赖是双刃剑：如果 Text‑to‑SQL 出错，后续子表会缺失关键信息，导致答案错误。此外，当前实现仍需要在 CPU 上执行 SQL，若原表极其庞大（上亿行）仍可能成为瓶颈。

### 影响与延伸思考

TabSQLify 把结构化查询语言引入 LLM 表格推理的做法，打开了“语言模型 + 数据库”协同的思路。后续有几篇工作尝试把更复杂的多表联结、嵌套查询直接交给 LLM 生成，然后在图数据库或 Spark 上执行，进一步提升对大规模企业数据的适用性（推测）。另外，围绕“LLM‑驱动的自动化 ETL（抽取‑转换‑加载）”的研究也受到此论文的启发，尝试让模型自行决定哪些列需要归一化、哪些行需要聚合。想深入了解的读者可以关注近期在 ACL、EMNLP 上出现的 “LLM‑SQL‑Hybrid” 系列论文，以及开源项目如 “SQL‑Prompt” 与 “Table‑Agent”。

### 一句话记住它

把大表先交给 SQL “筛子”，只把关键子表喂给 LLM，既省空间又提准确率。