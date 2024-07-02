# TCSR-SQL: Towards Table Content-aware Text-to-SQL with Self-retrieval

> **Date**：2024-07-01
> **arXiv**：https://arxiv.org/abs/2407.01183

## Abstract

Large Language Model-based (LLM-based) Text-to-SQL methods have achieved important progress in generating SQL queries for real-world applications. When confronted with table content-aware questions in real-world scenarios, ambiguous data content keywords and nonexistent database schema column names within the question lead to the poor performance of existing methods. To solve this problem, we propose a novel approach towards Table Content-aware Text-to-SQL with Self-Retrieval (TCSR-SQL). It leverages LLM's in-context learning capability to extract data content keywords within the question and infer possible related database schema, which is used to generate Seed SQL to fuzz search databases. The search results are further used to confirm the encoding knowledge with the designed encoding knowledge table, including column names and exact stored content values used in the SQL. The encoding knowledge is sent to obtain the final Precise SQL following multi-rounds of generation-execution-revision process. To validate our approach, we introduce a table-content-aware, question-related benchmark dataset, containing 2115 question-SQL pairs. Comprehensive experiments conducted on this benchmark demonstrate the remarkable performance of TCSR-SQL, achieving an improvement of at least 27.8% in execution accuracy compared to other state-of-the-art methods.

---

# TCSR‑SQL：面向表内容感知的自检索 Text‑to‑SQL 论文详细解读

### 背景：这个问题为什么难？

在把自然语言问题转成 SQL 查询的任务里，模型往往只看数据库的结构（表名、列名），而忽略了实际存储的数据。真实业务中，用户常会提到“去年销量最高的产品”或“库存低于 10 的商品”，这些词既不是表的列名，也不一定出现在问题里。传统方法把这些词当成噪声，导致生成的 SQL 与数据库实际内容不匹配，执行结果错误。根本原因是缺少一种机制，能够让模型在生成 SQL 前先了解表中到底有哪些具体值以及它们的分布。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问题自动翻译成对应的 SQL 语句，类似把口头指令变成数据库指令。
- **表内容感知（Table‑content‑aware）**：模型在生成 SQL 时会考虑表里实际存储的记录，而不仅仅是表结构。
- **自检索（Self‑retrieval）**：模型自己发起查询，把得到的结果再喂回自身，用来校正或补充信息。
- **Seed SQL**：根据初步抽取的关键词和推测的列名生成的“种子”查询，用来在数据库里做粗略搜索。
- **编码知识表（Encoding Knowledge Table）**：一种结构化的记忆，记录了列名、对应的具体值以及它们在 Seed SQL 中的使用方式。
- **多轮生成‑执行‑修正（Generation‑Execution‑Revision）**：模型循环三步：先生成 SQL → 执行得到结果 → 根据结果调整生成，直到满足需求。
- **执行准确率（Execution Accuracy）**：评估模型生成的 SQL 能否在真实数据库上得到正确答案的指标。

### 核心创新点
1. **从问题中主动抽取数据关键词 → 使用 LLM 的上下文学习能力让模型自行识别出潜在的数值或实体关键词 → 解决了传统方法把这些词当噪声的局限，使后续检索更有针对性。**  
2. **基于抽取的关键词和推测的列名生成 Seed SQL → 在真实数据库上进行模糊搜索 → 把搜索结果映射到编码知识表 → 让模型在生成正式 SQL 前拥有“实际数据的脚注”。**  
3. **引入多轮生成‑执行‑修正循环 → 每一次执行的返回值都会更新编码知识表，模型据此重新生成更精确的 SQL → 相比一次性生成的方式，显著提升了执行准确率。**  
4. **构建了专门的表内容感知基准数据集（2115 对问句‑SQL） → 为评估这类问题提供了统一标准，也让实验结果更具说服力。**

### 方法详解
整体思路可以看成三段式的“先找线索、再查证、最后写答案”。具体步骤如下：

1. **关键词提取与模糊检测**  
   - 输入原始自然语言问题，调用大模型（如 GPT‑4）进行 **in‑context learning**：给模型展示几例“问题 → 关键词”对，让它学会从新问题中挑出可能对应数据库中实际值的词（比如“2022 年”“销量最高”“库存不足 5”）。  
   - 同时，模型会尝试匹配这些词到数据库的列名，产生一个 **候选列集合**。这一步类似侦探先把现场的可疑线索标记出来。

2. **知识检索与对齐**  
   - 根据上一步得到的关键词和候选列，系统自动拼装一条 **Seed SQL**。这条 SQL 并不要求完美，只要能把关键词对应的行筛出来即可。  
   - 把 Seed SQL 发送到真实数据库执行，得到 **检索结果**（可能是一小批记录）。  
   - 检索结果被解析成 **编码知识表**：每一行记录对应的列名、具体值以及它们在 Seed SQL 中出现的位置。可以把它想象成一张“答案卡”，上面写着“这里的‘2022’对应的是‘year’列的值”。  
   - 这张卡随后会被喂回大模型，帮助它在后续生成时使用 **确切的列名和数值**，而不是模糊的猜测。

3. **SQL 生成与多轮修正**  
   - 大模型在接收到原问题、关键词、候选列以及编码知识表后，开始生成 **Precise SQL**。此时模型已经拥有“上下文+真实数据”的双重信息。  
   - 生成的 SQL 被立即执行。如果执行成功且返回的结果与用户期望相符（通过对比执行结果与问题中的暗示），流程结束。  
   - 若执行失败或返回的行数/字段不符合预期，系统会把 **执行错误信息**（比如“列名不存在”或“返回空集”）加入到编码知识表，触发 **修正**：模型重新生成 SQL，利用新的错误提示进行纠正。这个循环通常进行两到三次即可收敛。  
   - 关键的巧思在于把 **执行反馈** 当作额外的检索信号，让模型在每轮都“自我学习”，而不是一次性硬算。

整体框架可以用文字版流程图概括：

```
问题 → 关键词抽取 + 列候选 → 构造 Seed SQL → 执行 → 检索结果 → 编码知识表
      ↘                                 ↗
          多轮生成（带知识表） → Precise SQL → 执行 → (成功? → 结束 / 失败 → 错误加入知识表 → 继续)
```

### 实验与效果
- **数据集**：作者新建了一个专门的表内容感知基准，包含 2115 条真实业务场景下的问句‑SQL 对，覆盖了多种模糊关键词和不存在于 schema 中的实体。  
- **对比基线**：包括最新的 LLM‑based Text‑to‑SQL 系统（如 ChatGPT‑ZeroShot、Codex‑SQL）以及传统基于语义匹配的模型。  
- **主要指标**：执行准确率提升至少 **27.8%**，在最难的“内容模糊”子集上提升更明显。  
- **消融实验**：去掉关键词抽取、去掉 Seed SQL、或只做一次生成‑执行不进行修正，准确率分别下降约 10%、12% 和 15%，说明每个模块都对最终性能贡献显著。  
- **局限**：论文承认对极大规模数据库的模糊搜索成本仍然较高，且对检索结果的质量高度依赖数据库的索引情况；此外，当前实现仍然需要手工提供 few‑shot 示例来引导关键词抽取，自动化程度有提升空间。

### 影响与延伸思考
TCSR‑SQL 把 **执行反馈** 融入生成过程的思路在随后的一批工作中被广泛采纳，尤其是面向 **可解释 SQL 生成** 与 **交互式查询** 的系统。后续研究（如 “Self‑Check‑SQL” 与 “Iterative Retrieval‑Augmented Generation”）进一步扩展了自检索的范围，加入了跨表检索和多模态（表格+文本）信息。想继续深入，可以关注 **检索增强生成（RAG）** 在结构化数据上的应用、以及 **大模型自校正** 的理论分析。

### 一句话记住它
让大模型先自己去数据库里找“线索”，再把找到的真实值喂回去，循环修正，SQL 执行准确率直接飙升。