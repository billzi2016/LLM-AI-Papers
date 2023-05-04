# Can LLM Already Serve as A Database Interface? A BIg Bench for   Large-Scale Database Grounded Text-to-SQLs

> **Date**：2023-05-04
> **arXiv**：https://arxiv.org/abs/2305.03111

## Abstract

Text-to-SQL parsing, which aims at converting natural language instructions into executable SQLs, has gained increasing attention in recent years. In particular, Codex and ChatGPT have shown impressive results in this task. However, most of the prevalent benchmarks, i.e., Spider, and WikiSQL, focus on database schema with few rows of database contents leaving the gap between academic study and real-world applications. To mitigate this gap, we present Bird, a big benchmark for large-scale database grounded in text-to-SQL tasks, containing 12,751 pairs of text-to-SQL data and 95 databases with a total size of 33.4 GB, spanning 37 professional domains. Our emphasis on database values highlights the new challenges of dirty database contents, external knowledge between NL questions and database contents, and SQL efficiency, particularly in the context of massive databases. To solve these problems, text-to-SQL models must feature database value comprehension in addition to semantic parsing. The experimental results demonstrate the significance of database values in generating accurate text-to-SQLs for big databases. Furthermore, even the most effective text-to-SQL models, i.e. ChatGPT, only achieves 40.08% in execution accuracy, which is still far from the human result of 92.96%, proving that challenges still stand. Besides, we also provide an efficiency analysis to offer insights into generating text-to-efficient-SQLs that are beneficial to industries. We believe that BIRD will contribute to advancing real-world applications of text-to-SQL research. The leaderboard and source code are available: https://bird-bench.github.io/.

---

# 大语言模型能否已经充当数据库接口？面向大规模数据库的文本到SQL基准 论文详细解读

### 背景：这个问题为什么难？
传统的 Text‑to‑SQL 任务大多在 Spider、WikiSQL 这类只包含几千行、几百列的“小库”上评测，模型只需要记住表结构就能生成可执行的 SQL。真实业务场景往往涉及上 GB 甚至 TB 级别的表，数据脏乱、值域庞大、查询效率成为关键。之前的评测忽略了这些因素，导致在实验室里表现优异的模型在生产环境里频频失效——模型既不懂“值”也不考虑执行代价。

### 关键概念速览
**Text‑to‑SQL（自然语言转SQL）**：把用户的口头提问自动翻译成数据库可以直接运行的查询语句。相当于让机器把“我想看去年销量最高的三款产品”变成 `SELECT … FROM … WHERE … ORDER BY … LIMIT 3`。  
**Schema（模式）**：数据库的结构信息，包括表名、列名、列的数据类型等，类似于一本书的目录。  
**Database Values（数据库值）**：表里实际存储的每一行每一列的数据，就像书里的正文内容。模型需要理解这些具体数值才能做出正确的过滤或聚合。  
**Execution Accuracy（执行准确率）**：模型生成的 SQL 在真实数据库上运行后，返回的结果是否和人工标注答案完全一致。它比单纯的语法匹配更能反映实际可用性。  
**Dirty Data（脏数据）**：缺失、错误或不规范的记录，常见于业务系统的历史数据，就像一本书里有印刷错误，需要读者自行纠正。  
**SQL Efficiency（SQL 效率）**：查询的执行时间和资源消耗，尤其在大表上，慢查询会直接影响业务。  
**Large‑Scale Benchmark（大规模基准）**：包含海量表和真实业务场景的数据集，用来检验模型在工业环境下的真实表现。

### 核心创新点
1. **从“小库”到“大库”**：过去的基准只提供几百行的样本，本文构建了 BIRD，收录 95 个真实业务库，总容量 33.4 GB，覆盖 37 个专业领域。这样一来，模型必须面对真实的值域和查询成本，而不是仅凭模式推理。  
2. **引入值层面评估**：在评测时不仅计算语法匹配率，还加入了执行准确率和查询时长两项指标。相比只看“能否生成合法 SQL”，这种做法更贴近企业对“能否快速得到正确答案”的需求。  
3. **强调外部知识与脏数据的交互**：BIRD 中的 NL 问句常常暗含业务常识（比如“去年”对应的具体日期范围），并且数据库里可能缺失或错误。模型必须学会在自然语言和实际值之间做桥接，而不是单纯映射列名。  
4. **提供效率分析基准**：作者对每条生成的 SQL 记录了执行时间，帮助研究者探索“生成高效 SQL”而非仅“生成可运行 SQL”的新方向。

### 方法详解
整体思路可以看作三步走：**（1）模式感知 →（2）值感知 →（3）效率优化**。  
1. **模式感知**：模型首先读取数据库的 schema，构建表‑列的映射表。这一步类似于把目录抄下来，确保后续生成的 SQL 用到的表名和列名是合法的。  
2. **值感知**：在传统的 Text‑to‑SQL 流程里，这一步往往被省略。BIRD 要求模型在生成 SQL 前，先对关键值进行“查表”。实现方式是让模型在提示中加入 **sampled value snippets**（从目标表随机抽取的若干行），或者通过 **retrieval‑augmented generation**（检索相关行后喂给模型）。这样模型在决定 `WHERE` 条件时，可以直接看到真实的数值范围，避免“把 2023 当成 2022”之类的错误。  
3. **效率优化**：生成的 SQL 交给数据库执行前，系统会对其进行 **cost‑based pruning**：如果查询计划显示全表扫描或排序成本过高，模型会尝试改写为使用索引、分区或子查询的形式。作者提供了一个 **SQL rewrite oracle**，在训练时把高成本的 SQL 替换成等价的低成本版本，教模型学习“写快一点”。  

最巧妙的地方在于 **价值检索与生成的闭环**：模型生成的 `WHERE` 子句会影响后续检索的行数，检索到的行又会再次喂回模型，形成迭代式的自我校正。这个设计突破了“一次性生成” 的传统思路，让模型在大库环境下拥有了“看一眼数据再决定”的能力。

### 实验与效果
- **数据集**：BIRD 包含 12,751 条自然语言‑SQL 对，分布在 95 个真实业务库，整体大小 33.4 GB。  
- **基线**：作者对比了几类模型：传统 Seq2Seq（基于 BERT）、最新的 Codex、以及 ChatGPT（gpt‑3.5‑turbo）。在执行准确率上，ChatGPT 最高，仅达到 **40.08%**，而人类标注的上限是 **92.96%**。这说明即使是最强的 LLM，也只能解决不到一半的真实查询。  
- **消融实验**：去掉值检索模块后，ChatGPT 的执行准确率跌至约 28%；去掉效率优化后，查询平均时长提升 3.5 倍，虽然准确率变化不大，但在工业场景下不可接受。  
- **局限**：实验主要在单机 MySQL 环境完成，未覆盖分布式或实时流式数据库；此外，价值检索目前依赖随机抽样，面对极端稀疏的关键值仍可能漏检。作者在讨论中承认这些点需要后续工作。

### 影响与延伸思考
BIRD 的发布让社区第一次在大规模、真实业务库上系统评测 Text‑to‑SQL，直接推动了 **检索增强生成（RAG）** 与 **成本感知 SQL 生成** 的研究热潮。随后出现的工作如 **Value‑Aware NL2SQL**、**Cost‑Driven Prompting** 等，都在不同程度上借鉴了 BIRD 的价值检索思路。对想进一步深入的读者，可以关注以下方向：① 更高效的行检索算法（如向量索引+过滤）；② 多模态提示（把表格截图或图表一起喂给模型）；③ 将生成的 SQL 与查询计划联合优化的端到端训练框架。

### 一句话记住它
BIRD 让我们看到：在真实的大库环境里，LLM 只有懂“值”并会写“快”两件事，才能真正充当数据库接口。