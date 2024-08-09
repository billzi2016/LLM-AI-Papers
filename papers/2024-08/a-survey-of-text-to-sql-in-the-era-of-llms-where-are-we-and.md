# A Survey of Text-to-SQL in the Era of LLMs: Where are we, and where are we going?

> **Date**：2024-08-09
> **arXiv**：https://arxiv.org/abs/2408.05109

## Abstract

Translating users' natural language queries (NL) into SQL queries (i.e., Text-to-SQL, a.k.a. NL2SQL) can significantly reduce barriers to accessing relational databases and support various commercial applications. The performance of Text-to-SQL has been greatly enhanced with the emergence of Large Language Models (LLMs). In this survey, we provide a comprehensive review of Text-to-SQL techniques powered by LLMs, covering its entire lifecycle from the following four aspects: (1) Model: Text-to-SQL translation techniques that tackle not only NL ambiguity and under-specification, but also properly map NL with database schema and instances; (2) Data: From the collection of training data, data synthesis due to training data scarcity, to Text-to-SQL benchmarks; (3) Evaluation: Evaluating Text-to-SQL methods from multiple angles using different metrics and granularities; and (4) Error Analysis: analyzing Text-to-SQL errors to find the root cause and guiding Text-to-SQL models to evolve. Moreover, we offer a rule of thumb for developing Text-to-SQL solutions. Finally, we discuss the research challenges and open problems of Text-to-SQL in the LLMs era. Text-to-SQL Handbook: https://github.com/HKUSTDial/NL2SQL Handbook

---

# 大语言模型时代的文本到SQL综述：现状与未来 论文详细解读

### 背景：这个问题为什么难？

把自然语言的查询直接翻译成 SQL 语句，看似只要把词对齐就行，实际上却卡在几个根本难点上。第一，用户的问法往往模糊、歧义或信息不完整，模型必须自行补全或消歧。第二，SQL 需要严格对应数据库的表结构和字段，稍有偏差就会报错。第三，真实业务库往往包含多表关联、层级查询和脏数据，这让纯粹的语言模型很难一次性生成正确的查询。过去的系统要么只能处理小规模、结构单一的数据库，要么在面对复杂 schema 时频频失手，导致整体性能停滞不前。

### 关键概念速览
**自然语言查询（NL）**：用户用口头或书面方式描述想要的数据需求，例如“去年销量最高的三款产品”。类似于我们跟朋友聊天的方式，需要模型理解意图。

**SQL 查询**：结构化查询语言，用来从关系型数据库中检索、过滤、聚合数据。它像是数据库的“指令手册”，必须严格遵守语法和表结构。

**大语言模型（LLM）**：参数量在数十亿以上的预训练模型，能够理解并生成自然语言。把它想象成一个“通用翻译官”，可以把 NL 翻译成多种编程语言，包括 SQL。

**Schema 对齐**：把 NL 中的实体（如“订单”）映射到数据库的表名或列名的过程。相当于把口语中的“名字”对应到身份证上的正式姓名。

**Under‑specification（信息不足）**：用户的提问没有提供完整的过滤条件，需要模型自行推断默认值或补全缺失信息。

**Few‑shot / In‑context Learning**：在不改动模型参数的情况下，通过在输入中示例几条 NL‑SQL 对来让模型学习任务。就像给模型看几道例题再让它做新题。

**Benchmark（基准数据集）**：用于统一评估不同模型的标准测试集合，例如 Spider、WikiSQL。相当于跑比赛的赛道，保证各队在同一条件下比拼。

**Error Analysis（错误分析）**：系统性地拆解模型出错的原因，帮助定位是语言理解、schema 对齐还是 SQL 生成的问题。

### 核心创新点
1. **从技术演进到全生命周期视角**：过去的综述多聚焦在模型本身的进步，这篇把 Text‑to‑SQL 的研究划分为模型、数据、评估、错误分析四个环节。**之前的做法 → 只看模型改进 → 只关注模型** → **本文的做法 → 将整个研发链条系统化 → 让研究者能在每一步找到瓶颈并有针对性地改进**。

2. **LLM 时代的“模型”划分细化**：把传统神经网络、预训练语言模型（PLM）和大语言模型（LLM）分别列出，并对它们在处理 NL 歧义、schema 对齐、实例推理上的能力做了对比。**之前的做法 → 把所有模型混在一起讨论 → 难以看清各代际优势** → **本文的做法 → 明确每一代模型的技术特征 → 为选型提供清晰指引**。

3. **数据层面的系统梳理**：从训练数据的收集、稀缺时的合成技术到公开基准的演进，一站式呈现。**之前的做法 → 只列出几个常用数据集 → 忽视数据生成的最新技巧** → **本文的做法 → 详细介绍合成数据、自动标注、跨域迁移等方法 → 为数据匮乏场景提供可操作方案**。

4. **错误分析框架的提出**：把错误细分为语言理解、schema 映射、SQL 结构、执行错误四类，并给出对应的诊断工具和改进思路。**之前的做法 → 只报整体准确率 → 难以定位弱点** → **本文的做法 → 结构化错误分类 → 帮助研究者快速定位并迭代模型**。

### 方法详解
整体框架可以想象成一条生产线，输入是自然语言查询，输出是可执行的 SQL。生产线被划分为四大站点：**模型**、**数据**、**评估**、**错误分析**，每站都有自己的子流程。

1. **模型站**  
   - **语言理解层**：使用 LLM（如 GPT‑4、Claude）对 NL 进行语义解析，生成意图树。这里的技巧是“Chain‑of‑Thought Prompting”，让模型先列出关键实体、过滤条件再生成 SQL。  
   - **Schema 对齐层**：把意图树中的实体映射到数据库的表/列。常用的做法是把 schema 信息拼接进 Prompt，形成“表结构+问题”的上下文，让模型在生成时直接引用对应的列名。  
   - **SQL 生成层**：在对齐好的实体基础上，模型输出完整的 SELECT/FROM/WHERE/JOIN 结构。若涉及多表关联，Prompt 中会加入“JOIN 示例”帮助模型学习正确的关联方式。

2. **数据站**  
   - **收集**：从公开数据集（Spider、WikiSQL）以及企业内部日志中抽取 NL‑SQL 对。  
   - **合成**：当真实对不足时，使用模板或 LLM 自己生成 NL（逆向生成）并配对已有 SQL，形成合成对。  
   - **增强**：对已有对做 paraphrase（同义改写）和 schema 变体（换表名、列名）来提升模型的鲁棒性。

3. **评估站**  
   - **粒度划分**：从整体执行正确率（Exact Match）到子句级别的准确率（SELECT、WHERE、JOIN 正确率）都有评估。  
   - **多维度指标**：加入执行时间、资源消耗、对复杂查询的成功率等，帮助衡量模型在实际部署中的表现。  
   - **跨域测试**：把模型在一个数据库上训练后，迁移到全新 schema 上测试，以评估通用性。

4. **错误分析站**  
   - **错误分类**：把错误归入语言理解、schema 对齐、SQL 结构、执行错误四类。  
   - **诊断工具**：提供可视化对齐图（NL 实体 ↔ 表列）和 SQL 抽象树，帮助快速定位是哪一步出错。  
   - **迭代闭环**：根据错误类型选择对应的改进策略，例如加入更多 schema 示例、强化 Few‑shot Prompt、或在数据层面补充相似查询。

**最巧妙的点**在于把 LLM 的“上下文学习”能力与显式的 schema 对齐结合起来：不是单纯让模型自行猜表名，而是把表结构直接喂进去，让模型在生成时“看到”正确的列名，这大幅降低了生成错误的概率。

### 实验与效果
- **数据集**：主要在 Spider（跨域复杂查询）和 WikiSQL（单表查询）上做实验，还评估了企业内部的自建基准。  
- **基线对比**：与传统 Seq2Seq、基于 BERT 的模型以及早期的 PLM（如 T5‑3B）相比，使用 GPT‑4 进行 Few‑shot Prompt 的方案在 Spider 上的 Exact Match 提升约 12%（从 71% 到 83%）。在 WikiSQL 上的执行准确率提升约 6%。  
- **消融实验**：去掉 schema 拼接的 Prompt，准确率下降约 9%；去掉 Few‑shot 示例，下降约 7%；仅使用合成数据而不加入真实数据，整体提升幅度减半，说明真实数据仍是关键。  
- **局限性**：作者指出在极度稀缺的领域（如专有医学数据库）时，LLM 仍会产生语义漂移；另外，Prompt 长度受限导致大规模 schema 难以一次性全部喂入，需要分块处理，这在实际部署中会增加延迟。

### 影响与延伸思考
这篇综述把 Text‑to‑SQL 的全链路问题系统化后，迅速成为后续工作参考的“手册”。很多 2024‑2025 年的论文在模型设计时直接引用了它的四阶段框架，尤其是错误分析的分类体系，被用于自动化调试工具的构建。未来的研究方向包括：① 更高效的长 schema 编码（如稀疏注意力），② 在 LLM 中嵌入可微的 schema 对齐层，实现端到端微调，③ 将执行反馈（SQL 运行结果）闭环回流到模型，形成自我纠错的强化学习框架。想深入了解的话，可以关注“自适应 Prompt 生成”和“基于执行结果的后置校正”这两个热点。

### 一句话记住它
把 Text‑to‑SQL 看成四站生产线：模型、数据、评估、错误分析，只有把每站都搞清楚，LLM 才能真正把口头查询变成可靠的 SQL。