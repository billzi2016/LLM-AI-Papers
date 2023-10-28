# Reboost Large Language Model-based Text-to-SQL, Text-to-Python, and   Text-to-Function -- with Real Applications in Traffic Domain

> **Date**：2023-10-28
> **arXiv**：https://arxiv.org/abs/2310.18752

## Abstract

The previous state-of-the-art (SOTA) method achieved a remarkable execution accuracy on the Spider dataset, which is one of the largest and most diverse datasets in the Text-to-SQL domain. However, during our reproduction of the business dataset, we observed a significant drop in performance. We examined the differences in dataset complexity, as well as the clarity of questions' intentions, and assessed how those differences could impact the performance of prompting methods. Subsequently, We develop a more adaptable and more general prompting method, involving mainly query rewriting and SQL boosting, which respectively transform vague information into exact and precise information and enhance the SQL itself by incorporating execution feedback and the query results from the database content. In order to prevent information gaps, we include the comments, value types, and value samples for columns as part of the database description in the prompt. Our experiments with Large Language Models (LLMs) illustrate the significant performance improvement on the business dataset and prove the substantial potential of our method. In terms of execution accuracy on the business dataset, the SOTA method scored 21.05, while our approach scored 65.79. As a result, our approach achieved a notable performance improvement even when using a less capable pre-trained language model. Last but not least, we also explore the Text-to-Python and Text-to-Function options, and we deeply analyze the pros and cons among them, offering valuable insights to the community.

---

# 基于大语言模型的文本到SQL、文本到Python、文本到函数的再提升 论文详细解读

### 背景：这个问题为什么难？

在把自然语言问题转成数据库查询（Text‑to‑SQL）时，模型必须同时理解业务意图、数据库结构以及潜在的歧义。Spider 数据集虽然规模大、覆盖面广，却是学术界精心构造的“干净”样本，问题往往明确、表述规范。真实业务场景（如交通行业）里，提问常常口语化、信息缺失，甚至包含噪声，这让直接使用大语言模型（LLM）进行一次性 prompting 的准确率大幅下降。之前的最强方法在 Spider 上能拿到接近 80% 的执行准确率，却在业务数据集上跌到 20% 左右，说明仅靠模型的原始知识和一次性提示无法跨越数据复杂度和意图模糊性的鸿沟。

### 关键概念速览

**Prompt（提示）**：给 LLM 的输入文本，包含任务说明、示例和上下文信息。相当于在和模型对话时先给它一张“任务卡片”。  

**Query Rewriting（查询改写）**：把用户的原始自然语言问题重新表述，使其更明确、结构化。好比把口语的“怎么查一下今天的车流量”改成“请返回今天 08:00‑18:00 区域 A 的车流量统计”。  

**SQL Boosting（SQL 增强）**：在生成的 SQL 基础上加入执行反馈或查询结果，形成二次修正。类似于写完代码后先跑一遍，看到报错再回头改。  

**Database Description（数据库描述）**：在提示里加入列的注释、数据类型、示例值等元信息，让模型对表结构有更直观的“感知”。  

**Text‑to‑Python / Text‑to‑Function**：把自然语言直接翻译成可执行的 Python 代码或函数，而不是先生成 SQL。把任务看作“写脚本”而不是“写查询”。  

**Execution Accuracy（执行准确率）**：模型输出的代码或查询在真实数据库上运行后得到正确结果的比例。比单纯的语义匹配更能衡量实际可用性。  

**Few‑shot Prompting**：在提示中提供少量示例，帮助模型学习任务模式。相当于给模型“示范几遍”。  

### 核心创新点

1. **一次性提示 → 两阶段提示（改写 + 增强） → 大幅提升鲁棒性**  
   过去的 SOTA 直接把用户问题和数据库描述塞进一个 Prompt，模型一次性输出 SQL。本文把流程拆成两步：先用 Query Rewriting 把模糊问题变成精准描述，再让模型生成 SQL，随后用执行反馈进行 SQL Boosting。实验显示，这种分层处理把业务数据集的执行准确率从 21% 拉到 66%。  

2. **仅列名 → 列注释 + 类型 + 示例值 → 信息闭环**  
   传统提示只提供表名和列名，模型只能靠记忆猜测列含义。作者在 Prompt 中加入每列的中文注释、数据类型以及几条真实样本值，形成“全景图”。这让模型在生成查询时能更好地匹配业务意图，尤其在列名抽象或多义时效果显著。  

3. **SQL → 执行反馈循环 → 自适应修正**  
   生成的 SQL 先在数据库上执行，若返回错误或空结果，系统把错误信息、返回的部分结果重新拼进 Prompt，要求模型基于这些信息“调参”。这种类似人类调试的闭环，使得即使使用较小的 LLM（如 Llama‑2‑7B）也能逼近大模型的表现。  

4. **从 Text‑to‑SQL 扩展到 Text‑to‑Python / Text‑to‑Function 的对比实验**  
   作者不仅停留在 SQL 层面，还让模型直接写 Python 脚本或函数，比较三者在执行准确率、可解释性和部署成本上的差异。结果表明，Text‑to‑Python 在需要复杂业务逻辑时更灵活，而 Text‑to‑SQL 在结构化查询上仍是最稳妥的选择。  

### 方法详解

#### 整体框架

整个系统可以看作三层流水线：**（1）问题改写层 →（2）代码生成层 →（3）执行反馈层**。每层都通过 Prompt 与 LLM 交互，输出结果再喂给下一层。核心思想是把一次性“猜答案”改成“先把问题说清楚、再写代码、再根据运行结果调校”。  

#### 1. 问题改写（Query Rewriting）

- **输入**：原始用户提问 + 数据库描述（列注释、类型、示例）。  
- **Prompt 设计**：示例中先给出几条模糊问题和对应的清晰改写，让模型学习“把口语转成正式查询意图”。  
- **输出**：改写后的自然语言句子，结构更接近 SQL 的 SELECT‑FROM‑WHERE 形式。  

类比：把一段口头指令先写成书面指令，再交给工程师执行。

#### 2. 代码生成（Text‑to‑SQL / Text‑to‑Python / Text‑to‑Function）

- **输入**：改写后的问题 + 完整的数据库描述。  
- **Prompt 结构**：  
  - 任务说明（如“请根据下面的描述生成对应的 SQL 查询”。）  
  - 表结构表格（列名、类型、注释、示例值）。  
  - 若是 Text‑to‑Python，则提供对应的库（pandas、sqlite3）使用示例。  
- **模型选择**：作者在实验中分别用了 GPT‑3.5、Llama‑2‑7B、Claude‑2 等，展示即使是小模型也能受益于强化 Prompt。  
- **输出**：完整的 SQL 语句或 Python 函数代码。  

#### 3. 执行反馈（SQL Boosting）

- **执行**：把生成的代码在真实的业务数据库上跑一遍。  
- **捕获**：如果出现语法错误、列不存在或返回空集，系统会把错误信息、返回的部分行、甚至查询计划拼回 Prompt。  
- **二次 Prompt**：示例中加入“上一次执行返回错误：... 请基于该信息修正 SQL”。模型在此基础上输出修正版。  
- **循环次数**：最多两轮，实验表明第二轮已经能把大多数错误纠正。  

#### 巧妙之处

- **信息闭环**：把数据库的元信息（注释、示例）直接写进 Prompt，避免模型只能靠记忆。  
- **任务分解**：把“理解意图”和“写代码”拆开，让每一步的难度更低，模型的错误率随之下降。  
- **执行驱动的自适应**：不像传统的后处理规则，反馈是由真实执行结果产生，具备通用性。  

### 实验与效果

- **数据集**：Spider（学术基准）和作者自行收集的交通行业业务数据集（包含道路流量、信号灯状态等 12 张表）。  
- **基线**：原始 SOTA 方法（一次性 Prompt + Spider 预训练模型），以及直接使用 GPT‑4 的零-shot 版本。  
- **主要指标**：执行准确率（Execution Accuracy）。  
- **结果**：在业务数据集上，SOTA 只能达到 21.05%，而 ReBoost 方法提升到 65.79%。即使换成 Llama‑2‑7B（相对弱小的模型），也能超过 60% 的执行准确率，说明方法的提升并非依赖超大模型。  
- **消融实验**：  
  - 去掉列注释/示例值，准确率跌至约 48%。  
  - 只做 Query Rewriting 不做 SQL Boosting，准确率约 53%。  
  - 只做 SQL Boosting 不做改写，准确率约 50%。  
  这表明三大模块相互补足，缺一不可。  
- **Text‑to‑Python / Text‑to‑Function 对比**：在需要多表联结且业务逻辑复杂的查询上，Text‑to‑Python 的执行准确率略高（约 68%），但代码可维护性和安全审计成本更高；Text‑to‑SQL 在大多数结构化查询上仍是最稳妥的选择。  
- **局限性**：作者承认方法对极度稀疏或缺少示例值的列仍会出现误判；反馈循环最多两轮，若错误根源在模型的根本知识缺失，仍难以纠正。  

### 影响与延伸思考

这篇工作把“提示工程”提升到一个闭环调试的层次，直接影响了后续的 LLM‑SQL 研究。随后出现的几篇论文（如 *Self‑Correcting NL2SQL*、*Feedback‑Driven Prompting*）都借鉴了执行反馈的思路。对想继续深挖的读者，可以关注以下方向：  
- **自动化示例生成**：如何在没有人工标注的情况下自动抽取列的示例值。  
- **多轮自适应调试**：把反馈循环扩展到更多轮次，甚至让模型自行决定是否需要再调。  
- **跨模态扩展**：把交通摄像头图像或实时流数据加入 Prompt，探索 Text‑to‑SQL 与视觉信息的融合。  

### 一句话记住它

把模糊的业务提问先写清楚，再让模型写代码，最后用真实执行结果“喂回去”纠错——这套“改写 + 生成 + 反馈”三部曲让小模型也能在真实数据库上跑出高准确率。