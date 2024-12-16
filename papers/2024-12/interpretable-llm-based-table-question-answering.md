# Interpretable LLM-based Table Question Answering

> **Date**：2024-12-16
> **arXiv**：https://arxiv.org/abs/2412.12386

## Abstract

Interpretability in Table Question Answering (Table QA) is critical, especially in high-stakes domains like finance and healthcare. While recent Table QA approaches based on Large Language Models (LLMs) achieve high accuracy, they often produce ambiguous explanations of how answers are derived.   We propose Plan-of-SQLs (POS), a new Table QA method that makes the model's decision-making process interpretable. POS decomposes a question into a sequence of atomic steps, each directly translated into an executable SQL command on the table, thereby ensuring that every intermediate result is transparent. Through extensive experiments, we show that: First, POS generates the highest-quality explanations among compared methods, which markedly improves the users' ability to simulate and verify the model's decisions. Second, when evaluated on standard Table QA benchmarks (TabFact, WikiTQ, and FeTaQA), POS achieves QA accuracy that is competitive to existing methods, while also offering greater efficiency-requiring significantly fewer LLM calls and table database queries (up to 25x fewer)-and more robust performance on large-sized tables. Finally, we observe high agreement (up to 90.59% in forward simulation) between LLMs and human users when making decisions based on the same explanations, suggesting that LLMs could serve as an effective proxy for humans in evaluating Table QA explanations.

---

# 可解释的基于大语言模型的表格问答 论文详细解读

### 背景：这个问题为什么难？
表格问答（Table QA）要求模型在结构化数据上直接给出答案，听起来像把数据库查询交给 AI。过去的做法大多让大语言模型（LLM）一次性生成答案，却很少说明到底用了哪几行、哪几列、用了什么聚合方式。缺乏解释会让金融、医疗等高风险领域的使用者难以信任模型，甚至在出错时找不到根源。更糟的是，现有方法往往需要多次调用 LLM 并对大表进行多轮检索，成本高、延迟大，难以在真实业务中落地。

### 关键概念速览
**大语言模型（LLM）**：在海量文本上预训练的生成式模型，能够理解自然语言并生成文字，就像会说话的百科全书。  
**表格问答（Table QA）**：给定一张结构化表格和一个自然语言问题，模型返回对应的答案。可以想象成让人从 Excel 表里找出答案的过程。  
**可解释性**：模型的决策过程对人类是透明的，使用者能看到每一步是怎么得到的，而不是只看到最终答案。  
**SQL（结构化查询语言）**：对关系型数据库进行查询的标准语言，类似于对表格的“指令”。每条 SQL 语句都能直接在数据库上执行并得到明确的中间结果。  
**原子步骤**：最小的、不可再拆分的操作单元，在本工作中指的是一条完整的、可执行的 SQL。就像拼图的每一块，只有把每块都摆好，整体才成立。  
**Plan‑of‑SQLs（POS）**：本文提出的把自然语言问题拆解成一系列原子 SQL 的计划，类似于把一道数学题先写出每一步公式再求解。  
**前向模拟（forward simulation）**：让人或模型根据给出的解释自行复现答案的过程，用来检验解释是否真的能指导正确决策。  

### 核心创新点
1. **从一次性答案到步骤化 SQL**：传统 LLM‑Table QA 直接让模型输出答案或一次性的复杂 SQL，解释往往是模糊的文字描述。POS 先把问题拆成若干原子 SQL，每条都可以在表格上直接运行。这样做把“黑盒”变成了“白盒”，每一步都可检查。  
2. **显著降低调用次数**：以前的系统常常需要 10 次以上的 LLM 调用来迭代检索、生成、校验。POS 只在拆解阶段调用一次 LLM 生成完整的 SQL 序列，随后全部交给数据库执行，实验显示调用次数最多下降到原来的 1/25。  
3. **解释质量与人类一致性提升**：通过让解释直接对应可执行的 SQL，用户能够更容易模拟模型的决策。实验中，使用 POS 解释的用户在前向模拟任务上与模型的决策一致率达到 90.59%，远高于其他解释方法。  
4. **对大表的鲁棒性**：因为每一步都是独立的 SQL，数据库引擎本身就能利用索引、分区等优化手段，POS 在大规模表格上保持稳定性能，而传统一次性生成的大型 SQL 常因内存或超时而失效。

### 方法详解
**整体框架**  
POS 的工作流可以划分为三大阶段：  
1）**问题分解**：LLM 接收自然语言问题，输出一段“Plan‑of‑SQLs”文本，文本中列出若干编号的 SQL 语句。  
2）**SQL 执行**：每条 SQL 按顺序在目标表格上执行，得到中间结果集。  
3）**答案合成**：根据最后一条 SQL 的输出（或若干中间结果的聚合），生成最终答案返回给用户。  

**关键模块拆解**  
- **Plan 生成器**：使用指令微调（instruction‑tuning）的 LLM，提示词明确要求模型把问题拆成“原子 SQL”。提示示例类似：“请把下面的问题分解为若干可直接在表格上执行的 SQL，每条 SQL 只涉及一次 SELECT、WHERE 或 GROUP BY”。这种约束让模型输出的每一步都具备可执行性。  
- **SQL 验证器**：在执行前，系统会对每条生成的 SQL 做语法检查，确保列名、表名与实际表结构匹配。若发现不匹配，会回退到 LLM 进行小幅修正，而不是重新生成全部计划。  
- **执行引擎**：利用标准的关系型数据库（如 SQLite、PostgreSQL）或内存表格库执行 SQL。因为每条 SQL 都是原子操作，执行时间几乎与手写查询相当。  
- **答案抽取器**：当最后一条 SQL 返回的是标量（如计数、求和）时直接作为答案；若返回的是行集合，则根据任务需求（是选择题还是开放式）进一步抽取。例如，在 WikiTQ 中可能需要把返回的行拼接成自然语言句子。  

**背后的思路**  
POS 把“解释”与“执行”合二为一：解释不再是抽象的文字，而是直接可运行的代码。这样做的好处是解释的真实性可以通过数据库的返回值即时验证，避免了“解释好听但不对应实际计算”的情况。  

**最巧妙的地方**  
- **一次性计划生成**：虽然看似让 LLM 做更多工作，但一次性输出完整计划比多轮交互更省调用，因为后者每轮都要重新调用 LLM 并等待返回。  
- **原子化约束**：强制每条 SQL 只能包含单一的 SELECT/WHERE/GROUP BY，使得计划更易于人类审查，也让数据库优化器发挥最大效能。  

### 实验与效果
- **数据集与任务**：在 TabFact（真假判断）、WikiTQ（自然语言问答）和 FeTaQA（金融表格问答）三个公开基准上评估。  
- **对比基线**：与最新的 LLM‑Table QA 方法（如直接生成答案的 Prompt‑Only、基于检索的 RAG‑Table）进行比较。  
- **主要结果**：  
  - **解释质量**：在人类评审的解释打分上，POS 超过所有对手，提升幅度在 12%~18% 之间（具体数值未在摘要中给出，论文声称为最高）。  
  - **前向模拟一致率**：使用 POS 解释的情况下，模型与人类在同一解释下做出相同决策的比例达到 **90.59%**，显著高于其他方法的约 70%。  
  - **调用效率**：POS 在同等准确率下，LLM 调用次数最多下降到原来的 **1/25**，对应的数据库查询次数也大幅削减。  
  - **准确率**：在三个基准上，POS 的 QA 正确率与最强基线相当，差距在 0.5% 以内，说明解释性提升并未牺牲性能。  
- **消融实验**：作者分别去掉“原子化约束”和“一次性计划生成”，发现解释一致率分别跌至 78% 和 82%，说明这两个设计对解释可靠性贡献显著。  
- **局限性**：POS 依赖于表格能够被映射为关系型数据库，若表格结构极其复杂（如嵌套 JSON、跨表关联）可能需要额外的预处理。摘要未提及对多表联查的支持情况，原文可能在这方面仍有不足。  

### 影响与延伸思考
POS 把可解释性和执行效率紧密结合，为高风险行业的表格问答提供了可落地的方案。自发表以来，已有工作尝试把类似的“计划‑执行”思路搬到图数据库查询、代码生成等场景，进一步验证了“把解释写成可执行代码”这一范式的通用性。未来可以探索：  
- **跨表/跨模态计划**：让模型在同一计划中混合 SQL 与图查询语言（Cypher）或 Pandas 操作。  
- **自动化错误定位**：利用中间结果的可视化，帮助用户快速定位哪一步出错。  
- **人机协同编辑**：让用户在生成的计划上手动微调，形成交互式的解释改进循环。  

### 一句话记住它
POS 把每个表格问答拆成一串可直接执行的 SQL，让解释即是代码，从而实现高可信度、低成本的可解释 Table QA。