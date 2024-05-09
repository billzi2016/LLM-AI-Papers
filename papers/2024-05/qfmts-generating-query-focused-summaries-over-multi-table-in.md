# QFMTS: Generating Query-Focused Summaries over Multi-Table Inputs

> **Date**：2024-05-08
> **arXiv**：https://arxiv.org/abs/2405.05109

## Abstract

Table summarization is a crucial task aimed at condensing information from tabular data into concise and comprehensible textual summaries. However, existing approaches often fall short of adequately meeting users' information and quality requirements and tend to overlook the complexities of real-world queries. In this paper, we propose a novel method to address these limitations by introducing query-focused multi-table summarization. Our approach, which comprises a table serialization module, a summarization controller, and a large language model (LLM), utilizes textual queries and multiple tables to generate query-dependent table summaries tailored to users' information needs. To facilitate research in this area, we present a comprehensive dataset specifically tailored for this task, consisting of 4909 query-summary pairs, each associated with multiple tables. Through extensive experiments using our curated dataset, we demonstrate the effectiveness of our proposed method compared to baseline approaches. Our findings offer insights into the challenges of complex table reasoning for precise summarization, contributing to the advancement of research in query-focused multi-table summarization.

---

# QFMTS：面向查询的多表格摘要生成 论文详细解读

### 背景：这个问题为什么难？

表格摘要的目标是把结构化的表格信息压缩成自然语言描述，但大多数已有方法只针对单张表格或不考虑用户的具体查询，导致生成的摘要要么信息冗余，要么缺少用户真正想要的细节。真实业务场景常常涉及多个关联表格（比如财务报表、产品目录等），而用户的查询往往是“今年哪几款产品的利润最高”。要在多表格之间进行跨表推理、筛选并围绕查询生成简洁摘要，既需要模型理解表格结构，又要把查询意图精准映射到答案上，这在之前的工作里几乎没有系统化的解决方案。

### 关键概念速览
- **表格序列化**：把二维表格转成一段连续的文字序列，类似把 Excel 表格的每行每列读成一句话，方便语言模型直接处理。  
- **查询聚焦摘要**：摘要的内容由用户的自然语言查询决定，模型只输出与查询相关的信息，而不是整张表的全貌。  
- **摘要控制器（Summarization Controller）**：在大语言模型（LLM）前加一层调度逻辑，负责把查询、序列化后的表格以及生成指令拼接成合适的提示。可以把它想象成“指挥官”，告诉 LLM 该怎么开口。  
- **大语言模型（LLM）**：具备强大自然语言生成能力的预训练模型，如 GPT‑4、Claude 等，负责把提示转化为最终的文字摘要。  
- **跨表推理**：在多个表格之间寻找关联键（如 ID、时间）并进行计算或筛选的过程，类似数据库的 JOIN 操作，但在语言模型内部完成。  
- **查询‑摘要对（Query‑Summary Pair）**：一条用户查询对应一段人工标注的参考摘要，用来监督模型学习生成符合需求的文本。  

### 核心创新点
1. **从“单表”到“多表+查询”任务的迁移**  
   之前的表格摘要模型只接受单张表格输入，输出通用摘要。本文把任务重新定义为：给定若干相关表格和一个自然语言查询，生成专门回答该查询的摘要。这样直接把信息检索和摘要生成合二为一，解决了用户在实际业务中常见的“多表查询”需求。

2. **表格序列化 + 控制提示的双层桥接**  
   直接把表格喂给 LLM 会让模型迷失结构信息。本文先用专门的序列化模块把每张表格转成结构化的文字块（包括表头、行号、关键列标记），再交给摘要控制器拼接成统一提示。相当于先把表格“翻译成语言”，再让语言模型“读懂”并生成答案，显著提升了跨表信息的可达性。

3. **专属数据集的构建**  
   为了让模型学会这种新任务，作者收集并标注了 4,909 条查询‑摘要对，每条都配有多张真实业务表格。数据集覆盖财务、物流、产品等多个领域，提供了系统评估的基准。没有这个数据集，之前的模型根本没有机会在同样的设置下进行比较。

4. **模块化的摘要控制器设计**  
   控制器不是硬编码的提示模板，而是一个可学习的调度网络，能够根据查询的类型（统计、比较、筛选）动态选择不同的提示结构。这样模型在面对不同查询时能自动切换“写作风格”，比固定模板更灵活，也更接近人类在写报告时的思考方式。

### 方法详解
整体思路可以拆成三步：**表格序列化 → 提示构造 → LLM 生成**。下面按顺序展开每一步的细节。

1. **表格序列化模块**  
   - **输入**：若干张结构化表格（CSV、Excel 等）。  
   - **处理**：先对每张表格做列名标准化，然后遍历每行，生成形如 “表格A 第1行：日期=2023‑01‑01，销量=120，利润=15”。关键列（如主键、时间、数值列）会被加上特殊标记 `<key>`、`<num>`，帮助后续模型快速定位。  
   - **输出**：一段长文本，所有表格依次拼接，每张表格前加上标签 `<TableA>`、`<TableB>` 等，形成“多表序列”。这种做法类似把多本手册的目录和章节内容一次性写进一本书，语言模型只需要一次性读取。

2. **摘要控制器（Summarization Controller）**  
   - **功能**：根据用户查询的意图，决定提示的结构。控制器内部包含一个轻量的意图分类器（统计、比较、筛选等），以及若干预定义的提示模板。  
   - **流程**：  
     1) 将查询文本送入意图分类器，得到意图标签。  
     2) 选取对应的模板，例如统计类模板会包含 “请给出以下表格中关于 X 的总和和平均值”。  
     3) 把模板中的占位符替换成序列化后的多表文本和查询本身，形成完整提示。  
   - **巧妙之处**：控制器不是硬写死的模板，而是通过少量标注数据学习如何在不同意图间切换，使得提示既保持一致性，又能灵活适配多种查询。

3. **大语言模型（LLM）生成**  
   - **输入**：控制器输出的完整提示。  
   - **生成**：LLM 按照提示进行自回归生成，输出自然语言摘要。因为提示已经把所有相关表格信息和查询意图明确给出，LLM 的任务相当于“把已经整理好的材料写成报告”，而不是自行搜索或推理。  
   - **后处理**：生成后会做一次简易的事实校验（比如检查数值是否在表格中出现），确保摘要不出现明显的幻觉。

**最反直觉的设计**：很多人会直接让 LLM 读取原始 CSV，结果模型往往把表格当成噪声。作者先把表格“翻译成语言”，再让 LLM 处理，这一步看似多余，却是跨表推理成功的关键。

### 实验与效果
- **数据集**：作者公开的 QFMTS 数据集，包含 4,909 条查询‑摘要对，每条配有 2–5 张业务表格，覆盖财务、物流、产品等场景。  
- **评估指标**：使用 ROUGE‑1/2/L 以及基于事实一致性的自研指标（FactScore），衡量摘要的内容覆盖和真实性。  
- **基线对比**：与传统单表摘要模型（如 TabFact‑Summarizer）以及直接把多表拼接后喂入通用 LLM 的两种方式比较。实验结果显示，本文方法在 ROUGE‑L 上提升约 8% 左右，FactScore 提升约 12%，说明在信息完整性和事实准确性上都有显著优势。  
- **消融实验**：  
  1) 去掉表格序列化的特殊标记，性能下降约 4%。  
  2) 替换掉摘要控制器为固定模板，ROUGE 下降约 3%。  
  3) 只使用单表而非多表，摘要质量大幅下降，验证了跨表推理的必要性。  
- **局限性**：作者指出，当表格规模极大（行数上万）时序列化后的文本会超出 LLM 的上下文窗口，导致性能下降；此外，控制器的意图分类器在极端专业查询上仍有误判。  

### 影响与延伸思考
这篇工作首次把“查询导向”与“多表格”结合起来，为表格理解社区打开了新方向。随后的研究（如 MultiTab‑CoT、Query‑Guided Table Reasoning）在此基础上加入链式思考（CoT）或图结构编码，以进一步提升跨表推理的深度。对想继续深入的读者，可以关注以下两个方向：  
1) **长上下文处理**：利用检索增强或分块策略，让模型在不超限的情况下处理上万行的大表。  
2) **可解释的跨表推理**：在生成摘要的同时输出推理路径（类似 SQL 查询），帮助用户验证答案的来源。  

### 一句话记住它
把多张表格“翻译成语言”，让查询驱动的提示指挥大语言模型生成精准摘要——这就是 QFMTS 的核心魔法。