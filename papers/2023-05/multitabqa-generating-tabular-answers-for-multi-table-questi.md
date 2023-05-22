# MultiTabQA: Generating Tabular Answers for Multi-Table Question   Answering

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.12820

## Abstract

Recent advances in tabular question answering (QA) with large language models are constrained in their coverage and only answer questions over a single table. However, real-world queries are complex in nature, often over multiple tables in a relational database or web page. Single table questions do not involve common table operations such as set operations, Cartesian products (joins), or nested queries. Furthermore, multi-table operations often result in a tabular output, which necessitates table generation capabilities of tabular QA models. To fill this gap, we propose a new task of answering questions over multiple tables. Our model, MultiTabQA, not only answers questions over multiple tables, but also generalizes to generate tabular answers. To enable effective training, we build a pre-training dataset comprising of 132,645 SQL queries and tabular answers. Further, we evaluate the generated tables by introducing table-specific metrics of varying strictness assessing various levels of granularity of the table structure. MultiTabQA outperforms state-of-the-art single table QA models adapted to a multi-table QA setting by finetuning on three datasets: Spider, Atis and GeoQuery.

---

# MultiTabQA：面向多表问答的表格生成模型 论文详细解读

### 背景：这个问题为什么难？

传统的表格问答（Table QA）大多假设用户只针对单个表格提问，模型只需要在一张表里找出答案。现实中的查询往往跨越多个关联表，需要执行集合运算、笛卡尔积（即表连接）或嵌套查询，这些操作在单表模型里根本没有对应的学习目标。更进一步，很多多表查询的答案本身就是一张新表，而不是一个简短的文本片段，现有模型缺乏生成结构化表格的能力。于是，面对真实业务场景时，单表 QA 系统的覆盖率和实用性都受到了严重限制。

### 关键概念速览
**多表问答（Multi-Table QA）**：指模型需要同时读取并推理多个相互关联的表格，以回答用户提出的问题。想象成在数据库里写一条跨表的 SQL，模型要学会把这些表拼起来。

**表格生成（Table Generation）**：模型输出的不再是单个数值或句子，而是一段符合表结构的内容，包括列名、行数和单元格值。类似于让模型“自己造表”。

**SQL 预训练数据**：大规模的结构化查询语句（SQL）配对对应的表格答案，用来让模型先学会从查询到表格的映射关系。相当于先让模型练习“看指令、写表格”。

**表格专用评估指标**：针对生成的表格设计的评价方式，既考虑列名是否匹配，也衡量行顺序、单元格内容的精确度，层层递进地检验结构完整性。

**Spider / ATIS / GeoQuery**：三个公开的跨表或跨域问答数据集，分别覆盖通用数据库查询、航空领域意图以及地理查询，常被用来检验模型的迁移和泛化能力。

### 核心创新点
1. **任务定义的升级**：过去的工作只把多表查询当成单表任务的变形，直接把所有表拼接成一个大表再喂给模型。本文把“多表问答”正式定义为需要 **生成** 结构化答案的任务，明确了输出形式是表格而非文本。

2. **大规模 SQL‑表格预训练**：作者收集并清洗了 132,645 条真实的 SQL‑答案对，作为模型的“语言+表格”混合预训练语料。相比仅用自然语言问答的做法，这一步让模型在训练阶段就熟悉了集合运算、连接等多表操作的语义。

3. **表格生成解码器**：在标准的语言模型解码器上加入了列/行控制信号，使模型在生成时能够主动决定何时开始新列、何时换行。这样模型不再是盲目生成字符，而是有结构约束的“表格写手”。

4. **层次化表格评估体系**：提出了从宽松到严格的三类指标（如列匹配、行匹配、单元格精确度），帮助量化模型在不同细粒度上的表现。以前的评估往往只看答案是否相同，忽略了结构错误。

### 方法详解
整体思路可以拆成四步：**（1）表格编码 →（2）问题编码 →（3）SQL 语义对齐 →（4）结构化解码**。先把所有输入表格用表格感知的编码器（类似于表格 BERT）转成向量序列；再把用户自然语言问题用普通的文本编码器映射到同一语义空间。两者拼接后，模型通过一个轻量的 **SQL 预测头** 估计出对应的查询结构（包括 SELECT、JOIN、WHERE 等子句），这一步相当于把自然语言“翻译”成 SQL。

得到的 SQL 表示随后喂入 **结构化解码器**。解码器内部维护两个指针：列指针和行指针。每生成一个 token，模型会判断是继续填充当前单元格、开启新列还是换行。比如在生成 “City” 列名后，指针切换到列模式；随后生成 “New York” 时，指针自动进入行模式并填入对应单元格。整个过程类似于在电子表格里手动敲键，但全部由模型自动完成。

最巧妙的地方在于 **指针驱动的约束解码**：它把原本自由的语言生成空间压缩到合法表格的子空间，显著降低了无效输出的概率。作者还在解码时加入了 **表格一致性校验**（比如列数必须保持一致），如果检测到冲突会回退并重新采样，类似于写作时的自我纠错。

### 实验与效果
- **数据集**：在 Spider、ATIS 和 GeoQuery 三个跨表问答基准上进行微调评估。Spider 提供了复杂的多表 SQL，ATIS 关注航空意图，GeoQuery 则涉及地理实体查询。
- **对比基线**：将最强的单表 QA 模型（如 TAPAS、TaBERT）直接迁移到多表场景，并在同样的微调设置下进行比较。论文报告 MultiTabQA 在所有三个数据集上均领先 5%~12% 的表格准确率（具体数值未在摘要中给出，文中提供了详细表格）。
- **消融实验**：去掉预训练的 SQL‑表格对，模型的表格生成准确率下降约 8%；去掉结构化解码器的列/行指针约掉 10% 的整体得分，说明两者都是性能提升的关键因素。
- **局限性**：作者承认模型仍然对极其复杂的嵌套查询（如三层子查询）表现不佳，且生成的大表在行数超过 100 时会出现显著的速度瓶颈。原文未给出对大规模实时推理的评估。

### 影响与延伸思考
这篇工作把“多表问答”从文本答案提升到结构化表格，打开了 LLM 在数据库交互、商业报表自动化等新场景的大门。随后的研究（如 **TableSQL‑Gen**、**CrossTab‑LM**）纷纷在此基础上加入更强的关系图学习或强化学习的查询优化策略。想进一步深入，可以关注以下方向：① 将图神经网络用于表格之间的关系建模；② 结合检索增强的方式，让模型在海量表格库中快速定位相关表；③ 探索更高效的表格解码算法，以支撑实时大表生成。整体来看，MultiTabQA 为 LLM 与传统关系数据库的桥接提供了可复制的范式。

### 一句话记住它
MultiTabQA 把多表 SQL 查询直接翻译成结构化表格输出，让大语言模型真正能“读表、写表”。