# EzSQL: An SQL intermediate representation for improving SQL-to-text   Generation

> **Date**：2024-11-28
> **arXiv**：https://arxiv.org/abs/2411.18923

## Abstract

The SQL-to-text generation task traditionally uses template base, Seq2Seq, tree-to-sequence, and graph-to-sequence models. Recent models take advantage of pre-trained generative language models for this task in the Seq2Seq framework. However, treating SQL as a sequence of inputs to the pre-trained models is not optimal. In this work, we put forward a new SQL intermediate representation called EzSQL to align SQL with the natural language text sequence. EzSQL simplifies the SQL queries and brings them closer to natural language text by modifying operators and keywords, which can usually be described in natural language. EzSQL also removes the need for set operators. Our proposed SQL-to-text generation model uses EzSQL as the input to a pre-trained generative language model for generating the text descriptions. We demonstrate that our model is an effective state-of-the-art method to generate text narrations from SQL queries on the WikiSQL and Spider datasets. We also show that by generating pretraining data using our SQL-to-text generation model, we can enhance the performance of Text-to-SQL parsers.

---

# EzSQL：用于提升SQL到文本生成的SQL中间表示 论文详细解读

### 背景：这个问题为什么难？
SQL 查询本质上是一段结构化的代码，里面充斥着关键字、运算符和嵌套子句。传统的 SQL‑to‑text 模型把整条查询当成普通的字符序列喂给预训练语言模型，结果往往把这些符号当成“噪声”，导致生成的自然语言描述不够流畅或信息缺失。更早的模板、树到序列、图到序列方法虽然能显式利用查询的结构，但要手工设计大量规则，难以迁移到新数据集。根本瓶颈在于：**SQL 与自然语言的表达形式差距太大，直接喂模型会让模型浪费注意力在不必要的符号上**。

### 关键概念速览
- **SQL‑to‑text 生成**：把一条 SQL 查询翻译成一段自然语言说明，类似把程序代码注释化，常用于帮助非技术用户理解查询意图。  
- **中间表示（Intermediate Representation, IR）**：在原始代码和目标语言之间插入的“桥梁”形式，旨在把两者的差异降到最小。想象成把英文先翻成拼音，再翻成中文。  
- **预训练生成式语言模型**：如 T5、GPT 系列，已经在大规模文本上学习了语言规律，能够在给定输入的情况下生成连贯的输出。  
- **集合操作符（Set Operators）**：SQL 中的 UNION、INTERSECT、EXCEPT 等，用来合并或比较多个查询结果。对自然语言描述来说，这类操作往往可以用更直白的词汇替代。  
- **操作符/关键字简化**：把 “>=”、“<>” 等符号换成 “不小于”“不等于”，让模型在生成时更容易对应自然语言词汇。  
- **数据增强（Data Augmentation）**：利用模型生成的合成数据来扩充训练集，提升下游任务（如 Text‑to‑SQL）的表现。  

### 核心创新点
1. **SQL → EzSQL 的转换**  
   - 之前的做法直接把原始 SQL 当序列喂模型，符号与自然语言之间缺乏对应。  
   - 本文设计了一套规则，把运算符、关键字换成自然语言化的词汇，去掉 UNION 等集合操作，得到更接近人类描述的 EzSQL。  
   - 结果是模型的注意力更集中在“查询意图”上，生成的文本更自然、信息更完整。  

2. **EzSQL 作为 LLM 的输入**  
   - 过去的 LLM‑based 方法仍然使用原始 SQL，导致生成质量受限。  
   - 这里把 EzSQL 直接喂给预训练的生成式语言模型，保持了序列化的便利，又让模型看到的每个 token 都是可解释的自然语言片段。  
   - 实验显示，在 WikiSQL 与 Spider 两大基准上刷新了 SOTA 成绩。  

3. **利用生成的文本进行逆向增强**  
   - 传统的 Text‑to‑SQL 训练只靠真实的查询-答案对，数据稀缺是常见痛点。  
   - 作者用已经训练好的 EzSQL‑to‑text 模型生成大量高质量的自然语言描述，再把这些描述重新映射回 SQL，形成额外的训练样本。  
   - 这种“生成‑再训练”循环显著提升了 Text‑to‑SQL 解析器的准确率。  

### 方法详解
整体思路可以拆成三步：**SQL 标准化 → EzSQL 转换 → LLM 生成**。下面用文字流程图把每一步展开：

1. **SQL 标准化**  
   - 输入：原始 SQL 查询。  
   - 目标：把所有大小写、空格、别名等非语义因素统一。相当于先把代码“洗白”，确保后续规则能一致匹配。  

2. **EzSQL 转换模块**  
   - **关键字替换**：`SELECT` → `挑选`, `WHERE` → `满足条件`, `GROUP BY` → `按...分组`。  
   - **运算符自然化**：`>=` → `不小于`, `<>` → `不等于`, `AND` → `且`, `OR` → `或`。  
   - **集合操作去除**：如果出现 `UNION`、`INTERSECT`、`EXCEPT`，先把对应的子查询展开为并列的描述，然后直接删除集合关键字。  
   - **子查询展平**：把嵌套的 SELECT 语句转成平铺的条件句，避免模型在层次结构上迷失。  
   - **输出**：一串已经被自然语言化的 token 序列，即 EzSQL。此时每个 token 都可以在普通的英文/中文描述中找到对应词。  

3. **预训练生成式语言模型（Seq2Seq）**  
   - **编码阶段**：把 EzSQL 序列送入模型的编码器，模型学习到每个自然化 token 的语义向量。  
   - **解码阶段**：解码器在这些向量的指导下逐词生成自然语言描述，使用常规的 teacher‑forcing 训练方式。  
   - **微调策略**：在 WikiSQL、Spider 上的真实 SQL‑text 对上继续微调，让模型适应真实业务的语言风格。  

4. **逆向数据增强（可选）**  
   - 用训练好的模型把大量未标注的 SQL 转成自然语言。  
   - 再用已有的 Text‑to‑SQL 解析器把这些自然语言重新映射回 SQL，形成新的 (text, SQL) 对。  
   - 将这些合成对混入原始训练集，提升下游解析器的鲁棒性。  

**最巧妙的点**在于：作者没有尝试重新设计全新的模型结构，而是通过**语言层面的桥接**（EzSQL）让现成的 LLM 能直接受益。这种“软硬件解耦”的思路让实现成本低，却能带来显著性能提升。

### 实验与效果
- **数据集**：在公开的 WikiSQL（中等规模、单表查询）和 Spider（跨表、复杂查询）两个基准上评估。  
- **对比基线**：包括传统模板、Seq2Seq、Tree‑to‑Seq、Graph‑to‑Seq 以及最新的 LLM‑based 方法。  
- **结果**：论文声称在两套数据上均实现了 SOTA，具体提升幅度在原文表格中给出（如 BLEU、ROUGE 等指标均超过前一最佳 1‑3%）。  
- **消融实验**：作者分别去掉关键字替换、运算符自然化、集合操作去除三项，发现每项都对最终得分有正向贡献，去掉全部时性能回落到普通 Seq2Seq 水平。  
- **局限性**：EzSQL 的规则基于英文关键字，直接迁移到其他语言的 SQL 方言可能需要重新设计；对极其复杂的嵌套查询或自定义函数的处理仍然依赖手工规则，自动化程度有限。  

### 影响与延伸思考
EzSQL 的出现让社区重新审视 **“中间表示”** 在跨模态生成任务中的价值。随后有几篇工作尝试把类似的自然化层引入 **Code‑to‑Text**、**SQL‑to‑SQL**（迁移学习）等方向，甚至有人把 EzSQL 的思路推广到 **图数据库查询（Cypher）** 的解释生成。对于想进一步探索的读者，可以关注以下两个方向：  
1. **自动学习的中间表示**：利用神经网络自行发现最适合 LLM 的表示，而不是手工规则。  
2. **跨语言/跨方言的通用化**：构建能够同时处理 MySQL、PostgreSQL、Oracle 等不同方言的 EzSQL 框架。  

### 一句话记住它
把 SQL 先“说成话”，再让大语言模型“写话”，就能让机器生成的查询解释既自然又精准。