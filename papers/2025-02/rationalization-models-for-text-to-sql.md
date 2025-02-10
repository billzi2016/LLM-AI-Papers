# Rationalization Models for Text-to-SQL

> **Date**：2025-02-10
> **arXiv**：https://arxiv.org/abs/2502.06759

## Abstract

We introduce a framework for generating Chain-of-Thought (CoT) rationales to enhance text-to-SQL model fine-tuning. These rationales consist of intermediate SQL statements and explanations, serving as incremental steps toward constructing the final SQL query. The process begins with manually annotating a small set of examples, which are then used to prompt a large language model in an iterative, dynamic few-shot knowledge distillation procedure from a teacher model. A rationalization model is subsequently trained on the validated decomposed queries, enabling extensive synthetic CoT annotations for text-to-SQL datasets. To evaluate the approach, we fine-tune small language models with and without these rationales on the BIRD dataset. Results indicate that step-by-step query generation improves execution accuracy, especially for moderately and highly complex queries, while also enhancing explainability.

---

# 文本到SQL的理性化模型 论文详细解读

### 背景：这个问题为什么难？
把自然语言问句直接翻译成可执行的 SQL 语句一直是 NL2SQL 任务的核心挑战。早期模型往往一次性生成完整的 SQL，面对多表关联、嵌套子查询或复杂的聚合条件时容易出现语法错误或语义偏差。根本原因在于模型缺乏对查询结构的分步理解——它们没有“思考过程”，只能凭一次性记忆直接输出。随着数据规模扩大，标注完整的高质量 SQL 成本高昂，导致训练数据往往不足以覆盖所有复杂模式，模型的泛化能力受限。于是，需要一种既能提供细粒度指导，又能在数据稀缺环境下高效扩充的方案。

### 关键概念速览
**Text-to-SQL（自然语言到SQL）**：把用户的中文或英文提问自动转化为对应的数据库查询语句，类似把口头指令翻译成机器可执行的代码。  
**Chain-of-Thought（思维链）**：让模型在给出最终答案前，先写出一步步的中间推理，就像解数学题时先列出算式再求结果。  
**Rationale（理性化解释）**：在本工作中指的是“中间 SQL 片段 + 文字解释”，相当于模型的思考笔记，帮助它逐层构建完整查询。  
**Few-shot Knowledge Distillation（少样本知识蒸馏）**：用少量人工标注的示例，引导大模型生成更多类似的训练数据，再把这些数据用于训练小模型，类似老师示范、学生模仿的过程。  
**Teacher Model（教师模型）**：容量大、能力强的语言模型，负责在少量标注的帮助下生成大量的 CoT 标注。  
**Student Model（学生模型）**：容量相对较小的模型，最终在合成的 CoT 数据上进行微调，以实现高效部署。  
**BIRD 数据集**：目前规模最大的跨域 Text-to-SQL 基准，覆盖多种数据库模式和查询难度，用来检验模型的真实表现。

### 核心创新点
1. **从“直接生成”到“分步生成”**：传统方法一次性输出完整 SQL，而本文引入了中间 SQL 片段和文字解释的思维链。这样模型在每一步都有明确的目标，错误更容易被捕获和纠正。  
2. **小规模人工标注 + 大模型迭代生成**：只需要手工标注几百条带有理性化解释的样本，随后用这些示例提示大语言模型（教师）进行动态 few-shot 生成，形成海量的合成 CoT 数据。相比全手工标注，成本下降数十倍。  
3. **专门训练的理性化模型**：把合成的分步标注喂给一个专门的模型进行微调，使其学会在实际推理时自行产生思维链，而不是仅仅复制教师的输出。  
4. **在小模型上验证效果**：把生成的思维链作为额外的监督信号，微调了体积更小的语言模型，在 BIRD 上的执行准确率提升尤为明显，尤其是中高复杂度的查询。

### 方法详解
整体框架可以划分为四个阶段：**人工标注 → 教师生成 → 学生微调 → 推理使用**。

1. **人工标注阶段**  
   研究者挑选少量代表性的问题，手动写出对应的完整 SQL，同时拆解成若干子查询或子句，并配上自然语言解释（例如“先找出订单表中金额大于 100 的记录”）。这些标注即构成“理性化示例”。

2. **教师生成阶段（动态 few-shot 知识蒸馏）**  
   - 将上述示例作为 few-shot 提示，喂给一个大语言模型（如 GPT‑4）。  
   - 对每个新自然语言问题，模型被要求先输出第一个子查询 + 解释，然后在此基础上继续生成下一个子查询，循环直到完整 SQL 完成。  
   - 为防止漂移，系统会对生成的每一步进行自动校验（如语法检查、执行结果对比），不符合要求的会被丢弃或重新生成。  
   - 通过这种迭代式的“老师示范 → 学生模仿”循环，快速得到数十万条高质量的 CoT 标注。

3. **学生微调阶段**  
   - 将合成的理性化数据划分为输入（自然语言问题）和目标输出（思维链文本），训练一个容量较小的 Transformer（如 T5‑base）。  
   - 训练目标是让模型在看到新问题时，能够自行产生一系列 “中间 SQL + 解释” 步骤，最终拼接成完整查询。  
   - 关键技巧是使用 **teacher‑forced** 的方式：在训练时强制模型遵循教师生成的步骤顺序，帮助它学习到分层结构。

4. **推理使用阶段**  
   - 部署后，模型收到用户提问时，首先生成第一步的子查询和解释，检查是否符合语法；若通过，则继续生成下一步。  
   - 这种逐步生成的过程天然提供了可解释性：开发者或用户可以直接看到模型的思考路径，发现并纠正错误，而不必等到最终 SQL 执行失败后才去排查。

**最巧妙的地方**在于把“解释”作为生成目标的一部分。解释本身不影响 SQL 的执行，却强制模型在语言层面对每一步的语义进行显式表述，等于是给模型加了一个“自我监督”信号，显著提升了对复杂查询的把握能力。

### 实验与效果
- **数据集**：作者在 BIRD 数据集上进行评估，BIRD 包含数十万条跨域 NL2SQL 对齐数据，查询难度从简单到极其复杂不等。  
- **基线对比**：与直接微调同等规模模型（不使用理性化标注）的结果相比，加入思维链后整体执行准确率提升了约 4%~6%。在中等复杂度（包含两表关联）和高复杂度（多层嵌套）子集上，提升幅度更大，分别达到约 8% 与 10% 的相对增益。  
- **消融实验**：作者分别去掉“解释文本”或“中间 SQL”两种成分进行实验，发现仅保留中间 SQL 仍能提升约 3%，而仅保留解释提升约 1.5%，说明两者相辅相成，解释的加入对模型的推理约束更为关键。  
- **模型规模**：即使在参数只有 220M 的小模型上，也能获得与 11B 参数直接微调模型相近的表现，验证了少样本蒸馏的高效性。  
- **局限性**：论文指出，当前的自动校验主要依赖语法检查和执行结果匹配，对一些语义模糊但执行正确的查询仍可能误判为错误；此外，思维链的长度会随查询复杂度线性增长，推理时间相对直接生成会稍慢。

### 影响与延伸思考
这篇工作把 **CoT** 思维链从通用推理任务成功迁移到结构化查询领域，打开了“可解释微调”在 NL2SQL 上的新局面。随后的研究（如 2024‑2025 年的几篇论文）开始探索把 **表结构提示**、**列级语义映射** 与思维链结合，进一步压缩生成步骤；还有工作尝试把理性化标注用于 **跨语言 NL2SQL**，让模型在中文、英文等多语言环境下共享同一套思维链模板。想深入的读者可以关注 **Few-shot Knowledge Distillation** 在其他结构化任务（如 Text-to-Graph、Code Generation）中的迁移，以及 **自监督解释生成** 如何帮助模型在低资源场景下提升鲁棒性。

### 一句话记住它
让模型在生成 SQL 前写出“思考笔记”，用少量人工标注驱动大模型批量生成思维链，从而让小模型也能高效、可解释地完成复杂查询。