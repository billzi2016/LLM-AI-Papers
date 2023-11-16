# SUQL: Conversational Search over Structured and Unstructured Data with   Large Language Models

> **Date**：2023-11-16
> **arXiv**：https://arxiv.org/abs/2311.09818

## Abstract

While most conversational agents are grounded on either free-text or structured knowledge, many knowledge corpora consist of hybrid sources. This paper presents the first conversational agent that supports the full generality of hybrid data access for large knowledge corpora, through a language we developed called SUQL (Structured and Unstructured Query Language). Specifically, SUQL extends SQL with free-text primitives (summary and answer), so information retrieval can be composed with structured data accesses arbitrarily in a formal, succinct, precise, and interpretable notation. With SUQL, we propose the first semantic parser, an LLM with in-context learning, that can handle hybrid data sources.   Our in-context learning-based approach, when applied to the HybridQA dataset, comes within 8.9% exact match and 7.1% F1 of the SOTA, which was trained on 62K data samples. More significantly, unlike previous approaches, our technique is applicable to large databases and free-text corpora. We introduce a dataset consisting of crowdsourced questions and conversations on Yelp, a large, real restaurant knowledge base with structured and unstructured data. We show that our few-shot conversational agent based on SUQL finds an entity satisfying all user requirements 90.3% of the time, compared to 63.4% for a baseline based on linearization.

---

# SUQL：基于大语言模型的结构化与非结构化数据对话检索 论文详细解读

### 背景：这个问题为什么难？
传统对话系统要么只能在自由文本（比如网页、文档）里找答案，要么只能在严格的结构化库（比如关系型数据库）里查询。现实中的知识库往往是两者的混合：餐厅点评既有表格化的营业时间、评分，也有用户写的长评。早期方法只能把结构化信息“平铺”成文本再做检索，或者把非结构化内容硬塞进表格，导致查询表达不够精准、解释性差，而且在大规模数据库上成本爆炸。于是，如何让对话代理既能写 SQL 又能直接检索自由文本，成为了一个未被完整解决的难题。

### 关键概念速览
**结构化查询（SQL）**：一种用于关系型数据库的声明式语言，像在电子表格里写公式，只需要说“取哪些列、怎么过滤”。  
**自由文本检索（IR）**：在海量文档中找出与关键词匹配的段落，类似搜索引擎的关键词匹配。  
**SUQL**：在标准 SQL 基础上加入了 `summary` 与 `answer` 两个自由文本原语，使得一条查询可以同时执行表格过滤和全文摘要/问答。  
**大语言模型（LLM）**：像 GPT‑4 那样的深度学习模型，能够理解自然语言并生成代码或查询。  
**Few‑shot In‑context Learning**：不给模型显式微调，只在提示中提供几例输入输出，让模型“看着学”。  
**HybridQA 数据集**：一个公开的基准，包含需要同时查询表格和文档才能回答的问题。  
**Yelp 结构化/非结构化混合库**：作者自行收集的真实餐饮数据，表格字段和用户评论共存，用来评估对话式检索的实战表现。

### 核心创新点
1. **语言层面的统一**：过去的系统要么把结构化查询转成自然语言再交给检索模型，要么把全文检索结果再手工映射回表格。SUQL 直接在语言层面把两者合并，SQL 语句里可以嵌入 `summary(table, condition)` 或 `answer(document, question)`，实现“一条指令搞定”。  
2. **基于 LLM 的少样本语义解析**：不需要上万条标注的训练数据，作者只在提示里给出几条 SUQL 示例，利用大语言模型的上下文学习能力把用户自然语言转成 SUQL。相比传统的端到端训练，这大幅降低了标注成本。  
3. **可扩展到大规模库**：因为解析过程只产生一条 SUQL，后端执行时可以把结构化部分交给高效的关系型引擎，把自由文本部分交给专门的检索系统（如 BM25 或向量搜索），两者各自发挥最擅长的性能，避免了全表全文检索的瓶颈。  
4. **真实业务数据评估**：作者自行构建了 Yelp 对话数据集，展示了在真实、规模可观的混合库上，少样本 SUQL 代理的成功率从基线的 63.4% 提升到 90.3%，证明了方法的实用性。

### 方法详解
整体思路可以拆成三步：**（1）用户意图捕获 →（2）Few‑shot 语义解析 →（3）SUQL 执行与答案合成**。

1. **用户意图捕获**  
   对话系统先把用户的自然语言请求（可能是多轮对话的累计需求）转成一个完整的查询意图。这里不做特殊处理，只是把最近的几轮对话拼接成一个长句，交给后面的解析模块。

2. **Few‑shot 语义解析**  
   - **提示构造**：在 LLM 的输入前部放入 3–5 条手工编写的示例，每条示例包括自然语言问题、对应的 SUQL 代码以及执行结果的简要说明。示例覆盖了纯结构化查询、纯自由文本检索以及两者混合的情况。  
   - **模型调用**：使用 GPT‑4（或同等级的 LLM）进行一次前向推理，模型在看到示例后直接生成目标问题的 SUQL。因为 LLM 已经在海量代码和自然语言对齐上预训练，这一步不需要梯度更新。  
   - **后处理**：对生成的 SUQL 做语法检查（比如确保 SELECT、FROM、WHERE 正确闭合），若出现明显错误则回退到最相似的示例或请求用户澄清。

3. **SUQL 执行与答案合成**  
   - **结构化子句执行**：把 SUQL 中的标准 SQL 部分交给关系型数据库引擎（如 PostgreSQL），得到一组实体 ID 或属性集合。  
   - **自由文本子句执行**：对 `summary` 或 `answer` 原语，系统先根据结构化子句的输出限定检索范围（比如只在这些餐厅的评论里搜索），然后调用检索模型（BM25 + Rerank 或向量检索）得到相关文段，最后用 LLM 对这些文段进行摘要或直接生成答案。  
   - **结果拼接**：将结构化查询结果（如营业时间、价格区间）与自由文本生成的自然语言回答拼接，形成最终的对话回复。整个过程保持了可解释性：用户可以看到哪一步用了 SQL，哪一步用了文本检索。

**最巧妙的点**在于把结构化过滤作为检索的“硬过滤器”，大幅削减了自由文本检索的候选空间，从而在大库上仍然保持低延迟；同时，利用 LLM 的少样本学习能力，省去了昂贵的端到端训练。

### 实验与效果
- **数据集**：在公开的 HybridQA 基准上进行评估；另外自行收集并标注了 Yelp 对话数据集，包含数千条真实用户需求。  
- **对比基线**：HybridQA 上的最强模型是经过 62 K 标注样本训练的端到端系统。SUQL 的 few‑shot 方案在 Exact Match 上仅差 8.9%，在 F1 上差 7.1%，说明少样本方法已经逼近全监督水平。  
- **Yelp 场景**：使用线性化（把结构化表格拼成长文本）的方法作为基线，成功找到满足全部用户约束的实体的比例为 63.4%；SUQL 代理提升到 90.3%。  
- **消融实验**：论文展示了去掉结构化过滤、仅使用全文检索或仅使用 SQL 的两种极端配置，性能分别大幅下降，验证了两种子查询的互补性。  
- **局限性**：作者指出 SUQL 仍依赖于 LLM 的生成质量，极端长对话或高度歧义的需求仍可能导致解析错误；此外，当前实现只支持单表或单文档检索，跨表复杂联结仍是开放问题。

### 影响与延伸思考
这篇工作打开了“结构化+非结构化”统一查询的思路，随后出现的几篇论文（如 **HybridSQL**, **CoSQL‑Fusion**）都在尝试把自然语言直接映射到混合查询语言，或在提示工程上进一步压缩示例数量。对想继续深入的读者，可以关注以下方向：  
- **自动化提示生成**：让系统自行发现最具代表性的 few‑shot 示例，进一步降低人工成本。  
- **多表/图谱扩展**：把 SUQL 的原语推广到多表联结或图数据库查询。  
- **检索-生成闭环**：在自由文本子句中加入 LLM 的自我校验，使答案更可靠。  
- **实时对话优化**：结合对话状态追踪，让 SUQL 能够在多轮对话中增量修改查询。

### 一句话记住它
SUQL 用一条扩展了自由文本原语的 SQL，让大语言模型在几例提示下即可在结构化表格和海量文档之间自由跳转，实现高效、可解释的混合数据对话检索。