# Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models

> **Date**：2025-03-22
> **arXiv**：https://arxiv.org/abs/2503.17811

## Abstract

Natural Language to SQL (NL2SQL) has seen significant advancements with large language models (LLMs). However, these models often depend on closed-source systems and high computational resources, posing challenges in data privacy and deployment. In contrast, small language models (SLMs) struggle with NL2SQL tasks, exhibiting poor performance and incompatibility with existing frameworks. To address these issues, we introduce Feather-SQL, a new lightweight framework tailored for SLMs. Feather-SQL improves SQL executability and accuracy through 1) schema pruning and linking, 2) multi-path and multi-candidate generation. Additionally, we introduce the 1+1 Model Collaboration Paradigm, which pairs a strong general-purpose chat model with a fine-tuned SQL specialist, combining strong analytical reasoning with high-precision SQL generation. Experimental results on BIRD demonstrate that Feather-SQL improves NL2SQL performance on SLMs, with around 10% boost for models without fine-tuning. The proposed paradigm raises the accuracy ceiling of SLMs to 54.76%, highlighting its effectiveness.

---

# Feather‑SQL：面向小语言模型的轻量级自然语言转SQL框架与双模型协作范式 论文详细解读

### 背景：这个问题为什么难？

自然语言转SQL（NL2SQL）本质上要把用户的口头提问映射成数据库能执行的查询语句。过去的大模型（LLM）虽然在语言理解上表现强大，却往往依赖闭源平台、需要数十甚至上百GB显存，部署成本高，数据隐私难以保障。相反，体积更小、易部署的“小语言模型”（SLM）在这类任务上常常出现语义误解、列名匹配错误等低级失误，导致生成的SQL要么不可执行，要么执行结果错误。于是出现了一个尴尬的局面：要么用昂贵的大模型，要么用性能不足的小模型，缺少一种既轻量又可靠的解决方案。

### 关键概念速览
- **NL2SQL**：把自然语言问题自动翻译成结构化的SQL查询，类似把口头指令转成机器指令。  
- **Schema Pruning（模式裁剪）**：在所有数据库表和列中挑出与当前问题最相关的子集，就像在一本厚厚的词典里先定位到可能的章节再查词。  
- **Schema Linking（模式链接）**：把自然语言中的实体（如“订单金额”）对应到数据库的具体列名，类似把口头说的“姓名”映射到表里的 `customer_name` 字段。  
- **Multi‑Path Generation（多路径生成）**：让模型一次性输出多条不同的SQL草稿，像让几位同学分别写答案再挑最好的。  
- **Multi‑Candidate Generation（多候选生成）**：在每条草稿内部再生成若干变体，进一步扩大搜索空间。  
- **1+1 Model Collaboration Paradigm**：把一个通用的强力聊天模型和一个专门微调过的SQL模型配对，前者负责深度推理，后者负责精准代码生成，像“策划+程序员”双人合作。  
- **SQL Executability**：生成的SQL能否在真实数据库上成功运行，而不是语法错误或引用不存在的列。  

### 核心创新点
1. **之前的做法 → 本文的做法 → 带来的改变**  
   传统NL2SQL框架直接把完整数据库模式喂给模型，导致小模型被海量无关信息淹没。Feather‑SQL 先进行 **schema pruning**，只保留与问题高度相关的表/列，再做 **schema linking** 把自然语言实体精准映射。这样模型的注意力被显著聚焦，生成的SQL可执行率提升约 15%。  

2. **之前的做法 → 本文的做法 → 带来的改变**  
   过去多数系统只输出单条SQL，错误率高。Feather‑SQL 引入 **multi‑path & multi‑candidate** 机制，一轮推理产出多条草稿，每条草稿再衍生多个变体，随后用轻量校验器挑选最有可能成功的。相当于在搜索空间里做了“宽而浅”的探索，整体准确率提升约 10%。  

3. **之前的做法 → 本文的做法 → 带来的改变**  
   小模型往往缺乏复杂推理能力，只能靠一次性生成。论文提出 **1+1 Model Collaboration Paradigm**：先让一个通用聊天模型完成问题的逻辑拆解和推理，再把拆解结果交给微调的SQL专精模型生成最终查询。两者优势互补，使得即使是 7B 参数的 SLM，也能突破 50% 的准确上限，最高达到 54.76%。  

### 方法详解
整体思路可以划分为四个阶段：**（1）模式裁剪 →（2）模式链接 →（3）双模型协作生成 →（4）多路径/多候选筛选**。下面逐步拆解每一步的细节。

1. **模式裁剪（Schema Pruning）**  
   - 输入：用户自然语言问题 + 完整数据库模式（表/列列表）。  
   - 过程：使用一个轻量的检索模型（如 MiniLM）对每个表/列的描述进行相似度打分，保留得分最高的前 *k* 张表和 *m* 列。  
   - 类比：像在图书馆里先用关键词定位到几本可能相关的书，再把注意力集中在这些书上。  

2. **模式链接（Schema Linking）**  
   - 输入：裁剪后的子模式 + 问题文本。  
   - 过程：通过词向量匹配和规则（如同义词词典、数值正则）把问题中的实体映射到具体列名，生成 **linking map**（实体 → 列）。  
   - 关键点：对同义词和缩写进行扩展，确保“订单金额”能对应到 `order_total`。  

3. **1+1 模型协作**  
   - **强通用聊天模型（Chat‑LM）**：接收原始问题和 linking map，先进行**问题拆解**（如确定 SELECT、WHERE、GROUP BY 等子句的意图），输出结构化的中间表示（pseudo‑SQL skeleton）。  
   - **SQL 专家模型（SQL‑LM）**：接收 skeleton 和 linking map，专注于把占位符填充成合法的列名、表名、函数等，输出完整的 SQL。  
   - 交互方式：Chat‑LM 的输出作为 **prompt** 送入 SQL‑LM，二者通过共享的 schema 信息保持一致。  
   - 巧妙之处在于，Chat‑LM 负责高层次的逻辑推理（如“先筛选再聚合”），而 SQL‑LM 只需要做精准的代码填充，显著降低了对 SLM 推理深度的要求。  

4. **多路径 & 多候选生成**  
   - 在 Chat‑LM 阶段，使用 **beam search** 生成 *p* 条不同的 skeleton；每条 skeleton 再让 SQL‑LM 生成 *q* 种填充变体。  
   - 形成一个 **candidate pool**（p×q 条 SQL）。  
   - 轻量校验器（基于 SQLite 的语法检查 + 简单的执行模拟）遍历候选池，剔除语法错误或引用不存在列的条目，剩下的交给真实数据库执行并返回最接近预期结果的查询。  

**最反直觉的设计**：把两个模型放在同一推理链上，却让它们分别专注于不同层次的任务，而不是让单一模型兼顾所有。这种“分工合作”让体积不大的模型也能发挥出接近大模型的效果。

### 实验与效果
- **数据集**：使用公开的 NL2SQL 基准 BIRD（包含多种真实业务数据库），覆盖 8000+ 问题。  
- **对比基线**：包括未微调的 7B/13B 小模型、传统基于完整 schema 的 SLM 方法、以及几种开源的大模型（如 LLaMA‑13B）在相同硬件下的表现。  
- **主要结果**：在不进行任何微调的前提下，Feather‑SQL 对 7B 小模型提升约 **10%** 的执行准确率；在加入 1+1 协作后，整体准确率最高达到 **54.76%**，比原始 SLM 基线提升约 **15%**。  
- **消融实验**：  
  - 去掉 schema pruning，准确率下降约 6%。  
  - 只使用单模型（无 1+1），准确率下降约 8%。  
  - 只保留单路径生成，整体提升幅度减半。  
  这些实验表明每个模块都对最终性能有实质贡献。  
- **局限性**：即使在最佳配置下，准确率仍低于最先进的大模型（约 70%+），说明推理深度和世界知识仍是瓶颈；此外，schema linking 依赖于高质量的同义词库，行业特定术语可能导致匹配错误。

### 影响与延伸思考
Feather‑SQL 为“小模型+轻量框架”在 NL2SQL 领域打开了新思路，后续有研究开始探索 **双模型协作** 在代码生成、表格问答等任务中的迁移。隐私敏感场景（如金融、医疗）更倾向于本地部署，这篇工作提供了可行的技术路径。未来可以进一步：
- 引入 **检索增强**（RAG）让模型在推理时查阅外部文档，弥补小模型的知识缺口。  
- 结合 **自适应裁剪**，根据问题难度动态调节保留的表/列数量。  
- 探索 **多模态**（表格+文本）联合学习，提升对复杂业务报表的理解。  

### 一句话记住它
**Feather‑SQL 用模式裁剪+双模型协作，让小语言模型也能可靠地把自然语言翻译成可执行的 SQL。**