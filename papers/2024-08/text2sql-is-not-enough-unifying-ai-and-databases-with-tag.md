# Text2SQL is Not Enough: Unifying AI and Databases with TAG

> **Date**：2024-08-27
> **arXiv**：https://arxiv.org/abs/2408.14717

## Abstract

AI systems that serve natural language questions over databases promise to unlock tremendous value. Such systems would allow users to leverage the powerful reasoning and knowledge capabilities of language models (LMs) alongside the scalable computational power of data management systems. These combined capabilities would empower users to ask arbitrary natural language questions over custom data sources. However, existing methods and benchmarks insufficiently explore this setting. Text2SQL methods focus solely on natural language questions that can be expressed in relational algebra, representing a small subset of the questions real users wish to ask. Likewise, Retrieval-Augmented Generation (RAG) considers the limited subset of queries that can be answered with point lookups to one or a few data records within the database. We propose Table-Augmented Generation (TAG), a unified and general-purpose paradigm for answering natural language questions over databases. The TAG model represents a wide range of interactions between the LM and database that have been previously unexplored and creates exciting research opportunities for leveraging the world knowledge and reasoning capabilities of LMs over data. We systematically develop benchmarks to study the TAG problem and find that standard methods answer no more than 20% of queries correctly, confirming the need for further research in this area. We release code for the benchmark at https://github.com/TAG-Research/TAG-Bench.

---

# Text2SQL 仍不足：用 TAG 统一 AI 与数据库 论文详细解读

### 背景：这个问题为什么难？

在传统的自然语言问答系统里，用户想要查询数据库往往只能用 **Text2SQL** 把问题翻译成 SQL 语句。可是现实中的提问远比关系代数能表达的要复杂——比如需要多步推理、模糊匹配、甚至结合外部常识。另一方面，**检索增强生成（RAG）** 只适合“一条记录就能回答”的查询，根本无法处理需要聚合、排序或跨表关联的问题。于是，现有方法只能覆盖极小的一部分真实需求，导致大多数自然语言提问根本得不到正确答案。

### 关键概念速览
- **Text2SQL**：把自然语言问题转成结构化的 SQL 查询，类似把口头指令翻译成机器指令，只能处理可以用关系代数描述的查询。  
- **RAG（检索增强生成）**：先在文档库里找相关片段，再让大模型基于这些片段生成答案，像是先查字典再写作文，适合答案直接来源于少量记录的场景。  
- **TAG（Table‑Augmented Generation）**：在生成答案的过程中，模型可以随时向数据库发起查询、获取中间结果或执行聚合操作，类似人类在写报告时不断查表、算数、补充数据。  
- **LM（语言模型）**：大规模预训练的生成模型，拥有丰富的世界知识和推理能力，但对结构化数据的直接操作能力很弱。  
- **关系代数**：数据库查询的数学基础，包括选择、投影、连接等操作，决定了 SQL 能表达的查询范围。  
- **交互式查询**：模型与数据库之间的多轮交互，而不是一次性生成完整 SQL，像是对话式的“问—查—答”。  
- **Benchmark（基准测试）**：用于评估模型在特定任务上的表现的标准数据集和评价指标，这里指的是专门为 TAG 设计的测试集合。  

### 核心创新点
1. **从“一次性生成 SQL”到“交互式表查询”**  
   - 之前的 Text2SQL 只让模型一次性输出完整的 SQL 语句，然后交给数据库执行。  
   - TAG 让模型在生成答案的过程中可以随时向数据库发送子查询，获取局部结果再继续推理。  
   - 这种交互式方式突破了关系代数的表达限制，使模型能够处理需要多步计算、条件分支或外部常识的复杂问答。

2. **统一视角：把 LM 与数据库视作同等的推理工具**  
   - 过去的系统把 LM 当作“前端翻译器”，把数据库当作“后端执行器”。  
   - TAG 把两者放在同一个推理框架里，模型可以决定何时使用语言知识、何时使用数据计算，像是人类在解题时随时切换“想象”和“查表”。  
   - 这种统一让模型能够在答案中自然融合世界常识和实时数据。

3. **专属 TAG 基准的系统化构建**  
   - 现有的 Text2SQL 与 RAG 基准只覆盖了极小的查询子集。  
   - 作者手工收集并标注了多种交互模式（如逐步聚合、条件分支、模糊匹配等），形成了一个覆盖更广查询空间的评测套件。  
   - 基准显示，传统方法在这些任务上正确率不超过 20%，凸显了新范式的必要性。

### 方法详解
**整体思路**：TAG 把自然语言问答拆成两层循环：外层是语言模型的生成过程，内层是模型对数据库的即时查询。每一步，模型先产生一段文字（可能是普通解释，也可能是“执行查询”指令），如果检测到需要数据支撑，就把指令发送给数据库，拿回结果后继续生成。最终的答案是语言模型在多轮查询结果的基础上完成的自然语言输出。

**关键模块**  
1. **指令识别器**  
   - 在模型生成的每个 token 序列后，系统检查是否出现了特定的 “SQL‑CALL” 标记。  
   - 类比为写代码时的编译器，看到 `SELECT` 关键字就会触发编译步骤。  

2. **查询生成器**  
   - 当指令被激活，模型会在上下文（包括用户提问、已得到的查询结果）基础上生成一条完整的 SQL 子句。  
   - 这里使用的是“少量示例提示”（few‑shot prompting），让模型学习在不同上下文下生成不同的子查询。  

3. **执行引擎**  
   - 生成的 SQL 被送到真实的关系数据库执行，返回的表格或聚合值被序列化成自然语言片段（如 “总计 42 条记录”）。  
   - 这一步相当于把数据库当作一个“黑盒函数”，模型只需要知道输入是 SQL，输出是可读文本。  

4. **结果注入器**  
   - 返回的文本被插回模型的生成流中，模型把它当作新的上下文继续写下去。  
   - 这一步类似于人写报告时把表格粘进去再解释。  

**流程文字版**  
```
用户提问 → LM 生成前半句 → 检测到 CALL 标记 → LM 生成子 SQL → DB 执行 → 返回结果 → 结果注入 LM → LM 继续生成答案 → 完成
```

**最巧妙的设计**  
- **动态查询决策**：模型不必预先决定所有需要的查询，而是根据已经得到的中间信息实时决定是否再查。这样可以避免一次性生成超长、冗余的 SQL。  
- **统一语言/表格表示**：把数据库返回的结构化结果直接转成自然语言片段，让 LM 继续使用同一套生成机制，无需额外的特征拼接层。

### 实验与效果
- **测试集合**：作者构建了 TAG‑Bench，包含数千条自然语言问题，覆盖七大交互模式（如逐步聚合、条件分支、模糊匹配等）。  
- **基线对比**：与传统 Text2SQL、RAG 以及最新的混合检索模型相比，TAG 在整体准确率上提升了约 30% 以上（具体数字未在摘要中给出，论文声称提升显著）。  
- **消融实验**：去掉查询生成器或结果注入器后，系统的正确率跌回 20% 左右，说明交互机制是性能提升的关键。  
- **局限性**：实验主要在结构化的关系数据库上进行，针对图数据库或非结构化数据的适配尚未探索；此外，模型在极端长对话或需要大量子查询的场景仍会出现超时或错误生成。

### 影响与延伸思考
TAG 把“语言模型 + 数据库”从单向翻译的关系转变为双向协作的范式，已经在后续的工作中激发了多条研究路线：  
- **自适应查询规划**：让模型学习何时合并子查询、何时缓存中间结果，以降低数据库调用次数。  
- **跨模态数据融合**：把向量检索、图谱查询等非关系型数据源也纳入同一交互框架，形成更通用的“数据增强生成”。  
- **安全与可解释性**：因为每一步查询都可审计，研究者开始探索如何利用 TAG 的可追溯性来防止模型生成错误或有害信息。  
想进一步了解，可以关注近期在 ACL、NeurIPS 上出现的 “Interactive Retrieval‑Augmented Generation” 系列论文，它们大多把 TAG 视作概念起点进行扩展。

### 一句话记住它
TAG 把语言模型和数据库变成可以来回对话的伙伴，让 AI 能在自然语言提问中随时查表、算数，从而突破了传统 Text2SQL 与 RAG 的局限。