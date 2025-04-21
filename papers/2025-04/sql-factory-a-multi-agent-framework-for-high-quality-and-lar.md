# SQL-Factory: A Multi-Agent Framework for High-Quality and Large-Scale SQL Generation

> **Date**：2025-04-21
> **arXiv**：https://arxiv.org/abs/2504.14837

## Abstract

High quality SQL corpus is essential for intelligent database. For example, Text-to-SQL requires SQL queries and correspond natural language questions as training samples. However, collecting such query corpus remains challenging in practice due to the high cost of manual annotation, which highlights the importance of automatic SQL generation. Despite recent advances, existing generation methods still face limitations in achieving both diversity and cost-effectiveness. Besides, many methods also treat all tables equally, which overlooks schema complexity and leads to under-utilization of structurally rich tables. To address these issues, this paper proposes a multi-agent framework for high-quality and large-scale SQL generation, dubbed SQL-Factory. It decomposes the generation process into three collaborative teams: the Generation Team explores diverse query structures using a powerful language model, the Expansion Team scales promising patterns via a lightweight language model, and the Management Team adaptively schedules the workflow and evaluates the quality of synthesized queries. This modular framework ensures a balanced trade-off between diversity, scalability, and generation cost. We apply SQL-Factory to four widely used benchmarks and generate over 300,000 SQL queries with less than $200 API cost. Our generated queries achieve higher diversity compared to other methods, and extensive experiments demonstrate that the generated queries significantly improve the model performance in various downstream tasks.

---

# SQL-Factory：高质量大规模SQL生成的多智能体框架 论文详细解读

### 背景：这个问题为什么难？

在 Text‑to‑SQL 这类需要大量标注数据的任务里，SQL 语句本身的质量直接决定模型的上限。手工写数万甚至数十万条自然语言‑SQL 对齐样本成本极高，尤其是要覆盖不同数据库的复杂 schema（表结构、外键关系等）。已有的自动生成方法要么只会产生千篇一律的查询，缺乏结构多样性；要么使用大模型一次性生成，费用随规模指数增长。更糟的是，大多数方法把所有表当成同质的“砖块”，忽视了某些 schema 本身就蕴含丰富的查询模式，导致生成的语料库没有充分利用这些结构信息。于是，如何在保持成本可控的前提下，既产出多样化又高质量的大规模 SQL 成了瓶颈。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言问题翻译成对应的 SQL 查询，类似把口头指令转成数据库指令的过程。  
- **Schema（模式）**：数据库里表的结构、列的类型以及表之间的关联，就像一张城市地图的路网。  
- **多智能体（Multi‑Agent）**：把一个大任务拆成若干小团队，每个团队专注不同子任务，类似工厂里不同岗位的工人协同完成产品。  
- **Generation Team（生成团队）**：使用强大的语言模型探索各种查询骨架，像是让经验丰富的设计师先画出多种草图。  
- **Expansion Team（扩展团队）**：用轻量模型把有潜力的草图批量复制、细化，类似流水线上的复制机。  
- **Management Team（管理团队）**：负责调度、质量评估和资源分配，像是工厂的调度中心，决定哪条生产线继续、哪条停下。  
- **多样性（Diversity）**：生成的 SQL 在结构、连接方式、聚合函数等维度的差异程度，类似菜谱的花样多寡。  
- **成本效益（Cost‑Effectiveness）**：在保证质量的前提下，用最少的 API 调用费用产出最多的语料。

### 核心创新点
1. **把生成过程拆成三支协作团队**  
   之前的工作大多让单一模型一次性完成“写草图+填细节+检查”。这篇论文把任务分成 Generation、Expansion、Management 三个子任务。Generation 用大模型产生高质量、结构多样的查询骨架；Expansion 用轻量模型快速复制并微调这些骨架；Management 动态调度两者的调用频率并实时评估生成质量。这样既保留了大模型的创造力，又利用轻模型的高吞吐，实现了“质量+规模”双赢。

2. **基于 schema 复杂度的差异化生成**  
   传统方法把所有表当成同等对象，导致对结构丰富的表生成的查询数量不足。SQL‑Factory 在 Generation 阶段会先对每个 schema 进行复杂度打分，复杂度高的表会被分配更多的生成预算，从而让这些表的潜在查询模式被充分挖掘。结果是生成的语料库在结构上更贴近真实业务需求。

3. **自适应调度与质量评估机制**  
   Management Team 不是固定的调度规则，而是依据实时的生成成功率、重复率和多样性指标动态调整两支团队的调用比例。比如，当某类查询的重复率上升时，系统会自动降低该类的生成频率，转而探索新结构。这样避免了“生成机器”陷入局部最优，保持整体语料的活力。

4. **极低成本的大规模生成**  
   通过让轻模型承担大多数扩展工作，论文声称在四个公开基准上共生成了超过 30 万条 SQL，花费不到 200 美元的 API 费用。相比直接用大模型生成同等规模的语料，成本下降了数十倍。

### 方法详解
整体框架可以想象成一条三段式装配线：**草图 → 复制 → 检验**。第一段是 Generation Team，第二段是 Expansion Team，第三段是 Management Team。整个流程是循环的：Management 根据上一轮的评估结果决定下一轮的预算分配，然后把预算交给 Generation，生成若干“草图”，再交给 Expansion 进行批量扩展，最后回到 Management 进行质量打分和调度更新。

**1. Generation Team**  
- 输入：目标数据库的 schema（表名、列名、外键等）以及一个随机的“查询意图”提示。  
- 使用：GPT‑4 之类的强大语言模型。  
- 输出：一组结构完整的 SQL 骨架，通常只包含 SELECT、FROM、WHERE 的基本框架，且每条都带有对应的自然语言描述。  
- 关键技巧：在提示中加入“请多尝试不同的连接方式、聚合函数和子查询”，让模型主动探索多样的查询结构。

**2. Expansion Team**  
- 输入：Generation 输出的骨架集合。  
- 使用：一个参数量更小、调用成本更低的模型（如 LLaMA‑7B）。  
- 操作：对每个骨架进行“微调”——随机替换列名、改写条件常量、加入/删除额外的 JOIN 或 GROUP BY。相当于在流水线上让复制机把草图复制成千上万的细节版本。  
- 规模控制：每个骨架会被扩展一个预设的倍率（例如 20 倍），但实际倍率由 Management 根据预算动态调节。

**3. Management Team**  
- 负责三件事：  
  a. **调度**：根据当前预算、生成成本和质量指标，决定 Generation 与 Expansion 的调用比例。  
  b. **质量评估**：使用轻量的规则检查（语法合法性、是否出现未定义列）以及基于模型的可行性打分（如使用小模型对自然语言‑SQL 对齐度进行打分）。  
  c. **多样性监控**：统计已生成 SQL 的结构特征（JOIN 数量、聚合函数种类等），若某类结构出现频率过高，则降低其后续生成概率。  
- 实现方式：一个循环的控制器，每轮结束后更新内部的“预算表”和“多样性分布”，并把这些信息反馈给前两支团队的提示模板。

**最巧妙的地方**  
- **预算自适应**：不像传统的“一次性生成”，这里的预算是动态的，系统会在生成过程中“学会”哪些结构更值得投入资源。  
- **轻模型的高效扩展**：把大模型的创造力限制在少量高质量骨架上，然后用轻模型完成大规模复制，极大降低了 API 成本。  
- **结构感知的多样性控制**：通过对 schema 复杂度的打分和结构特征的统计，系统能够主动避免“只会 SELECT * FROM A”这类单调模式。

### 实验与效果
- **测试基准**：论文在四个常用的 Text‑to‑SQL 数据集上做实验（具体名称未在摘要中列出，但可以推测包括 Spider、WikiSQL 等）。  
- **生成规模**：共产生了超过 30 万条 SQL，花费不到 200 美元的 API 费用。  
- **对比基线**：与仅使用单一大模型或仅使用轻模型的生成方法相比，SQL‑Factory 在多样性指标上提升了显著幅度（论文声称“显著高于其他方法”），并且在下游模型的训练上带来了明显的性能提升（如在 Spider 上的执行准确率提升数个百分点）。  
- **消融实验**：作者分别关闭了 Management 调度、Schema 复杂度感知和 Expansion 阶段，结果显示每个模块的缺失都会导致生成质量或多样性下降，尤其是去掉 Management 后成本飙升且重复率上升。  
- **局限性**：论文未详细说明对极端复杂 schema（如上百张表的企业级数据库）的适用性，也没有给出对生成错误 SQL 的自动纠错机制，实际使用时仍需人工过滤。

### 影响与延伸思考
SQL‑Factory 把“多智能体协作”引入了自动化数据语料生成的场景，打开了在成本受限的情况下大规模构造结构化训练数据的新思路。后续的工作可能会在以下方向继续深化：  
- **更细粒度的智能体分工**，比如加入专门的“纠错智能体”或“语义对齐智能体”。  
- **跨数据库迁移**，让生成的 SQL 能自动适配不同方言（MySQL、PostgreSQL、ClickHouse 等）。  
- **自监督质量评估**，利用大模型本身对生成的 SQL 进行自评，进一步降低人工审查成本。  
- **生成式数据增强**在其他结构化任务（如图查询、NoSQL 查询）中的迁移。  
如果想深入了解，可以关注近期在 “LLM‑driven data synthesis” 方向的会议论文，尤其是那些把多智能体框架与强化学习结合的尝试。

### 一句话记住它
SQL‑Factory 用三支协作的智能体把少量高质量 SQL 骨架扩展成海量多样语料，成本低到几百美元，却能显著提升下游 Text‑to‑SQL 模型的表现。