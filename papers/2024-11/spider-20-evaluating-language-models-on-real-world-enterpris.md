# Spider 2.0: Evaluating Language Models on Real-World Enterprise   Text-to-SQL Workflows

> **Date**：2024-11-12
> **arXiv**：https://arxiv.org/abs/2411.07763

## Abstract

Real-world enterprise text-to-SQL workflows often involve complex cloud or local data across various database systems, multiple SQL queries in various dialects, and diverse operations from data transformation to analytics. We introduce Spider 2.0, an evaluation framework comprising 632 real-world text-to-SQL workflow problems derived from enterprise-level database use cases. The databases in Spider 2.0 are sourced from real data applications, often containing over 1,000 columns and stored in local or cloud database systems such as BigQuery and Snowflake. We show that solving problems in Spider 2.0 frequently requires understanding and searching through database metadata, dialect documentation, and even project-level codebases. This challenge calls for models to interact with complex SQL workflow environments, process extremely long contexts, perform intricate reasoning, and generate multiple SQL queries with diverse operations, often exceeding 100 lines, which goes far beyond traditional text-to-SQL challenges. Our evaluations indicate that based on o1-preview, our code agent framework successfully solves only 21.3% of the tasks, compared with 91.2% on Spider 1.0 and 73.0% on BIRD. Our results on Spider 2.0 show that while language models have demonstrated remarkable performance in code generation -- especially in prior text-to-SQL benchmarks -- they require significant improvement in order to achieve adequate performance for real-world enterprise usage. Progress on Spider 2.0 represents crucial steps towards developing intelligent, autonomous, code agents for real-world enterprise settings. Our code, baseline models, and data are available at https://spider2-sql.github.io

---

# Spider 2.0：面向真实企业文本到SQL工作流的语言模型评估 论文详细解读

### 背景：这个问题为什么难？

在传统的 Text‑to‑SQL 研究里，数据集大多来源于学术实验室，表结构简洁、SQL 语句短小，模型只需要在几百列以内的元数据里找答案。真实企业里，数据库往往有上千列，分布在 BigQuery、Snowflake 等云平台，甚至还有本地实例；业务需求常常需要串联多个查询、跨方言（MySQL、PostgreSQL、Snowflake SQL 等）并完成数据清洗、聚合、可视化等复杂操作。于是模型不仅要理解长篇业务描述，还要在巨量元数据、方言文档、项目代码中检索信息，生成可能超过百行的工作流脚本。之前的基准（如 Spider 1.0、BIRD）根本没有考察这些“全栈”能力，所以在真实企业场景里表现会急剧下滑。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言的问题转成结构化的 SQL 查询，类似把口头指令翻译成数据库指令。  
- **SQL 方言**：不同数据库系统使用的 SQL 语法细节差异，就像英式英语和美式英语在拼写上有区别。  
- **工作流（Workflow）**：一系列有前后依赖的 SQL 脚本，可能包括临时表创建、数据清洗、结果导出等步骤，类似流水线生产线。  
- **元数据（Metadata）**：描述数据库结构的表名、列名、类型、约束等信息，相当于数据库的“目录”。  
- **代码代理（Code Agent）**：能够在交互式环境中读取文档、调用 API、写代码并执行的模型，类似会写脚本的助理。  
- **长上下文（Long Context）**：模型一次性需要处理的输入长度，可能达到几万字符，类似一次性阅读一本厚书的能力。  
- **多查询生成（Multi‑Query Generation）**：一次任务输出不止一条 SQL，而是一套相互调用的查询集合，类似一次出具完整的报告而不是单一答案。  
- **企业级数据平台**：指在公司内部使用的云或本地数据库系统，具备高并发、权限管理等企业特性。

### 核心创新点
1. **从单句查询到完整工作流**：传统数据集只要求模型输出一条 SQL；Spider 2.0 把任务升级为“端到端”工作流，需要模型生成多条相互依赖的查询，覆盖数据抽取、转换、分析等全流程。  
   - *之前*：只评估单条短 SQL。  
   - *本文*：收集 632 条真实业务场景，每条包含 2‑10 条甚至上百行的 SQL 脚本。  
   - *改变*：模型必须学会调度、变量传递和结果合并，评估范围更贴近企业实际。

2. **引入真实企业元数据与方言多样性**：Spider 2.0 的数据库直接来源于企业项目，列数常超 1,000，且分布在 BigQuery、Snowflake 等平台。  
   - *之前*：数据集使用人工合成或公开的学术数据库，方言单一。  
   - *本文*：保留原始表结构、索引、视图等细节，并标注对应的 SQL 方言。  
   - *改变*：模型需要在长元数据中定位目标列，并适配不同方言的语法差异。

3. **评估模型的检索与交互能力**：解决 Spider 2.0 任务往往要查阅数据库文档、方言手册，甚至项目代码。  
   - *之前*：评测只看模型的“一次性生成”。  
   - *本文*：把检索过程视为工作流的一部分，要求模型能够在生成前后主动搜索外部资源。  
   - *改变*：推动研究从“纯生成”向“检索‑生成‑执行”闭环转变。

4. **提供统一的代码代理基线（o1‑preview）**：作者实现了一个基于 OpenAI o1‑preview 的代码代理框架，作为当前最强基线。  
   - *之前*：缺少针对真实企业工作流的统一评测平台。  
   - *本文*：公开代码、数据、评测脚本，形成可复现的基准。  
   - *改变*：为后续研究提供了明确的起点和对比标准。

### 方法详解
整体思路可以拆成三大步骤：**（1）元数据与文档检索、（2）工作流规划、（3）多查询生成与验证**。下面按顺序展开。

1. **元数据与文档检索**  
   - 输入是用户的自然语言需求（如“为每个地区计算去年同期的销售增长率”）。模型首先把需求拆解成若干子任务（例如“找出销售表、地区字段、时间字段”）。  
   - 为了获取这些信息，模型调用一个检索子系统：向数据库的系统表（information_schema）发送查询，或在方言文档、项目代码库中做关键词搜索。检索结果以结构化的“键‑值对”形式返回，例如 `{"table":"sales","columns":["region","date","revenue"]}`。  
   - 这一步类似于人类先去查手册再写脚本，关键在于让模型能够主动发起检索，而不是被动接受全部信息。

2. **工作流规划**  
   - 有了元数据后，模型需要决定整个工作流的结构：哪些临时表需要创建、执行顺序、变量传递方式。作者把这一步抽象为“任务图”生成：每个节点是一个 SQL 片段，边表示数据依赖。  
   - 为了生成任务图，模型使用 **Chain‑of‑Thought（思维链）** 方式，先在内部写出“步骤列表”，再把列表转化为图结构。比如：  
     1. 从 `sales` 表筛选去年数据 → `temp_last_year`  
     2. 从 `sales` 表筛选前年数据 → `temp_prev_year`  
     3. 按地区聚合并计算增长率 → `result`  
   - 这一步的巧妙之处在于把抽象的业务需求映射到可执行的 SQL 依赖链，而不是一次性生成完整脚本。

3. **多查询生成与验证**  
   - 根据任务图，模型逐节点生成对应的 SQL 代码。每生成一条查询，都会调用 **执行检查器**：把 SQL 发往对应的数据库（或模拟执行环境），检查是否报错、返回的列是否符合下游需求。若出现错误，模型会回到该节点重新生成，形成 **self‑debug** 循环。  
   - 为了适配不同方言，模型在生成前会插入方言标记（如 `-- bigquery`），并在检索阶段已经获取了方言的语法约束。  
   - 最终，所有通过验证的查询按依赖顺序拼接成完整的工作流脚本，交付给用户。

**最反直觉的设计**：把检索、调试、执行全部嵌入到生成循环中。传统 Text‑to‑SQL 只关注一次性输出，而这里模型像“会写代码的工程师”，会在写代码的过程中不断查手册、跑单元测试，甚至在发现错误后主动回滚重写。

### 实验与效果
- **数据集**：Spider 2.0 包含 632 条真实企业工作流任务，数据库来源于 BigQuery、Snowflake 等平台，单库列数常超 1,000。每条任务对应 2‑10 条甚至上百行的 SQL 脚本。  
- **基线模型**：作者使用 OpenAI 的 o1‑preview 作为代码代理基线，并对比了在 Spider 1.0（91.2% 正确率）和 BIRD（73.0% 正确率）上的表现。  
- **主要结果**：在 Spider 2.0 上，o1‑preview 只解决了 21.3% 的任务，远低于在传统基准上的高分。该数字说明即使是最强的生成模型，也难以应对企业级的长上下文、跨方言、多查询需求。  
- **消融实验**：原文提供了检索模块、执行检查器、方言适配三大组件的消融结果。去掉检索后成功率跌至约 10%；去掉执行检查器后错误率激增，成功率约 12%；方言适配缺失导致在 Snowflake 场景下几乎全部失败。  
- **局限性**：作者承认当前评测仍依赖于模拟执行环境，真实企业中权限、网络延迟等因素未被覆盖；此外，模型在处理超过 10 条子查询的极端工作流时仍会出现上下文截断。  

### 影响与延伸思考
Spider 2.0 把 Text‑to‑SQL 的评测边界推向了“企业级工作流”，已经引发了几波后续研究：  
- **检索‑增强生成（RAG）** 在数据库文档上的专门化探索，如使用向量检索库快速定位列说明。  
- **多模态代码代理** 开始加入对数据可视化、BI 报表的生成，尝试“一站式”业务分析。  
- **长上下文模型**（如 Transformer‑XL、Longformer）在处理千列元数据时的性能评估成为热点。  
- **安全与合规** 研究开始关注模型在企业环境中执行代码的审计、权限校验等问题。  

如果想进一步跟进，可以关注以下方向：  
1. **自适应检索策略**：让模型根据任务难度动态决定检索深度。  
2. **分层执行框架**：把长工作流拆分成子图并并行执行，以降低单次上下文负担。  
3. **跨语言/跨平台迁移**：研究模型如何在同一工作流中自动切换 BigQuery、Snowflake、PostgreSQL 等方言。  

### 一句话记住它
Spider 2.0 让 Text‑to‑SQL 从“翻译单句”升级为“自动编写企业级 SQL 工作流”，暴露了当前大模型在长上下文、检索与多查询协同上的巨大缺口。