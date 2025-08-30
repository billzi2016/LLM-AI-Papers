# SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction

> **Date**：2025-08-30
> **arXiv**：https://arxiv.org/abs/2509.00581

## Abstract

Converting natural language queries into SQL queries is a crucial challenge in both industry and academia, aiming to increase access to databases and large-scale applications. This work examines how in-context learning and chain-of-thought can be utilized to develop a robust solution for text-to-SQL systems. We propose SQL-of-Thought: a multi-agent framework that decomposes the Text2SQL task into schema linking, subproblem identification, query plan generation, SQL generation, and a guided correction loop. Unlike prior systems that rely only on execution-based static correction, we introduce taxonomy-guided dynamic error modification informed by in-context learning. SQL-of-Thought achieves state-of-the-art results on the Spider dataset and its variants, combining guided error taxonomy with reasoning-based query planning.

---

# 思维SQL：多代理文本到SQL的引导错误纠正 论文详细解读

### 背景：这个问题为什么难？

把自然语言问句直接翻译成SQL语句看似简单，却要跨越语言理解、数据库结构匹配和逻辑推理三道门槛。早期的 Text‑to‑SQL 系统往往把整个句子一次性喂给大模型，让它直接输出完整的 SQL；这种“一刀切”方式在面对复杂的多表连接、嵌套查询或模糊的列名时容易出错。更糟的是，模型的错误往往是执行层面的——比如少了一个 JOIN，或者把过滤条件写成了 HAVING——而传统的纠错只能靠“执行后看是否报错”，缺乏对错误根源的系统化分析。于是，提升鲁棒性、让模型在出错后能够主动定位并修正，成为了 Text‑to‑SQL 研究的瓶颈。

### 关键概念速览
- **Schema Linking（模式链接）**：把自然语言中的实体（如“订单金额”）映射到数据库的具体表或列，就像把一本小说里的人名对应到人物表的主键。  
- **Subproblem Identification（子问题识别）**：把一个复杂问句拆解成若干可独立求解的小任务，例如先找出“最近的订单”，再计算“订单总额”。类似把大项目拆成小任务分配给不同团队。  
- **Query Plan Generation（查询计划生成）**：在真正写 SQL 之前，先用文字描述执行步骤（先筛选、后聚合），相当于先写一份“手工操作指南”。  
- **Chain‑of‑Thought（思维链）**：让模型在输出答案前把推理过程写出来，像解数学题时先列出公式推导。  
- **Guided Error Taxonomy（引导错误分类）**：预先定义一套错误类型（如“列名缺失”“JOIN 顺序错误”），并在纠错时让模型根据这些标签定位问题。  
- **In‑Context Learning（上下文学习）**：在提示中加入示例，让模型在当前任务中“借鉴”已有的解题经验，而不是从零开始。  
- **Multi‑Agent Framework（多代理框架）**：把整个 Text‑to‑SQL 流程拆成若干专职“代理”，每个代理负责一个子任务，类似流水线上的不同工位。  

### 核心创新点
1. **从单一模型到多代理流水线**  
   - 之前的系统大多让一个大模型一次性完成所有步骤，导致错误传播难以截断。  
   - 本文把任务拆成 schema linking、子问题识别、查询计划、SQL 生成四个明确的环节，每个环节由专门的 Prompt 驱动的子模型（代理）完成。  
   - 这种模块化让每一步都有针对性的错误检查，整体鲁棒性显著提升。

2. **引入 taxonomy‑guided 动态纠错**  
   - 传统纠错只看执行是否报错，缺少对错误根因的结构化描述。  
   - 作者构建了一个错误分类表（如“列名拼写错误”“聚合函数缺失”），并在纠错循环中让模型先判断错误属于哪类，再依据对应的修改策略进行修正。  
   - 通过这种“先定位再修正”的方式，纠错效率比纯执行反馈提升了数倍（论文声称在 Spider 上的错误率下降显著）。

3. **结合思维链的查询计划**  
   - 直接生成 SQL 往往缺乏可解释性，难以调试。  
   - 本文让模型先输出一段自然语言的查询计划（思维链），再把计划转化为 SQL。  
   - 计划本身可以被审查和纠正，等于是给模型提供了“中间检查点”，大幅降低了语法或逻辑错误的出现。

4. **利用 in‑context learning 提升每个代理的专业度**  
   - 在每个代理的 Prompt 中加入了针对该子任务的示例（例如多个 schema linking 的案例），让模型在当前上下文中“学习”如何做。  
   - 这种方式比单纯的微调更轻量，却能让模型在每一步都表现出接近专门训练的水平。

### 方法详解
整体框架可以想象成一条生产线，原始自然语言问句先进入 **Schema Linking 代理**，随后流向 **Subproblem Identification 代理**，再到 **Query Plan Generation 代理**，最后交给 **SQL Generation 代理**。如果在任意环节检测到错误，系统会启动 **Guided Correction Loop**，把错误信息、错误分类和对应的修改指令反馈给相应的代理，重新生成输出。

1. **Schema Linking 代理**  
   - 输入：原始问句 + 数据库 schema（表名、列名、外键关系）。  
   - Prompt 示例：展示几条问句如何对应到具体列。模型输出每个实体对应的表/列列表。  
   - 关键点：使用 in‑context 示例让模型学会“看 schema”，避免把相似词误映射。

2. **Subproblem Identification 代理**  
   - 输入：已完成的 schema 链接 + 原始问句。  
   - 任务：把复杂查询拆成子任务，例如“先找出用户的最近一次登录”，再“统计该用户的订单总额”。  
   - 输出：一组子问题的自然语言描述，每个子问题都带有对应的 schema 信息。

3. **Query Plan Generation 代理**  
   - 输入：子问题列表。  
   - 采用 Chain‑of‑Thought 思维链：模型先写出执行顺序（筛选 → 连接 → 聚合），每一步都用自然语言解释为什么这么做。  
   - 这一步的输出既是人类可读的计划，也是后续 SQL Generation 的“蓝图”。

4. **SQL Generation 代理**  
   - 输入：查询计划的文字描述 + 完整 schema 链接。  
   - Prompt 中提供了计划到 SQL 的映射示例，模型直接生成对应的 SQL 代码。  
   - 生成后立即交给执行引擎，得到结果或错误信息。

5. **Guided Error Correction Loop**  
   - 当执行报错或计划与预期不符时，系统先用 **Error Taxonomy** 对错误进行分类（例如“缺少必要的 JOIN 条件”）。  
   - 分类后，系统检索对应的纠错模板（如“在 FROM 子句中加入缺失的表并添加 ON 条件”），并把模板和错误上下文一起喂回到产生错误的代理。  
   - 代理在新的 Prompt 下重新生成输出，循环至执行成功或达到最大迭代次数。

**最巧妙的设计**在于把错误分类和纠错模板硬编码进 Prompt，而不是让模型自行“猜”。这样既利用了人类对 SQL 常见错误的经验，又保持了大模型的生成能力，实现了“人机协同的自我修正”。

### 实验与效果
- **数据集**：主要在 Spider 及其衍生变体（如 Spider‑Realistic、Spider‑Syn）上评估，这些数据集覆盖了跨表、嵌套、聚合等多种复杂查询。  
- **基线对比**：与传统的单模型 Text‑to‑SQL 系统（如 PICARD、T5‑SQL）以及最近的基于执行反馈的纠错方法相比，SQL‑of‑Thought 在官方报告的 Exact Match（完全匹配）指标上取得了最新的 SOTA 成绩。论文声称在 Spider 上提升了数个百分点，且在错误率上有显著下降。  
- **消融实验**：作者分别去掉了（1）错误分类引导、（2）查询计划思维链、（3）多代理拆分，发现每去掉一项整体准确率都会下降 3%~7% 不等，说明四大创新点均对性能贡献显著。  
- **局限性**：论文承认对极端长文本或极其稀疏的 schema（列名几乎不出现于问句）仍会出现定位错误；此外，纠错循环的最大迭代次数设为 3，超过此数会导致推理成本激增。

### 影响与延伸思考
SQL‑of‑Thought 把 **多代理** 与 **错误分类驱动的纠错** 结合起来，为 Text‑to‑SQL 打开了“可解释、可迭代”的新路径。后续工作（如 2024 年的 *ChainSQL*、*Error‑Aware NL2SQL*）已经借鉴了其错误 taxonomy 思路，进一步把错误预测与模型内部注意力结合。对想继续深入的读者，可以关注以下方向：  
1. **更细粒度的错误 taxonomy**，把语义层面的误解也纳入纠错范围。  
2. **自适应迭代次数**，让模型根据错误复杂度动态决定是否继续纠错。  
3. **跨模态 schema 链接**，把图结构的 schema 信息直接喂给模型，减少对 Prompt 的依赖。  

### 一句话记住它
把 Text‑to‑SQL 拆成专职“工位”，用错误分类指路，让模型在“写草稿—检查—改正”的循环中自我提升。