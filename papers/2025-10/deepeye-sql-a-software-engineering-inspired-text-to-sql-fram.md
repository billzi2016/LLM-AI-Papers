# DeepEye-SQL: A Software-Engineering-Inspired Text-to-SQL Framework

> **Date**：2025-10-20
> **arXiv**：https://arxiv.org/abs/2510.17586

## Abstract

Large language models (LLMs) have advanced Text-to-SQL, yet existing solutions still fall short of system-level reliability. The limitation is not merely in individual modules -- e.g., schema linking, reasoning, and verification -- but more critically in the lack of structured orchestration that enforces correctness across the entire workflow. This gap motivates a paradigm shift: treating Text-to-SQL not as free-form language generation but as a software-engineering problem that demands structured, verifiable orchestration. We present DeepEye-SQL, a software-engineering-inspired framework that reframes Text-to-SQL as the development of a small software program, executed through a verifiable process guided by the Software Development Life Cycle (SDLC). DeepEye-SQL integrates four synergistic stages: it grounds user intent through robust schema linking, enforcing relational closure; enhances fault tolerance with N-version SQL generation; ensures deterministic verification via a ``Syntax-Logic-Quality'' tool-chain that intercepts errors pre-execution; and introduces confidence-aware selection that leverages execution-guided adjudication to resolve ambiguity beyond simple majority voting. Leveraging open-source MoE LLMs (~30B total, ~3B activated parameters) without any fine-tuning, DeepEye-SQL achieves 73.5% execution accuracy on BIRD-Dev, 75.07% on the official BIRD-Test leaderboard, and 89.8% on Spider-Test, outperforming state-of-the-art solutions that rely on larger models or extensive training. This highlights that principled orchestration, rather than LLM scaling alone, is key to achieving system-level reliability in Text-to-SQL.

---

# DeepEye‑SQL：受软件工程启发的文本到SQL框架 论文详细解读

### 背景：这个问题为什么难？

文本到SQL（Text‑to‑SQL）本质上要把自然语言查询翻译成能够在关系数据库上执行的代码。过去的模型大多把这当成一次性生成任务，靠大模型的语言能力直接输出 SQL。可是，SQL 语法严谨、表结构复杂、业务意图常常隐含，稍有偏差就会导致执行错误或返回空结果。现有方案虽然在 schema linking（把自然语言词映射到表/列）和后置验证上做了改进，但仍缺乏统一的、可追溯的工作流；每个子模块都是独立的“黑盒”，错误难以定位，整体可靠性难以保证。

### 关键概念速览
- **Schema Linking（模式链接）**：把用户提问中的实体（如“订单金额”）对应到数据库的表和列，就像把地图上的地名对应到实际的街道坐标。  
- **Robust Schema Linking（鲁棒模式链接）**：在普通链接的基础上加入容错策略，确保即使某一步识别错误也不会导致整个流程崩溃。  
- **N‑Version SQL Generation（N 版 SQL 生成）**：让模型用多种生成策略（骨架填充、示例驱动、递归拆解等）同时产出多条候选 SQL，类似软件开发中写多个实现版本以备比较。  
- **Syntax‑Logic‑Quality（语法‑逻辑‑质量）链**：一套自动化检查工具，先检查 SQL 语法是否合规，再验证逻辑（如列是否在 FROM 表中），最后评估质量指标（如是否使用了不必要的子查询）。  
- **Execution‑Guided Adjudication（执行引导裁决）**：把候选 SQL 实际跑一遍，根据返回结果和置信度决定最终答案，类似单元测试后选出最可靠的实现。  
- **Confidence‑Aware Selection（置信度感知选择）**：在裁决时考虑模型对每条候选的自信程度，避免单纯多数投票导致的错误。  
- **Software Development Life Cycle（SDLC，软件开发生命周期）**：传统软件工程中需求、设计、实现、测试、部署的循环过程，本文把它搬到文本到 SQL 的任务上。

### 核心创新点
1. **把 Text‑to‑SQL 当作软件开发而非一次性生成**  
   过去的系统把自然语言直接喂给大模型，让它一次性输出完整 SQL；DeepEye‑SQL 把任务拆成需求捕获、设计（schema linking）、实现（N‑版生成）和测试（Syntax‑Logic‑Quality + 执行裁决）四个阶段。这样每一步都有明确输入输出，错误可以在早期被捕获，整体可靠性大幅提升。

2. **鲁棒的 Schema Linking + 关系闭合**  
   传统链接只找出最可能的表/列，错一步就会导致后续生成全盘崩溃。DeepEye‑SQL 引入“关系闭合”概念：在识别出必要表后，自动补全它们之间的外键关系，形成完整的连接图。即使某个实体被误识别，闭合机制仍能提供备选路径，显著降低单点失效概率。

3. **N‑版 SQL 生成 + 多策略融合**  
   不是只用一种 prompting 方式，而是并行使用三种生成手段：① 基于固定骨架的填空，② 通过 few‑shot 示例的 in‑context learning，③ 递归拆解复杂查询为子问题。每种手段产生独立候选，形成“多版本”池，类似软件中写多个实现再挑最优。

4. **执行引导的置信度感知裁决**  
   传统做法往往用多数投票或模型打分决定最终 SQL，忽略了实际执行结果。DeepEye‑SQL 在候选池上跑轻量级单元测试，收集返回的行数、错误码等信息，再结合模型给出的置信度进行加权选择。这样即使多数模型倾向错误答案，只要执行表现差，系统也会倾向更可靠的候选。

### 方法详解
整体框架遵循 SDLC 四阶段循环：

1. **需求捕获（Intent Grounding）**  
   - 输入：用户自然语言查询 + 数据库 schema。  
   - 步骤：先用 LLM 做语义价值检索（Semantic Value Retrieval），把查询中的实体映射到实际数据值（如“去年销量最高的产品”对应具体的数值范围）。随后执行 Robust Schema Linking：模型列出所有可能涉及的表/列，并通过外键闭合生成完整的连接图。此阶段的输出是一组“需求对象”，包括目标表集合、过滤条件、聚合目标等。

2. **设计实现（N‑Version SQL Generation）**  
   - **骨架填充**：预定义常见查询模式（SELECT‑FROM‑WHERE、JOIN‑GROUP‑BY 等），模型只负责填充表名、列名、条件。  
   - **示例驱动 ICL**：提供少量相似查询‑SQL 对作为示例，模型在上下文中学习生成。  
   - **递归拆解**：对复杂自然语言先拆成子句，每个子句单独生成 SQL，再在代码层面拼接。  
   - 每种策略独立运行，产生多条候选 SQL，形成 N‑版池。

3. **测试验证（Syntax‑Logic‑Quality）**  
   - **Syntax 检查**：使用开源 SQL 解析器验证语法合法性。  
   - **Logic 检查**：确保所有 SELECT 列在 FROM 表中出现，JOIN 条件对应外键，WHERE 条件字段类型匹配。  
   - **Quality 评估**：计算查询复杂度、是否出现不必要的子查询、是否使用了最佳索引字段等指标。  
   - 不合格的候选直接剔除，剩余候选进入下一步。

4. **执行裁决（Confidence‑Aware Selection）**  
   - 对每条通过测试的 SQL 执行一次轻量级查询（通常加 LIMIT），收集返回行数、错误信息。  
   - 结合模型在生成时给出的置信度分数，使用加权投票或贝叶斯更新得到最终选项。  
   - 最终 SQL 通过完整执行后返回给用户。

**最巧妙的点**在于把“执行结果”当作第三方审判者，引入了软件工程里常见的单元测试概念，使得系统不再盲目相信模型输出，而是让真实数据库行为来纠错。

### 实验与效果
- **数据集**：BIRD（Dev / Test）和 Spider（Test），两者都是业界常用的 Text‑to‑SQL 基准。  
- **主要指标**：Execution Accuracy（执行正确率），即生成的 SQL 在真实数据库上返回的结果与标注答案完全一致。  
- **结果**：在 BIRD‑Dev 上达到 73.5%，在 BIRD‑Test 官方排行榜上 75.07%，Spider‑Test 上 89.8%。这些数字均超过了当时使用更大模型或经过大量微调的最先进系统。  
- **Baseline 对比**：与同类使用单一 LLM 生成的方案相比，DeepEye‑SQL 提升约 5‑10% 的执行准确率；与基于微调的强基线相比，差距在 2‑3% 之内，却只用了约 30B 参数的开源模型（实际激活约 3B），成本更低。  
- **消融实验**：论文分别去掉 Robust Schema Linking、N‑Version 生成、Syntax‑Logic‑Quality、Execution‑Guided 裁决四个模块，发现每去掉一项整体准确率下降 3‑6%，其中 Execution‑Guided 对 Spider 的提升最明显。  
- **局限**：仍依赖于轻量级执行环境，若数据库规模极大或查询涉及复杂事务，执行测试成本会升高；此外，框架对 schema 变化的适应性未在实验中评估。

### 影响与延伸思考
DeepEye‑SQL 把软件工程的“结构化、可验证”理念引入自然语言到代码的生成任务，开启了“LLM + SDLC” 的新思路。后续工作开始探索把 CI/CD（持续集成/持续部署）流水线、代码审查等概念进一步迁移到代码生成、数据分析等领域。对想深入的读者，可以关注以下方向：① 将更丰富的单元测试（如属性测试）加入生成流程；② 在大模型内部实现“自检”机制，减少外部执行开销；③ 把这种框架推广到多模态或跨数据库的查询场景。整体来看，系统化的工作流比单纯的模型规模更能提升实际可靠性，这一点在业界逐渐被认可。

### 一句话记住它
把 Text‑to‑SQL 当作一段小程序的开发，用多版本生成、自动化单元测试和执行裁决把“语言模型输出”硬核验证，可靠性大幅提升。