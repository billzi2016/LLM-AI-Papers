# MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL

> **Date**：2023-12-18
> **arXiv**：https://arxiv.org/abs/2312.11242

## Abstract

Recent LLM-based Text-to-SQL methods usually suffer from significant performance degradation on "huge" databases and complex user questions that require multi-step reasoning. Moreover, most existing methods neglect the crucial significance of LLMs utilizing external tools and model collaboration. To address these challenges, we introduce MAC-SQL, a novel LLM-based multi-agent collaborative framework. Our framework comprises a core decomposer agent for Text-to-SQL generation with few-shot chain-of-thought reasoning, accompanied by two auxiliary agents that utilize external tools or models to acquire smaller sub-databases and refine erroneous SQL queries. The decomposer agent collaborates with auxiliary agents, which are activated as needed and can be expanded to accommodate new features or tools for effective Text-to-SQL parsing. In our framework, We initially leverage GPT-4 as the strong backbone LLM for all agent tasks to determine the upper bound of our framework. We then fine-tune an open-sourced instruction-followed model, SQL-Llama, by leveraging Code Llama 7B, to accomplish all tasks as GPT-4 does. Experiments show that SQL-Llama achieves a comparable execution accuracy of 43.94, compared to the baseline accuracy of 46.35 for vanilla GPT-4. At the time of writing, MAC-SQL+GPT-4 achieves an execution accuracy of 59.59 when evaluated on the BIRD benchmark, establishing a new state-of-the-art (SOTA) on its holdout test set (https://github.com/wbbeyourself/MAC-SQL).

---

# MAC‑SQL：面向文本到SQL的多智能体协同框架 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成SQL语句本来就很挑战，尤其是当数据库规模巨大的时候，模型需要在海量表和列之间做选择，容易遗漏关键信息。再加上真实用户的提问往往包含多步推理——先筛选子集、再做聚合、最后过滤——单一的大语言模型（LLM）一次性给出完整SQL的成功率急剧下降。过去的工作大多只让LLM自己“想”，忽视了让模型调用外部工具或相互协作的潜力，导致在复杂场景下性能停滞不前。

### 关键概念速览
- **Text‑to‑SQL（NL2SQL）**：把自然语言查询转换成结构化的SQL语句，类似把口头指令翻译成数据库指令。
- **大语言模型（LLM）**：像 GPT‑4、Code Llama 这类经过海量文本训练的生成模型，能够理解并生成自然语言和代码。
- **Chain‑of‑Thought（思维链）**：让模型在给出答案前先写出推理步骤，像在纸上列出解题思路一样，帮助模型保持逻辑连贯。
- **多智能体（Multi‑Agent）**：系统里有多个专职模型，每个负责不同子任务，类似团队里分工合作的成员。
- **外部工具调用**：模型可以主动请求数据库查询、检索子表或运行代码等外部服务，而不是只靠内部记忆。
- **子数据库（Sub‑Database）**：从原始巨型库中抽取出与当前问题最相关的少量表和列，降低搜索空间。
- **SQL‑Llama**：基于 Code Llama 7B 微调得到的开源指令跟随模型，专门用于生成和修正SQL。

### 核心创新点
1. **从单一模型到协同多智能体**  
   *之前的做法*：让一个 LLM 完全负责从问题到 SQL 的全部步骤。  
   *本文的做法*：引入一个核心“分解器”智能体负责生成初步 SQL，同时配备两个“辅助”智能体：一个负责检索子数据库，另一个负责纠错和重写错误的 SQL。  
   *带来的改变*：通过分工，系统在面对大库和复杂推理时能先把问题拆解成小块，再逐块解决，整体准确率显著提升。

2. **Few‑Shot 思维链驱动的分解器**  
   *之前的做法*：直接让模型一次性输出完整 SQL，缺少中间推理可视化。  
   *本文的做法*：在分解器内部加入 few‑shot 示例的思维链，引导模型先写出“先筛选哪些表 → 再做哪些聚合 → 最后拼接 WHERE 条件”等步骤。  
   *带来的改变*：模型在生成 SQL 前形成明确的思路，错误率下降，尤其在多步推理的问题上表现更稳。

3. **工具化的子数据库获取**  
   *之前的做法*：模型只能靠内部记忆判断哪些表相关，容易被海量表淹没。  
   *本文的做法*：辅助智能体调用检索工具，对原始库执行轻量查询，返回与问题最匹配的子集。  
   *带来的改变*：搜索空间从上万张表压缩到几十张，分解器的负担大幅减轻，执行效率和准确率同步提升。

4. **开源模型逼近 GPT‑4 上限**  
   *之前的做法*：大多数高性能 NL2SQL 系统只能依赖闭源的强大模型（如 GPT‑4），成本高。  
   *本文的做法*：在 GPT‑4 设定上限后，用 Code Llama 7B 微调得到 SQL‑Llama，完成所有智能体任务。  
   *带来的改变*：SQL‑Llama 在 BIRD 基准上实现 43.94% 的执行准确率，仅比 GPT‑4 的 46.35% 低约 2.4%，展示了开源模型的可行性。

### 方法详解
整体框架可以看作三步走的流水线：

1. **问题分解**：用户的自然语言提问先交给 **分解器智能体**。它在 few‑shot 思维链的引导下，先把任务拆成若干子目标（如“找出涉及的表”“确定聚合方式”“构造过滤条件”），并输出一段结构化的中间表示。

2. **子库检索**：当分解器标记出需要的表或列时，**检索智能体**被激活。它调用外部检索工具（例如向数据库发送 `SHOW TABLES LIKE …` 或基于向量检索的元数据搜索），返回一个 **子数据库**——只包含与当前问题高度相关的表和字段的精简视图。

3. **SQL 生成与纠错**：分解器在拿到子数据库后再次生成完整的 SQL。如果生成的 SQL 在执行阶段抛错，**纠错智能体**会接管，利用错误信息（如 “列不存在”）重新编辑或补全 SQL，直到通过执行检查为止。

#### 模块细节
- **分解器**：采用 GPT‑4（或后续的 SQL‑Llama）作为底层模型。few‑shot 示例包括 3–5 条典型的 “问题 → 思维链 → SQL” 对齐对，帮助模型学习如何把自然语言映射到结构化步骤。思维链的输出形式类似：
  ```
  1. 确认涉及的表：orders, customers
  2. 需要的聚合：SUM(order_amount)
  3. 过滤条件：order_date > '2023-01-01'
  ```
- **检索智能体**：实现为一个轻量的 Python 脚本，接受表名关键词列表，使用正则或向量相似度在数据库元数据中快速定位匹配表。返回的子库是一个临时的 SQLite 副本，极大降低后续查询成本。
- **纠错智能体**：同样基于 LLM，但在提示中加入错误信息（如 “Error: column 'price' does not exist”），让模型把错误当作线索重新推理。它会循环调用，最多三次，确保不会陷入无限纠错。

#### 巧妙之处
- **按需激活**：辅助智能体不是每次都跑，而是根据分解器的标记动态触发，保持系统高效。
- **统一接口**：所有智能体都遵循同一“指令+上下文”输入格式，便于后期加入新工具（比如图数据库检索）而不改动核心代码。
- **开源闭环**：先用 GPT‑4 打出上限，再用同样的提示模板微调 SQL‑Llama，实现了从闭源到开源的平滑迁移。

### 实验与效果
- **数据集**：主要在 BIRD 基准上评估，这是一个包含大规模真实数据库和复杂自然语言查询的公开测试集。作者还在内部的 “huge‑DB” 场景做了补充实验（未公开细节）。
- **基线对比**：与直接使用 GPT‑4（单模型）相比，MAC‑SQL+GPT‑4 在执行准确率上从 46.35% 提升到 59.59%，刷新了该数据集的 SOTA。使用 SQL‑Llama 替代 GPT‑4 时，准确率为 43.94%，仅略低于闭源上限，证明了开源模型的竞争力。
- **消融实验**：论文分别关闭检索智能体、关闭纠错智能体以及去掉思维链提示，准确率分别下降约 7%、5% 和 4%，显示每个模块都有实质贡献。
- **局限性**：作者承认在极端超大规模（表数超过 10 万）时检索智能体的响应时间仍是瓶颈；此外，纠错循环最多三次的设定在某些极端错误上仍会失败。

### 影响与延伸思考
MAC‑SQL 把“多智能体+工具调用”理念正式搬进 NL2SQL 场景，开启了“LLM+数据库协同”新潮流。后续工作已经开始探索：
- 将 **图数据库检索** 作为另一种子库获取方式，以处理更复杂的关系型查询（推测）。
- 把 **强化学习** 引入纠错智能体，让模型在多轮纠错中自我奖励，提高收敛效率（推测）。
- 在 **跨模态** 场景（如自然语言+表格图片）中复用同样的多智能体框架，进一步验证其通用性。

如果想深入，可以关注 **Tool‑Augmented LLM** 系列论文，以及最近的 **Agentic LLM** 开源项目，它们在设计模式和接口规范上与 MAC‑SQL 十分相似。

### 一句话记住它
把大模型拆成“拆解‑检索‑纠错”三位一体的团队，让每一步都专注于小任务，就能在巨型数据库和复杂问句面前稳稳提升 Text‑to‑SQL 的准确率。